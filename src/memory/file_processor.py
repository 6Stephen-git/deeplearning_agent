"""
外部文件上传处理器。
基于LangChain工具，支持多种格式文档解析、分块并存储到向量库。
"""
import os
import hashlib
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.memory.memory_store import MemoryStore, MemoryType, MemoryPriority

logger = logging.getLogger(__name__)


def _extract_source_url_from_document_header(content: str) -> Optional[str]:
    """
    从正文首节提取 `SourceURL: ...`（rag_eval build_index 写入的溯源行）。
    分块后只有第一个 chunk 仍以该行开头，故在 process_uploaded_file 里对**整篇**先解析一次，
    再写入文件级 metadata，使同一文件的所有 chunk 都带 source_urls。
    """
    if not content:
        return None
    text = content.lstrip()
    if not text.startswith("SourceURL:"):
        return None
    first_line = text.split("\n", 1)[0]
    u = first_line.replace("SourceURL:", "", 1).strip()
    return u or None


class FileUploadProcessor:
    """
    文件上传处理器。
    职责：解析多种格式的文档，分块，嵌入，并存储到MemoryStore。
    """

    # 支持的文件扩展名映射
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.doc': Docx2txtLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.markdown': UnstructuredMarkdownLoader,
        '.csv': CSVLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
    }

    def __init__(self,
                 memory_store: MemoryStore,
                 embedding_model: Optional[Any] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        初始化文件处理器。

        Args:
            memory_store: MemoryStore实例，用于存储处理后的文档
            embedding_model: 嵌入模型，如果为None则使用memory_store的默认模型
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
        """
        self.memory_store = memory_store
        # 兼容历史参数：实际入库嵌入统一由 MemoryStore.embedder 生成
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

    def process_uploaded_file(self,
                              file_path: Union[str, Path],
                              research_topic: str,
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理单个上传文件，解析、分块并存储到向量库。

        Args:
            file_path: 文件路径
            research_topic: 关联的研究主题
            metadata: 额外的元数据

        Returns:
            处理结果统计
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 验证文件格式
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {ext}。支持: {list(self.SUPPORTED_EXTENSIONS.keys())}")

        # 1. 加载文档
        logger.info("[文件处理器] 开始处理文件: %s", file_path.name)
        documents = self._load_documents(file_path)

        if not documents:
            return {"success": False, "message": "文档内容为空"}

        # 文件级 source_urls：整篇首行 SourceURL（分块后多数 chunk 不再以该行开头，否则只有首块有 URL）
        merged_meta: Dict[str, Any] = dict(metadata or {})
        first_text = documents[0].page_content or ""
        header_url = _extract_source_url_from_document_header(first_text)
        if header_url:
            merged_meta.setdefault("source_urls", [header_url])

        # 2. 分割文本
        chunks = self._split_documents(documents, file_path.name)

        # 3. 准备元数据
        file_metadata = self._prepare_file_metadata(
            file_path, research_topic, merged_meta if merged_meta else None
        )

        # 4. 存储到向量库
        stored_count = self._store_chunks_to_memory(chunks, file_metadata)

        return {
            "success": True,
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "original_pages": len(documents),
            "chunks_created": len(chunks),
            "chunks_stored": stored_count,
            "research_topic": research_topic,
            "timestamp": datetime.now().isoformat()
        }

    def _load_documents(self, file_path: Path) -> List[Document]:
        """使用LangChain加载器加载文档"""
        ext = file_path.suffix.lower()
        loader_class = self.SUPPORTED_EXTENSIONS[ext]

        try:
            # 特殊处理CSV（需要指定列）
            if ext == '.csv':
                loader = loader_class(str(file_path), encoding='utf-8')
            elif ext == '.txt':
                # Windows 下 TextLoader 默认编码常为系统 locale（如 GBK），而 rag_eval 清洗产物为 UTF-8，
                # 不设 encoding 会先失败再走 except 里的 UTF-8 降级，日志里反复出现「文档加载失败」。
                loader = TextLoader(
                    str(file_path),
                    encoding='utf-8',
                    autodetect_encoding=True,
                )
            else:
                loader = loader_class(str(file_path))

            documents = loader.load()
            logger.info("[文件处理器] 成功加载文档 sections=%s", len(documents))
            return documents

        except Exception as e:
            logger.warning("[文件处理器] 文档加载失败: %s", e)
            # 降级方案：使用纯文本加载器
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                logger.info("[文件处理器] 使用纯文本加载器成功 sections=%s", len(documents))
                return documents
            except Exception as fallback_error:
                raise RuntimeError(f"所有加载方式都失败: {fallback_error}")

    def _split_documents(self, documents: List[Document], file_name: str) -> List[Document]:
        """分割文档为适合嵌入的块"""
        chunks = self.text_splitter.split_documents(documents)

        # 为每个块添加文件信息
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["file_name"] = file_name

        logger.info("[文件处理器] 文档分割完成 chunk_count=%s", len(chunks))
        return chunks

    def _prepare_file_metadata(self,
                               file_path: Path,
                               research_topic: str,
                               custom_metadata: Optional[Dict]) -> Dict[str, Any]:
        """准备文件元数据"""
        # 计算文件哈希，用于去重
        file_hash = self._calculate_file_hash(file_path)

        base_metadata = {
            "source_type": "uploaded_file",
            "file_name": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "file_hash": file_hash,
            "research_topic": research_topic,
            "upload_time": datetime.now().isoformat(),
            "processor_version": "1.0.0"
        }

        if custom_metadata:
            base_metadata.update(custom_metadata)

        return base_metadata

    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _store_chunks_to_memory(self,
                                chunks: List[Document],
                                metadata: Dict[str, Any]) -> int:
        """将文档块存储到MemoryStore"""
        stored_count = 0

        for chunk in chunks:
            try:
                # 合并块元数据和文件元数据
                chunk_metadata = {**metadata, **chunk.metadata}

                # 从正文头部提取 SourceURL（rag_eval_runner build_index 会写入）
                # 这样检索与评估可以按文档 URL 聚合，而不是被 chunk_id 绑死。
                content = chunk.page_content or ""
                extracted_urls: List[str] = []
                m = None
                if content.startswith("SourceURL:"):
                    try:
                        first_line, rest = content.split("\n", 1)
                    except ValueError:
                        first_line, rest = content, ""
                    m = first_line.replace("SourceURL:", "", 1).strip()
                    if m:
                        extracted_urls = [m]
                    # 去掉头部，避免影响嵌入
                    content = rest.lstrip("\n")
                if extracted_urls and not chunk_metadata.get("source_urls"):
                    chunk_metadata["source_urls"] = extracted_urls

                # 存储到MemoryStore
                self.memory_store.add_memory(
                    content=content,
                    memory_type=MemoryType.UPLOADED_DOC,
                    priority=MemoryPriority.MEDIUM,
                    metadata=chunk_metadata
                )
                stored_count += 1

            except Exception as e:
                logger.warning("[文件处理器] 存储块失败 chunk_id=%s err=%s", chunk.metadata.get("chunk_id"), e)
                continue

        return stored_count

    def process_directory(self,
                          directory_path: Union[str, Path],
                          research_topic: str,
                          recursive: bool = True) -> List[Dict[str, Any]]:
        """
        处理整个目录下的所有支持文件。

        Args:
            directory_path: 目录路径
            research_topic: 关联的研究主题
            recursive: 是否递归处理子目录

        Returns:
            每个文件的处理结果列表
        """
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise ValueError(f"不是有效目录: {directory_path}")

        results = []

        # 收集所有支持的文件
        pattern = "**/*" if recursive else "*"
        all_files = list(directory_path.glob(pattern))
        supported_files = [
            f for f in all_files
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        logger.info("[文件处理器] 在目录中发现支持文件=%s", len(supported_files))

        for file_path in supported_files:
            try:
                result = self.process_uploaded_file(file_path, research_topic)
                results.append(result)
                logger.info("[文件处理器] 完成: %s", file_path.name)
            except Exception as e:
                logger.warning("[文件处理器] 失败: %s - %s", file_path.name, e)
                results.append({
                    "success": False,
                    "file_name": str(file_path),
                    "error": str(e)
                })

        return results


class UploadedDocumentManager:
    """
    已上传文档管理器。
    提供对已上传文档的查询、管理和检索功能。
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def get_uploaded_documents(self, research_topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取所有已上传的文档信息"""
        # 这里需要实现从memory_store中查询特定类型的记忆
        # 假设memory_store有查询方法
        query_filter = {"source_type": "uploaded_file"}
        if research_topic:
            query_filter["research_topic"] = research_topic

        documents = self.memory_store.query_memories(
            filter_conditions=query_filter,
            limit=1000
        )

        # 按文件分组
        files_dict = {}
        for doc in documents:
            file_name = doc.metadata.get("file_name")
            if file_name not in files_dict:
                files_dict[file_name] = {
                    "file_name": file_name,
                    "research_topic": doc.metadata.get("research_topic"),
                    "upload_time": doc.metadata.get("upload_time"),
                    "total_chunks": 0,
                    "chunks": []
                }
            files_dict[file_name]["total_chunks"] += 1
            files_dict[file_name]["chunks"].append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            })

        return list(files_dict.values())

    def delete_uploaded_file(self, file_name: str, research_topic: str) -> bool:
        """删除指定文件的所有块"""
        # 实现从memory_store中删除特定文件的所有记忆
        # 这取决于memory_store是否支持按元数据删除
        try:
            # 伪代码，实际实现取决于memory_store的删除接口
            deleted_count = self.memory_store.delete_memories(
                filter_conditions={
                    "source_type": "uploaded_file",
                    "file_name": file_name,
                    "research_topic": research_topic
                }
            )
            logger.info("[文档管理器] 已删除文件 %s 块数=%s", file_name, deleted_count)
            return deleted_count > 0
        except NotImplementedError:
            logger.warning("[文档管理器] MemoryStore不支持按条件删除")
            return False