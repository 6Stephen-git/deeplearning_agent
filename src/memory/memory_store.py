# src/memory/memory_store.py
"""
记忆存储核心模块 (MemoryStore) - 向量数据库集成版。
职责：管理记忆的存储、检索、更新与统计，集成ChromaDB实现向量相似性搜索。
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import logging

# ChromaDB 导入
import chromadb
from chromadb.config import Settings
# 导入我们刚刚创建的嵌入工具
from .embedding_tool import embedding_tool

logger = logging.getLogger(__name__)

# 保留原有的枚举定义
class MemoryType(Enum):
    FACT = "fact"
    INSIGHT = "insight"
    METHOD = "method"
    SOURCE = "source"
    TASK_SUMMARY = "task_summary"
    REPORT = "report"
    UPLOADED_DOC = "uploaded_doc"

class MemoryPriority(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1

class MemoryRecord:
    def __init__(self, content: str, memory_type: MemoryType, priority: MemoryPriority = MemoryPriority.MEDIUM, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.priority = priority
        self.embedding = embedding
        self.metadata = metadata or {}
        self.metadata.setdefault("created_at", datetime.now().isoformat())
        self.metadata.setdefault("updated_at", self.metadata["created_at"])
        self.access_count = 0
        self.last_accessed = None
        self.related_memory_ids: List[str] = []
    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "content": self.content, "memory_type": self.memory_type.value, "priority": self.priority.value, "metadata": self.metadata, "embedding": self.embedding, "access_count": self.access_count, "last_accessed": self.last_accessed, "related_memory_ids": self.related_memory_ids}
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        record = cls(content=data["content"], memory_type=MemoryType(data["memory_type"]), priority=MemoryPriority(data["priority"]), metadata=data.get("metadata", {}), embedding=data.get("embedding"))
        record.id = data["id"]
        record.access_count = data.get("access_count", 0)
        record.last_accessed = data.get("last_accessed")
        record.related_memory_ids = data.get("related_memory_ids", [])
        return record


class MemoryStore:
    """
    记忆存储管理器（集成ChromaDB版）。
    """

    def __init__(self, persist_directory: str = "./data/memory_db"):
        """
        初始化记忆存储，连接ChromaDB。

        Args:
            persist_directory: ChromaDB 数据持久化目录。
        """
        self.persist_directory = persist_directory
        logger.info(f"正在初始化MemoryStore，数据目录: {persist_directory}")

        # 1. 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        # 2. 获取或创建集合 (Collection)
        #    集合名称固定为 'research_memories'
        #    我们指定一个嵌入函数，但注意：我们将自行生成嵌入，所以使用默认的。
        self.collection = self.client.get_or_create_collection(
            name="research_memories",
            # 指定距离函数，cosine 在文本相似度中常用
            metadata={"hnsw:space": "cosine"}
        )

        # 3. 初始化嵌入工具
        if embedding_tool is None:
            logger.warning("嵌入工具未正确初始化，向量检索功能将受限。")
        self.embedder = embedding_tool

        # 4. 初始化内存中的索引和统计（为了快速过滤和统计，我们仍保留部分内存结构）
        self._memory_registry: Dict[str, MemoryRecord] = {}
        self._topic_index: Dict[str, List[str]] = {}
        self._type_index: Dict[str, List[str]] = {}
        self.stats = {
            "total_memories": 0,
            "memories_by_type": {t.value: 0 for t in MemoryType},
            "memories_by_priority": {p.value: 0 for p in MemoryPriority},
            "total_accesses": 0
        }

        # 5. 启动时从ChromaDB加载元数据到内存索引（可选，提升过滤速度）
        #    这是一个简化实现。对于大量记忆，可能需要懒加载或更复杂的缓存。
        self._load_existing_memories()
        logger.info(f"MemoryStore 初始化完成。当前记忆数量: {self.stats['total_memories']}")

    def _load_existing_memories(self):
        """从ChromaDB加载已有记忆的元数据，构建内存索引。"""
        try:
            # 获取集合中的所有数据（限制数量，防止内存爆炸）
            all_data = self.collection.get(include=["metadatas"])
            if not all_data or not all_data['ids']:
                return

            for mid, metadata in zip(all_data['ids'], all_data['metadatas']):
                if not metadata:
                    continue
                # 从元数据中重建基础记忆对象（不包含embedding和content）
                mem_type = metadata.get("memory_type", "fact")
                priority = metadata.get("priority", 3)
                research_topic = metadata.get("research_topic")

                # 创建简化记录存入注册表
                record = MemoryRecord(
                    content="",  # 内容不加载到内存
                    memory_type=MemoryType(mem_type),
                    priority=MemoryPriority(priority),
                    metadata=metadata
                )
                record.id = mid
                self._memory_registry[mid] = record

                # 更新索引
                type_key = record.memory_type.value
                if type_key not in self._type_index:
                    self._type_index[type_key] = []
                self._type_index[type_key].append(mid)

                if research_topic:
                    if research_topic not in self._topic_index:
                        self._topic_index[research_topic] = []
                    self._topic_index[research_topic].append(mid)

                # 更新统计
                self.stats["total_memories"] += 1
                self.stats["memories_by_type"][type_key] += 1
                self.stats["memories_by_priority"][priority] += 1

            logger.info(f"从持久化存储加载了 {len(all_data['ids'])} 条记忆的索引。")
        except Exception as e:
            logger.error(f"从ChromaDB加载已有记忆索引失败: {e}")

    def add_memory(
        self,
        content: str,
        memory_type: Union[MemoryType, str],
        priority: Union[MemoryPriority, int] = MemoryPriority.MEDIUM,
        research_topic: Optional[str] = None,
        task_id: Optional[int] = None,
        source_urls: Optional[List[str]] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加一条新记忆到存储中（包含向量化）。

        Args:
            content: 记忆文本内容。
            ... (其他参数同前) ...

        Returns:
            新创建记忆的唯一ID。
        """
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        if isinstance(priority, int):
            priority = MemoryPriority(priority)

        # 构建基本元数据，过滤掉None值
        full_metadata = {}
        if research_topic is not None:
            full_metadata["research_topic"] = research_topic
        if task_id is not None:
            full_metadata["task_id"] = task_id
        full_metadata["confidence"] = confidence
        full_metadata["memory_type"] = memory_type.value
        full_metadata["priority"] = priority.value

        # 只添加非空的source_urls
        if source_urls:
            full_metadata["source_urls"] = source_urls

        # 记录嵌入配置，便于追踪同库向量空间一致性
        if self.embedder is not None:
            full_metadata.setdefault("embedding_provider", getattr(self.embedder, "provider", "unknown"))
            full_metadata.setdefault("embedding_model", getattr(self.embedder, "model_name", "unknown"))
            full_metadata.setdefault("embedding_dim", int(getattr(self.embedder, "embedding_dim", 0) or 0))

        # 合并额外元数据，同样需要过滤None值
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    # 如果是列表，确保非空
                    if isinstance(value, list) and len(value) == 0:
                        continue
                    full_metadata[key] = value

        # 1. 生成嵌入向量
        embedding = None
        if self.embedder and content.strip():
            embedding = self.embedder.generate_embedding(content)
            if not embedding:  # 生成失败
                logger.warning(f"为记忆内容生成嵌入失败，将仅存储元数据。内容预览: {content[:100]}")

        # 2. 创建内存记录对象
        memory = MemoryRecord(
            content=content,
            memory_type=memory_type,
            priority=priority,
            metadata=full_metadata,
            embedding=embedding
        )

        # 3. 存储到 ChromaDB
        try:
            # ChromaDB 的 add 方法
            self.collection.add(
                documents=[content],
                metadatas=[full_metadata],
                ids=[memory.id],
                embeddings=[embedding] if embedding else None
            )
        except Exception as e:
            logger.error(f"将记忆存储到ChromaDB失败: {e}")
            # 存储失败，可以在此处抛出异常或返回错误
            raise RuntimeError(f"无法保存记忆到数据库: {e}") from e

        # 4. 更新内存索引和统计
        self._memory_registry[memory.id] = memory
        self._update_indices(memory, research_topic)
        self._update_stats(memory, "add")

        logger.info(f"已添加向量记忆 ID:{memory.id[:8]} 类型:{memory_type.value} 主题:{research_topic} 向量:{'是' if embedding else '否'}")
        return memory.id

    def search_by_similarity(
        self,
        query: str,
        n_results: int = 10,
        research_topic: Optional[str] = None,
        memory_type: Optional[Union[MemoryType, str]] = None,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        基于向量相似性进行语义检索。

        Args:
            query: 查询文本。
            n_results: 返回的结果数量。
            research_topic: 可选，按研究主题过滤（ChromaDB metadata过滤）。
            memory_type: 可选，按记忆类型过滤。
            where_filter: 可选，传递给ChromaDB的更复杂的metadata过滤字典。

        Returns:
            一个列表，每个元素是 (MemoryRecord, similarity_score) 的元组。
        """
        if not query.strip():
            logger.warning("相似性搜索的查询文本为空。")
            return []

        if not self.embedder:
            logger.error("嵌入工具不可用，无法执行向量搜索。")
            return []

        # 1. 为查询文本生成嵌入
        query_embedding = self.embedder.generate_embedding(query)
        if not query_embedding:
            return []

        # 构建 ChromaDB 查询过滤器
        conditions = []

        # 添加研究主题条件
        if research_topic:
            conditions.append({"research_topic": {"$eq": research_topic}})

        # 添加记忆类型条件
        if memory_type:
            mem_type_val = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
            conditions.append({"memory_type": {"$eq": mem_type_val}})

        # 添加自定义过滤器条件
        if where_filter:
            conditions.append(where_filter)

        # 组合所有条件
        if len(conditions) == 0:
            final_where = None
        elif len(conditions) == 1:
            final_where = conditions[0]
        else:
            # 多个条件用 $and 组合
            final_where = {"$and": conditions}

        # 3. 执行向量查询
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=final_where if final_where else None,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            logger.error(f"执行ChromaDB向量查询失败: {e}")
            return []

        # 4. 格式化返回结果
        returned_memories = []
        if results and results['ids'] and results['ids'][0]:
            for i, mem_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i]  # 距离，越小越相似

                # 将距离转换为相似度分数 (余弦距离范围大致为[0,2]，转换为[0,1]的相似度)
                similarity_score = 1.0 - (distance / 2.0) if distance is not None else 0.0
                similarity_score = max(0.0, min(1.0, similarity_score))  # 钳制到[0,1]

                # 从内存注册表获取完整记录，或从元数据重建
                memory = self._memory_registry.get(mem_id)
                if not memory:
                    # 从元数据和文档重建一个简化记录
                    memory = MemoryRecord.from_dict({
                        "id": mem_id,
                        "content": document,
                        "memory_type": metadata.get("memory_type", "fact"),
                        "priority": metadata.get("priority", 3),
                        "metadata": metadata
                    })
                returned_memories.append((memory, similarity_score))

        logger.info(f"向量搜索完成: 查询“{query[:50]}...” 返回 {len(returned_memories)} 条结果。")
        return returned_memories

    def search_memories(
        self,
        query: Optional[str] = None,
        memory_type: Optional[Union[MemoryType, str]] = None,
        research_topic: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 20,
        use_hybrid: bool = False
    ) -> List[MemoryRecord]:
        """
        增强版多条件记忆检索。
        现在支持两种模式：
        1. 仅关键词/过滤检索 (use_hybrid=False)
        2. 混合检索：结合向量语义搜索和过滤 (use_hybrid=True)

        Args:
            use_hybrid: 如果为True且query不为空，则使用混合策略（向量相似性+过滤）。
                        否则使用传统的关键词匹配过滤。
        """
        if use_hybrid and query:
            # 混合检索模式：先做向量搜索，再进行过滤和排序
            vector_results = self.search_by_similarity(
                query=query,
                n_results=limit * 2,  # 多取一些，方便后续过滤
                research_topic=research_topic,
                memory_type=memory_type
            )

            # 应用额外的过滤（如置信度）和排序
            filtered = []
            for memory, sim_score in vector_results:
                if memory.metadata.get("confidence", 1.0) < min_confidence:
                    continue
                # 可选：将相似度分数存入metadata临时字段，供排序参考
                memory.metadata["_similarity_score"] = sim_score
                filtered.append(memory)

            # 排序：相似度 > 优先级 > 置信度
            filtered.sort(key=lambda m: (
                -m.metadata.get("_similarity_score", 0.0),
                -m.priority.value,
                -m.metadata.get("confidence", 0.0)
            ))
            result = filtered[:limit]
            # 清理临时字段
            for m in result:
                m.metadata.pop("_similarity_score", None)
            return result
        else:
            # 传统关键词过滤模式 (与原逻辑类似，但通过内存索引快速筛选)
            candidate_ids = self._get_candidate_ids(memory_type, research_topic)
            candidates = [self._memory_registry[mid] for mid in candidate_ids]

            filtered = []
            for memory in candidates:
                if memory.metadata.get("confidence", 1.0) < min_confidence:
                    continue
                if query and query.lower() not in memory.content.lower():
                    continue
                filtered.append(memory)

            filtered.sort(key=lambda m: (
                -m.priority.value,
                -m.metadata.get("confidence", 0.0),
                m.metadata.get("created_at", "")
            ))
            return filtered[:limit]

    # 以下辅助方法 _update_indices, _update_stats, _get_candidate_ids, get_stats 等逻辑不变，
    # 但需要注意它们现在操作的是基于元数据的内存记录，而非完整内容。
    def _update_indices(self, memory: MemoryRecord, research_topic: Optional[str]):
        type_key = memory.memory_type.value
        if type_key not in self._type_index:
            self._type_index[type_key] = []
        self._type_index[type_key].append(memory.id)
        if research_topic:
            if research_topic not in self._topic_index:
                self._topic_index[research_topic] = []
            self._topic_index[research_topic].append(memory.id)
    def _update_stats(self, memory: MemoryRecord, operation: str):
        if operation == "add":
            self.stats["total_memories"] += 1
            self.stats["memories_by_type"][memory.memory_type.value] += 1
            self.stats["memories_by_priority"][memory.priority.value] += 1
    def _get_candidate_ids(self, memory_type: Optional[MemoryType], research_topic: Optional[str]) -> List[str]:
        if memory_type:
            type_candidates = set(self._type_index.get(memory_type.value, []))
        else:
            type_candidates = set(self._memory_registry.keys())
        if research_topic:
            topic_candidates = set(self._topic_index.get(research_topic, []))
        else:
            topic_candidates = set(self._memory_registry.keys())
        return list(type_candidates & topic_candidates)
    def get_stats(self, detail: bool = False) -> Dict[str, Any]:
        base_stats = {
            "total_memories": self.stats["total_memories"],
            "total_accesses": self.stats["total_accesses"],
            "persist_directory": self.persist_directory,
            "chroma_collection_count": self.collection.count()
        }
        if detail:
            base_stats.update({
                "by_type": self.stats["memories_by_type"],
                "by_priority": self.stats["memories_by_priority"],
                "by_topic": {topic: len(ids) for topic, ids in self._topic_index.items()}
            })
        return base_stats

    def query_memories(
        self,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        include_content: bool = True
    ) -> List[MemoryRecord]:
        """
        基于ChromaDB的metadata过滤查询记忆（用于上传文档管理等场景）。

        Args:
            filter_conditions: ChromaDB where 过滤条件（metadata过滤字典）。
            limit: 最大返回数量。
            include_content: 是否返回文档内容（documents）。

        Returns:
            MemoryRecord 列表。
        """
        def _normalize_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not where:
                return None
            # 若已经是 Chroma where 表达式（含 $and/$or 或操作符），直接使用
            if any(str(k).startswith("$") for k in where.keys()):
                return where
            # 将 {"a":1,"b":2} 规范化为 {"$and":[{"a":{"$eq":1}},{"b":{"$eq":2}}]}
            clauses = [{k: {"$eq": v}} for k, v in where.items()]
            if len(clauses) == 1:
                return clauses[0]
            return {"$and": clauses}

        try:
            includes = ["metadatas"]
            if include_content:
                includes.append("documents")

            data = self.collection.get(
                where=_normalize_where(filter_conditions),
                limit=limit,
                include=includes
            )
        except Exception as e:
            logger.error(f"query_memories 查询失败: {e}")
            return []

        ids = (data or {}).get("ids") or []
        metadatas = (data or {}).get("metadatas") or []
        documents = (data or {}).get("documents") or ([""] * len(ids))

        results: List[MemoryRecord] = []
        for i, mid in enumerate(ids):
            md = metadatas[i] or {}
            doc = documents[i] if include_content and i < len(documents) else ""

            try:
                mem_type = md.get("memory_type", "fact")
                priority = md.get("priority", 3)
                record = MemoryRecord(
                    content=doc or "",
                    memory_type=MemoryType(mem_type),
                    priority=MemoryPriority(int(priority)),
                    metadata=md
                )
                record.id = mid
                results.append(record)
            except Exception:
                # 元数据不规范时做最小化回退
                record = MemoryRecord(
                    content=doc or "",
                    memory_type=MemoryType.FACT,
                    priority=MemoryPriority.MEDIUM,
                    metadata=md
                )
                record.id = mid
                results.append(record)

        return results

    def delete_memories(self, filter_conditions: Dict[str, Any]) -> int:
        """
        按metadata条件删除记忆（用于删除上传文件等管理操作）。

        Returns:
            实际删除数量。
        """
        def _normalize_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not where:
                return None
            if any(str(k).startswith("$") for k in where.keys()):
                return where
            clauses = [{k: {"$eq": v}} for k, v in where.items()]
            if len(clauses) == 1:
                return clauses[0]
            return {"$and": clauses}

        try:
            data = self.collection.get(
                where=_normalize_where(filter_conditions),
                include=["metadatas"]
            )
            ids = (data or {}).get("ids") or []
            if not ids:
                return 0

            self.collection.delete(ids=ids)

            # 尽量同步内存索引（最佳努力，不保证完整一致）
            for mid in ids:
                rec = self._memory_registry.pop(mid, None)
                if rec:
                    try:
                        type_key = rec.memory_type.value
                        if type_key in self._type_index:
                            self._type_index[type_key] = [x for x in self._type_index[type_key] if x != mid]
                        topic = rec.metadata.get("research_topic")
                        if topic and topic in self._topic_index:
                            self._topic_index[topic] = [x for x in self._topic_index[topic] if x != mid]
                        # 更新统计（简单扣减）
                        self.stats["total_memories"] = max(0, self.stats["total_memories"] - 1)
                        self.stats["memories_by_type"][type_key] = max(
                            0, self.stats["memories_by_type"].get(type_key, 0) - 1
                        )
                    except Exception:
                        pass

            return len(ids)
        except Exception as e:
            logger.error(f"delete_memories 删除失败: {e}")
            raise NotImplementedError(f"delete_memories not available: {e}") from e