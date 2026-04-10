# src/memory/embedding_tool.py
"""
嵌入生成工具（阿里云DashScope版）。
主选：DashScope文本嵌入API，备选：哈希回退。
"""
import os
import logging
import hashlib
import time
from typing import List, Optional, Dict, Any
from src.config import ensure_project_env_loaded

ensure_project_env_loaded()

logger = logging.getLogger(__name__)


class EmbeddingTool:
    """嵌入生成工具类（支持DashScope API）。"""

    def __init__(self, provider: Optional[str] = None):
        """
        初始化嵌入工具。

        Args:
            provider: 嵌入服务提供商，默认从环境变量读取，支持 'dashscope' 和 'hash'。
        """
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER", "dashscope").lower()
        self.model_name = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v2")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

        # 嵌入向量维度（DashScope v2模型为1536维）
        self.embedding_dim = 1536 if self.provider == "dashscope" else 256

        # 初始化客户端
        self.client = None
        self._initialize_client()

        logger.info(f"嵌入工具初始化完成，提供商: {self.provider}, 模型: {self.model_name}")

    def _initialize_client(self):
        """根据配置初始化嵌入客户端。"""
        if self.provider == "dashscope":
            self._init_dashscope_client()
        elif self.provider == "hash":
            logger.info("使用哈希嵌入回退策略。")
        else:
            logger.warning(f"不支持的嵌入提供商: {self.provider}，将使用哈希回退。")
            self.provider = "hash"

    def _init_dashscope_client(self):
        """初始化阿里云DashScope嵌入客户端。"""
        if not self.api_key:
            logger.warning("未配置DASHSCOPE_API_KEY，无法使用DashScope嵌入服务，将回退到哈希策略。")
            self.provider = "hash"
            return

        try:
            # 使用OpenAI兼容模式调用DashScope
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=30.0,  # 30秒超时
                max_retries=2  # 重试2次
            )
            # 不在初始化阶段发起联网请求（避免 import 即卡住/失败）。
            # 维度在首次真实调用后可自动纠正（若模型返回维度与默认不同）。
            logger.info("DashScope嵌入客户端初始化成功（未执行联网自检）。")

        except ImportError:
            logger.error("未安装openai库，请运行: pip install openai>=1.0.0")
            self.provider = "hash"
        except Exception as e:
            logger.error(f"初始化DashScope嵌入客户端失败: {e}，将回退到哈希策略。")
            self.provider = "hash"
            self.client = None

    def generate_embedding(self, text: str) -> List[float]:
        """
        为单条文本生成嵌入向量。

        Args:
            text: 输入文本。

        Returns:
            文本的向量表示（List[float]）。
        """
        if not text or not text.strip():
            return [0.0] * self.embedding_dim

        if self.provider == "dashscope" and self.client:
            return self._generate_dashscope_embedding(text)
        else:
            return self._generate_hash_embedding(text)

    def _generate_dashscope_embedding(self, text: str) -> List[float]:
        """使用DashScope API生成嵌入。"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text],
            )

            embedding = response.data[0].embedding
            # 纠正维度（如果服务端返回维度与预期不同）
            if embedding and len(embedding) != self.embedding_dim:
                self.embedding_dim = len(embedding)
            return embedding

        except Exception as e:
            logger.error(f"DashScope嵌入生成失败: {e}，将回退到哈希嵌入。")
            # 评估/索引构建等场景可开启严格模式：失败直接抛错，避免悄悄回退导致检索评估失真
            strict = os.getenv("EMBEDDING_STRICT", "false").lower() in ("1", "true", "yes", "y")
            if strict:
                raise
            # 单次失败后，不切换提供商，仅本次回退
            return self._generate_hash_embedding(text)

    def _generate_hash_embedding(self, text: str) -> List[float]:
        """生成基于哈希的回退嵌入。"""
        # 使用MD5哈希生成确定性嵌入
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()

        # 将哈希字节转换为浮点数列表
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            if len(embedding) >= self.embedding_dim:
                break
            chunk = hash_bytes[i:min(i + 4, len(hash_bytes))]
            value = int.from_bytes(chunk, 'big') / (2 ** 32)
            embedding.append(float(value))

        # 填充到目标维度
        while len(embedding) < self.embedding_dim:
            embedding.append(0.0)

        # 归一化
        norm = (sum(x * x for x in embedding) ** 0.5) or 1.0
        embedding = [x / norm for x in embedding]

        return embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文本生成嵌入向量。

        Args:
            texts: 输入文本列表。

        Returns:
            向量列表，顺序与输入文本对应。
        """
        if not texts:
            return []

        if self.provider == "dashscope" and self.client:
            return self._generate_dashscope_embeddings_batch(texts)
        else:
            return [self._generate_hash_embedding(text) for text in texts]

    def _generate_dashscope_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """使用DashScope API批量生成嵌入。"""
        try:
            # DashScope API支持批量，但注意tokens限制
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]
            return embeddings

        except Exception as e:
            logger.error(f"DashScope批量嵌入生成失败: {e}，将回退到哈希嵌入。")
            strict = os.getenv("EMBEDDING_STRICT", "false").lower() in ("1", "true", "yes", "y")
            if strict:
                raise
            return [self._generate_hash_embedding(text) for text in texts]


# 创建全局工具实例
try:
    embedding_tool = EmbeddingTool()
    logger.info(f"全局嵌入工具初始化完成，使用: {embedding_tool.provider}")
except Exception as e:
    logger.error(f"全局嵌入工具初始化失败: {e}")
    # 创建最小功能的回退实例
    embedding_tool = EmbeddingTool(provider="hash")