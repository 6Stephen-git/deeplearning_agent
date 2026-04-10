# src/llm_client.py
"""
LLM客户端集中导出模块。
职责：安全地从环境变量读取配置，创建并导出一个全局可用的、异步的Qwen模型客户端实例。
"""
import os
import logging
from typing import Optional

from langchain_openai import ChatOpenAI  # 使用OpenAI兼容接口调用阿里云Qwen
from src.config import ensure_project_env_loaded

ensure_project_env_loaded()
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def create_async_llm_client(
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """
    创建并返回一个配置好的异步 Qwen 模型客户端。

    未传入的参数从环境变量读取（与历史行为一致）。
    HyDE/MQE 等长生成请用 `create_enhancer_llm_client()`，避免默认短超时 + 多次重试导致总耗时过长、易超时。
    """
    # 兼容两种 key 命名：
    # - DASHSCOPE_API_KEY：DashScope 原生
    # - OPENAI_API_KEY：OpenAI 兼容（便于本地/CI 统一）
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "无法创建LLM客户端: 未找到环境变量 'DASHSCOPE_API_KEY' 或 'OPENAI_API_KEY'。\n"
            "  请检查项目根目录下的 .env 文件或系统环境变量，并确保至少设置其中一个。"
        )

    model_name = model_name if model_name is not None else os.getenv("QWEN_MODEL_NAME", "qwen-turbo")
    if temperature is None:
        temperature = float(os.getenv("QWEN_TEMPERATURE", 0.1))
    else:
        temperature = float(temperature)
    base_url = os.getenv(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    timeout = _env_int("QWEN_TIMEOUT", 30) if timeout is None else int(timeout)
    max_retries = _env_int("QWEN_MAX_RETRIES", 2) if max_retries is None else int(max_retries)
    if max_tokens is None:
        env_max_tokens = os.getenv("QWEN_MAX_TOKENS")
        if env_max_tokens is not None and str(env_max_tokens).strip():
            max_tokens = int(env_max_tokens)

    kwargs = dict(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        streaming=False,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    return ChatOpenAI(**kwargs)


def create_enhancer_llm_client():
    """
    专用于 HyDE / MQE（先 LLM 生成再向量检索）的客户端。

    背景：全局 `llm_client` 默认 QWEN_TIMEOUT=30s 且 QWEN_MAX_RETRIES=2，
    单次请求若超时，底层会重试，总等待可达约 90s+；HyDE 需生成一段假设答案，
    易触发 `Request timed out`，表现为「没有假设答案（回退原 query）」且「等很久」。

    环境变量（可选）：
    - ENHANCER_QWEN_TIMEOUT：单次 HTTP 读超时秒数，默认 300（HyDE/MQE 生成较慢，120 易误判超时）
    - ENHANCER_QWEN_MAX_RETRIES：失败重试次数，默认 0
    - ENHANCER_MAX_TOKENS：最大生成 token，默认 600
    """
    timeout = _env_int("ENHANCER_QWEN_TIMEOUT", 300)
    max_retries = _env_int("ENHANCER_QWEN_MAX_RETRIES", 0)
    return create_async_llm_client(
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=_env_int("ENHANCER_MAX_TOKENS", 600),
    )


_enhancer_llm_client_singleton = None
_report_llm_client_singleton = None


def get_enhancer_llm_client():
    """懒加载单例，供 HyDE/MQE 与测试复用，避免重复构造客户端。"""
    global _enhancer_llm_client_singleton
    if _enhancer_llm_client_singleton is None:
        _enhancer_llm_client_singleton = create_enhancer_llm_client()
    return _enhancer_llm_client_singleton


def create_report_llm_client():
    """
    专用于最终报告润色的客户端。

    默认使用更长超时、较少重试和更高 max_tokens，降低报告阶段 Request timed out 的概率。
    """
    report_timeout = _env_int("REPORT_QWEN_TIMEOUT", 180)
    report_max_retries = _env_int("REPORT_QWEN_MAX_RETRIES", 1)
    report_max_tokens = _env_int("REPORT_MAX_TOKENS", _env_int("QWEN_MAX_TOKENS", 3000))
    return create_async_llm_client(
        timeout=report_timeout,
        max_retries=report_max_retries,
        max_tokens=report_max_tokens,
    )


def get_report_llm_client():
    """懒加载报告专用单例，避免重复构造客户端。"""
    global _report_llm_client_singleton
    if _report_llm_client_singleton is None:
        _report_llm_client_singleton = create_report_llm_client()
    return _report_llm_client_singleton


# 导出全局单例（短超时，适合规划等常规短调用）
try:
    llm_client = create_async_llm_client()
except Exception as e:
    llm_client = None
    logger.warning("默认 llm_client 初始化失败（通常是缺少密钥）: %s", e)
