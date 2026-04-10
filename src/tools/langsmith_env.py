"""
LangSmith 环境变量与追踪开关。

用于：
- RAG 评估：`eval_langsmith`（@traceable 上报 runs）
- 可选：深度研究图 `run_research.py`（需显式打开开关）

所需（云端 LangSmith）：
- LANGSMITH_API_KEY
- LANGSMITH_PROJECT（可选，默认 rag-eval / research-agent）
"""

from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def configure_langsmith_tracing(
    *,
    project_name: Optional[str] = None,
    force_tracing: bool = True,
) -> bool:
    """
    若已配置 API Key，则设置 LANGSMITH_TRACING_V2 并统一密钥环境变量。

    Returns:
        True 表示已检测到 Key 并完成配置；False 表示无 Key，调用方应跳过上报。
    """
    key = (
        os.getenv("LANGSMITH_API_KEY", "").strip()
        or os.getenv("LANGCHAIN_API_KEY", "").strip()
    )
    if not key:
        return False

    os.environ["LANGSMITH_API_KEY"] = key
    if project_name:
        os.environ["LANGSMITH_PROJECT"] = project_name.strip()

    if force_tracing:
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        # 兼容旧文档 / 部分 LangChain 组件
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    return True


def try_enable_langsmith_for_research() -> None:
    """
    当 LANGSMITH_TRACE_RESEARCH=1/true 且存在 API Key 时，为 `run_research` 打开追踪。

    LangGraph + LangChain 会在 TRACING_V2 开启时自动将步骤写入 LangSmith（取决于版本）。
    """
    raw = os.getenv("LANGSMITH_TRACE_RESEARCH", "").lower()
    if raw not in ("1", "true", "yes", "y"):
        return
    project = os.getenv("LANGSMITH_PROJECT", "research-agent").strip() or "research-agent"
    if configure_langsmith_tracing(project_name=project, force_tracing=True):
        logger.info("[LangSmith] 已启用研究流程追踪，项目=%s", project)
