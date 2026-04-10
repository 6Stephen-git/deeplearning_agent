"""
不生成报告的结束节点。
当达到最大轮次且质量远低于阈值时，用于输出提示并结束流程。
"""

import logging
from src.state import GraphState

logger = logging.getLogger(__name__)


async def no_report_node(state: GraphState) -> GraphState:
    research_topic = state.get("research_topic", "")
    reason = state.get("report_block_reason", "质量不足，建议重新研究。")
    logger.warning("[结束节点] 主题=%s 未生成报告，原因=%s", research_topic, reason)
    # 明确标记
    state["final_report"] = None
    return state

