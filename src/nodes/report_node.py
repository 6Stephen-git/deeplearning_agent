# src/nodes/report_node.py
"""
报告生成节点模块。
职责：整合所有子任务的研究成果，生成一份完整的最终研究报告。
"""
import os
import logging
from typing import List, Any

from src.memory.memory_store import MemoryType, MemoryPriority
from src.state import GraphState, TaskStatus
from src.tools.llm_client import get_report_llm_client
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        def _decorator(fn):  # type: ignore[no-untyped-def]
            return fn
        return _decorator

logger = logging.getLogger(__name__)


@traceable(name="report", run_type="chain")
async def report_node(state: GraphState) -> GraphState:
    """
    报告节点主函数。
    整合所有子任务结果，生成最终的综合性研究报告。

    Args:
        state: 当前图状态，必须包含 `research_topic` 和 `task_results`。

    Returns:
        更新后的状态，主要更新了：
        - `final_report`: 生成的最终报告文本 (Markdown格式)。
        - `sub_tasks[*].status`: 所有任务状态最终更新为 DONE。
    """
    research_topic = state["research_topic"]
    logger.info("[报告节点] 开始生成最终报告 topic=%s", research_topic)
    if state.get("report_warning"):
        logger.warning("[报告节点] 提醒: %s", state["report_warning"])

    # 若已被证据门控拦截，则不生成报告（兜底，避免误入 report 节点时产生幻觉）
    if state.get("report_allowed") is False:
        logger.warning("[报告节点] 报告被拦截: %s", state.get("report_block_reason", "无可验证证据"))
        state["final_report"] = None
        return state

    # 获取工作记忆的增强上下文
    memory_context = ""
    if state.get("working_memory"):
        try:
            # 获取格式化的完整上下文（包含相关记忆和会话历史）
            context, stats = state["working_memory"].get_context_for_llm(
                token_limit=3000,  # 为报告留出足够空间
                include_recent_items=10,
                include_memories=True
            )

            memory_context = f"""
    ## 研究上下文与历史参考
    {context}

    ## 当前研究发现总结
    以下是本次研究中各个子任务的详细发现：
    """
            logger.info(
                "[报告节点] 加载工作记忆上下文 tokens=%s memory_count=%s context_items=%s",
                stats["estimated_tokens"],
                stats["memory_count"],
                stats["context_item_count"],
            )

        except Exception as e:
            logger.warning("[报告节点] 获取工作记忆上下文失败: %s", e)
            memory_context = ""

    # 构建报告生成提示（整合记忆上下文）
    # ... 原有报告生成逻辑，但增强system_prompt ...

    # 报告生成后，将最终报告存储到长期记忆
    if state.get("working_memory") and state.get("final_report"):
        try:
            state["working_memory"].store_important_findings(
                content=state["final_report"],
                memory_type=MemoryType.REPORT,
                priority=MemoryPriority.CRITICAL,
                research_topic=research_topic,
                confidence=0.95
            )
            logger.info("[报告节点] 已将最终报告存储到长期记忆")
        except Exception as e:
            logger.warning("[报告节点] 存储最终报告失败: %s", e)

    # 1. 准备报告素材：过滤并格式化所有有效结果
    valid_results = [r for r in state["task_results"] if r is not None]
    if not valid_results:
        # 无有效结果的降级处理
        state["final_report"] = _generate_fallback_report(research_topic)
        logger.warning("[报告节点] 无有效结果，已生成降级报告")
        _mark_all_tasks_done(state)
        return state

    logger.info("[报告节点] 正在整合子任务成果 count=%s", len(valid_results))

    # 2. 构建用于整合报告的提示词
    formatted_input = _format_results_for_results(valid_results)
    max_input_chars = max(3000, int(os.getenv("REPORT_INPUT_MAX_CHARS", "14000")))
    if len(formatted_input) > max_input_chars:
        logger.warning(
            "[报告节点] 报告输入过长 chars=%s，已截断为 %s",
            len(formatted_input),
            max_input_chars,
        )
        formatted_input = formatted_input[:max_input_chars]
    system_prompt, user_prompt = _build_report_prompts(research_topic, formatted_input)

    # 3. 调用LLM生成最终报告
    try:
        report_llm_client = get_report_llm_client()
        response = await report_llm_client.agenerate(messages=[[system_prompt, user_prompt]])
        final_report = response.generations[0][0].text
    except Exception as e:
        logger.warning("[报告节点] 调用LLM生成报告失败: %s", e)
        final_report = _generate_fallback_report_from_results(valid_results, research_topic)

    # 4. 更新状态
    state["final_report"] = final_report
    _mark_all_tasks_done(state)

    logger.info("[报告节点] 报告生成完成 report_len=%s", len(final_report))
    return state


def _format_results_for_results(valid_results: List[dict]) -> str:
    """将所有有效子任务的结果格式化为一个连贯的文本块，供LLM整合。"""
    formatted_text = ""
    for result in valid_results:
        formatted_text += f"## {result.get('title', '未知标题')}\n\n"
        formatted_text += f"{result.get('summary', '无总结内容')}\n\n"
        # 可选：附带来源信息
        if result.get("sources"):
            formatted_text += "**参考来源**:\n"
            for src in result["sources"][:3]:  # 只列前几个关键来源
                formatted_text += f"- {src}\n"
        formatted_text += "\n---\n\n"
    return formatted_text


def _build_report_prompts(research_topic: str, formatted_results: str):
    """构建用于生成最终报告的系统提示和用户提示。"""
    system_prompt = SystemMessage(content="""你是一名高级研究助理。你的任务是根据提供的多个子任务研究总结，整合、润色并生成一份完整、结构清晰、语言流畅的最终研究报告。

# 报告结构要求
请生成一份正式的Markdown格式报告，应包含以下部分：
1.  **标题**：清晰反映研究主题。
2.  **概述/摘要**：简要介绍研究背景、目的和核心发现。
3.  **主体内容**：将提供的子任务总结有机地整合起来，按照合理的逻辑顺序（如：从概念到应用，从现状到趋势）组织成多个章节。避免简单罗列，要建立内容间的联系。
4.  **总结与展望**：归纳核心结论，并可能提出未来的研究方向或挑战。
5.  **参考文献**：如有必要，可列出关键信息来源。

# 证据与引用要求（强制）
- 你只能基于输入材料写作，禁止编造输入中不存在的事实、数据、结论或来源。
- 每个章节都必须包含“证据与来源”小节：
  - 使用引用标号 `[1] [2] ...` 指向输入材料中的来源线索
  - 在小节中列出对应的URL（至少1个；能列更多更好）
- 如果某章节输入材料不足以支持结论，必须显式写出“证据不足/缺乏直接证据”，并给出下一步检索建议；不得用泛泛叙述冒充结论。

# 写作风格
- 专业、客观、严谨。
- 确保信息准确，不捏造原文未提及的内容。
- 对提供的材料进行必要的去重、归纳和语言润色，使报告读起来是一个整体。""")

    user_prompt = HumanMessage(
        content=f"""请围绕以下研究主题，整合我已完成的各项子研究，撰写最终报告。

**研究主题**：{research_topic}

**各项子研究总结如下**：
{formatted_results}

请开始生成报告。"""
    )
    return system_prompt, user_prompt


def _generate_fallback_report(research_topic: str) -> str:
    """当没有任何有效结果时的降级报告。"""
    return f"""# 关于 {research_topic} 的研究报告

## 执行状态

系统未能成功收集到关于此主题的研究结果。这可能由于以下原因导致：
- 网络搜索未能返回相关信息。
- 在信息处理过程中发生意外错误。

建议您：
1.  检查研究主题的表述是否清晰、具体。
2.  确认网络连接及API服务状态。
3.  重新尝试执行研究流程。"""


def _generate_fallback_report_from_results(valid_results: List[dict], research_topic: str) -> str:
    """当LLM调用失败时，直接拼接所有结果作为降级报告。"""
    report = f"# 关于 {research_topic} 的研究报告\n\n"
    report += "> 注：此报告由系统直接整合生成，未经AI润色。\n\n"
    for result in valid_results:
        report += f"## {result.get('title', '')}\n\n"
        report += f"{result.get('summary', '')}\n\n"
        urls = result.get("source_urls") or result.get("sources") or []
        if urls:
            report += "**证据与来源（来自搜索结果链接）**:\n"
            for u in urls[:5]:
                report += f"- {u}\n"
            report += "\n"
    return report


def _mark_all_tasks_done(state: GraphState):
    """将所有子任务的状态标记为 DONE。"""
    for task in state["sub_tasks"]:
        task["status"] = TaskStatus.DONE
    logger.info("[报告节点] 所有子任务状态已更新为 DONE")