# src/nodes/execution_node.py
"""
异步执行节点模块。
使用真实的异步搜索工具 (`async_search_tool`) 并发执行子任务。
"""
import asyncio
import logging
import os
from typing import List, Dict, Any
from src.state import GraphState, TaskStatus
from src.tools.llm_client import llm_client
from src.tools.async_search_tool import async_search_tool
from src.config import ensure_project_env_loaded
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        def _decorator(fn):  # type: ignore[no-untyped-def]
            return fn
        return _decorator

ensure_project_env_loaded()
logger = logging.getLogger(__name__)


def _task_timeout_s() -> float:
    raw = os.getenv("EXECUTION_TASK_TIMEOUT_S", "120")
    try:
        return max(5.0, float(raw))
    except Exception:
        return 120.0


@traceable(name="execute", run_type="chain")
async def execute_node(state: GraphState) -> GraphState:
    """
    异步执行节点主函数。
    并发启动所有待处理子任务，不等待其完成。

    Args:
        state: 当前图状态，应包含已规划的 `sub_tasks`。

    Returns:
        更新后的状态，主要更新了：
        - `sub_tasks[*].status`: 从 PENDING 变为 RUNNING
        - `active_tasks`: 填充了并发异步任务的句柄列表
    """

    # 获取当前轮次
    current_cycle = state.get("current_cycle", 1)
    logger.info("[执行节点] 开始执行，研究轮次=%s", current_cycle)

    # 重置任务状态（如果需要）
    if state.get("need_deeper_research", False):
        state["need_deeper_research"] = False
        # 重置任务状态：将上一轮完成的任务状态重置为PENDING
        for task in state.get("sub_tasks", []):
            if task.get("status") == TaskStatus.SUMMARIZED:
                task["status"] = TaskStatus.PENDING

    # 筛选出所有待处理的任务
    pending_tasks = [t for t in state["sub_tasks"] if t["status"] == TaskStatus.PENDING]
    if not pending_tasks:
        logger.info("[执行节点] 无待处理任务，跳过执行阶段")
        return state

    logger.info("[执行节点] 发现待处理任务=%s，开始并发执行", len(pending_tasks))

    active_tasks = []
    timeout_s = _task_timeout_s()
    for task in pending_tasks:
        async_task = asyncio.create_task(
            _execute_single_task_with_timeout(task, state["current_cycle"], timeout_s),
            name=f"Task-{task['id']}"
        )
        active_tasks.append(async_task)
        task["status"] = TaskStatus.RUNNING
        logger.info("[执行节点] 已发起任务 id=%s title=%s", task["id"], task["title"])

    state["active_tasks"].extend(active_tasks)
    logger.info("[执行节点] 并发任务已发起，总数=%s timeout_s=%.1f", len(active_tasks), timeout_s)
    return state


async def _execute_single_task_with_timeout(
    task: Dict[str, Any],
    research_cycle: int,
    timeout_s: float,
) -> Dict[str, Any]:
    task_id = task["id"]
    try:
        return await asyncio.wait_for(
            _execute_single_task(task, research_cycle),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("[任务-%s] 执行超时 timeout_s=%.1f", task_id, timeout_s)
        return {
            "task_id": task_id,
            "title": task.get("title", ""),
            "search_results": [],
            "summary": f"## 执行超时\n\n任务处理超过 {timeout_s:.1f}s，已自动终止。",
            "sources": [],
            "source_urls": [],
            "quality_score": None,
            "research_cycle": research_cycle,
            "timeout": True,
            "error_type": "timeout",
        }


@traceable(name="execute_single_task", run_type="chain")
async def _execute_single_task(task: Dict[str, Any], research_cycle: int) -> Dict[str, Any]:
    """
    执行单个子任务的完整异步流程：搜索 -> 总结。
    使用真实的 `async_search_tool` 进行搜索。

    Args:
        task: 子任务字典，包含 id, title, query 等信息。
        research_cycle: 当前研究轮次。

    Returns:
        该子任务的结果字典。
    """
    task_id = task["id"]
    logger.info("[任务-%s] 开始执行 title=%s", task_id, task.get("title", ""))

    result = {
        "task_id": task_id,
        "title": task["title"],
        "search_results": [],
        "summary": "",
        "sources": [],
        "source_urls": [],
        "quality_score": None,
        "research_cycle": research_cycle,
    }

    try:
        # 步骤 1: 调用真实的异步搜索工具
        search_query = task["query"]
        search_response = await async_search_tool.ainvoke(search_query)

        if search_response.get("error"):
            raise RuntimeError(search_response["error"])

        search_results = search_response.get("results", [])
        result["search_results"] = search_results
        urls = [r["url"] for r in search_results if "url" in r]
        result["sources"] = urls
        result["source_urls"] = urls
        logger.info("[任务-%s] 搜索完成 result_count=%s", task_id, len(search_results))

        # 步骤 2: 异步总结
        if search_results:
            summary = await _summarize_search_results(task, search_results)
            result["summary"] = summary
            logger.info("[任务-%s] 总结完成 summary_len=%s", task_id, len(summary))
        else:
            result["summary"] = "## 无相关信息\n\n未能找到相关信息。"
            logger.warning("[任务-%s] 未获得搜索结果", task_id)

    except asyncio.CancelledError:
        logger.warning("[任务-%s] 任务被取消", task_id)
        raise
    except Exception as e:
        logger.exception("[任务-%s] 执行异常: %s", task_id, e)
        result["summary"] = f"## 执行错误\n\n任务处理失败: {str(e)}"
        result["error_type"] = type(e).__name__
        result["timeout"] = False
    finally:
        logger.info("[任务-%s] 执行结束", task_id)
        return result


async def _summarize_search_results(task: Dict[str, Any], search_results: List[Dict]) -> str:
    """
    调用LLM总结搜索结果。
    """
    system_prompt = SystemMessage(
        content=(
            "你是一个严谨的信息总结专家。\n"
            "你只能基于提供的【搜索结果】进行总结，不得编造、不得补全不存在的事实、不得将不同实体/姓名/机构张冠李戴。\n"
            "若搜索结果不足以支撑某个结论，请明确写“无法从给定证据确认/暂无公开证据”。\n"
            "所有关键结论都必须带引用标号（如 [1], [2]），且引用必须来自下方对应编号的搜索结果。\n"
        )
    )

    # 为降低超时概率，限制输入规模并截断 snippet
    formatted_results = ""
    for i, res in enumerate((search_results or [])[:3], 1):
        title = res.get("title", "无标题")
        snippet = (res.get("snippet", "无摘要") or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
        url = res.get("url", "无链接")
        formatted_results += f"[{i}] {title}\n"
        formatted_results += f"   摘要: {snippet}\n"
        formatted_results += f"   链接: {url}\n\n"

    user_prompt = HumanMessage(
        content=f"请针对以下研究子问题，总结提供的资料：\n\n"
                f"**子问题**：{task['title']}\n"
                f"**研究意图**：{task['intent']}\n\n"
                f"**搜索结果**：\n{formatted_results}\n"
                f"请生成Markdown格式的总结："
    )

    try:
        response = await llm_client.agenerate(messages=[[system_prompt, user_prompt]])
        return response.generations[0][0].text
    except Exception as e:
        # 降级：LLM 超时/失败时，直接基于 snippet 生成可用摘要，避免任务整体变成“执行错误”
        fallback = f"# {task.get('title', '')}\n\n"
        fallback += "## 摘要（降级生成：LLM总结失败）\n\n"
        fallback += f"- 研究意图：{task.get('intent', '')}\n"
        fallback += f"- 失败原因：{str(e)[:120]}\n\n"
        fallback += "## 关键信息线索（来自搜索摘要）\n\n"
        for i, res in enumerate((search_results or [])[:3], 1):
            fallback += f"- [{i}] {res.get('title','无标题')}（{res.get('url','')}）\n"
            snip = (res.get("snippet", "") or "").strip().replace("\n", " ")
            if snip:
                fallback += f"  - {snip[:300]}{'...' if len(snip) > 300 else ''}\n"
        return fallback
