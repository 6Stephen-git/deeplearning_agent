# src/graph.py
"""
主图定义模块。
使用 LangGraph 构建深度研究助手的核心工作流，实现"规划->执行->聚合->报告"的循环条件逻辑。
"""
from typing import Literal
import logging
from langgraph.graph import StateGraph, END
from src.state import GraphState, TaskStatus
from src.nodes.planning_node import plan_node
from src.nodes.execution_node import execute_node
from src.nodes.aggregate_node import aggregate_node
from src.nodes.report_node import report_node
from src.nodes.initialize_memory_node import initialize_memory_node
from src.nodes.no_report_node import no_report_node

logger = logging.getLogger(__name__)


def create_research_graph() -> StateGraph:
    """
    创建并返回配置完整的深度研究助手工作流图。

    图结构：
        [Start] → initialize_memory_node → plan_node → has_tasks? → execute_node → aggregate_node
                    ↑                                              |
                    |                                              v
                    ----------- need_deeper? <---------- should_continue?
                                                              |
                                                              v
                                                    report_node → [End]
    """
    # 1. 初始化一个以 GraphState 为状态类型的 StateGraph，确保每个节点接收StateGraph作为输入，并返回更新后的StateGraph
    workflow = StateGraph(GraphState)
    logger.info("[主图] 初始化 StateGraph")

    # 2. 添加节点
    workflow.add_node("initialize_memory", initialize_memory_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("report", report_node)
    workflow.add_node("no_report", no_report_node)
    logger.info("[主图] 所有功能节点添加完毕")

    # 3. 设置入口点
    workflow.set_entry_point("initialize_memory")
    logger.info("[主图] 设置入口点 initialize_memory")

    # 4. 定义条件路由函数
    def _has_pending_tasks(state: GraphState) -> Literal["execute", "report"]:  # 类型注解，指定返回值只能是这两个字符串之一。
        """
        条件路由：判断是否有待处理的任务。
        在 `plan` 节点之后调用，决定是进入 `execute` 节点执行任务，还是直接跳到 `report` 节点。

        Returns:
            "execute": 如果存在状态为 PENDING 的子任务。
            "report":  如果所有子任务均已完成（或初始无任务）。
        """
        pending_tasks = [t for t in state.get("sub_tasks", []) if t["status"] == TaskStatus.PENDING]
        has_tasks = len(pending_tasks) > 0

        if has_tasks:
            logger.info("[条件路由 has_tasks] 有待处理任务=%s，前往 execute", len(pending_tasks))
            return "execute"
        else:
            logger.info("[条件路由 has_tasks] 无待处理任务，前往 report")
            return "report"

    def _should_continue_research(state: GraphState) -> Literal["plan", "report", "no_report"]:
        """
        条件路由：判断当前轮次研究后，是否需要进行更深入的研究。
        在 `aggregate` 节点之后调用，决定是回到 `execute` 节点开始新一轮研究，还是前往 `report` 节点结束。

        Returns:
            "plan":   如果 `need_deeper_research` 标志为 True，且未达到最大轮次限制，则进入下一轮重新规划。
            "report": 如果不需要或不能再进行深入研究。
        """
        need_deeper = state.get("need_deeper_research", False)
        current_cycle = state.get("current_cycle", 1)
        max_cycles = state.get("max_cycles", 5)
        report_allowed = state.get("report_allowed", True)

        # 注意：current_cycle 可能已在聚合节点内递增到“下一轮”的编号；
        # 因此这里用 <= 允许进入最大轮次那一轮的规划与执行，确保能产出最后一轮评分。
        if need_deeper and current_cycle <= max_cycles:
            logger.info("[条件路由 should_continue] 当前第%s轮，需要深入研究，前往 plan", current_cycle)
            return "plan"
        else:
            if not report_allowed:
                logger.info("[条件路由 should_continue] 报告被拦截，前往 no_report")
                return "no_report"
            logger.info("[条件路由 should_continue] 研究完成，前往 report")
            return "report"

    # 5. 添加边（定义节点间的流转关系）

    # 5.1 从 initialize_memory 到 plan
    workflow.add_edge("initialize_memory", "plan")

    # 5.2 从 `plan` 节点出来后，根据条件路由 `_has_pending_tasks` 决定去哪
    workflow.add_conditional_edges(
        "plan",
        _has_pending_tasks,
        {
            "execute": "execute",  # 有待处理任务 -> 执行节点
            "report": "report",  # 无任务 -> 直接生成报告
        }
    )
    # 5.3 `execute` 节点之后总是到 `aggregate` 节点（收集并发结果）
    workflow.add_edge("execute", "aggregate")
    # 5.4 `aggregate` 节点之后，根据条件路由 `_should_continue_research` 决定循环还是结束
    workflow.add_conditional_edges(
        "aggregate",
        _should_continue_research,  # 条件边的条件判断函数
        {
            "plan": "plan",  # 需要深入研究 -> 回到规划节点（开启新一轮，重规划子任务）
            "report": "report",  # 研究完成 -> 前往报告节点
            "no_report": "no_report",  # 不生成报告 -> 输出提示后结束
        }
    )
    # 5.5 `report` 节点是终点，连接到 END
    workflow.add_edge("report", END)
    workflow.add_edge("no_report", END)

    logger.info("[主图] 节点与条件边添加完成")
    return workflow


# 编译图，创建可执行的应用实例
research_graph = create_research_graph()
app = research_graph.compile()

logger.info("深度研究助手 LangGraph 工作流编译成功")
