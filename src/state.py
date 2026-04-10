# src/state.py
"""
状态定义模块。
定义深度研究助手LangGraph工作流的核心状态（State）结构。
此状态对象将在所有图节点间流转，承载全部输入、中间结果和最终输出。
"""
from typing import TypedDict, List, Optional, Any, Annotated
from langgraph.graph.message import add_messages
import asyncio
from enum import Enum


class TaskStatus(Enum):
    """子任务状态枚举，明确任务生命周期。"""
    PENDING = "pending"
    """等待执行"""
    RUNNING = "running"
    """正在执行（用于监控）"""
    SUMMARIZED = "summarized"
    """已完成总结"""
    NEEDS_DEEPER = "needs_deeper"
    """需要深入研究（由反思节点设置）"""
    DONE = "done"
    """最终完成，准备用于报告生成"""


class GraphState(TypedDict):
    """
    深度研究助手的核心状态定义。
    这是整个LangGraph工作流中流转的唯一上下文对象。
    所有节点的读取和修改都必须基于此结构中定义的字段。
    """
    # ========== 输入层 Input Layer ==========
    research_topic: str
    """用户输入的研究主题。此字段由工作流入口初始化。"""

    # ========== 规划层 Planning Layer ==========
    sub_tasks: List[dict]
    """
    由`规划节点`生成的子任务列表。
    每个任务是一个字典，推荐结构如下：
    {
        "id": int,                    # 任务唯一ID，从1开始递增
        "title": str,                 # 任务标题（中文，用于前端展示和报告）
        "intent": str,                # 研究意图，说明为何要研究此子问题
        "query": str,                 # 搜索查询词（优化后的英文关键词，以提高搜索结果质量）
        "status": TaskStatus,         # 任务当前状态，用于控制流
        "assigned_agent": Optional[str], # 预留字段，用于未来扩展多智能体路由
    }
    注意：此列表的顺序即为初始规划的逻辑顺序，但异步执行不保证完成顺序。
    """

    # ========== 异步执行层 Async Execution Layer ==========
    active_tasks: List[asyncio.Task]
    """
    用于管理并发执行的异步任务（asyncio.Task）句柄列表。
    **关键设计**：在并发执行阶段，`执行节点`将为每个`PENDING`状态的任务创建一个异步任务
    （该任务封装了`搜索->总结`的完整流程），并将其`Task`对象追加到此列表。
    随后，`聚合节点`将`await`此列表中所有任务，并收集它们的结果。
    此机制是实现高效I/O并发的核心。
    """

    # ========== 结果存储层 Result Storage Layer ==========
    task_results: List[Optional[dict]]
    """
    所有子任务的结果存储列表。列表索引与`task_id-1`对应。
    每个元素对应一个子任务的最终结果，初始为`None`，任务完成后填充为字典，结构如下：
    {
        "task_id": int,               # 对应子任务ID
        "title": str,                 # 任务标题
        "search_results": List[dict], # 原始搜索结果，每个结果包含title, url, snippet等
        "summary": str,               # 该任务的文本总结（Markdown格式）
        "sources": List[str],         # 所引用的来源URL列表，用于报告参考文献
        "quality_score": Optional[float], # 反思节点给出的质量评分（0-1），用于评估信息完备性
        "research_cycle": int,        # 此结果是在第几轮研究中产生的（用于追踪深度）
    }
    **设计原因**：使用与`sub_tasks`平行的列表存储结果，避免了在异步并发中修改`sub_tasks`字典可能导致的竞争条件，
    并通过索引实现O(1)时间复杂度的结果查找与更新。
    """

    # ========== 控制流层 Control Flow Layer ==========
    need_deeper_research: bool
    """
    全局反思标志。
    当任一子任务在`反思节点`中被判定为信息不足（`quality_score`过低）时，此标志被设置为`True`。
    控制流会根据此标志决定是否启动新一轮的深入研究循环。
    """
    current_cycle: int
    """当前研究循环的轮次（从0开始）。用于限制最大研究深度，防止无限循环。"""
    max_cycles: int
    """最大研究轮次限制。可从配置读取，默认为3，作为安全护栏。"""

    # ========== 输出层 Output Layer ==========
    final_report: Optional[str]
    """由`报告生成节点`生成的最终Markdown格式研究报告。初始为None，工作流完成后填充。"""

    # ========== LangGraph 内置特性集成层 ==========
    messages: Annotated[List[Any], add_messages]
    """
    LangGraph 的消息历史。
    **最佳实践**：将所有与LLM的交互（如规划、总结、反思的Prompt和Completion）通过`add_messages`自动记录于此。
    优势：
    1.  LangGraph会自动管理此列表的上下文窗口，在调用LLM时将其作为历史消息传入。
    2.  为整个工作流提供了完整的、结构化的可观察性，极大方便调试和审计追溯。
    3.  无需手动拼接和管理对话历史。
    """

    working_memory: Optional[Any]  # 存储WorkingMemory实例
    """当前研究会话的工作记忆实例，由initialize_memory_node创建"""

    # (deprecated duplicated block removed)

    # ========== 智能评估层 Intelligent Evaluation Layer ==========
    task_quality_profiles: List[Optional[dict]]
    """
    由`aggregate_node`中的TaskDiagnoser生成的任务质量画像列表。
    每个元素是一个字典，包含任务ID、各项指标分数、综合评分、标签和建议。
    列表索引与`task_results`和`sub_tasks`对应。
    """

    deficiency_report: Optional[str]
    """
    由`aggregate_node`中的ResearchCycleDiagnoser生成的轮次缺陷报告。
    人类可读的文本，总结了本轮研究的主要问题和改进方向。
    """

    # Report gating / early-stop
    report_allowed: bool
    """Whether generating final report is allowed."""

    report_block_reason: Optional[str]
    """Reason when report_allowed is False."""

    report_warning: Optional[str]
    """Soft warning when report is allowed but quality is borderline."""

    task_metric_scores: List[dict]
    """Compact task scorecards for tracing/debugging."""

    evidence_pass_rate: float
    """Evidence alignment pass rate in current cycle."""

    targeted_instructions: List[dict]
    """
    定向研究指令列表。
    每个指令是一个字典，格式如：
    {
        "task_id": int,           # 针对哪个任务
        "type": str,             # 指令类型，如"EXPAND_SOURCES", "CLARIFY_CONFLICT"
        "suggestion": str        # 具体建议
    }
    用于指导下一轮`plan_node`生成更有针对性的子任务。
    """

    last_cycle_score: Optional[float]
    """上一轮研究的平均综合质量评分，用于追踪进度。"""
