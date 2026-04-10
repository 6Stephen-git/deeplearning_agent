# src/nodes/aggregate_node.py
"""
聚合节点模块（集成定向深度研究引擎版）。
职责：收集并发任务结果，调用智能评估引擎进行任务级诊断，生成缺陷报告，并做出定向研究决策。
"""
import asyncio
import os
import math
import logging
from typing import List, Dict, Optional, Tuple, Any

from src.memory.memory_store import MemoryType, MemoryPriority
from src.state import GraphState, TaskStatus
from src.config import ensure_project_env_loaded
# 导入新评估模块
from src.evaluator.task_diagnoser import TaskDiagnoser, ResearchCycleDiagnoser
from src.evaluator.schemas import TaskQualityProfile, CycleDeficiencyReport
from src.tools.llm_client import llm_client

# LangSmith tracing（可选依赖：未安装时降级为 no-op）
try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        def _decorator(fn):  # type: ignore[no-untyped-def]
            return fn

        return _decorator

ensure_project_env_loaded()
logger = logging.getLogger(__name__)

# 从环境变量读取评估配置
# 基础评估权重
METRIC_VALIDITY_WEIGHT = float(os.getenv("METRIC_VALIDITY_WEIGHT", "0.15"))
METRIC_SATURATION_WEIGHT = float(os.getenv("METRIC_SATURATION_WEIGHT", "0.20"))
METRIC_DIVERSITY_WEIGHT = float(os.getenv("METRIC_DIVERSITY_WEIGHT", "0.25"))
METRIC_CONFLICT_WEIGHT = float(os.getenv("METRIC_CONFLICT_WEIGHT", "0.15"))
METRIC_FACT_DENSITY_WEIGHT = float(os.getenv("METRIC_FACT_DENSITY_WEIGHT", "0.15"))
METRIC_NOVELTY_WEIGHT = float(os.getenv("METRIC_NOVELTY_WEIGHT", "0.10"))

# 决策阈值
TASK_STORAGE_THRESHOLD = float(os.getenv("TASK_STORAGE_THRESHOLD", "70"))
MIN_TASK_STORAGE_THRESHOLD = float(os.getenv("MIN_TASK_STORAGE_THRESHOLD", "55"))
# 非线性衰减系数：第1轮下降最快，之后逐步变慢（对数归一化）。
# 该参数表示从第1轮到第2轮的阈值下降幅度（后续下降会自动变缓）。
TASK_STORAGE_THRESHOLD_DECAY_PER_CYCLE = float(os.getenv("TASK_STORAGE_THRESHOLD_DECAY_PER_CYCLE", "5"))

# 质量检测阈值（总体应小于记忆存储阈值），同样采用“先快后慢”的非线性衰减，
# 且衰减参数应大于记忆侧，表示更激进地降低质量门槛以避免后期空转。
CYCLE_CONTINUE_THRESHOLD = float(os.getenv("CYCLE_CONTINUE_THRESHOLD", "62"))
MIN_CYCLE_CONTINUE_THRESHOLD = float(os.getenv("MIN_CYCLE_CONTINUE_THRESHOLD", "55"))
CYCLE_THRESHOLD_DECAY_PER_CYCLE = float(os.getenv("CYCLE_THRESHOLD_DECAY_PER_CYCLE", "8"))
MAX_RESEARCH_CYCLES = int(os.getenv("MAX_RESEARCH_CYCLES", "3"))
RESEARCH_MIN_SUMMARY_LENGTH = int(os.getenv("RESEARCH_MIN_SUMMARY_LENGTH", "200"))


def _fast_then_slow_decay(base: float, min_val: float, step_drop: float, cycle_index: int) -> float:
    """
    非线性阈值衰减：第1轮下降快，后续下降逐步变慢。
    使用对数归一化，使得：
    - cycle_index=0 => drop=0
    - cycle_index=1 => drop=step_drop
    - 后续每增加1轮，增量下降幅度会变小（log 的边际递减）
    """
    i = max(0, int(cycle_index))
    if i == 0:
        return max(min_val, base)
    # 归一化：log(1+i) / log(2) 使 i=1 时比例为1
    ratio = math.log1p(i) / math.log(2)
    return max(min_val, base - step_drop * ratio)


@traceable(name="aggregate", run_type="chain")
async def aggregate_node(state: GraphState) -> GraphState:
    """
    聚合节点主函数（集成智能评估引擎）。
    1. 等待所有并发任务完成。
    2. 对每个任务进行细粒度质量诊断。
    3. 生成轮次缺陷报告与全局决策。
    4. 根据诊断结果筛选信息存入记忆。
    5. 为下一轮研究生成定向指令。
    """
    logger.info("[聚合节点] 开始收集与评估第%s轮研究结果", state.get("current_cycle", 1))

    if not state.get("active_tasks"):
        logger.info("[聚合节点] 无活跃任务，跳过聚合阶段")
        return state

    # 1. 等待所有并发任务完成
    try:
        task_result_list = await asyncio.gather(
            *state["active_tasks"],
            return_exceptions=True
        )
    except Exception as e:
        logger.exception("[聚合节点] 等待任务完成异常: %s", e)
        task_result_list = []

    # 2. 遍历结果，逐个处理并更新状态
    for task_return in task_result_list:
        if isinstance(task_return, Exception):
            logger.warning("[聚合节点] 任务执行失败: %s", task_return)
            continue
        if not isinstance(task_return, dict):
            logger.warning("[聚合节点] 收到非预期任务结果类型: %s", type(task_return))
            continue

        task_id = task_return.get("task_id")
        if not task_id:
            logger.warning("[聚合节点] 结果中缺少任务ID title=%s", task_return.get("title", "未知"))
            continue

        result_index = task_id - 1
        if 0 <= result_index < len(state["task_results"]):
            state["task_results"][result_index] = task_return
        else:
            logger.warning("[聚合节点] 任务ID超出结果列表范围 id=%s", task_id)
            continue

        for sub_task in state["sub_tasks"]:
            if sub_task["id"] == task_id:
                sub_task["status"] = TaskStatus.SUMMARIZED
                logger.info("[聚合节点] 任务结果已接收 id=%s title=%s", task_id, sub_task["title"])
                break

    # 3. 清空活跃任务列表
    state["active_tasks"].clear()
    valid_result_count = len([r for r in state["task_results"] if r and r.get("summary")])
    logger.info("[聚合节点] 结果收集完成，有效结果=%s/%s", valid_result_count, len(state["task_results"]))

    # ===== 硬早停：本轮完全无搜索证据（任意轮触发）=====
    # 口径：所有任务 search_results 为空（或 sources/source_urls 为空）=> 直接停止，不进入后续评分/深挖/报告
    try:
        total_tasks = len(state.get("task_results", []) or [])
        empty_hits = 0
        for r in (state.get("task_results", []) or []):
            if not isinstance(r, dict):
                continue
            sr = r.get("search_results", []) or []
            src = r.get("source_urls") or r.get("sources") or []
            if len(sr) == 0 or len(src) == 0:
                empty_hits += 1
        if total_tasks > 0 and empty_hits >= total_tasks:
            state["need_deeper_research"] = False
            state["report_allowed"] = False
            state["report_warning"] = None
            state["report_block_reason"] = (
                f"无可验证证据：本轮全部任务搜索结果为空/来源为空（{empty_hits}/{total_tasks}）。"
                f"为避免编造/张冠李戴，停止研究并不生成报告。"
            )
            logger.warning("[聚合节点] 触发硬早停: %s", state["report_block_reason"])
            return state
    except Exception as e:
        logger.exception("[聚合节点] 硬早停判定失败: %s", e)

    # ===== 初始化智能评估引擎 =====
    working_memory = state.get("working_memory")

    # 构建评估器配置（含动态阈值：轮次越大，阈值适度下调，避免后期因“新颖性下降/深挖变难”被固定阈值卡死）
    current_cycle = state.get("current_cycle", 1)
    cycle_index = max(0, current_cycle - 1)
    effective_cycle_threshold = _fast_then_slow_decay(
        base=CYCLE_CONTINUE_THRESHOLD,
        min_val=MIN_CYCLE_CONTINUE_THRESHOLD,
        step_drop=CYCLE_THRESHOLD_DECAY_PER_CYCLE,
        cycle_index=cycle_index,
    )
    logger.info(
        "[聚合节点] 动态继续阈值 base=%.1f cycle=%s step_drop=%.1f min=%.1f effective=%.1f",
        CYCLE_CONTINUE_THRESHOLD,
        current_cycle,
        CYCLE_THRESHOLD_DECAY_PER_CYCLE,
        MIN_CYCLE_CONTINUE_THRESHOLD,
        effective_cycle_threshold,
    )

    # 单条任务存入长期记忆的动态阈值（非线性衰减）
    storage_cycle_index = max(0, current_cycle - 1)
    effective_task_storage_threshold = _fast_then_slow_decay(
        base=TASK_STORAGE_THRESHOLD,
        min_val=MIN_TASK_STORAGE_THRESHOLD,
        step_drop=TASK_STORAGE_THRESHOLD_DECAY_PER_CYCLE,
        cycle_index=storage_cycle_index,
    )
    logger.info(
        "[聚合节点] 动态存储阈值 base=%.1f cycle=%s step_drop=%.1f min=%.1f effective=%.1f",
        TASK_STORAGE_THRESHOLD,
        current_cycle,
        TASK_STORAGE_THRESHOLD_DECAY_PER_CYCLE,
        MIN_TASK_STORAGE_THRESHOLD,
        effective_task_storage_threshold,
    )

    evaluator_config = {
        'metric_weights': {
            'validity': METRIC_VALIDITY_WEIGHT,
            'saturation': METRIC_SATURATION_WEIGHT,
            'diversity': METRIC_DIVERSITY_WEIGHT,
            'conflict': METRIC_CONFLICT_WEIGHT,
            'fact_density': METRIC_FACT_DENSITY_WEIGHT,
            'novelty': METRIC_NOVELTY_WEIGHT,
        },
        'storage_threshold': effective_task_storage_threshold,
        'cycle_continue_threshold': effective_cycle_threshold,
        'max_cycles': MAX_RESEARCH_CYCLES,
        'conflict_detection_enabled': llm_client is not None and METRIC_CONFLICT_WEIGHT > 0,
    }

    memory_store = working_memory.memory_store if working_memory else None
    task_diagnoser = TaskDiagnoser(
        llm_client=llm_client,
        memory_store=memory_store,
        config=evaluator_config
    )
    cycle_diagnoser = ResearchCycleDiagnoser(config=evaluator_config)

    # ===== 执行任务级并行诊断 =====
    research_topic = state.get("research_topic", "")
    task_profiles = []
    diagnostic_tasks = []

    logger.info("[聚合节点] 启动智能评估引擎，有效任务=%s", valid_result_count)

    for i, result in enumerate(state["task_results"]):
        if result and result.get("summary"):
            task_context = {
                "task_id": i + 1,
                "research_cycle": current_cycle,
                "research_topic": research_topic,
            }
            # 为每个任务创建诊断协程
            task = task_diagnoser.diagnose_task(result, task_context)
            diagnostic_tasks.append(task)

    if diagnostic_tasks:
        # 并行执行所有任务诊断
        try:
            task_profiles: List[TaskQualityProfile] = await asyncio.gather(*diagnostic_tasks)
            logger.info("[聚合节点] 任务级诊断完成，画像数量=%s", len(task_profiles))
        except Exception as e:
            logger.exception("[聚合节点] 任务诊断异常: %s", e)
            task_profiles = []
    else:
        logger.info("[聚合节点] 无有效任务可供诊断")
        task_profiles = []

    # ===== 生成轮次缺陷报告与全局决策 =====
    deficiency_report: CycleDeficiencyReport = cycle_diagnoser.generate_deficiency_report(
        profiles=task_profiles,
        current_cycle=current_cycle
    )

    # 打印详细的评估报告
    logger.info("[聚合节点] 第%s轮研究质量评估完成", current_cycle)

    # ===== 根据诊断结果筛选并存储到长期记忆 =====
    if working_memory and task_profiles:
        try:
            stored_count = 0
            immediate_count = 0
            verify_count = 0
            reject_count = 0

            for profile in task_profiles:
                # 新机制：单条任务结果达到阈值即存入长期记忆（保留高质量输出）
                if profile.composite_score >= effective_task_storage_threshold:
                    # 获取对应的原始结果
                    result_index = profile.task_id - 1
                    if 0 <= result_index < len(state["task_results"]):
                        result = state["task_results"][result_index]
                        if result and "summary" in result:
                            # confidence：与综合分挂钩（composite_score 为 0–100），映射到 (0,1]，
                            # 避免凡入库都写死 0.9，便于后续按 min_confidence 过滤或人工区分质量。
                            confidence = max(0.01, min(1.0, float(profile.composite_score) / 100.0))
                            # 存入长期记忆
                            working_memory.store_important_findings(
                                content=result["summary"],
                                memory_type=MemoryType.TASK_SUMMARY,
                                priority=MemoryPriority.HIGH,
                                research_topic=research_topic,
                                confidence=confidence,
                                metadata={
                                    "task_id": profile.task_id,
                                    "composite_score": profile.composite_score,
                                    "research_cycle": current_cycle,
                                    "diagnostic_tags": profile.tags,
                                }
                            )
                            stored_count += 1
                            immediate_count += 1
                            logger.info("[聚合节点] 任务因高质量入库 id=%s score=%.1f", profile.task_id, profile.composite_score)

                elif profile.storage_suggestion == "VERIFY":
                    verify_count += 1
                    logger.info("[聚合节点] 任务建议核实后存储 id=%s score=%.1f", profile.task_id, profile.composite_score)
                elif profile.storage_suggestion == "REJECT":
                    reject_count += 1
                    validity_score = profile.get_metric_score('validity')
                    if validity_score < 50:
                        logger.info("[聚合节点] 任务因无效被丢弃 id=%s validity=%.1f", profile.task_id, validity_score)
                    else:
                        logger.info("[聚合节点] 任务因低分被丢弃 id=%s score=%.1f", profile.task_id, profile.composite_score)

            if stored_count > 0:
                logger.info("[聚合节点] 高质量发现已入库 count=%s immediate=%s verify=%s", stored_count, immediate_count, verify_count)

        except Exception as e:
            logger.exception("[聚合节点] 存储到工作记忆失败: %s", e)

    # ===== 更新状态：注入评估结果与决策 =====
    # 将任务质量画像列表存入状态
    state["task_quality_profiles"] = [profile.to_dict() for profile in task_profiles]

    # 将“评分卡”写入状态，便于 LangSmith trace 查看（仅原始分数，不含证据/理由，避免过大）
    # 结构示例：
    # [
    #   {"task_id": 1, "composite_score": 72.3, "metrics": {"validity": 70.0, "saturation": 55.2, ...}},
    #   ...
    # ]
    scorecard: List[Dict[str, Any]] = []
    for profile in task_profiles:
        scorecard.append(
            {
                "task_id": int(profile.task_id),
                "research_cycle": int(profile.research_cycle),
                "composite_score": float(profile.composite_score or 0.0),
                "metrics": {
                    name: float(ms.score)
                    for name, ms in (profile.metrics or {}).items()
                },
            }
        )
    state["task_metric_scores"] = scorecard

    # 将缺陷报告存入状态
    state["deficiency_report"] = deficiency_report.report_text
    state["last_cycle_score"] = deficiency_report.average_composite_score

    # ===== 无证据早停（任意轮触发）=====
    # 判定口径：以 evidence_alignment（门控指标，0/100）为准；有效任务比例 < 20% 视为“无证据”
    evidence_early_stop = False
    try:
        valid_count = 0
        total = len(task_profiles)
        for p in task_profiles:
            if p.get_metric_score("evidence_alignment") >= 50:
                valid_count += 1
        pass_rate = (valid_count / total) if total else 0.0
        state["evidence_pass_rate"] = pass_rate  # 方便 LangSmith 审计
        if total > 0 and pass_rate < 0.20:
            evidence_early_stop = True
            state["report_warning"] = None
            state["report_allowed"] = False
            state["report_block_reason"] = (
                f"无可验证证据：本轮证据对齐通过率 {pass_rate*100:.1f}% (<20%)。"
                f"为避免编造/张冠李戴，停止研究并不生成报告。"
            )
            logger.warning("[聚合节点] 触发证据早停: %s", state["report_block_reason"])
    except Exception as e:
        logger.exception("[聚合节点] 早停判定失败: %s", e)

    # ===== 最大轮次报告门控（按阈值差距分档）=====
    # 规则：
    # - 低于阈值 0~5 分：正常输出报告
    # - 低于阈值 5~10 分：输出报告并提醒“报告质量有待考量”
    # - 低于阈值 >10 分：不输出报告，提示需要重新研究
    max_cycles = state.get("max_cycles", MAX_RESEARCH_CYCLES)
    threshold_used = float(evaluator_config.get("cycle_continue_threshold", CYCLE_CONTINUE_THRESHOLD))
    avg_score = float(deficiency_report.average_composite_score or 0.0)
    gap = max(0.0, threshold_used - avg_score)

    # 注意：不得覆盖「无证据早停」已写入的 report_allowed / report_block_reason
    if not evidence_early_stop:
        state["report_allowed"] = True
        state["report_warning"] = None
        state["report_block_reason"] = None

    if current_cycle >= max_cycles and not evidence_early_stop:
        # 到达最大轮次时才启用该门控策略（不得覆盖无证据早停）
        if gap <= 5.0:
            state["report_allowed"] = True
        elif gap <= 10.0:
            state["report_allowed"] = True
            state["report_warning"] = f"平均分 {avg_score:.1f} 低于阈值 {threshold_used:.1f} 约 {gap:.1f} 分，报告质量有待考量。"
        else:
            state["report_allowed"] = False
            state["report_block_reason"] = (
                f"平均分 {avg_score:.1f} 低于阈值 {threshold_used:.1f} 约 {gap:.1f} 分，"
                f"报告质量差，建议调整主题表述/检索策略后重新研究。"
            )
        logger.info(
            "[聚合节点] 最大轮次报告门控 avg=%.1f threshold=%.1f gap=%.1f allowed=%s",
            avg_score,
            threshold_used,
            gap,
            state["report_allowed"],
        )

    # 生成定向研究指令（基于缺陷分类和具体评分）
    targeted_instructions = []
    for task_id in deficiency_report.get_tasks_by_category("has_conflicts"):
        # 查找对应的质量画像
        profile = next((p for p in task_profiles if p.task_id == task_id), None)
        if profile:
            targeted_instructions.append({
                "task_id": task_id,
                "type": "CLARIFY_CONFLICT",
                "suggestion": f"针对已识别的争议点，使用搜索词如 'debate controversy pros and cons' 来探索正反双方论据。原查询: {state['sub_tasks'][task_id - 1].get('query', '')}",
                "original_query": state['sub_tasks'][task_id - 1].get('query', '') if task_id - 1 < len(
                    state['sub_tasks']) else ""
            })

    # 保留基于来源多样性的优化建议，但不再通过特定缺陷类别触发；
    # 这里直接遍历已有的任务质量画像，根据 diversity 指标生成可选优化指令。
    for profile in task_profiles:
        diversity_score = profile.get_metric_score('diversity')
        if diversity_score < 40:
            if diversity_score < 20:
                suggestion = f"可进一步补充权威和多样化来源(评分:{diversity_score:.1f})，例如在查询中添加 'site:.edu OR site:.gov OR site:.org'。"
            else:
                suggestion = f"来源多样性有提升空间(评分:{diversity_score:.1f})，可尝试添加不同机构限定词，如 'university research institute industry'。"

            targeted_instructions.append({
                "task_id": profile.task_id,
                "type": "OPTIMIZE_SOURCES",
                "suggestion": suggestion,
                "original_query": state['sub_tasks'][profile.task_id - 1].get('query', '') if profile.task_id - 1 < len(
                    state['sub_tasks']) else "",
                "diversity_score": diversity_score
            })

    for task_id in deficiency_report.get_tasks_by_category("needs_depth"):
        profile = next((p for p in task_profiles if p.task_id == task_id), None)
        if profile:
            saturation_score = profile.get_metric_score('saturation')
            if saturation_score < 40:
                suggestion = f"信息深度严重不足(评分:{saturation_score:.1f})。在查询中添加 'technical details implementation case study' 获取具体信息。"
            else:
                suggestion = f"信息深度不足(评分:{saturation_score:.1f})。添加 'how to implement best practices examples' 获取更实用信息。"

            targeted_instructions.append({
                "task_id": task_id,
                "type": "DEEPEN_ANALYSIS",
                "suggestion": suggestion,
                "original_query": state['sub_tasks'][task_id - 1].get('query', '') if task_id - 1 < len(
                    state['sub_tasks']) else "",
                "saturation_score": saturation_score
            })

    # 针对“难以检索/结果稀少/强比较过窄导致低分”的任务，生成拓宽范围指令
    # 触发条件（任一命中即可）：
    # - 搜索结果为空/来源为空 且 综合评分很低
    # - 搜索结果很少/来源很少/多样性极低，且综合评分偏低
    # - 标题明显是强比较/强绑定两个实体（如 A 与 B 对比/比较分析），且综合评分偏低
    for profile in task_profiles:
        try:
            result_index = profile.task_id - 1
            if not (0 <= result_index < len(state.get("task_results", []))):
                continue
            result = state["task_results"][result_index] or {}
            search_results = result.get("search_results", []) or []
            sources = result.get("sources", []) or []

            composite = profile.composite_score if profile.composite_score is not None else 0.0
            diversity = profile.get_metric_score("diversity")

            # 获取标题/查询用于判定“过窄强比较”
            task_title = ""
            original_query = ""
            if profile.task_id - 1 < len(state.get("sub_tasks", [])):
                task_title = (state["sub_tasks"][profile.task_id - 1].get("title", "") or "").strip()
                original_query = (state["sub_tasks"][profile.task_id - 1].get("query", "") or "").strip()

            compare_markers = ("比较", "对比", "对照", "vs", "VS", "与")
            is_strong_compare = any(m in task_title for m in compare_markers) and ("比较" in task_title or "对比" in task_title or "vs" in task_title.lower())

            # 空/稀少结果信号（常见于关键词过窄、强限定、或绑定两个实体导致可检索性差）
            empty_signal = (len(search_results) == 0) or (len(sources) == 0)
            sparse_signal = (len(search_results) <= 1) or (len(sources) <= 1) or (diversity is not None and diversity < 20)

            should_broaden = (
                (empty_signal and composite < 55) or
                (sparse_signal and composite < 65) or
                (is_strong_compare and composite < 70 and sparse_signal)
            )

            if should_broaden:
                original_query = ""
                if profile.task_id - 1 < len(state.get("sub_tasks", [])):
                    original_query = state["sub_tasks"][profile.task_id - 1].get("query", "") or ""

                targeted_instructions.append({
                    "task_id": profile.task_id,
                    "type": "BROADEN_SCOPE",
                    "suggestion": (
                        "上一轮检索结果偏少/为空导致质量受限。下一轮请适度拓宽限定范围，并优先采用“复合约束拆解”策略："
                        "如果原任务同时绑定多个专有名词/实体、多个限定条件或强对比要求，先拆成若干更可检索的子任务"
                        "（分别覆盖关键实体/子概念/上位概念/代表性案例/权威框架），再在后续任务中进行整合、对照或归纳。"
                        "查询可引入 framework/guidelines/standards/policy/requirements/case study 等通用学术词汇，并避免过强同时限定。"
                    ),
                    "original_query": original_query,
                    "composite_score": composite,
                    "empty_results": len(search_results) == 0,
                    "empty_sources": len(sources) == 0,
                    "sparse_results": len(search_results) <= 1,
                    "sparse_sources": len(sources) <= 1,
                    "diversity_score": diversity,
                    "task_title": task_title
                })
        except Exception:
            # 诊断/结果结构异常时不阻断主流程
            continue

    state["targeted_instructions"] = targeted_instructions

    # ===== 做出全局循环决策 =====
    max_cycles = state.get("max_cycles", MAX_RESEARCH_CYCLES)
    need_deeper = False
    decision_reason = ""

    if evidence_early_stop:
        need_deeper = False
        decision_reason = state.get("report_block_reason") or "无可验证证据，已早停。"
    elif current_cycle >= max_cycles:
        decision_reason = f"已达到最大研究轮次 ({max_cycles})。"
    elif deficiency_report.global_continue_suggestion:
        need_deeper = True
        # 在这里递增轮次
        state["current_cycle"] += 1
        if deficiency_report.average_composite_score < evaluator_config.get("cycle_continue_threshold", CYCLE_CONTINUE_THRESHOLD):
            threshold_used = evaluator_config.get("cycle_continue_threshold", CYCLE_CONTINUE_THRESHOLD)
            decision_reason = f"轮次平均分{deficiency_report.average_composite_score:.1f}未达阈值{threshold_used:.1f}。"
        else:
            decision_reason = f"存在需要深入解决的缺陷：{deficiency_report.primary_deficiency}。"
    else:
        decision_reason = "研究质量已达标，且无显著缺陷需深入。"

    state["need_deeper_research"] = need_deeper

    if need_deeper:
        # 注意：state["current_cycle"] 已经在上面递增了
        logger.info("[聚合节点] 决策: %s 将进入第%s轮", decision_reason, state["current_cycle"])
        if targeted_instructions:
            logger.info("[聚合节点] 已生成定向研究指令 count=%s", len(targeted_instructions))
    else:
        logger.info("[聚合节点] 决策: %s，流程结束进入报告阶段", decision_reason)
    return state