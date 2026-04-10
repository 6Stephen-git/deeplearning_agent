# src/nodes/planning_node.py
"""
规划节点模块（支持定向深度研究）。
利用工作记忆、历史评估结果，生成智能的、有针对性研究子任务。
"""
import json
import os
import logging
from datetime import datetime
from typing import List, Any
from src.state import GraphState, TaskStatus
from src.tools.llm_client import llm_client, get_enhancer_llm_client
from langchain_core.messages import HumanMessage, SystemMessage
from src.memory.query_enhancer import HyDEEnhancer, MQEEnhancer
from src.config import ensure_project_env_loaded

try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        def _decorator(fn):  # type: ignore[no-untyped-def]
            return fn
        return _decorator

ensure_project_env_loaded()
logger = logging.getLogger(__name__)

def _planning_debug_enabled() -> bool:
    return os.getenv("SHOW_PLANNING_DEBUG", "false").lower() == "true"

def _recent_years_hint() -> str:
    """近2~3年的年份提示（动态），用于搜索查询约束。"""
    y = datetime.now().year
    return f"{y-2}, {y-1}, {y}"


@traceable(name="plan", run_type="chain")
async def plan_node(state: GraphState) -> GraphState:
    """
    规划节点主函数（支持定向深度研究）。
    基于工作记忆、历史评估结果，生成（或调整）下一轮研究的子任务列表。
    """
    research_topic = state["research_topic"]
    prev_sub_tasks_snapshot = list(state.get("sub_tasks", []) or [])
    logger.info("[规划节点] 开始第%s轮研究规划: %s", state.get("current_cycle", 1), research_topic)

    # 全局反幻觉/证据真实性约束：会注入到规划提示中（影响子任务 query/intent 的生成）
    evidence_integrity_rules = """
## ✅ 证据真实性与正确性（最高优先级，不得违背）
- 若缺乏公开证据，应明确承认“未找到/无法确认”，宁可不给结论也不要编造。
- 严禁张冠李戴：不同姓名/机构/地点即便相似也不得互相替代；如出现不一致实体，必须在任务中显式区分并提示需核验。
- 生成 query 时必须覆盖研究主题中的关键实体/关键词（含专业术语），避免过度泛化导致错配。
"""

    working_memory = state.get("working_memory")
    memory_context = ""

    if working_memory:
        # 1. 根据环境变量决定启用哪种增强策略
        enable_hyde = os.getenv("ENABLE_HYDE", "true").lower() == "true"
        enable_mqe = os.getenv("ENABLE_MQE", "false").lower() == "true"

        # 初始化增强器实例
        query_enhancer = None

        if enable_hyde and enable_mqe:
            # 如果两者都开启，发出警告并默认优先使用 HyDE (可调整此策略)
            logger.warning("[规划节点] ENABLE_HYDE 与 ENABLE_MQE 同时开启，优先使用 HyDE")
            enable_mqe = False

        if enable_hyde:
            # 初始化 HyDE 增强器（使用加长超时、少重试的客户端，避免默认 30s×多次重试导致超时与总耗时过长）
            try:
                hyde_enhancer = HyDEEnhancer(
                    llm_client=get_enhancer_llm_client(),
                    answer_length=os.getenv("HYDE_ANSWER_LENGTH", "medium")
                )
                query_enhancer = hyde_enhancer
                logger.info("[规划节点] 已启用 HyDE 查询增强器")
            except Exception as e:
                logger.warning("[规划节点] HyDE增强器初始化失败: %s，将使用基础检索", e)

        elif enable_mqe:
            # 初始化 MQE 增强器
            try:
                num_variants = int(os.getenv("MQE_NUM_VARIANTS", "3"))
                mqe_enhancer = MQEEnhancer(
                    llm_client=get_enhancer_llm_client(),
                    num_variants=num_variants
                    # 如需配置 focus_areas，可在此从环境变量读取
                )
                query_enhancer = mqe_enhancer
                logger.info("[规划节点] 已启用 MQE 查询增强器，变体数量=%s", num_variants)
            except Exception as e:
                logger.warning("[规划节点] MQE增强器初始化失败: %s，将使用基础检索", e)

        # 2. 检索相关历史记忆（传入增强器实例）
        try:
            logger.info("[规划节点] 正在检索相关历史记忆")
            related_memories = await working_memory.retrieve_relevant_memories(
                current_topic=research_topic,
                strategy="hybrid",
                limit=3,
                query_enhancer=query_enhancer  # 关键：传入增强器，可能为 None/HyDEEnhancer/MQEEnhancer
            )

            if related_memories:
                logger.info("[规划节点] 检索到相关历史记忆 count=%s", len(related_memories))
                # 构建记忆上下文提示
                memory_context = "\n## 相关历史知识（供参考）\n"
                for i, memory in enumerate(related_memories, 1):
                    memory_context += f"{i}. [{memory.memory_type.value}] {memory.content}\n"
                memory_context += "\n"
            else:
                logger.info("[规划节点] 未检索到相关历史记忆")
                memory_context = ""

            # 记录规划动作到工作记忆...
            working_memory.add_context(
                item_type="plan_started",
                content=f"开始为研究主题 '{research_topic}' 进行规划",
                metadata={
                    "research_topic": research_topic,
                    "related_memories_count": len(related_memories),
                    "query_enhancer": query_enhancer.__class__.__name__ if query_enhancer else "None"
                }
            )

        except Exception as e:
            logger.warning("[规划节点] 工作记忆处理失败: %s", e)
            memory_context = ""

    # 3. 获取上一轮的智能评估结果，用于定向规划
    deficiency_report = state.get("deficiency_report")
    targeted_instructions = state.get("targeted_instructions", [])
    last_cycle_score = state.get("last_cycle_score")
    current_cycle = state.get("current_cycle", 1)
    max_cycles = state.get("max_cycles", 5)

    # 若聚合节点已判定“无证据/不生成报告”，则停止后续规划（避免在缺乏证据基础上继续深挖导致幻觉/错配）
    if state.get("report_allowed") is False and state.get("report_block_reason"):
        logger.warning("[规划节点] 已触发早停: %s", state.get("report_block_reason"))
        state["need_deeper_research"] = False
        return state

    # 上一轮（或上一版）子任务标题清单，用于强制去重与递进深化
    previous_titles: List[str] = []
    try:
        previous_titles = [t.get("title", "").strip() for t in state.get("sub_tasks", []) if t.get("title")]
    except Exception:
        previous_titles = []

    has_targeted_instructions = bool(targeted_instructions)
    used_directed_planning = False
    directed_planning_context = ""

    if deficiency_report and has_targeted_instructions:
        logger.info("[规划节点] 基于上一轮评估结果进行强制性任务重构")
        logger.info("[规划节点] 收到定向研究指令 count=%s", len(targeted_instructions))
        used_directed_planning = True

        # 构建强制性的任务重构指令
        directed_planning_context = f"""
    ## 🔁 上一轮研究质量评估结果（你必须严格遵守的改进指令）
    上一轮研究的综合评分为：**{last_cycle_score:.1f}/100**。

    **明确的缺陷诊断**（来自智能评估引擎）：
    {deficiency_report}

    ## ⚠️ 强制性指令：你必须进行任务结构性重构
    你**绝对不能**重复上一轮的任务结构。对于下面列出的每个有缺陷的任务，你必须执行**彻底的任务重构**，而不仅仅是修改搜索查询。

    **缺陷类型与强制性重构对照表**（你必须严格遵循）：
    | 缺陷类型 | 必须执行的重构操作 | 禁止的操作示例 |
    |----------|-------------------|----------------|
    | **来源多样性不足** | 1. 任务标题必须包含以下至少一个关键词：`权威研究`、`学术机构`、`政策文件`、`跨机构比较`。<br>2. 查询必须包含域名限定：`site:.edu`、`site:.gov`、`site:.ac.uk` 或权威来源关键词：`university study`、`research report`。<br>3. 研究意图必须说明聚焦于高质量、多元化的信息来源。 | ❌ 禁止：`人工智能伦理的基本概念`<br>✅ 必须：`美国顶尖大学关于AI伦理核心原则的对比研究` |
    | **信息深度不足** | 1. 任务标题必须从概念层面转向具体技术、案例或实现细节，必须包含：`技术细节`、`实现方案`、`案例分析`、`具体框架`。<br>2. 查询必须包含具体技术术语、方法名、案例名称或`implementation`、`case study`、`technical details`。<br>3. 研究意图必须说明探究具体机制、步骤或实证。 | ❌ 禁止：`人工智能的伦理挑战`<br>✅ 必须：`自动驾驶AI伦理决策的LSTM-CRF混合框架技术分析` |
    | **存在观点冲突** | 1. 任务标题尽量包含`争议`、`辩论`、`对立观点`、`学术分歧`等关键词。<br>2. 查询必须包含`debate`、`controversy`、`pros and cons`、`criticism`。<br>3. 研究意图必须说明对比分析不同学术立场。 | ❌ 禁止：`AI的偏见问题`<br>✅ 必须：`算法公平性定义中个体平等与群体平等之争的学术辩论` |
    | **信息新颖性低** | 1. 任务标题必须强调`最新进展`、`前沿趋势`、`近两到三年更新`。<br>2. 查询必须包含近2~3年的年份（例如：{_recent_years_hint()}）以及`latest`、`recent`、`emerging`。<br>3. 研究意图必须说明探索该领域的最新动态。 | ❌ 禁止：`人工智能的历史发展`<br>✅ 必须：`近三年多模态大模型在医疗诊断中的最新突破性应用` |

    ## 📋 本轮你必须重构的具体任务
    """

        for instr in targeted_instructions[:5]:  # 最多展示5条指令
            task_id = instr.get("task_id")
            suggestion = instr.get("suggestion", "")
            instr_type = instr.get("type", "GENERAL")
            original_query = instr.get("original_query", "")

            # 获取上一轮对应的任务标题（用于对比）
            prev_title = ""
            for sub_task in state.get("sub_tasks", []):
                if sub_task.get("id") == task_id:
                    prev_title = sub_task.get("title", "")
                    break

            # 为每种缺陷类型提供强制的重构模板
            mandatory_template = ""
            if instr_type == "EXPAND_SOURCES":
                mandatory_template = f"""
    **任务{task_id}重构要求**（来源多样性不足）：
    - 上一轮标题：`{prev_title}`
    - **你必须改为**：`[具体机构类型]关于[原主题]的[研究类型]分析`
    - **必须包含的关键词**：权威、研究、报告、对比
    - **查询必须包含**：`{original_query} site:.edu OR "research report" OR "white paper"`
    - **示例标题**：`斯坦福大学与MIT关于人工智能伦理原则的对比研究报告`
    """
            elif instr_type == "DEEPEN_ANALYSIS":
                mandatory_template = f"""
    **任务{task_id}重构要求**（信息深度不足）：
    - 上一轮标题：`{prev_title}`
    - **你必须改为**：`[原主题]在[具体应用场景]的[技术细节]分析`
    - **必须包含的关键词**：技术细节、实现、案例、框架
    - **查询必须包含**：`{original_query} implementation OR "case study" OR technical details`
    - **示例标题**：`人工智能伦理在自动驾驶决策系统中的LSTM-CRF混合框架技术实现`
    """
            elif instr_type == "CLARIFY_CONFLICT":
                mandatory_template = f"""
    **任务{task_id}重构要求**（存在观点冲突）：
    - 上一轮标题：`{prev_title}`
    - **你必须改为**：`关于[原主题]中[具体争议点]的学术辩论`
    - **必须包含的关键词**：争议、辩论、对立观点
    - **查询必须包含**：`{original_query} debate OR controversy OR "pros and cons"`
    - **示例标题**：`关于算法公平性个体平等与群体平等对立的学术争议分析`
    """
            elif instr_type == "OPTIMIZE_SOURCES":
                mandatory_template = f"""
    **任务{task_id}重构要求**（来源可用性/多样性优化）：
    - 上一轮标题：`{prev_title}`
    - **你必须改为**：`[原主题]在[可检索的具体子领域/场景]的权威来源梳理与对比`
    - **必须包含的关键词**：权威、对比、报告、指南（任选其二）
    - **查询必须包含**：`{original_query} site:.edu OR site:.gov OR "report" OR "guidelines"`
    - **目的**：提升可检索性与来源多样性，避免再次“结果很少/来源为0”
    """
            elif instr_type == "BROADEN_SCOPE":
                mandatory_template = f"""
    **任务{task_id}重构要求**（资料难以检索，需要拓宽范围）：
    - 上一轮标题：`{prev_title}`
    - 上一轮查询：`{original_query}`
    - **你必须改为（通用策略）**：
      - 将主题限定从“过窄/过具体/过多复合条件”调整为“相邻可检索领域/更常用英文表述”，但仍保持与研究主题强相关
      - 如果原任务包含**复合约束**（例如绑定多个专有名词/实体、多个限定条件、强对比要求），你必须先进行**约束拆解**：
        1) 拆成若干更易检索的子任务（分别覆盖关键实体、子概念、上位概念、代表性案例或权威框架）
        2) 再在后续子任务中做整合、对照或归纳（而不是一开始就把所有约束绑在同一个查询里）
    - **查询必须**：
      - 减少过强同时限定（避免一次性绑定太多专有名词）
      - 加入更通用但强相关的学术关键词（如 framework, guidelines, standards, policy, requirements, case study）
      - 适当加入近2~3年年份（例如：{_recent_years_hint()}）以提高时效性
    - **示例查询结构**（示意）：
      - 拆解检索：`(core concept/entity) framework OR guidelines OR standards {datetime.now().year-2} {datetime.now().year-1} {datetime.now().year}`
      - 案例检索：`(core concept) case study {datetime.now().year-2} {datetime.now().year-1} {datetime.now().year}`
      - 整合/对照：`compare (concept A) (concept B) requirements framework {datetime.now().year-2} {datetime.now().year-1} {datetime.now().year}`（仅在确有对比需求时使用）
    """

            directed_planning_context += mandatory_template + "\n"

        directed_planning_context += """
    ## ✅ 最终输出验证清单（你必须逐一检查）
    在生成最终JSON输出前，请确认你的规划满足以下所有条件：

    1. **标题重构检查**：
       - 对于上面列出的每个任务，新标题**绝对不能**与上一轮标题相同
       - 新标题必须包含对应缺陷类型所要求的关键词
       - 新标题必须体现研究焦点从宽泛到具体、从概念到细节的转移

    2. **查询修改检查**：
       - 每个任务的查询必须包含所要求的限定词、域名过滤或关键词
       - 查询必须与上一轮有显著差异，确保搜索结果不同

    3. **整体结构检查**：
       - 所有任务共同体现从基础到深入的递进逻辑
       - 本轮的研究层次必须比上一轮更深入、更具体
       - 至少有一个任务专门探究该主题的最新进展（近2~3年，例如：{_recent_years_hint()}）

    **⚠️ 重要提醒**：如果你重复上一轮的任务结构，系统将无法解决已识别的缺陷，研究质量不会提升。你必须进行彻底的任务重构。
    """

    elif deficiency_report and not has_targeted_instructions and state.get("current_cycle", 1) > 1:
        # 有缺陷报告但无具体指令（例如第一轮后）
        logger.info("[规划节点] 参考上一轮评估报告进行规划 score=%.1f", last_cycle_score)
        directed_planning_context = f"""
    ## 📈 上一轮研究概况
    上一轮研究的综合评分为：**{last_cycle_score:.1f}/100**。

    **评估摘要**：
    {deficiency_report[:500]}...

    ## 📋 本轮规划要求
    基于以上评估，请确保：
    1. 避免重复上一轮已充分研究的内容
    2. 针对报告中提到的不足，设计更深入、更具体的子任务
    3. 任务标题应体现研究焦点的深化
    4. 至少有一个任务专门探究该主题的最新进展（近2~3年，例如：{_recent_years_hint()}）
    """

    # 构建整合了记忆上下文和定向指令的增强版系统提示词
    enhanced_system_content = f"""你是一个资深的研究规划专家。你的任务是将一个宏观的研究主题，拆解成一系列具体、可独立搜索查询的子任务。

    {memory_context if memory_context else ""}
    {directed_planning_context if directed_planning_context else ""}

    # 🔁 迭代与质量提升策略（必须执行）
    - 当前轮次：第{current_cycle}轮 / 最大{max_cycles}轮
    - 上一轮平均分：{f"{last_cycle_score:.1f}/100" if last_cycle_score is not None else "未知"}
    - **本轮目标**：在可检索性与信息深度之间取得平衡，使“本轮平均质量评分”尽量高于上一轮（如果上一轮分数未知，也要按更高标准设计任务）。
    - **下一轮深挖要求**：你在本轮生成的任务必须为“下一轮继续深挖”留下空间（例如：本轮先建立权威框架与关键争议点，下一轮深入机制/案例/实施细节/评估指标）。

    # 🚫 标题强制去重（严禁重复）
    以下是上一轮（或上一版）已使用过的子任务标题（仅供你去重对照）：
    {json.dumps(previous_titles, ensure_ascii=False)}
    - 你生成的每个新 `title` 必须与上述标题**完全不同**，且不得只是同义改写（要换角度、换限定范围或换研究对象）。
    - 你生成的 3-5 个新 `title` 之间也必须彼此不同。

    # 🧭 “难检索导致低分”时的拓宽规则（允许但要有边界）
    - 如果上一轮出现“未获得搜索结果/来源为0/独立域名为0/资料难以检索”等迹象（可能会在评估报告或定向指令中体现），下一轮你可以**适度拓宽限定范围**：
      - 从过窄的实体/术语 → 相邻概念、上位概念、或更常用英文表述
      - 从单一国家/机构 → 跨国家/跨机构比较（仍需与主题强相关）
      - 从极具体实现 → “框架/标准/治理/政策/案例”层面以获取可检索资料
    - 但禁止跑题：拓宽必须仍围绕研究主题的核心问题。

    # 📄 输出格式要求
    你必须返回一个**纯粹的、合法的JSON数组**，不能包含任何其他解释性文字。
    数组应包含3到5个对象，每个对象代表一个子任务，且必须包含以下三个字段：
    1. `title`: 子任务的中文标题，清晰说明研究方面。**必须体现研究焦点的深化**。
    2. `intent`: 用一句话中文说明研究此任务的目的。**必须说明本任务与前几轮的区别**。
    3. `query`: 用于搜索引擎的查询字符串。应使用**英文关键词**，并包含近2~3年的年份（例如：{_recent_years_hint()}）以提高搜索结果时效性和相关性。

    # 🧱 首轮与后续轮次的拆分策略（关键）
    - {"**第1轮（首轮）**：优先“广覆盖”而非“细钻研”。请围绕主题先建立全景框架，任务应覆盖：核心概念/主流方向/典型应用场景/关键挑战（可含治理或评估维度）。避免一开始就绑定过细技术路线、过窄地区对比或强限定实体组合。" if current_cycle == 1 else "**第2轮及以后**：在首轮覆盖基础上做“定向深挖”，优先补足上一轮缺陷（证据不足、深度不足、冲突未厘清、时效性不足）。"}
    - {"首轮每个任务的标题尽量保持“主题级-子域级”，减少“具体论文级/具体项目级”限定，以保证后续轮次仍有深入空间。" if current_cycle == 1 else "后续轮次可引入更具体的技术机制、对照维度与实施细节，但必须明确回应上一轮评估缺陷。"}

    # 🎯 子任务设计原则
    - **深度递进性**: 任务间应有明确的逻辑递进关系，体现从基础到深入、从概念到具体、从现状到前沿的层次深化。{"**首轮重点是覆盖面与结构化框架。**" if current_cycle == 1 else "**本轮研究层次必须比上一轮更深入。**"}
    - **问题针对性**: 每个任务应针对上一轮识别出的具体缺陷进行设计，直接回应质量评估报告中的改进要求。
    - **可执行性**: `query`字段必须能直接用于搜索引擎，使用英文关键词，并确保能解决上一轮识别的问题。

    # ⚠️ 特别注意
    {"**强制性要求**：你必须严格遵守上述所有重构指令。系统将对你的输出进行验证，如果检测到任务标题重复或未体现重构要求，本轮研究将被视为失败。你的核心任务是解决已识别的缺陷，推动研究向更深入的层面发展。" if used_directed_planning else ""}
    {"请参考上述历史知识，避免重复，并着重探索新的或需要更新的方面。" if memory_context and not used_directed_planning else ""}
    {f"请确保至少有一个子任务专门探究该主题的最新进展（近2~3年，例如：{_recent_years_hint()}）。" if not used_directed_planning else ""}

    # 📋 输出验证（请在生成JSON前自行检查）
    1. 检查每个任务的`title`是否体现了研究焦点的深化
    2. 检查`query`是否包含必要的限定词以解决上一轮问题
    3. 检查整体任务列表是否呈现清晰的递进逻辑
    4. 确保没有简单重复上一轮的任务结构

    # 📖 示例（正确的递进式重构）
    研究主题: "多模态大模型的最新进展"
    **第一轮标题示例**：`多模态大模型的基本概念`
    **第二轮重构后标题**：`近三年顶尖实验室多模态大模型架构的技术细节分析`
    **查询**：`multimodal large language model architecture technical details {_recent_years_hint()} MIT Stanford`
    """

    # 将“证据真实性与正确性”规则注入系统提示，减少后续 query 生成跑题/错配
    enhanced_system_content = (enhanced_system_content or "") + "\n" + evidence_integrity_rules
    system_prompt = SystemMessage(content=enhanced_system_content)
    user_prompt = HumanMessage(
        content=f"请针对以下研究主题，生成研究子任务列表：\n\n研究主题：{research_topic}"
    )

    # 调试信息（避免泄露系统提示词内容）
    logger.info("[规划节点] 系统提示词构造完成 prompt_len=%s", len(enhanced_system_content))

    # 记录定向规划的具体信息
    if used_directed_planning and _planning_debug_enabled():
        logger.info("[规划节点] 定向规划已启用 instructions=%s", len(targeted_instructions))
        # 仅展示前2条指令摘要，避免日志刷屏
        for instr in targeted_instructions[:2]:
            logger.info(
                "[规划节点] 指令预览 task_id=%s type=%s suggestion=%s",
                instr.get("task_id"),
                str(instr.get("type", "")),
                str(instr.get("suggestion", ""))[:80],
            )

    # 调用LLM
    llm_output = ""
    try:
        response = await llm_client.agenerate(messages=[[system_prompt, user_prompt]])
        llm_output = response.generations[0][0].text
        logger.info("[规划节点] LLM输出接收成功 output_len=%s", len(llm_output))
    except Exception as e:
        logger.warning("[规划节点] 调用Qwen API失败: %s", e)
        llm_output = json.dumps(_get_fallback_tasks(research_topic))
        logger.info("[规划节点] 已启用降级规划方案")

    # 解析LLM输出
    tasks = _parse_llm_output_to_tasks(llm_output)
    if not tasks:
        logger.warning("[规划节点] 无法解析有效任务列表，使用降级任务")
        # 使用降级任务
        tasks = _get_fallback_tasks(research_topic)

    # 格式化任务并更新状态
    formatted_sub_tasks = []
    for idx, task in enumerate(tasks[:5], start=1):  # 限制最多5个子任务
        formatted_sub_tasks.append({
            "id": idx,
            "title": task.get("title", "").strip(),
            "intent": task.get("intent", "").strip(),
            "query": task.get("query", "").strip(),
            "status": TaskStatus.PENDING,
            "assigned_agent": None
        })

    # 将规划结果记录到工作记忆
    if working_memory:
        try:
            task_summary = f"为研究主题 '{research_topic}' 规划了 {len(formatted_sub_tasks)} 个子任务:\n"
            for task in formatted_sub_tasks:
                task_summary += f"- {task['title']}\n"

            # 记录规划元数据
            planning_method = "定向规划" if used_directed_planning else "标准规划"
            directed_instruction_count = len(targeted_instructions) if has_targeted_instructions else 0

            working_memory.add_context(
                item_type="plan_result",
                content=task_summary,
                metadata={
                    "research_topic": research_topic,
                    "task_count": len(formatted_sub_tasks),
                    "research_cycle": state.get("current_cycle", 1),
                    "planning_method": planning_method,
                    "directed_instruction_count": directed_instruction_count,
                    "used_deficiency_report": deficiency_report is not None,
                }
            )
            logger.info("[规划节点] 规划结果已记录到工作记忆 method=%s", planning_method)
        except Exception as e:
            logger.warning("[规划节点] 记录规划结果到工作记忆失败: %s", e)

    # 更新GraphState
    state["sub_tasks"] = formatted_sub_tasks
    state["task_results"] = [None] * len(formatted_sub_tasks)
    # 注意：current_cycle 不应在这里重置为1，应该保持其原值
    # state["current_cycle"] = 1  # 移除这行，由聚合节点控制循环
    state["max_cycles"] = state.get("max_cycles", 4)
    state["need_deeper_research"] = False
    state["active_tasks"] = []

    logger.info("[规划节点] 规划完成，子任务数量=%s", len(formatted_sub_tasks))
    # 默认仅输出标题概览，避免输出大量查询词造成日志噪音
    for task in formatted_sub_tasks:
        logger.info("[规划节点] 子任务 id=%s title=%s", task["id"], task["title"])
        if _planning_debug_enabled():
            logger.info("[规划节点] 子任务查询 query=%s", task["query"])

    # 在规划完成后，添加查询对比验证（与上一轮快照对比）
    if used_directed_planning and state.get("current_cycle", 1) > 1 and prev_sub_tasks_snapshot:
        logger.info("[规划节点] 查询变化验证")
        for task in formatted_sub_tasks:
            task_id = task["id"]
            new_query = task["query"]

            # 查找上一轮对应的查询（如果有的话）
            prev_query = ""
            for prev_task in prev_sub_tasks_snapshot:
                if prev_task.get("id") == task_id:
                    prev_query = prev_task.get("query", "")
                    break

            if prev_query and new_query != prev_query:
                logger.info("[规划节点] 任务%s查询已更新 old=%s new=%s", task_id, prev_query[:80], new_query[:80])
            elif prev_query:
                logger.warning("[规划节点] 任务%s查询未变化，需要更强定向指导", task_id)
    return state


def _parse_llm_output_to_tasks(llm_output: str) -> List[dict]:
    """
    从LLM的响应文本中解析出任务列表。
    """
    if not llm_output:
        return []

    text = llm_output.strip()
    parsed_data = None

    # 策略1: 尝试直接解析
    try:
        parsed_data = json.loads(text)
        if isinstance(parsed_data, list):
            return parsed_data
    except json.JSONDecodeError:
        pass

    # 策略2: 尝试提取被 ```json ... ``` 包裹的JSON
    import re
    json_code_block_pattern = r'```(?:json)?\s*\n([\s\S]*?)\n\s*```'
    match = re.search(json_code_block_pattern, text, re.IGNORECASE)
    if match:
        try:
            parsed_data = json.loads(match.group(1).strip())
            if isinstance(parsed_data, list):
                return parsed_data
        except json.JSONDecodeError:
            pass

    # 策略3: 使用正则表达式查找最外层的 JSON 数组
    array_pattern = r'\[\s*\{.*\}\s*\]'
    match = re.search(array_pattern, text, re.DOTALL)
    if match:
        try:
            parsed_data = json.loads(match.group().strip())
            if isinstance(parsed_data, list):
                return parsed_data
        except json.JSONDecodeError:
            pass

    return []


def _get_fallback_tasks(research_topic: str) -> List[dict]:
    """
    降级函数：当LLM服务不可用时，生成基础任务列表。
    """
    return [
        {
            "title": f"{research_topic}的基本概念与定义",
            "intent": f"了解{research_topic}的基础定义、核心概念与基本原理",
            "query": f"{research_topic} basic concepts definition {_recent_years_hint()}"
        },
        {
            "title": f"{research_topic}的最新发展与应用",
            "intent": f"了解{research_topic}的最新技术进展、实际应用与典型案例",
            "query": f"{research_topic} latest developments applications {_recent_years_hint()}"
        },
        {
            "title": f"{research_topic}面临的挑战与未来趋势",
            "intent": f"了解{research_topic}当前面临的主要问题、争议以及未来发展方向",
            "query": f"{research_topic} challenges future trends {_recent_years_hint()}"
        }
    ]