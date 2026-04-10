# src/memory/query_enhancer.py
"""
查询增强器模块。
职责：提供高级检索策略，如 HyDE (假设文档嵌入)，用于在检索前优化查询表达。
设计：遵循策略模式，便于未来扩展（如 MQE, Query2Doc 等）。
"""
import asyncio
import logging
from typing import Optional, Dict, Any, Union, List
from abc import ABC, abstractmethod
import re
logger = logging.getLogger(__name__)


class BaseQueryEnhancer(ABC):
    """
    查询增强器抽象基类。
    所有具体的增强策略（如 HyDE, MQE）都应继承并实现此接口。
    """

    @abstractmethod
    async def enhance(
        self,
        original_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Union[str, List[str]]:
        """
        增强原始查询的核心抽象方法。

        Args:
            original_query: 用户原始的查询文本。
            context: 可选的上下文信息字典，可用于更精准的增强。
                    例如，可能包含研究主题、任务意图等。

        Returns:
            增强后的查询文本。对于某些增强器（如 MQE），可能返回一个查询列表。
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class HyDEEnhancer(BaseQueryEnhancer):
    """
    HyDE (Hypothetical Document Embeddings) 查询增强器。

    原理：不直接使用原始查询进行向量检索，而是先利用LLM
          根据该查询“生成”一个假设性的理想答案文档，然后用这个生成的、
          更丰富、更贴近知识库文档风格的文本来进行检索。

    优势：对于开放式、研究性的查询，生成的假设答案在语义空间上
          可能比简短的原始查询更接近知识库中的高质量文档，从而提高召回相关性。

    注意：请传入 `get_enhancer_llm_client()`（见 llm_client），勿使用默认全局 `llm_client`；
    后者默认 30s 超时且会重试，长文本生成易超时并回退为原 query。
    """

    def __init__(self, llm_client, answer_length: str = "medium"):
        """
        初始化 HyDE 增强器。

        Args:
            llm_client: 用于生成假设答案的 LLM 客户端，需支持异步生成。
            answer_length: 期望生成的假设答案长度，可选 'short', 'medium', 'detailed'。
        """
        self.llm_client = llm_client
        self.answer_length = answer_length
        self._length_map = {
            "short": "两到三句话",
            "medium": "一段话（约100-200字）",
            "detailed": "一个较完整的段落（约200-300字）"
        }
        logger.info(f"HyDEEnhancer 初始化完成，答案长度模式: {answer_length}")
        # 最近一次 enhance 是否未得到真实假设答案（回退为原始 query）；供调用方区分“成功 HyDE”与“静默回退”
        self.last_used_fallback: bool = False
        self.last_fallback_reason: Optional[str] = None

    async def enhance(
        self,
        original_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        使用 HyDE 策略增强查询：生成一个假设性答案。

        Args:
            original_query: 原始查询（例如研究主题或子任务意图）。
            context: 可选上下文，当前版本暂未使用，为未来扩展预留。

        Returns:
            由LLM生成的假设性答案文本，将用于后续的向量检索。
        """
        self.last_used_fallback = False
        self.last_fallback_reason = None

        if not original_query or not original_query.strip():
            logger.warning("收到空查询，无法进行 HyDE 增强。")
            self.last_used_fallback = True
            self.last_fallback_reason = "empty_query"
            return original_query

        logger.info(f"开始 HyDE 查询增强，原始查询: '{original_query[:50]}...'")

        # 构建生成假设答案的提示词
        length_hint = self._length_map.get(self.answer_length, "一段话")
        prompt = self._build_hyde_prompt(original_query, length_hint)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            system_prompt = SystemMessage(
                content="你是一个知识渊博的研究助手。你的任务是针对一个提问或研究主题，生成一份假设性的、全面的答案草案。"
            )
            human_prompt = HumanMessage(content=prompt)

            # 异步调用LLM生成假设答案
            response = await self.llm_client.agenerate(messages=[[system_prompt, human_prompt]])
            hypothetical_answer = response.generations[0][0].text.strip()

            if not hypothetical_answer:
                logger.warning("LLM 返回了空的假设答案，将回退到原始查询。")
                self.last_used_fallback = True
                self.last_fallback_reason = "empty_llm_response"
                return original_query

            logger.info(f"HyDE 增强完成。生成假设答案长度: {len(hypothetical_answer)} 字符。")
            logger.debug(f"假设答案预览: {hypothetical_answer[:150]}...")
            return hypothetical_answer

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"HyDE 增强过程失败: {e}，将回退到原始查询。")
            print(f"[HyDE] 增强失败（已回退为原始 query）: {type(e).__name__}: {e}")
            self.last_used_fallback = True
            self.last_fallback_reason = f"exception:{type(e).__name__}"
            # 关键：增强失败时，安全地返回原始查询，不影响主流程
            return original_query

    def _build_hyde_prompt(self, query: str, length_hint: str) -> str:
        """构建用于生成假设答案的提示词。"""
        return f"""请针对以下研究问题或主题，生成一份{length_hint}的、全面的假设性答案。

**研究问题/主题**：{query}

**请生成假设答案时注意**：
1.  **全面性**：覆盖该主题的关键方面、核心概念、重要事实和潜在推论。
2.  **学术/专业语气**：使用严谨、客观的书面语，类似于一份研究报告或百科条目中的片段。
3.  **基于通用知识**：这是一份“假设”答案，请基于公开的、通用的知识和逻辑进行生成，无需担心绝对正确性，但应合理可信。
4.  **结构化**：答案应内在逻辑清晰，可以自然地包含背景、要点、总结等元素。

请直接输出这份假设性答案，不要包含“答案：”、“假设答案：”等引导词。"""



class MQEEnhancer(BaseQueryEnhancer):
    """
    多查询扩展（Multiple Query Expansion）增强器。

    在**同一信息需求**下生成少量子查询（同义改写、术语别名、短子问句等），
    用于向量检索召回；默认避免「综述式多角度」扩展导致检索跑偏。
    """

    # 受控扩展：每条对应一个变体 slot，按 num_variants 截取使用前几条
    _DEFAULT_EXPANSION_STRATEGIES: List[str] = [
        "同义改写：换句式或近义表达，保留原问中的实体、数字、否定与范围约束，不新增话题。",
        "术语层面：对原问中的专业词给出常见别名、缩写展开或英文说法（问题目标与原问一致）。",
        "更短等价问法：把原问改写为更短的 1 句问法，但必须保留所有约束条件（不要把它变成只问其中一个子点）。",
        "紧凑关键词：用更短的关键词串重写，便于向量匹配（语义须与原问等价）。",
        "显式约束：若原问含多个条件，用另一种说法把同一组条件写全，勿丢掉任一条件。",
    ]

    def __init__(self, llm_client, num_variants: int = 4, focus_areas: Optional[List[str]] = None):
        """
        初始化 MQE 增强器。

        Args:
            llm_client: 用于生成查询变体的 LLM 客户端。
            num_variants: 期望生成的查询变体数量。
            focus_areas: 可选，每条为一条「扩展策略说明」，长度宜与 num_variants 一致；
                         若为 None，则使用内置的受控扩展策略（非综述式大角度）。
        """
        self.llm_client = llm_client
        # 偏召回：允许略多变体，但仍限制上界以控制成本与噪声
        self.num_variants = max(1, min(num_variants, 6))
        self.focus_areas = focus_areas if focus_areas is not None else list(self._DEFAULT_EXPANSION_STRATEGIES)
        logger.info(f"MQEEnhancer 初始化完成，变体数量: {num_variants}")
        # 最近一次 enhance 是否回退为原始 query（供评估层诊断）
        self.last_used_fallback: bool = False
        self.last_fallback_reason: Optional[str] = None

    async def enhance(
        self,
        original_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        使用 MQE 策略增强查询：生成多个相关的查询变体。

        Args:
            original_query: 原始查询（例如研究主题或子任务意图）。
            context: 可选上下文，可包含如 `research_topic` 等信息，用于更精准的扩展。

        Returns:
            一个包含多个查询变体的字符串列表。如果增强失败，则返回包含原始查询的列表。
        """
        self.last_used_fallback = False
        self.last_fallback_reason = None

        if not original_query or not original_query.strip():
            logger.warning("收到空查询，MQE 增强将返回空列表。")
            self.last_used_fallback = True
            self.last_fallback_reason = "empty_query"
            return []

        logger.info(f"开始 MQE 查询扩展，原始查询: '{original_query[:50]}...'，目标变体数: {self.num_variants}")

        # 构建生成多查询变体的提示词
        prompt = self._build_mqe_prompt(original_query, context)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            system_prompt = SystemMessage(
                content=(
                    "你是面向向量检索的查询改写助手。必须在不改变用户核心意图的前提下，"
                    "生成少量用于召回的子查询；禁止引入原问未涉及的新主题、新领域或泛泛的伦理/趋势讨论。"
                )
            )
            human_prompt = HumanMessage(content=prompt)

            response = await self.llm_client.agenerate(messages=[[system_prompt, human_prompt]])
            response_text = response.generations[0][0].text.strip()

            # 解析LLM返回的文本，提取查询列表
            expanded_queries = self._parse_response_to_queries(response_text, original_query)

            if not expanded_queries:
                logger.warning("未能从LLM响应中解析出有效的查询变体，将回退到原始查询。")
                print("[MQE] 解析未得到有效变体（LLM 输出格式可能不符），已回退为原始 query。")
                self.last_used_fallback = True
                self.last_fallback_reason = "empty_parsed_variants"
                return [original_query]

            logger.info(f"MQE 扩展完成。生成 {len(expanded_queries)} 个查询变体。")
            for i, q in enumerate(expanded_queries, 1):
                logger.debug(f"  变体 {i}: {q[:60]}...")
            return expanded_queries

        except Exception as e:
            logger.error(f"MQE 扩展过程失败: {e}，将回退到原始查询。")
            print(f"[MQE] 增强失败（已回退为原始 query）: {type(e).__name__}: {e}")
            # 关键：增强失败时，安全地返回仅包含原始查询的列表
            self.last_used_fallback = True
            self.last_fallback_reason = f"exception:{type(e).__name__}"
            return [original_query]

    def _context_scope_note(self, original_query: str, context: Optional[Dict]) -> str:
        """
        可选背景句：仅当 context 中的 research_topic 像「自然语言领域」而非内部 slug 时使用。
        避免把 rag_eval_ai 等 Chroma topic 误当成要扩展的用户问题（否则会整体带偏）。
        """
        if not context:
            return ""
        rt = context.get("research_topic")
        if rt is None or str(rt).strip() == "":
            return ""
        rts = str(rt).strip()
        oq = (original_query or "").strip()
        if rts == oq:
            return ""
        # 典型内部 topic id：仅字母数字下划线连字符
        if re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", rts):
            return ""
        return f"\n**背景提示**（仅供理解领域，子查询仍必须直接回答上面的用户问题，不得改成讨论本标签）：{rts}\n"

    def _build_mqe_prompt(self, original_query: str, context: Optional[Dict]) -> str:
        """构建用于生成多查询变体的提示词（受控扩展、防跑偏）。"""
        scope = self._context_scope_note(original_query, context)
        strategies = self.focus_areas[: self.num_variants]
        strategy_lines = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(strategies))

        # 语料多为英文技术文档时，对中文问题生成英文关键词/短问句能显著提升召回覆盖
        # 规则：保持信息需求等价，不引入新主题；仅做“跨语言/同义/关键词化”的表达变化
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", original_query or ""))
        needs_english = has_cjk and not re.search(r"[A-Za-z]{3,}", original_query or "")
        bilingual_rule = (
            "\n**语料语言提示**：知识库/技术文档可能以英文为主。请确保输出中至少包含 2 条英文（或中英混合）查询，使用常见英文技术术语/缩写，但仍需与用户问题等价。\n"
            if needs_english
            else ""
        )

        return f"""请基于下面的**用户问题**，生成恰好 {self.num_variants} 条**短检索查询**（子查询），用于向量库召回。
所有子查询必须与用户问题指向**同一信息需求**：可以换说法、拆术语、略写或补全省略成分，但**不得**改成另一个问题、不得引入用户未提及的新主题（例如不要随意扩展到伦理、行业全景、无关年份热点）。

**用户问题**：
{original_query.strip()}
{scope}
{bilingual_rule}
**每条子查询对应的扩展方式**（第 i 条尽量贴近第 i 条策略，仍须与用户问题等价）：
{strategy_lines}

**硬性约束**：
- 不要随意加入「2024」「2025」等年份，除非用户问题里已经提到时间。
- 不要写长段落；每条子查询建议不超过 40 字（必要术语可略长）。
- 不要输出解释、不要输出序号以外的多余行；不要在行尾追加括号解释。
- 每条必须是“可直接用于检索”的短 query：尽量包含关键实体/术语/缩写。

**输出格式**（严格遵守）：
查询1：[子查询1]
查询2：[子查询2]
...
查询N：[子查询N]

**示例**（用户问题已给定，仅演示格式与「等价改写」尺度；请按真实用户问题生成内容）：
用户问题：RAG 评测里 nDCG 与 MRR 有什么区别？
查询1：RAG 评估中 nDCG 和 MRR 差异
查询2：信息检索评测指标 nDCG MRR 对比
查询3：排序评价 nDCG 对比 MRR 适用场景
"""

    def _parse_response_to_queries(self, response_text: str, original_query: str) -> List[str]:
        """从LLM的响应文本中解析出查询变体列表。"""
        queries: List[str] = []
        lines = [line.strip() for line in (response_text or "").split("\n") if line.strip()]

        # 偏召回：但解析必须严格，避免把解释性文本吞进来造成噪声召回
        # 仅接受形如：查询1: xxx / 查询1：xxx / Query 1: xxx
        pat = re.compile(r"^(?:查询|query)\s*(\d+)\s*[:：]\s*(.+)$", re.IGNORECASE)
        for line in lines:
            m = pat.match(line)
            if not m:
                continue
            content = (m.group(2) or "").strip()
            if content:
                queries.append(content)

        # 后处理：去重，并确保数量不超过设定值
        unique_queries = []
        seen = set()
        for q in queries:
            if q not in seen and len(q) > 5:  # 过滤过短的可能无效查询
                seen.add(q)
                unique_queries.append(q)
            if len(unique_queries) >= self.num_variants:
                break

        # 如果解析失败或数量不足，用原始查询补充
        if not unique_queries:
            unique_queries.append(original_query)
        elif len(unique_queries) < self.num_variants:
            unique_queries.append(original_query)  # 确保至少包含原始意图

        return unique_queries
