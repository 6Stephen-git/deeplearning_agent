# src/evaluator/task_diagnoser.py
"""
任务诊断器模块。
包含 TaskDiagnoser 和 ResearchCycleDiagnoser 类，负责对研究结果进行细粒度质量评估。
"""
import asyncio
import logging
import re
import json
from typing import List, Dict, Optional, Tuple, Any, Set
from urllib.parse import urlparse
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.evaluator.schemas import TaskQualityProfile, MetricScore, CycleDeficiencyReport

logger = logging.getLogger(__name__)


class TaskDiagnoser:
    """
    任务诊断器。
    职责：对单个任务结果进行多维度深度评估，生成质量画像和优化建议。
    """

    def __init__(self, llm_client, memory_store=None, config: Optional[Dict[str, Any]] = None):
        """
        初始化诊断器。

        Args:
            llm_client: 用于高级语义分析的LLM客户端。
            memory_store: 用于评估信息新颖性的记忆存储（可选）。
            config: 评估配置字典，包含各指标的权重、阈值等。
        """
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.config = config or {}

        # 默认指标权重配置 (可从config覆盖)
        self.metric_weights = self.config.get('metric_weights', {
            'validity': 0.15,      # 结果有效性
            'saturation': 0.20,    # 信息饱和度
            'diversity': 0.25,     # 来源多样性 (提升权重)
            'conflict': 0.15,      # 观点冲突性
            'fact_density': 0.15,  # 事实密度 (新增)
            'novelty': 0.10,       # 信息新颖性 (新增)
        })

        # 权威域名配置
        self.authoritative_domains = self.config.get('authoritative_domains', [
            '.edu', '.gov', '.ac.',
            'nature.com', 'science.org', 'arxiv.org', 'springer.com', 'ieee.org'
        ])

        # 决策阈值
        # 存储到长期记忆的综合评分阈值：默认 65 分
        self.storage_threshold = self.config.get('storage_threshold', 65.0)
        self.conflict_detection_enabled = self.config.get('conflict_detection_enabled', True)

        logger.info(f"TaskDiagnoser 初始化完成，指标权重: {self.metric_weights}")

    async def diagnose_task(self, task_result: Dict[str, Any],
                            task_context: Dict[str, Any]) -> TaskQualityProfile:
        """
        诊断单个任务，生成其质量画像。

        Args:
            task_result: 来自 `task_results` 的单个任务结果字典。
            task_context: 任务上下文，包含 task_id, research_cycle, research_topic 等。

        Returns:
            任务质量画像对象。
        """
        task_id = task_context.get('task_id', 0)
        research_cycle = task_context.get('research_cycle', 1)

        logger.info(f"[诊断器] 开始诊断任务 {task_id} (第{research_cycle}轮)")

        # 1. 创建基础画像对象
        profile = TaskQualityProfile(
            task_id=task_id,
            research_cycle=research_cycle
        )

        try:
            # 并行计算基础指标
            basic_metrics_task = asyncio.create_task(
                self._compute_basic_metrics(profile, task_result, task_context)
            )

            # 只有启用冲突检测时才执行语义分析
            semantic_analysis_task = None
            if self.conflict_detection_enabled and self.llm_client is not None:
                # 只有明确启用且有客户端时，才创建语义分析任务
                semantic_analysis_task = asyncio.create_task(
                    self._perform_semantic_analysis(profile, task_result, task_context)
                )
            else:
                # 否则，预先设置一个默认的冲突指标
                profile.metrics['conflict'] = MetricScore(
                    name='观点冲突性',
                    score=0.0,
                    weight=self.metric_weights.get('conflict', 0.0),
                    evidence="冲突检测未启用"
                )

            # 评估信息新颖性
            novelty_assessment_task = asyncio.create_task(
                self._assess_novelty(profile, task_result, task_context)
            ) if self.memory_store else None

            # 等待所有并行任务完成
            await basic_metrics_task
            if semantic_analysis_task:  # 只有存在时才等待
                await semantic_analysis_task
            if novelty_assessment_task:
                await novelty_assessment_task

            # 5. 生成标签与决策建议
            self._generate_tags_and_suggestions(profile)

            # 6. 计算最终综合评分
            profile.calculate_composite()

            logger.info(f"[诊断器] 任务 {task_id} 诊断完成，综合评分: {profile.composite_score:.1f}")
            logger.debug(f"[诊断器] 任务 {task_id} 质量画像: {profile.to_dict()}")

        except Exception as e:
            logger.error(f"[诊断器] 诊断任务 {task_id} 时发生异常: {e}", exc_info=True)
            # 在异常情况下，至少设置一个基础的有效性指标
            profile.metrics['validity'] = MetricScore(
                name='结果有效性',
                score=0.0,
                weight=self.metric_weights.get('validity', 0.15),
                evidence=f"诊断过程异常: {str(e)[:100]}"
            )

        return profile

    async def _compute_basic_metrics(self, profile: TaskQualityProfile,
                                     task_result: Dict[str, Any],
                                     context: Dict[str, Any]) -> None:
        """计算基础、可量化的指标"""
        summary = task_result.get("summary", "")
        sources = task_result.get("source_urls") or task_result.get("sources") or []
        search_results = task_result.get("search_results") or []
        research_topic = (context.get("research_topic") or "").strip()

        # === 0. 关键词抽取（从 research_topic 中抽取所有关键词，含专业术语） ===
        # 设计目标：宁可保守判无证据，也不允许张冠李戴或基于无关来源生成“高分总结”。
        required_keywords = self._extract_required_keywords(research_topic)
        normalized_keywords = [k.lower() for k in required_keywords if k and k.strip()]

        # 构建证据文本（title/snippet/url + sources）
        evidence_text = self._build_evidence_text(search_results=search_results, sources=list(sources))

        # 关键词命中率：用于相关性/证据门控
        kw_total = max(1, len(normalized_keywords))
        kw_hits = sum(1 for kw in normalized_keywords if kw and kw in evidence_text)
        kw_hit_ratio = kw_hits / kw_total if kw_total else 0.0

        # 证据对齐门控（MVP）：至少 20% 关键词在证据中出现，且总结中也命中至少 20%
        summary_lc = (summary or "").lower()
        sum_hits = sum(1 for kw in normalized_keywords if kw and kw in summary_lc)
        sum_hit_ratio = sum_hits / kw_total if kw_total else 0.0

        evidence_alignment_pass = (kw_hit_ratio >= 0.20) and (sum_hit_ratio >= 0.20)
        profile.metrics["evidence_alignment"] = MetricScore(
            name="证据对齐",
            score=100.0 if evidence_alignment_pass else 0.0,
            weight=0.0,  # 门控指标：不参与综合分，仅用于 validity/reliability 的硬约束
            evidence=f"keywords_hit={kw_hits}/{kw_total}, summary_hit={sum_hits}/{kw_total}"
        )

        # 1. 结果有效性（强化“可追溯证据”要求）
        has_summary = bool(summary and len(summary.strip()) > 10)
        has_sources = len(sources) > 0
        has_citations = bool(re.search(r'\[\d+\]', summary or ""))
        # 只有通过证据对齐门控，才认为“有效”。否则即使有总结/有来源也按无效处理（防张冠李戴）。
        is_valid = has_summary and evidence_alignment_pass
        profile.metrics['validity'] = MetricScore(
            name='结果有效性',
            score=(
                100.0 if (has_summary and has_sources and has_citations)
                else 70.0 if (has_summary and has_sources)
                else 40.0 if has_summary
                else 0.0
            ),
            weight=self.metric_weights.get('validity', 0.15),
            evidence=(
                "总结有效且含来源与引用标号" if (has_summary and has_sources and has_citations)
                else "总结有效且含来源" if (has_summary and has_sources)
                else "总结有效但缺少可追溯来源/引用" if has_summary
                else "总结文本为空或过短"
            )
        )

        # 若证据对齐失败，强制将 validity 拉低到 0（防止 sources/格式刷分）
        if not evidence_alignment_pass:
            profile.metrics["validity"].score = 0.0
            profile.metrics["validity"].evidence = "证据对齐失败：疑似无证据或来源与主题关键词不匹配"

        if not is_valid:
            # 如果总结无效，跳过其他指标计算
            return

        # 2. 信息饱和度 (增强版)
        length_score = min(len(summary) / 5, 100)  # 原逻辑：每5字符1分，上限100

        # 简单句子分割估算信息点
        sentences = [s.strip() for s in re.split(r'[。！？.!?]', summary) if len(s.strip()) > 5]
        # 估算子句/信息单元 (通过分号、逗号等分割)
        clauses = [c.strip() for c in re.split(r'[；，,;]', summary) if len(c.strip()) > 3]

        info_density = min(len(sentences) * 8 + len(clauses) * 2, 60)  # 结合句子和子句
        saturation_score = 0.6 * length_score + 0.4 * info_density

        # 可靠性：若证据对齐通过则高，否则低（这里 is_valid 已要求对齐通过）
        reliability = 100.0 if evidence_alignment_pass else 0.0
        saturation_final = 0.4 * saturation_score + 0.6 * reliability

        profile.metrics['saturation'] = MetricScore(
            name='信息饱和度',
            score=saturation_final,
            weight=self.metric_weights.get('saturation', 0.20),
            evidence=f"长度:{len(summary)}字, 句子:{len(sentences)}, 单元:{len(clauses)}, reliability:{reliability:.0f}"
        )

        # 3. 来源多样性 (增强版)
        unique_domains = self._extract_unique_domains(sources)
        domain_count = len(unique_domains)

        # 计算权威域名数量
        authoritative_count = 0
        for domain in unique_domains:
            if any(auth_domain in domain for auth_domain in self.authoritative_domains):
                authoritative_count += 1

        # 多样性评分：基础分 + 权威加分
        base_diversity_score = min(domain_count * 20, 60)  # 每个域名20分，基础分上限60
        authority_bonus = min(authoritative_count * 15, 40)  # 每个权威域名15分，加成上限40
        diversity_raw = base_diversity_score + authority_bonus

        # 相关性：基于关键词在证据中的命中比例（0-100）
        relevance_score = min(100.0, kw_hit_ratio * 120.0)
        # 多样性封顶：相关性不足时，即便域名多也不允许高分
        diversity_score = min(diversity_raw, relevance_score + 10.0)

        profile.metrics['diversity'] = MetricScore(
            name='来源多样性',
            score=diversity_score,
            weight=self.metric_weights.get('diversity', 0.25),
            evidence=f"域名:{domain_count} 权威:{authoritative_count} relevance:{relevance_score:.1f}"
        )

        # 4. 事实密度 (新增指标)
        # 统计总结中特定类型的实体作为"事实"代理
        quoted_terms = re.findall(r'"[^"]+"|\'[^\']+\'|「[^」]+」|"[^"]+"', summary)

        # 查找可能的技术术语/专有名词 (连续大写字母开头或包含数字字母组合)
        tech_terms = re.findall(r'\b[A-Z][A-Za-z0-9]+\b', summary)  # 如GPT-4, Transformer
        proper_nouns = re.findall(r'\b(?:[A-Z][a-z]+ )+[A-Z][a-z]+\b', summary)  # 如Artificial Intelligence

        # 查找数据点 (百分比、数值+单位)
        percentages = re.findall(r'\d+\.?\d*%', summary)
        numeric_with_units = re.findall(r'\d+\.?\d*\s*(?:GHz|MB|ms|s|ms|倍|年|月)', summary)

        fact_count = (len(quoted_terms) * 2 +  # 引用的内容权重更高
                     len(tech_terms) // 2 +   # 技术术语
                     len(proper_nouns) // 2 + # 专有名词
                     len(percentages) +
                     len(numeric_with_units))

        fact_density_score = min(fact_count * 8, 100)  # 调整系数，避免过高

        profile.metrics['fact_density'] = MetricScore(
            name='事实密度',
            score=fact_density_score,
            weight=self.metric_weights.get('fact_density', 0.15),
            evidence=f"检测到引用{len(quoted_terms)}处，技术术语{len(tech_terms)}个，数据点{len(percentages)+len(numeric_with_units)}个"
        )

    async def _perform_semantic_analysis(self, profile: TaskQualityProfile,
                                         task_result: Dict[str, Any],
                                         context: Dict[str, Any]) -> None:
        if not self.conflict_detection_enabled:
            profile.metrics['conflict'] = MetricScore(
                name='观点冲突性',
                score=0.0,
                weight=self.metric_weights.get('conflict', 0.0),
                evidence="冲突检测已禁用"
            )
            return
        """执行需要LLM的深层语义分析（观点冲突检测）"""
        summary = task_result.get("summary", "")
        if not summary or len(summary) < 30:  # 过短的总结跳过LLM分析
            profile.metrics['conflict'] = MetricScore(
                name='观点冲突性',
                score=0.0,
                weight=self.metric_weights.get('conflict', 0.15),
                evidence="总结文本过短，跳过冲突检测"
            )
            return

        try:
            # 使用修复后的检测方法
            conflict_result = await self._detect_conflict_with_llm(summary)

            if conflict_result and conflict_result.get('has_conflict', False):
                conflict_score = 100.0
                evidence = f"检测到观点冲突: {conflict_result.get('key_point', '未知')}"

                profile.metrics['conflict'] = MetricScore(
                    name='观点冲突性',
                    score=conflict_score,
                    weight=self.metric_weights.get('conflict', 0.0),  # 权重从配置读取
                    evidence=evidence
                )

                profile.tags.append("存在观点冲突")
                if 'key_point' in conflict_result and conflict_result['key_point']:
                    profile.key_findings.append(f"争议焦点: {conflict_result['key_point']}")
            else:
                profile.metrics['conflict'] = MetricScore(
                    name='观点冲突性',
                    score=0.0,
                    weight=self.metric_weights.get('conflict', 0.0),
                    evidence="未检测到明显观点冲突"
                )

        except Exception as e:
            logger.error(f"语义分析过程失败: {e}")
            # 出错时设置一个默认值，避免影响整个诊断
            profile.metrics['conflict'] = MetricScore(
                name='观点冲突性',
                score=0.0,
                weight=self.metric_weights.get('conflict', 0.0),
                evidence=f"检测过程异常: {str(e)[:50]}"
            )

    async def _detect_conflict_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """使用LLM检测文本中的观点冲突"""
        # 安全保护：如果未启用冲突检测，直接返回
        if not self.llm_client or not self.conflict_detection_enabled:
            logger.debug("冲突检测被禁用或LLM客户端不可用，跳过检测。")
            return None

        text_preview = text[:800] + "..." if len(text) > 800 else text

        prompt = f"""请分析以下研究总结，判断其是否提及了学术观点分歧、方法论争议、未决问题或对立看法。

    总结文本：请严格按照以下JSON格式回答，不要有任何额外的解释、标记或文字：
{{
    "has_conflict": true/false,
    "key_point": "简要描述冲突点或争议焦点，如无冲突则留空",
    "confidence": 0.0-1.0之间的置信度分数
}}"""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            import json

            sys_msg = SystemMessage(
                content="你是一个严谨的学术文本分析专家。请准确判断文本中是否存在观点冲突。只返回要求的JSON。")
            human_msg = HumanMessage(content=prompt)

            messages = [sys_msg, human_msg]

            # === 修复点2：采用更通用的异步调用模式 ===
            # 方法1: 尝试直接异步调用（最常见）
            try:
                # 注意：这里直接使用 await 调用客户端的异步方法
                response = await self.llm_client.agenerate(messages=[messages])
            except AttributeError:
                # 方法2: 如果客户端没有 agenerate，尝试 ainvoke
                logger.warning("llm_client.agenerate 不可用，尝试 ainvoke...")
                response = await self.llm_client.ainvoke(messages)
            except Exception as e:
                logger.error(f"异步调用LLM失败: {e}")
                return None

            # === 修复点3：更健壮的响应解析 ===
            result_text = ""

            # 尝试多种方式提取响应文本
            if hasattr(response, 'generations') and response.generations:
                # 标准 ChatResult 格式
                result_text = response.generations[0][0].text
            elif hasattr(response, 'content'):
                # 直接是 AIMessage 格式
                result_text = response.content
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                # 某些封装格式
                result_text = response.message.content
            elif isinstance(response, str):
                # 直接是字符串
                result_text = response
            else:
                # 最后尝试转换为字符串
                result_text = str(response)

            result_text = result_text.strip()

            # 清理可能存在的代码块标记
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)

            # 移除可能的 ```json ``` 包装
            result_text = re.sub(r'^```json\s*|\s*```$', '', result_text, flags=re.IGNORECASE)
            result_text = result_text.strip()

            if not result_text:
                logger.warning("从LLM响应中提取到的文本为空。")
                return None

            # 解析JSON
            try:
                result_dict = json.loads(result_text)
                # 验证必需字段
                if 'has_conflict' not in result_dict:
                    logger.warning("LLM响应缺少 'has_conflict' 字段。")
                    return None
                return result_dict
            except json.JSONDecodeError as e:
                logger.error(f"无法解析LLM的JSON响应。原始文本: {result_text[:200]}... 错误: {e}")
                return None

        except Exception as e:
            logger.error(f"冲突检测过程中发生未预期错误: {e}", exc_info=True)
            return None

    async def _assess_novelty(self, profile: TaskQualityProfile,
                              task_result: Dict[str, Any],
                              context: Dict[str, Any]) -> None:
        """评估信息相对于记忆库的新颖性"""
        if not self.memory_store:
            profile.metrics['novelty'] = MetricScore(
                name='信息新颖性',
                score=50.0,  # 无法评估时给中间分
                weight=self.metric_weights.get('novelty', 0.10),
                evidence="记忆库不可用，新颖性未评估"
            )
            return

        summary = task_result.get("summary", "")
        if not summary:
            profile.metrics['novelty'] = MetricScore(
                name='信息新颖性',
                score=0.0,
                weight=self.metric_weights.get('novelty', 0.10),
                evidence="无总结文本"
            )
            return

        try:
            # 在记忆库中检索相似记忆
            research_topic = context.get('research_topic', '')
            similar_memories = self.memory_store.search_by_similarity(
                query=summary[:500],  # 使用前500字符查询
                n_results=3,
                research_topic=research_topic
            )

            # 4:6（记忆相似度:时效性）
            if not similar_memories:
                novelty_memory = 100.0
                max_similarity = 0.0
            else:
                max_similarity = max([score for _, score in similar_memories])
                novelty_memory = max(0.0, 100.0 - (max_similarity * 100.0))

            # 时效性：从证据侧抽取年份，越接近当前年份越高；抽不到年份则给 50
            search_results = task_result.get("search_results") or []
            sources = task_result.get("source_urls") or task_result.get("sources") or []
            evidence_text = self._build_evidence_text(search_results=search_results, sources=list(sources))
            novelty_time = self._freshness_score_from_text(evidence_text)

            novelty_score = 0.4 * novelty_memory + 0.6 * novelty_time
            evidence = f"memory:{novelty_memory:.1f}(sim={max_similarity:.2f}) time:{novelty_time:.1f}"

            if novelty_score > 70:
                profile.tags.append("信息新颖")

            profile.metrics['novelty'] = MetricScore(
                name='信息新颖性',
                score=float(novelty_score),
                weight=self.metric_weights.get('novelty', 0.10),
                evidence=evidence
            )

        except Exception as e:
            logger.error(f"新颖性评估失败: {e}")
            profile.metrics['novelty'] = MetricScore(
                name='信息新颖性',
                score=50.0,
                weight=self.metric_weights.get('novelty', 0.10),
                evidence=f"评估过程出错: {str(e)[:50]}"
            )

    def _extract_required_keywords(self, topic: str) -> List[str]:
        """
        从 research_topic 抽取关键词（含专业术语）。
        策略：
        1) 抽取英文术语 token（agent, llm, rag...）
        2) 抽取中文连续片段，并按常见功能词拆分，避免整句作为单关键词
        3) 去重与轻量停用词过滤
        """
        if not topic:
            return []

        t = re.sub(r"[\(\)（）\[\]【】{}<>《》“”\"'`]", " ", topic)
        t = re.sub(r"[\s,，;；、/|：:。.!?？\n\r\t]+", " ", t).strip()

        kws: List[str] = []

        # 英文/数字术语
        for m in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", t):
            kws.append(m)

        # 中文片段（2字以上）
        zh_segments = re.findall(r"[\u4e00-\u9fff]{2,}", t)
        splitter = r"(?:在|的|与|和|及|以及|有关|关于|面向|针对|应用|研究|分析|总结|报告|领域|开发|现状|趋势)"
        for seg in zh_segments:
            parts = [p for p in re.split(splitter, seg) if p and len(p) >= 2]
            if parts:
                kws.extend(parts)
            else:
                # 兜底：避免整句 token 完全无法命中
                if 2 <= len(seg) <= 8:
                    kws.append(seg)
                elif len(seg) > 8:
                    kws.append(seg[:4])
                    kws.append(seg[-4:])

        # 若仍为空，回退到按空格切分
        if not kws:
            kws = [x for x in t.split(" ") if x]

        stopwords = {
            "的", "和", "与", "及", "以及", "研究", "分析", "总结", "报告",
            "相关", "方面", "问题", "情况", "进行", "如何"
        }
        cleaned: List[str] = []
        for tok in kws:
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() in stopwords:
                continue
            if len(tok) == 1 and not re.match(r"[A-Za-z]", tok):
                continue
            cleaned.append(tok)

        # 去重保持顺序
        seen: Set[str] = set()
        out: List[str] = []
        for k in cleaned:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _build_evidence_text(self, search_results: List[Dict[str, Any]], sources: List[str]) -> str:
        parts: List[str] = []
        for r in (search_results or [])[:8]:
            if not isinstance(r, dict):
                continue
            parts.append(str(r.get("title") or ""))
            parts.append(str(r.get("snippet") or ""))
            parts.append(str(r.get("url") or ""))
        for u in (sources or [])[:12]:
            parts.append(str(u))
        return "\n".join([p for p in parts if p]).lower()

    def _freshness_score_from_text(self, text: str) -> float:
        """
        从证据文本中粗略提取年份，计算时效性分数（0-100）。
        - 抽不到年份 => 50
        - 年份越接近当前年份越高（差 0 年=100，差 1 年=85，差 2 年=70，差>=5 年=30）
        """
        if not text:
            return 50.0
        years = [int(y) for y in re.findall(r"(19\\d{2}|20\\d{2})", text)]
        if not years:
            return 50.0
        y = max(years)
        now = datetime.now().year
        d = max(0, now - y)
        if d == 0:
            return 100.0
        if d == 1:
            return 85.0
        if d == 2:
            return 70.0
        if d == 3:
            return 55.0
        if d == 4:
            return 40.0
        return 30.0

    def _generate_tags_and_suggestions(self, profile: TaskQualityProfile) -> None:
        """基于评估结果生成标签和决策建议"""
        # 1. 存储建议逻辑
        validity_score = profile.get_metric_score('validity')

        if validity_score < 50:
            profile.storage_suggestion = "REJECT"
        elif profile.composite_score >= self.storage_threshold:
            profile.storage_suggestion = "IMMEDIATE"
        elif profile.composite_score >= 60:
            profile.storage_suggestion = "VERIFY"
        else:
            profile.storage_suggestion = "REJECT"

        # 2. 生成下一步研究建议
        suggestions = []

        # 来源多样性建议（仅作为一般性优化提示，不再形成单独缺陷标签）
        diversity_score = profile.get_metric_score('diversity')
        if diversity_score < 40:
            if diversity_score < 20:
                suggestions.append("可进一步补充信息来源以提升多样性")
            else:
                suggestions.append("可适当增加权威来源（如.edu, .gov, 知名期刊）以优化结果")

        # 信息深度建议
        saturation_score = profile.get_metric_score('saturation')
        if saturation_score < 50:
            suggestions.append("需要更深入的技术细节、实现原理或具体案例")
            profile.tags.append("需深化")

        # 观点冲突建议
        if "存在观点冲突" in profile.tags:
            suggestions.append("针对已识别的争议点，进行正反双方的论据深入收集")
            profile.tags.append("需厘清争议")

        # 新颖性建议
        novelty_score = profile.get_metric_score('novelty')
        if novelty_score < 30:
            suggestions.append("信息与已有知识高度重复，建议转向该主题下更新或更小众的细分方向")
            profile.tags.append("需创新角度")

        # 事实密度建议
        fact_density_score = profile.get_metric_score('fact_density')
        if fact_density_score < 40:
            suggestions.append("需要补充具体数据、引用或实证案例")
            profile.tags.append("需增事实")

        if suggestions:
            # 取最重要的3条建议
            priority_suggestions = suggestions[:3]
            profile.next_research_suggestion = "；".join(priority_suggestions)

    def _extract_unique_domains(self, urls: List[str]) -> List[str]:
        """从URL列表中提取独立域名"""
        domains = set()
        for url in urls:
            if not url or not isinstance(url, str):
                continue

            try:
                parsed = urlparse(url)
                domain = parsed.netloc

                if domain and domain not in ["localhost", "127.0.0.1"]:
                    # 移除www前缀和端口号
                    if domain.startswith("www."):
                        domain = domain[4:]
                    if ":" in domain:
                        domain = domain.split(":")[0]
                    domains.add(domain.lower())
            except Exception:
                continue

        return list(domains)


class ResearchCycleDiagnoser:
    """
    研究轮次诊断器。
    职责：整合所有任务的诊断结果，生成轮次级缺陷报告和全局决策。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # 研究质量阈值：默认 60 分（低于此阈值建议继续下一轮）
        self.cycle_continue_threshold = self.config.get('cycle_continue_threshold', 60.0)

    def generate_deficiency_report(self, profiles: List[TaskQualityProfile],current_cycle: int) -> CycleDeficiencyReport:
        """
        基于所有任务的质量画像，生成轮次缺陷报告。

        Args:
            profiles: 本轮所有任务的质量画像列表。
            current_cycle: 当前研究轮次。

        Returns:
            轮次缺陷报告对象。
        """
        if not profiles:
            return CycleDeficiencyReport(
                cycle_number=current_cycle,
                report_text="无任何有效任务结果，无法生成报告。",
                average_composite_score=0.0,
                total_tasks=0,
                valid_tasks=0,
                global_continue_suggestion=False,
                primary_deficiency="无数据"
            )

        # 1. 按问题分类任务
        categories = {
            "high_quality": [],      # 高质量，可立即存储
            "needs_depth": [],       # 需要更深入信息
            "has_conflicts": [],     # 存在观点冲突
            "low_novelty": [],       # 新颖性不足
            "invalid": []            # 无效结果
        }

        valid_profiles = []
        total_composite = 0.0

        for profile in profiles:
            validity_score = profile.get_metric_score('validity')

            if validity_score < 50:
                categories["invalid"].append(profile.task_id)
                continue

            valid_profiles.append(profile)
            total_composite += profile.composite_score

            # 分类逻辑
            if profile.composite_score >= 80:
                categories["high_quality"].append(profile.task_id)
            # 不再将来源多样性不足作为单独缺陷类别参与决策，只保留深度、冲突等核心缺陷
            elif profile.get_metric_score('saturation') < 50:
                categories["needs_depth"].append(profile.task_id)
            elif "存在观点冲突" in profile.tags:
                categories["has_conflicts"].append(profile.task_id)
            elif profile.get_metric_score('novelty') < 30:
                categories["low_novelty"].append(profile.task_id)
            else:
                categories["high_quality"].append(profile.task_id)

        # 2. 计算统计信息
        avg_score = total_composite / len(valid_profiles) if valid_profiles else 0.0
        total_tasks = len(profiles)
        valid_tasks = len(valid_profiles)

        # 3. 生成报告文本
        report_parts = []
        report_parts.append(f"第{current_cycle}轮研究质量评估报告")
        report_parts.append("=" * 40)

        if categories["high_quality"]:
            report_parts.append(f"高质量任务 ({len(categories['high_quality'])}个): {', '.join(map(str, categories['high_quality']))}")
        # 不再单独在报告中突出与来源相关的缺陷类别
        if categories["needs_depth"]:
            report_parts.append(f"需深化任务 ({len(categories['needs_depth'])}个): {', '.join(map(str, categories['needs_depth']))}")
        if categories["has_conflicts"]:
            report_parts.append(f"有争议任务 ({len(categories['has_conflicts'])}个): {', '.join(map(str, categories['has_conflicts']))}")
        if categories["low_novelty"]:
            report_parts.append(f"低新颖性任务 ({len(categories['low_novelty'])}个): {', '.join(map(str, categories['low_novelty']))}")
        if categories["invalid"]:
            report_parts.append(f"无效任务 ({len(categories['invalid'])}个): {', '.join(map(str, categories['invalid']))}")

        report_parts.append("")
        report_parts.append(f"轮次统计:")
        report_parts.append(f"   - 总任务数: {total_tasks}")
        report_parts.append(f"   - 有效任务数: {valid_tasks}")
        report_parts.append(f"   - 平均质量分: {avg_score:.1f}/100")

        # 4. 确定主要缺陷和全局决策
        primary_deficiency = "无显著缺陷"
        global_continue = False

        # 确定最主要缺陷类别
        deficiency_priority = ["invalid", "has_conflicts", "needs_depth", "low_novelty"]
        for category in deficiency_priority:
            if categories.get(category):
                primary_deficiency = self._get_deficiency_name(category)
                break

        # 全局决策逻辑（强约束版）
        # 业务要求：只有当本轮平均综合质量分「严格大于」阈值时，才允许进入报告阶段；
        # 否则必须继续下一轮（在未达到最大轮次前）。
        max_cycles = self.config.get('max_cycles', 3)
        if current_cycle < max_cycles and avg_score <= self.cycle_continue_threshold:
            global_continue = True
            report_parts.append(
                f"决策: 平均分{avg_score:.1f} 不高于阈值 {self.cycle_continue_threshold}，"
                f"且当前轮次为第{current_cycle}轮，建议继续第{current_cycle+1}轮研究"
            )
        else:
            # 当达到最大轮次或者平均分超过阈值时，结束研究流程，进入报告阶段
            if avg_score > self.cycle_continue_threshold:
                reason = "平均分已超过质量阈值，研究质量达标"
            else:
                reason = f"已达到最大轮次 {max_cycles}，按安全策略结束研究流程"
            report_parts.append(f"决策: {reason}，可进入报告生成阶段")

        report_text = "\n".join(report_parts)

        return CycleDeficiencyReport(
            cycle_number=current_cycle,
            task_categories=categories,
            report_text=report_text,
            average_composite_score=avg_score,
            total_tasks=total_tasks,
            valid_tasks=valid_tasks,
            global_continue_suggestion=global_continue,
            primary_deficiency=primary_deficiency
        )

    def _get_deficiency_name(self, category: str) -> str:
        """将内部类别名转换为可读的缺陷名称"""
        names = {
            "invalid": "无效结果",
            "has_conflicts": "观点冲突",
            "needs_depth": "深度不足",
            "low_novelty": "新颖性低",
            "high_quality": "高质量"
        }
        return names.get(category, category)