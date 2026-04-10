# src/memory/working_memory.py
"""
工作记忆（WorkingMemory）模块。
职责：管理当前研究任务的会话生命周期，协调长期记忆检索与上下文维护。
设计理念：模拟人类的工作记忆，作为短期上下文与长期知识库之间的桥梁。
"""
import logging
import asyncio
import re
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from .memory_store import MemoryStore, MemoryRecord, MemoryType, MemoryPriority
from .query_enhancer import BaseQueryEnhancer

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    工作记忆类。
    每个研究会话（session）对应一个独立实例，贯穿整个LangGraph工作流。
    """

    def __init__(
            self,
            session_id: str,
            memory_store: MemoryStore,
            max_context_items: int = 20
    ):
        """
        初始化工作记忆。

        Args:
            session_id: 当前研究会话的唯一标识符。
            memory_store: 记忆存储实例，用于访问长期记忆。
            max_context_items: 工作记忆容量（最大上下文项数），防止上下文无限膨胀。
        """
        self.session_id = session_id
        self.memory_store = memory_store
        self.max_context_items = max_context_items

        # 工作记忆缓冲区：存储当前会话的上下文项
        # 每个项格式：{"type": str, "content": str, "timestamp": datetime, "metadata": dict}
        self.context_buffer: List[Dict[str, Any]] = []

        # 当前检索到的相关记忆缓存
        self.relevant_memories: List[MemoryRecord] = []

        # 会话元数据
        self.metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "context_item_count": 0,
            "retrieval_count": 0
        }

        logger.info(f"工作记忆初始化完成，会话ID: {session_id}")

    def add_context(
            self,
            item_type: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加一项内容到工作记忆上下文。

        Args:
            item_type: 上下文项类型，如 'user_query', 'plan_result', 'task_summary', 'search_result'。
            content: 上下文内容文本。
            metadata: 附加元数据。

        Returns:
            添加项的ID（基于时间戳的哈希）。
        """
        from hashlib import md5
        import json

        # 生成该项的唯一ID
        item_id = md5(
            f"{item_type}_{content}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        context_item = {
            "id": item_id,
            "type": item_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # 添加到缓冲区开头（新的在前）
        self.context_buffer.insert(0, context_item)

        # 维护缓冲区大小限制
        if len(self.context_buffer) > self.max_context_items:
            removed = self.context_buffer.pop()  # 移除最旧的项
            logger.debug(f"工作记忆达到上限，移除了旧项: {removed['type']}")

        self.metadata["context_item_count"] = len(self.context_buffer)
        self.metadata["last_updated"] = datetime.now().isoformat()

        logger.debug(f"添加上下文项: {item_type}, ID: {item_id}, 内容长度: {len(content)}")
        return item_id

    async def retrieve_relevant_memories(
            self,
            current_topic: str,
            current_task_intent: Optional[str] = None,
            strategy: str = "hybrid",
            limit: int = 5,
            query_enhancer: Optional[BaseQueryEnhancer] = None
    ) -> List[MemoryRecord]:
        """
        检索与当前研究主题/任务相关的历史记忆（支持多查询增强）。

        Args:
            current_topic: 当前研究主题。
            current_task_intent: 当前子任务的研究意图（可选，用于更精确的检索）。
            strategy: 检索策略，'hybrid'（混合）或 'semantic'（语义）。
            limit: 最大返回记忆数量。
            query_enhancer: 可选的查询增强器实例（如 HyDEEnhancer, MQEEnhancer）。

        Returns:
            相关记忆记录列表，按相关性排序。
        """
        # 构建基础检索查询
        raw_query = current_topic
        if current_task_intent:
            raw_query = f"{current_topic} {current_task_intent}"

        # === 查询增强与类型处理 ===
        effective_queries = raw_query  # 可能被增强为 str 或 List[str]
        enhancement_used = "None"
        is_multi_query = False

        if query_enhancer is not None:
            try:
                logger.info(f"正在使用 {query_enhancer} 对查询进行增强...")
                enhanced_result = await query_enhancer.enhance(
                    original_query=raw_query,
                    context={"research_topic": current_topic, "intent": current_task_intent}
                )

                if enhanced_result:  # 增强器可能返回 str 或 List[str]
                    effective_queries = enhanced_result
                    enhancement_used = query_enhancer.__class__.__name__

                    # 判断是否为多查询 (MQE)
                    if isinstance(effective_queries, list):
                        is_multi_query = True
                        logger.info(f"MQE 扩展成功。收到 {len(effective_queries)} 个查询变体。")
                        for i, q in enumerate(effective_queries, 1):
                            logger.debug(f"  MQE 变体 {i}: '{q[:60]}...'")
                    else:
                        # 单个查询 (如 HyDE 或其他返回 str 的增强器)
                        logger.info(f"查询增强成功 (单条)。原始: '{raw_query[:50]}...'")
                        logger.debug(f"增强后查询预览: '{effective_queries[:100]}...'")
                else:
                    logger.warning("查询增强器返回了空结果，将使用原始查询。")
                    effective_queries = raw_query

            except Exception as e:
                logger.error(f"查询增强过程失败: {e}，将回退到原始查询。")
                effective_queries = raw_query

        # 记录最终用于检索的查询（类型）
        query_type = "多查询列表" if is_multi_query else "单查询"
        logger.info(f"最终检索输入: {query_type} (增强策略: {enhancement_used})")

        # === 多查询并行检索逻辑 ===
        all_memory_results = []  # 存储 (memory, similarity_score, source_query) 的列表

        if is_multi_query and isinstance(effective_queries, list):
            # MQE 模式：并行检索多个查询变体
            # 若变体列表未显式包含「原始 raw_query」，先并入原句检索，再与各变体在 _merge 中取 max，
            # 避免某条记忆最匹配用户原问题、却不匹配任何变体时，合并分数反而低于纯 baseline。
            variant_list = [q for q in effective_queries if q and str(q).strip()]
            rq = (raw_query or "").strip()
            vnorm = {str(q).strip() for q in variant_list}
            if rq and rq not in vnorm:
                variant_list.insert(0, raw_query)

            retrieval_tasks = []
            queries_for_tasks: List[str] = []
            for query in variant_list:
                if not query or not query.strip():
                    continue
                queries_for_tasks.append(query.strip())

                task = self._retrieve_single_query(
                    query=query,
                    strategy=strategy,
                    current_topic=current_topic,
                    limit_per_query=limit * 2
                )
                retrieval_tasks.append(task)

            if retrieval_tasks:
                # 并行执行所有查询的检索
                logger.info(f"开始并行检索 {len(retrieval_tasks)} 个查询变体...")
                try:
                    task_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

                    # 处理每个查询的检索结果
                    for i, result in enumerate(task_results):
                        if isinstance(result, Exception):
                            logger.error(f"第 {i + 1} 个查询检索失败: {result}")
                            continue

                        memories_with_scores = result
                        if memories_with_scores:
                            src_q = queries_for_tasks[i] if i < len(queries_for_tasks) else ""
                            for memory, score in memories_with_scores:
                                all_memory_results.append((memory, score, src_q))

                except Exception as e:
                    logger.error(f"并行检索执行失败: {e}")

        else:
            # 单查询模式
            single_query = effective_queries if isinstance(effective_queries, str) else raw_query
            logger.info(f"执行单查询检索: '{single_query[:50]}...'")

            memories_with_scores = await self._retrieve_single_query(
                query=single_query,
                strategy=strategy,
                current_topic=current_topic,
                limit_per_query=limit * 2
            )

            for memory, score in memories_with_scores:
                all_memory_results.append((memory, score, single_query))

        # === 结果合并、去重与排序 ===
        final_memories = self._merge_and_rank_memories(
            all_memory_results=all_memory_results,
            limit=limit
        )

        # 缓存最终结果
        self.relevant_memories = final_memories
        self.metadata["retrieval_count"] += 1
        # 记录增强统计
        self.metadata.setdefault("enhancement_stats", {})
        self.metadata["enhancement_stats"][enhancement_used] = self.metadata["enhancement_stats"].get(enhancement_used,
                                                                                                      0) + 1
        if is_multi_query:
            self.metadata.setdefault("mqe_stats", {})
            self.metadata["mqe_stats"]["multi_query_count"] = self.metadata["mqe_stats"].get("multi_query_count", 0) + 1

        logger.info(
            f"检索完成。合并后得到 {len(final_memories)} 条唯一记忆 (使用策略: {strategy}, 增强: {enhancement_used})")
        return final_memories

    async def _retrieve_single_query(
            self,
            query: str,
            strategy: str,
            current_topic: str,
            limit_per_query: int
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        辅助方法：针对单个查询执行检索，返回（记忆，相似度分数）的列表。

        Args:
            query: 单个查询字符串。
            strategy: 检索策略。
            current_topic: 研究主题，用于过滤。
            limit_per_query: 每个查询返回的最大结果数。

        Returns:
            包含记忆和对应相似度分数的元组列表。
        """
        try:
            if strategy == "hybrid":
                # 使用混合检索
                results = self.memory_store.search_by_similarity(
                    query=query,
                    n_results=limit_per_query,
                    research_topic=current_topic
                )
                return results
            elif strategy == "semantic":
                # 纯语义检索
                results = self.memory_store.search_by_similarity(
                    query=query,
                    n_results=limit_per_query,
                    research_topic=current_topic
                )
                return results
            else:
                logger.warning(f"未知的检索策略 '{strategy}'，使用语义检索。")
                return self.memory_store.search_by_similarity(
                    query=query,
                    n_results=limit_per_query,
                    research_topic=current_topic
                )
        except Exception as e:
            logger.error(f"单查询检索失败 (查询: '{query[:30]}...'): {e}")
            return []

    def _merge_and_rank_memories(
            self,
            all_memory_results: List[Tuple[MemoryRecord, float, str]],
            limit: int
    ) -> List[MemoryRecord]:
        """
        辅助方法：合并来自多个查询的检索结果，去重并按综合分数排序。

        Args:
            all_memory_results: 列表，元素为 (memory, score, source_query)。
            limit: 最终返回的记忆数量限制。

        Returns:
            排序后的唯一记忆列表。
        """
        if not all_memory_results:
            return []

        # 第一步：按记忆ID分组，合并来自不同查询的分数
        memory_scores = {}  # memory_id -> {'memory': MemoryRecord, 'scores': List[float], 'queries': List[str]}

        for memory, score, source_query in all_memory_results:
            mem_id = memory.id
            if mem_id not in memory_scores:
                memory_scores[mem_id] = {
                    'memory': memory,
                    'scores': [],
                    'queries': [],
                    'max_score': 0.0
                }

            memory_scores[mem_id]['scores'].append(score)
            memory_scores[mem_id]['queries'].append(source_query[:30])
            if score > memory_scores[mem_id]['max_score']:
                memory_scores[mem_id]['max_score'] = score

        # 第二步：为每个唯一记忆计算综合得分
        scored_memories = []
        for mem_id, data in memory_scores.items():
            memory = data['memory']
            scores = data['scores']

            # 综合评分策略
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = data['max_score']
            query_count = len(data['queries'])

            # 综合分数 = 平均分 * (1 + 广度奖励) + 记忆自身优先级加权
            breadth_bonus = 0.1 * min(query_count - 1, 2)
            priority_bonus = memory.priority.value * 0.05

            composite_score = avg_score * (1.0 + breadth_bonus) + priority_bonus

            scored_memories.append((memory, composite_score, avg_score, max_score, query_count))

        # 第三步：按综合分数降序排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # 第四步：记录排序详情（调试用）
        if scored_memories and logger.isEnabledFor(logging.DEBUG):
            logger.debug("记忆排序详情 (Top 5):")
            for i, (memory, comp_score, avg_score, max_score, q_count) in enumerate(scored_memories[:5]):
                logger.debug(f"  {i + 1}. ID:{memory.id[:8]} 综合:{comp_score:.3f} 平均:{avg_score:.3f} "
                             f"最高:{max_score:.3f} 来自{q_count}个查询")

        # 返回排序后的记忆对象列表
        return [mem for mem, _, _, _, _ in scored_memories[:limit]]

    def get_context_for_llm(
            self,
            token_limit: Optional[int] = None,
            include_recent_items: int = 10,
            include_memories: bool = True
    ) -> Tuple[str, Dict[str, int]]:
        """
        将工作记忆格式化为适合LLM提示词的上下文字符串。

        Args:
            token_limit: 粗略的token数量限制（基于字符数估算）。
            include_recent_items: 最多包含的最近上下文项数量。
            include_memories: 是否包含相关记忆。

        Returns:
            Tuple[formatted_context, stats]
            - formatted_context: 格式化的上下文字符串
            - stats: 统计信息，如字符数、项数等
        """
        # 选择要包含的上下文项（最近的N项）
        recent_items = self.context_buffer[:include_recent_items]

        # 构建格式化的上下文
        context_parts = []

        # 1. 添加相关记忆（如果有）
        memory_count = 0
        if include_memories and self.relevant_memories:
            context_parts.append("## 相关背景知识")
            for i, memory in enumerate(self.relevant_memories, 1):
                # 格式化记忆显示
                confidence = memory.metadata.get("confidence", 1.0)
                confidence_str = f" (置信度: {confidence:.2f})" if confidence < 1.0 else ""

                context_parts.append(
                    f"{i}. **{memory.memory_type.value}**{confidence_str}\n"
                    f"   {memory.content}"
                )
                memory_count += 1
            context_parts.append("")  # 空行分隔

        # 2. 添加上下文历史
        context_item_count = 0
        if recent_items:
            context_parts.append("## 当前会话记录")
            for item in recent_items:
                # 格式化时间戳
                time_str = datetime.fromisoformat(
                    item['timestamp']).strftime("%H:%M:%S")

                context_parts.append(
                    f"[{time_str}] **{item['type']}**\n"
                    f"{item['content']}"
                )
                context_item_count += 1
                context_parts.append("")  # 项之间空行

        formatted_context = "\n".join(context_parts).strip()

        # 统计信息
        stats = {
            "total_characters": len(formatted_context),
            "memory_count": memory_count,
            "context_item_count": context_item_count,
            "estimated_tokens": len(formatted_context) // 3
        }

        # 简单的token限制处理
        if token_limit and stats["estimated_tokens"] > token_limit:
            logger.warning(
                f"上下文超过token限制 ({stats['estimated_tokens']} > {token_limit})，进行截断"
            )
            max_chars = token_limit * 3
            if len(formatted_context) > max_chars:
                formatted_context = formatted_context[:max_chars] + "\n...（上下文已截断）"

        logger.debug(
            f"生成LLM上下文: {stats['memory_count']}记忆, {stats['context_item_count']}项, "
            f"{stats['total_characters']}字符, ~{stats['estimated_tokens']}tokens"
        )

        return formatted_context, stats

    def store_important_findings(
            self,
            content: str,
            memory_type: MemoryType = MemoryType.INSIGHT,
            priority: MemoryPriority = MemoryPriority.HIGH,
            research_topic: Optional[str] = None,
            confidence: float = 0.9,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        将重要发现存储到长期记忆。

        Args:
            content: 要存储的内容。
            memory_type: 记忆类型。
            priority: 记忆优先级。
            research_topic: 研究主题（如果为None，则尝试从上下文中推断）。
            confidence: 置信度分数。

        Returns:
            存储的记忆ID。
        """
        if not research_topic and self.context_buffer:
            for item in self.context_buffer:
                if item["type"] == "user_query" and "research_topic" in item.get("metadata", {}):
                    research_topic = item["metadata"]["research_topic"]
                    break

        if not research_topic:
            research_topic = f"session_{self.session_id[:8]}"
            logger.warning(f"无法确定研究主题，使用会话ID: {research_topic}")

        # 存储到长期记忆
        extra_metadata = metadata or {}
        if not isinstance(extra_metadata, dict):
            extra_metadata = {"_metadata": str(extra_metadata)}
        memory_id = self.memory_store.add_memory(
            content=content,
            memory_type=memory_type,
            priority=priority,
            research_topic=research_topic,
            confidence=confidence,
            metadata={
                "session_id": self.session_id,
                "stored_from": "working_memory",
                "context_item_count": len(self.context_buffer),
                **extra_metadata,
            }
        )

        logger.info(
            f"存储重要发现到长期记忆: ID={memory_id[:8]}, "
            f"类型={memory_type.value}, 主题={research_topic}"
        )

        # 运行时可见反馈：默认关闭，避免日志泄露内容预览
        show_writes = os.getenv("SHOW_LONGTERM_MEMORY_WRITES", "false").lower() == "true"
        if show_writes:
            logger.info(
                "[长期记忆] 已写入 ID=%s 类型=%s 主题=%s content_len=%s",
                memory_id[:8],
                memory_type.value,
                research_topic,
                len(content or ""),
            )

        return memory_id

    def clear_context_buffer(self, keep_recent: int = 5) -> int:
        """
        清理上下文缓冲区，保留最近的N项。

        Args:
            keep_recent: 保留的最近项数。

        Returns:
            清理的项数。
        """
        if len(self.context_buffer) <= keep_recent:
            return 0

        removed_count = len(self.context_buffer) - keep_recent
        self.context_buffer = self.context_buffer[:keep_recent]

        self.metadata["context_item_count"] = len(self.context_buffer)
        logger.info(f"清理上下文缓冲区，移除了 {removed_count} 项，保留 {keep_recent} 项")

        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """
        获取工作记忆的统计信息。

        Returns:
            统计信息字典。
        """
        stats = self.metadata.copy()
        stats.update({
            "buffer_size": len(self.context_buffer),
            "relevant_memories_count": len(self.relevant_memories),
            "context_types": {},
            "oldest_item_age_minutes": 0
        })

        # 统计上下文类型分布
        for item in self.context_buffer:
            item_type = item["type"]
            stats["context_types"][item_type] = stats["context_types"].get(item_type, 0) + 1

        # 计算最旧项的年龄（分钟）
        if self.context_buffer:
            oldest_time = datetime.fromisoformat(self.context_buffer[-1]["timestamp"])
            age_minutes = (datetime.now() - oldest_time).total_seconds() / 60
            stats["oldest_item_age_minutes"] = round(age_minutes, 1)

        return stats
