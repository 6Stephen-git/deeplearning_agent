"""
生成 AI 资讯/论文解读方向的评估资产：
1) 扩容 URL 清单（120+）
2) 80 条 query 数据集（含主题桶、难度、gold_answer）

用法：
  python scripts/generate_ai_eval_assets.py
  python scripts/generate_ai_eval_assets.py --query-count 80 --url-count 140
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _expanded_url_pool(profile: str = "mixed") -> List[str]:
    # 以官方文档/论文平台/研究博客为主，尽量减少二手转载站点
    seeds = [
        "https://arxiv.org/",
        "https://openreview.net/",
        "https://proceedings.mlr.press/",
        "https://aclanthology.org/",
        "https://docs.langchain.com/",
        "https://docs.langchain.com/langsmith",
        "https://platform.openai.com/docs/",
        "https://ai.google.dev/",
        "https://www.anthropic.com/engineering",
        "https://www.microsoft.com/en-us/research/blog/",
        "https://huggingface.co/blog",
        "https://developer.nvidia.com/blog",
        "https://aws.amazon.com/blogs/machine-learning/",
        "https://cloud.google.com/blog/products/ai-machine-learning",
        "https://azure.microsoft.com/en-us/blog/topics/ai-machine-learning/",
        "https://engineering.fb.com/",
        "https://deepmind.google/discover/blog/",
        "https://www.databricks.com/blog",
        "https://cohere.com/blog",
        "https://www.sequoiacap.com/article/agentic-ai-2025/",
    ]

    arxiv_topics = [
        "https://arxiv.org/list/cs.AI/recent",
        "https://arxiv.org/list/cs.CL/recent",
        "https://arxiv.org/list/cs.LG/recent",
        "https://arxiv.org/list/cs.IR/recent",
        "https://arxiv.org/list/cs.HC/recent",
        "https://arxiv.org/list/cs.CY/recent",
        "https://arxiv.org/list/stat.ML/recent",
        "https://arxiv.org/list/cs.SE/recent",
    ]

    docs = [
        "https://platform.openai.com/docs/guides/function-calling",
        "https://platform.openai.com/docs/guides/structured-outputs",
        "https://platform.openai.com/docs/guides/retrieval",
        "https://platform.openai.com/docs/guides/evals",
        "https://docs.anthropic.com/",
        "https://docs.anthropic.com/en/docs/build-with-claude/tool-use",
        "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering",
        "https://docs.langchain.com/docs/introduction",
        "https://docs.langchain.com/docs/concepts/rag",
        "https://docs.langchain.com/docs/concepts/retrieval",
        "https://docs.langchain.com/docs/concepts/evaluation",
        "https://docs.langchain.com/langgraph",
        "https://smith.langchain.com/",
        "https://docs.llamaindex.ai/",
        "https://docs.llamaindex.ai/en/stable/optimizing/basic_strategies/basic_strategies/",
        "https://docs.weaviate.io/",
        "https://docs.pinecone.io/",
        "https://cookbook.openai.com/",
        "https://ai.google.dev/gemini-api/docs",
        "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
    ]

    blog_hosts = [
        "https://huggingface.co/blog",
        "https://www.anthropic.com/engineering",
        "https://openai.com/index/",
        "https://deepmind.google/discover/blog/",
        "https://www.microsoft.com/en-us/research/blog/",
        "https://developer.nvidia.com/blog",
        "https://aws.amazon.com/blogs/machine-learning/",
        "https://cloud.google.com/blog/products/ai-machine-learning",
        "https://www.databricks.com/blog",
        "https://cohere.com/blog",
        "https://blog.google/technology/ai/",
        "https://blogs.nvidia.com/blog/category/deep-learning/",
        "https://engineering.atspotify.com/",
        "https://engineering.linkedin.com/blog",
        "https://netflixtechblog.com/",
        "https://engineering.salesforce.com/",
        "https://dropbox.tech/machine-learning",
        "https://research.facebook.com/blog/",
        "https://stability.ai/news",
        "https://www.pinecone.io/learn/",
        "https://weaviate.io/blog",
        "https://milvus.io/blog",
        "https://qdrant.tech/blog/",
        "https://www.elastic.co/blog",
        "https://www.snowflake.com/blog/",
    ]
    blog_paths = [
        "agent",
        "agents",
        "rag",
        "retrieval",
        "evaluation",
        "benchmark",
        "tool-use",
        "function-calling",
        "multimodal",
        "reasoning",
        "safety",
        "alignment",
        "evaluation",
        "evals",
        "enterprise",
        "production",
        "retrieval",
        "search",
        "vector-database",
        "memory",
        "orchestration",
        "benchmarks",
        "latency",
        "cost",
        "prompt-engineering",
        "fine-tuning",
        "observability",
        "guardrails",
    ]
    blog_urls: List[str] = []
    for host in blog_hosts:
        for p in blog_paths:
            blog_urls.append(f"{host.rstrip('/')}/{p}")

    arxiv_keywords = [
        "agent",
        "agentic",
        "retrieval-augmented generation",
        "RAG evaluation",
        "tool use",
        "function calling",
        "long context",
        "reranking",
        "groundedness",
        "faithfulness",
        "multimodal agent",
        "benchmark",
        "alignment",
        "safety",
        "hallucination",
        "self-reflection",
        "planning",
        "memory",
        "multi-agent",
        "orchestration",
    ]
    arxiv_search_urls: List[str] = []
    for kw in arxiv_keywords:
        q = kw.replace(" ", "+")
        arxiv_search_urls.append(f"https://arxiv.org/search/?query={q}&searchtype=all&source=header")
        arxiv_search_urls.append(f"https://arxiv.org/search/cs?query={q}&searchtype=all&abstracts=show&order=-announced_date_first&size=200")
    # 增加高成功率 arXiv 检索组合，提升可下载文档总量
    years = ["2023", "2024", "2025", "2026"]
    for kw in arxiv_keywords:
        base_kw = kw.replace(" ", "+")
        for y in years:
            q = f"{base_kw}+{y}"
            arxiv_search_urls.append(
                f"https://arxiv.org/search/cs?query={q}&searchtype=all&abstracts=show&order=-announced_date_first&size=200"
            )
            arxiv_search_urls.append(
                f"https://arxiv.org/search/?query={q}&searchtype=all&source=header"
            )

    hf_tags = [
        "rag",
        "agent",
        "llm",
        "evaluation",
        "safety",
        "multimodal",
        "benchmark",
        "reasoning",
        "inference",
        "tool-use",
    ]
    hf_urls = [f"https://huggingface.co/blog?tag={t}" for t in hf_tags]

    if profile == "arxiv-heavy":
        extra: List[str] = []
        # 通过主题词 x 年份 x 排序扩展高成功率URL
        years = ["2021", "2022", "2023", "2024", "2025", "2026"]
        sorts = ["-announced_date_first", "-submitted_date", "relevance"]
        for kw in arxiv_keywords:
            k2 = kw.replace(" ", "+")
            for y in years:
                for srt in sorts:
                    q = f"{k2}+{y}"
                    extra.append(
                        f"https://arxiv.org/search/cs?query={q}&searchtype=all&abstracts=show&order={srt}&size=200"
                    )
        all_urls = seeds[:1] + arxiv_topics + arxiv_search_urls + extra
    else:
        all_urls = seeds + arxiv_topics + docs + blog_urls + arxiv_search_urls + hf_urls
    # 保序去重
    out: List[str] = []
    seen = set()
    for u in all_urls:
        u = u.strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _bucket_queries() -> Dict[str, List[Dict[str, str]]]:
    return {
        "definition": [
            {"query": "2024-2026 年 agentic AI 的通用定义如何演进？", "gold": "应比较主流定义中的规划、工具调用、记忆与自主性边界。"},
            {"query": "AI Agent 与 workflow automation 的边界是什么？", "gold": "应区分自治决策与固定流程编排，并给出可验证例子。"},
            {"query": "RAG 与 Agent 的能力边界在企业场景中如何划分？", "gold": "应说明检索补全与行动执行的职责边界。"},
            {"query": "什么是 groundedness，在 Agent 评测中的作用是什么？", "gold": "应解释回答可证据支持性及其在质量门控中的意义。"},
            {"query": "什么是 tool calling 的结构化约束，为什么重要？", "gold": "应解释参数结构化、错误恢复和可观测性的关系。"},
            {"query": "HyDE 与 Query Rewrite 的本质差异是什么？", "gold": "应对比假设答案向量化与文本改写的机制差异。"},
            {"query": "MQE 与单 query 检索相比，核心优势与代价是什么？", "gold": "应指出召回覆盖提升与噪声、时延、成本的权衡。"},
            {"query": "什么是 multi-agent orchestration？何时不该使用？", "gold": "应给出协作收益条件与复杂度上升风险。"},
            {"query": "LLM-as-a-judge 在评估里的常见偏差有哪些？", "gold": "应指出立场偏置、长度偏置与提示词敏感性。"},
            {"query": "生产级 Agent 的最小可用能力清单是什么？", "gold": "应覆盖可观测性、重试、限流、回退与评估闭环。"},
        ],
        "architecture": [
            {"query": "企业级 Agent 系统常见的分层架构是什么？", "gold": "应覆盖任务层、工具层、记忆层、评估层及网关。"},
            {"query": "LangGraph 类工作流与代码驱动 orchestration 的取舍", "gold": "应比较可视化治理、灵活性与调试成本。"},
            {"query": "RAG 系统中两阶段检索+重排架构如何设计？", "gold": "应描述召回层与精排层职责及接口。"},
            {"query": "长上下文 + 检索混合架构的典型实现模式", "gold": "应比较纯长上下文与检索增强的组合策略。"},
            {"query": "多工具代理如何做权限隔离与沙箱执行？", "gold": "应说明工具白名单、参数审计和执行边界。"},
            {"query": "在线 Agent 的状态管理应该如何分层？", "gold": "应区分会话态、任务态、长期记忆态。"},
            {"query": "可回放的 tracing 架构应包含哪些关键字段？", "gold": "应包含输入、输出、中间决策、检索证据和耗时。"},
            {"query": "评估服务与在线推理服务分离部署的优缺点", "gold": "应比较资源隔离、一致性与维护成本。"},
            {"query": "混合检索（向量+关键词）在 Agent 中的架构位置", "gold": "应说明召回融合与后续重排的衔接。"},
            {"query": "多租户 Agent 平台的隔离架构最佳实践", "gold": "应覆盖数据隔离、模型配额、审计与计费。"},
        ],
        "evaluation": [
            {"query": "Agent 检索评估为什么要同时看 P@k、R@k、MRR、nDCG？", "gold": "应解释每个指标反映的排序与覆盖特性。"},
            {"query": "如何判断 HyDE 变好是偶然还是稳定提升？", "gold": "应包含多折评估与置信区间方法。"},
            {"query": "评估里如何区分检索不足和生成幻觉？", "gold": "应给出证据链检查与claim-level核验方法。"},
            {"query": "URL级金标与chunk级金标各自偏差是什么？", "gold": "应比较粒度偏差与评估目的的匹配关系。"},
            {"query": "增强器回退（fallback）为什么要单独统计？", "gold": "应说明回退混算会掩盖增强真实效果。"},
            {"query": "如何设计公平的 gold_pool 标注流程？", "gold": "应强调合并候选池避免 baseline 天然占优。"},
            {"query": "A/B 评测中 query 难度分层的必要性", "gold": "应说明不同难度样本对结论稳定性的影响。"},
            {"query": "如何评估多轮 Agent 的任务完成质量？", "gold": "应包含任务成功率、证据对齐和过程可解释性。"},
            {"query": "评估报告中为什么要包含失败案例分析？", "gold": "应说明平均分无法暴露系统性失效模式。"},
            {"query": "生产评估如何连接离线指标与线上业务指标？", "gold": "应给出离线-线上映射与监控策略。"},
        ],
        "engineering": [
            {"query": "Agent 生产系统如何做超时与重试策略设计？", "gold": "应给出分层超时、幂等与退避机制。"},
            {"query": "如何降低 MQE 带来的时延与成本？", "gold": "应包含变体上限、并发控制与缓存策略。"},
            {"query": "HyDE 在高并发下的资源治理策略有哪些？", "gold": "应说明限流、队列和降级策略。"},
            {"query": "向量库扩容后如何保证评估可复现？", "gold": "应要求固化版本、快照与配置记录。"},
            {"query": "Agent API 如何做请求幂等和结果缓存？", "gold": "应说明幂等键、过期策略与一致性约束。"},
            {"query": "文件上传到记忆库的工程风险有哪些？", "gold": "应覆盖格式解析失败、编码异常和恶意输入。"},
            {"query": "异步搜索聚合器如何实现健康回退？", "gold": "应描述多源重试、降级与监控。"},
            {"query": "如何建立检索质量的 nightly regression？", "gold": "应给出定时评测、阈值告警与回滚门禁。"},
            {"query": "Agent 服务容器化部署常见失败点与排查路径", "gold": "应覆盖依赖缺失、网络超时和环境变量配置。"},
            {"query": "LangSmith tracing 成本控制的工程手段", "gold": "应说明采样策略与字段裁剪方法。"},
        ],
        "risk_safety": [
            {"query": "Agent 系统中的 prompt injection 主要攻击面有哪些？", "gold": "应覆盖外部文档污染、工具调用注入和越权指令。"},
            {"query": "如何防止工具调用越权与敏感操作误触发？", "gold": "应给出权限模型、审批门与审计机制。"},
            {"query": "模型幻觉在高风险业务中的治理策略", "gold": "应包含证据门控、人工复核和拒答机制。"},
            {"query": "多智能体协作中的责任归因如何实现？", "gold": "应说明可追踪链路与操作日志。"},
            {"query": "安全评测中 red teaming 应覆盖哪些场景？", "gold": "应覆盖越狱、数据泄露、越权调用等。"},
            {"query": "Agent 的隐私合规风险与数据最小化策略", "gold": "应说明PII识别、脱敏和最小保留。"},
            {"query": "模型更新后如何做安全回归评测？", "gold": "应给出固定测试集与风险等级判定。"},
            {"query": "RAG 外部来源可信度如何分级治理？", "gold": "应包含来源白名单与质量评分。"},
            {"query": "在线系统如何监控异常行为并自动熔断？", "gold": "应覆盖异常阈值、报警和自动降级。"},
            {"query": "如何处理“看起来合理但证据不一致”的回答？", "gold": "应给出证据优先与冲突显式披露策略。"},
        ],
        "cases": [
            {"query": "客服场景 Agent 与 FAQ-RAG 的效果差异如何评估？", "gold": "应比较任务完成率、时延、满意度与成本。"},
            {"query": "代码助理场景中 Agent 如何结合仓库搜索工具？", "gold": "应给出检索、计划、修复与验证闭环。"},
            {"query": "财务分析 Agent 的证据可追溯链路如何设计？", "gold": "应包含来源引用、版本记录和审计。"},
            {"query": "教育辅导 Agent 如何平衡个性化与准确性？", "gold": "应给出评测维度与风险控制。"},
            {"query": "医疗问答中 Agent 系统的上线门槛是什么？", "gold": "应强调高风险审查与人工兜底。"},
            {"query": "制造业知识库场景下 RAG 索引策略实践", "gold": "应说明文档类型差异与检索配置。"},
            {"query": "法律检索 Agent 如何控制误引用风险？", "gold": "应强调法条版本、出处核验与拒答。"},
            {"query": "电商运营 Agent 的实验设计方法", "gold": "应包含A/B、转化率、成本与稳定性指标。"},
            {"query": "企业内网知识助手如何处理权限分级文档？", "gold": "应给出检索时权限过滤与审计策略。"},
            {"query": "多语言客服 Agent 的检索与评估挑战", "gold": "应覆盖跨语种检索与翻译误差影响。"},
        ],
        "timeliness": [
            {"query": "2025 年后函数调用规范有哪些关键变化？", "gold": "应列出结构化输出与工具接口演进。"},
            {"query": "近一年开源 Agent 框架的活跃趋势如何？", "gold": "应给出版本更新频率与社区指标。"},
            {"query": "最近半年 RAG 评测基准有哪些新方向？", "gold": "应概述新任务类型和评价维度。"},
            {"query": "2026 年初主流模型在长上下文能力的差异", "gold": "应比较上下文窗口、稳定性与成本。"},
            {"query": "近期关于 Agent 可靠性的代表论文结论", "gold": "应给出可复现实验结论与局限。"},
            {"query": "最新产业报告对 AI Agent ROI 的判断", "gold": "应包含场景边界、收益与失败条件。"},
            {"query": "过去一年 tool-use benchmark 的变化趋势", "gold": "应总结指标变化和测试集偏差。"},
            {"query": "新发布模型在检索增强任务上的公开表现", "gold": "应引用官方或第三方可验证评测。"},
            {"query": "近期企业对 Agent 治理框架的共识是什么？", "gold": "应概述审计、合规、安全三层框架。"},
            {"query": "当前对“全自动Agent”的主流谨慎观点有哪些？", "gold": "应说明可靠性和责任边界争议。"},
        ],
        "comparison": [
            {"query": "HyDE 与 MQE 在 precision/recall 上的典型取舍", "gold": "应比较不同任务下的优势和副作用。"},
            {"query": "向量检索 vs 混合检索在 Agent 场景的表现对比", "gold": "应包含召回、精度、延迟和维护成本。"},
            {"query": "Cross-encoder 重排与轻量重排方法的工程权衡", "gold": "应比较效果收益与推理成本。"},
            {"query": "单代理 vs 多代理在复杂任务中的收益边界", "gold": "应给出何时收益转负的条件。"},
            {"query": "规则路由 vs LLM 路由在任务分发中的稳定性对比", "gold": "应对比可解释性与泛化能力。"},
            {"query": "静态提示词 vs 动态规划提示词的效果差异", "gold": "应说明任务类型依赖与风险。"},
            {"query": "上下文压缩策略 A/B 对回答忠实度影响", "gold": "应包含压缩率与信息损失权衡。"},
            {"query": "检索 top-k=5 与 top-k=20 对最终回答质量影响", "gold": "应比较召回提升与噪声引入。"},
            {"query": "离线评估高分但线上效果一般的常见原因", "gold": "应说明分布偏移与奖励错配。"},
            {"query": "不同 embedding 模型对同一 query 集的敏感性", "gold": "应给出稳定性和成本对比分析。"},
        ],
    }


def _difficulty_for_index(i: int) -> str:
    # 80 条时：24 easy, 40 medium, 16 hard
    if i < 24:
        return "easy"
    if i < 64:
        return "medium"
    return "hard"


def build_queries(target_count: int = 80) -> List[Dict[str, str]]:
    buckets = _bucket_queries()
    rows: List[Dict[str, str]] = []
    idx = 1
    for bucket, items in buckets.items():
        for item in items:
            rows.append(
                {
                    "id": f"ai_q{idx:03d}",
                    "query": item["query"],
                    "relevant_ids": [],
                    "gold_answer": item["gold"],
                    "domain": "ai_news_paper",
                    "topic_bucket": bucket,
                    "difficulty": _difficulty_for_index(idx - 1),
                }
            )
            idx += 1
    # 按需求截断（默认 80）
    return rows[: max(1, min(target_count, len(rows)))]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_lines(path: Path, lines: List[str], limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = lines[: max(1, min(limit, len(lines)))]
    path.write_text("\n".join(chosen) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-out", default="./data/rag_eval_ai_queries_80.gold.jsonl")
    ap.add_argument("--url-out", default="./data/rag_eval_ai_urls_expanded.txt")
    ap.add_argument("--query-count", type=int, default=80)
    ap.add_argument("--url-count", type=int, default=140)
    ap.add_argument("--url-profile", choices=["mixed", "arxiv-heavy"], default="mixed")
    args = ap.parse_args()

    qrows = build_queries(target_count=args.query_count)
    urls = _expanded_url_pool(profile=args.url_profile)

    qpath = Path(args.query_out)
    upath = Path(args.url_out)
    write_jsonl(qpath, qrows)
    write_lines(upath, urls, limit=args.url_count)

    print(
        f"[assets] queries={len(qrows)} -> {qpath}\n"
        f"[assets] urls={min(len(urls), args.url_count)} -> {upath}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

