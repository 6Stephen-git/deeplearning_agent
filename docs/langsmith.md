# LangSmith 效果测试与追踪

## 1. 环境变量

在仓库根目录 `.env` 中配置（与 [LangSmith 文档](https://docs.smith.langchain.com/) 一致）：

| 变量 | 说明 |
|------|------|
| `LANGSMITH_API_KEY` | 必填。可在 LangSmith → Settings → API Keys 创建。 |
| `LANGCHAIN_API_KEY` | 可选。若未设 `LANGSMITH_API_KEY`，可用此项作为兼容别名。 |
| `LANGSMITH_PROJECT` | 可选。RAG 评估默认 `rag-eval`；研究脚本默认 `research-agent`（见下）。 |
| `LANGSMITH_ENDPOINT` | 可选。自建 LangSmith 时填写 API 地址。 |

RAG 评估命令会自动设置 `LANGSMITH_TRACING_V2=true`（无需手写）。

## 2. RAG：把检索 + 回答写入 LangSmith

1. 安装依赖：`pip install langsmith`（已写入 `requirements.txt`）。
2. 配置好向量评估数据与索引（见 `rag_manual_gold_labeling.md`）。
3. 在项目根目录执行：

```bat
python -m src.evaluator.rag_eval_runner eval_langsmith
```

- 对 `data/rag_eval_queries.jsonl`（或 `RAG_EVAL_QUERIES_PATH`）中的每条 query，按 **`RAG_EVAL_MODES`**（默认 `baseline,hyde,mqe`）各跑一遍：检索 → 基于 chunk 生成答案。
- 在 LangSmith **Projects** 中打开 `LANGSMITH_PROJECT` 对应项目，即可看到 runs，便于配置 **在线评估器**（如 LLM-as-judge）、对比实验、筛选坏例。

可选：

- `RAG_EVAL_LANGSMITH_CHUNK_CHARS`：上报到 LangSmith 的每条 chunk 正文最大字符数（默认 2000），防止单条 trace 过大。

## 3. 深度研究图（可选追踪）

运行 `run_research.py` 时，若希望尽量把 LangGraph/LangChain 步骤同步到 LangSmith，可设置：

```env
LANGSMITH_TRACE_RESEARCH=true
LANGSMITH_PROJECT=research-agent
```

并配置 `LANGSMITH_API_KEY`。启动时会打印 `[LangSmith] 已启用研究流程追踪...`。

> 实际可见的 span 粒度取决于 LangGraph / LangChain 版本与是否使用带追踪的 Runnable；若看不到节点级 trace，以 RAG 的 `eval_langsmith` 为准即可。

## 4. 与本地 `eval_retrieval` 的区别

| 命令 | 作用 |
|------|------|
| `eval_retrieval` | 本地算 P@k、R@k、MRR、nDCG 等，**不上传** LangSmith。 |
| `eval_langsmith` | **不上报**上述标量；只把每次「检索 + 答案」作为 run **上传**，便于在云端做定性/在线评测。 |

二者可配合：本地指标看排序质量，LangSmith 看答案与引用是否 grounded。
