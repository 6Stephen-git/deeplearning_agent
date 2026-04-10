# deeplearning_agent

## 环境变量与密钥

- `.env` 建议只放**非敏感默认配置**。
- 真实密钥请放在 `.env.local`（本机私有，不共享）或系统环境变量中。
- 变量优先级：**系统环境变量 > `.env.local` > `.env`**。
- 可复制 `.env.example` 作为本地模板。

## 联网搜索（执行节点）

默认使用 **Serper**（`https://serper.dev`，返回 Google 有机结果 JSON）。在项目根 `.env` 中设置：

- `SEARCH_BACKEND=serper`（默认）
- `SERPER_API_KEY=<你的 Key>`

仍可使用 **Tavily**：`SEARCH_BACKEND=tavily` 并设置 `TAVILY_API_KEY`。

**注意**：`.env` 里**不要重复写同一变量**（如两行 `SERPER_API_KEY`）；python-dotenv 通常以**最后一次为准**，空行会覆盖前面的密钥。

若出现 **`Server disconnected`**（网络不稳、跨境延迟）：可加大 `SEARCH_HTTP_TIMEOUT_S`（默认 60）与 `SEARCH_HTTP_RETRIES`（默认 3），例如 `SEARCH_HTTP_TIMEOUT_S=90`。

## LangSmith（RAG 效果测试 / 追踪）

1. 在 `.env` 中设置 `LANGSMITH_API_KEY`（或 `LANGCHAIN_API_KEY`），可选 `LANGSMITH_PROJECT=rag-eval`。
2. 安装：`pip install langsmith`（已列入 `requirements.txt`）。
3. 上传 RAG 检索+回答到 LangSmith 供在线评测：

   `python -m src.evaluator.rag_eval_runner eval_langsmith`

4. 深度研究脚本可选开启追踪：设 `LANGSMITH_TRACE_RESEARCH=true` 后运行 `python run_research.py ...`。

详见 [`docs/langsmith.md`](docs/langsmith.md)。

## 后端服务化（FastAPI + Redis + Celery）

本仓库已提供后端 MVP：

- API 服务：`python -m src.backend.main`
- Celery Worker：`celery -A src.backend.celery_app.celery_app worker -l info --concurrency=2`
- Redis：默认 `redis://localhost:6379`

推荐一键启动（Docker）：

```bash
docker compose up --build
```

核心接口：

- `GET /health`
- `POST /tasks/research`（提交研究任务）
- `POST /tasks/eval`（提交评估任务）
- `GET /tasks/{task_id}`（查询任务状态/结果）

说明：

- 支持任务幂等与结果缓存（Redis，TTL 可配）。
- 可用 `BACKEND_API_KEY` 启用简单 API Key 鉴权（请求头：`X-API-Key`）。
- 压测脚本：`python scripts/load_test_tasks.py --url http://127.0.0.1:8000 --n 20 --c 5`

## 开发环境：`import src` 找不到？

直接运行子路径下的脚本（如 `python src/evaluator/rag_eval_runner.py`）时，若未把**项目根目录**加入 Python 路径，会出现 `ModuleNotFoundError: No module named 'src'`。

**推荐（任选其一）：**

1. **可编辑安装（推荐，一次配置全局生效）**  
   在项目根目录执行：
   ```bash
   .\venv\Scripts\pip install -e .
   ```
   之后从任意工作目录运行脚本，只要用的是该 venv，`import src` 均可用。

2. **模块方式运行（无需安装）**  
   ```bash
   cd <项目根目录>
   python -m src.evaluator.rag_eval_runner demo
   ```

3. **VS Code / Cursor**  
   已提供 `.vscode/settings.json`，为终端注入 `PYTHONPATH=项目根`（需用工作区打开项目根）。

## RAG 离线评估卡住？

`eval_retrieval` 默认对每条 query 跑 **baseline + HyDE + MQE**（后两者各需 LLM），10 条可能要 **数十分钟**，且首条就会等较久。

- 运行中会打印 **`[RAG-EVAL] 进度 i/N`**；若长时间无输出，多半是首条在等待 LLM。
- **只测向量、快速冒烟**（秒级）：
  ```bat
  set RAG_EVAL_MODES=baseline
  python -m src.evaluator.rag_eval_runner eval_retrieval
  ```
  PowerShell：`$env:RAG_EVAL_MODES="baseline"; python -m src.evaluator.rag_eval_runner eval_retrieval`

### HyDE / MQE 总超时、回退 baseline？

- 默认已将 **HTTP 读超时** 与 **asyncio 外层等待** 调高（`ENHANCER_QWEN_TIMEOUT` 默认 **300s**；外层默认约 **max(480, HTTP+120)**）。
- 若仍 `Request timed out`，可在 `.env` 或终端继续加大（**外层须大于 HTTP**），例如：

```bat
set ENHANCER_QWEN_TIMEOUT=600
set RAG_EVAL_ENHANCE_TIMEOUT_S=900
```

仅评估链路单独加大 HTTP：设 `RAG_EVAL_LLM_HTTP_TIMEOUT_S=600`。跑 `eval_retrieval` 时会打印当前生效的秒数。

### 离线评估 P/R/MRR 全是 0？

多为 **`data/rag_eval_queries.jsonl` 里的 `relevant_ids` 与当前向量库不一致**（重建 `build_index` 后 chunk UUID 会变，旧 id 全部失效）。运行 `eval_retrieval` 时会自动校验并打印警告。

**一键对齐当前库（弱监督口径，会备份原文件为 `rag_eval_queries.jsonl.bak`）：**

```bat
python -m src.evaluator.rag_eval_runner refresh_eval_ids_union
```

说明：
- 已彻底移除 `refresh_eval_ids`（过去容易被误解为 baseline Top‑k 回写，导致评测结论偏置）。
- `refresh_eval_ids_union` 会回写为 **baseline/HyDE/MQE 三路候选 Top‑k 的去重并集**（k 由 `RAG_EVAL_GOLD_CANDIDATES_K` 控制，默认 5），适合做难例分析与快速迭代。
- 若要严肃对比，请使用 **gold_pool + 人工标注** 或 URL/文档级金标（见下文）。

`data/rag_eval_queries.jsonl` 的路径**固定相对项目根目录**解析（与当前终端在哪个文件夹无关）。

评估输出除 **P@k / R@k / MRR** 外还有 **nDCG@k**（二元相关：命中金标 chunk 或 URL 则该 rank 上 rel=1），用于看排序质量；若金标为 baseline Top-k，nDCG 同样会偏向 baseline。

手动标注仍可用：`search_chunks "关键词"` 查当前 id 后写入 jsonl。

**人工金标注完整流程（步骤、规范、质检、避坑）**：见 [`docs/rag_manual_gold_labeling.md`](docs/rag_manual_gold_labeling.md)。

**开始一轮人工金标**：在仓库根目录执行  
`python -m src.evaluator.rag_eval_runner gold_candidates`  
查看每条 query 的候选 chunk 与当前 `relevant_ids`，再编辑 jsonl 并跑 `eval_retrieval`。

**不要只从 baseline 的 Top-k 里勾选金标**（会低估 HyDE/MQE 等召回改进）。更宽候选池：`python -m src.evaluator.rag_eval_runner gold_pool`（baseline+HyDE+MQE 去重合并，需 LLM）。详见 [`docs/rag_manual_gold_labeling.md`](docs/rag_manual_gold_labeling.md)「方法学」。

`gold_candidates` / `search_chunks` 里的 **preview 默认只显示前 220 字**（可调 `RAG_EVAL_GOLD_PREVIEW_CHARS`），**完整正文在向量库里**。**url** 若显示 N/A：多为旧索引未写入每块 URL；从带 `SourceURL:` 的清洗文本重建索引后，同一文件的 chunk 会带上来源 URL（见 `docs/rag_manual_gold_labeling.md` 说明）。

若预览里仍有 **「复制页面」、HTML 碎片** 等：语料来自网页抓取，旧版清洗较弱；已在 `build_index` 中加强去导航/去 UI 文案。**需重新 `python -m src.evaluator.rag_eval_runner build_index`** 后 chunk 才会变干净。

金标时：**`gold_candidates` 会打印 jsonl 里的 `gold_answer` 作题意对照**；看某 chunk 全文用 `python -m src.evaluator.rag_eval_runner dump_chunk <uuid>`。检索 Top-k 只是候选，噪声进上下文会干扰 LLM——产品上需 **重排 / 减小 k / 过滤**，与「检索评估」测的是两回事。
