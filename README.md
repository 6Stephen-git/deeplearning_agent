# DeepLearning Agent

基于 **LangGraph** 的深度研究型智能体：自动拆解主题、并发联网检索、结合向量记忆与可选文档上传，在多轮反思后生成结构化研究报告。仓库同时提供 **RAG 离线评测链路** 与 **FastAPI + Redis + Celery** 的任务化后端，适合从本地实验平滑过渡到服务部署。

---

## 项目能做什么

| 方向 | 说明 |
|------|------|
| **深度研究** | 规划 → 异步执行子任务（搜索 + 总结）→ 聚合 → 按需多轮深入 → 报告，适合调研、竞品分析、技术扫盲等长链路任务。 |
| **记忆与 RAG** | ChromaDB 向量库 + 通义等嵌入；支持 HyDE / MQE 等查询增强；可上传 PDF 等文档参与检索。 |
| **评测与可观测** | 检索指标（P@k、R@k、MRR、nDCG 等）、可选 **LangSmith** 追踪；便于迭代检索策略与提示词。 |
| **工程化交付** | 异步任务 API、幂等与结果缓存、Docker 一键拉起 API + Worker + Redis。 |

---

## 核心特点

- **图式工作流**：状态在节点间显式流转，规划、执行、聚合、报告职责清晰，易于扩展新节点或策略。
- **高并发 I/O**：子任务级异步执行，适合多查询并行检索与总结。
- **搜索可插拔**：默认 **Serper**（Google 有机结果 JSON），可切换 **Tavily**；超时与重试可配置，适配不稳定网络。
- **LLM 与嵌入**：默认对接阿里云 **DashScope** 兼容 OpenAI API（如 Qwen）；嵌入与模型名通过环境变量配置。
- **RAG 评测工具链**：从建索引、`eval_retrieval`、金标候选生成到与 LangSmith 联动，覆盖实验到对比分析。
- **后端 MVP**：`POST` 提交研究/评估任务，`GET` 轮询状态；可选 `X-API-Key` 鉴权；附压测脚本。

---

## 技术栈

Python 3.10+ · LangGraph · LangChain · FastAPI · Celery · Redis · ChromaDB · sentence-transformers（及可选 DashScope 嵌入）等。

---

## 快速开始

### 1. 环境

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

复制环境变量模板并按说明填写密钥（详见下方「配置说明」）：

```bash
copy .env.example .env
# 编辑 .env；敏感项建议使用 .env.local 或系统环境变量
```

变量优先级：**系统环境变量 > `.env.local` > `.env`**。

### 2. 命令行跑一轮深度研究

```bash
python run_research.py "你的研究主题"
# 可选：--upload 某.pdf 或目录，将文档纳入向量检索
```

### 3. Docker 启动 API + Worker + Redis

```bash
docker compose up --build
```

默认暴露 **API `http://127.0.0.1:8000`**。核心接口示例：

- `GET /health`
- `POST /tasks/research` — 提交研究任务  
- `POST /tasks/eval` — 提交评估任务  
- `GET /tasks/{task_id}` — 查询状态与结果  

---

## 配置说明（摘要）

| 类别 | 要点 |
|------|------|
| **LLM** | `DASHSCOPE_API_KEY`、`QWEN_*`、`REPORT_*` 等（见 `.env.example`）。 |
| **搜索** | `SEARCH_BACKEND=serper` 或 `tavily`；`SERPER_API_KEY` / `TAVILY_API_KEY`。勿在 `.env` 中重复同一键（后者会覆盖前者）。 |
| **嵌入** | `EMBEDDING_PROVIDER`、`DASHSCOPE_EMBEDDING_MODEL` 等。 |
| **检索增强** | `ENABLE_HYDE`、`ENABLE_MQE` 等。 |
| **后端** | `REDIS_URL`、`CELERY_*`、`BACKEND_API_KEY`（可选）、报告目录与 TTL 等。 |
| **LangSmith** | `LANGSMITH_API_KEY`、`LANGSMITH_PROJECT`；深度研究可选 `LANGSMITH_TRACE_RESEARCH=true`。 |

联网搜索若出现 **`Server disconnected`**，可适当增大 `SEARCH_HTTP_TIMEOUT_S`、`SEARCH_HTTP_RETRIES`。

---

## 常用开发命令

```bash
# 模块方式运行（避免 PYTHONPATH 问题）
python -m src.evaluator.rag_eval_runner demo

# RAG：上传 LangSmith 评测（需配置 LangSmith）
python -m src.evaluator.rag_eval_runner eval_langsmith

# 后端本地（需本机 Redis）
python -m src.backend.main
celery -A src.backend.celery_app.celery_app worker -l info --concurrency=2

# 压测（示例）
python scripts/load_test_tasks.py --url http://127.0.0.1:8000 --n 20 --c 5
```

若直接运行子路径脚本报 `No module named 'src'`，请使用 **`pip install -e .`** 或 **`python -m ...`**，或在本机配置 `PYTHONPATH` 指向项目根（仓库内可提供 VS Code/Cursor 工作区设置）。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/langsmith.md](docs/langsmith.md) | LangSmith 与 RAG 评测上传 |
| [docs/rag_manual_gold_labeling.md](docs/rag_manual_gold_labeling.md) | 人工金标、候选池、评测口径与避坑 |
| [docs/backend_upgrade_mvp.md](docs/backend_upgrade_mvp.md) | 后端 MVP 相关说明 |

RAG 离线评估较慢时，可设 `RAG_EVAL_MODES=baseline` 做快速冒烟；金标 ID 与向量库不一致时见评测脚本说明与 `refresh_eval_ids_union` 等命令（详见 `docs/rag_manual_gold_labeling.md`）。

---

## 仓库结构（节选）

```
src/
  graph.py              # LangGraph 主流程
  state.py              # 图状态
  nodes/                # 规划、执行、聚合、报告等节点
  memory/               # 向量记忆、嵌入、查询增强
  tools/                # LLM、异步搜索等
  backend/              # FastAPI、Celery、Redis 存储
  evaluator/            # RAG 评测与索引构建入口
run_research.py         # 命令行深度研究入口
scripts/                # 压测、数据与标注辅助脚本
```

---

## 测试

```bash
pytest
```

---

## 致谢

构建于 LangGraph、LangChain 与开源向量库生态；搜索与模型服务依赖第三方 API，使用前请遵守相应服务条款与配额。

若本项目对你有帮助，欢迎 Star 与 Issue/PR。
