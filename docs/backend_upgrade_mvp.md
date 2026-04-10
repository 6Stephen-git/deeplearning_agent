# 后端升级 MVP 说明（5天版）

## 架构

```mermaid
flowchart LR
client[Client] --> api[FastAPI]
api --> redis[(Redis)]
api --> worker[CeleryWorker]
worker --> graph[LangGraphApp]
worker --> eval[RagEvalRunner]
api --> langsmith[LangSmith]
worker --> langsmith
```

## 启动

### Docker

```bash
docker compose up --build
```

### 本地

1. 启动 Redis（默认 `localhost:6379`）。
2. API：

```bash
python -m src.backend.main
```

3. Worker：

```bash
celery -A src.backend.celery_app.celery_app worker -l info --concurrency=2
```

## API 示例

提交研究任务：

```bash
curl -X POST http://127.0.0.1:8000/tasks/research \
  -H "Content-Type: application/json" \
  -d "{\"research_topic\":\"RAG工程化实践\",\"max_cycles\":2}"
```

查询任务状态：

```bash
curl http://127.0.0.1:8000/tasks/<task_id>
```

提交评估任务：

```bash
curl -X POST http://127.0.0.1:8000/tasks/eval \
  -H "Content-Type: application/json" \
  -d "{\"mode\":\"retrieval\",\"k\":5,\"eval_modes\":\"baseline,hyde,mqe\"}"
```

## 可观测与可靠性

- 结构化日志字段：`request_id`、`path`、`method`、`duration_ms`。
- 任务状态映射：`PENDING/RUNNING/SUCCEEDED/FAILED/TIMEOUT`。
- Celery 任务启用超时与重试退避。
- LangSmith 可通过环境变量打开（见 `docs/langsmith.md`）。

## 压测

```bash
python scripts/load_test_tasks.py --url http://127.0.0.1:8000 --n 50 --c 10
```

记录并留存以下指标作为投递素材：

- 平均响应时间（提交接口）。
- p95 延迟。
- 失败率与重试后成功率。
- 缓存命中后延迟下降幅度。
