from __future__ import annotations

import logging
import json
import tempfile
import time
import uuid
from pathlib import Path
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.backend.celery_app import celery_app
from src.backend.logging_utils import configure_logging
from src.backend.redis_store import RedisStore
from src.backend.schemas import (
    EvalTaskRequest,
    HealthResponse,
    MemoryUploadResponse,
    ResearchTaskRequest,
    TaskStatusResponse,
    TaskSubmitResponse,
    TopicRegistryItem,
)
from src.backend.settings import get_settings
from src.backend.tasks import run_eval_task, run_research_task
from src.memory.file_processor import FileUploadProcessor
from src.memory.memory_store import MemoryStore
from src.memory.topic_registry import list_topics, register_topic, topic_to_db_path, topic_to_db_suffix
from src.tools.langsmith_env import configure_langsmith_tracing

settings = get_settings()
configure_logging()
logger = logging.getLogger("backend.api")
redis_store = RedisStore(settings.redis_url)
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

app = FastAPI(title=settings.api_title, version="0.1.0")


def _map_celery_status(status: str) -> str:
    if status == "SUCCESS":
        return "SUCCEEDED"
    if status == "STARTED":
        return "RUNNING"
    if status == "FAILURE":
        return "FAILED"
    if status == "REVOKED":
        return "TIMEOUT"
    return "PENDING"


def _extract_error(result: AsyncResult) -> Optional[str]:
    if result.state != "FAILURE":
        return None
    try:
        return str(result.result)
    except Exception:
        return "unknown failure"


def _celery_result_to_dict(async_result: AsyncResult) -> Optional[Dict[str, Any]]:
    """将 Celery 成功结果规范为 dict，供提交接口直接返回。"""
    if async_result.state != "SUCCESS":
        return None
    try:
        r = async_result.result
        if isinstance(r, dict):
            return r
        return {"result": r}
    except Exception:
        return None


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = settings.api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


@app.on_event("startup")
def _on_startup() -> None:
    if settings.enable_langsmith:
        configure_langsmith_tracing(
            project_name=settings.api_title.replace(" ", "-").lower(),
            force_tracing=True,
        )


@app.middleware("http")
async def _request_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request_id_ctx.set(rid)
    started = time.time()
    logger.info(
        "request.start",
        extra={
            "request_id": rid,
            "path": request.url.path,
            "method": request.method,
        },
    )
    response = await call_next(request)
    duration_ms = round((time.time() - started) * 1000, 2)
    response.headers["X-Request-ID"] = rid
    logger.info(
        "request.end",
        extra={
            "request_id": rid,
            "path": request.url.path,
            "method": request.method,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.exception_handler(Exception)
async def _handle_uncaught(_: Request, exc: Exception):
    logger.exception("request.error", extra={"request_id": request_id_ctx.get("-")})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health(_: None = Depends(_require_api_key)) -> HealthResponse:
    return HealthResponse(
        redis="ok" if redis_store.is_healthy() else "down",
        celery_broker=settings.celery_broker_url,
        celery_backend=settings.celery_result_backend,
    )


@app.get("/memory/topics", response_model=List[TopicRegistryItem])
def get_memory_topics(_: None = Depends(_require_api_key)) -> List[TopicRegistryItem]:
    return [TopicRegistryItem(**item) for item in list_topics()]


@app.post("/memory/upload", response_model=MemoryUploadResponse)
async def upload_memory_file(
    research_topic: str = Form(...),
    file: UploadFile = File(...),
    metadata_json: Optional[str] = Form(default=None),
    _: None = Depends(_require_api_key),
) -> MemoryUploadResponse:
    topic = (research_topic or "").strip()
    if len(topic) < 2:
        raise HTTPException(status_code=400, detail="research_topic too short")
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing file name")

    register_topic(topic)
    persist_dir = topic_to_db_path(topic)
    store = MemoryStore(persist_directory=persist_dir)
    processor = FileUploadProcessor(memory_store=store)

    extra_meta: Optional[Dict[str, Any]] = None
    if metadata_json:
        try:
            parsed = json.loads(metadata_json)
            if isinstance(parsed, dict):
                extra_meta = parsed
        except Exception:
            raise HTTPException(status_code=400, detail="invalid metadata_json (must be JSON object)")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        result = processor.process_uploaded_file(
            file_path=tmp_path,
            research_topic=topic,
            metadata=extra_meta,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return MemoryUploadResponse(
        success=bool(result.get("success", True)),
        file_name=str(file.filename),
        research_topic=topic,
        db_suffix=topic_to_db_suffix(topic),
        persist_directory=persist_dir,
        chunks_created=int(result.get("chunks_created", result.get("total_chunks", 0) or 0)),
        stored_count=int(result.get("stored_count", result.get("chunks_stored", 0) or 0)),
        message=result.get("message"),
    )


def _submit_task(
    task_type: str,
    payload: Dict[str, Any],
    force_refresh: bool,
    use_cache: bool,
) -> TaskSubmitResponse:
    fp = redis_store.payload_hash(task_type, payload, version="v1")
    cache_key = f"task:cache:{task_type}:{fp}"
    map_key = f"task:idem:{task_type}:{fp}"

    if use_cache and not force_refresh:
        cached = redis_store.get_json(cache_key)
        if cached is not None:
            return TaskSubmitResponse(
                task_type=task_type, status="SUCCEEDED", cached=True, result=cached
            )

    if not force_refresh:
        claim_token = f"pending:{uuid.uuid4().hex}"
        acquired = redis_store.set_text_if_absent(
            map_key,
            claim_token,
            settings.task_idempotency_ttl_s,
        )
        if not acquired:
            existing_task_id = redis_store.get_text(map_key)
            if existing_task_id:
                if existing_task_id.startswith("pending:"):
                    return TaskSubmitResponse(
                        task_type=task_type,
                        status="PENDING",
                        cached=False,
                        result=None,
                    )
                async_res = AsyncResult(existing_task_id, app=celery_app)
                mapped = _map_celery_status(async_res.state)
                done_result: Optional[Dict[str, Any]] = None
                if mapped == "SUCCEEDED":
                    done_result = _celery_result_to_dict(async_res)
                return TaskSubmitResponse(
                    task_id=existing_task_id,
                    task_type=task_type,
                    status=mapped,
                    cached=False,
                    result=done_result,
                )

        try:
            if task_type == "research":
                async_result = run_research_task.delay(payload)
            else:
                async_result = run_eval_task.delay(payload)
        except Exception:
            redis_store.delete_key(map_key)
            raise
        redis_store.set_text(map_key, async_result.id, settings.task_idempotency_ttl_s)
    else:
        if task_type == "research":
            async_result = run_research_task.delay(payload)
        else:
            async_result = run_eval_task.delay(payload)

    redis_store.set_json(
        f"task:meta:{async_result.id}",
        {"task_type": task_type, "fingerprint": fp},
        settings.task_idempotency_ttl_s,
    )
    return TaskSubmitResponse(
        task_id=async_result.id,
        task_type=task_type,
        status="PENDING",
        cached=False,
    )


@app.post("/tasks/research", response_model=TaskSubmitResponse)
def submit_research(
    req: ResearchTaskRequest,
    _: None = Depends(_require_api_key),
) -> TaskSubmitResponse:
    payload = {
        "research_topic": req.research_topic.strip(),
        "max_cycles": req.max_cycles,
        "request_id": request_id_ctx.get("-"),
    }
    return _submit_task(
        task_type="research",
        payload=payload,
        force_refresh=req.force_refresh,
        use_cache=req.use_cache,
    )


@app.post("/tasks/eval", response_model=TaskSubmitResponse)
def submit_eval(
    req: EvalTaskRequest,
    _: None = Depends(_require_api_key),
) -> TaskSubmitResponse:
    payload = {
        "mode": req.mode,
        "k": req.k,
        "research_topic": req.research_topic,
        "eval_modes": req.eval_modes,
        "request_id": request_id_ctx.get("-"),
    }
    return _submit_task(
        task_type="eval",
        payload=payload,
        force_refresh=req.force_refresh,
        use_cache=req.use_cache,
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(
    task_id: str,
    _: None = Depends(_require_api_key),
) -> TaskStatusResponse:
    task_id = task_id.strip()
    result = AsyncResult(task_id, app=celery_app)
    status = _map_celery_status(result.state)
    meta = redis_store.get_json(f"task:meta:{task_id}") or {}
    task_type = str(meta.get("task_type") or "")
    data: Optional[Dict[str, Any]] = None
    if status == "SUCCEEDED":
        data = result.result if isinstance(result.result, dict) else {"result": result.result}
        task_type = task_type or ("eval" if "eval_modes" in data else "research")
        fp = str(meta.get("fingerprint") or "")
        if fp:
            redis_store.set_json(
                f"task:cache:{task_type}:{fp}",
                data,
                settings.task_result_cache_ttl_s,
            )
    return TaskStatusResponse(
        task_id=task_id,
        task_type=task_type or None,
        status=status,
        result=data,
        error=_extract_error(result),
    )
