from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict

from src.backend.celery_app import celery_app
from src.backend.service import (
    run_eval_sync,
    run_research_sync,
    build_effective_config_snapshot,
    write_run_config_snapshot,
)
from src.backend.settings import get_settings

settings = get_settings()
logger = logging.getLogger("backend.tasks")

def _as_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_filename(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        return "report"
    s = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s or "report"


def _maybe_save_report(task_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    在 celery worker 侧自动落盘报告，避免 docker logs 截断导致看不到完整内容。

    由环境变量控制：
    - BACKEND_SAVE_REPORTS: true/false（默认 true）
    - BACKEND_REPORT_DIR: 保存目录（默认 ./reports）
    """
    if not _as_bool("BACKEND_SAVE_REPORTS", True):
        return result
    final_report = result.get("final_report")
    if not isinstance(final_report, str) or not final_report.strip():
        return result

    out_dir = Path(os.getenv("BACKEND_REPORT_DIR", "./reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    topic = _safe_filename(str(result.get("research_topic") or "research"))
    path = out_dir / f"{task_id}__{topic}.md"
    try:
        path.write_text(final_report, encoding="utf-8")
        result["report_path"] = str(path)
        snapshot = result.get("effective_config") or build_effective_config_snapshot()
        result["run_config_path"] = write_run_config_snapshot(str(path), snapshot)
    except Exception:
        # 落盘失败不应导致任务失败；保持原结果
        return result
    return result


@celery_app.task(
    bind=True,
    name="backend.run_research",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    soft_time_limit=settings.task_default_timeout_s,
    time_limit=settings.task_default_timeout_s + 30,
)
def run_research_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = str(payload.get("request_id") or "-")
    logger.info(
        "task.research.start",
        extra={
            "request_id": request_id,
            "task_id": getattr(getattr(self, "request", None), "id", "-"),
            "topic": payload.get("research_topic", ""),
        },
    )
    result = run_research_sync(
        research_topic=payload["research_topic"],
        max_cycles=int(payload["max_cycles"]),
    )
    task_id = getattr(getattr(self, "request", None), "id", None) or ""
    if task_id:
        result = _maybe_save_report(task_id, result)
    logger.info(
        "task.research.end",
        extra={
            "request_id": request_id,
            "task_id": task_id or "-",
            "topic": payload.get("research_topic", ""),
        },
    )
    return result


@celery_app.task(
    bind=True,
    name="backend.run_eval",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 1},
    soft_time_limit=settings.task_default_timeout_s,
    time_limit=settings.task_default_timeout_s + 30,
)
def run_eval_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = str(payload.get("request_id") or "-")
    task_id = getattr(getattr(self, "request", None), "id", "-")
    logger.info(
        "task.eval.start",
        extra={"request_id": request_id, "task_id": task_id, "topic": payload.get("research_topic", "")},
    )
    result = run_eval_sync(
        mode=payload["mode"],
        k=int(payload["k"]),
        research_topic=payload["research_topic"],
        eval_modes=payload["eval_modes"],
    )
    logger.info(
        "task.eval.end",
        extra={"request_id": request_id, "task_id": task_id, "topic": payload.get("research_topic", "")},
    )
    return result
