from __future__ import annotations

import os
from dataclasses import dataclass

from src.config import ensure_project_env_loaded

ensure_project_env_loaded()


def _as_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class BackendSettings:
    api_title: str = os.getenv("BACKEND_API_TITLE", "DeepLearning Agent API")
    api_host: str = os.getenv("BACKEND_API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("BACKEND_API_PORT", "8000"))
    api_key: str = os.getenv("BACKEND_API_KEY", "").strip()

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://localhost:6379/1"))

    task_default_timeout_s: int = int(os.getenv("TASK_DEFAULT_TIMEOUT_S", "1800"))
    task_idempotency_ttl_s: int = int(os.getenv("TASK_IDEMPOTENCY_TTL_S", "1800"))
    task_result_cache_ttl_s: int = int(os.getenv("TASK_RESULT_CACHE_TTL_S", "1800"))

    default_max_cycles: int = int(os.getenv("BACKEND_DEFAULT_MAX_CYCLES", "3"))
    enable_langsmith: bool = _as_bool("BACKEND_ENABLE_LANGSMITH", True)


def get_settings() -> BackendSettings:
    return BackendSettings()
