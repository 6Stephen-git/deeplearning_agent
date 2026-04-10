from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


TaskType = Literal["research", "eval"]
TaskStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "TIMEOUT"]


class ResearchTaskRequest(BaseModel):
    research_topic: str = Field(..., min_length=2, description="研究主题")
    max_cycles: int = Field(3, ge=1, le=8)
    use_cache: bool = True
    force_refresh: bool = False
    

class EvalTaskRequest(BaseModel):
    mode: Literal["retrieval", "langsmith"] = "retrieval"
    k: int = Field(5, ge=1, le=20)
    research_topic: str = "rag_eval"
    eval_modes: str = Field("baseline,hyde,mqe", description="传给 RAG_EVAL_MODES")
    use_cache: bool = True
    force_refresh: bool = False


class TaskSubmitResponse(BaseModel):
    task_id: Optional[str] = None
    task_type: TaskType
    status: TaskStatus
    cached: bool = False
    result: Optional[Dict[str, Any]] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    task_type: Optional[TaskType] = None
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    redis: Literal["ok", "down"]
    celery_broker: str
    celery_backend: str


class MemoryUploadResponse(BaseModel):
    success: bool
    file_name: str
    research_topic: str
    db_suffix: str
    persist_directory: str
    chunks_created: int = 0
    stored_count: int = 0
    message: Optional[str] = None


class TopicRegistryItem(BaseModel):
    research_topic: str
    db_suffix: str
    persist_directory: str
    updated_at: str
