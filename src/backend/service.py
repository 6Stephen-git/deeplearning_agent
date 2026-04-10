from __future__ import annotations

import asyncio
import os
import time
import json
from pathlib import Path
from typing import Any, Dict

from src.graph import app
from src.state import GraphState
from src.tools.langsmith_env import try_enable_langsmith_for_research

_SENSITIVE_ENV_KEYS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
)


def _is_sensitive_key(name: str) -> bool:
    up = (name or "").upper()
    return any(mark in up for mark in _SENSITIVE_ENV_KEYS)


def _masked_value(name: str, value: str) -> str:
    if not value:
        return ""
    if not _is_sensitive_key(name):
        return value
    if len(value) <= 6:
        return "***"
    return value[:3] + "***" + value[-3:]


def build_effective_config_snapshot() -> Dict[str, Any]:
    keys = [
        "QWEN_MODEL_NAME",
        "QWEN_TIMEOUT",
        "QWEN_MAX_RETRIES",
        "REPORT_QWEN_TIMEOUT",
        "REPORT_QWEN_MAX_RETRIES",
        "SEARCH_BACKEND",
        "SEARCH_HTTP_TIMEOUT_S",
        "SEARCH_HTTP_RETRIES",
        "SEARCH_LOG_QUERIES",
        "EMBEDDING_PROVIDER",
        "DASHSCOPE_EMBEDDING_MODEL",
        "ENABLE_HYDE",
        "ENABLE_MQE",
        "MQE_NUM_VARIANTS",
        "HYDE_ANSWER_LENGTH",
        "MAX_RESEARCH_CYCLES",
        "EXECUTION_TASK_TIMEOUT_S",
        "SHOW_LONGTERM_MEMORY_WRITES",
        "LANGSMITH_TRACE_RESEARCH",
    ]
    env = {}
    for key in keys:
        raw = os.getenv(key, "")
        env[key] = _masked_value(key, str(raw))
    return {"env": env}


def write_run_config_snapshot(report_path: str, snapshot: Dict[str, Any]) -> str:
    report = Path(report_path)
    out_path = report.with_name(report.stem + ".run_config.json")
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _build_initial_state(research_topic: str, max_cycles: int) -> GraphState:
    return {
        "research_topic": research_topic,
        "sub_tasks": [],
        "active_tasks": [],
        "task_results": [],
        "need_deeper_research": False,
        "current_cycle": 1,
        "max_cycles": max_cycles,
        "final_report": None,
        "messages": [],
        "working_memory": None,
        "task_quality_profiles": [],
        "deficiency_report": None,
        "targeted_instructions": [],
        "last_cycle_score": None,
        "report_allowed": True,
        "report_block_reason": None,
        "report_warning": None,
        "task_metric_scores": [],
        "evidence_pass_rate": 0.0,
    }


async def run_research_async(research_topic: str, max_cycles: int) -> Dict[str, Any]:
    t0 = time.time()
    # 研究工作流追踪：在 worker 进程内按环境变量显式启用
    try_enable_langsmith_for_research()
    state = _build_initial_state(research_topic=research_topic, max_cycles=max_cycles)
    final_state = await app.ainvoke(state)
    valid_results = len([r for r in final_state.get("task_results", []) if r is not None])
    effective_config = build_effective_config_snapshot()
    return {
        "research_topic": final_state.get("research_topic", research_topic),
        "current_cycle": final_state.get("current_cycle"),
        "sub_task_count": len(final_state.get("sub_tasks", [])),
        "valid_result_count": valid_results,
        "final_report": final_state.get("final_report"),
        "elapsed_s": round(time.time() - t0, 3),
        "effective_config": effective_config,
    }


def run_research_sync(research_topic: str, max_cycles: int) -> Dict[str, Any]:
    return asyncio.run(run_research_async(research_topic=research_topic, max_cycles=max_cycles))


async def run_eval_async(
    mode: str,
    k: int,
    research_topic: str,
    eval_modes: str,
) -> Dict[str, Any]:
    from src.evaluator.rag_eval_runner import (
        eval_retrieval,
        eval_with_langsmith,
        load_eval_queries,
    )

    t0 = time.time()
    old_modes = os.getenv("RAG_EVAL_MODES")
    os.environ["RAG_EVAL_MODES"] = eval_modes
    try:
        queries = load_eval_queries()
        eval_result = None
        if mode == "langsmith":
            await eval_with_langsmith(queries, k=k, research_topic=research_topic)
        else:
            eval_result = await eval_retrieval(queries, k=k, research_topic=research_topic)
    finally:
        if old_modes is None:
            os.environ.pop("RAG_EVAL_MODES", None)
        else:
            os.environ["RAG_EVAL_MODES"] = old_modes
    resp = {
        "mode": mode,
        "k": k,
        "research_topic": research_topic,
        "eval_modes": eval_modes,
        "query_count": len(queries),
        "elapsed_s": round(time.time() - t0, 3),
        "note": "retrieval 模式会返回结构化指标与报告路径；LangSmith 模式请到控制台查看 run。",
        "effective_config": build_effective_config_snapshot(),
    }
    if eval_result:
        resp["diag_stats"] = eval_result.get("diag_stats")
        resp["overall_stats"] = eval_result.get("overall_stats")
        resp["effective_stats"] = eval_result.get("effective_stats")
        resp["diag_summary"] = eval_result.get("diag_summary")
        resp["overall_summary"] = eval_result.get("overall_summary")
        resp["effective_summary"] = eval_result.get("effective_summary")
        resp["evaluation_scope_count"] = eval_result.get("evaluation_scope_count")
        resp["report_paths"] = eval_result.get("report_paths")
    return resp


def run_eval_sync(mode: str, k: int, research_topic: str, eval_modes: str) -> Dict[str, Any]:
    return asyncio.run(
        run_eval_async(
            mode=mode,
            k=k,
            research_topic=research_topic,
            eval_modes=eval_modes,
        )
    )
