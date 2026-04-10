from __future__ import annotations

import asyncio
import importlib
import os
from unittest.mock import patch

from src.backend import api as api_module
from src.nodes import execution_node as execution_module
from src.tools.async_search_tool import AsyncSearchTool


def test_submit_task_idempotency_uses_atomic_claim():
    calls = {"delay": 0}

    class _R:
        id = "task-123"

    with patch.object(api_module.redis_store, "set_text_if_absent", side_effect=[True, False]), patch.object(
        api_module.redis_store, "set_text"
    ), patch.object(api_module.redis_store, "set_json"), patch.object(
        api_module.redis_store, "get_json", return_value=None
    ), patch.object(
        api_module.redis_store, "get_text", return_value="pending:xyz"
    ), patch.object(
        api_module.run_research_task, "delay", side_effect=lambda payload: calls.__setitem__("delay", calls["delay"] + 1) or _R()
    ):
        first = api_module._submit_task("research", {"research_topic": "x", "max_cycles": 1}, False, False)
        second = api_module._submit_task("research", {"research_topic": "x", "max_cycles": 1}, False, False)

    assert first.task_id == "task-123"
    assert first.status == "PENDING"
    assert second.status == "PENDING"
    assert calls["delay"] == 1


def test_execute_single_task_timeout_wrapper():
    async def _slow(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return {"task_id": 1}

    with patch.object(execution_module, "_execute_single_task", side_effect=_slow):
        result = asyncio.run(
            execution_module._execute_single_task_with_timeout(
                {"id": 1, "title": "t"},
                1,
                0.01,
            )
        )
    assert result["timeout"] is True
    assert result["error_type"] == "timeout"


def test_env_loader_keeps_process_env_priority(monkeypatch):
    import src.config.env as envmod

    monkeypatch.setenv("SEARCH_BACKEND", "tavily")
    envmod._ENV_LOADED = False
    envmod.ensure_project_env_loaded()
    assert os.getenv("SEARCH_BACKEND") == "tavily"

    # 回滚状态，避免污染其他用例
    envmod._ENV_LOADED = False
    importlib.reload(envmod)


def test_search_logs_are_redacted_by_default(monkeypatch, caplog):
    monkeypatch.setenv("SEARCH_BACKEND", "serper")
    monkeypatch.setenv("SERPER_API_KEY", "dummy")
    monkeypatch.setenv("SEARCH_LOG_QUERIES", "false")

    tool = AsyncSearchTool()

    async def _fake_post(_payload, extra_headers=None):
        return {"organic": []}

    with patch.object(tool, "_post_json_with_retry", side_effect=_fake_post):
        with caplog.at_level("INFO"):
            asyncio.run(tool._search_with_serper("very sensitive query"))

    log_text = "\n".join([r.message for r in caplog.records])
    assert "very sensitive query" not in log_text
    assert "\"q\":\"***\"" in log_text
