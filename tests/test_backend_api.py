from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.backend.api import app


def test_health_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["redis"] in ("ok", "down")


def test_submit_research_returns_task_id():
    client = TestClient(app)

    class _R:
        id = "task-123"

    with patch("src.backend.api.run_research_task.delay", return_value=_R()):
        resp = client.post(
            "/tasks/research",
            json={"research_topic": "RAG 应用实践", "max_cycles": 2},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "task-123"
    assert body["status"] == "PENDING"


def test_submit_eval_returns_task_id():
    client = TestClient(app)

    class _R:
        id = "task-456"

    with patch("src.backend.api.run_eval_task.delay", return_value=_R()):
        resp = client.post(
            "/tasks/eval",
            json={"mode": "retrieval", "k": 5, "eval_modes": "baseline"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "task-456"
    assert body["status"] == "PENDING"
