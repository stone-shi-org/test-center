import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.config import get_settings
from backend.app.db.base import Base, utc_now
from backend.app.db.models.run import Run
from backend.app.db.models.result import Result
from backend.app.db.models.artifact import Artifact
from backend.app.db.session import get_db, configure_engine_pragmas
from backend.app.main import create_app
from backend.app.runner.worker import RunWorker, get_worker
from backend.app.runner.events import get_broadcaster




def test_api_testers_list_and_detail(app_and_client):
    client, _, _ = app_and_client

    # List testers
    res = client.get("/api/testers")
    assert res.status_code == 200
    testers = res.json()
    assert isinstance(testers, list)
    assert any(t["name"] == "dummy" for t in testers)

    # Get specific tester
    res = client.get("/api/testers/dummy")
    assert res.status_code == 200
    dummy_info = res.json()
    assert dummy_info["name"] == "dummy"
    assert "properties" in dummy_info["config_schema"]
    assert "smoke" in dummy_info["suites"]

    # Not found
    res = client.get("/api/testers/nonexistent")
    assert res.status_code == 404


def test_api_runs_crud_and_execution(app_and_client):
    client, session_factory, artifacts_dir = app_and_client

    # 1. Create Run
    create_payload = {
        "tester": "dummy",
        "suites": ["smoke"],
        "config": {
            "duration_seconds": 0.1,
            "step_delay": 0.05,
        },
        "target": {"name": "test-env"},
    }
    res = client.post("/api/runs", json=create_payload)
    assert res.status_code == 201
    run_data = res.json()
    run_id = run_data["id"]
    assert run_data["status"] == "queued"
    assert run_data["config_snapshot"]["tester"] == "dummy"

    # 2. List Runs
    res = client.get("/api/runs")
    assert res.status_code == 200
    runs = res.json()
    assert any(r["id"] == run_id for r in runs)

    # 3. Execute with worker
    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True

    # 4. Get Run Detail
    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
    detail = res.json()
    assert detail["status"] == "completed"
    assert len(detail["results"]) == 1
    assert detail["results"][0]["status"] == "pass"
    assert len(detail["results"][0]["cases"]) == 1
    assert len(detail["results"][0]["artifacts"]) >= 2

    # 5. Get Logs
    res = client.get(f"/api/runs/{run_id}/logs")
    assert res.status_code == 200
    logs_data = res.json()
    assert logs_data["run_id"] == run_id
    assert len(logs_data["lines"]) > 0

    # 6. Download Artifact
    artifact_id = detail["results"][0]["artifacts"][0]["id"]
    res = client.get(f"/api/runs/{run_id}/artifacts/{artifact_id}/download")
    assert res.status_code == 200
    assert len(res.content) > 0


def test_api_cancel_run(app_and_client):
    client, _, _ = app_and_client

    # Create run
    create_payload = {
        "tester": "dummy",
        "config": {"duration_seconds": 5.0},
    }
    res = client.post("/api/runs", json=create_payload)
    assert res.status_code == 201
    run_id = res.json()["id"]

    # Cancel while queued
    res = client.post(f"/api/runs/{run_id}/cancel")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # Verify status
    res = client.get(f"/api/runs/{run_id}")
    assert res.json()["status"] == "cancelled"


def test_api_sse_log_stream_completed_run(app_and_client):
    client, session_factory, artifacts_dir = app_and_client

    # Create and finish a run
    db = session_factory()
    run = Run(
        status="completed",
        queued_at=utc_now(),
        started_at=utc_now(),
        finished_at=utc_now(),
        config_snapshot={"tester": "dummy"},
        target_snapshot={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id

    # Create a mock run.log
    run_dir = artifacts_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.log").write_text("[2026-10-16T12:00:00Z] [INFO] Sample log line\n")
    db.close()

    # Stream SSE
    with client.stream("GET", f"/api/runs/{run_id}/logs/stream") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = list(response.iter_lines())
        assert any("Sample log line" in ev for ev in events)
        assert any("completed" in ev for ev in events)


def test_api_websocket_log_stream(app_and_client):
    client, session_factory, artifacts_dir = app_and_client

    # Create and finish a run
    db = session_factory()
    run = Run(
        status="completed",
        queued_at=utc_now(),
        started_at=utc_now(),
        finished_at=utc_now(),
        config_snapshot={"tester": "dummy"},
        target_snapshot={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id

    # Create a mock run.log
    run_dir = artifacts_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.log").write_text("[2026-10-16T12:00:00Z] [INFO] WebSocket test line\n")
    db.close()

    with client.websocket_connect(f"/api/runs/{run_id}/ws") as websocket:
        data = websocket.receive_text()
        parsed = json.loads(data)
        assert parsed["type"] == "log"
        assert "WebSocket test line" in parsed["message"]


def test_run_without_definition_id_end_to_end(app_and_client):
    """Runs triggered without a test_definition_id work end-to-end across worker, results, JUnit, Compare, and notifications."""
    from backend.app.runner.notifications import _pair_results_for_comparison

    client, session_factory, _ = app_and_client

    # Create a target
    res_t = client.post(
        "/api/targets",
        json={"name": "Direct Run Target", "kind": "dummy", "base_url": "http://127.0.0.1:8000"},
    )
    assert res_t.status_code == 201
    target_id = res_t.json()["id"]

    defs_before = client.get("/api/test-definitions").json()

    # Trigger two runs without test_definition_id (as the simplified Trigger page does)
    run_ids = []
    for duration in (0.05, 0.08):
        res_run = client.post(
            "/api/runs",
            json={
                "tester": "dummy",
                "suites": ["smoke"],
                "config": {"duration_seconds": duration, "step_delay": 0.01},
                "target_id": target_id,
                "timeout": 30.0,
            },
        )
        assert res_run.status_code == 201
        r_json = res_run.json()
        assert r_json["config_snapshot"].get("test_definition_id") is None
        assert r_json["config_snapshot"]["timeout"] == 30.0
        assert r_json["target_snapshot"]["id"] == target_id
        run_ids.append(r_json["id"])

    # No new TestDefinition rows created
    defs_after = client.get("/api/test-definitions").json()
    assert len(defs_after) == len(defs_before)

    # Execute both runs with worker
    worker = get_worker()
    loop = asyncio.new_event_loop()
    for _ in run_ids:
        assert loop.run_until_complete(worker.poll_and_execute_once()) is True
    loop.close()

    # Check Run Detail, Results list, and JUnit XML for both runs
    result_ids = []
    for rid in run_ids:
        detail = client.get(f"/api/runs/{rid}").json()
        assert detail["status"] == "completed"
        assert len(detail["results"]) == 1
        res_item = detail["results"][0]
        assert res_item["test_definition_id"] is None
        assert res_item["definition_name"] is None
        assert res_item["status"] == "pass"
        result_ids.append(res_item["id"])

        res_junit_run = client.get(f"/api/runs/{rid}/junit.xml")
        assert res_junit_run.status_code == 200
        assert "<testsuites" in res_junit_run.text

        res_junit_result = client.get(f"/api/results/{res_item['id']}/junit.xml")
        assert res_junit_result.status_code == 200
        assert "<testsuites" in res_junit_result.text

    # Compare the two definition-less runs
    res_cmp = client.get(f"/api/compare?run_ids={run_ids[0]},{run_ids[1]}")
    assert res_cmp.status_code == 200
    cmp_data = res_cmp.json()
    assert cmp_data["tester"] == "dummy"
    assert len(cmp_data["runs"]) == 2
    assert all(r["test_definition_id"] is None for r in cmp_data["runs"])

    # Notifications pairing fallback by tester when test_definition_id is None
    db = session_factory()
    try:
        r0 = db.get(Result, result_ids[0])
        r1 = db.get(Result, result_ids[1])
        pairs = _pair_results_for_comparison([r0], [r1])
        assert len(pairs) == 1
        assert pairs[0][0].id == r0.id
        assert pairs[0][1].id == r1.id
    finally:
        db.close()


def test_run_datetime_utc_serialization_and_db_persistence(app_and_client):
    """Verify queued_at, started_at, and finished_at are stored and returned with UTC timezone."""
    client, session_factory, artifacts_dir = app_and_client

    db = session_factory()
    run = Run(
        status="completed",
        queued_at=datetime(2026, 10, 21, 14, 0, 0, tzinfo=timezone.utc),
        started_at=datetime(2026, 10, 21, 14, 0, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 10, 21, 14, 0, 5, tzinfo=timezone.utc),
        config_snapshot={"tester": "dummy"},
        target_snapshot={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Verify model tzinfo is preserved from DB
    assert run.queued_at.tzinfo == timezone.utc
    assert run.started_at.tzinfo == timezone.utc
    assert run.finished_at.tzinfo == timezone.utc
    run_id = run.id
    db.close()

    # Verify API serializes with explicit timezone (+00:00 or Z)
    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
    data = res.json()
    assert "+00:00" in data["queued_at"] or data["queued_at"].endswith("Z")
    assert "+00:00" in data["started_at"] or data["started_at"].endswith("Z")
    assert "+00:00" in data["finished_at"] or data["finished_at"].endswith("Z")

    # Verify list runs API also includes timezone
    res_list = client.get("/api/runs?limit=1")
    assert res_list.status_code == 200
    first_run = res_list.json()[0]
    assert "+00:00" in first_run["queued_at"] or first_run["queued_at"].endswith("Z")

