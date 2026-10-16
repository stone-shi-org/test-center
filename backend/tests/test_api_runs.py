import asyncio
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


@pytest.fixture
def app_and_client(tmp_path: Path):
    db_path = tmp_path / "test_api.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    configure_engine_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    app = create_app()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Configure worker to use the test session factory and artifacts dir
    worker = get_worker()
    worker.session_factory = TestingSessionLocal
    worker.artifacts_dir = tmp_path / "artifacts"

    # Also update settings
    settings = get_settings()
    settings.DATA_DIR = tmp_path
    settings.DATABASE_URL = db_url

    client = TestClient(app)
    return client, TestingSessionLocal, tmp_path / "artifacts"


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
