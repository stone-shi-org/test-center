import asyncio
import json
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.app.db.base import Base, utc_now
from backend.app.db.models.run import Run
from backend.app.db.models.result import Result
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.artifact import Artifact
from backend.app.db.session import configure_engine_pragmas
from backend.app.runner.worker import RunWorker
from backend.app.runner.recovery import recover_stale_runs


@pytest.fixture
def test_db_session(tmp_path: Path):
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    configure_engine_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return TestingSessionLocal


@pytest.mark.asyncio
async def test_worker_end_to_end_dummy(tmp_path: Path, test_db_session):
    artifacts_dir = tmp_path / "artifacts"
    worker = RunWorker(session_factory=test_db_session, artifacts_dir=artifacts_dir)

    # Insert a queued run
    db: Session = test_db_session()
    run = Run(
        status="queued",
        queued_at=utc_now(),
        config_snapshot={
            "tester": "dummy",
            "config": {
                "duration_seconds": 0.1,
                "step_delay": 0.05,
                "fail_case": False,
                "create_artifacts": True,
            },
            "suites": ["smoke"],
        },
        target_snapshot={"name": "test-env"},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()

    # Worker executes the queued run
    executed = await worker.poll_and_execute_once()
    assert executed is True

    # Verify Run in DB
    db = test_db_session()
    updated_run = db.get(Run, run_id)
    assert updated_run is not None
    assert updated_run.status == "completed"
    assert updated_run.started_at is not None
    assert updated_run.finished_at is not None
    assert len(updated_run.results) == 1

    result = updated_run.results[0]
    assert result.status == "pass"
    assert result.tester == "dummy"
    assert "duration_ms" in result.metrics
    assert len(result.cases) == 1
    assert result.cases[0].suite == "smoke"
    assert result.cases[0].status == "pass"

    # Verify artifacts in DB and on disk
    artifact_names = [a.name for a in result.artifacts]
    assert "run.log" in artifact_names
    assert "result.json" in artifact_names
    assert "sample_output.txt" in artifact_names
    assert "metrics_summary.json" in artifact_names

    run_dir = artifacts_dir / str(run_id)
    assert (run_dir / "run.log").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "sample_output.txt").exists()
    assert (run_dir / "metrics_summary.json").exists()

    db.close()


@pytest.mark.asyncio
async def test_worker_cancel_running(tmp_path: Path, test_db_session):
    artifacts_dir = tmp_path / "artifacts"
    worker = RunWorker(session_factory=test_db_session, artifacts_dir=artifacts_dir)

    # Insert a run with long duration
    db: Session = test_db_session()
    run = Run(
        status="queued",
        queued_at=utc_now(),
        config_snapshot={
            "tester": "dummy",
            "config": {
                "duration_seconds": 10.0,
                "step_delay": 0.2,
            },
        },
        target_snapshot={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()

    # Launch execution as background task
    exec_task = asyncio.create_task(worker.poll_and_execute_once())

    # Wait until run transitions to running
    for _ in range(50):
        await asyncio.sleep(0.05)
        db = test_db_session()
        r = db.get(Run, run_id)
        status = r.status if r else None
        db.close()
        if status == "running":
            break

    # Cancel the run
    cancelled = await worker.cancel_run(run_id)
    assert cancelled is True

    # Wait for execution task to complete
    await asyncio.wait_for(exec_task, timeout=5.0)

    # Verify status in DB
    db = test_db_session()
    updated_run = db.get(Run, run_id)
    assert updated_run.status == "cancelled"
    assert updated_run.finished_at is not None
    assert len(updated_run.results) == 1
    assert updated_run.results[0].status == "cancelled"
    db.close()


@pytest.mark.asyncio
async def test_worker_cancel_queued(tmp_path: Path, test_db_session):
    artifacts_dir = tmp_path / "artifacts"
    worker = RunWorker(session_factory=test_db_session, artifacts_dir=artifacts_dir)

    # Insert a queued run
    db: Session = test_db_session()
    run = Run(
        status="queued",
        queued_at=utc_now(),
        config_snapshot={"tester": "dummy"},
        target_snapshot={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()

    # Cancel while still queued
    cancelled = await worker.cancel_run(run_id)
    assert cancelled is True

    db = test_db_session()
    updated_run = db.get(Run, run_id)
    assert updated_run.status == "cancelled"
    assert updated_run.finished_at is not None
    db.close()

    # Worker poll should not execute it
    executed = await worker.poll_and_execute_once()
    assert executed is False


@pytest.mark.asyncio
async def test_worker_timeout(tmp_path: Path, test_db_session):
    artifacts_dir = tmp_path / "artifacts"
    # Set a very short default timeout (0.3s)
    worker = RunWorker(
        session_factory=test_db_session,
        artifacts_dir=artifacts_dir,
        default_timeout=0.3,
    )

    db: Session = test_db_session()
    run = Run(
        status="queued",
        queued_at=utc_now(),
        config_snapshot={
            "tester": "dummy",
            "config": {
                "duration_seconds": 5.0,
                "step_delay": 0.2,
            },
        },
        target_snapshot={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()

    executed = await worker.poll_and_execute_once()
    assert executed is True

    db = test_db_session()
    updated_run = db.get(Run, run_id)
    assert updated_run.status == "error"
    assert "timed out" in updated_run.results[0].summary.lower()
    db.close()


def test_recover_stale_runs_on_restart(tmp_path: Path, test_db_session):
    artifacts_dir = tmp_path / "artifacts"
    db: Session = test_db_session()

    # Create one stale running run, one completed run, and one queued run
    stale_run = Run(
        status="running",
        started_at=utc_now(),
        config_snapshot={"tester": "dummy"},
        target_snapshot={},
    )
    completed_run = Run(
        status="completed",
        started_at=utc_now(),
        finished_at=utc_now(),
        config_snapshot={"tester": "dummy"},
        target_snapshot={},
    )
    queued_run = Run(
        status="queued",
        queued_at=utc_now(),
        config_snapshot={"tester": "dummy"},
        target_snapshot={},
    )
    db.add_all([stale_run, completed_run, queued_run])
    db.commit()
    db.refresh(stale_run)
    db.refresh(completed_run)
    db.refresh(queued_run)

    stale_id = stale_run.id
    completed_id = completed_run.id
    queued_id = queued_run.id
    db.close()

    # Simulate backend restart recovery
    db = test_db_session()
    recovered = recover_stale_runs(db, artifacts_dir)
    assert recovered == [stale_id]

    r_stale = db.get(Run, stale_id)
    assert r_stale.status == "error"
    assert r_stale.finished_at is not None
    assert len(r_stale.results) == 1
    assert r_stale.results[0].status == "error"
    assert "restart" in r_stale.results[0].summary.lower()

    # Check log file
    log_file = artifacts_dir / str(stale_id) / "run.log"
    assert log_file.exists()
    assert "backend restart" in log_file.read_text()

    # Verify others untouched
    assert db.get(Run, completed_id).status == "completed"
    assert db.get(Run, queued_id).status == "queued"
    db.close()
