import pytest
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError

from backend.app.db.models import (
    Target,
    TestDefinition,
    Job,
    Run,
    Result,
    CaseResult,
    Artifact,
)
from backend.app.db.metrics import (
    format_metric,
    validate_metric,
    validate_metrics_dict,
)


def test_target_crud_and_snapshot(db_session):
    target = Target(
        name="Local STT Service",
        kind="live_stt",
        base_url="http://localhost:8001",
        auth_env_var="LOCAL_STT_KEY",
        extra={"model": "whisper-large-v3", "stream": True},
    )
    db_session.add(target)
    db_session.commit()
    db_session.refresh(target)

    assert target.id is not None
    assert target.name == "Local STT Service"
    assert target.kind == "live_stt"
    assert target.auth_env_var == "LOCAL_STT_KEY"
    assert target.extra["model"] == "whisper-large-v3"
    assert target.created_at is not None
    assert target.updated_at is not None

    snapshot = target.to_snapshot()
    assert snapshot["id"] == target.id
    assert snapshot["name"] == "Local STT Service"
    assert snapshot["kind"] == "live_stt"
    assert snapshot["base_url"] == "http://localhost:8001"
    assert snapshot["auth_env_var"] == "LOCAL_STT_KEY"
    assert snapshot["extra"] == {"model": "whisper-large-v3", "stream": True}

    # Unique constraint test
    duplicate_target = Target(
        name="Local STT Service",
        kind="live_stt",
        base_url="http://localhost:8002",
    )
    db_session.add(duplicate_target)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_test_definition_crud(db_session):
    target = Target(
        name="Embeddings Target",
        kind="embedding",
        base_url="http://localhost:8002",
    )
    db_session.add(target)
    db_session.commit()

    definition = TestDefinition(
        name="Embedding Throughput & Norms",
        tester="embedding",
        suites=["throughput", "vector_norms"],
        config={"batch_size": 32, "max_concurrency": 8},
        target_id=target.id,
    )
    db_session.add(definition)
    db_session.commit()
    db_session.refresh(definition)

    assert definition.id is not None
    assert definition.target_id == target.id
    assert definition.target.name == "Embeddings Target"
    assert definition.suites == ["throughput", "vector_norms"]
    assert definition.config["batch_size"] == 32

    snapshot = definition.to_snapshot()
    assert snapshot["id"] == definition.id
    assert snapshot["tester"] == "embedding"
    assert snapshot["suites"] == ["throughput", "vector_norms"]


def test_job_crud(db_session):
    now = datetime.now(timezone.utc)
    job = Job(
        name="Nightly Benchmark Suite",
        description="Bundled nightly regression suite",
        schedule="0 2 * * *",
        enabled=True,
        run_mode="parallel",
        missed_run_policy="run_once",
        next_run_at=now,
        definition_ids=[1, 2, 3],
        overrides={"1": {"suites": ["smoke"], "config": {"batch_size": 16}}},
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    assert job.id is not None
    assert job.name == "Nightly Benchmark Suite"
    assert job.description == "Bundled nightly regression suite"
    assert job.schedule == "0 2 * * *"
    assert job.enabled is True
    assert job.run_mode == "parallel"
    assert job.missed_run_policy == "run_once"
    assert job.next_run_at is not None
    assert job.definition_ids == [1, 2, 3]
    assert job.overrides["1"]["config"]["batch_size"] == 16

    snap = job.to_snapshot()
    assert snap["id"] == job.id
    assert snap["run_mode"] == "parallel"
    assert snap["missed_run_policy"] == "run_once"
    assert snap["next_run_at"] is not None
    assert snap["definition_ids"] == [1, 2, 3]
    assert snap["overrides"]["1"]["suites"] == ["smoke"]

    # Unique constraint test
    dup_job = Job(name="Nightly Benchmark Suite", schedule="manual")
    db_session.add(dup_job)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_run_snapshots_and_immutability(db_session):
    # Setup target and definitions
    target = Target(
        name="LLM Test Target",
        kind="llm",
        base_url="http://localhost:8000/v1",
        extra={"model": "gpt-4o"},
    )
    db_session.add(target)
    db_session.commit()

    def1 = TestDefinition(
        name="LLM Reasoning",
        tester="llm",
        suites=["reasoning"],
        config={"temperature": 0.0},
        target_id=target.id,
    )
    db_session.add(def1)
    db_session.commit()

    now = datetime.now(timezone.utc)
    run = Run(
        status="running",
        queued_at=now,
        started_at=now,
        git_info={"commit": "abc1234", "branch": "main", "dirty": False},
    )
    run.snapshot_target(target)
    run.snapshot_config([def1])
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    assert run.target_snapshot["name"] == "LLM Test Target"
    assert run.target_snapshot["extra"]["model"] == "gpt-4o"
    assert len(run.config_snapshot) == 1
    assert run.config_snapshot[0]["config"]["temperature"] == 0.0

    # Mutate target and definition; verify snapshot remains unmodified
    target.name = "Renamed Target"
    target.extra = {"model": "gpt-4o-mini"}
    def1.config = {"temperature": 0.7}
    db_session.commit()
    db_session.refresh(run)

    assert run.target_snapshot["name"] == "LLM Test Target"
    assert run.target_snapshot["extra"]["model"] == "gpt-4o"
    assert run.config_snapshot[0]["config"]["temperature"] == 0.0


def test_result_metrics_and_better_direction(db_session):
    run = Run(status="completed")
    db_session.add(run)
    db_session.commit()

    result = Result(
        run_id=run.id,
        tester="llm",
        status="pass",
        summary="All tests passed successfully",
        metrics={
            "ttft_ms": {"value": 142.5, "unit": "ms", "better": "lower"},
            "tps": {"value": 85.0, "unit": "tok/s", "better": "higher"},
        },
    )
    db_session.add(result)
    db_session.commit()
    db_session.refresh(result)

    assert result.metrics["ttft_ms"]["value"] == 142.5
    assert result.metrics["ttft_ms"]["better"] == "lower"
    assert result.metrics["tps"]["better"] == "higher"

    # Test helper method add_metric
    result.add_metric(name="p95_ms", value=320.0, unit="ms", better="lower")
    db_session.commit()
    db_session.refresh(result)

    assert "p95_ms" in result.metrics
    assert result.metrics["p95_ms"] == {"value": 320.0, "unit": "ms", "better": "lower"}

    # Invalid better direction should raise ValueError
    with pytest.raises(ValueError, match="better"):
        result.add_metric(name="invalid_metric", value=10, better="faster")


def test_case_result_crud_and_metrics(db_session):
    run = Run(status="completed")
    db_session.add(run)
    db_session.commit()

    result = Result(run_id=run.id, tester="live_stt", status="pass")
    db_session.add(result)
    db_session.commit()

    case = CaseResult(
        result_id=result.id,
        suite="accuracy",
        case_id="meeting_session_01",
        status="pass",
        detail={"raw_transcript": "Hello world", "ground_truth": "Hello world"},
    )
    case.add_metric("wer", 0.042, unit="ratio", better="lower")
    case.add_metric("der", 0.015, unit="ratio", better="lower")
    db_session.add(case)
    db_session.commit()
    db_session.refresh(case)

    assert case.id is not None
    assert case.suite == "accuracy"
    assert case.case_id == "meeting_session_01"
    assert case.status == "pass"
    assert case.metrics["wer"]["better"] == "lower"
    assert case.detail["raw_transcript"] == "Hello world"
    assert case.result.id == result.id


def test_artifact_crud(db_session):
    run = Run(status="completed")
    db_session.add(run)
    db_session.commit()

    result = Result(run_id=run.id, tester="live_stt", status="pass")
    db_session.add(result)
    db_session.commit()

    artifact = Artifact(
        result_id=result.id,
        name="recording_audio.wav",
        path="artifacts/run_1/recording_audio.wav",
        mime="audio/wav",
        size_bytes=1048576,
    )
    db_session.add(artifact)
    db_session.commit()
    db_session.refresh(artifact)

    assert artifact.id is not None
    assert artifact.name == "recording_audio.wav"
    assert artifact.mime == "audio/wav"
    assert artifact.size_bytes == 1048576
    assert artifact.created_at is not None
    assert artifact.result.id == result.id


def test_cascade_deletes(db_session):
    # Setup full hierarchy
    job = Job(name="Bench Job")
    db_session.add(job)
    target = Target(name="T1", kind="llm", base_url="http://t1")
    db_session.add(target)
    db_session.commit()

    def1 = TestDefinition(name="D1", tester="llm", target_id=target.id)
    db_session.add(def1)
    run = Run(job_id=job.id, status="completed")
    db_session.add(run)
    db_session.commit()

    result = Result(run_id=run.id, test_definition_id=def1.id, tester="llm", status="pass")
    db_session.add(result)
    db_session.commit()

    case = CaseResult(result_id=result.id, suite="s1", case_id="c1", status="pass")
    artifact = Artifact(result_id=result.id, name="log.txt", path="artifacts/log.txt", mime="text/plain")
    db_session.add_all([case, artifact])
    db_session.commit()

    # Verify entities exist
    assert db_session.query(Result).count() == 1
    assert db_session.query(CaseResult).count() == 1
    assert db_session.query(Artifact).count() == 1

    # Deleting run cascades to result, case_result, and artifact
    db_session.delete(run)
    db_session.commit()

    assert db_session.query(Result).count() == 0
    assert db_session.query(CaseResult).count() == 0
    assert db_session.query(Artifact).count() == 0

    # Deleting Target sets target_id on TestDefinition to NULL (ON DELETE SET NULL)
    db_session.delete(target)
    db_session.commit()
    db_session.refresh(def1)
    assert def1.target_id is None

    # Deleting Job sets job_id on Run to NULL
    run2 = Run(job_id=job.id, status="pending")
    db_session.add(run2)
    db_session.commit()
    db_session.delete(job)
    db_session.commit()
    db_session.refresh(run2)
    assert run2.job_id is None


def test_metric_validation_helpers():
    # Valid metrics
    m1 = format_metric(12.5, unit="ms", better="lower")
    assert m1 == {"value": 12.5, "unit": "ms", "better": "lower"}
    validate_metric(m1)

    m2 = format_metric(99, better="higher")
    assert m2 == {"value": 99, "better": "higher"}
    validate_metric(m2)

    validate_metrics_dict({"latency": m1, "score": m2})

    # Invalid metrics
    with pytest.raises(ValueError, match="better"):
        format_metric(10, better="more")

    with pytest.raises(ValueError, match="number"):
        format_metric("not-a-number")

    with pytest.raises(ValueError, match="value"):
        validate_metric({"unit": "ms"})

    with pytest.raises(ValueError, match="better"):
        validate_metric({"value": 10, "better": "invalid"})
