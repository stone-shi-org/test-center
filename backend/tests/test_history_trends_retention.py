"""Tests for TES-29: Filterable run history, per-metric trend charts per target/model, and retention settings."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from backend.app.db.models.artifact import Artifact
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.job import Job
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run
from backend.app.db.models.target import Target
from backend.app.db.models.test_definition import TestDefinition
from backend.app.runner.worker import get_worker


def _seed_history_runs(session_factory, artifacts_dir: Path):
    """Seed a rich set of historical runs across testers, targets, models, jobs, statuses, and dates."""
    db = session_factory()
    try:
        t_a100 = Target(
            name="vllm-a100",
            kind="llm",
            base_url="http://a100:8000/v1",
            extra={"model": "Llama-3-8B"},
        )
        t_h100 = Target(
            name="vllm-h100",
            kind="llm",
            base_url="http://h100:8000/v1",
            extra={"model": "Llama-3-70B"},
        )
        t_embed = Target(
            name="bge-server",
            kind="embedding",
            base_url="http://embed:8001/v1",
            extra={"model": "bge-large-en"},
        )
        db.add_all([t_a100, t_h100, t_embed])
        db.commit()
        db.refresh(t_a100)
        db.refresh(t_h100)
        db.refresh(t_embed)

        job_nightly = Job(
            name="Nightly LLM Benchmark",
            schedule="0 2 * * *",
            enabled=True,
            run_mode="sequential",
            definition_ids=[],
        )
        db.add(job_nightly)
        db.commit()
        db.refresh(job_nightly)

        base_time = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Run 1: 15 days ago, LLM on vllm-a100 (Llama-3-8B), part of Nightly job, PASS
        t1 = base_time
        r1 = Run(
            job_id=job_nightly.id,
            status="completed",
            queued_at=t1,
            started_at=t1,
            finished_at=t1 + timedelta(seconds=10),
            config_snapshot={
                "job_id": job_nightly.id,
                "job_name": job_nightly.name,
                "tester": "llm",
                "config": {"model": "Llama-3-8B"},
            },
            target_snapshot=t_a100.to_snapshot(),
        )
        db.add(r1)
        db.flush()
        res1 = Result(
            run_id=r1.id,
            tester="llm",
            status="pass",
            summary="Run 1 pass",
            metrics={
                "ttft_ms": {"value": 120.0, "unit": "ms", "better": "lower"},
                "tps": {"value": 85.0, "unit": "tok/s", "better": "higher"},
            },
        )
        db.add(res1)
        db.flush()
        r1_dir = artifacts_dir / str(r1.id)
        r1_dir.mkdir(parents=True, exist_ok=True)
        r1_log = r1_dir / "run.log"
        r1_log.write_text("Run 1 log content\n" * 10, encoding="utf-8")
        db.add(
            Artifact(
                result_id=res1.id,
                name="run.log",
                path=str(r1_log),
                mime="text/plain",
                size_bytes=r1_log.stat().st_size,
            )
        )

        # Run 2: 10 days later (Oct 11), LLM on vllm-a100 (Llama-3-8B), part of Nightly job, PASS (improved ttft)
        t2 = base_time + timedelta(days=10)
        r2 = Run(
            job_id=job_nightly.id,
            status="completed",
            queued_at=t2,
            started_at=t2,
            finished_at=t2 + timedelta(seconds=9),
            config_snapshot={
                "job_id": job_nightly.id,
                "job_name": job_nightly.name,
                "tester": "llm",
                "config": {"model": "Llama-3-8B"},
            },
            target_snapshot=t_a100.to_snapshot(),
        )
        db.add(r2)
        db.flush()
        res2 = Result(
            run_id=r2.id,
            tester="llm",
            status="pass",
            summary="Run 2 pass",
            metrics={
                "ttft_ms": {"value": 95.0, "unit": "ms", "better": "lower"},
                "tps": {"value": 92.0, "unit": "tok/s", "better": "higher"},
            },
        )
        db.add(res2)
        db.flush()
        r2_dir = artifacts_dir / str(r2.id)
        r2_dir.mkdir(parents=True, exist_ok=True)
        r2_log = r2_dir / "run.log"
        r2_log.write_text("Run 2 log content\n" * 10, encoding="utf-8")
        db.add(
            Artifact(
                result_id=res2.id,
                name="run.log",
                path=str(r2_log),
                mime="text/plain",
                size_bytes=r2_log.stat().st_size,
            )
        )

        # Run 3: 12 days later (Oct 13), LLM on vllm-h100 (Llama-3-70B), manual run, FAIL
        t3 = base_time + timedelta(days=12)
        r3 = Run(
            job_id=None,
            status="completed",
            queued_at=t3,
            started_at=t3,
            finished_at=t3 + timedelta(seconds=15),
            config_snapshot={
                "tester": "llm",
                "config": {"model": "Llama-3-70B"},
            },
            target_snapshot=t_h100.to_snapshot(),
        )
        db.add(r3)
        db.flush()
        res3 = Result(
            run_id=r3.id,
            tester="llm",
            status="fail",
            summary="Run 3 had 1 failure",
            metrics={
                "ttft_ms": {"value": 210.0, "unit": "ms", "better": "lower"},
                "tps": {"value": 48.0, "unit": "tok/s", "better": "higher"},
            },
        )
        db.add(res3)
        db.flush()
        r3_dir = artifacts_dir / str(r3.id)
        r3_dir.mkdir(parents=True, exist_ok=True)
        r3_log = r3_dir / "run.log"
        r3_log.write_text("Run 3 log content\n" * 10, encoding="utf-8")
        db.add(
            Artifact(
                result_id=res3.id,
                name="run.log",
                path=str(r3_log),
                mime="text/plain",
                size_bytes=r3_log.stat().st_size,
            )
        )

        # Run 4: 15 days later (Oct 16), Embedding on bge-server, manual run, PASS
        t4 = base_time + timedelta(days=15)
        r4 = Run(
            job_id=None,
            status="completed",
            queued_at=t4,
            started_at=t4,
            finished_at=t4 + timedelta(seconds=5),
            config_snapshot={
                "tester": "embedding",
                "config": {"model": "bge-large-en"},
            },
            target_snapshot=t_embed.to_snapshot(),
        )
        db.add(r4)
        db.flush()
        res4 = Result(
            run_id=r4.id,
            tester="embedding",
            status="pass",
            summary="Embedding pass",
            metrics={
                "recall@5": {"value": 0.94, "unit": "ratio", "better": "higher"},
                "latency_p95_ms": {"value": 18.5, "unit": "ms", "better": "lower"},
            },
        )
        db.add(res4)
        db.flush()
        r4_dir = artifacts_dir / str(r4.id)
        r4_dir.mkdir(parents=True, exist_ok=True)
        r4_log = r4_dir / "run.log"
        r4_log.write_text("Run 4 log content\n" * 10, encoding="utf-8")
        db.add(
            Artifact(
                result_id=res4.id,
                name="run.log",
                path=str(r4_log),
                mime="text/plain",
                size_bytes=r4_log.stat().st_size,
            )
        )

        db.commit()
        return {
            "targets": {"a100": t_a100.id, "h100": t_h100.id, "embed": t_embed.id},
            "job_id": job_nightly.id,
            "runs": [r1.id, r2.id, r3.id, r4.id],
        }
    finally:
        db.close()


def test_run_history_filtering(app_and_client):
    """Verify GET /api/runs filters by tester, target, model, job, status, and date range."""
    client, session_factory, artifacts_dir = app_and_client
    seeded = _seed_history_runs(session_factory, artifacts_dir)
    r1_id, r2_id, r3_id, r4_id = seeded["runs"]

    # 1. Filter by tester
    res_llm = client.get("/api/runs?tester=llm")
    assert res_llm.status_code == 200
    assert [r["id"] for r in res_llm.json()] == [r3_id, r2_id, r1_id]

    res_emb = client.get("/api/runs?tester=embedding")
    assert res_emb.status_code == 200
    assert [r["id"] for r in res_emb.json()] == [r4_id]

    # 2. Filter by target_id and target name
    res_t_a100 = client.get(f"/api/runs?target_id={seeded['targets']['a100']}")
    assert res_t_a100.status_code == 200
    assert [r["id"] for r in res_t_a100.json()] == [r2_id, r1_id]

    res_t_name = client.get("/api/runs?target=vllm-h100")
    assert res_t_name.status_code == 200
    assert [r["id"] for r in res_t_name.json()] == [r3_id]

    # 3. Filter by model
    res_model = client.get("/api/runs?model=Llama-3-70B")
    assert res_model.status_code == 200
    assert [r["id"] for r in res_model.json()] == [r3_id]

    # 4. Filter by job_id, job name, and manual
    res_job = client.get(f"/api/runs?job_id={seeded['job_id']}")
    assert res_job.status_code == 200
    assert [r["id"] for r in res_job.json()] == [r2_id, r1_id]

    res_job_name = client.get("/api/runs?job=Nightly")
    assert res_job_name.status_code == 200
    assert [r["id"] for r in res_job_name.json()] == [r2_id, r1_id]

    res_manual = client.get("/api/runs?job=manual")
    assert res_manual.status_code == 200
    assert [r["id"] for r in res_manual.json()] == [r4_id, r3_id]

    # 5. Filter by status (including result status pass/fail)
    res_pass = client.get("/api/runs?status=pass")
    assert res_pass.status_code == 200
    assert [r["id"] for r in res_pass.json()] == [r4_id, r2_id, r1_id]

    res_fail = client.get("/api/runs?status=fail")
    assert res_fail.status_code == 200
    assert [r["id"] for r in res_fail.json()] == [r3_id]

    # 6. Filter by date range
    res_dates = client.get("/api/runs?date_from=2026-10-10&date_to=2026-10-14")
    assert res_dates.status_code == 200
    assert [r["id"] for r in res_dates.json()] == [r3_id, r2_id]

    res_single_date = client.get("/api/runs?date=2026-10-01")
    assert res_single_date.status_code == 200
    assert [r["id"] for r in res_single_date.json()] == [r1_id]

    # Invalid date returns 400
    res_bad_date = client.get("/api/runs?date_from=not-a-date")
    assert res_bad_date.status_code == 400


def test_per_metric_trend_charts_api(app_and_client):
    """Verify GET /api/runs/trends returns per-target/model time series and available metrics."""
    client, session_factory, artifacts_dir = app_and_client
    seeded = _seed_history_runs(session_factory, artifacts_dir)
    r1_id, r2_id, r3_id, _ = seeded["runs"]

    # Query trends for llm tester and metric ttft_ms
    res = client.get("/api/runs/trends?tester=llm&metric=ttft_ms")
    assert res.status_code == 200
    data = res.json()

    assert data["metric"] == "ttft_ms"
    assert data["unit"] == "ms"
    assert data["better"] == "lower"

    metric_names = [m["name"] for m in data["available_metrics"]]
    assert "ttft_ms" in metric_names
    assert "tps" in metric_names

    # Should have 2 target/model series: 'vllm-a100 / Llama-3-8B' (2 points) and 'vllm-h100 / Llama-3-70B' (1 point)
    series_by_key = {s["series_key"]: s for s in data["series"]}
    assert "vllm-a100 / Llama-3-8B" in series_by_key
    assert "vllm-h100 / Llama-3-70B" in series_by_key

    a100_series = series_by_key["vllm-a100 / Llama-3-8B"]
    assert a100_series["target_name"] == "vllm-a100"
    assert a100_series["model"] == "Llama-3-8B"
    # Points ordered chronologically (r1 then r2)
    assert [pt["run_id"] for pt in a100_series["points"]] == [r1_id, r2_id]
    assert [pt["value"] for pt in a100_series["points"]] == [120.0, 95.0]

    h100_series = series_by_key["vllm-h100 / Llama-3-70B"]
    assert [pt["run_id"] for pt in h100_series["points"]] == [r3_id]
    assert [pt["value"] for pt in h100_series["points"]] == [210.0]

    # Also verify /api/trends alias works
    res_alias = client.get("/api/trends?tester=embedding")
    assert res_alias.status_code == 200
    assert res_alias.json()["metric"] in ("recall@5", "latency_p95_ms")


def test_retention_settings_and_cleanup_modes(app_and_client):
    """Verify retention policy persistence, dry_run preview, artifacts_only cleanup, and full run cleanup."""
    client, session_factory, artifacts_dir = app_and_client
    seeded = _seed_history_runs(session_factory, artifacts_dir)
    r1_id, r2_id, r3_id, r4_id = seeded["runs"]

    # 1. Inspect initial retention settings and storage stats
    res_init = client.get("/api/runs/retention")
    assert res_init.status_code == 200
    init_data = res_init.json()
    assert init_data["enabled"] is False
    assert init_data["stats"]["total_runs"] == 4
    assert init_data["stats"]["total_artifacts"] == 4
    assert init_data["stats"]["artifacts_size_bytes"] > 0

    # 2. Update retention settings: keep last 2 runs, artifacts_only=True
    res_upd = client.put(
        "/api/runs/retention",
        json={
            "enabled": True,
            "max_runs": 2,
            "max_age_days": None,
            "artifacts_only": True,
            "keep_failed": False,
        },
    )
    assert res_upd.status_code == 200
    upd_data = res_upd.json()
    assert upd_data["enabled"] is True
    assert upd_data["max_runs"] == 2
    assert upd_data["artifacts_only"] is True
    assert upd_data["stats"]["eligible_runs_count"] == 2
    assert set(upd_data["stats"]["eligible_run_ids"]) == {r1_id, r2_id}

    # 3. Preview cleanup with dry_run=True
    res_dry = client.post("/api/runs/retention/cleanup", json={"dry_run": True})
    assert res_dry.status_code == 200
    dry_data = res_dry.json()
    assert dry_data["dry_run"] is True
    assert dry_data["cleaned_runs"] == 2
    assert dry_data["deleted_runs"] == 0  # because artifacts_only=True
    assert (artifacts_dir / str(r1_id)).exists()

    # 4. Execute cleanup in artifacts_only=True mode
    res_clean_art = client.post("/api/runs/retention/cleanup", json={"dry_run": False})
    assert res_clean_art.status_code == 200
    clean_art_data = res_clean_art.json()
    assert clean_art_data["artifacts_only"] is True
    assert clean_art_data["cleaned_runs"] == 2
    assert clean_art_data["deleted_artifacts"] == 2
    assert not (artifacts_dir / str(r1_id)).exists()
    assert not (artifacts_dir / str(r2_id)).exists()
    assert (artifacts_dir / str(r3_id)).exists()
    assert (artifacts_dir / str(r4_id)).exists()

    # Verify Run 1 and Run 2 still exist in DB and still appear in trend charts!
    res_runs_after_art = client.get("/api/runs")
    assert len(res_runs_after_art.json()) == 4
    res_trends = client.get("/api/runs/trends?tester=llm&metric=ttft_ms")
    assert len(res_trends.json()["points"]) == 3

    # 5. Now execute full run deletion for runs beyond max_runs=2 (with keep_failed=True)
    res_clean_full = client.post(
        "/api/runs/retention/cleanup",
        json={
            "max_runs": 1,
            "artifacts_only": False,
            "keep_failed": True,
            "dry_run": False,
        },
    )
    assert res_clean_full.status_code == 200
    clean_full_data = res_clean_full.json()
    # r4 is index 0 (kept), r3 is failed (kept because keep_failed=True), r2 and r1 are deleted!
    assert set(clean_full_data["deleted_run_ids"]) == {r1_id, r2_id}

    remaining_ids = [r["id"] for r in client.get("/api/runs").json()]
    assert remaining_ids == [r4_id, r3_id]

    # 6. Test DELETE /api/runs/{run_id}
    res_del = client.delete(f"/api/runs/{r3_id}")
    assert res_del.status_code == 204
    assert not (artifacts_dir / str(r3_id)).exists()
    assert [r["id"] for r in client.get("/api/runs").json()] == [r4_id]


def test_auto_retention_on_worker_completion(app_and_client):
    """Verify worker automatically applies retention policy after completing a run when enabled."""
    client, session_factory, artifacts_dir = app_and_client
    _seed_history_runs(session_factory, artifacts_dir)

    # Configure auto-retention to keep at most 2 terminal runs
    client.put(
        "/api/runs/retention",
        json={
            "enabled": True,
            "max_runs": 2,
            "artifacts_only": False,
        },
    )

    # Queue and execute a new dummy run
    res_create = client.post(
        "/api/runs",
        json={
            "tester": "dummy",
            "suites": ["smoke"],
            "config": {"duration_seconds": 0.05, "step_delay": 0.01},
        },
    )
    assert res_create.status_code == 201
    new_run_id = res_create.json()["id"]

    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True

    # Only the 2 newest terminal runs should remain
    runs = client.get("/api/runs").json()
    assert len(runs) == 2
    assert runs[0]["id"] == new_run_id
