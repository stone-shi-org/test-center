"""Tests for TES-44: Named run baselines API, run responses/filters, deletion protection, retention protection, and compare integration."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.app.db.models.artifact import Artifact
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run
from backend.app.db.models.target import Target


def _seed_baseline_test_data(session_factory, artifacts_dir: Path) -> dict[str, int]:
    db = session_factory()
    try:
        t_llm = Target(
            name="vllm-prod",
            kind="llm",
            base_url="http://vllm-prod:8000/v1",
            extra={"model": "Qwen2.5-72B"},
        )
        t_stt = Target(
            name="whisper-prod",
            kind="live_stt",
            base_url="grpc://whisper:50051",
            extra={"model": "whisper-large-v3"},
        )
        db.add_all([t_llm, t_stt])
        db.commit()
        db.refresh(t_llm)
        db.refresh(t_stt)

        base_time = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Run 1: old completed LLM run (20 days ago) with artifacts
        r1 = Run(
            status="completed",
            queued_at=base_time,
            started_at=base_time,
            finished_at=base_time + timedelta(seconds=15),
            config_snapshot={"tester": "llm", "config": {"model": "Qwen2.5-72B"}},
            target_snapshot=t_llm.to_snapshot(),
        )
        # Run 2: newer completed LLM run (15 days ago) with artifacts
        t2 = base_time + timedelta(days=5)
        r2 = Run(
            status="completed",
            queued_at=t2,
            started_at=t2,
            finished_at=t2 + timedelta(seconds=12),
            config_snapshot={"tester": "llm", "config": {"model": "Qwen2.5-72B"}},
            target_snapshot=t_llm.to_snapshot(),
        )
        # Run 3: completed STT run (10 days ago)
        t3 = base_time + timedelta(days=10)
        r3 = Run(
            status="completed",
            queued_at=t3,
            started_at=t3,
            finished_at=t3 + timedelta(seconds=20),
            config_snapshot={"tester": "live_stt", "config": {"model": "whisper-large-v3"}},
            target_snapshot=t_stt.to_snapshot(),
        )
        # Run 4: failed LLM run (5 days ago)
        t4 = base_time + timedelta(days=15)
        r4 = Run(
            status="failed",
            queued_at=t4,
            started_at=t4,
            finished_at=t4 + timedelta(seconds=5),
            config_snapshot={"tester": "llm", "config": {"model": "Qwen2.5-7B"}},
            target_snapshot=t_llm.to_snapshot(),
        )
        # Run 5: active (running) run
        t5 = base_time + timedelta(days=19)
        r5 = Run(
            status="running",
            queued_at=t5,
            started_at=t5,
            config_snapshot={"tester": "llm", "config": {"model": "Qwen2.5-72B"}},
            target_snapshot=t_llm.to_snapshot(),
        )
        # Run 6: queued run
        r6 = Run(
            status="queued",
            queued_at=t5,
            config_snapshot={"tester": "llm", "config": {"model": "Qwen2.5-72B"}},
            target_snapshot=t_llm.to_snapshot(),
        )

        db.add_all([r1, r2, r3, r4, r5, r6])
        db.flush()

        res1 = Result(
            run_id=r1.id,
            tester="llm",
            status="pass",
            summary="LLM v1 pass",
            metrics={
                "ttft_ms": {"value": 110.0, "unit": "ms", "better": "lower"},
                "tps": {"value": 90.0, "unit": "tok/s", "better": "higher"},
            },
        )
        res2 = Result(
            run_id=r2.id,
            tester="llm",
            status="pass",
            summary="LLM v2 pass",
            metrics={
                "ttft_ms": {"value": 95.0, "unit": "ms", "better": "lower"},
                "tps": {"value": 105.0, "unit": "tok/s", "better": "higher"},
            },
        )
        res3 = Result(
            run_id=r3.id,
            tester="live_stt",
            status="pass",
            summary="STT pass",
            metrics={
                "wer": {"value": 4.2, "unit": "%", "better": "lower"},
            },
        )
        res4 = Result(
            run_id=r4.id,
            tester="llm",
            status="fail",
            summary="LLM fail",
            metrics={"ttft_ms": {"value": 450.0, "unit": "ms", "better": "lower"}},
        )
        db.add_all([res1, res2, res3, res4])
        db.flush()

        for run_obj, res_obj in ((r1, res1), (r2, res2), (r3, res3)):
            r_dir = artifacts_dir / str(run_obj.id)
            r_dir.mkdir(parents=True, exist_ok=True)
            log_path = r_dir / "run.log"
            log_path.write_text(f"Log for run {run_obj.id}\n" * 5, encoding="utf-8")
            db.add(
                Artifact(
                    result_id=res_obj.id,
                    name="run.log",
                    path=str(log_path),
                    mime="text/plain",
                    size_bytes=log_path.stat().st_size,
                )
            )

        db.commit()
        return {
            "r1": r1.id,
            "r2": r2.id,
            "r3": r3.id,
            "r4": r4.id,
            "r5": r5.id,
            "r6": r6.id,
        }
    finally:
        db.close()


def test_baseline_crud_uniqueness_and_move_history(app_and_client):
    client, session_factory, artifacts_dir = app_and_client
    ids = _seed_baseline_test_data(session_factory, artifacts_dir)

    # 1. Queued and running runs return 422
    res_running = client.post(
        f"/api/runs/{ids['r5']}/baseline",
        json={"name": "active-baseline"},
    )
    assert res_running.status_code == 422

    res_queued = client.post(
        f"/api/runs/{ids['r6']}/baseline",
        json={"name": "queued-baseline"},
    )
    assert res_queued.status_code == 422

    # 2. Save completed run r1 as baseline
    res_create = client.post(
        f"/api/runs/{ids['r1']}/baseline",
        json={"name": "Prod-LLM-Golden", "description": "Initial Qwen 72B golden"},
    )
    assert res_create.status_code == 201
    b1 = res_create.json()
    assert b1["name"] == "Prod-LLM-Golden"
    assert b1["run_id"] == ids["r1"]
    assert b1["description"] == "Initial Qwen 72B golden"
    assert b1["tester"] == "llm"
    assert b1["target"] == "vllm-prod"
    assert b1["model"] == "Qwen2.5-72B"
    assert b1["status"] == "completed"
    assert "ttft_ms" in b1["metrics"]
    assert b1["history"] == []

    # 3. Case-insensitive uniqueness across ALL testers: trying to save r3 (live_stt) with same name returns 409 with current run_id
    res_dup = client.post(
        f"/api/runs/{ids['r3']}/baseline",
        json={"name": "prod-llm-golden"},
    )
    assert res_dup.status_code == 409
    dup_body = res_dup.json()
    assert dup_body["run_id"] == ids["r1"]
    assert dup_body["current_run_id"] == ids["r1"]

    # 4. Move baseline to r2 with move=True: points name at r2 and appends r1 to history
    res_move = client.post(
        f"/api/runs/{ids['r2']}/baseline",
        json={"name": "Prod-LLM-Golden", "description": "Updated to r2", "move": True},
    )
    assert res_move.status_code == 201
    moved = res_move.json()
    assert moved["id"] == b1["id"]
    assert moved["run_id"] == ids["r2"]
    assert moved["description"] == "Updated to r2"
    assert len(moved["history"]) == 1
    assert moved["history"][0]["run_id"] == ids["r1"]
    assert moved["history"][0]["from"] is not None
    assert moved["history"][0]["to"] is not None

    # 5. GET /api/baselines/{id} returns detail including history
    res_detail = client.get(f"/api/baselines/{b1['id']}")
    assert res_detail.status_code == 200
    assert res_detail.json()["run_id"] == ids["r2"]
    assert len(res_detail.json()["history"]) == 1

    # 6. Run r1 detail shows it in previous_baselines, and r2 detail shows it in baselines
    r1_detail = client.get(f"/api/runs/{ids['r1']}").json()
    assert r1_detail["baselines"] == []
    assert any(pb["name"] == "Prod-LLM-Golden" for pb in r1_detail["previous_baselines"])

    r2_detail = client.get(f"/api/runs/{ids['r2']}").json()
    assert r2_detail["baselines"] == [{"id": b1["id"], "name": "Prod-LLM-Golden"}]

    # 7. A single run can have multiple baseline names, and failed runs can be baselined
    res_second = client.post(
        f"/api/runs/{ids['r2']}/baseline",
        json={"name": "Release-2026.10"},
    )
    assert res_second.status_code == 201
    r2_after = client.get(f"/api/runs/{ids['r2']}").json()
    assert len(r2_after["baselines"]) == 2

    res_failed_baseline = client.post(
        f"/api/runs/{ids['r4']}/baseline",
        json={"name": "Known-Failure-Repro"},
    )
    assert res_failed_baseline.status_code == 201
    assert res_failed_baseline.json()["status"] == "failed"

    # 8. PATCH /api/baselines/{id} renames and enforces case-insensitive uniqueness
    res_patch_conflict = client.patch(
        f"/api/baselines/{b1['id']}",
        json={"name": "release-2026.10"},
    )
    assert res_patch_conflict.status_code == 409

    res_patch_ok = client.patch(
        f"/api/baselines/{b1['id']}",
        json={"name": "Prod-LLM-Current", "description": "Renamed baseline"},
    )
    assert res_patch_ok.status_code == 200
    assert res_patch_ok.json()["name"] == "Prod-LLM-Current"
    assert res_patch_ok.json()["description"] == "Renamed baseline"


def test_list_baselines_filters_and_run_deletion_protection(app_and_client):
    client, session_factory, artifacts_dir = app_and_client
    ids = _seed_baseline_test_data(session_factory, artifacts_dir)

    b_llm = client.post(
        f"/api/runs/{ids['r1']}/baseline",
        json={"name": "LLM-Qwen-Baseline", "description": "Production LLM"},
    ).json()
    b_stt = client.post(
        f"/api/runs/{ids['r3']}/baseline",
        json={"name": "STT-Whisper-Baseline", "description": "Production STT"},
    ).json()

    # List all baselines
    all_b = client.get("/api/baselines").json()
    assert len(all_b) == 2

    # Filter by q
    q_res = client.get("/api/baselines?q=whisper").json()
    assert [b["id"] for b in q_res] == [b_stt["id"]]

    # Filter by tester
    tester_res = client.get("/api/baselines?tester=llm").json()
    assert [b["id"] for b in tester_res] == [b_llm["id"]]

    # Filter by target
    target_res = client.get("/api/baselines?target=vllm-prod").json()
    assert [b["id"] for b in target_res] == [b_llm["id"]]

    # Filter by model
    model_res = client.get("/api/baselines?model=whisper-large-v3").json()
    assert [b["id"] for b in model_res] == [b_stt["id"]]

    # GET /api/runs?baseline=true returns only baselined runs
    baselined_runs = client.get("/api/runs?baseline=true").json()
    assert {r["id"] for r in baselined_runs} == {ids["r1"], ids["r3"]}
    for r in baselined_runs:
        assert len(r["baselines"]) == 1

    non_baselined_runs = client.get("/api/runs?baseline=false").json()
    assert ids["r1"] not in {r["id"] for r in non_baselined_runs}
    assert ids["r2"] in {r["id"] for r in non_baselined_runs}

    # DELETE /api/runs/{id} on a baselined run returns 409 "run is baseline 'X'; remove the baseline first"
    del_blocked = client.delete(f"/api/runs/{ids['r1']}")
    assert del_blocked.status_code == 409
    assert del_blocked.json()["detail"] == "run is baseline 'LLM-Qwen-Baseline'; remove the baseline first"

    # DELETE /api/baselines/{id} removes only the label, then run can be deleted
    del_label = client.delete(f"/api/baselines/{b_llm['id']}")
    assert del_label.status_code == 204

    # Run still exists after removing baseline label
    r1_still_there = client.get(f"/api/runs/{ids['r1']}")
    assert r1_still_there.status_code == 200
    assert r1_still_there.json()["baselines"] == []

    # Now deleting the run succeeds
    del_run_ok = client.delete(f"/api/runs/{ids['r1']}")
    assert del_run_ok.status_code == 204


def test_retention_protects_current_baselines_but_not_history(app_and_client):
    client, session_factory, artifacts_dir = app_and_client
    ids = _seed_baseline_test_data(session_factory, artifacts_dir)

    # Save r1 as baseline, then move it to r2 so r1 is only in history and r2 is current baseline
    client.post(f"/api/runs/{ids['r1']}/baseline", json={"name": "Golden-LLM"})
    client.post(f"/api/runs/{ids['r2']}/baseline", json={"name": "Golden-LLM", "move": True})
    # Also add a second baseline name on r2 to verify distinct run count in baseline_runs_protected
    client.post(f"/api/runs/{ids['r2']}/baseline", json={"name": "Golden-LLM-Alias"})

    # Check retention status reports 1 baseline run protected
    ret_status = client.get("/api/runs/retention").json()
    assert ret_status["stats"]["baseline_runs_protected"] == 1

    # 1. Test artifacts_only mode with max_runs=1:
    # Terminal runs are r4 (newest), r3, r2 (baseline), r1 (oldest, only in history).
    # r2 must be skipped even in artifacts_only mode; r1 and r3 should be cleaned.
    cleanup_artifacts = client.post(
        "/api/runs/retention/cleanup",
        json={"max_runs": 1, "artifacts_only": True, "dry_run": False},
    ).json()
    assert ids["r2"] not in cleanup_artifacts["cleaned_run_ids"]
    assert ids["r1"] in cleanup_artifacts["cleaned_run_ids"]
    assert (artifacts_dir / str(ids["r2"]) / "run.log").exists()
    assert not (artifacts_dir / str(ids["r1"]) / "run.log").exists()

    # 2. Test full-delete mode with max_runs=1:
    # r2 (current baseline) is protected; r1 (only in history) and r3 are deleted.
    cleanup_full = client.post(
        "/api/runs/retention/cleanup",
        json={"max_runs": 1, "artifacts_only": False, "dry_run": False},
    ).json()
    assert ids["r2"] not in cleanup_full["deleted_run_ids"]
    assert ids["r1"] in cleanup_full["deleted_run_ids"]
    assert client.get(f"/api/runs/{ids['r2']}").status_code == 200
    assert client.get(f"/api/runs/{ids['r1']}").status_code == 404


def test_compare_includes_baselines_and_resolves_baseline_names(app_and_client):
    client, session_factory, artifacts_dir = app_and_client
    ids = _seed_baseline_test_data(session_factory, artifacts_dir)

    client.post(f"/api/runs/{ids['r1']}/baseline", json={"name": "v1-golden"})

    # Compare by run_ids includes baselines metadata on each run
    cmp_res = client.get(f"/api/compare?run_ids={ids['r1']},{ids['r2']}").json()
    r1_meta = next(r for r in cmp_res["runs"] if r["run_id"] == ids["r1"])
    r2_meta = next(r for r in cmp_res["runs"] if r["run_id"] == ids["r2"])
    assert r1_meta["baselines"] == [{"id": 1, "name": "v1-golden"}]
    assert r2_meta["baselines"] == []

    # Compare using ?baselines=v1-golden&run_ids=<r2> resolves the baseline name and sets it as reference run
    cmp_by_name = client.get(f"/api/compare?baselines=v1-golden&run_ids={ids['r2']}").json()
    assert cmp_by_name["baseline_run_id"] == ids["r1"]
    assert cmp_by_name["run_ids"] == [ids["r1"], ids["r2"]]
