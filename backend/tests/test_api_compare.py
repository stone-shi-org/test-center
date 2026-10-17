"""Tests for the Compare API (/api/compare) and metric/case alignment & delta engine."""

from __future__ import annotations

import pytest

from backend.app.api.compare import compute_case_status_diff, compute_metric_delta
from backend.app.db.base import utc_now
from backend.app.db.models.artifact import Artifact
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run


def _seed_run_with_result(
    session_factory,
    *,
    tester: str = "llm",
    run_status: str = "completed",
    result_status: str = "pass",
    target_snapshot: dict | None = None,
    config_snapshot: dict | None = None,
    metrics: dict | None = None,
    cases: list[dict] | None = None,
    artifacts: list[dict] | None = None,
    summary: str | None = None,
) -> tuple[int, int]:
    """Helper to seed a Run, Result, CaseResults, and Artifacts in the test database."""
    db = session_factory()
    try:
        now = utc_now()
        run = Run(
            status=run_status,
            queued_at=now,
            started_at=now,
            finished_at=now,
            config_snapshot=config_snapshot or {"tester": tester, "config": {}},
            target_snapshot=target_snapshot or {"name": "default-target", "base_url": "http://localhost:8000"},
        )
        db.add(run)
        db.flush()

        res = Result(
            run_id=run.id,
            tester=tester,
            status=result_status,
            metrics=metrics or {},
            summary=summary or f"{tester} result",
        )
        db.add(res)
        db.flush()

        for c in cases or []:
            db.add(
                CaseResult(
                    result_id=res.id,
                    suite=c.get("suite", "default"),
                    case_id=c["case_id"],
                    status=c.get("status", "pass"),
                    metrics=c.get("metrics", {}),
                    detail=c.get("detail", {}),
                )
            )

        for a in artifacts or []:
            db.add(
                Artifact(
                    result_id=res.id,
                    name=a["name"],
                    path=a.get("path", f"/tmp/{a['name']}"),
                    mime=a.get("mime", "text/plain"),
                    size_bytes=a.get("size_bytes", 128),
                )
            )

        db.commit()
        return run.id, res.id
    finally:
        db.close()


def test_unit_compute_metric_delta_and_case_status_diff():
    """Unit test metric delta calculations, better direction, thresholds, and case status diffs."""
    # better="lower": increase is regression, decrease is improvement
    d_reg = compute_metric_delta(100.0, 125.0, better="lower", threshold=0.0)
    assert d_reg["abs_delta"] == 25.0
    assert d_reg["pct_delta"] == 25.0
    assert d_reg["regression"] is True
    assert d_reg["improvement"] is False
    assert d_reg["flag"] == "regression"

    d_imp = compute_metric_delta(100.0, 80.0, better="lower", threshold=0.0)
    assert d_imp["abs_delta"] == -20.0
    assert d_imp["pct_delta"] == -20.0
    assert d_imp["regression"] is False
    assert d_imp["improvement"] is True
    assert d_imp["flag"] == "improvement"

    # better="higher": increase is improvement, decrease is regression
    d_high_imp = compute_metric_delta(50.0, 60.0, better="higher", threshold=0.0)
    assert d_high_imp["abs_delta"] == 10.0
    assert d_high_imp["pct_delta"] == 20.0
    assert d_high_imp["improvement"] is True
    assert d_high_imp["regression"] is False
    assert d_high_imp["flag"] == "improvement"

    d_high_reg = compute_metric_delta(50.0, 40.0, better="higher", threshold=0.0)
    assert d_high_reg["abs_delta"] == -10.0
    assert d_high_reg["pct_delta"] == -20.0
    assert d_high_reg["regression"] is True
    assert d_high_reg["improvement"] is False
    assert d_high_reg["flag"] == "regression"

    # Configurable threshold suppresses small fluctuations
    d_within = compute_metric_delta(100.0, 103.0, better="lower", threshold=5.0)
    assert d_within["abs_delta"] == 3.0
    assert d_within["pct_delta"] == 3.0
    assert d_within["regression"] is False
    assert d_within["improvement"] is False
    assert d_within["flag"] == "unchanged"

    d_at_thresh = compute_metric_delta(100.0, 105.0, better="lower", threshold=5.0)
    assert d_at_thresh["regression"] is True
    assert d_at_thresh["flag"] == "regression"

    # Zero baseline handling
    d_zero_same = compute_metric_delta(0.0, 0.0, better="lower", threshold=0.0)
    assert d_zero_same["abs_delta"] == 0.0
    assert d_zero_same["pct_delta"] == 0.0
    assert d_zero_same["flag"] == "unchanged"

    d_zero_reg = compute_metric_delta(0.0, 0.05, better="lower", threshold=5.0)
    assert d_zero_reg["abs_delta"] == 0.05
    assert d_zero_reg["pct_delta"] is None
    assert d_zero_reg["regression"] is True
    assert d_zero_reg["flag"] == "regression"

    # Missing on one side
    d_missing = compute_metric_delta(100.0, None, better="lower", threshold=0.0)
    assert d_missing["abs_delta"] is None
    assert d_missing["pct_delta"] is None
    assert d_missing["regression"] is False
    assert d_missing["improvement"] is False
    assert d_missing["flag"] == "missing"

    # Case status transitions
    assert compute_case_status_diff("pass", "fail")["status_regression"] is True
    assert compute_case_status_diff("pass", "error")["status_regression"] is True
    assert compute_case_status_diff("fail", "error")["status_regression"] is True
    assert compute_case_status_diff("fail", "pass")["status_improvement"] is True
    assert compute_case_status_diff("error", "pass")["status_improvement"] is True
    assert compute_case_status_diff("pass", "pass")["status_changed"] is False
    assert compute_case_status_diff(None, "pass")["status_change"] == "added"
    assert compute_case_status_diff("pass", None)["status_change"] == "missing"


def test_compare_two_runs_metrics_and_cases(app_and_client):
    """Compare 2 runs across different targets/models of the same tester via GET and POST."""
    client, session_factory, _ = app_and_client

    run1_id, res1_id = _seed_run_with_result(
        session_factory,
        tester="llm",
        result_status="pass",
        target_snapshot={
            "id": 1,
            "name": "vllm-a100",
            "kind": "openai-compatible",
            "base_url": "http://target-a:8000/v1",
            "extra": {"model": "Qwen2.5-7B-Instruct"},
        },
        config_snapshot={
            "tester": "llm",
            "suites": ["coding", "reasoning"],
            "config": {"model": "Qwen2.5-7B-Instruct", "temperature": 0.05},
        },
        metrics={
            "ttft_p50_ms": {"value": 100.0, "unit": "ms", "better": "lower"},
            "latency_p95_ms": {"value": 500.0, "unit": "ms", "better": "lower"},
            "tps": {"value": 80.0, "unit": "tokens/s", "better": "higher"},
            "pass_rate": {"value": 100.0, "unit": "%", "better": "higher"},
            "prompt_tokens": {"value": 250, "unit": "tokens"},
            "baseline_only_metric": {"value": 12.5, "unit": "ms", "better": "lower"},
        },
        cases=[
            {
                "suite": "coding",
                "case_id": "coding_lru_cache",
                "status": "pass",
                "metrics": {
                    "ttft_ms": {"value": 95.0, "unit": "ms", "better": "lower"},
                    "tps": {"value": 85.0, "unit": "tokens/s", "better": "higher"},
                },
                "detail": {"passed": True, "eval_message": "All unit tests passed"},
            },
            {
                "suite": "reasoning",
                "case_id": "reasoning_riddle",
                "status": "fail",
                "metrics": {
                    "ttft_ms": {"value": 105.0, "unit": "ms", "better": "lower"},
                    "tps": {"value": 75.0, "unit": "tokens/s", "better": "higher"},
                },
                "detail": {"passed": False, "eval_message": "Wrong final answer"},
            },
            {
                "suite": "coding",
                "case_id": "coding_removed_in_run2",
                "status": "pass",
                "metrics": {"ttft_ms": {"value": 90.0, "unit": "ms", "better": "lower"}},
                "detail": {"passed": True},
            },
        ],
        artifacts=[{"name": "llm_report.txt"}],
    )

    run2_id, res2_id = _seed_run_with_result(
        session_factory,
        tester="llm",
        result_status="fail",
        target_snapshot={
            "id": 2,
            "name": "vllm-h100",
            "kind": "openai-compatible",
            "base_url": "http://target-b:8000/v1",
            "extra": {"model": "Qwen2.5-32B-Instruct"},
        },
        config_snapshot={
            "tester": "llm",
            "suites": ["coding", "reasoning"],
            "config": {"model": "Qwen2.5-32B-Instruct", "temperature": 0.1},
        },
        metrics={
            "ttft_p50_ms": {"value": 102.0, "unit": "ms", "better": "lower"},  # +2% (slight slowdown)
            "latency_p95_ms": {"value": 650.0, "unit": "ms", "better": "lower"},  # +30% (regression)
            "tps": {"value": 100.0, "unit": "tokens/s", "better": "higher"},  # +25% (improvement)
            "pass_rate": {"value": 50.0, "unit": "%", "better": "higher"},  # -50% (regression)
            "prompt_tokens": {"value": 250, "unit": "tokens"},  # unchanged
            "run2_only_metric": {"value": 4.8, "unit": "score", "better": "higher"},
        },
        cases=[
            {
                "suite": "coding",
                "case_id": "coding_lru_cache",
                "status": "fail",  # Regressed from pass -> fail
                "metrics": {
                    "ttft_ms": {"value": 133.0, "unit": "ms", "better": "lower"},  # +40% regression
                    "tps": {"value": 90.0, "unit": "tokens/s", "better": "higher"},
                    "judge_score": {"value": 2, "unit": "score", "better": "higher"},  # Only on run2
                },
                "detail": {"passed": False, "eval_message": "AssertionError"},
            },
            {
                "suite": "reasoning",
                "case_id": "reasoning_riddle",
                "status": "pass",  # Improved from fail -> pass
                "metrics": {
                    "ttft_ms": {"value": 84.0, "unit": "ms", "better": "lower"},  # -20% improvement
                    "tps": {"value": 90.0, "unit": "tokens/s", "better": "higher"},  # +20% improvement
                },
                "detail": {"passed": True, "eval_message": "Solved"},
            },
            {
                "suite": "reasoning",
                "case_id": "reasoning_added_in_run2",
                "status": "pass",
                "metrics": {"ttft_ms": {"value": 88.0, "unit": "ms", "better": "lower"}},
                "detail": {"passed": True},
            },
        ],
        artifacts=[{"name": "llm_report.txt"}],
    )

    # 1. GET /api/compare with comma-separated run_ids and default threshold=0
    resp = client.get(f"/api/compare?run_ids={run1_id},{run2_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["tester"] == "llm"
    assert data["baseline_run_id"] == run1_id
    assert data["baseline_result_id"] == res1_id
    assert data["run_ids"] == [run1_id, run2_id]
    assert data["result_ids"] == [res1_id, res2_id]

    # Verify target/model metadata & config_diff
    assert len(data["runs"]) == 2
    assert data["runs"][0]["target_name"] == "vllm-a100"
    assert data["runs"][0]["model"] == "Qwen2.5-7B-Instruct"
    assert data["runs"][0]["is_baseline"] is True
    assert data["runs"][1]["target_name"] == "vllm-h100"
    assert data["runs"][1]["model"] == "Qwen2.5-32B-Instruct"
    assert data["runs"][1]["is_baseline"] is False

    assert data["config_diff"]["targets_differ"] is True
    assert data["config_diff"]["models_differ"] is True
    assert "model" in data["config_diff"]["config_keys_changed"]
    assert "temperature" in data["config_diff"]["config_keys_changed"]

    # Verify aligned metrics
    mmap = data["metric_map"]
    assert set(mmap.keys()) == {
        "ttft_p50_ms",
        "latency_p95_ms",
        "tps",
        "pass_rate",
        "prompt_tokens",
        "baseline_only_metric",
        "run2_only_metric",
    }

    # latency_p95_ms: 500 -> 650 (+150, +30%, regression because better=lower)
    lat_m = mmap["latency_p95_ms"]
    assert lat_m["baseline_value"] == 500.0
    assert lat_m["target_value"] == 650.0
    assert lat_m["abs_delta"] == 150.0
    assert lat_m["pct_delta"] == 30.0
    assert lat_m["regression"] is True
    assert lat_m["improvement"] is False
    assert lat_m["flag"] == "regression"

    # tps: 80 -> 100 (+20, +25%, improvement because better=higher)
    tps_m = mmap["tps"]
    assert tps_m["baseline_value"] == 80.0
    assert tps_m["target_value"] == 100.0
    assert tps_m["abs_delta"] == 20.0
    assert tps_m["pct_delta"] == 25.0
    assert tps_m["regression"] is False
    assert tps_m["improvement"] is True
    assert tps_m["flag"] == "improvement"

    # Missing metrics on either side
    base_only = mmap["baseline_only_metric"]
    assert base_only["missing_in_some"] is True
    assert base_only["values"][0]["present"] is True
    assert base_only["values"][0]["value"] == 12.5
    assert base_only["values"][1]["present"] is False
    assert base_only["values"][1]["value"] is None
    assert base_only["values"][1]["abs_delta"] is None
    assert base_only["values"][1]["pct_delta"] is None
    assert base_only["values"][1]["flag"] == "missing"

    run2_only = mmap["run2_only_metric"]
    assert run2_only["missing_in_some"] is True
    assert run2_only["values"][0]["present"] is False
    assert run2_only["values"][0]["value"] is None
    assert run2_only["values"][1]["present"] is True
    assert run2_only["values"][1]["value"] == 4.8
    assert run2_only["values"][1]["abs_delta"] is None
    assert run2_only["values"][1]["pct_delta"] is None
    assert run2_only["values"][1]["flag"] == "missing"

    # Verify per-case diff
    cmap = data["case_map"]
    assert set(cmap.keys()) == {
        "coding_lru_cache",
        "reasoning_riddle",
        "coding_removed_in_run2",
        "reasoning_added_in_run2",
    }

    c_lru = cmap["coding_lru_cache"]
    assert c_lru["baseline_status"] == "pass"
    assert c_lru["target_status"] == "fail"
    assert c_lru["status_changed"] is True
    assert c_lru["status_change"] == "regression"
    assert c_lru["has_status_regression"] is True
    assert c_lru["has_metric_regression"] is True
    assert c_lru["metric_changes"]["ttft_ms"]["abs_delta"] == 38.0
    assert c_lru["metric_changes"]["ttft_ms"]["pct_delta"] == 40.0
    assert c_lru["metric_changes"]["ttft_ms"]["regression"] is True
    assert c_lru["metric_changes"]["judge_score"]["missing_in_some"] is True

    c_riddle = cmap["reasoning_riddle"]
    assert c_riddle["baseline_status"] == "fail"
    assert c_riddle["target_status"] == "pass"
    assert c_riddle["status_changed"] is True
    assert c_riddle["status_change"] == "improvement"
    assert c_riddle["has_status_improvement"] is True
    assert c_riddle["metric_changes"]["ttft_ms"]["abs_delta"] == -21.0
    assert c_riddle["metric_changes"]["ttft_ms"]["pct_delta"] == -20.0
    assert c_riddle["metric_changes"]["ttft_ms"]["improvement"] is True

    c_removed = cmap["coding_removed_in_run2"]
    assert c_removed["missing_in_some"] is True
    assert c_removed["status_change"] == "missing"
    assert c_removed["values"][0]["present"] is True
    assert c_removed["values"][1]["present"] is False

    c_added = cmap["reasoning_added_in_run2"]
    assert c_added["missing_in_some"] is True
    assert c_added["status_change"] == "added"
    assert c_added["values"][0]["present"] is False
    assert c_added["values"][1]["present"] is True

    # 2. Re-query with threshold=5.0 -> ttft_p50_ms (+2%) is now within threshold ('unchanged')
    resp_thresh = client.post(
        "/api/compare",
        json={
            "run_ids": [run1_id, run2_id],
            "threshold": 5.0,
        },
    )
    assert resp_thresh.status_code == 200
    thresh_data = resp_thresh.json()
    ttft_m = thresh_data["metric_map"]["ttft_p50_ms"]
    assert ttft_m["pct_delta"] == 2.0
    assert ttft_m["regression"] is False
    assert ttft_m["flag"] == "unchanged"
    # latency_p95_ms (+30%) still exceeds 5% threshold
    assert thresh_data["metric_map"]["latency_p95_ms"]["regression"] is True


def test_compare_three_runs_and_custom_baseline_and_result_ids(app_and_client):
    """Compare 3 runs, test selecting middle run as baseline, and test comparing by result_ids."""
    client, session_factory, _ = app_and_client

    r1, res1 = _seed_run_with_result(
        session_factory,
        tester="embedding",
        metrics={
            "recall_at_1": {"value": 80.0, "unit": "%", "better": "higher"},
            "latency_p50_ms": {"value": 20.0, "unit": "ms", "better": "lower"},
        },
        cases=[
            {"suite": "retrieval", "case_id": "q01", "status": "pass", "metrics": {"rank": {"value": 1, "better": "lower"}}},
        ],
    )
    r2, res2 = _seed_run_with_result(
        session_factory,
        tester="embedding",
        metrics={
            "recall_at_1": {"value": 85.0, "unit": "%", "better": "higher"},
            "latency_p50_ms": {"value": 25.0, "unit": "ms", "better": "lower"},
        },
        cases=[
            {"suite": "retrieval", "case_id": "q01", "status": "pass", "metrics": {"rank": {"value": 1, "better": "lower"}}},
        ],
    )
    r3, res3 = _seed_run_with_result(
        session_factory,
        tester="embedding",
        metrics={
            "recall_at_1": {"value": 90.0, "unit": "%", "better": "higher"},
            "latency_p50_ms": {"value": 15.0, "unit": "ms", "better": "lower"},
        },
        cases=[
            {"suite": "retrieval", "case_id": "q01", "status": "fail", "metrics": {"rank": {"value": 3, "better": "lower"}}},
        ],
    )

    # Compare 3 runs using result_ids and setting res2 as baseline
    resp = client.get(
        f"/api/compare?result_ids={res1}&result_ids={res2}&result_ids={res3}&baseline_result_id={res2}"
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["baseline_run_id"] == r2
    assert data["baseline_result_id"] == res2
    assert len(data["runs"]) == 3

    recall_row = data["metric_map"]["recall_at_1"]
    assert recall_row["baseline_value"] == 85.0
    # Run 1 vs Baseline (Run 2): 80 vs 85 -> -5.0 (-5.8824%), regression
    cell_r1 = recall_row["by_run_id"][str(r1)]
    assert cell_r1["abs_delta"] == -5.0
    assert cell_r1["regression"] is True
    # Run 2 is baseline
    cell_r2 = recall_row["by_run_id"][str(r2)]
    assert cell_r2["is_baseline"] is True
    assert cell_r2["abs_delta"] == 0.0
    assert cell_r2["flag"] == "baseline"
    # Run 3 vs Baseline (Run 2): 90 vs 85 -> +5.0 (+5.8824%), improvement
    cell_r3 = recall_row["by_run_id"][str(r3)]
    assert cell_r3["abs_delta"] == 5.0
    assert cell_r3["improvement"] is True

    # Verify case q01 across 3 runs
    q01_row = data["case_map"]["q01"]
    assert q01_row["by_run_id"][str(r1)]["status_change"] == "unchanged"
    assert q01_row["by_run_id"][str(r2)]["status_change"] == "baseline"
    assert q01_row["by_run_id"][str(r3)]["status_change"] == "regression"
    assert q01_row["by_run_id"][str(r3)]["metric_deltas"]["rank"]["abs_delta"] == 2.0
    assert q01_row["by_run_id"][str(r3)]["metric_deltas"]["rank"]["regression"] is True


def test_compare_cross_model_case_alignment_and_filters(app_and_client):
    """Verify cross-model case alignment for live-stt and changed_only / regressed_only / suite filters."""
    client, session_factory, _ = app_and_client

    r1, _ = _seed_run_with_result(
        session_factory,
        tester="live-stt",
        target_snapshot={"name": "stt-node-1", "base_url": "http://stt1:8000"},
        config_snapshot={"tester": "live-stt", "config": {"transcription_model": "whisper-large-v3"}},
        metrics={
            "latency_mean_ms": {"value": 120.0, "unit": "ms", "better": "lower"},
            "pass_rate": {"value": 100.0, "unit": "%", "better": "higher"},
        },
        cases=[
            {
                "suite": "transcriptions_json",
                "case_id": "transcriptions_json:whisper-large-v3:json",
                "status": "pass",
                "metrics": {"latency_ms": {"value": 120.0, "unit": "ms", "better": "lower"}},
                "detail": {"model": "whisper-large-v3", "response_format": "json"},
            },
            {
                "suite": "admin",
                "case_id": "admin:api_health",
                "status": "pass",
                "metrics": {"latency_ms": {"value": 10.0, "unit": "ms", "better": "lower"}},
                "detail": {"endpoint": "/api/health"},
            },
        ],
    )

    r2, _ = _seed_run_with_result(
        session_factory,
        tester="live-stt",
        target_snapshot={"name": "stt-node-1", "base_url": "http://stt1:8000"},
        config_snapshot={"tester": "live-stt", "config": {"transcription_model": "parakeet-tdt-0.6b-v2"}},
        metrics={
            "latency_mean_ms": {"value": 60.0, "unit": "ms", "better": "lower"},
            "pass_rate": {"value": 100.0, "unit": "%", "better": "higher"},
        },
        cases=[
            {
                "suite": "transcriptions_json",
                "case_id": "transcriptions_json:parakeet-tdt-0.6b-v2:json",
                "status": "pass",
                "metrics": {"latency_ms": {"value": 60.0, "unit": "ms", "better": "lower"}},
                "detail": {"model": "parakeet-tdt-0.6b-v2", "response_format": "json"},
            },
            {
                "suite": "admin",
                "case_id": "admin:api_health",
                "status": "pass",
                "metrics": {"latency_ms": {"value": 10.0, "unit": "ms", "better": "lower"}},
                "detail": {"endpoint": "/api/health"},
            },
        ],
    )

    # Compare the two single-model live-stt runs
    resp = client.get(f"/api/compare?run_ids={r1},{r2}")
    assert resp.status_code == 200
    data = resp.json()

    # The model-parameterized case IDs should align under 'transcriptions_json:json'
    assert "transcriptions_json:json" in data["case_map"]
    aligned_case = data["case_map"]["transcriptions_json:json"]
    assert aligned_case["values"][0]["original_case_id"] == "transcriptions_json:whisper-large-v3:json"
    assert aligned_case["values"][1]["original_case_id"] == "transcriptions_json:parakeet-tdt-0.6b-v2:json"
    assert aligned_case["metric_changes"]["latency_ms"]["abs_delta"] == -60.0
    assert aligned_case["metric_changes"]["latency_ms"]["pct_delta"] == -50.0
    assert aligned_case["metric_changes"]["latency_ms"]["improvement"] is True

    # Test changed_only=True -> unchanged metric 'pass_rate' and unchanged case 'admin:api_health' are excluded
    resp_changed = client.get(f"/api/compare?run_ids={r1},{r2}&changed_only=true")
    assert resp_changed.status_code == 200
    changed_data = resp_changed.json()
    assert "latency_mean_ms" in changed_data["metric_map"]
    assert "pass_rate" not in changed_data["metric_map"]
    assert "transcriptions_json:json" in changed_data["case_map"]
    assert "admin:api_health" not in changed_data["case_map"]

    # Test regressed_only=True -> no regressions in r2 vs r1
    resp_reg = client.get(f"/api/compare?run_ids={r1},{r2}&regressed_only=true")
    assert resp_reg.status_code == 200
    reg_data = resp_reg.json()
    assert len(reg_data["metrics"]) == 0
    assert len(reg_data["cases"]) == 0

    # Test suite filter
    resp_suite = client.get(f"/api/compare?run_ids={r1},{r2}&suite=admin")
    assert resp_suite.status_code == 200
    suite_data = resp_suite.json()
    assert list(suite_data["case_map"].keys()) == ["admin:api_health"]


def test_compare_validation_errors_and_secret_scrubbing(app_and_client):
    """Verify HTTP 400/404 validation errors and guarantee secret fields are scrubbed."""
    client, session_factory, _ = app_and_client

    r_llm, _ = _seed_run_with_result(
        session_factory,
        tester="llm",
        config_snapshot={"tester": "llm", "config": {"api_key": "sk-secret-do-not-leak-123"}},
        metrics={"tps": {"value": 50.0, "better": "higher"}},
    )
    r_llm2, _ = _seed_run_with_result(
        session_factory,
        tester="llm",
        metrics={"tps": {"value": 60.0, "better": "higher"}},
    )
    r_emb, _ = _seed_run_with_result(
        session_factory,
        tester="embedding",
        metrics={"mrr": {"value": 0.9, "better": "higher"}},
    )

    # 1. Missing or < 2 runs
    assert client.get("/api/compare").status_code == 400
    assert client.get(f"/api/compare?run_ids={r_llm}").status_code == 400
    assert client.get(f"/api/compare?run_ids={r_llm},{r_llm}").status_code == 400

    # 2. Invalid integer in run_ids
    assert client.get("/api/compare?run_ids=1,not_an_int").status_code == 400

    # 3. Non-existent run_id or result_id
    assert client.get(f"/api/compare?run_ids={r_llm},999999").status_code == 404
    assert client.post("/api/compare", json={"result_ids": [1, 999999]}).status_code == 404

    # 4. Different testers rejected by default
    res_diff = client.get(f"/api/compare?run_ids={r_llm},{r_emb}")
    assert res_diff.status_code == 400
    assert "different testers" in res_diff.json()["detail"].lower()

    # 5. Secret scrubbing in compare response
    res_ok = client.get(f"/api/compare?run_ids={r_llm},{r_llm2}")
    assert res_ok.status_code == 200
    assert "sk-secret-do-not-leak-123" not in res_ok.text


def test_compare_multi_result_runs_and_per_metric_thresholds(app_and_client):
    """Verify comparing runs that contain multiple results (e.g. multi-tester job runs) and per-metric thresholds."""
    client, session_factory, _ = app_and_client

    db = session_factory()
    try:
        now = utc_now()
        run_a = Run(status="completed", queued_at=now, started_at=now, finished_at=now)
        run_b = Run(status="completed", queued_at=now, started_at=now, finished_at=now)
        db.add_all([run_a, run_b])
        db.flush()

        # Each run has both an 'embedding' and an 'llm' result
        db.add_all([
            Result(
                run_id=run_a.id,
                tester="embedding",
                status="pass",
                metrics={
                    "mrr": {"value": 0.80, "better": "higher"},
                    "latency_p50_ms": {"value": 100.0, "unit": "ms", "better": "lower"},
                },
            ),
            Result(
                run_id=run_a.id,
                tester="llm",
                status="pass",
                metrics={"tps": {"value": 50.0, "unit": "tokens/s", "better": "higher"}},
            ),
            Result(
                run_id=run_b.id,
                tester="embedding",
                status="pass",
                metrics={
                    "mrr": {"value": 0.76, "better": "higher"},  # -5% change
                    "latency_p50_ms": {"value": 110.0, "unit": "ms", "better": "lower"},  # +10% change
                },
            ),
            Result(
                run_id=run_b.id,
                tester="llm",
                status="pass",
                metrics={"tps": {"value": 75.0, "unit": "tokens/s", "better": "higher"}},
            ),
        ])
        # Also create an empty run with no results
        empty_run = Run(status="queued", queued_at=now)
        db.add(empty_run)
        db.commit()
        run_a_id, run_b_id, empty_run_id = run_a.id, run_b.id, empty_run.id
    finally:
        db.close()

    # Empty run fails with 400
    res_empty = client.get(f"/api/compare?run_ids={run_a_id},{empty_run_id}")
    assert res_empty.status_code == 400
    assert "no results" in res_empty.json()["detail"].lower()

    # Select 'llm' results from multi-result runs via tester parameter
    res_llm = client.get(f"/api/compare?run_ids={run_a_id},{run_b_id}&tester=llm")
    assert res_llm.status_code == 200
    llm_data = res_llm.json()
    assert llm_data["tester"] == "llm"
    assert llm_data["metric_map"]["tps"]["pct_delta"] == 50.0
    assert llm_data["metric_map"]["tps"]["improvement"] is True

    # Select 'embedding' results with per-metric thresholds:
    # global threshold=2.0%, but latency_p50_ms override=15.0%
    res_emb = client.post(
        "/api/compare",
        json={
            "run_ids": [run_a_id, run_b_id],
            "tester": "embedding",
            "threshold": 2.0,
            "metric_thresholds": {"latency_p50_ms": 15.0},
        },
    )
    assert res_emb.status_code == 200
    emb_data = res_emb.json()
    assert emb_data["tester"] == "embedding"
    # mrr (-5%) exceeds global 2% threshold -> regression
    assert emb_data["metric_map"]["mrr"]["regression"] is True
    # latency_p50_ms (+10%) is below its 15% per-metric threshold -> unchanged
    assert emb_data["metric_map"]["latency_p50_ms"]["regression"] is False
    assert emb_data["metric_map"]["latency_p50_ms"]["flag"] == "unchanged"


def test_compare_end_to_end_executed_runs(app_and_client):
    """Execute 2 real runs via RunWorker across different targets and compare them via /api/compare."""
    import asyncio
    from backend.app.runner.worker import get_worker

    client, _, _ = app_and_client

    r1 = client.post(
        "/api/runs",
        json={
            "tester": "dummy",
            "suites": ["smoke", "load"],
            "config": {"duration_seconds": 0.05, "step_delay": 0.02, "fail_case": False},
            "target": {"name": "staging-target", "base_url": "http://staging:8000", "extra": {"model": "v1"}},
        },
    ).json()["id"]

    r2 = client.post(
        "/api/runs",
        json={
            "tester": "dummy",
            "suites": ["smoke", "load"],
            "config": {"duration_seconds": 0.05, "step_delay": 0.02, "fail_case": True},
            "target": {"name": "canary-target", "base_url": "http://canary:8000", "extra": {"model": "v2"}},
        },
    ).json()["id"]

    worker = get_worker()
    loop = asyncio.new_event_loop()
    assert loop.run_until_complete(worker.poll_and_execute_once()) is True
    assert loop.run_until_complete(worker.poll_and_execute_once()) is True
    loop.close()

    resp = client.get(f"/api/compare?run_ids={r1},{r2}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["tester"] == "dummy"
    assert data["summary"]["has_regression"] is True
    assert data["metric_map"]["success_rate"]["baseline_value"] == 100.0
    assert data["metric_map"]["success_rate"]["target_value"] == 50.0
    assert data["metric_map"]["success_rate"]["abs_delta"] == -50.0
    assert data["metric_map"]["success_rate"]["pct_delta"] == -50.0
    assert data["metric_map"]["success_rate"]["regression"] is True

    # case_load_001 regressed from pass -> fail in r2
    assert data["case_map"]["case_load_001"]["baseline_status"] == "pass"
    assert data["case_map"]["case_load_001"]["target_status"] == "fail"
    assert data["case_map"]["case_load_001"]["status_regression"] is True
    assert data["case_map"]["case_smoke_001"]["status_changed"] is False

