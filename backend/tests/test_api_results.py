"""Tests for the results, test cases, and artifact retrieval APIs."""

import asyncio
from pathlib import Path
import pytest
from backend.app.runner.worker import get_worker


def test_results_and_artifacts_api(app_and_client):
    client, session_factory, artifacts_dir = app_and_client

    # 1. Trigger a run that completes with results, cases, and artifacts
    run_payload = {
        "tester": "dummy",
        "suites": ["smoke"],
        "config": {"duration_seconds": 0.1, "step_delay": 0.05},
    }
    res_run = client.post("/api/runs", json=run_payload)
    assert res_run.status_code == 201
    run_id = res_run.json()["id"]

    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True

    # 2. Query /api/runs/{run_id}/results
    res_run_results = client.get(f"/api/runs/{run_id}/results")
    assert res_run_results.status_code == 200
    run_results = res_run_results.json()
    assert len(run_results) == 1
    result_id = run_results[0]["id"]
    assert run_results[0]["status"] == "pass"

    # 3. Query /api/results
    res_list = client.get("/api/results")
    assert res_list.status_code == 200
    all_results = res_list.json()
    assert any(r["id"] == result_id for r in all_results)

    # Filter by tester
    res_by_tester = client.get("/api/results?tester=dummy")
    assert res_by_tester.status_code == 200
    assert any(r["id"] == result_id for r in res_by_tester.json())

    # Filter by status
    res_by_status = client.get("/api/results?status=pass")
    assert res_by_status.status_code == 200
    assert any(r["id"] == result_id for r in res_by_status.json())

    # Filter by run_id
    res_by_run = client.get(f"/api/results?run_id={run_id}")
    assert res_by_run.status_code == 200
    assert len(res_by_run.json()) == 1
    assert res_by_run.json()[0]["id"] == result_id
    assert res_by_run.json()[0]["case_count"] >= 1
    assert res_by_run.json()[0]["artifact_count"] >= 2

    # 4. Get result detail
    res_detail = client.get(f"/api/results/{result_id}")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["id"] == result_id
    assert detail["run_id"] == run_id
    assert len(detail["cases"]) >= 1
    assert len(detail["artifacts"]) >= 2

    # Result not found
    res_404 = client.get("/api/results/999999")
    assert res_404.status_code == 404

    # 5. List cases for result
    res_cases = client.get(f"/api/results/{result_id}/cases")
    assert res_cases.status_code == 200
    cases = res_cases.json()
    assert len(cases) >= 1
    assert cases[0]["suite"] == "smoke"
    assert cases[0]["status"] == "pass"

    # Filter cases by suite
    res_cases_filter = client.get(f"/api/results/{result_id}/cases?suite=smoke")
    assert res_cases_filter.status_code == 200
    assert len(res_cases_filter.json()) >= 1

    # Filter cases by status
    res_cases_fail = client.get(f"/api/results/{result_id}/cases?status=fail")
    assert res_cases_fail.status_code == 200
    assert len(res_cases_fail.json()) == 0

    # 6. List artifacts for result
    res_artifacts = client.get(f"/api/results/{result_id}/artifacts")
    assert res_artifacts.status_code == 200
    artifacts = res_artifacts.json()
    assert len(artifacts) >= 2
    artifact_id = artifacts[0]["id"]

    # 7. Download artifact via result endpoint
    res_download = client.get(f"/api/results/{result_id}/artifacts/{artifact_id}/download")
    assert res_download.status_code == 200
    assert len(res_download.content) > 0
