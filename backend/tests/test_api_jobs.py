"""Tests for Jobs model, CRUD API, ordering, overrides, and sequential/parallel manual runs."""

import asyncio
from backend.app.runner.worker import get_worker


def _create_target(client, name="Job Test Target", kind="dummy", base_url="http://127.0.0.1:8000"):
    res = client.post(
        "/api/targets",
        json={"name": name, "kind": kind, "base_url": base_url},
    )
    assert res.status_code == 201
    return res.json()


def _create_definition(
    client,
    name="Def 1",
    tester="dummy",
    suites=None,
    config=None,
    target_id=None,
):
    res = client.post(
        "/api/test-definitions",
        json={
            "name": name,
            "tester": tester,
            "suites": suites if suites is not None else ["smoke"],
            "config": config if config is not None else {"duration_seconds": 0.1, "step_delay": 0.02},
            "target_id": target_id,
        },
    )
    assert res.status_code == 201
    return res.json()


def test_job_crud_ordering_and_overrides(app_and_client):
    client, _, _ = app_and_client

    t1 = _create_target(client, name="Target Alpha")
    t2 = _create_target(client, name="Target Beta")

    d1 = _create_definition(client, name="Smoke Check", suites=["smoke"], target_id=t1["id"])
    d2 = _create_definition(client, name="Load Check", suites=["load"], target_id=t1["id"])
    d3 = _create_definition(client, name="Edge Cases Check", suites=["edge_cases"], target_id=t2["id"])

    # 1. Create Job with ordered definition_ids, run_mode, and per-definition overrides
    create_payload = {
        "name": "Daily Full Suite",
        "description": "Runs load and smoke definitions in order",
        "schedule": "manual",
        "enabled": True,
        "run_mode": "sequential",
        "definition_ids": [d2["id"], d1["id"]],
        "overrides": {
            str(d2["id"]): {
                "suites": ["smoke", "load"],
                "config": {"duration_seconds": 0.25},
                "target_id": t2["id"],
            }
        },
    }
    res_create = client.post("/api/jobs", json=create_payload)
    assert res_create.status_code == 201
    job = res_create.json()
    job_id = job["id"]

    assert job["name"] == "Daily Full Suite"
    assert job["description"] == "Runs load and smoke definitions in order"
    assert job["run_mode"] == "sequential"
    assert job["definition_ids"] == [d2["id"], d1["id"]]
    assert len(job["definitions"]) == 2

    # Verify ordering and effective overrides on resolved definitions
    first_def = job["definitions"][0]
    assert first_def["order"] == 0
    assert first_def["test_definition_id"] == d2["id"]
    assert first_def["name"] == "Load Check"
    assert first_def["effective_suites"] == ["smoke", "load"]
    assert first_def["effective_config"]["duration_seconds"] == 0.25
    assert first_def["effective_target_id"] == t2["id"]
    assert first_def["effective_target"]["name"] == "Target Beta"

    second_def = job["definitions"][1]
    assert second_def["order"] == 1
    assert second_def["test_definition_id"] == d1["id"]
    assert second_def["effective_suites"] == ["smoke"]
    assert second_def["effective_target_id"] == t1["id"]

    # 2. Duplicate name returns 409 Conflict
    res_dup = client.post("/api/jobs", json={"name": "Daily Full Suite"})
    assert res_dup.status_code == 409

    # 3. List Jobs with filters
    res_list = client.get("/api/jobs")
    assert res_list.status_code == 200
    assert any(j["id"] == job_id for j in res_list.json())

    res_mode_filter = client.get("/api/jobs?run_mode=sequential")
    assert res_mode_filter.status_code == 200
    assert any(j["id"] == job_id for j in res_mode_filter.json())

    # 4. Get Job by ID
    res_get = client.get(f"/api/jobs/{job_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == job_id

    # 5. Reorder definitions via PUT /api/jobs/{id}/order
    res_order = client.put(
        f"/api/jobs/{job_id}/order",
        json={"definition_ids": [d3["id"], d1["id"], d2["id"]]},
    )
    assert res_order.status_code == 200
    ordered_job = res_order.json()
    assert ordered_job["definition_ids"] == [d3["id"], d1["id"], d2["id"]]
    assert [d["test_definition_id"] for d in ordered_job["definitions"]] == [
        d3["id"],
        d1["id"],
        d2["id"],
    ]

    # 6. Update Job (change run_mode to parallel and update definitions inline)
    res_update = client.patch(
        f"/api/jobs/{job_id}",
        json={
            "name": "Parallel Suite",
            "run_mode": "parallel",
            "definitions": [
                {
                    "test_definition_id": d1["id"],
                    "suites": ["smoke"],
                    "config": {"duration_seconds": 0.15},
                },
                {
                    "test_definition_id": d3["id"],
                    "config": {"duration_seconds": 0.2},
                },
            ],
        },
    )
    assert res_update.status_code == 200
    updated = res_update.json()
    assert updated["name"] == "Parallel Suite"
    assert updated["run_mode"] == "parallel"
    assert updated["definition_ids"] == [d1["id"], d3["id"]]
    assert updated["definitions"][0]["effective_config"]["duration_seconds"] == 0.15
    assert updated["definitions"][1]["effective_config"]["duration_seconds"] == 0.2

    # 7. Delete Job
    res_del = client.delete(f"/api/jobs/{job_id}")
    assert res_del.status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_job_validations(app_and_client):
    client, _, _ = app_and_client
    d1 = _create_definition(client, name="Valid Def")

    # Invalid run_mode
    res_bad_mode = client.post(
        "/api/jobs",
        json={"name": "Bad Mode Job", "run_mode": "random_mode", "definition_ids": [d1["id"]]},
    )
    assert res_bad_mode.status_code == 422

    # Nonexistent TestDefinition ID
    res_bad_def = client.post(
        "/api/jobs",
        json={"name": "Bad Def Job", "definition_ids": [999999]},
    )
    assert res_bad_def.status_code == 404
    assert "TestDefinition 999999 not found" in res_bad_def.json()["detail"]

    # Nonexistent Target ID in override
    res_bad_target = client.post(
        "/api/jobs",
        json={
            "name": "Bad Target Override Job",
            "definition_ids": [d1["id"]],
            "overrides": {str(d1["id"]): {"target_id": 888888}},
        },
    )
    assert res_bad_target.status_code == 404
    assert "Target 888888 not found" in res_bad_target.json()["detail"]

    # Invalid suite in override
    res_bad_suite = client.post(
        "/api/jobs",
        json={
            "name": "Bad Suite Override Job",
            "definition_ids": [d1["id"]],
            "overrides": {str(d1["id"]): {"suites": ["unknown_suite_xyz"]}},
        },
    )
    assert res_bad_suite.status_code == 422
    assert "Unknown suites" in res_bad_suite.json()["detail"]

    # Invalid config value in override against tester JSON Schema
    res_bad_cfg = client.post(
        "/api/jobs",
        json={
            "name": "Bad Config Override Job",
            "definition_ids": [d1["id"]],
            "overrides": {str(d1["id"]): {"config": {"duration_seconds": -10}}},
        },
    )
    assert res_bad_cfg.status_code == 422


def test_job_manual_run_sequential_groups_results(app_and_client):
    """Triggering 'Run now' on a sequential job executes definitions in order and groups per-tester results."""
    client, _, _ = app_and_client

    target = _create_target(client, name="Seq Target")
    d1 = _create_definition(
        client,
        name="Step 1 Smoke",
        suites=["smoke"],
        config={"duration_seconds": 0.05, "step_delay": 0.01, "fail_case": False},
        target_id=target["id"],
    )
    d2 = _create_definition(
        client,
        name="Step 2 Edge Cases",
        suites=["edge_cases"],
        config={"duration_seconds": 0.05, "step_delay": 0.01, "fail_case": False},
        target_id=target["id"],
    )

    res_job = client.post(
        "/api/jobs",
        json={
            "name": "Sequential Bundled Job",
            "run_mode": "sequential",
            "definition_ids": [d1["id"], d2["id"]],
            "overrides": {
                str(d2["id"]): {"config": {"fail_case": True}},
            },
        },
    )
    assert res_job.status_code == 201
    job_id = res_job.json()["id"]

    # Trigger 'Run now'
    res_run = client.post(f"/api/jobs/{job_id}/run")
    assert res_run.status_code == 201
    queued_run = res_run.json()
    run_id = queued_run["id"]
    assert queued_run["job_id"] == job_id
    assert queued_run["job_name"] == "Sequential Bundled Job"
    assert queued_run["status"] == "queued"
    assert queued_run["config_snapshot"]["run_mode"] == "sequential"
    assert len(queued_run["config_snapshot"]["definitions"]) == 2
    assert (
        queued_run["config_snapshot"]["definitions"][1]["config"]["fail_case"]
        is True
    )

    # Execute queued run with worker
    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True

    # Verify Run groups both per-tester results in definition order
    res_detail = client.get(f"/api/runs/{run_id}")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["status"] == "completed"
    assert detail["job_id"] == job_id
    assert len(detail["results"]) == 2

    res_step_1 = detail["results"][0]
    assert res_step_1["test_definition_id"] == d1["id"]
    assert res_step_1["definition_name"] == "Step 1 Smoke"
    assert res_step_1["tester"] == "dummy"
    assert res_step_1["status"] == "pass"
    assert len(res_step_1["cases"]) == 1
    assert res_step_1["cases"][0]["suite"] == "smoke"

    res_step_2 = detail["results"][1]
    assert res_step_2["test_definition_id"] == d2["id"]
    assert res_step_2["definition_name"] == "Step 2 Edge Cases"
    assert res_step_2["tester"] == "dummy"
    assert res_step_2["status"] == "fail"
    assert len(res_step_2["cases"]) == 1
    assert res_step_2["cases"][0]["suite"] == "edge_cases"
    assert res_step_2["cases"][0]["status"] == "fail"

    # Verify /api/jobs/{job_id}/runs lists this run
    res_job_runs = client.get(f"/api/jobs/{job_id}/runs")
    assert res_job_runs.status_code == 200
    job_runs = res_job_runs.json()
    assert len(job_runs) == 1
    assert job_runs[0]["id"] == run_id
    assert len(job_runs[0]["results"]) == 2

    # Verify combined run logs include both steps
    res_logs = client.get(f"/api/runs/{run_id}/logs")
    assert res_logs.status_code == 200
    raw_logs = res_logs.json()["raw"]
    assert "Step 1/2" in raw_logs
    assert "Step 2/2" in raw_logs


def test_job_manual_run_parallel_and_runtime_overrides(app_and_client):
    """Triggering a parallel job runs definitions concurrently and applies runtime overrides."""
    client, _, _ = app_and_client

    d1 = _create_definition(
        client,
        name="Parallel Def A",
        suites=["smoke"],
        config={"duration_seconds": 0.08, "step_delay": 0.02},
    )
    d2 = _create_definition(
        client,
        name="Parallel Def B",
        suites=["load"],
        config={"duration_seconds": 0.08, "step_delay": 0.02},
    )

    res_job = client.post(
        "/api/jobs",
        json={
            "name": "Parallel Eval Job",
            "run_mode": "parallel",
            "definition_ids": [d1["id"], d2["id"]],
        },
    )
    assert res_job.status_code == 201
    job_id = res_job.json()["id"]

    # Trigger with runtime override on d2
    res_run = client.post(
        f"/api/jobs/{job_id}/run",
        json={
            "overrides": {
                str(d2["id"]): {
                    "suites": ["smoke", "load"],
                    "config": {"fail_case": True},
                }
            }
        },
    )
    assert res_run.status_code == 201
    run_id = res_run.json()["id"]
    assert res_run.json()["config_snapshot"]["run_mode"] == "parallel"

    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["status"] == "completed"
    assert len(detail["results"]) == 2
    assert detail["results"][0]["test_definition_id"] == d1["id"]
    assert detail["results"][0]["status"] == "pass"
    assert detail["results"][1]["test_definition_id"] == d2["id"]
    assert detail["results"][1]["status"] == "fail"
    assert len(detail["results"][1]["cases"]) == 2


def test_trigger_empty_job_and_post_runs_with_job_id(app_and_client):
    client, _, _ = app_and_client

    # Empty job cannot be triggered
    res_empty = client.post("/api/jobs", json={"name": "Empty Shell Job", "definition_ids": []})
    assert res_empty.status_code == 201
    empty_id = res_empty.json()["id"]

    res_fail_run = client.post(f"/api/jobs/{empty_id}/run")
    assert res_fail_run.status_code == 422
    assert "no test definitions" in res_fail_run.json()["detail"].lower()

    # Non-empty job triggered via POST /api/runs {"job_id": id}
    d1 = _create_definition(client, name="Single Job Def")
    client.patch(f"/api/jobs/{empty_id}", json={"definition_ids": [d1["id"]]})

    res_via_runs = client.post("/api/runs", json={"job_id": empty_id})
    assert res_via_runs.status_code == 201
    assert res_via_runs.json()["job_id"] == empty_id
    assert len(res_via_runs.json()["config_snapshot"]["definitions"]) == 1
