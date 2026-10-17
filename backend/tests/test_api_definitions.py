"""Tests for test definitions REST API and run trigger endpoint."""

import asyncio
import pytest
from backend.app.runner.worker import get_worker


def test_test_definition_crud(app_and_client):
    client, session_factory, _ = app_and_client

    # 1. Create Target
    res_t = client.post(
        "/api/targets",
        json={
            "name": "Target For Def",
            "kind": "dummy",
            "base_url": "http://127.0.0.1:8000",
            "auth_env_var": "MY_TEST_KEY",
        },
    )
    assert res_t.status_code == 201
    target_id = res_t.json()["id"]

    # 2. Create TestDefinition
    def_payload = {
        "name": "Dummy Fast Suite",
        "tester": "dummy",
        "suites": ["smoke"],
        "config": {
            "duration_seconds": 0.2,
            "step_delay": 0.05,
        },
        "target_id": target_id,
    }
    res_d = client.post("/api/test-definitions", json=def_payload)
    assert res_d.status_code == 201
    def_data = res_d.json()
    def_id = def_data["id"]
    assert def_data["name"] == "Dummy Fast Suite"
    assert def_data["tester"] == "dummy"
    assert def_data["suites"] == ["smoke"]
    assert def_data["config"]["duration_seconds"] == 0.2
    assert def_data["target_id"] == target_id
    assert def_data["target"]["name"] == "Target For Def"

    # 3. List TestDefinitions
    res_list = client.get("/api/test-definitions")
    assert res_list.status_code == 200
    defs = res_list.json()
    assert any(d["id"] == def_id for d in defs)

    # Filter by tester
    res_tester_filter = client.get("/api/test-definitions?tester=dummy")
    assert res_tester_filter.status_code == 200
    assert any(d["id"] == def_id for d in res_tester_filter.json())

    # Filter by target_id
    res_target_filter = client.get(f"/api/test-definitions?target_id={target_id}")
    assert res_target_filter.status_code == 200
    assert any(d["id"] == def_id for d in res_target_filter.json())

    # 4. Get by ID
    res_get = client.get(f"/api/test-definitions/{def_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == def_id

    # Nonexistent ID
    res_404 = client.get("/api/test-definitions/999999")
    assert res_404.status_code == 404

    # 5. Update TestDefinition
    update_payload = {
        "name": "Updated Dummy Suite",
        "config": {"duration_seconds": 0.5},
    }
    res_update = client.patch(f"/api/test-definitions/{def_id}", json=update_payload)
    assert res_update.status_code == 200
    assert res_update.json()["name"] == "Updated Dummy Suite"
    assert res_update.json()["config"]["duration_seconds"] == 0.5

    # 6. Delete TestDefinition
    res_del = client.delete(f"/api/test-definitions/{def_id}")
    assert res_del.status_code == 204

    # Verify deleted
    res_after = client.get(f"/api/test-definitions/{def_id}")
    assert res_after.status_code == 404


def test_test_definition_validations(app_and_client):
    client, _, _ = app_and_client

    # Case A: Nonexistent target_id
    res_bad_target = client.post(
        "/api/test-definitions",
        json={"name": "Bad Target Def", "tester": "dummy", "target_id": 99999},
    )
    assert res_bad_target.status_code == 404
    assert "Target 99999 not found" in res_bad_target.json()["detail"]

    # Case B: Unknown suite for tester
    res_bad_suite = client.post(
        "/api/test-definitions",
        json={"name": "Bad Suite Def", "tester": "dummy", "suites": ["nonexistent-suite"]},
    )
    assert res_bad_suite.status_code == 422
    assert "Unknown suites" in res_bad_suite.json()["detail"]

    # Case C: Invalid config against tester schema
    res_bad_config = client.post(
        "/api/test-definitions",
        json={
            "name": "Bad Config Def",
            "tester": "dummy",
            "config": {"duration_seconds": -5.0},  # schema requires >= 0
        },
    )
    assert res_bad_config.status_code == 422


def test_definitions_alias_route(app_and_client):
    """Verify that /api/definitions works as an alias for /api/test-definitions."""
    client, _, _ = app_and_client

    # Create via /api/definitions
    res_create = client.post(
        "/api/definitions",
        json={"name": "Alias Route Test", "tester": "dummy", "suites": ["smoke"]},
    )
    assert res_create.status_code == 201
    def_id = res_create.json()["id"]

    # List via /api/definitions
    res_list = client.get("/api/definitions")
    assert res_list.status_code == 200
    assert any(d["id"] == def_id for d in res_list.json())

    # Get via /api/definitions/{id}
    res_get = client.get(f"/api/definitions/{def_id}")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "Alias Route Test"


def test_trigger_manual_run_from_test_definition(app_and_client):
    """Verify triggering a manual run directly from a test definition."""
    client, session_factory, artifacts_dir = app_and_client

    # Create Target
    res_t = client.post(
        "/api/targets",
        json={"name": "Trigger Target", "kind": "dummy", "base_url": "http://127.0.0.1:8000"},
    )
    assert res_t.status_code == 201
    target_id = res_t.json()["id"]

    # Create TestDefinition
    res_d = client.post(
        "/api/test-definitions",
        json={
            "name": "Def To Trigger",
            "tester": "dummy",
            "suites": ["smoke"],
            "config": {"duration_seconds": 0.1, "step_delay": 0.05},
            "target_id": target_id,
        },
    )
    assert res_d.status_code == 201
    def_id = res_d.json()["id"]

    # Trigger Run via /api/test-definitions/{id}/run
    res_run = client.post(f"/api/test-definitions/{def_id}/run", json={"timeout": 30.0})
    assert res_run.status_code == 201
    run_data = res_run.json()
    run_id = run_data["id"]
    assert run_data["status"] == "queued"
    assert run_data["config_snapshot"]["test_definition_id"] == def_id
    assert run_data["target_snapshot"]["id"] == target_id

    # Execute with worker
    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True

    # Verify run completed
    res_detail = client.get(f"/api/runs/{run_id}")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["status"] == "completed"
    assert len(detail["results"]) == 1
    # Check that Result record has test_definition_id
    assert detail["results"][0]["status"] == "pass"

    # Also verify via /api/results
    res_results = client.get(f"/api/results?test_definition_id={def_id}")
    assert res_results.status_code == 200
    results_list = res_results.json()
    assert len(results_list) >= 1
    assert results_list[0]["run_id"] == run_id
    assert results_list[0]["test_definition_id"] == def_id
