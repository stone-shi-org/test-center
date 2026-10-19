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


def test_delete_definition_blocked_when_used_by_job(app_and_client):
    """Deleting a template referenced in Job.definition_ids returns 409 Conflict."""
    client, _, _ = app_and_client

    res_d1 = client.post(
        "/api/test-definitions",
        json={"name": "Template In Use", "tester": "dummy", "suites": ["smoke"]},
    )
    assert res_d1.status_code == 201
    def1_id = res_d1.json()["id"]

    res_d2 = client.post(
        "/api/test-definitions",
        json={"name": "Unused Template", "tester": "dummy", "suites": ["smoke"]},
    )
    assert res_d2.status_code == 201
    def2_id = res_d2.json()["id"]

    res_job = client.post(
        "/api/jobs",
        json={
            "name": "Nightly Regression Job",
            "run_mode": "sequential",
            "definition_ids": [def1_id],
        },
    )
    assert res_job.status_code == 201
    job_id = res_job.json()["id"]

    # Deleting template used by job is blocked with 409
    res_del_blocked = client.delete(f"/api/test-definitions/{def1_id}")
    assert res_del_blocked.status_code == 409
    assert "Nightly Regression Job" in res_del_blocked.json()["detail"]

    # Deleting unused template succeeds
    res_del_unused = client.delete(f"/api/test-definitions/{def2_id}")
    assert res_del_unused.status_code == 204

    # After deleting the job, the template can be deleted
    res_del_job = client.delete(f"/api/jobs/{job_id}")
    assert res_del_job.status_code == 204

    res_del_ok = client.delete(f"/api/test-definitions/{def1_id}")
    assert res_del_ok.status_code == 204


def test_starter_templates_endpoint(app_and_client):
    """Verify GET /api/test-definitions/starters and /api/definitions/starters."""
    client, _, _ = app_and_client

    # 1. Main endpoint
    res = client.get("/api/test-definitions/starters")
    assert res.status_code == 200
    starters = res.json()
    assert len(starters) >= 13

    # Check structure
    for s in starters:
        assert "tester" in s
        assert "name" in s
        assert "description" in s
        assert "suites" in s
        assert "config" in s
        assert isinstance(s["suites"], list)
        assert isinstance(s["config"], dict)

    # 2. Filter by tester
    res_stt = client.get("/api/test-definitions/starters?tester=live-stt")
    assert res_stt.status_code == 200
    stt_starters = res_stt.json()
    assert len(stt_starters) == 6
    assert all(s["tester"] == "live-stt" for s in stt_starters)
    stt_names = [s["name"] for s in stt_starters]
    assert "HTTP smoke" in stt_names
    assert "HTTP full contract" in stt_names
    assert "Quick accuracy (WER)" in stt_names

    res_llm = client.get("/api/test-definitions/starters?tester=llm")
    assert res_llm.status_code == 200
    llm_starters = res_llm.json()
    assert len(llm_starters) == 3
    assert any(s["name"] == "Reasoning, high effort" for s in llm_starters)

    # 3. Alias endpoint /api/definitions/starters
    res_alias = client.get("/api/definitions/starters")
    assert res_alias.status_code == 200
    assert len(res_alias.json()) == len(starters)


def test_all_starter_templates_valid():
    """Validate every starter template against its plugin schema and suites."""
    from backend.app.testers.starters import STARTER_TEMPLATES
    from backend.app.testers.registry import tester_registry

    assert len(STARTER_TEMPLATES) >= 13

    for starter in STARTER_TEMPLATES:
        tester_name = starter["tester"]
        plugin = tester_registry.get(tester_name)
        assert plugin is not None

        # Verify suites belong to the tester
        allowed_suites = set(plugin.suites) | set(getattr(plugin, "suite_aliases", set()))
        for suite in starter["suites"]:
            assert suite in allowed_suites, f"Suite '{suite}' in starter '{starter['name']}' not allowed in tester '{tester_name}'"

        # Verify config validates and applies defaults cleanly
        validated_cfg = plugin.validate_and_apply_config(dict(starter["config"]))
        assert isinstance(validated_cfg, dict)


def test_definition_and_run_config_do_not_contain_stray_suites(app_and_client):
    """Verify TestDefinition and Run config never retain stray 'suites' inside config."""
    client, _, _ = app_and_client

    # 1. Create a live-stt template with top-level suites=['accuracy_wer'] and conflicting config.suites
    res_def = client.post(
        "/api/test-definitions",
        json={
            "name": "Live-STT Accuracy WER Only",
            "tester": "live-stt",
            "suites": ["accuracy_wer"],
            "config": {
                "models": ["parakeet-tdt-0.6b-v3"],
                "suites": ["transcriptions_json", "admin"],
            },
        },
    )
    assert res_def.status_code == 201
    def_data = res_def.json()
    assert def_data["suites"] == ["accuracy_wer"]
    assert "suites" not in def_data["config"]

    # 2. Trigger run via /api/test-definitions/{id}/run
    res_def_run = client.post(f"/api/test-definitions/{def_data['id']}/run", json={})
    assert res_def_run.status_code == 201
    def_run_snap = res_def_run.json()["config_snapshot"]
    assert def_run_snap["suites"] == ["accuracy_wer"]
    assert "suites" not in def_run_snap["config"]

    # 3. Trigger run directly via /api/runs (as TriggerView does)
    res_direct_run = client.post(
        "/api/runs",
        json={
            "tester": "live-stt",
            "suites": ["accuracy_wer"],
            "config": {
                "models": ["parakeet-tdt-0.6b-v3"],
                "suites": ["transcriptions_json"],
            },
        },
    )
    assert res_direct_run.status_code == 201
    direct_snap = res_direct_run.json()["config_snapshot"]
    assert direct_snap["suites"] == ["accuracy_wer"]
    assert "suites" not in direct_snap["config"]


