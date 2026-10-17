"""Tests for the targets REST API and secret safety checks."""

import pytest
from backend.app.db.models.target import Target
from backend.app.db.models.test_definition import TestDefinition


def test_target_crud_lifecycle(app_and_client):
    client, session_factory, _ = app_and_client

    # 1. Create target with env-var reference auth
    create_payload = {
        "name": "Local FastSTT Target",
        "kind": "live-stt",
        "base_url": "http://127.0.0.1:9000",
        "auth_env_var": "FASTSTT_API_KEY",
        "extra": {
            "model": "whisper-large-v3",
            "concurrency": 4,
        },
    }
    res = client.post("/api/targets", json=create_payload)
    assert res.status_code == 201
    target_data = res.json()
    target_id = target_data["id"]
    assert target_data["name"] == "Local FastSTT Target"
    assert target_data["kind"] == "live-stt"
    assert target_data["base_url"] == "http://127.0.0.1:9000"
    assert target_data["auth_env_var"] == "FASTSTT_API_KEY"
    assert target_data["extra"]["model"] == "whisper-large-v3"
    assert "created_at" in target_data
    assert "updated_at" in target_data

    # 2. Duplicate target name rejected with 409 Conflict
    res_dup = client.post("/api/targets", json=create_payload)
    assert res_dup.status_code == 409
    assert "already exists" in res_dup.json()["detail"]

    # 3. List targets
    res_list = client.get("/api/targets")
    assert res_list.status_code == 200
    targets = res_list.json()
    assert any(t["id"] == target_id for t in targets)

    # Filter by kind
    res_filter = client.get("/api/targets?kind=live-stt")
    assert res_filter.status_code == 200
    assert any(t["id"] == target_id for t in res_filter.json())

    res_empty_filter = client.get("/api/targets?kind=nonexistent-kind")
    assert res_empty_filter.status_code == 200
    assert len(res_empty_filter.json()) == 0

    # 4. Get target by ID
    res_get = client.get(f"/api/targets/{target_id}")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "Local FastSTT Target"

    # Nonexistent target
    res_404 = client.get("/api/targets/999999")
    assert res_404.status_code == 404

    # 5. Update target (PUT & PATCH)
    update_payload = {
        "name": "Updated STT Target",
        "base_url": "http://127.0.0.1:9001",
        "auth_env_var": "NEW_FASTSTT_KEY",
        "extra": {"model": "whisper-large-v3-turbo"},
    }
    res_update = client.patch(f"/api/targets/{target_id}", json=update_payload)
    assert res_update.status_code == 200
    updated = res_update.json()
    assert updated["name"] == "Updated STT Target"
    assert updated["base_url"] == "http://127.0.0.1:9001"
    assert updated["auth_env_var"] == "NEW_FASTSTT_KEY"
    assert updated["extra"]["model"] == "whisper-large-v3-turbo"

    # 6. Delete target
    res_del = client.delete(f"/api/targets/{target_id}")
    assert res_del.status_code == 204

    # Verify deleted
    res_after = client.get(f"/api/targets/{target_id}")
    assert res_after.status_code == 404

    # Delete non-existent
    res_del_404 = client.delete(f"/api/targets/{target_id}")
    assert res_del_404.status_code == 404


def test_target_auth_env_var_validation(app_and_client):
    client, _, _ = app_and_client

    # Case A: Raw secret token starting with sk-
    bad_payload_sk = {
        "name": "Invalid Secret Target 1",
        "kind": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "auth_env_var": "sk-proj-1234567890abcdef",
    }
    res = client.post("/api/targets", json=bad_payload_sk)
    assert res.status_code == 422
    assert "environment variable reference only" in str(res.json())

    # Case B: Invalid variable name with spaces
    bad_payload_spaces = {
        "name": "Invalid Secret Target 2",
        "kind": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "auth_env_var": "MY SECRET KEY",
    }
    res = client.post("/api/targets", json=bad_payload_spaces)
    assert res.status_code == 422

    # Case C: Invalid characters in env var
    bad_payload_special = {
        "name": "Invalid Secret Target 3",
        "kind": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "auth_env_var": "KEY:WITH:COLONS",
    }
    res = client.post("/api/targets", json=bad_payload_special)
    assert res.status_code == 422


def test_target_extra_secrets_rejection(app_and_client):
    client, _, _ = app_and_client

    # Attempting to store raw secrets directly in extra should be rejected
    payload_secret_in_extra = {
        "name": "Insecure Target",
        "kind": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "auth_env_var": "VALID_ENV_VAR",
        "extra": {
            "api_key": "sk-actual-secret-value",
        },
    }
    res = client.post("/api/targets", json=payload_secret_in_extra)
    assert res.status_code == 422
    assert "Direct secrets are forbidden" in str(res.json())


def test_target_deletion_sets_definition_target_id_null(app_and_client):
    client, session_factory, _ = app_and_client

    # Create target
    res_t = client.post(
        "/api/targets",
        json={"name": "Cascade Test Target", "kind": "dummy", "base_url": "http://localhost:8000"},
    )
    assert res_t.status_code == 201
    target_id = res_t.json()["id"]

    # Create test definition referencing this target
    res_d = client.post(
        "/api/test-definitions",
        json={
            "name": "Target Ref Definition",
            "tester": "dummy",
            "target_id": target_id,
        },
    )
    assert res_d.status_code == 201
    def_id = res_d.json()["id"]
    assert res_d.json()["target_id"] == target_id

    # Delete the target
    res_del = client.delete(f"/api/targets/{target_id}")
    assert res_del.status_code == 204

    # Verify definition still exists but target_id is set to None
    res_def_after = client.get(f"/api/test-definitions/{def_id}")
    assert res_def_after.status_code == 200
    assert res_def_after.json()["target_id"] is None
