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


def test_target_capabilities_crud(app_and_client):
    client, _, _ = app_and_client

    # 1. Default inferred capabilities when not explicitly provided
    res_default = client.post(
        "/api/targets",
        json={
            "name": "Default Embedding Target",
            "kind": "embedding",
            "base_url": "http://localhost:8001/v1",
        },
    )
    assert res_default.status_code == 201
    assert res_default.json()["capabilities"] == ["embedding"]

    # 2. Explicit multi-capability target (e.g. chat + embedding)
    res_multi = client.post(
        "/api/targets",
        json={
            "name": "Multi Cap Gateway",
            "kind": "openai-compatible",
            "base_url": "http://localhost:8000/v1",
            "capabilities": ["chat", "embeddings", "chat"],
            "extra": {"model": "qwen2.5-7b"},
        },
    )
    assert res_multi.status_code == 201
    data = res_multi.json()
    target_id = data["id"]
    assert data["capabilities"] == ["chat", "embedding"]
    assert data["extra"]["capabilities"] == ["chat", "embedding"]
    assert data["extra"]["model"] == "qwen2.5-7b"

    # 3. Update capabilities via PATCH
    res_patch = client.patch(
        f"/api/targets/{target_id}",
        json={"capabilities": ["embedding"]},
    )
    assert res_patch.status_code == 200
    patched = res_patch.json()
    assert patched["capabilities"] == ["embedding"]
    assert patched["extra"]["capabilities"] == ["embedding"]
    assert patched["extra"]["model"] == "qwen2.5-7b"


def test_target_test_endpoint_capabilities(app_and_client, monkeypatch):
    import httpx
    import backend.app.api.targets as targets_api

    client, _, _ = app_and_client
    secret_val = "sk-secret-test-key-99999"
    monkeypatch.setenv("TEST_GATEWAY_KEY", secret_val)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        auth_header = request.headers.get("Authorization", "")
        if request.url.host == "auth-fail.local":
            return httpx.Response(401, json={"error": f"Invalid key {secret_val}"})
        if path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "mock-chat-v1", "type": "asr", "engine": "parakeet", "default": True},
                        {"id": "kokoro-82m-v1", "type": "tts", "engine": "kokoro", "default": True},
                        {"id": "pyannote/speaker-diarization-community-1", "type": "diarization", "engine": "pyannote", "default": True},
                    ]
                },
            )
        if path.endswith("/chat/completions"):
            assert auth_header == f"Bearer {secret_val}"
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
            )
        if path.endswith("/embeddings"):
            assert auth_header == f"Bearer {secret_val}"
            return httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
            )
        if path.endswith("/v1/audio/transcriptions"):
            return httpx.Response(200, json={"text": "hello world"})
        if path.endswith("/v1/audio/speech"):
            return httpx.Response(200, content=b"RIFF____WAVEfmt ", headers={"content-type": "audio/wav"})
        if path.endswith("/v1/audio/diarization"):
            return httpx.Response(200, json={"segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5}]})
        return httpx.Response(404, json={"error": "not found"})

    mock_transport = httpx.MockTransport(mock_handler)
    monkeypatch.setattr(targets_api, "_get_http_transport", lambda: mock_transport)

    # Create target with chat and embedding capabilities
    res_create = client.post(
        "/api/targets",
        json={
            "name": "Unified AI Gateway",
            "kind": "openai-compatible",
            "base_url": "http://gateway.local/v1",
            "auth_env_var": "TEST_GATEWAY_KEY",
            "capabilities": ["chat", "embedding"],
            "extra": {"model": "gpt-4o-mini"},
        },
    )
    assert res_create.status_code == 201
    target_id = res_create.json()["id"]

    # 1. Test default configured capabilities (both chat and embedding)
    res_test_all = client.post(f"/api/targets/{target_id}/test")
    assert res_test_all.status_code == 200
    body_all = res_test_all.json()
    assert body_all["ok"] is True
    assert body_all["target_id"] == target_id
    assert body_all["auth_configured"] is True
    assert body_all["capabilities_tested"] == ["chat", "embedding"]
    assert len(body_all["results"]) == 2
    assert body_all["results"][0]["capability"] == "chat"
    assert body_all["results"][0]["ok"] is True
    assert body_all["results"][0]["detail"]["response_preview"] == "OK"
    assert body_all["results"][1]["capability"] == "embedding"
    assert body_all["results"][1]["ok"] is True
    assert body_all["results"][1]["detail"]["dimensions"] == 4
    assert secret_val not in res_test_all.text

    # 2. Test picking a single capability (embedding only)
    res_test_embed = client.post(
        f"/api/targets/{target_id}/test",
        json={"capabilities": ["embedding"], "model": "text-embed-custom"},
    )
    assert res_test_embed.status_code == 200
    body_embed = res_test_embed.json()
    assert body_embed["ok"] is True
    assert body_embed["capabilities_tested"] == ["embedding"]
    assert len(body_embed["results"]) == 1
    assert body_embed["results"][0]["model"] == "text-embed-custom"

    # 3. Test live-stt capabilities (asr, tts, diarization) via ad-hoc test endpoint
    res_adhoc_stt = client.post(
        "/api/targets/test",
        json={
            "name": "Live STT Check",
            "kind": "live-stt",
            "base_url": "http://stt.local",
            "capabilities": ["asr", "tts", "diarization"],
        },
    )
    assert res_adhoc_stt.status_code == 200
    body_stt = res_adhoc_stt.json()
    assert body_stt["ok"] is True
    assert body_stt["capabilities_tested"] == ["asr", "tts", "diarization"]
    assert all(r["ok"] is True for r in body_stt["results"])

    # 4. Test failure case & verify secret redaction
    res_fail = client.post(
        "/api/targets/test",
        json={
            "name": "Failing Target",
            "kind": "openai-compatible",
            "base_url": "http://auth-fail.local/v1",
            "auth_env_var": "TEST_GATEWAY_KEY",
            "capabilities": ["chat"],
        },
    )
    assert res_fail.status_code == 200
    body_fail = res_fail.json()
    assert body_fail["ok"] is False
    assert body_fail["results"][0]["ok"] is False
    assert body_fail["results"][0]["status_code"] == 401
    assert secret_val not in res_fail.text
    assert "***REDACTED***" in (body_fail["results"][0]["error"] or "")

    # 5. Non-existent target returns 404
    res_404 = client.post("/api/targets/999999/test")
    assert res_404.status_code == 404



def test_target_test_grpc_capability(app_and_client):
    from backend.app.testers.live_stt.grpc_client import MockLiveSTTGRPCServer

    client, _, _ = app_and_client

    # Live-stt targets include grpc in their inferred capabilities
    res = client.post(
        "/api/targets",
        json={"name": "stt-grpc-default", "kind": "live-stt", "base_url": "http://stt.local"},
    )
    assert res.status_code == 201
    assert "grpc" in res.json()["capabilities"]

    # Invalid grpc_port is rejected
    bad = client.post(
        "/api/targets",
        json={
            "name": "stt-bad-port",
            "kind": "live-stt",
            "base_url": "http://stt.local",
            "extra": {"grpc_port": "abc"},
        },
    )
    assert bad.status_code == 422

    with MockLiveSTTGRPCServer() as server:
        port = int(server.target.rsplit(":", 1)[1])
        ok = client.post(
            "/api/targets/test",
            json={
                "name": "grpc-ok",
                "kind": "live-stt",
                "base_url": "http://127.0.0.1:1",
                "capabilities": ["grpc"],
                "extra": {"grpc_port": port},
            },
        )
        assert ok.status_code == 200
        body = ok.json()
        assert body["capabilities_tested"] == ["grpc"]
        assert body["ok"] is True, body
        assert body["results"][0]["detail"]["grpc_target"] == f"127.0.0.1:{port}"

    # Nothing listening on the port -> failure
    fail = client.post(
        "/api/targets/test",
        json={
            "name": "grpc-down",
            "kind": "live-stt",
            "base_url": "http://127.0.0.1:1",
            "capabilities": ["grpc"],
            "extra": {"grpc_port": port},
            "timeout_seconds": 2,
        },
    )
    assert fail.status_code == 200
    assert fail.json()["ok"] is False
