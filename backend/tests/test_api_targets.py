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


def test_target_models_discovery_endpoint(app_and_client, monkeypatch):
    """Verify GET /api/targets/{id}/models and POST /api/targets/models with mock and unreachable targets."""
    import httpx
    from backend.app.api import targets as targets_api

    client, _, _ = app_and_client
    secret_val = "sk-live-stt-models-secret-token-777"
    monkeypatch.setenv("LIVE_STT_MODELS_KEY", secret_val)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        auth_header = request.headers.get("Authorization", "")
        if host == "unreachable.local":
            raise httpx.ConnectError(f"Connection refused to {host}")
        if host == "auth-fail.local":
            return httpx.Response(401, json={"error": {"message": f"Invalid key {secret_val}"}})
        assert auth_header == f"Bearer {secret_val}"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "parakeet-ctc-0.6b",
                        "type": "asr",
                        "engine": "parakeet",
                        "default": True,
                        "streaming_capable": True,
                    },
                    {
                        "id": "whisper-large-v3-turbo",
                        "type": "asr",
                        "engine": "whisper",
                        "default": False,
                        "streaming_capable": False,
                    },
                    {
                        "id": "pyannote/speaker-diarization-community-1",
                        "type": "diarization",
                        "default": True,
                        "supports_num_speakers_hint": True,
                    },
                    {
                        "id": "kokoro-82m",
                        "type": "tts",
                        "engine": "kokoro",
                        "default": True,
                        "sample_rate": 24000,
                        "supports_streaming": True,
                        "default_voice": "af_heart",
                        "supported_voices": ["af_heart", "af_bella"],
                    },
                ],
            },
        )

    monkeypatch.setattr(targets_api, "_get_http_transport", lambda: httpx.MockTransport(mock_handler))

    # 1. Create target and query GET /api/targets/{id}/models
    res_create = client.post(
        "/api/targets",
        json={
            "name": "Mock Live-STT Models Target",
            "kind": "live-stt",
            "base_url": "http://stt-models.local/v1",
            "auth_env_var": "LIVE_STT_MODELS_KEY",
        },
    )
    assert res_create.status_code == 201
    target_id = res_create.json()["id"]

    res_models = client.get(f"/api/targets/{target_id}/models")
    assert res_models.status_code == 200
    models = res_models.json()
    assert len(models) == 4
    by_id = {m["id"]: m for m in models}
    assert by_id["parakeet-ctc-0.6b"]["grpc_streaming"] is True
    assert by_id["parakeet-ctc-0.6b"]["is_default"] is True
    assert by_id["parakeet-ctc-0.6b"]["sample_rate"] is None
    assert by_id["whisper-large-v3-turbo"]["grpc_streaming"] is False
    assert by_id["pyannote/speaker-diarization-community-1"]["supports_num_speakers_hint"] is True
    assert by_id["pyannote/speaker-diarization-community-1"]["sample_rate"] is None
    assert by_id["kokoro-82m"]["sample_rate"] == 24000
    assert by_id["kokoro-82m"]["supported_voices"] == ["af_heart", "af_bella"]
    assert by_id["kokoro-82m"]["default_voice"] == "af_heart"

    # 2. Ad-hoc POST /api/targets/models
    res_adhoc = client.post(
        "/api/targets/models",
        json={
            "base_url": "http://stt-models.local",
            "auth_env_var": "LIVE_STT_MODELS_KEY",
        },
    )
    assert res_adhoc.status_code == 200
    assert len(res_adhoc.json()) == 4

    # 3. Unreachable target returns 502 with clear message
    res_down_create = client.post(
        "/api/targets",
        json={
            "name": "Unreachable Live-STT Target",
            "kind": "live-stt",
            "base_url": "http://unreachable.local",
        },
    )
    assert res_down_create.status_code == 201
    down_id = res_down_create.json()["id"]

    res_down = client.get(f"/api/targets/{down_id}/models")
    assert res_down.status_code == 502
    assert "unreachable" in res_down.json()["detail"].lower()

    # 4. Auth failure returns 502 with redacted message (never leaks secret)
    res_auth_fail = client.post(
        "/api/targets/models",
        json={
            "base_url": "http://auth-fail.local",
            "auth_env_var": "LIVE_STT_MODELS_KEY",
        },
    )
    assert res_auth_fail.status_code == 502
    assert secret_val not in res_auth_fail.text
    assert "authentication failed" in res_auth_fail.json()["detail"].lower()

    # 5. Non-existent target returns 404
    res_404 = client.get("/api/targets/999999/models")
    assert res_404.status_code == 404



def test_target_test_pyannote_ai_checks_key_without_submitting_a_job(app_and_client, monkeypatch):
    """A pyannote-ai target's Test button authenticates via POST /v1/media/input only (free),
    never /v1/diarize (billed), and defaults to the diarization capability."""
    import httpx
    import backend.app.api.targets as targets_api

    client, _, _ = app_and_client
    secret_val = "sk-pyannote-secret-123"
    monkeypatch.setenv("TEST_PYANNOTE_AI_KEY", secret_val)
    seen: list[str] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.headers.get("Authorization") != f"Bearer {secret_val}":
            return httpx.Response(401, json={"message": f"bad key {request.headers.get('Authorization')}"})
        if request.url.path == "/v1/media/input":
            return httpx.Response(201, json={"url": "https://uploads.example/put"})
        return httpx.Response(404)

    monkeypatch.setattr(targets_api, "_get_http_transport", lambda: httpx.MockTransport(mock_handler))

    res = client.post(
        "/api/targets/test",
        json={
            "name": "pyannote.ai",
            "kind": "pyannote-ai",
            "base_url": "https://api.pyannote.ai/v1",
            "auth_env_var": "TEST_PYANNOTE_AI_KEY",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["capabilities_tested"] == ["diarization"]
    assert body["ok"] is True, body
    assert body["results"][0]["detail"]["backend"] == "pyannote_ai"
    assert seen == ["/v1/media/input"]

    monkeypatch.setenv("TEST_PYANNOTE_AI_KEY", "wrong")
    res_bad = client.post(
        "/api/targets/test",
        json={
            "name": "pyannote.ai",
            "kind": "pyannote-ai",
            "base_url": "https://api.pyannote.ai/v1",
            "auth_env_var": "TEST_PYANNOTE_AI_KEY",
        },
    )
    bad = res_bad.json()
    assert bad["ok"] is False
    assert bad["results"][0]["status_code"] == 401


def test_target_models_pyannote_ai_returns_builtin_catalog_without_network(app_and_client, monkeypatch):
    """pyannote.ai has no /v1/models: the model picker must get the fixed pyannote.ai catalog."""
    import httpx
    import backend.app.api.targets as targets_api

    client, _, _ = app_and_client

    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    monkeypatch.setattr(targets_api, "_get_http_transport", lambda: httpx.MockTransport(no_network))

    res_create = client.post(
        "/api/targets",
        json={
            "name": "pyannote.ai cloud",
            "kind": "pyannote-ai",
            "base_url": "https://api.pyannote.ai/v1",
            "auth_env_var": "PYANNOTE_AI_API_KEY",
        },
    )
    assert res_create.status_code == 201
    res = client.get(f"/api/targets/{res_create.json()['id']}/models")
    assert res.status_code == 200, res.text
    models = {m["id"]: m for m in res.json()}
    assert set(models) == {"pyannote.ai/precision-2", "pyannote.ai/precision-3", "pyannote.ai/community-1"}
    assert models["pyannote.ai/precision-2"]["is_default"] is True
    assert all(m["type"] == "diarization" and m["engine"] == "pyannote_ai" for m in models.values())

    # Ad-hoc discovery: detected by kind, or by host alone
    for body in (
        {"base_url": "https://example.invalid", "kind": "pyannote-ai"},
        {"base_url": "https://api.pyannote.ai/v1"},
    ):
        res_adhoc = client.post("/api/targets/models", json=body)
        assert res_adhoc.status_code == 200, res_adhoc.text
        assert len(res_adhoc.json()) == 3


def test_target_test_google_cloud_checks_key_without_submitting_audio(app_and_client, monkeypatch):
    """A google-cloud target's Test button mints a token and reads the model card only (free),
    never generateContent/recognize (billed), and defaults to the diarization capability."""
    import json as _json

    import httpx
    import backend.app.api.targets as targets_api
    import backend.app.testers.live_stt.google_cloud as google_cloud

    client, _, _ = app_and_client
    sa = {"type": "service_account", "project_id": "demo-proj", "private_key": "-----BEGIN PRIVATE KEY-----\nx\n",
          "client_email": "sa@demo-proj.iam.gserviceaccount.com"}
    monkeypatch.setenv("TEST_GOOGLE_SA_JSON", _json.dumps(sa))
    monkeypatch.setattr(google_cloud, "_service_account_token_provider", lambda info: (lambda: "ya29.ok"))
    seen: list[str] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.url.host}{request.url.path}")
        if request.headers.get("Authorization") != "Bearer ya29.ok":
            return httpx.Response(401, json={"error": {"message": "invalid authentication"}})
        if request.url.path == "/v1beta1/publishers/google/models/gemini-3.5-transcribe":
            return httpx.Response(200, json={"name": "publishers/google/models/gemini-3.5-transcribe"})
        return httpx.Response(404)

    monkeypatch.setattr(targets_api, "_get_http_transport", lambda: httpx.MockTransport(mock_handler))
    payload = {"name": "Google Cloud", "kind": "google-cloud", "base_url": "https://aiplatform.googleapis.com",
               "auth_env_var": "TEST_GOOGLE_SA_JSON"}
    res = client.post("/api/targets/test", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["capabilities_tested"] == ["diarization"]
    assert body["ok"] is True, body
    assert body["results"][0]["detail"]["backend"] == "google_cloud"
    assert "demo-proj" in body["results"][0]["message"]
    assert seen == ["aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-3.5-transcribe"]

    monkeypatch.setenv("TEST_GOOGLE_SA_JSON", "/nope/missing.json")
    bad = client.post("/api/targets/test", json=payload).json()
    assert bad["ok"] is False
    assert "neither an existing key file" in bad["results"][0]["error"]


def test_target_models_google_cloud_returns_builtin_catalog_without_network(app_and_client, monkeypatch):
    import httpx
    import backend.app.api.targets as targets_api

    client, _, _ = app_and_client

    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    monkeypatch.setattr(targets_api, "_get_http_transport", lambda: httpx.MockTransport(no_network))
    res_create = client.post(
        "/api/targets",
        json={"name": "Google Cloud speech", "kind": "google-cloud", "base_url": "https://aiplatform.googleapis.com",
              "auth_env_var": "GOOGLE_APPLICATION_CREDENTIALS"},
    )
    assert res_create.status_code == 201
    res = client.get(f"/api/targets/{res_create.json()['id']}/models")
    assert res.status_code == 200, res.text
    models = {m["id"]: m for m in res.json()}
    assert set(models) == {"google/gemini-3.5-transcribe", "google/chirp_3"}
    assert models["google/gemini-3.5-transcribe"]["is_default"] is True
    assert all(m["type"] == "diarization" and m["engine"] == "google_cloud" for m in models.values())
    for body in (
        {"base_url": "https://example.invalid", "kind": "google-cloud"},
        {"base_url": "https://us-central1-aiplatform.googleapis.com"},
    ):
        res_adhoc = client.post("/api/targets/models", json=body)
        assert res_adhoc.status_code == 200, res_adhoc.text
        assert len(res_adhoc.json()) == 2
