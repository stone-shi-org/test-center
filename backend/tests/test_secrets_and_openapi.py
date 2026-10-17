"""Tests ensuring secrets are never leaked/returned by the API and verifying OpenAPI docs."""

import asyncio
import os
import pytest
from backend.app.runner.worker import get_worker


def test_secrets_never_returned_by_api(app_and_client, monkeypatch):
    """Verify that secret values stored in environment variables are NEVER returned by any API endpoint."""
    client, session_factory, artifacts_dir = app_and_client

    secret_key_value = "super-confidential-token-xyz-987654321"
    monkeypatch.setenv("MOCK_OPENAI_KEY", secret_key_value)

    # 1. Create Target referencing the env var
    res_t = client.post(
        "/api/targets",
        json={
            "name": "Secret Protected Target",
            "kind": "openai-compatible",
            "base_url": "https://api.openai.com/v1",
            "auth_env_var": "MOCK_OPENAI_KEY",
            "extra": {"model": "gpt-4o"},
        },
    )
    assert res_t.status_code == 201
    target_data = res_t.json()
    target_id = target_data["id"]
    assert target_data["auth_env_var"] == "MOCK_OPENAI_KEY"
    assert secret_key_value not in res_t.text

    # 2. Create Test Definition with this target
    res_d = client.post(
        "/api/test-definitions",
        json={
            "name": "Secret Protected Def",
            "tester": "dummy",
            "suites": ["smoke"],
            "config": {"duration_seconds": 0.1, "step_delay": 0.05},
            "target_id": target_id,
        },
    )
    assert res_d.status_code == 201
    def_id = res_d.json()["id"]
    assert secret_key_value not in res_d.text

    # 3. Trigger a Run from this definition
    res_r = client.post(f"/api/test-definitions/{def_id}/run")
    assert res_r.status_code == 201
    run_id = res_r.json()["id"]
    assert secret_key_value not in res_r.text

    # Execute run
    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True

    # 4. Check all API endpoints to guarantee secret is NEVER in the response
    endpoints_to_check = [
        f"/api/targets",
        f"/api/targets/{target_id}",
        f"/api/test-definitions",
        f"/api/test-definitions/{def_id}",
        f"/api/definitions",
        f"/api/definitions/{def_id}",
        f"/api/runs",
        f"/api/runs/{run_id}",
        f"/api/runs/{run_id}/results",
        f"/api/runs/{run_id}/logs",
        f"/api/results",
    ]

    for endpoint in endpoints_to_check:
        res = client.get(endpoint)
        assert res.status_code == 200, f"Endpoint {endpoint} failed: {res.text}"
        assert (
            secret_key_value not in res.text
        ), f"Secret leaked in endpoint {endpoint}!"

    # Get the result ID and check result endpoints
    res_results = client.get(f"/api/results?run_id={run_id}")
    assert res_results.status_code == 200
    results = res_results.json()
    assert len(results) >= 1
    result_id = results[0]["id"]

    for endpoint in [
        f"/api/results/{result_id}",
        f"/api/results/{result_id}/cases",
        f"/api/results/{result_id}/artifacts",
    ]:
        res = client.get(endpoint)
        assert res.status_code == 200, f"Endpoint {endpoint} failed: {res.text}"
        assert (
            secret_key_value not in res.text
        ), f"Secret leaked in endpoint {endpoint}!"


def test_openapi_documentation_enabled(app_and_client):
    """Verify that OpenAPI documentation is enabled and covers all core entities."""
    client, _, _ = app_and_client

    # 1. Check /openapi.json
    res_json = client.get("/openapi.json")
    assert res_json.status_code == 200
    schema = res_json.json()
    assert "openapi" in schema
    assert "paths" in schema
    paths = schema["paths"]

    # Verify key routes are present in OpenAPI schema
    assert "/api/targets" in paths
    assert "/api/targets/{target_id}" in paths
    assert "/api/test-definitions" in paths
    assert "/api/test-definitions/{definition_id}" in paths
    assert "/api/test-definitions/{definition_id}/run" in paths
    assert "/api/jobs" in paths
    assert "/api/jobs/{job_id}" in paths
    assert "/api/jobs/{job_id}/run" in paths
    assert "/api/testers" in paths
    assert "/api/testers/{name}" in paths
    assert "/api/runs" in paths
    assert "/api/runs/{run_id}" in paths
    assert "/api/runs/{run_id}/results" in paths
    assert "/api/runs/{run_id}/logs" in paths
    assert "/api/runs/{run_id}/logs/stream" in paths
    assert "/api/results" in paths
    assert "/api/results/{result_id}" in paths
    assert "/api/results/{result_id}/cases" in paths
    assert "/api/results/{result_id}/artifacts" in paths
    assert "/api/compare" in paths

    # 2. Check /docs (Swagger UI)
    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
    assert "swagger-ui" in res_docs.text.lower() or "html" in res_docs.headers.get("content-type", "").lower()

    # 3. Check /redoc (ReDoc)
    res_redoc = client.get("/redoc")
    assert res_redoc.status_code == 200
