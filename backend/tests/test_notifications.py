"""Tests for per-job failure and regression notifications (webhook and optional email)."""

import asyncio
import hashlib
import hmac
import json
from typing import Any
import httpx
import pytest

from backend.app.runner.worker import get_worker


def _create_definition(
    client,
    name: str = "Notif Test Def",
    tester: str = "dummy",
    suites: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    res = client.post(
        "/api/test-definitions",
        json={
            "name": name,
            "tester": tester,
            "suites": suites if suites is not None else ["smoke"],
            "config": (
                config
                if config is not None
                else {"duration_seconds": 0.05, "step_delay": 0.01, "create_artifacts": False}
            ),
        },
    )
    assert res.status_code == 201
    return res.json()


def _execute_next_queued_run() -> bool:
    worker = get_worker()
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(worker.poll_and_execute_once())
    finally:
        loop.close()


def test_job_notification_settings_crud_and_validation(app_and_client, monkeypatch):
    """Test creating, updating, validating, and retrieving per-job notification settings."""
    client, _, _ = app_and_client
    d1 = _create_definition(client, name="CRUD Def")

    secret_val = "top-secret-webhook-signing-key-999"
    monkeypatch.setenv("TEST_WEBHOOK_SECRET", secret_val)

    # 1. Create job with notifications dict
    res_create = client.post(
        "/api/jobs",
        json={
            "name": "Webhook Alert Job",
            "definition_ids": [d1["id"]],
            "notifications": {
                "enabled": True,
                "on_failure": True,
                "on_regression": True,
                "regression_threshold": 5.0,
                "metric_thresholds": {"latency_ms": 10.0},
                "webhook_url": "https://hooks.example.com/test-center",
                "webhook_secret_env_var": "TEST_WEBHOOK_SECRET",
                "webhook_headers": {"X-Custom-Team": "speech-infra"},
                "email_to": "dev1@example.com, dev2@example.com",
                "email_subject_prefix": "[Speech QA]",
            },
        },
    )
    assert res_create.status_code == 201
    job = res_create.json()
    job_id = job["id"]

    # Ensure raw secret value is never exposed in response
    assert secret_val not in res_create.text

    notifs = job["notifications"]
    assert notifs["enabled"] is True
    assert notifs["on_failure"] is True
    assert notifs["on_regression"] is True
    assert notifs["regression_threshold"] == 5.0
    assert notifs["metric_thresholds"] == {"latency_ms": 10.0}
    assert notifs["webhook_enabled"] is True
    assert notifs["webhook_url"] == "https://hooks.example.com/test-center"
    assert notifs["webhook_secret_env_var"] == "TEST_WEBHOOK_SECRET"
    assert notifs["webhook_headers"] == {"X-Custom-Team": "speech-infra"}
    assert notifs["email_enabled"] is True
    assert notifs["email_to"] == ["dev1@example.com", "dev2@example.com"]
    assert notifs["email_subject_prefix"] == "[Speech QA]"

    # 2. GET /api/jobs/{job_id}/notifications
    res_get_notif = client.get(f"/api/jobs/{job_id}/notifications")
    assert res_get_notif.status_code == 200
    assert res_get_notif.json()["webhook_url"] == "https://hooks.example.com/test-center"
    assert secret_val not in res_get_notif.text

    # 3. Update via PATCH /api/jobs/{job_id}/notifications
    res_patch_notif = client.patch(
        f"/api/jobs/{job_id}/notifications",
        json={
            "on_failure": False,
            "regression_threshold": 2.5,
            "email_enabled": False,
        },
    )
    assert res_patch_notif.status_code == 200
    updated_notif = res_patch_notif.json()
    assert updated_notif["on_failure"] is False
    assert updated_notif["on_regression"] is True
    assert updated_notif["regression_threshold"] == 2.5
    assert updated_notif["email_enabled"] is False
    assert updated_notif["webhook_url"] == "https://hooks.example.com/test-center"

    # 4. Validation errors (invalid webhook URL, invalid email, negative threshold)
    res_bad_url = client.post(
        "/api/jobs",
        json={
            "name": "Bad Webhook Job",
            "definition_ids": [d1["id"]],
            "notifications": {"webhook_url": "ftp://invalid-scheme.example.com"},
        },
    )
    assert res_bad_url.status_code == 422
    assert "http://" in res_bad_url.json()["detail"]

    res_bad_email = client.put(
        f"/api/jobs/{job_id}/notifications",
        json={"email_to": ["not-an-email-address"]},
    )
    assert res_bad_email.status_code == 422
    assert "email" in res_bad_email.json()["detail"].lower()

    res_bad_threshold = client.put(
        f"/api/jobs/{job_id}/notifications",
        json={"regression_threshold": -5.0},
    )
    assert res_bad_threshold.status_code == 422


def test_webhook_notification_triggers_on_failed_run(app_and_client, monkeypatch):
    """Verify webhook notification fires when a job run fails and includes HMAC signature."""
    client, _, _ = app_and_client
    captured_requests: list[dict[str, Any]] = []

    async def mock_post(self, url: str, **kwargs):
        content = kwargs.get("content", b"")
        headers = kwargs.get("headers", {})
        body_json = json.loads(content.decode("utf-8")) if isinstance(content, bytes) else kwargs.get("json")
        captured_requests.append({
            "url": str(url),
            "headers": dict(headers),
            "raw_body": content,
            "json": body_json,
        })
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setenv("JOB_HMAC_SECRET", "my-hmac-secret-123")

    d1 = _create_definition(
        client,
        name="Fail Trigger Def",
        suites=["smoke"],
        config={"duration_seconds": 0.02, "step_delay": 0.01, "fail_case": False, "create_artifacts": False},
    )

    res_job = client.post(
        "/api/jobs",
        json={
            "name": "Failure Alert Job",
            "definition_ids": [d1["id"]],
            "notifications": {
                "enabled": True,
                "on_failure": True,
                "on_regression": False,
                "webhook_url": "https://hooks.example.com/on-fail",
                "webhook_secret_env_var": "JOB_HMAC_SECRET",
            },
        },
    )
    assert res_job.status_code == 201
    job_id = res_job.json()["id"]

    # Run 1: Passes -> should NOT send webhook
    res_run1 = client.post(f"/api/jobs/{job_id}/run")
    assert res_run1.status_code == 201
    run1_id = res_run1.json()["id"]
    assert _execute_next_queued_run() is True
    assert len(captured_requests) == 0

    run1_detail = client.get(f"/api/runs/{run1_id}").json()
    assert run1_detail["notification_report"]["triggered"] is False
    assert run1_detail["notification_report"]["failed"] is False

    # Run 2: Fails via runtime override -> SHOULD send webhook
    res_run2 = client.post(
        f"/api/jobs/{job_id}/run",
        json={"config": {"fail_case": True}},
    )
    assert res_run2.status_code == 201
    run2_id = res_run2.json()["id"]
    assert _execute_next_queued_run() is True

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "https://hooks.example.com/on-fail"
    assert req["headers"]["X-TestCenter-Job-Id"] == str(job_id)
    assert req["headers"]["X-TestCenter-Run-Id"] == str(run2_id)

    # Verify HMAC-SHA256 signature header
    expected_sig = hmac.new(
        b"my-hmac-secret-123",
        req["raw_body"],
        hashlib.sha256,
    ).hexdigest()
    assert req["headers"]["X-TestCenter-Signature"] == f"sha256={expected_sig}"

    payload = req["json"]
    assert payload["event"] == "job_notification"
    assert payload["job_id"] == job_id
    assert payload["job_name"] == "Failure Alert Job"
    assert payload["run_id"] == run2_id
    assert payload["failed"] is True
    assert "failure" in payload["reasons"]
    assert len(payload["failure"]["failed_results"]) == 1
    assert len(payload["failure"]["failed_cases"]) == 1
    assert "Failure Alert Job" in payload["text"]

    run2_detail = client.get(f"/api/runs/{run2_id}").json()
    assert run2_detail["notification_report"]["triggered"] is True
    assert run2_detail["notification_report"]["reasons"] == ["failure"]
    assert run2_detail["notifications_sent"][0]["status"] == "sent"
    assert run2_detail["notifications_sent"][0]["status_code"] == 200


def test_webhook_notification_triggers_on_regression_reusing_compare_logic(app_and_client, monkeypatch):
    """Verify regression vs previous run reuses compare logic and respects regression_threshold."""
    client, _, _ = app_and_client
    captured_requests: list[dict[str, Any]] = []

    async def mock_post(self, url: str, **kwargs):
        content = kwargs.get("content", b"")
        body_json = json.loads(content.decode("utf-8")) if isinstance(content, bytes) else kwargs.get("json")
        captured_requests.append({"url": str(url), "json": body_json})
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    d1 = _create_definition(
        client,
        name="Perf Regression Def",
        suites=["smoke"],
        config={
            "duration_seconds": 0.02,
            "step_delay": 0.01,
            "fail_case": False,
            "create_artifacts": False,
            "throughput": 100.0,
            "latency_ms": 20.0,
        },
    )

    res_job = client.post(
        "/api/jobs",
        json={
            "name": "Regression Alert Job",
            "definition_ids": [d1["id"]],
            "notifications": {
                "enabled": True,
                "on_failure": False,
                "on_regression": True,
                "regression_threshold": 10.0,  # Require >= 10% degradation
                "webhook_url": "https://hooks.example.com/on-regression",
            },
        },
    )
    assert res_job.status_code == 201
    job_id = res_job.json()["id"]

    # Run 1 (Baseline): throughput=100.0 (better=higher), latency_ms=20.0 (better=lower)
    r1 = client.post(f"/api/jobs/{job_id}/run").json()
    assert _execute_next_queued_run() is True
    assert len(captured_requests) == 0

    # Run 2: Minor fluctuation within 10% threshold (throughput=95.0 [-5%], latency_ms=21.0 [+5%])
    # Should NOT trigger regression alert because threshold is 10%
    r2 = client.post(
        f"/api/jobs/{job_id}/run",
        json={"config": {"throughput": 95.0, "latency_ms": 21.0}},
    ).json()
    assert _execute_next_queued_run() is True
    assert len(captured_requests) == 0

    r2_detail = client.get(f"/api/runs/{r2['id']}").json()
    assert r2_detail["notification_report"]["triggered"] is False
    assert r2_detail["notification_report"]["regressed"] is False
    assert r2_detail["notification_report"]["baseline_run_id"] == r1["id"]

    # Run 3: Significant regression vs Run 2 (throughput drops 95 -> 60 [-36.8%], latency_ms rises 21 -> 35 [+66.7%])
    # Note: Run 3 passes (status='pass'), but regresses on metrics vs previous Run 2!
    r3 = client.post(
        f"/api/jobs/{job_id}/run",
        json={"config": {"throughput": 60.0, "latency_ms": 35.0}},
    ).json()
    assert _execute_next_queued_run() is True

    assert len(captured_requests) == 1
    payload = captured_requests[0]["json"]
    assert payload["job_id"] == job_id
    assert payload["run_id"] == r3["id"]
    assert payload["baseline_run_id"] == r2["id"]
    assert payload["failed"] is False
    assert payload["regressed"] is True
    assert payload["reasons"] == ["regression"]
    assert payload["compare_url"] == f"/compare?runs={r2['id']},{r3['id']}&baseline={r2['id']}"

    reg_names = {reg["name"] for reg in payload["regressions"] if reg.get("name")}
    assert "throughput" in reg_names
    assert "latency_ms" in reg_names

    tp_reg = next(reg for reg in payload["regressions"] if reg.get("name") == "throughput" and reg["type"] == "metric")
    assert tp_reg["baseline_value"] == 95.0
    assert tp_reg["value"] == 60.0
    assert tp_reg["better"] == "higher"
    assert tp_reg["pct_delta"] < -30.0


def test_email_notification_optional_and_test_endpoint(app_and_client, monkeypatch):
    """Verify optional SMTP email notification delivery and POST /api/jobs/{id}/notifications/test."""
    client, _, _ = app_and_client
    captured_webhooks: list[dict[str, Any]] = []
    sent_emails: list[Any] = []

    async def mock_post(self, url: str, **kwargs):
        content = kwargs.get("content", b"")
        body_json = json.loads(content.decode("utf-8")) if isinstance(content, bytes) else kwargs.get("json")
        captured_webhooks.append({"url": str(url), "json": body_json})
        return httpx.Response(200, json={"ok": True})

    class DummySMTP:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port
            self.started_tls = False
            self.logged_in = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def starttls(self):
            self.started_tls = True

        def login(self, user, password):
            self.logged_in = (user, password)

        def send_message(self, msg):
            sent_emails.append({
                "host": self.host,
                "port": self.port,
                "tls": self.started_tls,
                "login": self.logged_in,
                "subject": msg["Subject"],
                "from": msg["From"],
                "to": msg["To"],
                "body": msg.get_content(),
            })

    import smtplib
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setattr(smtplib, "SMTP", DummySMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.internal.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "mailer@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-secret-pass")
    monkeypatch.setenv("SMTP_FROM", "qa-alerts@example.com")

    d1 = _create_definition(
        client,
        name="Email & Webhook Def",
        suites=["smoke"],
        config={"duration_seconds": 0.02, "step_delay": 0.01, "fail_case": True, "create_artifacts": False},
    )

    res_job = client.post(
        "/api/jobs",
        json={
            "name": "Dual Channel Job",
            "definition_ids": [d1["id"]],
            "notifications": {
                "enabled": True,
                "on_failure": True,
                "on_regression": True,
                "webhook_url": "https://hooks.example.com/dual",
                "email_to": ["oncall@example.com", "lead@example.com"],
                "email_subject_prefix": "[QA Alert]",
            },
        },
    )
    assert res_job.status_code == 201
    job_id = res_job.json()["id"]

    # 1. Test endpoint POST /api/jobs/{job_id}/notifications/test
    res_test = client.post(f"/api/jobs/{job_id}/notifications/test")
    assert res_test.status_code == 200
    test_data = res_test.json()
    assert test_data["ok"] is True
    assert len(test_data["deliveries"]) == 2
    assert test_data["deliveries"][0]["channel"] == "webhook"
    assert test_data["deliveries"][0]["status"] == "sent"
    assert test_data["deliveries"][1]["channel"] == "email"
    assert test_data["deliveries"][1]["status"] == "sent"

    assert len(captured_webhooks) == 1
    assert captured_webhooks[0]["json"]["is_test"] is True
    assert len(sent_emails) == 1
    assert "[QA Alert] Test notification for job 'Dual Channel Job'" in sent_emails[0]["subject"]
    assert sent_emails[0]["to"] == "oncall@example.com, lead@example.com"
    assert sent_emails[0]["login"] == ("mailer@example.com", "smtp-secret-pass")

    # 2. Trigger real failing run -> both webhook and email should be sent in order (webhook first, email second)
    captured_webhooks.clear()
    sent_emails.clear()

    res_run = client.post(f"/api/jobs/{job_id}/run")
    assert res_run.status_code == 201
    run_id = res_run.json()["id"]
    assert _execute_next_queued_run() is True

    assert len(captured_webhooks) == 1
    assert len(sent_emails) == 1
    assert f"Job 'Dual Channel Job' (Run #{run_id}): FAILURE" in sent_emails[0]["subject"]
    assert "Failure:" in sent_emails[0]["body"]
