"""Tests for Bamboo / CI integration: token-protected trigger, status polling, JUnit XML export, and status report-back."""

import asyncio
import base64
import hashlib
import hmac
import json
from typing import Any
import xml.etree.ElementTree as ET

import httpx
import pytest

from backend.app.db.models import CaseResult, Result, Run
from backend.app.runner.junit import build_run_junit_xml
from backend.app.runner.worker import get_worker


def _create_definition(
    client,
    name: str = "Bamboo Test Def",
    tester: str = "dummy",
    suites: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    res = client.post(
        "/api/test-definitions",
        json={
            "name": name,
            "tester": tester,
            "suites": suites if suites is not None else ["smoke", "load"],
            "config": (
                config
                if config is not None
                else {"duration_seconds": 0.05, "step_delay": 0.01, "create_artifacts": False}
            ),
        },
    )
    assert res.status_code == 201
    return res.json()


def _create_job(
    client,
    definition_ids: list[int],
    name: str = "bamboo-nightly-suite",
    run_mode: str = "sequential",
    enabled: bool = True,
) -> dict[str, Any]:
    res = client.post(
        "/api/jobs",
        json={
            "name": name,
            "description": "CI triggered regression job",
            "run_mode": run_mode,
            "enabled": enabled,
            "definition_ids": definition_ids,
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


def test_bamboo_token_protection_and_secret_non_leakage(app_and_client, monkeypatch):
    """Verify Bamboo endpoints enforce token authentication and never leak the secret value."""
    client, _, _ = app_and_client
    d1 = _create_definition(client)
    job = _create_job(client, [d1["id"]], name="auth-check-job")
    job_id = job["id"]

    # 1. When BAMBOO_API_TOKEN is not configured on server -> 401
    monkeypatch.delenv("BAMBOO_API_TOKEN", raising=False)
    monkeypatch.delenv("BAMBOO_TOKEN", raising=False)
    monkeypatch.delenv("CI_API_TOKEN", raising=False)
    monkeypatch.delenv("CI_TOKEN", raising=False)

    res_unconfigured = client.post(f"/api/bamboo/jobs/{job_id}/trigger")
    assert res_unconfigured.status_code == 401
    assert "not configured" in res_unconfigured.json()["detail"].lower()

    # 2. Configure secret token in environment
    secret_token = "bamboo-super-secret-token-xyz-987654"
    monkeypatch.setenv("BAMBOO_API_TOKEN", secret_token)

    # Missing token -> 401
    res_missing = client.post(f"/api/bamboo/jobs/{job_id}/trigger")
    assert res_missing.status_code == 401
    assert secret_token not in res_missing.text

    # Wrong token -> 401
    res_wrong = client.post(
        f"/api/bamboo/jobs/{job_id}/trigger",
        headers={"Authorization": "Bearer wrong-token-value"},
    )
    assert res_wrong.status_code == 401
    assert secret_token not in res_wrong.text

    # 3. Valid token via Authorization: Bearer <token>
    res_bearer = client.post(
        f"/api/bamboo/jobs/{job_id}/trigger",
        headers={"Authorization": f"Bearer {secret_token}"},
    )
    assert res_bearer.status_code == 201
    assert secret_token not in res_bearer.text
    run_id_1 = res_bearer.json()["run_id"]

    # 4. Valid token via X-Bamboo-Token header on poll endpoint
    res_header = client.get(
        f"/api/bamboo/runs/{run_id_1}",
        headers={"X-Bamboo-Token": secret_token},
    )
    assert res_header.status_code == 200
    assert res_header.json()["run_id"] == run_id_1
    assert secret_token not in res_header.text

    # 5. Valid token via Basic auth (username:token)
    basic_creds = base64.b64encode(f"bamboo:{secret_token}".encode("utf-8")).decode("ascii")
    res_basic = client.get(
        f"/api/bamboo/runs/{run_id_1}/status",
        headers={"Authorization": f"Basic {basic_creds}"},
    )
    assert res_basic.status_code == 200
    assert secret_token not in res_basic.text

    # 6. Valid token via query parameter ?token=...
    res_query = client.get(f"/api/bamboo/runs/{run_id_1}/junit.xml?token={secret_token}")
    assert res_query.status_code == 200
    assert secret_token not in res_query.text

    # 7. Check info endpoints never leak the raw secret
    res_info = client.get("/api/bamboo/info")
    assert res_info.status_code == 200
    assert res_info.json()["token_configured"] is True
    assert res_info.json()["token_env_var"] == "BAMBOO_API_TOKEN"
    assert secret_token not in res_info.text

    res_job_bamboo = client.get(f"/api/jobs/{job_id}/bamboo")
    assert res_job_bamboo.status_code == 200
    assert res_job_bamboo.json()["token_configured"] is True
    assert secret_token not in res_job_bamboo.text


def test_bamboo_trigger_by_id_and_name_and_poll_lifecycle(app_and_client, monkeypatch):
    """Test triggering a job from Bamboo by ID or name, polling status, and checking exit codes."""
    client, _, _ = app_and_client
    token = "ci-trigger-token-123"
    monkeypatch.setenv("BAMBOO_API_TOKEN", token)
    headers = {"Authorization": f"Bearer {token}"}

    d_pass = _create_definition(
        client,
        name="Pass Suite Def",
        suites=["smoke", "load"],
        config={"duration_seconds": 0.05, "step_delay": 0.01, "fail_case": False, "create_artifacts": False},
    )
    job = _create_job(client, [d_pass["id"]], name="speech-qa-gate")
    job_id = job["id"]

    # Trigger by job name with Bamboo build metadata
    res_trig = client.post(
        "/api/bamboo/jobs/speech-qa-gate/trigger",
        headers=headers,
        json={
            "plan_key": "TES-SPEECH",
            "build_number": 42,
            "build_result_key": "TES-SPEECH-42",
            "branch": "feature/bamboo-ci",
            "commit": "a1b2c3d4e5f6",
            "build_url": "https://bamboo.example.com/browse/TES-SPEECH-42",
        },
    )
    assert res_trig.status_code == 201
    trig_data = res_trig.json()
    run_id = trig_data["run_id"]
    assert trig_data["job_id"] == job_id
    assert trig_data["job_name"] == "speech-qa-gate"
    assert trig_data["status"] == "queued"
    assert trig_data["state"] == "Queued"
    assert trig_data["finished"] is False
    assert trig_data["passed"] is False
    assert trig_data["exit_code"] is None
    assert trig_data["poll_url"] == f"/api/bamboo/runs/{run_id}"
    assert trig_data["junit_url"] == f"/api/bamboo/runs/{run_id}/junit.xml"
    assert trig_data["bamboo"]["build_result_key"] == "TES-SPEECH-42"
    assert trig_data["bamboo"]["plan_key"] == "TES-SPEECH"
    assert trig_data["bamboo"]["build_number"] == 42

    # Poll while queued
    res_poll_queued = client.get(f"/api/bamboo/runs/{run_id}", headers=headers)
    assert res_poll_queued.status_code == 200
    assert res_poll_queued.json()["finished"] is False
    assert res_poll_queued.json()["state"] == "Queued"

    # Execute the queued run with worker
    assert _execute_next_queued_run() is True

    # Poll after completion
    res_poll_done = client.get(f"/api/bamboo/runs/{run_id}", headers=headers)
    assert res_poll_done.status_code == 200
    done_data = res_poll_done.json()
    assert done_data["status"] == "completed"
    assert done_data["result_status"] == "pass"
    assert done_data["state"] == "Successful"
    assert done_data["finished"] is True
    assert done_data["passed"] is True
    assert done_data["exit_code"] == 0
    assert done_data["counts"]["total"] >= 2
    assert done_data["counts"]["passed"] == done_data["counts"]["total"]
    assert done_data["counts"]["failed"] == 0
    assert done_data["counts"]["errors"] == 0
    assert done_data["duration_seconds"] is not None

    # Also test polling via job-scoped route and /api/ci alias
    res_job_poll = client.get(
        f"/api/ci/jobs/speech-qa-gate/runs/{run_id}",
        headers={"X-CI-Token": token},
    )
    assert res_job_poll.status_code == 200
    assert res_job_poll.json()["state"] == "Successful"

    # Verify Run detail endpoint includes bamboo metadata and junit_url
    res_run_detail = client.get(f"/api/runs/{run_id}")
    assert res_run_detail.status_code == 200
    run_detail = res_run_detail.json()
    assert run_detail["junit_url"] == f"/api/runs/{run_id}/junit.xml"
    assert run_detail["bamboo"]["build_result_key"] == "TES-SPEECH-42"
    assert run_detail["git_info"]["commit"] == "a1b2c3d4e5f6"
    assert run_detail["git_info"]["branch"] == "feature/bamboo-ci"


def test_bamboo_failed_run_poll_and_overlap_protection(app_and_client, monkeypatch):
    """Test polling a failed run (exit_code=1, state=Failed) and prevent_overlap behavior."""
    client, _, _ = app_and_client
    token = "ci-fail-token-456"
    monkeypatch.setenv("BAMBOO_API_TOKEN", token)
    headers = {"X-Bamboo-Token": token}

    d_fail = _create_definition(
        client,
        name="Failing Def",
        suites=["smoke"],
        config={"duration_seconds": 0.05, "step_delay": 0.01, "fail_case": True, "create_artifacts": False},
    )
    job = _create_job(client, [d_fail["id"]], name="failing-ci-job")

    # Trigger first run via generic /api/bamboo/trigger
    res1 = client.post(
        "/api/bamboo/trigger",
        headers=headers,
        json={"job_id": job["id"], "prevent_overlap": True},
    )
    assert res1.status_code == 201
    run_id = res1.json()["run_id"]

    # Trigger overlapping run with prevent_overlap=True -> 409 Conflict
    res_overlap = client.post(
        "/api/bamboo/trigger",
        headers=headers,
        json={"job_id": job["id"], "prevent_overlap": True},
    )
    assert res_overlap.status_code == 409

    # Execute run
    assert _execute_next_queued_run() is True

    # Poll failed run status
    res_poll = client.get(f"/api/bamboo/runs/{run_id}/status", headers=headers)
    assert res_poll.status_code == 200
    body = res_poll.json()
    assert body["finished"] is True
    assert body["passed"] is False
    assert body["state"] == "Failed"
    assert body["result_status"] == "fail"
    assert body["exit_code"] == 1
    assert body["counts"]["failed"] >= 1
    assert len(body["failed_cases"]) >= 1


def test_junit_xml_export_structure_and_bamboo_parser_compatibility(app_and_client, monkeypatch):
    """Verify JUnit XML export produces standard XML consumable by Bamboo's JUnit parser."""
    client, session_factory, _ = app_and_client
    token = "junit-export-token"
    monkeypatch.setenv("BAMBOO_API_TOKEN", token)

    # Create a run with diverse case statuses (pass, fail, error, skip) and control chars
    session = session_factory()
    try:
        run = Run(
            job_id=None,
            status="failed",
            config_snapshot={
                "job_name": "tts-benchmark-job",
                "tester": "tts",
                "bamboo": {
                    "plan_key": "TES-TTS",
                    "build_number": 108,
                    "build_result_key": "TES-TTS-108",
                    "branch": "main",
                },
            },
            target_snapshot={"name": "tts-staging", "base_url": "http://tts.internal:8000"},
            git_info={"commit": "deadbeef1234", "branch": "main"},
        )
        session.add(run)
        session.flush()

        result = Result(
            run_id=run.id,
            tester="tts",
            status="fail",
            metrics={"rtf": 0.42, "mos": 4.15},
            summary="3/4 cases passed, 1 failed",
        )
        session.add(result)
        session.flush()

        cases = [
            CaseResult(
                result_id=result.id,
                suite="latency",
                case_id="case_short_utterance",
                status="pass",
                metrics={"duration_ms": 125.0, "rtf": 0.35},
                detail={"Note": "Clean synthesis", "stdout": "Synthesized 2.1s audio"},
            ),
            CaseResult(
                result_id=result.id,
                suite="latency",
                case_id="case_long_paragraph",
                status="fail",
                metrics={"duration_ms": 840.0, "rtf": 1.25},
                detail={
                    "error": "RTF 1.25 exceeded threshold 1.0\x00\x07",
                    "traceback": "AssertionError: RTF 1.25 > 1.0 at line 42",
                },
            ),
            CaseResult(
                result_id=result.id,
                suite="robustness",
                case_id="case_invalid_ssml",
                status="error",
                metrics={"duration_seconds": 0.31},
                detail={
                    "error": "Timeout waiting for TTS worker",
                    "stderr": "WorkerReadTimeout: 30s elapsed",
                },
            ),
            CaseResult(
                result_id=result.id,
                suite="robustness",
                case_id="case_multilingual_zh",
                status="skip",
                metrics={},
                detail={"reason": "zh voice pack not installed on staging target"},
            ),
        ]
        session.add_all(cases)
        session.commit()
        run_id = run.id
        result_id = result.id
    finally:
        session.close()

    # 1. Fetch JUnit XML via token-protected Bamboo endpoint
    res_bamboo_xml = client.get(
        f"/api/bamboo/runs/{run_id}/junit.xml",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_bamboo_xml.status_code == 200
    assert "application/xml" in res_bamboo_xml.headers["content-type"]
    assert f"run-{run_id}-junit.xml" in res_bamboo_xml.headers.get(
        "content-disposition", ""
    )

    xml_text = res_bamboo_xml.text
    assert xml_text.startswith('<?xml version="1.0" encoding="UTF-8"?>')

    # Parse XML with ElementTree (validates well-formedness & stripped control characters)
    root = ET.fromstring(xml_text)
    assert root.tag == "testsuites"
    assert root.attrib["tests"] == "4"
    assert root.attrib["failures"] == "1"
    assert root.attrib["errors"] == "1"
    assert root.attrib["skipped"] == "1"

    suites = root.findall("testsuite")
    assert len(suites) == 2

    suite_by_name = {s.attrib["name"]: s for s in suites}
    assert "tts.latency" in suite_by_name
    assert "tts.robustness" in suite_by_name

    latency_suite = suite_by_name["tts.latency"]
    assert latency_suite.attrib["tests"] == "2"
    assert latency_suite.attrib["failures"] == "1"
    assert latency_suite.attrib["errors"] == "0"

    # Check suite properties include bamboo metadata and metrics
    props = {
        p.attrib["name"]: p.attrib["value"]
        for p in latency_suite.findall("./properties/property")
    }
    assert props["run_id"] == str(run_id)
    assert props["tester"] == "tts"
    assert props["bamboo.build_result_key"] == "TES-TTS-108"
    assert props["metric.rtf"] == "0.42"

    # Check testcases in latency suite
    latency_cases = {c.attrib["name"]: c for c in latency_suite.findall("testcase")}
    pass_case = latency_cases["case_short_utterance"]
    assert pass_case.find("failure") is None
    assert pass_case.find("error") is None
    assert float(pass_case.attrib["time"]) == pytest.approx(0.125, rel=1e-2)

    fail_case = latency_cases["case_long_paragraph"]
    failure_el = fail_case.find("failure")
    assert failure_el is not None
    assert "RTF 1.25 exceeded threshold 1.0" in failure_el.attrib["message"]
    assert "AssertionError" in (failure_el.text or "")

    # Check error and skipped cases in robustness suite
    rob_cases = {c.attrib["name"]: c for c in suite_by_name["tts.robustness"].findall("testcase")}
    err_case = rob_cases["case_invalid_ssml"]
    err_el = err_case.find("error")
    assert err_el is not None
    assert "Timeout waiting for TTS worker" in err_el.attrib["message"]

    skip_case = rob_cases["case_multilingual_zh"]
    skip_el = skip_case.find("skipped")
    assert skip_el is not None
    assert "zh voice pack not installed" in skip_el.attrib.get("message", "")

    # 2. Fetch JUnit XML via standard /api/runs/{run_id}/junit.xml and /api/results/{result_id}/junit.xml
    res_run_xml = client.get(f"/api/runs/{run_id}/junit.xml")
    assert res_run_xml.status_code == 200
    assert ET.fromstring(res_run_xml.text).attrib["tests"] == "4"

    res_result_xml = client.get(f"/api/results/{result_id}/junit.xml")
    assert res_result_xml.status_code == 200
    res_result_root = ET.fromstring(res_result_xml.text)
    assert res_result_root.attrib["tests"] == "4"


def test_junit_xml_artifact_auto_created_on_worker_completion(app_and_client, monkeypatch):
    """Verify worker automatically writes junit.xml artifact when a run completes."""
    client, _, _ = app_and_client
    token = "worker-junit-token"
    monkeypatch.setenv("BAMBOO_API_TOKEN", token)

    d1 = _create_definition(client, name="Artifact JUnit Def", suites=["smoke"])
    job = _create_job(client, [d1["id"]], name="artifact-junit-job")

    res_trig = client.post(
        f"/api/bamboo/jobs/{job['id']}/trigger",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_trig.status_code == 201
    run_id = res_trig.json()["run_id"]

    assert _execute_next_queued_run() is True

    # List run artifacts and verify junit.xml was registered
    res_arts = client.get(f"/api/runs/{run_id}/artifacts")
    assert res_arts.status_code == 200
    artifacts = res_arts.json()
    junit_arts = [a for a in artifacts if a["name"] == "junit.xml"]
    assert len(junit_arts) == 1
    assert junit_arts[0]["mime"] == "application/xml"

    # Download artifact content and verify it's valid JUnit XML
    art_id = junit_arts[0]["id"]
    res_dl = client.get(f"/api/runs/{run_id}/artifacts/{art_id}/download")
    assert res_dl.status_code == 200
    root = ET.fromstring(res_dl.text)
    assert root.tag == "testsuites"
    assert int(root.attrib["tests"]) >= 1


def test_bamboo_status_report_back_webhook_and_rest_api(app_and_client, monkeypatch):
    """Verify status report-back dispatches webhook callback and Bamboo REST API comment/labels."""
    client, _, _ = app_and_client
    token = "report-back-secret-token"
    bamboo_creds = "bamboo-bot:bamboo-pass-123"
    monkeypatch.setenv("BAMBOO_API_TOKEN", token)
    monkeypatch.setenv("BAMBOO_API_CREDS", bamboo_creds)

    captured_requests: list[dict[str, Any]] = []

    async def mock_post(self, url: str, **kwargs):
        content = kwargs.get("content", b"")
        headers = kwargs.get("headers", {})
        body_json = json.loads(content.decode("utf-8")) if isinstance(content, bytes) else kwargs.get("json", {})
        captured_requests.append(
            {
                "url": str(url),
                "headers": dict(headers),
                "raw_body": content,
                "json": body_json,
            }
        )
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    d1 = _create_definition(client, name="Report Back Def", suites=["smoke"])
    job = _create_job(client, [d1["id"]], name="report-back-job")

    # Trigger run with both callback_url and bamboo_base_url + build_result_key
    res_trig = client.post(
        f"/api/bamboo/jobs/{job['id']}/trigger",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "plan_key": "TES-MAIN",
            "build_number": 77,
            "build_result_key": "TES-MAIN-77",
            "callback_url": "https://ci-receiver.example.com/bamboo/callback",
            "bamboo_base_url": "https://bamboo.example.com",
            "report_comment": True,
            "report_labels": True,
        },
    )
    assert res_trig.status_code == 201
    run_id = res_trig.json()["run_id"]

    # Execute run -> should automatically report status back to callback_url and Bamboo REST API
    assert _execute_next_queued_run() is True

    urls_called = [req["url"] for req in captured_requests]
    assert "https://ci-receiver.example.com/bamboo/callback" in urls_called
    assert "https://bamboo.example.com/rest/api/latest/result/TES-MAIN-77/comment" in urls_called
    assert "https://bamboo.example.com/rest/api/latest/result/TES-MAIN-77/label" in urls_called

    # Verify HMAC signature on callback_url request
    callback_req = next(
        req for req in captured_requests if req["url"] == "https://ci-receiver.example.com/bamboo/callback"
    )
    cb_headers = callback_req["headers"]
    cb_payload = callback_req["json"]
    assert cb_payload["event"] == "test_center.bamboo.run_completed"
    assert cb_payload["run_id"] == run_id
    assert cb_payload["state"] == "Successful"
    assert cb_payload["passed"] is True
    assert cb_payload["exit_code"] == 0
    assert "X-Test-Center-Signature" in cb_headers
    expected_sig = hmac.new(
        token.encode("utf-8"),
        callback_req["raw_body"],
        hashlib.sha256,
    ).hexdigest()
    assert cb_headers["X-Test-Center-Signature"] == f"sha256={expected_sig}"

    # Verify Bamboo REST API comment contains summary and JUnit status
    comment_req = next(
        req
        for req in captured_requests
        if req["url"].endswith("/rest/api/latest/result/TES-MAIN-77/comment")
    )
    assert "PASSED" in comment_req["json"]["content"]
    assert f"Run #{run_id}" in comment_req["json"]["content"]

    # Verify run's bamboo_report is persisted and returned on poll
    res_poll = client.get(
        f"/api/bamboo/runs/{run_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_poll.status_code == 200
    report = res_poll.json()["bamboo_report"]
    assert report["reported"] is True
    assert report["ok"] is True
    assert len(report["deliveries"]) == 2


def test_synthetic_junit_testcase_on_crash_and_manual_report_endpoint(app_and_client, monkeypatch):
    """Verify crashed runs without case records still emit a failing synthetic testcase in JUnit XML."""
    client, _, _ = app_and_client
    token = "crash-and-report-token"
    monkeypatch.setenv("BAMBOO_API_TOKEN", token)
    headers = {"Authorization": f"Bearer {token}"}

    d_crash = _create_definition(
        client,
        name="Crashing Def",
        suites=["smoke"],
        config={"duration_seconds": 0.02, "step_delay": 0.01, "error": True, "create_artifacts": False},
    )
    job = _create_job(client, [d_crash["id"]], name="crash-ci-job")

    res_trig = client.post(
        f"/api/bamboo/jobs/{job['id']}/trigger",
        headers=headers,
    )
    assert res_trig.status_code == 201
    run_id = res_trig.json()["run_id"]

    # Execute crashing run
    assert _execute_next_queued_run() is True

    # Poll status -> Failed with exit_code=1
    res_poll = client.get(f"/api/bamboo/runs/{run_id}", headers=headers)
    assert res_poll.status_code == 200
    poll_body = res_poll.json()
    assert poll_body["state"] == "Failed"
    assert poll_body["result_status"] == "error"
    assert poll_body["exit_code"] == 1

    # Export JUnit XML -> should have 1 synthetic testcase with <error> element
    res_xml = client.get(f"/api/bamboo/runs/{run_id}/junit.xml", headers=headers)
    assert res_xml.status_code == 200
    root = ET.fromstring(res_xml.text)
    assert int(root.attrib["tests"]) == 1
    assert int(root.attrib["errors"]) == 1
    err_el = root.find(".//testcase/error")
    assert err_el is not None
    assert "return code 1" in err_el.attrib.get("message", "") or "crash" in err_el.attrib.get("message", "").lower()


