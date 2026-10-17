"""Tests for TES-35: Artifact viewer (HTML/JSON/audio/log) and download."""

import asyncio
import json
import os
from pathlib import Path
from backend.app.runner.worker import get_worker


def _create_completed_run(client) -> int:
    run_payload = {
        "tester": "dummy",
        "suites": ["smoke"],
        "config": {"duration_seconds": 0.05, "step_delay": 0.01},
    }
    res_run = client.post("/api/runs", json=run_payload)
    assert res_run.status_code == 201
    run_id = res_run.json()["id"]

    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True
    return run_id


def test_artifact_viewer_html_json_audio_log_and_download(app_and_client):
    client, _session_factory, artifacts_dir = app_and_client
    run_id = _create_completed_run(client)

    run_art_dir = Path(artifacts_dir) / str(run_id)
    run_art_dir.mkdir(parents=True, exist_ok=True)

    # Create HTML, JSON, JSONL, WAV audio, and log artifacts in data/artifacts/<run_id>/
    html_file = run_art_dir / "report.html"
    html_file.write_text(
        "<!doctype html><html><body><h1>Run Report</h1><script>alert('xss')</script></body></html>",
        encoding="utf-8",
    )

    json_file = run_art_dir / "metrics_tree.json"
    json_payload = {
        "run_id": run_id,
        "summary": {"wer": 0.042, "passed": True},
        "cases": [{"id": "c1", "wer": 0.0}, {"id": "c2", "wer": 0.084}],
    }
    json_file.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    jsonl_file = run_art_dir / "events.jsonl"
    jsonl_file.write_text(
        '{"seq": 1, "event": "start"}\n{"seq": 2, "event": "stop"}\n',
        encoding="utf-8",
    )

    wav_file = run_art_dir / "sample_output.wav"
    # Minimal valid 44-byte WAV header + 4 bytes silence
    wav_bytes = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00"
    )
    wav_file.write_bytes(wav_bytes)

    log_file = run_art_dir / "execution.log"
    log_file.write_text(
        "[10:00:01] INFO Starting benchmark suite\n"
        "[10:00:02] DEBUG Connected to websocket\n"
        "[10:00:03] WARN High latency spike detected (420ms)\n"
        "[10:00:04] ERROR Stream closed unexpectedly on utterance_04\n"
        "[10:00:05] INFO Completed run cleanup\n",
        encoding="utf-8",
    )

    # 1. List artifacts for the run (verifies auto-discovery of files in data/artifacts/<run_id>/)
    res_list = client.get(f"/api/runs/{run_id}/artifacts")
    assert res_list.status_code == 200
    items = res_list.json()
    by_name = {item["name"]: item for item in items}

    assert "report.html" in by_name
    assert "metrics_tree.json" in by_name
    assert "events.jsonl" in by_name
    assert "sample_output.wav" in by_name
    assert "execution.log" in by_name

    assert by_name["report.html"]["kind"] == "html"
    assert by_name["metrics_tree.json"]["kind"] == "json"
    assert by_name["events.jsonl"]["kind"] == "json"
    assert by_name["sample_output.wav"]["kind"] == "audio"
    assert by_name["execution.log"]["kind"] == "log"

    # 2. Safe HTML preview (sandboxed CSP headers + inline disposition)
    html_id = by_name["report.html"]["id"]
    res_html_preview = client.get(f"/api/runs/{run_id}/artifacts/{html_id}/preview")
    assert res_html_preview.status_code == 200
    assert "text/html" in res_html_preview.headers["content-type"]
    csp = res_html_preview.headers.get("content-security-policy", "")
    assert "sandbox" in csp
    assert "default-src 'none'" in csp
    assert res_html_preview.headers.get("x-content-type-options") == "nosniff"
    assert "inline" in res_html_preview.headers.get("content-disposition", "")
    assert "<h1>Run Report</h1>" in res_html_preview.text

    # Also test lookup by relative filename under data/artifacts/<run_id>/
    res_html_by_name = client.get(f"/api/runs/{run_id}/artifacts/report.html/preview")
    assert res_html_by_name.status_code == 200
    assert "sandbox" in res_html_by_name.headers.get("content-security-policy", "")

    # 3. JSON structured content endpoint (JSON object & JSONL)
    json_id = by_name["metrics_tree.json"]["id"]
    res_json_content = client.get(f"/api/runs/{run_id}/artifacts/{json_id}/content")
    assert res_json_content.status_code == 200
    json_body = res_json_content.json()
    assert json_body["kind"] == "json"
    assert json_body["json_valid"] is True
    assert json_body["json_data"]["summary"]["wer"] == 0.042
    assert len(json_body["json_data"]["cases"]) == 2

    res_jsonl_content = client.get(f"/api/runs/{run_id}/artifacts/events.jsonl/content")
    assert res_jsonl_content.status_code == 200
    jsonl_body = res_jsonl_content.json()
    assert jsonl_body["kind"] == "json"
    assert jsonl_body["json_valid"] is True
    assert isinstance(jsonl_body["json_data"], list)
    assert len(jsonl_body["json_data"]) == 2
    assert jsonl_body["json_data"][0]["event"] == "start"

    # 4. Audio preview and attachment download
    audio_id = by_name["sample_output.wav"]["id"]
    res_audio_preview = client.get(f"/api/runs/{run_id}/artifacts/{audio_id}/preview")
    assert res_audio_preview.status_code == 200
    assert res_audio_preview.headers["content-type"].startswith("audio/")
    assert res_audio_preview.content == wav_bytes

    res_audio_download = client.get(f"/api/runs/{run_id}/artifacts/{audio_id}/download")
    assert res_audio_download.status_code == 200
    assert "attachment" in res_audio_download.headers.get("content-disposition", "")
    assert res_audio_download.content == wav_bytes

    # 5. Log viewer content with search, level filter, and tail
    log_id = by_name["execution.log"]["id"]
    res_log_all = client.get(f"/api/runs/{run_id}/artifacts/{log_id}/content")
    assert res_log_all.status_code == 200
    log_all = res_log_all.json()
    assert log_all["kind"] == "log"
    assert log_all["total_lines"] == 5
    assert log_all["filtered_lines"] == 5

    res_log_err = client.get(
        f"/api/runs/{run_id}/artifacts/{log_id}/content?level=ERROR"
    )
    assert res_log_err.status_code == 200
    log_err = res_log_err.json()
    assert log_err["filtered_lines"] == 1
    assert "utterance_04" in log_err["lines"][0]

    res_log_search = client.get(
        f"/api/runs/{run_id}/artifacts/{log_id}/content?search=latency"
    )
    assert res_log_search.status_code == 200
    log_search = res_log_search.json()
    assert log_search["filtered_lines"] == 1
    assert "latency spike" in log_search["lines"][0]

    res_log_tail = client.get(
        f"/api/runs/{run_id}/artifacts/{log_id}/content?tail=2"
    )
    assert res_log_tail.status_code == 200
    log_tail = res_log_tail.json()
    assert len(log_tail["lines"]) == 2
    assert "utterance_04" in log_tail["lines"][0]
    assert "Completed run cleanup" in log_tail["lines"][1]


def test_artifact_security_and_isolation(app_and_client):
    client, _session_factory, artifacts_dir = app_and_client
    run_a = _create_completed_run(client)
    run_b = _create_completed_run(client)

    # Create a secret file outside the run's artifact directory
    secret_file = Path(artifacts_dir) / "secret_outside.txt"
    secret_file.write_text("TOP_SECRET", encoding="utf-8")

    # Create a symlink inside run_a's artifact dir pointing outside
    run_a_dir = Path(artifacts_dir) / str(run_a)
    symlink_path = run_a_dir / "escape_link.log"
    try:
        os.symlink(secret_file, symlink_path)
    except OSError:
        symlink_path = None

    # 1. Path traversal attempts via '..' must be rejected
    res_traversal = client.get(f"/api/runs/{run_a}/artifacts/..%2Fsecret_outside.txt/download")
    assert res_traversal.status_code in (400, 403, 404)

    # 2. Symlink escaping run directory must not be listed or served
    res_list = client.get(f"/api/runs/{run_a}/artifacts")
    assert res_list.status_code == 200
    names = [a["name"] for a in res_list.json()]
    assert "escape_link.log" not in names

    if symlink_path is not None:
        res_symlink = client.get(f"/api/runs/{run_a}/artifacts/escape_link.log/download")
        assert res_symlink.status_code in (400, 403, 404)

    # 3. Cross-run artifact ID access must be rejected with 404
    run_b_artifacts = client.get(f"/api/runs/{run_b}/artifacts").json()
    assert len(run_b_artifacts) >= 1
    run_b_art_id = run_b_artifacts[0]["id"]

    res_cross = client.get(f"/api/runs/{run_a}/artifacts/{run_b_art_id}/download")
    assert res_cross.status_code == 404
