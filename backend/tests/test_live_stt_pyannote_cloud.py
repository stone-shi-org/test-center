"""pyannote.ai cloud target for the live-stt tester's accuracy_diarization suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.app.testers.base import TestContext, TesterConfigError
from backend.app.testers.live_stt import LiveSTTTester
from backend.app.testers.live_stt.client import LiveSTTClientError
from backend.app.testers.live_stt.pyannote_cloud import (
    PyannoteCloudClient,
    is_pyannote_ai_target,
    map_pyannote_output,
    pyannote_ai_models,
)
from backend.tests.test_live_stt_accuracy import _create_synthetic_notsofar_split

API_KEY = "sk-test-pyannote"
TARGET = {"name": "pyannote.ai", "kind": "pyannote-ai", "base_url": "https://api.pyannote.ai/v1"}

# MTG_90001 reference: Alice "Hello Bob, let's start the quarterly review." / Bob "Sounds good I have the metrics ready."
WORDS_A = ["Hello", "Bob,", "let's", "start", "the", "quarterly", "review."]
WORDS_B = ["Sounds", "good", "I", "have", "the", "metrics", "ready."]


def _job_output() -> dict[str, Any]:
    words = [
        {"start": 0.2 + i * 0.17, "end": 0.35 + i * 0.17, "text": t, "speaker": "SPEAKER_00"}
        for i, t in enumerate(WORDS_A)
    ] + [
        {"start": 1.6 + i * 0.17, "end": 1.75 + i * 0.17, "text": t, "speaker": "SPEAKER_01"}
        for i, t in enumerate(WORDS_B)
    ]
    return {
        "diarization": [
            {"speaker": "SPEAKER_00", "start": 0.2, "end": 1.4},
            {"speaker": "SPEAKER_01", "start": 1.6, "end": 2.8},
        ],
        "wordLevelTranscription": words,
        "turnLevelTranscription": [
            {"speaker": "SPEAKER_00", "start": 0.2, "end": 1.4, "text": " ".join(WORDS_A)},
            {"speaker": "SPEAKER_01", "start": 1.6, "end": 2.8, "text": " ".join(WORDS_B)},
        ],
    }


class FakePyannoteAI:
    """Minimal api.pyannote.ai: media/input -> presigned PUT -> diarize -> jobs (running, then succeeded)."""

    def __init__(self, job_status: str = "succeeded") -> None:
        self.calls: list[tuple[str, str, dict[str, str]]] = []
        self.jobs: list[dict[str, Any]] = []
        self.uploads: list[int] = []
        self.polls: dict[str, int] = {}
        self.job_status = job_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        self.calls.append((request.method, f"{url.host}{url.path}", dict(request.headers)))
        if url.host == "uploads.example":
            self.uploads.append(len(request.read()))
            return httpx.Response(200)
        if request.headers.get("authorization") != f"Bearer {API_KEY}":
            return httpx.Response(401, json={"message": "Unauthorized"})
        if url.path == "/v1/media/input":
            key = json.loads(request.read())["url"]
            assert key.startswith("media://test-center/")
            return httpx.Response(201, json={"url": f"https://uploads.example/put?key={key}"})
        if url.path == "/v1/diarize":
            body = json.loads(request.read())
            self.jobs.append(body)
            return httpx.Response(200, json={"jobId": f"job-{len(self.jobs)}", "status": "created"})
        if url.path.startswith("/v1/jobs/"):
            job_id = url.path.rsplit("/", 1)[-1]
            self.polls[job_id] = self.polls.get(job_id, 0) + 1
            if self.polls[job_id] == 1:
                return httpx.Response(200, json={"jobId": job_id, "status": "running"})
            if self.job_status != "succeeded":
                return httpx.Response(200, json={"jobId": job_id, "status": self.job_status, "error": "audio too short"})
            return httpx.Response(200, json={"jobId": job_id, "status": "succeeded", "output": _job_output()})
        return httpx.Response(404)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _run(tmp_path: Path, fake: FakePyannoteAI, **cfg: Any):
    ds_root = _create_synthetic_notsofar_split(tmp_path)
    logs: list[str] = []
    ctx = TestContext(
        run_id=300,
        artifacts_dir=tmp_path / "pyannote_ai",
        emit_stdout=False,
        on_event=lambda ev: logs.append(str(ev.get("message", ""))) if ev.get("type") == "log" else None,
    )
    config = {
        "suites": ["accuracy_diarization"],
        "dataset_dir": str(ds_root),
        "meetings": ["MTG_90001"],
        "api_key": API_KEY,
        "pyannote_ai_poll_interval_sec": 0,
        "use_cache": False,
        "_transport": fake.transport,
        **cfg,
    }
    return LiveSTTTester().run(config, dict(TARGET), ctx), logs


def test_is_pyannote_ai_target_detection():
    assert is_pyannote_ai_target({"kind": "pyannote-ai", "base_url": ""})
    assert is_pyannote_ai_target({"kind": "custom", "base_url": "https://api.pyannote.ai/v1"})
    assert is_pyannote_ai_target({"kind": "live-stt", "base_url": "http://x", "extra": {"backend": "pyannote_ai"}})
    assert not is_pyannote_ai_target({"kind": "live-stt", "base_url": "https://live-stt-2.local.shifamily.com"})


def test_catalog_flags_only_precision_models_as_one_pass():
    by_id = {m.id: m for m in pyannote_ai_models()}
    assert set(by_id) == {"pyannote.ai/precision-2", "pyannote.ai/precision-3", "pyannote.ai/community-1"}
    assert by_id["pyannote.ai/precision-2"].is_default
    assert by_id["pyannote.ai/precision-2"].produces_transcript
    assert by_id["pyannote.ai/precision-3"].produces_transcript
    assert not by_id["pyannote.ai/community-1"].produces_transcript  # no transcription on community-1


def test_map_output_uses_diarization_turns_and_same_speaker_words():
    out = _job_output()
    # An overlapping backchannel: SPEAKER_01 says "yeah" inside SPEAKER_00's turn
    out["diarization"].append({"speaker": "SPEAKER_01", "start": 0.9, "end": 1.1})
    out["wordLevelTranscription"].append({"start": 0.95, "end": 1.05, "text": "yeah", "speaker": "SPEAKER_01"})
    body = map_pyannote_output(out, "pyannote.ai/precision-2")
    segs = body["segments"]
    assert [(s["speaker"], s["start"], s["end"]) for s in segs] == [
        ("SPEAKER_00", 0.2, 1.4), ("SPEAKER_01", 0.9, 1.1), ("SPEAKER_01", 1.6, 2.8),
    ]
    assert segs[0]["text"] == " ".join(WORDS_A)  # "yeah" not stolen by the overlapping SPEAKER_00 turn
    assert segs[1]["text"] == "yeah"
    assert body["num_speakers"] == 2 and len(body["words"]) == 15 and body["task"] == "diarize"


def test_accuracy_diarization_scores_pyannote_ai_one_pass(tmp_path: Path):
    fake = FakePyannoteAI()
    res, logs = _run(tmp_path, fake, suites=["accuracy_diarization", "admin", "transcriptions_json"])

    assert [c.case_id for c in res.cases] == ["accuracy_diarization:pyannote.ai/precision-2:MTG_90001"]
    case = res.cases[0]
    assert case.status == "pass"
    assert case.detail["one_pass_transcript"] is True
    assert case.metrics["cpwer"]["value"] == 0.0 and case.metrics["wer"]["value"] == 0.0
    assert res.metrics["joint_cpwer"]["value"] == 0.0
    assert "der" in res.metrics and "joint_der" in res.metrics

    # The job asked for speaker-attributed transcription with the default STT model
    assert fake.jobs == [{
        "url": fake.jobs[0]["url"],
        "model": "precision-2",
        "transcription": True,
        "transcriptionConfig": {"model": "parakeet-tdt-0.6b-v3"},
    }]
    assert fake.uploads and fake.uploads[0] > 1000
    # API key never sent to the presigned upload host; no live-stt endpoints touched
    for method, where, headers in fake.calls:
        if where.startswith("uploads.example"):
            assert "authorization" not in headers
        else:
            assert where.startswith("api.pyannote.ai/v1/")
    assert any("pyannote.ai target: skipping transcriptions_json, admin" in m for m in logs), [m for m in logs if "skipping" in m]


def test_pyannote_ai_results_are_cached_per_transcribe_model(tmp_path: Path):
    fake = FakePyannoteAI()
    cache = str(tmp_path / "cache.sqlite")
    _run(tmp_path, fake, use_cache=True, cache_path=cache)
    _run(tmp_path, fake, use_cache=True, cache_path=cache)
    assert len(fake.jobs) == 1  # second run: cache hit, no new (billed) job
    _run(tmp_path, fake, use_cache=True, cache_path=cache, pyannote_ai_transcribe_model="faster-whisper-large-v3-turbo")
    assert len(fake.jobs) == 2
    assert fake.jobs[1]["transcriptionConfig"] == {"model": "faster-whisper-large-v3-turbo"}


def test_community_1_is_diarization_only_and_hints_are_sent(tmp_path: Path):
    fake = FakePyannoteAI()
    res, _ = _run(tmp_path, fake, models=["pyannote.ai/community-1"], use_num_speakers_hint=True)
    assert fake.jobs[0]["model"] == "community-1"
    assert "transcription" not in fake.jobs[0]
    assert fake.jobs[0]["minSpeakers"] == 2 and fake.jobs[0]["maxSpeakers"] == 2
    assert res.cases[0].detail["one_pass_transcript"] is False


def test_failed_job_is_errored_case_not_cached(tmp_path: Path):
    fake = FakePyannoteAI(job_status="failed")
    res, _ = _run(tmp_path, fake)
    assert res.cases[0].status == "error"
    assert "audio too short" in res.cases[0].detail["eval_message"]
    assert res.metrics["accuracy_diarization_errored_cases"]["value"] == 1


def test_only_non_diarization_suites_is_config_error(tmp_path: Path):
    with pytest.raises(TesterConfigError, match="only the accuracy_diarization suite"):
        _run(tmp_path, FakePyannoteAI(), suites=["accuracy_wer"])


def test_rate_limit_retried_with_retry_after():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(201, json={"url": "https://uploads.example/x"})

    sleeps: list[float] = []
    client = PyannoteCloudClient(api_key=API_KEY, sleep=sleeps.append)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        resp = client._request(http, "POST", "https://api.pyannote.ai/v1/media/input", json={})
    assert resp.status_code == 201 and hits["n"] == 2 and sleeps == [7.0]


def test_connection_check_and_missing_key():
    fake = FakePyannoteAI()
    ok, status, _, msg = PyannoteCloudClient(api_key=API_KEY, transport=fake.transport).test_connection()
    assert (ok, status) == (True, 201) and "no job submitted" in msg
    assert fake.jobs == []
    ok, status, _, msg = PyannoteCloudClient(api_key="wrong", transport=fake.transport).test_connection()
    assert (ok, status) == (False, 401) and "rejected" in msg

    with pytest.raises(LiveSTTClientError, match="no API key"):
        PyannoteCloudClient(api_key=None).diarize(b"RIFF", model="pyannote.ai/precision-2")


def test_compare_aligns_pyannote_ai_case_with_nemotron_case(tmp_path: Path):
    """Compare's model-agnostic alignment puts the pyannote.ai and Nemotron rows for one meeting together."""
    from types import SimpleNamespace

    from backend.app.api.compare import _extract_model_agnostic_case_id

    res, _ = _run(tmp_path, FakePyannoteAI())
    py_case = res.cases[0]
    nem_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    nem_case = SimpleNamespace(
        suite="accuracy_diarization",
        case_id=f"accuracy_diarization:{nem_id}:MTG_90001",
        detail={"model": nem_id},
    )
    py_ns = SimpleNamespace(suite=py_case.suite, case_id=py_case.case_id, detail=py_case.detail)
    assert _extract_model_agnostic_case_id(py_ns) == _extract_model_agnostic_case_id(nem_case) == "accuracy_diarization:MTG_90001"
