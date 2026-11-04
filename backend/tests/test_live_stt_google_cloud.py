"""Google Cloud (Gemini 3.5 Transcribe / Chirp 3) target for the live-stt tester's accuracy_diarization suite."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.app.testers.base import TestContext, TesterConfigError
from backend.app.testers.live_stt import LiveSTTTester
from backend.app.testers.live_stt.client import LiveSTTClientError
from backend.app.testers.live_stt.fixtures import make_pcm_wav_bytes
from backend.app.testers.live_stt.google_cloud import (
    GoogleCloudClient,
    google_cloud_models,
    is_google_cloud_target,
    load_service_account_info,
    map_chirp_results,
    map_gemini_response,
)
from backend.tests.test_live_stt_accuracy import _create_synthetic_notsofar_split

TOKEN = "ya29.test-token"
PROJECT = "demo-proj"
TARGET = {"name": "Google Cloud", "kind": "google-cloud", "base_url": "https://aiplatform.googleapis.com"}

# MTG_90001 reference: Alice "Hello Bob, let's start the quarterly review." / Bob "Sounds good I have the metrics ready."
WORDS_A = ["Hello", "Bob,", "let's", "start", "the", "quarterly", "review."]
WORDS_B = ["Sounds", "good", "I", "have", "the", "metrics", "ready."]


def _words(tokens: list[str], t0: float) -> list[dict[str, str]]:
    return [
        {"word": t, "startOffset": f"{t0 + i * 0.17:.3f}s", "endOffset": f"{t0 + 0.15 + i * 0.17:.3f}s"}
        for i, t in enumerate(tokens)
    ]


def _gemini_response() -> dict[str, Any]:
    parts = []
    for spk, toks, t0 in (("spk:0", WORDS_A, 0.2), ("spk:1", WORDS_B, 1.6)):
        text = " ".join(toks)
        parts.append({"text": text, "audioTranscription": {"text": text, "speakerLabel": spk, "words": _words(toks, t0)}})
    return {"candidates": [{"content": {"role": "model", "parts": parts}}], "usageMetadata": {"promptTokenCount": 75}}


def _chirp_results() -> list[dict[str, Any]]:
    words = [{**w, "speakerLabel": "1"} for w in _words(WORDS_A, 0.2)] + [
        {**w, "speakerLabel": "2"} for w in _words(WORDS_B, 1.6)
    ]
    return [{"alternatives": [{"transcript": " ".join(WORDS_A + WORDS_B), "words": words}], "languageCode": "en-US"}]


class FakeGoogle:
    """Minimal Vertex AI generateContent + Speech V2 recognize/batchRecognize/operations + GCS."""

    def __init__(self, batch_polls: int = 1) -> None:
        self.calls: list[tuple[str, str]] = []
        self.gemini: list[dict[str, Any]] = []
        self.chirp: list[dict[str, Any]] = []
        self.uploads: list[str] = []
        self.deletes: list[str] = []
        self.batch_polls = batch_polls
        self._polls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        self.calls.append((request.method, f"{url.host}{url.path}"))
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"error": {"code": 401, "message": "Request had invalid authentication"}})
        if url.host == "aiplatform.googleapis.com" and url.path.endswith("gemini-3.5-transcribe:generateContent"):
            assert url.path.startswith(f"/v1/projects/{PROJECT}/locations/global/")
            self.gemini.append(json.loads(request.read()))
            return httpx.Response(200, json=_gemini_response())
        if url.host == "aiplatform.googleapis.com" and url.path == "/v1beta1/publishers/google/models/gemini-3.5-transcribe":
            return httpx.Response(200, json={"name": "publishers/google/models/gemini-3.5-transcribe"})
        if url.host == "us-speech.googleapis.com":
            rec = f"/v2/projects/{PROJECT}/locations/us/recognizers/_"
            if url.path == f"{rec}:recognize":
                body = json.loads(request.read())
                self.chirp.append(body)
                return httpx.Response(200, json={"results": _chirp_results()})
            if url.path == f"{rec}:batchRecognize":
                body = json.loads(request.read())
                self.chirp.append(body)
                self._uri = body["files"][0]["uri"]
                return httpx.Response(200, json={"name": f"projects/{PROJECT}/locations/us/operations/op-1"})
            if url.path == f"/v2/projects/{PROJECT}/locations/us/operations/op-1":
                self._polls += 1
                if self._polls < self.batch_polls:
                    return httpx.Response(200, json={"name": "op-1", "done": False})
                return httpx.Response(200, json={"name": "op-1", "done": True, "response": {"results": {
                    self._uri: {"inlineResult": {"transcript": {"results": _chirp_results()}}}}}})
        if url.host == "storage.googleapis.com":
            if request.method == "POST" and url.path == "/upload/storage/v1/b/my-bucket/o":
                self.uploads.append(url.params["name"])
                return httpx.Response(200, json={"name": url.params["name"]})
            if request.method == "DELETE":
                self.deletes.append(url.path)
                return httpx.Response(204)
        return httpx.Response(404, json={"error": {"message": f"no route {url.host}{url.path}"}})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _client(fake: FakeGoogle, **kw: Any) -> GoogleCloudClient:
    return GoogleCloudClient(project=PROJECT, token_provider=lambda: TOKEN, transport=fake.transport,
                             poll_interval_sec=0, sleep=lambda s: None, **kw)


def _run(tmp_path: Path, fake: FakeGoogle, **cfg: Any):
    ds_root = _create_synthetic_notsofar_split(tmp_path)
    logs: list[str] = []
    ctx = TestContext(
        run_id=301,
        artifacts_dir=tmp_path / "google_cloud",
        emit_stdout=False,
        on_event=lambda ev: logs.append(str(ev.get("message", ""))) if ev.get("type") == "log" else None,
    )
    config = {
        "suites": ["accuracy_diarization"],
        "dataset_dir": str(ds_root),
        "meetings": ["MTG_90001"],
        "google_project": PROJECT,
        "google_poll_interval_sec": 0,
        "_google_token_provider": lambda: TOKEN,
        "use_cache": False,
        "_transport": fake.transport,
        **cfg,
    }
    return LiveSTTTester().run(config, dict(TARGET), ctx), logs


def test_is_google_cloud_target_detection():
    assert is_google_cloud_target({"kind": "google-cloud", "base_url": ""})
    assert is_google_cloud_target({"kind": "vertex-ai", "base_url": ""})
    assert is_google_cloud_target({"kind": "custom", "base_url": "https://us-central1-aiplatform.googleapis.com"})
    assert is_google_cloud_target({"kind": "custom", "base_url": "https://us-speech.googleapis.com/v2"})
    assert is_google_cloud_target({"kind": "live-stt", "base_url": "http://x", "extra": {"backend": "google_cloud"}})
    assert not is_google_cloud_target({"kind": "live-stt", "base_url": "https://live-stt-2.local.shifamily.com"})
    assert not is_google_cloud_target({"kind": "pyannote-ai", "base_url": "https://api.pyannote.ai/v1"})


def test_catalog_both_models_one_pass():
    by_id = {m.id: m for m in google_cloud_models()}
    assert set(by_id) == {"google/gemini-3.5-transcribe", "google/chirp_3"}
    assert by_id["google/gemini-3.5-transcribe"].is_default
    assert all(m.produces_transcript for m in by_id.values())
    assert by_id["google/chirp_3"].supports_num_speakers_hint
    assert not by_id["google/gemini-3.5-transcribe"].supports_num_speakers_hint


def test_map_gemini_parts_are_speaker_turns():
    body = map_gemini_response(_gemini_response(), "google/gemini-3.5-transcribe")
    segs = body["segments"]
    assert [(s["speaker"], s["start"], s["text"]) for s in segs] == [
        ("SPEAKER_00", 0.2, " ".join(WORDS_A)), ("SPEAKER_01", 1.6, " ".join(WORDS_B)),
    ]
    assert segs[0]["end"] == pytest.approx(0.2 + 0.15 + 6 * 0.17, abs=1e-3)
    assert body["num_speakers"] == 2 and len(body["words"]) == 14
    assert {w["speaker"] for w in body["words"]} == {"SPEAKER_00", "SPEAKER_01"}
    assert body["usage"] == {"promptTokenCount": 75}


def test_map_chirp_groups_consecutive_speaker_words():
    results = _chirp_results()
    # A backchannel from speaker 2 inside speaker 1's turn splits it into three turns
    results[0]["alternatives"][0]["words"].insert(3, {"word": "yeah", "startOffset": "0.700s",
                                                      "endOffset": "0.720s", "speakerLabel": "2"})
    body = map_chirp_results(results, "google/chirp_3")
    assert [(s["speaker"], s["text"]) for s in body["segments"]] == [
        ("SPEAKER_01", "Hello Bob, let's"), ("SPEAKER_02", "yeah"), ("SPEAKER_01", "start the quarterly review."),
        ("SPEAKER_02", " ".join(WORDS_B)),
    ]
    assert body["num_speakers"] == 2


def test_accuracy_diarization_scores_both_google_models(tmp_path: Path):
    fake = FakeGoogle()
    res, logs = _run(tmp_path, fake, models=["google/gemini-3.5-transcribe", "google/chirp_3"],
                     suites=["accuracy_diarization", "admin"])
    ids = sorted(c.case_id for c in res.cases)
    assert ids == ["accuracy_diarization:google/chirp_3:MTG_90001",
                   "accuracy_diarization:google/gemini-3.5-transcribe:MTG_90001"]
    for case in res.cases:
        assert case.status == "pass", case.detail.get("eval_message")
        assert case.detail["one_pass_transcript"] is True
        assert case.metrics["cpwer"]["value"] == 0.0 and case.metrics["wer"]["value"] == 0.0

    g = fake.gemini[0]
    assert g["generationConfig"]["audioTranscriptionConfig"] == {
        "diarization": True, "wordTimestamp": True, "languageCodes": ["en-US"],
    }
    assert base64.b64decode(g["contents"][0]["parts"][0]["inlineData"]["data"])[:4] == b"RIFF"
    c = fake.chirp[0]
    assert c["config"]["model"] == "chirp_3" and c["config"]["features"]["enableWordTimeOffsets"] is True
    assert "content" in c  # short meeting: synchronous recognize, no GCS
    assert fake.uploads == []
    assert all(not where.startswith("live-stt") for _, where in fake.calls)
    assert any("Google Cloud target: skipping admin" in m for m in logs)


def test_results_cached(tmp_path: Path):
    fake = FakeGoogle()
    cache = str(tmp_path / "cache.sqlite")
    _run(tmp_path, fake, use_cache=True, cache_path=cache)
    _run(tmp_path, fake, use_cache=True, cache_path=cache)
    assert len(fake.gemini) == 1  # second run: cache hit, nothing billed


def test_speaker_hint_sent_to_chirp(tmp_path: Path):
    fake = FakeGoogle()
    _run(tmp_path, fake, models=["google/chirp_3"], use_num_speakers_hint=True)
    assert fake.chirp[0]["config"]["features"]["diarizationConfig"] == {"minSpeakerCount": 2, "maxSpeakerCount": 2}


def test_chirp_long_audio_without_bucket_is_clear_error():
    long_wav = make_pcm_wav_bytes(duration_sec=61.0, sample_rate=16000)
    res = _client(FakeGoogle()).diarize(long_wav, model="google/chirp_3")
    assert res.status_code == 400
    assert "gcs_bucket" in res.body["error"]["message"]


def test_chirp_long_audio_batch_via_gcs_and_cleanup():
    fake = FakeGoogle(batch_polls=2)
    long_wav = make_pcm_wav_bytes(duration_sec=61.0, sample_rate=16000)
    res = _client(fake, gcs_bucket="gs://my-bucket/").diarize(long_wav, model="google/chirp_3", filename="MTG_1.wav")
    assert res.status_code == 200, res.body
    assert res.body["google_cloud"]["mode"] == "batch"
    assert [s["speaker"] for s in res.body["segments"]] == ["SPEAKER_01", "SPEAKER_02"]
    assert len(fake.uploads) == 1 and fake.uploads[0].startswith("test-center/") and fake.uploads[0].endswith("MTG_1.wav")
    assert fake.chirp[0]["files"][0]["uri"] == f"gs://my-bucket/{fake.uploads[0]}"
    assert len(fake.deletes) == 1  # uploaded object removed after the job


def test_api_errors_and_unknown_model_are_error_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"code": 403, "message": "Permission 'aiplatform.endpoints.predict' denied"}})

    client = GoogleCloudClient(project=PROJECT, token_provider=lambda: TOKEN, transport=httpx.MockTransport(handler))
    res = client.diarize(make_pcm_wav_bytes(duration_sec=1.0), model="google/gemini-3.5-transcribe")
    assert res.status_code == 403 and "aiplatform.endpoints.predict" in res.body["error"]["message"]
    res = client.diarize(b"RIFF", model="google/nope")
    assert res.status_code == 400 and "Unknown Google Cloud model" in res.body["error"]["message"]


def test_retry_on_429_then_success():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, json={"error": {"message": "quota"}})
        return httpx.Response(200, json=_gemini_response())

    sleeps: list[float] = []
    client = GoogleCloudClient(project=PROJECT, token_provider=lambda: TOKEN,
                               transport=httpx.MockTransport(handler), sleep=sleeps.append)
    res = client.diarize(make_pcm_wav_bytes(duration_sec=1.0))
    assert res.status_code == 200 and hits["n"] == 2 and sleeps == [3.0]


def test_failed_call_is_errored_case(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Audio exceeds 15 minutes"}})

    ds_root = _create_synthetic_notsofar_split(tmp_path)
    ctx = TestContext(run_id=302, artifacts_dir=tmp_path / "gc_err", emit_stdout=False)
    res = LiveSTTTester().run({
        "suites": ["accuracy_diarization"], "dataset_dir": str(ds_root), "meetings": ["MTG_90001"],
        "google_project": PROJECT, "_google_token_provider": lambda: TOKEN, "use_cache": False,
        "_transport": httpx.MockTransport(handler),
    }, dict(TARGET), ctx)
    assert res.cases[0].status == "error"
    assert "15 minutes" in res.cases[0].detail["eval_message"]


def test_only_non_diarization_suites_is_config_error(tmp_path: Path):
    with pytest.raises(TesterConfigError, match="Google Cloud targets support only the accuracy_diarization"):
        _run(tmp_path, FakeGoogle(), suites=["accuracy_wer"])


def test_credentials_parsing(tmp_path: Path):
    info = {"type": "service_account", "project_id": "p1", "private_key": "-----BEGIN PRIVATE KEY-----\nx\n",
            "client_email": "sa@p1.iam.gserviceaccount.com"}
    raw = json.dumps(info)
    key_file = tmp_path / "sa.json"
    key_file.write_text(raw)
    assert load_service_account_info(raw)["project_id"] == "p1"
    assert load_service_account_info(str(key_file))["project_id"] == "p1"
    assert load_service_account_info(base64.b64encode(raw.encode()).decode())["project_id"] == "p1"
    with pytest.raises(LiveSTTClientError, match="no credentials"):
        load_service_account_info(None)
    with pytest.raises(LiveSTTClientError, match="not a service-account key"):
        load_service_account_info(json.dumps({"type": "authorized_user"}))
    with pytest.raises(LiveSTTClientError, match="neither an existing key file"):
        load_service_account_info("/nope/missing.json")


def test_missing_credentials_is_config_error(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    ds_root = _create_synthetic_notsofar_split(tmp_path)
    ctx = TestContext(run_id=303, artifacts_dir=tmp_path / "gc_nokey", emit_stdout=False)
    res = LiveSTTTester().run({"suites": ["accuracy_diarization"], "dataset_dir": str(ds_root),
                               "meetings": ["MTG_90001"], "use_cache": False}, dict(TARGET), ctx)
    assert [c.status for c in res.cases] == ["error"]
    assert "has no credentials" in res.cases[0].detail["eval_message"]


def test_connection_check():
    fake = FakeGoogle()
    ok, status, _, msg = _client(fake).test_connection()
    assert (ok, status) == (True, 200) and "no audio submitted" in msg
    assert fake.gemini == [] and fake.chirp == []
    bad = GoogleCloudClient(project=PROJECT, token_provider=lambda: "wrong", transport=fake.transport)
    ok, status, _, msg = bad.test_connection()
    assert (ok, status) == (False, 401) and "invalid authentication" in msg
