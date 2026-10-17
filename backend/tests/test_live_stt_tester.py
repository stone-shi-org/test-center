"""Comprehensive test suite for the live-stt HTTP Tester plugin (TES-23).

Covers:
1. Built-in 16kHz mono 16-bit PCM WAV audio fixtures and TTS text fixtures
2. Response schema validators for /v1/models, /v1/audio/transcriptions (JSON, verbose_json, SSE),
   /v1/audio/diarization (speaker segments + word-level alignment), and /v1/audio/speech (WAV/MP3/PCM)
3. Latency percentiles, TTFT/TTFB, and Real-Time Factor (RTF) metrics calculation
4. /v1/models discovery and per-engine / per-model parametrization
5. Full LiveSTTTester plugin execution across all suites, error handling, and cancellation
6. End-to-end REST API + Worker subprocess integration test against a mock live-stt HTTP server
"""

from __future__ import annotations

import asyncio
import http.server
import io
import json
from pathlib import Path
import threading
from typing import Any
import wave
import httpx
import pytest

from backend.app.runner.worker import get_worker
from backend.app.testers.base import (
    TestContext,
    TesterCancelledError,
    TesterConfigError,
    TesterError,
)
from backend.app.testers.registry import get_tester, list_testers
from backend.app.testers.live_stt import (
    AudioFixture,
    BUILTIN_AUDIO_FIXTURES,
    BUILTIN_TTS_FIXTURES,
    DiscoveredModels,
    LiveSTTClientError,
    LiveSTTHTTPClient,
    LiveSTTTester,
    ModelInfo,
    SchemaCheckResult,
    TTSFixture,
    WordTimestamp,
    compute_latency_and_rtf_summary,
    compute_percentiles,
    compute_rtf,
    format_live_stt_report,
    get_audio_fixture,
    get_builtin_audio_fixtures,
    get_builtin_tts_fixtures,
    infer_model_info,
    normalize_live_stt_suites,
    parse_wav_audio_info,
    select_parametrized_models,
    synthesize_speech_like_wav,
    validate_diarization_response,
    validate_models_response,
    validate_transcription_json_response,
    validate_transcription_sse_response,
    validate_transcription_verbose_json_response,
    validate_tts_response,
)


# ===========================================================================
# Helpers for building mock WAV & /v1/models payloads
# ===========================================================================

def _build_mock_tts_wav(duration_sec: float = 0.8, sample_rate: int = 24000) -> bytes:
    """Create a valid mono 16-bit PCM WAV byte payload for TTS responses."""
    words = (
        WordTimestamp("hello", 0.05, min(0.35, duration_sec * 0.45)),
        WordTimestamp("speech", min(0.40, duration_sec * 0.5), max(0.45, duration_sec - 0.05)),
    )
    return synthesize_speech_like_wav(
        duration_sec=duration_sec,
        words=words,
        sample_rate=sample_rate,
    )


def _build_sample_models_payload() -> dict[str, Any]:
    """Return a realistic live-stt GET /v1/models payload with multiple engines."""
    return {
        "object": "list",
        "data": [
            {
                "id": "parakeet-ctc-0.6b",
                "object": "model",
                "created": 0,
                "owned_by": "live-stt",
                "type": "asr",
                "default": True,
                "engine": "parakeet",
                "sample_rate": 16000,
                "supports_streaming": True,
            },
            {
                "id": "parakeet-ctc-1.1b",
                "object": "model",
                "created": 0,
                "owned_by": "live-stt",
                "type": "asr",
                "default": False,
                "engine": "parakeet",
                "sample_rate": 16000,
                "supports_streaming": True,
            },
            {
                "id": "parakeet-tdt-0.6b-v3",
                "object": "model",
                "created": 0,
                "owned_by": "live-stt",
                "type": "asr",
                "default": False,
                "engine": "parakeet_tdt",
                "sample_rate": 16000,
                "supports_streaming": True,
            },
            {
                "id": "whisper-large-v3-turbo",
                "object": "model",
                "created": 0,
                "owned_by": "live-stt",
                "type": "asr",
                "default": False,
                "engine": "whisper",
                "sample_rate": 16000,
                "supports_streaming": True,
            },
            {
                "id": "pyannote/speaker-diarization-community-1",
                "object": "model",
                "created": 0,
                "owned_by": "live-stt",
                "type": "diarization",
                "default": True,
                "engine": "pyannote",
                "sample_rate": 16000,
                "supports_num_speakers_hint": True,
            },
            {
                "id": "kokoro-82m",
                "object": "model",
                "created": 0,
                "owned_by": "live-stt",
                "type": "tts",
                "default": True,
                "engine": "kokoro",
                "sample_rate": 24000,
                "default_voice": "af_heart",
                "supported_voices": ["af_heart", "af_bella", "am_adam"],
                "supports_streaming": True,
            },
            {
                "id": "piper-en-us-lessac-medium",
                "object": "model",
                "created": 0,
                "owned_by": "live-stt",
                "type": "tts",
                "default": False,
                "engine": "piper",
                "sample_rate": 22050,
                "default_voice": "en_US-lessac-medium",
                "supported_voices": ["en_US-lessac-medium"],
                "supports_streaming": True,
            },
        ],
    }


def _create_mock_live_stt_transport(
    *,
    models_payload: dict[str, Any] | None = None,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Create an httpx.MockTransport simulating all live-stt HTTP endpoints."""
    recorded_requests: list[httpx.Request] = []
    active_models = models_payload or _build_sample_models_payload()
    wav_24k = _build_mock_tts_wav(duration_sec=0.8, sample_rate=24000)
    wav_22k = _build_mock_tts_wav(duration_sec=0.8, sample_rate=22050)

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        path = request.url.path

        if request.method == "GET" and path.endswith("/v1/models"):
            return httpx.Response(200, json=active_models)

        if request.method == "POST" and path.endswith("/v1/audio/transcriptions"):
            raw_body = request.content.decode("latin-1", errors="ignore")
            is_stream = 'name="stream"\r\n\r\ntrue' in raw_body
            is_verbose = 'name="response_format"\r\n\r\nverbose_json' in raw_body

            if is_stream:
                sse_lines = (
                    'data: {"type": "transcript.text.delta", "delta": "Hello world "}\n\n'
                    'data: {"type": "transcript.text.done", "text": "Hello world from live stt."}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream; charset=utf-8"},
                    content=sse_lines.encode("utf-8"),
                )

            if is_verbose:
                return httpx.Response(
                    200,
                    json={
                        "task": "transcribe",
                        "language": "en",
                        "duration": 3.0,
                        "text": "The quick brown fox jumps over the lazy dog.",
                        "words": [
                            {"word": "The", "start": 0.12, "end": 0.30, "confidence": 0.99},
                            {"word": "quick", "start": 0.35, "end": 0.62, "confidence": 0.98},
                            {"word": "brown", "start": 0.68, "end": 0.98, "confidence": 0.97},
                            {"word": "fox", "start": 1.04, "end": 1.32, "confidence": 0.99},
                        ],
                    },
                )

            return httpx.Response(200, json={"text": "Hello world testing."})

        if request.method == "POST" and path.endswith("/v1/audio/diarization"):
            raw_body = request.content.decode("latin-1", errors="ignore")
            include_text = 'name="include_text"\r\n\r\ntrue' in raw_body
            seg0_text = "good morning everyone" if include_text else ""
            seg1_text = "hello thanks joining" if include_text else ""
            return httpx.Response(
                200,
                json={
                    "task": "diarize",
                    "model": "pyannote/speaker-diarization-community-1",
                    "duration": 4.0,
                    "num_speakers": 2,
                    "segments": [
                        {
                            "id": 0,
                            "speaker": "speaker_0",
                            "label": "SPEAKER_00",
                            "start": 0.10,
                            "end": 1.85,
                            "text": seg0_text,
                        },
                        {
                            "id": 1,
                            "speaker": "speaker_1",
                            "label": "SPEAKER_01",
                            "start": 2.05,
                            "end": 3.90,
                            "text": seg1_text,
                        },
                    ],
                    "speakers": [
                        {
                            "id": "speaker_0",
                            "label": "SPEAKER_00",
                            "total_speech_duration": 1.75,
                            "segment_count": 1,
                        },
                        {
                            "id": "speaker_1",
                            "label": "SPEAKER_01",
                            "total_speech_duration": 1.85,
                            "segment_count": 1,
                        },
                    ],
                },
            )

        if request.method == "POST" and path.endswith("/v1/audio/speech"):
            body = json.loads(request.content.decode("utf-8"))
            fmt = body.get("response_format", "wav")
            model_name = str(body.get("model", ""))
            if fmt == "mp3":
                return httpx.Response(
                    200,
                    headers={"content-type": "audio/mpeg"},
                    content=b"ID3\x04\x00\x00\x00\x00\x00\x10" + (b"\xff\xfb\x90\x64" * 80),
                )
            if fmt == "pcm":
                return httpx.Response(
                    200,
                    headers={"content-type": "audio/pcm"},
                    content=b"\x10\x02" * 12000,
                )
            chosen_wav = wav_22k if "piper" in model_name.lower() else wav_24k
            return httpx.Response(
                200,
                headers={"content-type": "audio/wav"},
                content=chosen_wav,
            )

        return httpx.Response(404, json={"detail": f"Unhandled route {path}"})

    return httpx.MockTransport(handler), recorded_requests


# ===========================================================================
# 1. Built-in Audio & TTS Fixtures Tests
# ===========================================================================

def test_builtin_audio_fixtures_valid_16khz_mono_pcm():
    """Verify all built-in audio fixtures produce valid 16kHz mono 16-bit PCM WAV data."""
    fixtures = get_builtin_audio_fixtures()
    assert len(fixtures) == 3
    ids = [f.id for f in fixtures]
    assert ids == ["short_speech", "medium_speech", "two_speaker_dialog"]

    for fx in fixtures:
        info = parse_wav_audio_info(fx.wav_bytes)
        assert info is not None, f"Fixture {fx.id} failed WAV header parse"
        assert info["sample_rate"] == 16000
        assert info["channels"] == 1
        assert info["sample_width"] == 2
        assert abs(info["duration_sec"] - fx.duration_sec) < 0.01
        assert info["n_frames"] == int(fx.duration_sec * 16000)

    # Verify two_speaker_dialog has word-level timestamp alignment fixtures
    dialog_fx = get_audio_fixture("two_speaker_dialog")
    assert dialog_fx.expected_speakers == 2
    words_payload = dialog_fx.words_payload()
    assert len(words_payload) == 6
    for w in words_payload:
        assert "text" in w and "start_sec" in w and "end_sec" in w
        assert w["end_sec"] > w["start_sec"]


def test_builtin_tts_fixtures_catalog():
    """Verify built-in TTS fixtures exist and have non-empty input prompts."""
    tts_fixtures = get_builtin_tts_fixtures()
    assert len(tts_fixtures) == 2
    assert [f.id for f in tts_fixtures] == ["tts_batch", "tts_stream"]
    assert tts_fixtures[0].stream is False
    assert tts_fixtures[1].stream is True
    for fx in tts_fixtures:
        assert len(fx.text.strip()) > 10


# ===========================================================================
# 2. Schema Validation Tests
# ===========================================================================

def test_validate_models_response():
    """Verify /v1/models schema validation for valid and invalid payloads."""
    valid_res = validate_models_response(_build_sample_models_payload())
    assert valid_res.valid is True

    # Invalid object type
    bad_obj = validate_models_response({"object": "not_list", "data": [{"id": "m1", "object": "model"}]})
    assert bad_obj.valid is False
    assert "object_is_list" in bad_obj.summary_message

    # Empty data list
    empty_data = validate_models_response({"object": "list", "data": []})
    assert empty_data.valid is False
    assert "data_is_non_empty_list" in empty_data.summary_message


def test_validate_transcription_json_and_verbose_json():
    """Verify /v1/audio/transcriptions JSON and verbose_json schema validators."""
    # Valid basic JSON
    res_ok = validate_transcription_json_response({"text": "Hello world"})
    assert res_ok.valid is True

    # Missing text field
    res_missing = validate_transcription_json_response({"transcript": "Hello"})
    assert res_missing.valid is False
    assert "text_field_is_string" in res_missing.summary_message

    # Error field present
    res_err = validate_transcription_json_response({"text": "", "error": "bad audio"})
    assert res_err.valid is False
    assert "no_error_field" in res_err.summary_message

    # Valid verbose_json
    verbose_payload = {
        "task": "transcribe",
        "language": "en",
        "duration": 1.5,
        "text": "Hello world",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
            {"word": "world", "start": 0.6, "end": 1.2, "confidence": 0.95},
        ],
    }
    res_verbose_ok = validate_transcription_verbose_json_response(verbose_payload)
    assert res_verbose_ok.valid is True

    # Invalid verbose_json with end < start in words
    bad_words_payload = {
        **verbose_payload,
        "words": [{"word": "Hello", "start": 1.0, "end": 0.4, "confidence": 0.9}],
    }
    res_bad_words = validate_transcription_verbose_json_response(bad_words_payload)
    assert res_bad_words.valid is False
    assert "word_items_schema" in res_bad_words.summary_message


def test_validate_transcription_sse_response():
    """Verify /v1/audio/transcriptions stream=true SSE schema validation."""
    events = [
        {"type": "transcript.text.delta", "delta": "Hello "},
        {"type": "transcript.text.done", "text": "Hello world"},
    ]
    res_ok = validate_transcription_sse_response(
        events=events,
        saw_done=True,
        content_type="text/event-stream; charset=utf-8",
        status_code=200,
    )
    assert res_ok.valid is True

    # Missing [DONE] sentinel
    res_no_done = validate_transcription_sse_response(
        events=events,
        saw_done=False,
        content_type="text/event-stream",
        status_code=200,
    )
    assert res_no_done.valid is False
    assert "sse_done_sentinel" in res_no_done.summary_message

    # Wrong Content-Type
    res_bad_ct = validate_transcription_sse_response(
        events=events,
        saw_done=True,
        content_type="application/json",
        status_code=200,
    )
    assert res_bad_ct.valid is False
    assert "content_type_event_stream" in res_bad_ct.summary_message


def test_validate_diarization_response():
    """Verify /v1/audio/diarization schema validator including speaker & word alignment checks."""
    valid_diar = {
        "task": "diarize",
        "model": "pyannote/speaker-diarization-community-1",
        "duration": 4.0,
        "num_speakers": 2,
        "segments": [
            {"id": 0, "speaker": "speaker_0", "label": "SPEAKER_00", "start": 0.1, "end": 1.8, "text": "Hello there"},
            {"id": 1, "speaker": "speaker_1", "label": "SPEAKER_01", "start": 2.0, "end": 3.9, "text": "General Kenobi"},
        ],
        "speakers": [
            {"id": "speaker_0", "label": "SPEAKER_00", "total_speech_duration": 1.7, "segment_count": 1},
            {"id": "speaker_1", "label": "SPEAKER_01", "total_speech_duration": 1.9, "segment_count": 1},
        ],
    }
    res_ok = validate_diarization_response(valid_diar, expect_text=True)
    assert res_ok.valid is True

    # Missing aligned text when expect_text=True
    no_text_diar = {
        **valid_diar,
        "num_speakers": 1,
        "segments": [
            {"id": 0, "speaker": "speaker_0", "label": "SPEAKER_00", "start": 0.1, "end": 1.8, "text": ""},
        ],
        "speakers": [
            {"id": "speaker_0", "label": "SPEAKER_00", "total_speech_duration": 1.7, "segment_count": 1},
        ],
    }
    res_no_text = validate_diarization_response(no_text_diar, expect_text=True)
    assert res_no_text.valid is False
    assert "segments_words_text_aligned" in res_no_text.summary_message

    # Speaker count mismatch with speakers list
    mismatch_diar = {
        **valid_diar,
        "num_speakers": 3,
    }
    res_mismatch = validate_diarization_response(mismatch_diar, expect_text=True)
    assert res_mismatch.valid is False
    assert "speakers_list_matches_num_speakers" in res_mismatch.summary_message


def test_validate_tts_response():
    """Verify /v1/audio/speech WAV, MP3, and PCM response validators."""
    wav_bytes = _build_mock_tts_wav(duration_sec=0.6, sample_rate=24000)
    res_wav, dur_wav = validate_tts_response(
        wav_bytes,
        "audio/wav",
        status_code=200,
        response_format="wav",
        expected_sample_rate=24000,
    )
    assert res_wav.valid is True
    assert abs(dur_wav - 0.6) < 0.02

    # Sample rate mismatch should fail sample_rate_matches_model
    res_sr_mismatch, _ = validate_tts_response(
        wav_bytes,
        "audio/wav",
        status_code=200,
        response_format="wav",
        expected_sample_rate=22050,
    )
    assert res_sr_mismatch.valid is False
    assert "sample_rate_matches_model" in res_sr_mismatch.summary_message

    # MP3 validation
    mp3_bytes = b"ID3\x04\x00\x00\x00\x00\x00\x10" + (b"\xff\xfb\x90\x64" * 50)
    res_mp3, dur_mp3 = validate_tts_response(mp3_bytes, "audio/mpeg", response_format="mp3")
    assert res_mp3.valid is True
    assert dur_mp3 > 0

    # PCM validation
    pcm_bytes = b"\x01\x02" * 2400
    res_pcm, dur_pcm = validate_tts_response(
        pcm_bytes,
        "audio/pcm",
        response_format="pcm",
        expected_sample_rate=24000,
    )
    assert res_pcm.valid is True
    assert dur_pcm == pytest.approx(0.1, rel=1e-3)


# ===========================================================================
# 3. Latency & Real-Time Factor (RTF) Metrics Tests
# ===========================================================================

def test_compute_rtf_and_summary_metrics():
    """Verify RTF computation and latency/RTF percentile summary aggregation."""
    # 300ms latency for 1.5s audio -> RTF = 0.2
    rtf = compute_rtf(300.0, 1.5)
    assert rtf == pytest.approx(0.2, rel=1e-4)
    assert compute_rtf(100.0, 0.0) == 0.0

    pct = compute_percentiles([100.0, 200.0, 300.0, 400.0, 500.0])
    assert pct[50] == 300.0
    assert pct[95] > 400.0
    assert pct[99] > 480.0

    summary = compute_latency_and_rtf_summary(
        latencies_ms=[150.0, 180.0, 400.0, 240.0],
        rtfs=[0.10, 0.12, 0.10, 0.30],
    )
    assert summary["latency_mean_ms"] == pytest.approx(242.5, rel=1e-3)
    assert summary["latency_min_ms"] == 150.0
    assert summary["latency_max_ms"] == 400.0
    assert summary["rtf_mean"] == pytest.approx(0.155, rel=1e-3)
    assert summary["rtf_min"] == 0.10
    assert summary["rtf_max"] == 0.30
    assert summary["rtf_p50"] == pytest.approx(0.11, rel=1e-3)


# ===========================================================================
# 4. /v1/models Discovery & Per-Engine Parametrization Tests
# ===========================================================================

def test_models_discovery_and_per_engine_parametrization():
    """Verify /v1/models discovery categorizes ASR/diarization/TTS models and parametrizes per engine."""
    transport, _ = _create_mock_live_stt_transport()
    client = LiveSTTHTTPClient(base_url="http://localhost:8000/v1", transport=transport)
    discovered = client.discover_models()

    assert discovered.status_code == 200
    assert discovered.total_count == 7
    assert len(discovered.asr_models) == 4
    assert len(discovered.diarization_models) == 1
    assert len(discovered.tts_models) == 2

    # 1. per_engine mode: should pick 1 representative per engine (parakeet, parakeet_tdt, whisper)
    asr_per_engine = select_parametrized_models(
        discovered.asr_models,
        model_type="asr",
        mode="per_engine",
    )
    assert [m.id for m in asr_per_engine] == [
        "parakeet-ctc-0.6b",       # default for 'parakeet' engine
        "parakeet-tdt-0.6b-v3",    # representative for 'parakeet_tdt' engine
        "whisper-large-v3-turbo",  # representative for 'whisper' engine
    ]

    tts_per_engine = select_parametrized_models(
        discovered.tts_models,
        model_type="tts",
        mode="per_engine",
    )
    assert [m.id for m in tts_per_engine] == [
        "kokoro-82m",
        "piper-en-us-lessac-medium",
    ]

    # 2. all mode: should return all 4 ASR models
    asr_all = select_parametrized_models(
        discovered.asr_models,
        model_type="asr",
        mode="all",
    )
    assert len(asr_all) == 4

    # 3. default mode: should return only the default ASR model
    asr_default = select_parametrized_models(
        discovered.asr_models,
        model_type="asr",
        mode="default",
    )
    assert [m.id for m in asr_default] == ["parakeet-ctc-0.6b"]

    # 4. explicit override
    asr_explicit = select_parametrized_models(
        discovered.asr_models,
        model_type="asr",
        mode="per_engine",
        explicit_models=["parakeet-ctc-1.1b", "custom-whisper-asr"],
    )
    assert [m.id for m in asr_explicit] == ["parakeet-ctc-1.1b", "custom-whisper-asr"]
    assert asr_explicit[0].engine == "parakeet"
    assert asr_explicit[1].engine == "whisper"


# ===========================================================================
# 5. LiveSTTTester Plugin Execution, Config Validation & Error Handling Tests
# ===========================================================================

def test_live_stt_tester_registration_and_aliases():
    """Verify LiveSTTTester is registered in the global registry with expected suites and aliases."""
    tester = get_tester("live-stt")
    assert isinstance(tester, LiveSTTTester)
    assert tester.name == "live-stt"
    assert tester.suites == ["transcriptions_json", "transcriptions_sse", "diarization", "tts"]

    # Verify aliases resolve to the same plugin
    assert get_tester("live_stt") is tester
    assert get_tester("live-stt-http") is tester
    assert get_tester("live_stt_http") is tester
    assert any(t.name == "live-stt" for t in list_testers())

    # Verify suite alias normalization
    assert normalize_live_stt_suites(["transcriptions", "speech"], tester.suites) == [
        "transcriptions_json",
        "transcriptions_sse",
        "tts",
    ]


def test_live_stt_tester_config_validation():
    """Verify config schema defaults and validation rules."""
    tester = LiveSTTTester()
    cfg = tester.validate_and_apply_config({})
    assert cfg["model_parametrization"] == "per_engine"
    assert cfg["repetitions"] == 1
    assert cfg["discover_models"] is True
    assert cfg["tts_response_format"] == "wav"
    assert cfg["tts_stream_format"] == "pcm"

    with pytest.raises(TesterConfigError, match="model_parametrization"):
        tester.validate_and_apply_config({"model_parametrization": "invalid_mode"})

    with pytest.raises(TesterConfigError, match="repetitions"):
        tester.validate_and_apply_config({"repetitions": 0})

    with pytest.raises(TesterConfigError, match="timeout_seconds"):
        tester.validate_and_apply_config({"timeout_seconds": 0})


def test_live_stt_tester_full_execution_per_engine(tmp_path: Path):
    """Run all 4 suites with per_engine parametrization and verify cases, metrics, and artifacts."""
    tester = LiveSTTTester()
    transport, requests = _create_mock_live_stt_transport()
    ctx = TestContext(
        run_id=501,
        artifacts_dir=tmp_path / "artifacts_501",
        emit_stdout=False,
    )

    result = tester.run(
        {
            "_transport": transport,
            "suites": ["transcriptions_json", "transcriptions_sse", "diarization", "tts"],
            "model_parametrization": "per_engine",
        },
        {"base_url": "http://localhost:8000/v1"},
        ctx,
    )

    assert result.status == "pass"
    # Expected cases:
    # 1. transcriptions_json: 3 ASR engines * 2 formats (json + verbose_json) = 6 cases
    # 2. transcriptions_sse: 3 ASR engines * 1 stream case = 3 cases
    # 3. diarization: 1 diarization engine * 2 modes (segments + words_alignment) = 2 cases
    # 4. tts: 2 TTS engines * 2 modes (batch + stream) = 4 cases
    # Total = 6 + 3 + 2 + 4 = 15 cases
    assert len(result.cases) == 15
    assert all(c.status == "pass" for c in result.cases)

    # Check summary metrics
    m = result.metrics
    assert m["cases_total"]["value"] == 15
    assert m["cases_passed"]["value"] == 15
    assert m["pass_rate"]["value"] == 100.0
    assert m["models_discovered"]["value"] == 7
    assert m["models_tested"]["value"] == 6  # 3 ASR + 1 diarization + 2 TTS
    assert m["engines_tested"]["value"] == 6  # parakeet, parakeet_tdt, whisper, pyannote, kokoro, piper
    assert m["rtf_mean"]["value"] >= 0.0
    assert "transcription_json_rtf_mean" in m
    assert "transcription_sse_ttft_mean_ms" in m
    assert "diarization_rtf_mean" in m
    assert "tts_rtf_mean" in m
    assert "tts_ttfb_mean_ms" in m

    # Check artifacts written to disk
    assert len(result.artifacts) == 2
    report_path = Path(result.artifacts[0].path)
    json_path = Path(result.artifacts[1].path)
    assert report_path.exists()
    assert json_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "LIVE-STT HTTP ENDPOINT FUNCTIONAL & LATENCY REPORT" in report_text
    assert "parakeet-ctc-0.6b" in report_text
    assert "kokoro-82m" in report_text

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_payload["tester"] == "live-stt"
    assert len(json_payload["cases"]) == 15


def test_live_stt_tester_error_and_cancellation_handling(tmp_path: Path):
    """Verify missing base_url, API error handling, and cancellation behavior."""
    tester = LiveSTTTester()
    transport, _ = _create_mock_live_stt_transport()

    # 1. Missing base_url raises TesterConfigError
    ctx1 = TestContext(run_id=502, artifacts_dir=tmp_path / "a1", emit_stdout=False)
    with pytest.raises(TesterConfigError, match="base_url"):
        tester.run({}, {}, ctx1)

    # 2. Pre-cancelled context raises TesterCancelledError
    ctx_cancel = TestContext(
        run_id=503,
        artifacts_dir=tmp_path / "a2",
        emit_stdout=False,
    )
    ctx_cancel.set_cancelled()
    with pytest.raises(TesterCancelledError):
        tester.run(
            {"_transport": transport, "suites": ["tts"]},
            {"base_url": "http://localhost:8000/v1"},
            ctx_cancel,
        )

    # 3. Transport failure raises TesterError
    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused to live-stt")

    ctx_err = TestContext(run_id=504, artifacts_dir=tmp_path / "a3", emit_stdout=False)
    with pytest.raises(TesterError, match="live-stt API error"):
        tester.run(
            {"_transport": httpx.MockTransport(failing_handler)},
            {"base_url": "http://localhost:8000/v1"},
            ctx_err,
        )


# ===========================================================================
# 6. End-to-End FastAPI + Worker Subprocess Integration Test
# ===========================================================================

def test_live_stt_worker_subprocess_end_to_end(app_and_client, monkeypatch):
    """End-to-end integration: trigger a live-stt run via REST API and execute in worker subprocess."""
    client, session_factory, artifacts_dir = app_and_client

    secret_token = "sk-live-stt-secret-bearer-token-999"
    monkeypatch.setenv("LIVE_STT_API_KEY", secret_token)

    received_auth_headers: list[str] = []
    models_payload = _build_sample_models_payload()
    wav_24k = _build_mock_tts_wav(duration_sec=0.75, sample_rate=24000)
    wav_22k = _build_mock_tts_wav(duration_sec=0.75, sample_rate=22050)

    class MockLiveSTTServerHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            received_auth_headers.append(self.headers.get("Authorization", ""))
            if self.path.endswith("/v1/models"):
                body = json.dumps(models_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            received_auth_headers.append(self.headers.get("Authorization", ""))
            content_len = int(self.headers.get("Content-Length", 0))
            raw_bytes = self.rfile.read(content_len)

            if self.path.endswith("/v1/audio/transcriptions"):
                raw_text = raw_bytes.decode("latin-1", errors="ignore")
                if 'name="stream"\r\n\r\ntrue' in raw_text:
                    sse_payload = (
                        'data: {"type": "transcript.text.delta", "delta": "Testing "}\n\n'
                        'data: {"type": "transcript.text.done", "text": "Testing live stt streaming."}\n\n'
                        "data: [DONE]\n\n"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Content-Length", str(len(sse_payload)))
                    self.end_headers()
                    self.wfile.write(sse_payload)
                    return

                if 'name="response_format"\r\n\r\nverbose_json' in raw_text:
                    resp_obj = {
                        "task": "transcribe",
                        "language": "en",
                        "duration": 3.0,
                        "text": "Testing live stt verbose json.",
                        "words": [
                            {"word": "Testing", "start": 0.05, "end": 0.45, "confidence": 0.99},
                            {"word": "live", "start": 0.48, "end": 0.80, "confidence": 0.98},
                            {"word": "stt", "start": 0.82, "end": 1.15, "confidence": 0.97},
                        ],
                    }
                else:
                    resp_obj = {"text": "Testing live stt json."}

                body = json.dumps(resp_obj).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path.endswith("/v1/audio/diarization"):
                raw_text = raw_bytes.decode("latin-1", errors="ignore")
                include_text = 'name="include_text"\r\n\r\ntrue' in raw_text
                resp_obj = {
                    "task": "diarize",
                    "model": "pyannote/speaker-diarization-community-1",
                    "duration": 4.0,
                    "num_speakers": 2,
                    "segments": [
                        {
                            "id": 0,
                            "speaker": "speaker_0",
                            "label": "SPEAKER_00",
                            "start": 0.1,
                            "end": 1.8,
                            "text": "Hello speaker one" if include_text else "",
                        },
                        {
                            "id": 1,
                            "speaker": "speaker_1",
                            "label": "SPEAKER_01",
                            "start": 2.0,
                            "end": 3.8,
                            "text": "Hello speaker two" if include_text else "",
                        },
                    ],
                    "speakers": [
                        {"id": "speaker_0", "label": "SPEAKER_00", "total_speech_duration": 1.7, "segment_count": 1},
                        {"id": "speaker_1", "label": "SPEAKER_01", "total_speech_duration": 1.8, "segment_count": 1},
                    ],
                }
                body = json.dumps(resp_obj).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path.endswith("/v1/audio/speech"):
                req_json = json.loads(raw_bytes.decode("utf-8"))
                fmt = req_json.get("response_format", "wav")
                model_name = str(req_json.get("model", ""))
                if fmt == "pcm":
                    pcm_payload = b"\x10\x02" * 12000
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/pcm")
                    self.send_header("Content-Length", str(len(pcm_payload)))
                    self.end_headers()
                    self.wfile.write(pcm_payload)
                    return

                chosen_wav = wav_22k if "piper" in model_name.lower() else wav_24k
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(chosen_wav)))
                self.end_headers()
                self.wfile.write(chosen_wav)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockLiveSTTServerHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        # 1. Verify /api/testers/live-stt metadata endpoint
        res_tester = client.get("/api/testers/live-stt")
        assert res_tester.status_code == 200
        tester_info = res_tester.json()
        assert tester_info["name"] == "live-stt"
        assert tester_info["suites"] == [
            "transcriptions_json",
            "transcriptions_sse",
            "diarization",
            "tts",
        ]

        # 2. Create Target pointing to mock live-stt server
        res_target = client.post(
            "/api/targets",
            json={
                "name": "Local live-stt Service",
                "kind": "live-stt",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "auth_env_var": "LIVE_STT_API_KEY",
            },
        )
        assert res_target.status_code == 201
        target_id = res_target.json()["id"]

        # 3. Create TestDefinition for live-stt
        res_def = client.post(
            "/api/test-definitions",
            json={
                "name": "live-stt HTTP Full Suite",
                "tester": "live-stt",
                "suites": ["transcriptions_json", "transcriptions_sse", "diarization", "tts"],
                "config": {
                    "model_parametrization": "per_engine",
                },
                "target_id": target_id,
            },
        )
        assert res_def.status_code == 201
        def_id = res_def.json()["id"]

        # 4. Trigger Run from TestDefinition
        res_run = client.post(f"/api/test-definitions/{def_id}/run")
        assert res_run.status_code == 201
        run_id = res_run.json()["id"]

        # 5. Execute queued run via worker subprocess
        worker = get_worker()
        loop = asyncio.new_event_loop()
        executed = loop.run_until_complete(worker.poll_and_execute_once())
        loop.close()
        assert executed is True

        # Verify bearer token was passed on every HTTP call and never leaked in API responses
        assert len(received_auth_headers) == 16  # 1 /v1/models + 15 test cases
        assert all(h == f"Bearer {secret_token}" for h in received_auth_headers)

        # 6. Verify Run status and metrics via /api/runs/{run_id}
        res_get_run = client.get(f"/api/runs/{run_id}")
        assert res_get_run.status_code == 200
        assert secret_token not in res_get_run.text
        run_data = res_get_run.json()
        assert run_data["status"] == "completed"

        results = run_data["results"]
        assert len(results) == 1
        res_item = results[0]
        assert res_item["status"] == "pass"
        metrics = res_item["metrics"]
        assert metrics["cases_total"]["value"] == 15
        assert metrics["cases_passed"]["value"] == 15
        assert metrics["pass_rate"]["value"] == 100.0
        assert metrics["rtf_mean"]["value"] >= 0.0
        assert metrics["latency_p95_ms"]["value"] > 0.0

        # 7. Verify per-case results via /api/results/{result_id}/cases
        result_id = res_item["id"]
        res_cases = client.get(f"/api/results/{result_id}/cases")
        assert res_cases.status_code == 200
        cases = res_cases.json()
        assert len(cases) == 15
        assert all(c["status"] == "pass" for c in cases)

        # 8. Verify generated artifacts via /api/results/{result_id}/artifacts
        res_artifacts = client.get(f"/api/results/{result_id}/artifacts")
        assert res_artifacts.status_code == 200
        artifact_names = {a["name"] for a in res_artifacts.json()}
        assert "live_stt_report.txt" in artifact_names
        assert "live_stt_results.json" in artifact_names

    finally:
        server.shutdown()
        server.server_close()
