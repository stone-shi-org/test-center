"""Comprehensive test suite for the live-stt HTTP Tester plugin (TES-23 & TES-24).

Covers:
1. Built-in 16kHz mono 16-bit PCM WAV audio fixtures, bad audio fixtures, and TTS text fixtures
2. Response schema validators for /v1/models, /v1/audio/transcriptions (JSON, verbose_json, SSE),
   /v1/audio/diarization (speaker segments + word-level alignment), /v1/audio/speech (WAV/MP3/PCM),
   read-only admin endpoints (/api/health, /api/stats, /api/version, /api/config, /api/settings,
   /api/backends, /api/diarization-engines, /api/tts-engines, /metrics, /dashboard), and error shapes
3. Latency percentiles, TTFT/TTFB, and Real-Time Factor (RTF) metrics calculation
4. /v1/models discovery and per-engine / per-model parametrization
5. Read-only admin/contract suite verification (confirming zero mutating calls by default),
   error-path suite (bad audio, unknown model, missing fields, status codes & error body shapes),
   and concurrency/admission limit suite (503 when over budget, slot recovery, stats counters)
6. End-to-end REST API + Worker subprocess integration test against a mock live-stt HTTP server
"""

from __future__ import annotations

import asyncio
import http.server
import io
import json
from pathlib import Path
import re
import threading
import time
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
    AdminEndpointResult,
    AudioFixture,
    BUILTIN_AUDIO_FIXTURES,
    BUILTIN_BAD_AUDIO_FIXTURES,
    BUILTIN_TTS_FIXTURES,
    BadAudioFixture,
    ConcurrentBurstResult,
    DiscoveredModels,
    LiveSTTClientError,
    LiveSTTHTTPClient,
    LiveSTTTester,
    ModelInfo,
    SchemaCheckResult,
    TTSFixture,
    WordTimestamp,
    build_multipart_body,
    compute_latency_and_rtf_summary,
    compute_percentiles,
    compute_rtf,
    extract_error_message,
    format_live_stt_report,
    get_audio_fixture,
    get_bad_audio_fixture,
    get_builtin_audio_fixtures,
    get_builtin_bad_audio_fixtures,
    get_builtin_tts_fixtures,
    infer_model_info,
    make_pcm_wav_bytes,
    normalize_live_stt_suites,
    parse_wav_audio_info,
    select_parametrized_models,
    synthesize_speech_like_wav,
    validate_backends_response,
    validate_config_response,
    validate_dashboard_html_response,
    validate_diarization_response,
    validate_engine_status_response,
    validate_error_response,
    validate_health_response,
    validate_models_response,
    validate_prometheus_metrics_response,
    validate_settings_response,
    validate_stats_response,
    validate_transcription_json_response,
    validate_transcription_sse_response,
    validate_transcription_verbose_json_response,
    validate_tts_response,
    validate_version_response,
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


def _parse_multipart_form(content: bytes, content_type: str) -> tuple[dict[str, str], dict[str, bytes]]:
    """Parse multipart/form-data fields and file payloads for mock live-stt validation."""
    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    m = re.search(r"boundary=([^\s;]+)", content_type or "")
    if not m:
        return fields, files
    boundary = m.group(1).strip('"').encode("utf-8")
    for part in content.split(b"--" + boundary):
        if not part or part.startswith(b"--") or b"\r\n\r\n" not in part:
            continue
        header_block, body = part.split(b"\r\n\r\n", 1)
        if body.endswith(b"\r\n"):
            body = body[:-2]
        header_str = header_block.decode("utf-8", errors="ignore")
        name_m = re.search(r'name="([^"]+)"', header_str)
        if not name_m:
            continue
        field_name = name_m.group(1)
        if 'filename="' in header_str:
            files[field_name] = body
        else:
            fields[field_name] = body.decode("utf-8", errors="replace")
    return fields, files


def _validate_16khz_mono_16bit_wav(wav_bytes: bytes) -> str | None:
    """Validate WAV bytes match live-stt's strict 16kHz mono 16-bit PCM requirement."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
    except Exception:
        return "could not parse WAV file"
    if sr != 16000 or ch != 1 or sw != 2:
        return (
            f"unsupported WAV format: {sr}Hz, {ch}ch, {sw * 8}-bit "
            "(only 16kHz mono 16-bit PCM accepted)"
        )
    if nframes == 0:
        return "WAV file contains no audio frames"
    return None


def _create_mock_live_stt_transport(
    *,
    models_payload: dict[str, Any] | None = None,
    max_concurrent_calls: int = 2,
    transcribe_hold_sec: float = 0.03,
    draining: bool = False,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Create an httpx.MockTransport simulating all live-stt HTTP, admin, error, and admission endpoints."""
    recorded_requests: list[httpx.Request] = []
    active_models = models_payload or _build_sample_models_payload()
    wav_24k = _build_mock_tts_wav(duration_sec=0.8, sample_rate=24000)
    wav_22k = _build_mock_tts_wav(duration_sec=0.8, sample_rate=22050)

    known_asr_ids = {
        m["id"] for m in active_models.get("data", []) if m.get("type") == "asr"
    } | {"realtime_eou_120m-v1"}
    known_diar_ids = {
        m["id"] for m in active_models.get("data", []) if m.get("type") == "diarization"
    }
    known_tts_ids = {
        m["id"] for m in active_models.get("data", []) if m.get("type") == "tts"
    } | {"kokoro-82m-v1"}

    state_lock = threading.Lock()
    state: dict[str, Any] = {
        "active_calls": 0,
        "active_workers": 0,
        "max_concurrent_calls": max_concurrent_calls,
        "max_workers": max_concurrent_calls + 2,
        "draining": draining,
        "transcribe_active": 0,
        "transcribe_completed_total": 0,
        "transcribe_failed_total": 0,
        "transcribe_rejected_vram_total": 0,
        "transcribe_rejected_capacity_total": 0,
        "diarization_completed_total": 0,
        "tts_completed_total": 0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        path = request.url.path

        # -------------------------------------------------------------------
        # Read-only Admin & Telemetry GET Endpoints
        # -------------------------------------------------------------------
        if request.method == "GET" and path.endswith("/v1/models"):
            return httpx.Response(200, json=active_models)

        if request.method == "GET" and path.endswith("/api/health"):
            with state_lock:
                used = state["active_calls"]
                total = state["max_concurrent_calls"]
                is_draining = state["draining"]
            status_str = "draining" if is_draining else "ok"
            http_code = 200 if status_str == "ok" else 503
            return httpx.Response(
                http_code,
                json={
                    "status": status_str,
                    "backend": "cuda",
                    "backends": {
                        "parakeet": "cuda",
                        "whisper": "cuda",
                        "parakeet_tdt": "cuda",
                    },
                    "model": "parakeet-ctc-0.6b",
                    "capacity": {
                        "used": used,
                        "total": total,
                        "admitting": (not is_draining) and (used < total),
                    },
                },
            )

        if request.method == "GET" and path.endswith("/api/stats"):
            with state_lock:
                snap = dict(state)
            return httpx.Response(
                200,
                json={
                    "active_calls": snap["active_calls"],
                    "active_workers": snap["active_workers"],
                    "max_concurrent_calls": snap["max_concurrent_calls"],
                    "max_workers": snap["max_workers"],
                    "draining": snap["draining"],
                    "gpu": {
                        "free_vram_mb": 12400,
                        "total_vram_mb": 16384,
                        "used_vram_mb": 3984,
                        "utilization_pct": 18,
                        "reserve_mb": 2048,
                    },
                    "diarization": {
                        "active": 0,
                        "completed_total": snap["diarization_completed_total"],
                        "failed_total": 0,
                        "rejected_vram_total": 0,
                        "active_requests": [],
                    },
                    "transcribe": {
                        "active": snap["transcribe_active"],
                        "completed_total": snap["transcribe_completed_total"],
                        "failed_total": snap["transcribe_failed_total"],
                        "rejected_vram_total": snap["transcribe_rejected_vram_total"],
                        "rejected_capacity_total": snap["transcribe_rejected_capacity_total"],
                        "active_requests": [],
                    },
                    "tts": {
                        "active": 0,
                        "completed_total": snap["tts_completed_total"],
                        "failed_total": 0,
                        "rejected_vram_total": 0,
                        "rejected_capacity_total": 0,
                        "active_requests": [],
                        "warm_worker": None,
                    },
                    "stt_worker_pool": {
                        "standby_workers": [],
                        "standby_count": 0,
                    },
                    "warmed_workers": [],
                },
            )

        if request.method == "GET" and path.endswith("/api/version"):
            return httpx.Response(
                200,
                json={
                    "hash": "a1b2c3d",
                    "timestamp": "2026-10-16T00:00:00Z",
                    "parakeet_ref": "v0.3.0",
                    "whisper_ref": "v1.1.0",
                },
            )

        if request.method == "GET" and path.endswith("/api/config"):
            return httpx.Response(
                200,
                json={
                    "host": "0.0.0.0",
                    "port": 8080,
                    "parakeet_backend": "cuda",
                    "whisper_backend": "cuda",
                    "parakeet_tdt_backend": "cuda",
                    "default_model": "parakeet-ctc-0.6b",
                    "max_concurrent_calls": max_concurrent_calls,
                    "reserve_slots": 2,
                    "gpu_vram_reserve_mb": 2048,
                },
            )

        if request.method == "GET" and path.endswith("/api/settings"):
            return httpx.Response(
                200,
                json={
                    "settings": {
                        "default_model": "parakeet-ctc-0.6b",
                        "max_concurrent_calls": max_concurrent_calls,
                        "parakeet_backend": "cuda",
                        "whisper_backend": "cuda",
                    },
                    "hot_reload_fields": ["default_model", "max_concurrent_calls", "reserve_slots"],
                    "restart_required_fields": ["host", "port", "parakeet_backend", "whisper_backend"],
                    "store_available": True,
                },
            )

        if request.method == "GET" and path.endswith("/api/backends"):
            return httpx.Response(
                200,
                json={
                    "store_available": True,
                    "registered": [
                        {"engine": "parakeet", "device": "cuda", "name": "parakeet-cuda"},
                        {"engine": "whisper", "device": "cuda", "name": "whisper-cuda"},
                        {"engine": "parakeet_tdt", "device": "cuda", "name": "parakeet_tdt-cuda"},
                    ],
                    "installed": [
                        {"engine": "parakeet", "device": "cuda", "installed": True, "active": True},
                        {"engine": "whisper", "device": "cuda", "installed": True, "active": True},
                    ],
                },
            )

        if request.method == "GET" and path.endswith("/api/diarization-engines"):
            return httpx.Response(
                200,
                json={
                    "engines": {
                        "pyannote": {
                            "installed": True,
                            "active_install": False,
                            "install_error": None,
                        },
                        "nemo_sortformer": {
                            "installed": False,
                            "active_install": False,
                            "install_error": None,
                        },
                    }
                },
            )

        if request.method == "GET" and path.endswith("/api/tts-engines"):
            return httpx.Response(
                200,
                json={
                    "engines": {
                        "kokoro": {
                            "installed": True,
                            "active_install": False,
                            "install_error": None,
                        },
                        "piper": {
                            "installed": True,
                            "active_install": False,
                            "install_error": None,
                        },
                    }
                },
            )

        if request.method == "GET" and path.endswith("/metrics"):
            prom_text = (
                "# HELP live_stt_streams_active Currently active WebSocket audio streams.\n"
                "# TYPE live_stt_streams_active gauge\n"
                "live_stt_streams_active 0.0\n"
                "# HELP live_stt_transcribe_requests_total Total /v1/audio/transcriptions requests by model and outcome.\n"
                "# TYPE live_stt_transcribe_requests_total counter\n"
                'live_stt_transcribe_requests_total{model="parakeet-ctc-0.6b",outcome="completed"} 5.0\n'
                "# HELP live_stt_diarization_requests_total Total /v1/audio/diarization requests by model and outcome.\n"
                "# TYPE live_stt_diarization_requests_total counter\n"
                'live_stt_diarization_requests_total{model="pyannote/speaker-diarization-community-1",outcome="completed"} 2.0\n'
                "# HELP live_stt_tts_requests_total Total /v1/audio/speech requests by model and outcome.\n"
                "# TYPE live_stt_tts_requests_total counter\n"
                'live_stt_tts_requests_total{model="kokoro-82m",outcome="completed"} 2.0\n'
                "# HELP live_stt_admission_rejections_total Total requests rejected by admission gates.\n"
                "# TYPE live_stt_admission_rejections_total counter\n"
                'live_stt_admission_rejections_total{endpoint="transcribe",reason="capacity"} 0.0\n'
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/plain; version=0.0.4; charset=utf-8"},
                content=prom_text.encode("utf-8"),
            )

        if request.method == "GET" and path.endswith("/dashboard"):
            html_doc = "<!DOCTYPE html><html><head><title>live-stt Dashboard</title></head><body><h1>live-stt</h1></body></html>"
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=html_doc.encode("utf-8"),
            )

        # -------------------------------------------------------------------
        # POST /v1/audio/transcriptions (with validation & admission budget)
        # -------------------------------------------------------------------
        if request.method == "POST" and path.endswith("/v1/audio/transcriptions"):
            ct = request.headers.get("content-type", "")
            fields, files = _parse_multipart_form(request.content, ct)
            if "file" not in files or not files["file"]:
                return httpx.Response(
                    400,
                    json={"error": {"message": "missing 'file' field in multipart upload"}},
                )

            wav_err = _validate_16khz_mono_16bit_wav(files["file"])
            if wav_err:
                return httpx.Response(400, json={"error": {"message": wav_err}})

            model_id = fields.get("model", "parakeet-ctc-0.6b")
            if model_id not in known_asr_ids:
                return httpx.Response(
                    400,
                    json={"error": {"message": f"unknown model {model_id!r}"}},
                )

            resp_fmt = fields.get("response_format", "json")
            if resp_fmt not in ("json", "verbose_json"):
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": f"unsupported response_format {resp_fmt!r} (expected 'json' or 'verbose_json')"
                        }
                    },
                )

            is_stream = fields.get("stream", "false").lower() == "true"
            is_verbose = resp_fmt == "verbose_json"

            # Admission budget check (WorkerBudget)
            with state_lock:
                if state["draining"]:
                    state["transcribe_rejected_capacity_total"] += 1
                    return httpx.Response(
                        503,
                        json={"error": {"message": "server is draining, not accepting new calls"}},
                    )
                if state["active_calls"] >= state["max_concurrent_calls"]:
                    state["transcribe_rejected_capacity_total"] += 1
                    return httpx.Response(
                        503,
                        json={"error": {"message": "at capacity, try again shortly"}},
                    )
                state["active_calls"] += 1
                state["active_workers"] += 1
                state["transcribe_active"] += 1

            try:
                if transcribe_hold_sec > 0:
                    time.sleep(transcribe_hold_sec)

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
            finally:
                with state_lock:
                    state["active_calls"] -= 1
                    state["active_workers"] -= 1
                    state["transcribe_active"] -= 1
                    state["transcribe_completed_total"] += 1

        # -------------------------------------------------------------------
        # POST /v1/audio/diarization (with validation)
        # -------------------------------------------------------------------
        if request.method == "POST" and path.endswith("/v1/audio/diarization"):
            ct = request.headers.get("content-type", "")
            fields, files = _parse_multipart_form(request.content, ct)
            if "file" not in files or not files["file"]:
                return httpx.Response(
                    400,
                    json={"error": {"message": "missing 'file' field in multipart upload"}},
                )

            wav_err = _validate_16khz_mono_16bit_wav(files["file"])
            if wav_err:
                return httpx.Response(400, json={"error": {"message": wav_err}})

            model_id = fields.get("model", "pyannote/speaker-diarization-community-1")
            if model_id not in known_diar_ids:
                return httpx.Response(
                    400,
                    json={"error": {"message": f"Unknown diarization model {model_id!r}"}},
                )

            if "words" in fields and fields["words"]:
                try:
                    parsed_words = json.loads(fields["words"])
                    if not isinstance(parsed_words, list):
                        raise ValueError("words must be a JSON array")
                except Exception as exc:
                    return httpx.Response(
                        400,
                        json={"error": {"message": f"'words' field is not valid JSON: {exc}"}},
                    )

            for int_field in ("num_speakers", "min_speakers", "max_speakers"):
                if int_field in fields and fields[int_field]:
                    try:
                        val = int(fields[int_field])
                        if val < 1:
                            raise ValueError("must be >= 1")
                    except Exception:
                        return httpx.Response(
                            400,
                            json={"error": {"message": f"invalid integer for {int_field!r}: {fields[int_field]!r}"}},
                        )

            include_text = fields.get("include_text", "false").lower() == "true"
            seg0_text = "good morning everyone" if include_text else ""
            seg1_text = "hello thanks joining" if include_text else ""
            with state_lock:
                state["diarization_completed_total"] += 1
            return httpx.Response(
                200,
                json={
                    "task": "diarize",
                    "model": model_id,
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

        # -------------------------------------------------------------------
        # POST /v1/audio/speech (with validation)
        # -------------------------------------------------------------------
        if request.method == "POST" and path.endswith("/v1/audio/speech"):
            try:
                body = json.loads(request.content.decode("utf-8"))
            except Exception as exc:
                return httpx.Response(400, json={"error": f"Invalid JSON body: {exc}"})

            input_text = body.get("input")
            if not isinstance(input_text, str) or not input_text.strip():
                return httpx.Response(
                    400,
                    json={"error": "'input' field is required and must be a non-empty string"},
                )

            model_name = str(body.get("model", "kokoro-82m"))
            if model_name not in known_tts_ids:
                return httpx.Response(
                    400,
                    json={"error": f"Unknown TTS model: {model_name!r}"},
                )

            fmt = body.get("response_format", "wav")
            with state_lock:
                state["tts_completed_total"] += 1
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

        return httpx.Response(404, json={"error": "not found"})

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
    assert tester.suites == [
        "transcriptions_json",
        "transcriptions_sse",
        "diarization",
        "tts",
        "admin",
        "error_handling",
        "concurrency",
    ]

    # Verify aliases resolve to the same plugin
    assert get_tester("live_stt") is tester
    assert get_tester("live-stt-http") is tester
    assert get_tester("live_stt_http") is tester
    assert get_tester("live-stt-admin") is tester
    assert get_tester("live-stt-contract") is tester
    assert any(t.name == "live-stt" for t in list_testers())

    # Verify suite alias normalization
    assert normalize_live_stt_suites(["transcriptions", "speech"], tester.suites) == [
        "transcriptions_json",
        "transcriptions_sse",
        "tts",
    ]
    assert normalize_live_stt_suites(["contract"], tester.suites) == [
        "admin",
        "error_handling",
        "concurrency",
    ]
    assert normalize_live_stt_suites(["models", "errors", "admission"], tester.suites) == [
        "admin",
        "error_handling",
        "concurrency",
    ]


def test_live_stt_tester_config_validation():
    """Verify config schema defaults and validation rules."""
    tester = LiveSTTTester()
    cfg = tester.validate_and_apply_config({})
    assert cfg["model_parametrization"] == "per_engine"
    assert cfg["repetitions"] == 1
    assert cfg["discover_models"] is True
    assert cfg["include_extended_admin"] is True
    assert cfg["allow_mutating_admin"] is False
    assert cfg["tts_response_format"] == "wav"
    assert cfg["tts_stream_format"] == "pcm"

    with pytest.raises(TesterConfigError, match="model_parametrization"):
        tester.validate_and_apply_config({"model_parametrization": "invalid_mode"})

    with pytest.raises(TesterConfigError, match="repetitions"):
        tester.validate_and_apply_config({"repetitions": 0})

    with pytest.raises(TesterConfigError, match="timeout_seconds"):
        tester.validate_and_apply_config({"timeout_seconds": 0})


def test_live_stt_tester_full_execution_per_engine(tmp_path: Path):
    """Run all 4 core audio suites with per_engine parametrization and verify cases, metrics, and artifacts."""
    tester = LiveSTTTester()
    transport, requests = _create_mock_live_stt_transport(transcribe_hold_sec=0.0)
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
    transport, _ = _create_mock_live_stt_transport(transcribe_hold_sec=0.0)

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
            "admin",
            "error_handling",
            "concurrency",
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


# ===========================================================================
# 7. TES-24: Admin/Contract, Error-Path, and Concurrency Limit Tests
# ===========================================================================

def test_bad_audio_fixtures_catalog():
    """Verify built-in bad audio fixtures produce corrupt, 8kHz, stereo, and 8-bit WAVs."""
    bad_fixtures = get_builtin_bad_audio_fixtures()
    assert len(bad_fixtures) == 4
    assert [f.id for f in bad_fixtures] == [
        "corrupt_non_wav",
        "wrong_sample_rate_8khz",
        "stereo_2ch_wav",
        "wrong_bit_depth_8bit",
    ]

    corrupt = get_bad_audio_fixture("corrupt_non_wav")
    assert parse_wav_audio_info(corrupt.wav_bytes) is None
    assert _validate_16khz_mono_16bit_wav(corrupt.wav_bytes) == "could not parse WAV file"

    fx_8k = get_bad_audio_fixture("wrong_sample_rate_8khz")
    info_8k = parse_wav_audio_info(fx_8k.wav_bytes)
    assert info_8k is not None and info_8k["sample_rate"] == 8000
    assert "8000Hz" in (_validate_16khz_mono_16bit_wav(fx_8k.wav_bytes) or "")

    fx_stereo = get_bad_audio_fixture("stereo_2ch_wav")
    info_stereo = parse_wav_audio_info(fx_stereo.wav_bytes)
    assert info_stereo is not None and info_stereo["channels"] == 2
    assert "2ch" in (_validate_16khz_mono_16bit_wav(fx_stereo.wav_bytes) or "")

    fx_8bit = get_bad_audio_fixture("wrong_bit_depth_8bit")
    info_8bit = parse_wav_audio_info(fx_8bit.wav_bytes)
    assert info_8bit is not None and info_8bit["sample_width"] == 1
    assert "8-bit" in (_validate_16khz_mono_16bit_wav(fx_8bit.wav_bytes) or "")


def test_validate_admin_and_contract_schemas():
    """Verify schema validators for /v1/models (strict), /api/health, /api/stats, /metrics, and error shapes."""
    # 1. Strict /v1/models catalog validation
    strict_ok = validate_models_response(
        _build_sample_models_payload(),
        status_code=200,
        content_type="application/json",
        strict_catalog=True,
    )
    assert strict_ok.valid is True

    # 2. /api/health: 200 when status=='ok' (even at capacity used==total!), 503 when draining/degraded
    health_at_cap = {
        "status": "ok",
        "backend": "cuda",
        "backends": {"parakeet": "cuda", "whisper": "cuda", "parakeet_tdt": "cuda"},
        "model": "parakeet-ctc-0.6b",
        "capacity": {"used": 4, "total": 4, "admitting": False},
    }
    assert validate_health_response(health_at_cap, status_code=200).valid is True
    assert validate_health_response(health_at_cap, status_code=503).valid is False

    health_draining = {
        **health_at_cap,
        "status": "draining",
    }
    assert validate_health_response(health_draining, status_code=503).valid is True
    assert validate_health_response(health_draining, status_code=200).valid is False

    # 3. /api/stats validation
    valid_stats = {
        "active_calls": 1,
        "active_workers": 1,
        "max_concurrent_calls": 4,
        "max_workers": 6,
        "draining": False,
        "gpu": {
            "free_vram_mb": 10240,
            "total_vram_mb": 16384,
            "used_vram_mb": 6144,
            "utilization_pct": 25,
            "reserve_mb": 2048,
        },
        "diarization": {
            "active": 0,
            "completed_total": 3,
            "failed_total": 0,
            "rejected_vram_total": 0,
            "active_requests": [],
        },
        "transcribe": {
            "active": 1,
            "completed_total": 10,
            "failed_total": 0,
            "rejected_vram_total": 0,
            "rejected_capacity_total": 2,
            "active_requests": [{"id": "req-1"}],
        },
        "tts": {
            "active": 0,
            "completed_total": 5,
            "failed_total": 0,
            "rejected_vram_total": 0,
            "rejected_capacity_total": 0,
            "active_requests": [],
            "warm_worker": None,
        },
        "stt_worker_pool": {"standby_workers": [], "standby_count": 0},
        "warmed_workers": [],
    }
    assert validate_stats_response(valid_stats, status_code=200).valid is True
    bad_stats = {**valid_stats, "gpu": {"reserve_mb": -1}}
    assert validate_stats_response(bad_stats, status_code=200).valid is False

    # 4. /metrics Prometheus validation & forbidden PII label check
    valid_prom = (
        "# HELP live_stt_streams_active Active streams\n"
        "# TYPE live_stt_streams_active gauge\n"
        "live_stt_streams_active 0\n"
        "live_stt_transcribe_requests_total 1\n"
        "live_stt_diarization_requests_total 1\n"
        "live_stt_tts_requests_total 1\n"
        "live_stt_admission_rejections_total 0\n"
    )
    assert validate_prometheus_metrics_response(valid_prom, status_code=200).valid is True

    pii_prom = valid_prom + 'live_stt_streams_active{stream_id="secret-call-123"} 1\n'
    res_pii = validate_prometheus_metrics_response(pii_prom, status_code=200)
    assert res_pii.valid is False
    assert "no_high_cardinality_pii_labels" in res_pii.summary_message

    # 5. Error body shape validation (nested vs flat)
    nested_err = {"error": {"message": "unsupported WAV format: 8000Hz, 1ch, 16-bit"}}
    assert validate_error_response(
        nested_err,
        status_code=400,
        expected_status=400,
        shape="nested",
        expected_substring="8000Hz",
    ).valid is True
    assert validate_error_response(
        nested_err,
        status_code=400,
        expected_status=400,
        shape="flat",
    ).valid is False

    flat_err = {"error": "Unknown TTS model: 'bad-tts'"}
    assert validate_error_response(
        flat_err,
        status_code=400,
        expected_status=400,
        shape="flat",
        expected_substring="Unknown TTS model",
    ).valid is True


def test_live_stt_admin_suite_read_only_no_mutating_calls(tmp_path: Path):
    """Run the 'admin' suite and verify all read-only endpoints pass and zero mutating calls occur."""
    tester = LiveSTTTester()
    transport, requests = _create_mock_live_stt_transport(transcribe_hold_sec=0.0)
    ctx = TestContext(
        run_id=601,
        artifacts_dir=tmp_path / "artifacts_601",
        emit_stdout=False,
    )

    result = tester.run(
        {
            "_transport": transport,
            "suites": ["admin"],
            "discover_models": False,
            "include_extended_admin": True,
        },
        {"base_url": "http://localhost:8000"},
        ctx,
    )

    assert result.status == "pass"
    assert len(result.cases) == 11
    assert all(c.status == "pass" for c in result.cases)
    case_ids = [c.case_id for c in result.cases]
    assert case_ids == [
        "admin:v1_models",
        "admin:api_health",
        "admin:api_stats",
        "admin:api_version",
        "admin:api_config",
        "admin:api_settings",
        "admin:api_backends",
        "admin:api_diarization_engines",
        "admin:api_tts_engines",
        "admin:metrics",
        "admin:dashboard",
    ]

    # Verify strictly read-only behavior: every single HTTP request was GET, zero POST/PUT/DELETE/PATCH
    assert len(requests) == 11
    assert all(req.method == "GET" for req in requests)
    assert all(c.detail.get("read_only") is True and c.detail.get("mutating") is False for c in result.cases)
    assert "admin_latency_mean_ms" in result.metrics


def test_live_stt_error_handling_suite_bad_audio_unknown_model_missing_fields(tmp_path: Path):
    """Run the 'error_handling' suite and verify bad audio, unknown models, missing fields, and error shapes."""
    tester = LiveSTTTester()
    transport, requests = _create_mock_live_stt_transport(transcribe_hold_sec=0.0)
    ctx = TestContext(
        run_id=602,
        artifacts_dir=tmp_path / "artifacts_602",
        emit_stdout=False,
    )

    result = tester.run(
        {
            "_transport": transport,
            "suites": ["error_handling"],
            "discover_models": False,
        },
        {"base_url": "http://localhost:8000"},
        ctx,
    )

    assert result.status == "pass"
    assert len(result.cases) == 15
    assert all(c.status == "pass" for c in result.cases)

    by_id = {c.case_id: c for c in result.cases}
    # Missing fields
    assert by_id["error_handling:transcriptions:missing_file"].detail["status_code"] == 400
    assert by_id["error_handling:diarization:missing_file"].detail["status_code"] == 400
    assert by_id["error_handling:tts:missing_input"].detail["status_code"] == 400

    # Bad audio (corrupt, 8kHz, stereo, 8-bit)
    assert by_id["error_handling:transcriptions:bad_audio_corrupt_non_wav"].detail["status_code"] == 400
    assert by_id["error_handling:transcriptions:bad_audio_wrong_sample_rate_8khz"].detail["status_code"] == 400
    assert by_id["error_handling:transcriptions:bad_audio_stereo_2ch_wav"].detail["status_code"] == 400
    assert by_id["error_handling:transcriptions:bad_audio_wrong_bit_depth_8bit"].detail["status_code"] == 400
    assert by_id["error_handling:diarization:bad_audio_corrupt_non_wav"].detail["status_code"] == 400

    # Unknown model
    assert by_id["error_handling:transcriptions:unknown_model"].detail["status_code"] == 400
    assert by_id["error_handling:diarization:unknown_model"].detail["status_code"] == 400
    assert by_id["error_handling:tts:unknown_model"].detail["status_code"] == 400

    # Invalid parameters & 404
    assert by_id["error_handling:transcriptions:unsupported_response_format"].detail["status_code"] == 400
    assert by_id["error_handling:diarization:invalid_words_json"].detail["status_code"] == 400
    assert by_id["error_handling:diarization:invalid_num_speakers"].detail["status_code"] == 400
    assert by_id["error_handling:admin:unknown_route_404"].detail["status_code"] == 404
    assert "error_handling_latency_mean_ms" in result.metrics


def test_live_stt_concurrency_suite_503_over_budget_and_recovery(tmp_path: Path):
    """Run the 'concurrency' suite and verify 503 when over max_concurrent_calls budget and slot release."""
    tester = LiveSTTTester()
    transport, _ = _create_mock_live_stt_transport(
        max_concurrent_calls=2,
        transcribe_hold_sec=0.06,
    )
    ctx = TestContext(
        run_id=603,
        artifacts_dir=tmp_path / "artifacts_603",
        emit_stdout=False,
    )

    result = tester.run(
        {
            "_transport": transport,
            "suites": ["concurrency"],
            "discover_models": False,
            "concurrency_burst_size": 5,
        },
        {"base_url": "http://localhost:8000"},
        ctx,
    )

    assert result.status == "pass"
    assert len(result.cases) == 3
    assert [c.case_id for c in result.cases] == [
        "concurrency:pre_check_budget",
        "concurrency:over_budget_burst",
        "concurrency:post_burst_recovery_and_stats",
    ]
    assert all(c.status == "pass" for c in result.cases)

    burst_case = result.cases[1]
    assert burst_case.detail["max_concurrent_calls"] == 2
    assert burst_case.detail["burst_size"] == 5
    assert burst_case.detail["ok_200_count"] == 2
    assert burst_case.detail["rejected_503_count"] == 3
    assert burst_case.detail["other_status_count"] == 0

    post_case = result.cases[2]
    assert post_case.detail["recovery_status_code"] == 200
    assert post_case.detail["active_calls_after"] == 0
    assert post_case.detail["rejected_delta"] == 3
    assert "concurrency_latency_mean_ms" in result.metrics


def test_live_stt_contract_suite_alias_and_full_7_suites(tmp_path: Path):
    """Run all 7 suites (functional + admin/contract + error_handling + concurrency) together."""
    tester = LiveSTTTester()
    transport, _ = _create_mock_live_stt_transport(
        max_concurrent_calls=2,
        transcribe_hold_sec=0.05,
    )
    ctx = TestContext(
        run_id=604,
        artifacts_dir=tmp_path / "artifacts_604",
        emit_stdout=False,
    )

    result = tester.run(
        {
            "_transport": transport,
            "model_parametrization": "per_engine",
            "concurrency_burst_size": 4,
        },
        {"base_url": "http://localhost:8000"},
        ctx,
    )

    assert result.status == "pass"
    # 15 functional audio cases + 11 admin cases + 15 error_handling cases + 3 concurrency cases = 44 cases
    assert len(result.cases) == 44
    assert all(c.status == "pass" for c in result.cases)
    assert result.metrics["cases_total"]["value"] == 44
    assert result.metrics["cases_passed"]["value"] == 44
    assert result.metrics["pass_rate"]["value"] == 100.0
    assert "admin_latency_mean_ms" in result.metrics
    assert "error_handling_latency_mean_ms" in result.metrics
    assert "concurrency_latency_mean_ms" in result.metrics

    report_text = Path(result.artifacts[0].path).read_text(encoding="utf-8")
    assert "Admin / Contract" in report_text
    assert "Error Handling" in report_text
    assert "Concurrency Limits" in report_text

