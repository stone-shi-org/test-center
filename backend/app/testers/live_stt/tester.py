"""live-stt HTTP endpoint tester plugin: transcriptions, diarization, TTS, admin, error handling, and concurrency."""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Any

from backend.app.testers.base import (
    TesterPlugin,
    TestContext,
    TesterResult,
    TesterConfigError,
    TesterError,
)
from backend.app.testers.live_stt.client import (
    DEFAULT_ASR_MODEL,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_TTS_MODEL,
    DiscoveredModels,
    LiveSTTClientError,
    LiveSTTHTTPClient,
    ModelInfo,
    select_parametrized_models,
)
from backend.app.testers.live_stt.fixtures import (
    BUILTIN_AUDIO_FIXTURES,
    BUILTIN_BAD_AUDIO_FIXTURES,
    BUILTIN_TTS_FIXTURES,
    get_audio_fixture,
    get_bad_audio_fixture,
)
from backend.app.testers.live_stt.metrics import (
    compute_latency_and_rtf_summary,
    compute_percentiles,
    compute_rtf,
)
from backend.app.testers.live_stt.report import format_live_stt_report
from backend.app.testers.live_stt.schemas import (
    SchemaCheckResult,
    extract_error_message,
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


_SUITE_EXPANSIONS: dict[str, list[str]] = {
    "transcriptions_json": ["transcriptions_json"],
    "transcription_json": ["transcriptions_json"],
    "json": ["transcriptions_json"],
    "transcriptions_sse": ["transcriptions_sse"],
    "transcription_sse": ["transcriptions_sse"],
    "sse": ["transcriptions_sse"],
    "stream": ["transcriptions_sse"],
    "transcriptions": ["transcriptions_json", "transcriptions_sse"],
    "transcription": ["transcriptions_json", "transcriptions_sse"],
    "asr": ["transcriptions_json", "transcriptions_sse"],
    "diarization": ["diarization"],
    "diarize": ["diarization"],
    "tts": ["tts"],
    "speech": ["tts"],
    "admin": ["admin"],
    "models": ["admin"],
    "stats": ["admin"],
    "health": ["admin"],
    "error_handling": ["error_handling"],
    "errors": ["error_handling"],
    "error": ["error_handling"],
    "concurrency": ["concurrency"],
    "concurrency_limits": ["concurrency"],
    "admission": ["concurrency"],
    "contract": ["admin", "error_handling", "concurrency"],
    "admin_contract": ["admin", "error_handling", "concurrency"],
}


def normalize_live_stt_suites(raw_suites: list[str] | None, available: list[str]) -> list[str]:
    """Normalize suite names and expand shorthand aliases (e.g. 'transcriptions', 'contract')."""
    if not raw_suites:
        return list(available)
    expanded: list[str] = []
    for item in raw_suites:
        key = str(item).strip().lower()
        mapped_list = _SUITE_EXPANSIONS.get(key, [str(item).strip()])
        for s in mapped_list:
            if s not in expanded:
                expanded.append(s)
    valid = [s for s in available if s in expanded]
    return valid if valid else expanded


class LiveSTTTester(TesterPlugin):
    """Tester for live-stt HTTP endpoints and admin/contract verification:

    - POST /v1/audio/transcriptions (JSON and verbose_json)
    - POST /v1/audio/transcriptions (stream=true Server-Sent Events)
    - POST /v1/audio/diarization (speaker segments and word-level text alignment)
    - POST /v1/audio/speech (batch WAV/MP3/PCM and chunked streaming TTS)
    - Read-only checks of GET /v1/models, /api/health, /api/stats, /api/version,
      /api/config, /api/settings, /api/backends, /api/*-engines, /metrics, /dashboard
      (no mutating calls by default)
    - Error-path contract tests (bad audio, unknown model, missing fields, status codes & error body shapes)
    - Concurrency/admission tests (503 when over budget, capacity tracking, slot recovery)
    """

    name = "live-stt"
    description = (
        "Functional, latency, and contract tester for live-stt HTTP endpoints: "
        "POST /v1/audio/transcriptions (JSON + stream=true SSE), "
        "POST /v1/audio/diarization, POST /v1/audio/speech (TTS), "
        "read-only admin/telemetry endpoints (/v1/models, /api/stats, /api/health, /metrics), "
        "error handling, and concurrency/admission limits."
    )
    suites = [
        "transcriptions_json",
        "transcriptions_sse",
        "diarization",
        "tts",
        "admin",
        "error_handling",
        "concurrency",
    ]
    suite_aliases = set(_SUITE_EXPANSIONS.keys())

    config_schema = {
        "type": "object",
        "properties": {
            "suites": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "transcriptions_json",
                        "transcriptions_sse",
                        "diarization",
                        "tts",
                        "admin",
                        "error_handling",
                        "concurrency",
                    ],
                },
                "default": [
                    "transcriptions_json",
                    "transcriptions_sse",
                    "diarization",
                    "tts",
                    "admin",
                    "error_handling",
                    "concurrency",
                ],
                "description": "HTTP endpoint and contract test suites to execute",
            },
            "model_parametrization": {
                "type": "string",
                "enum": ["per_engine", "all", "default"],
                "default": "per_engine",
                "description": (
                    "How to parametrize models discovered from /v1/models: "
                    "'per_engine' (one model per distinct engine), 'all' (every model), or 'default' (default model only)"
                ),
            },
            "discover_models": {
                "type": "boolean",
                "default": True,
                "description": "Query GET /v1/models to discover and parametrize engines and models",
            },
            "include_extended_admin": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Include extended read-only admin checks (/api/version, /api/config, /api/settings, "
                    "/api/backends, /api/diarization-engines, /api/tts-engines, /metrics, /dashboard) "
                    "in addition to /v1/models, /api/health, and /api/stats"
                ),
            },
            "allow_mutating_admin": {
                "type": "boolean",
                "default": False,
                "description": "Mutating admin calls are strictly disabled by default",
            },
            "concurrency_burst_size": {
                "type": "integer",
                "minimum": 1,
                "maximum": 64,
                "description": (
                    "Optional explicit number of concurrent transcription requests for the concurrency suite; "
                    "defaults to max_concurrent_calls + 2 discovered from /api/stats or /api/health"
                ),
            },
            "engines": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional filter of engine names or model IDs to test (e.g. ['parakeet', 'whisper', 'pyannote', 'kokoro'])",
            },
            "transcription_model": {
                "type": "string",
                "description": "Optional single ASR model override for /v1/audio/transcriptions",
            },
            "transcription_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional explicit list of ASR models to parametrize over",
            },
            "diarization_model": {
                "type": "string",
                "description": "Optional single model override for /v1/audio/diarization",
            },
            "diarization_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional explicit list of diarization models to parametrize over",
            },
            "tts_model": {
                "type": "string",
                "description": "Optional single model override for /v1/audio/speech",
            },
            "tts_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional explicit list of TTS models to parametrize over",
            },
            "tts_voice": {
                "type": "string",
                "description": "Optional explicit voice override for TTS synthesis",
            },
            "tts_response_format": {
                "type": "string",
                "enum": ["wav", "mp3", "pcm", "opus", "aac", "flac"],
                "default": "wav",
                "description": "Audio format for non-streaming TTS tests",
            },
            "tts_stream_format": {
                "type": "string",
                "enum": ["pcm", "wav", "mp3"],
                "default": "pcm",
                "description": "Audio format for streaming TTS tests",
            },
            "language": {
                "type": "string",
                "default": "en",
                "description": "Language code passed to transcription endpoints",
            },
            "repetitions": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 20,
                "description": "Number of repetitions per test case for latency and RTF statistics",
            },
            "timeout_seconds": {
                "type": "number",
                "default": 120.0,
                "minimum": 1.0,
                "description": "HTTP request timeout in seconds",
            },
        },
    }

    def validate_and_apply_config(self, raw_config: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize optional None fields and suite aliases before JSON Schema validation."""
        cleaned: dict[str, Any] = {}
        if isinstance(raw_config, dict):
            for k, v in raw_config.items():
                if v is None:
                    continue
                if k == "suites" and isinstance(v, list):
                    cleaned[k] = normalize_live_stt_suites(v, self.suites)
                else:
                    cleaned[k] = v
        return super().validate_and_apply_config(cleaned)

    @staticmethod
    def _resolve_api_key(target: dict[str, Any], config: dict[str, Any]) -> str | None:
        auth_env_var = target.get("auth_env_var")
        api_key: str | None = None
        if auth_env_var and isinstance(auth_env_var, str):
            api_key = os.environ.get(auth_env_var)
        if not api_key:
            api_key = config.get("api_key") or os.environ.get("LIVE_STT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return api_key

    def _resolve_client(self, config: dict[str, Any], target: dict[str, Any]) -> LiveSTTHTTPClient:
        base_url = target.get("base_url") or config.get("base_url")
        if not base_url or not str(base_url).strip():
            raise TesterConfigError("Target definition is missing required 'base_url'")

        api_key = self._resolve_api_key(target, config)
        timeout = float(config.get("timeout_seconds", 120.0))
        transport = config.get("_transport")

        return LiveSTTHTTPClient(
            base_url=str(base_url),
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )

    def _resolve_parametrized_models(
        self,
        cfg: dict[str, Any],
        target: dict[str, Any],
        discovered: DiscoveredModels | None,
    ) -> tuple[list[ModelInfo], list[ModelInfo], list[ModelInfo]]:
        """Resolve parametrized ASR, diarization, and TTS models from /v1/models and config overrides."""
        mode = str(cfg.get("model_parametrization", "per_engine"))
        engines_filter = cfg.get("engines")
        extra = target.get("extra") if isinstance(target.get("extra"), dict) else {}

        # Explicit overrides for ASR
        explicit_asr: list[str] = []
        if isinstance(cfg.get("transcription_models"), list) and cfg["transcription_models"]:
            explicit_asr = [str(m) for m in cfg["transcription_models"] if str(m).strip()]
        elif cfg.get("transcription_model"):
            explicit_asr = [str(cfg["transcription_model"])]
        elif extra.get("transcription_model") or extra.get("model"):
            explicit_asr = [str(extra.get("transcription_model") or extra.get("model"))]

        # Explicit overrides for Diarization
        explicit_diar: list[str] = []
        if isinstance(cfg.get("diarization_models"), list) and cfg["diarization_models"]:
            explicit_diar = [str(m) for m in cfg["diarization_models"] if str(m).strip()]
        elif cfg.get("diarization_model"):
            explicit_diar = [str(cfg["diarization_model"])]
        elif extra.get("diarization_model"):
            explicit_diar = [str(extra["diarization_model"])]

        # Explicit overrides for TTS
        explicit_tts: list[str] = []
        if isinstance(cfg.get("tts_models"), list) and cfg["tts_models"]:
            explicit_tts = [str(m) for m in cfg["tts_models"] if str(m).strip()]
        elif cfg.get("tts_model"):
            explicit_tts = [str(cfg["tts_model"])]
        elif extra.get("tts_model"):
            explicit_tts = [str(extra["tts_model"])]

        disc_asr = discovered.asr_models if discovered else []
        disc_diar = discovered.diarization_models if discovered else []
        disc_tts = discovered.tts_models if discovered else []

        asr_models = select_parametrized_models(
            discovered=disc_asr,
            model_type="asr",
            mode=mode,
            explicit_models=explicit_asr or None,
            engines_filter=engines_filter,
            fallback=DEFAULT_ASR_MODEL,
        )
        diar_models = select_parametrized_models(
            discovered=disc_diar,
            model_type="diarization",
            mode=mode,
            explicit_models=explicit_diar or None,
            engines_filter=engines_filter,
            fallback=DEFAULT_DIARIZATION_MODEL,
        )
        tts_models = select_parametrized_models(
            discovered=disc_tts,
            model_type="tts",
            mode=mode,
            explicit_models=explicit_tts or None,
            engines_filter=engines_filter,
            fallback=DEFAULT_TTS_MODEL,
        )

        return asr_models, diar_models, tts_models

    def run(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
        ctx: TestContext,
    ) -> TesterResult:
        cfg = self.validate_and_apply_config(config)
        client = self._resolve_client(cfg, target)

        raw_suites = cfg.get("suites") or target.get("suites") or self.suites
        suites_to_run = normalize_live_stt_suites(raw_suites, self.suites)
        suites_to_run = [s for s in self.suites if s in suites_to_run]
        if not suites_to_run:
            suites_to_run = list(self.suites)

        repetitions = int(cfg.get("repetitions", 1))
        language = str(cfg.get("language", "en") or "en")
        tts_format = str(cfg.get("tts_response_format", "wav") or "wav")
        tts_stream_format = str(cfg.get("tts_stream_format", "pcm") or "pcm")
        tts_voice_override = cfg.get("tts_voice")
        discover_enabled = bool(cfg.get("discover_models", True))
        param_mode = str(cfg.get("model_parametrization", "per_engine"))

        ctx.check_cancelled()
        ctx.log(
            f"Starting live-stt HTTP tester for target='{target.get('name', 'default')}', "
            f"base_url='{client.root_url}', suites={suites_to_run}, "
            f"parametrization='{param_mode}', repetitions={repetitions}"
        )

        discovered: DiscoveredModels | None = None
        models_schema_valid: bool | None = None
        if discover_enabled:
            ctx.log(f"Querying GET {client.models_url} for model discovery and engine parametrization...")
            try:
                discovered = client.discover_models()
            except LiveSTTClientError as e:
                ctx.log(f"Error calling {client.models_url}: {e}", level="error")
                raise TesterError(f"live-stt API error: {e}") from e

            if discovered.status_code == 200 and discovered.raw_body is not None:
                schema_check = validate_models_response(discovered.raw_body)
                models_schema_valid = schema_check.valid
                ctx.log(
                    f"Discovered {discovered.total_count} models from /v1/models "
                    f"(ASR={len(discovered.asr_models)}, Diarization={len(discovered.diarization_models)}, "
                    f"TTS={len(discovered.tts_models)}) in {discovered.latency_ms:.2f}ms | "
                    f"{schema_check.summary_message}"
                )
            else:
                ctx.log(
                    f"GET /v1/models returned non-200 or invalid payload ({discovered.error}); "
                    "falling back to built-in default models",
                    level="warning",
                )

        asr_models, diar_models, tts_models = self._resolve_parametrized_models(
            cfg=cfg,
            target=target,
            discovered=discovered,
        )

        include_extended_admin = bool(cfg.get("include_extended_admin", True))

        # Calculate total planned cases for progress reporting
        planned_cases = 0
        if "transcriptions_json" in suites_to_run:
            planned_cases += len(asr_models) * 2
        if "transcriptions_sse" in suites_to_run:
            planned_cases += len(asr_models)
        if "diarization" in suites_to_run:
            planned_cases += len(diar_models) * 2
        if "tts" in suites_to_run:
            for tm in tts_models:
                planned_cases += 2 if tm.supports_streaming else 1
        if "admin" in suites_to_run:
            planned_cases += 11 if include_extended_admin else 3
        if "error_handling" in suites_to_run:
            planned_cases += 15
        if "concurrency" in suites_to_run:
            planned_cases += 3
        planned_cases = max(1, planned_cases)
        completed_cases = 0

        result = TesterResult(status="pass")

        all_latencies_ms: list[float] = []
        all_rtfs: list[float] = []

        trans_json_latencies: list[float] = []
        trans_json_rtfs: list[float] = []

        trans_sse_latencies: list[float] = []
        trans_sse_ttfts: list[float] = []
        trans_sse_rtfs: list[float] = []

        diar_latencies: list[float] = []
        diar_rtfs: list[float] = []

        tts_latencies: list[float] = []
        tts_ttfbs: list[float] = []
        tts_rtfs: list[float] = []

        admin_latencies: list[float] = []
        error_latencies: list[float] = []
        concurrency_latencies: list[float] = []

        tested_engines: set[str] = set()
        tested_models: set[str] = set()

        t_run_start = time.perf_counter()

        # -------------------------------------------------------------------
        # Suite 1: POST /v1/audio/transcriptions (JSON and verbose_json)
        # -------------------------------------------------------------------
        if "transcriptions_json" in suites_to_run:
            short_fix = get_audio_fixture("short_speech")
            medium_fix = get_audio_fixture("medium_speech")

            for m_info in asr_models:
                tested_engines.add(m_info.engine)
                tested_models.add(m_info.id)

                # Case 1a: response_format="json"
                ctx.check_cancelled()
                case_id_json = f"transcriptions_json:{m_info.id}:json"
                ctx.log(
                    f"Running [{completed_cases + 1}/{planned_cases}] {case_id_json} "
                    f"(engine={m_info.engine}, fixture={short_fix.id}, {short_fix.duration_sec}s)..."
                )

                rep_lats: list[float] = []
                rep_rtfs: list[float] = []
                last_res = None
                for _ in range(repetitions):
                    ctx.check_cancelled()
                    try:
                        res = client.transcribe_json(
                            wav_bytes=short_fix.wav_bytes,
                            model=m_info.id,
                            response_format="json",
                            language=language,
                            filename=f"{short_fix.id}.wav",
                        )
                    except LiveSTTClientError as e:
                        ctx.log(f"Error calling {client.transcriptions_url}: {e}", level="error")
                        raise TesterError(f"live-stt API error: {e}") from e

                    last_res = res
                    rtf_val = compute_rtf(res.latency_ms, short_fix.duration_sec)
                    rep_lats.append(res.latency_ms)
                    rep_rtfs.append(rtf_val)
                    all_latencies_ms.append(res.latency_ms)
                    all_rtfs.append(rtf_val)
                    trans_json_latencies.append(res.latency_ms)
                    trans_json_rtfs.append(rtf_val)

                assert last_res is not None
                schema_res = validate_transcription_json_response(
                    body=last_res.body,
                    content_type=last_res.content_type,
                    status_code=last_res.status_code,
                )
                case_status = "pass" if schema_res.valid else "fail"
                mean_lat = round(float(statistics.mean(rep_lats)), 2)
                mean_rtf = round(float(statistics.mean(rep_rtfs)), 4)
                transcript_text = (
                    last_res.body.get("text", "")
                    if isinstance(last_res.body, dict) and isinstance(last_res.body.get("text"), str)
                    else ""
                )
                word_cnt = len(transcript_text.split()) if transcript_text.strip() else 0

                result.add_case(
                    case_id=case_id_json,
                    suite="transcriptions_json",
                    status=case_status,
                    metrics={
                        "latency_ms": {"value": mean_lat, "unit": "ms", "better": "lower"},
                        "rtf": {"value": mean_rtf, "unit": "x", "better": "lower"},
                        "audio_duration_sec": {"value": short_fix.duration_sec, "unit": "s"},
                        "word_count": {"value": word_cnt, "unit": "words"},
                    },
                    detail={
                        "endpoint": "/v1/audio/transcriptions",
                        "model": m_info.id,
                        "engine": m_info.engine,
                        "response_format": "json",
                        "stream": False,
                        "fixture_id": short_fix.id,
                        "audio_duration_sec": short_fix.duration_sec,
                        "status_code": last_res.status_code,
                        "content_type": last_res.content_type,
                        "text": transcript_text,
                        "response_body": last_res.body,
                        "passed": schema_res.valid,
                        "eval_message": schema_res.summary_message,
                        "checks": schema_res.checks,
                        "errors": schema_res.errors,
                    },
                )
                completed_cases += 1
                ctx.log(
                    f"  {case_id_json} -> {case_status.upper()} | "
                    f"Latency={mean_lat:.2f}ms, RTF={mean_rtf:.4f}x | {schema_res.summary_message}"
                )
                ctx.progress(completed_cases, planned_cases, f"Completed {case_id_json}")

                # Case 1b: response_format="verbose_json"
                ctx.check_cancelled()
                case_id_vjson = f"transcriptions_json:{m_info.id}:verbose_json"
                ctx.log(
                    f"Running [{completed_cases + 1}/{planned_cases}] {case_id_vjson} "
                    f"(engine={m_info.engine}, fixture={medium_fix.id}, {medium_fix.duration_sec}s)..."
                )

                v_lats: list[float] = []
                v_rtfs: list[float] = []
                last_vres = None
                for _ in range(repetitions):
                    ctx.check_cancelled()
                    try:
                        vres = client.transcribe_json(
                            wav_bytes=medium_fix.wav_bytes,
                            model=m_info.id,
                            response_format="verbose_json",
                            language=language,
                            filename=f"{medium_fix.id}.wav",
                        )
                    except LiveSTTClientError as e:
                        ctx.log(f"Error calling {client.transcriptions_url}: {e}", level="error")
                        raise TesterError(f"live-stt API error: {e}") from e

                    last_vres = vres
                    rtf_val = compute_rtf(vres.latency_ms, medium_fix.duration_sec)
                    v_lats.append(vres.latency_ms)
                    v_rtfs.append(rtf_val)
                    all_latencies_ms.append(vres.latency_ms)
                    all_rtfs.append(rtf_val)
                    trans_json_latencies.append(vres.latency_ms)
                    trans_json_rtfs.append(rtf_val)

                assert last_vres is not None
                v_schema = validate_transcription_verbose_json_response(
                    body=last_vres.body,
                    content_type=last_vres.content_type,
                    status_code=last_vres.status_code,
                )
                v_status = "pass" if v_schema.valid else "fail"
                v_mean_lat = round(float(statistics.mean(v_lats)), 2)
                v_mean_rtf = round(float(statistics.mean(v_rtfs)), 4)
                v_body = last_vres.body if isinstance(last_vres.body, dict) else {}
                v_text = v_body.get("text", "") if isinstance(v_body.get("text"), str) else ""
                v_words = v_body.get("words", []) if isinstance(v_body.get("words"), list) else []
                v_reported_dur = float(v_body.get("duration", 0.0) or 0.0) if isinstance(v_body.get("duration"), (int, float)) else 0.0

                result.add_case(
                    case_id=case_id_vjson,
                    suite="transcriptions_json",
                    status=v_status,
                    metrics={
                        "latency_ms": {"value": v_mean_lat, "unit": "ms", "better": "lower"},
                        "rtf": {"value": v_mean_rtf, "unit": "x", "better": "lower"},
                        "audio_duration_sec": {"value": medium_fix.duration_sec, "unit": "s"},
                        "reported_duration_sec": {"value": round(v_reported_dur, 3), "unit": "s"},
                        "word_count": {"value": len(v_words), "unit": "words"},
                    },
                    detail={
                        "endpoint": "/v1/audio/transcriptions",
                        "model": m_info.id,
                        "engine": m_info.engine,
                        "response_format": "verbose_json",
                        "stream": False,
                        "fixture_id": medium_fix.id,
                        "audio_duration_sec": medium_fix.duration_sec,
                        "status_code": last_vres.status_code,
                        "content_type": last_vres.content_type,
                        "text": v_text,
                        "words_count": len(v_words),
                        "response_body": last_vres.body,
                        "passed": v_schema.valid,
                        "eval_message": v_schema.summary_message,
                        "checks": v_schema.checks,
                        "errors": v_schema.errors,
                    },
                )
                completed_cases += 1
                ctx.log(
                    f"  {case_id_vjson} -> {v_status.upper()} | "
                    f"Latency={v_mean_lat:.2f}ms, RTF={v_mean_rtf:.4f}x | {v_schema.summary_message}"
                )
                ctx.progress(completed_cases, planned_cases, f"Completed {case_id_vjson}")

        # -------------------------------------------------------------------
        # Suite 2: POST /v1/audio/transcriptions (stream=true SSE)
        # -------------------------------------------------------------------
        if "transcriptions_sse" in suites_to_run:
            sse_fix = get_audio_fixture("short_speech")

            for m_info in asr_models:
                tested_engines.add(m_info.engine)
                tested_models.add(m_info.id)

                ctx.check_cancelled()
                case_id_sse = f"transcriptions_sse:{m_info.id}:stream"
                ctx.log(
                    f"Running [{completed_cases + 1}/{planned_cases}] {case_id_sse} "
                    f"(engine={m_info.engine}, fixture={sse_fix.id}, {sse_fix.duration_sec}s)..."
                )

                s_lats: list[float] = []
                s_ttfts: list[float] = []
                s_rtfs: list[float] = []
                last_sres = None
                for _ in range(repetitions):
                    ctx.check_cancelled()
                    try:
                        sres = client.transcribe_sse(
                            wav_bytes=sse_fix.wav_bytes,
                            model=m_info.id,
                            language=language,
                            filename=f"{sse_fix.id}.wav",
                        )
                    except LiveSTTClientError as e:
                        ctx.log(f"Error calling {client.transcriptions_url} (SSE): {e}", level="error")
                        raise TesterError(f"live-stt API error: {e}") from e

                    last_sres = sres
                    rtf_val = compute_rtf(sres.latency_ms, sse_fix.duration_sec)
                    s_lats.append(sres.latency_ms)
                    s_ttfts.append(sres.ttft_ms)
                    s_rtfs.append(rtf_val)

                    all_latencies_ms.append(sres.latency_ms)
                    all_rtfs.append(rtf_val)
                    trans_sse_latencies.append(sres.latency_ms)
                    trans_sse_ttfts.append(sres.ttft_ms)
                    trans_sse_rtfs.append(rtf_val)

                assert last_sres is not None
                sse_schema = validate_transcription_sse_response(
                    events=last_sres.events,
                    saw_done=last_sres.saw_done,
                    content_type=last_sres.content_type,
                    status_code=last_sres.status_code,
                )
                sse_status = "pass" if sse_schema.valid else "fail"
                s_mean_lat = round(float(statistics.mean(s_lats)), 2)
                s_mean_ttft = round(float(statistics.mean(s_ttfts)), 2)
                s_mean_rtf = round(float(statistics.mean(s_rtfs)), 4)

                result.add_case(
                    case_id=case_id_sse,
                    suite="transcriptions_sse",
                    status=sse_status,
                    metrics={
                        "ttft_ms": {"value": s_mean_ttft, "unit": "ms", "better": "lower"},
                        "latency_ms": {"value": s_mean_lat, "unit": "ms", "better": "lower"},
                        "rtf": {"value": s_mean_rtf, "unit": "x", "better": "lower"},
                        "audio_duration_sec": {"value": sse_fix.duration_sec, "unit": "s"},
                        "sse_events": {"value": len(last_sres.events), "unit": "events"},
                    },
                    detail={
                        "endpoint": "/v1/audio/transcriptions",
                        "model": m_info.id,
                        "engine": m_info.engine,
                        "stream": True,
                        "fixture_id": sse_fix.id,
                        "audio_duration_sec": sse_fix.duration_sec,
                        "status_code": last_sres.status_code,
                        "content_type": last_sres.content_type,
                        "text": last_sres.text,
                        "events": last_sres.events,
                        "saw_done": last_sres.saw_done,
                        "passed": sse_schema.valid,
                        "eval_message": sse_schema.summary_message,
                        "checks": sse_schema.checks,
                        "errors": sse_schema.errors,
                    },
                )
                completed_cases += 1
                ctx.log(
                    f"  {case_id_sse} -> {sse_status.upper()} | "
                    f"TTFT={s_mean_ttft:.2f}ms, Latency={s_mean_lat:.2f}ms, RTF={s_mean_rtf:.4f}x | "
                    f"{sse_schema.summary_message}"
                )
                ctx.progress(completed_cases, planned_cases, f"Completed {case_id_sse}")

        # -------------------------------------------------------------------
        # Suite 3: POST /v1/audio/diarization
        # -------------------------------------------------------------------
        if "diarization" in suites_to_run:
            dialog_fix = get_audio_fixture("two_speaker_dialog")
            words_payload = dialog_fix.words_payload()

            for m_info in diar_models:
                tested_engines.add(m_info.engine)
                tested_models.add(m_info.id)

                # Case 3a: Speaker segmentation (segments & speakers schema)
                ctx.check_cancelled()
                case_id_seg = f"diarization:{m_info.id}:segments"
                ctx.log(
                    f"Running [{completed_cases + 1}/{planned_cases}] {case_id_seg} "
                    f"(engine={m_info.engine}, fixture={dialog_fix.id}, {dialog_fix.duration_sec}s)..."
                )

                d_lats: list[float] = []
                d_rtfs: list[float] = []
                last_dres = None
                min_spk = 1 if m_info.supports_num_speakers_hint else None
                max_spk = 2 if m_info.supports_num_speakers_hint else None

                for _ in range(repetitions):
                    ctx.check_cancelled()
                    try:
                        dres = client.diarize(
                            wav_bytes=dialog_fix.wav_bytes,
                            model=m_info.id,
                            words=None,
                            include_text=False,
                            min_speakers=min_spk,
                            max_speakers=max_spk,
                            response_format="verbose_json",
                            filename=f"{dialog_fix.id}.wav",
                        )
                    except LiveSTTClientError as e:
                        ctx.log(f"Error calling {client.diarization_url}: {e}", level="error")
                        raise TesterError(f"live-stt API error: {e}") from e

                    last_dres = dres
                    rtf_val = compute_rtf(dres.latency_ms, dialog_fix.duration_sec)
                    d_lats.append(dres.latency_ms)
                    d_rtfs.append(rtf_val)
                    all_latencies_ms.append(dres.latency_ms)
                    all_rtfs.append(rtf_val)
                    diar_latencies.append(dres.latency_ms)
                    diar_rtfs.append(rtf_val)

                assert last_dres is not None
                d_schema = validate_diarization_response(
                    body=last_dres.body,
                    content_type=last_dres.content_type,
                    status_code=last_dres.status_code,
                    expect_text=False,
                )
                d_status = "pass" if d_schema.valid else "fail"
                d_mean_lat = round(float(statistics.mean(d_lats)), 2)
                d_mean_rtf = round(float(statistics.mean(d_rtfs)), 4)
                d_body = last_dres.body if isinstance(last_dres.body, dict) else {}
                num_spk = int(d_body.get("num_speakers", 0) or 0) if isinstance(d_body.get("num_speakers"), int) else 0
                segs = d_body.get("segments", []) if isinstance(d_body.get("segments"), list) else []

                result.add_case(
                    case_id=case_id_seg,
                    suite="diarization",
                    status=d_status,
                    metrics={
                        "latency_ms": {"value": d_mean_lat, "unit": "ms", "better": "lower"},
                        "rtf": {"value": d_mean_rtf, "unit": "x", "better": "lower"},
                        "audio_duration_sec": {"value": dialog_fix.duration_sec, "unit": "s"},
                        "num_speakers": {"value": num_spk, "unit": "speakers"},
                        "segment_count": {"value": len(segs), "unit": "segments"},
                    },
                    detail={
                        "endpoint": "/v1/audio/diarization",
                        "model": m_info.id,
                        "engine": m_info.engine,
                        "mode": "segments",
                        "fixture_id": dialog_fix.id,
                        "audio_duration_sec": dialog_fix.duration_sec,
                        "status_code": last_dres.status_code,
                        "content_type": last_dres.content_type,
                        "num_speakers": num_spk,
                        "segment_count": len(segs),
                        "response_body": last_dres.body,
                        "passed": d_schema.valid,
                        "eval_message": d_schema.summary_message,
                        "checks": d_schema.checks,
                        "errors": d_schema.errors,
                    },
                )
                completed_cases += 1
                ctx.log(
                    f"  {case_id_seg} -> {d_status.upper()} | "
                    f"Latency={d_mean_lat:.2f}ms, RTF={d_mean_rtf:.4f}x, Speakers={num_spk} | "
                    f"{d_schema.summary_message}"
                )
                ctx.progress(completed_cases, planned_cases, f"Completed {case_id_seg}")

                # Case 3b: Diarization with ASR word timestamps alignment (words + include_text=true)
                ctx.check_cancelled()
                case_id_words = f"diarization:{m_info.id}:words_alignment"
                ctx.log(
                    f"Running [{completed_cases + 1}/{planned_cases}] {case_id_words} "
                    f"(engine={m_info.engine}, words={len(words_payload)})..."
                )

                dw_lats: list[float] = []
                dw_rtfs: list[float] = []
                last_dwres = None
                for _ in range(repetitions):
                    ctx.check_cancelled()
                    try:
                        dwres = client.diarize(
                            wav_bytes=dialog_fix.wav_bytes,
                            model=m_info.id,
                            words=words_payload,
                            include_text=True,
                            min_speakers=min_spk,
                            max_speakers=max_spk,
                            response_format="verbose_json",
                            filename=f"{dialog_fix.id}.wav",
                        )
                    except LiveSTTClientError as e:
                        ctx.log(f"Error calling {client.diarization_url}: {e}", level="error")
                        raise TesterError(f"live-stt API error: {e}") from e

                    last_dwres = dwres
                    rtf_val = compute_rtf(dwres.latency_ms, dialog_fix.duration_sec)
                    dw_lats.append(dwres.latency_ms)
                    dw_rtfs.append(rtf_val)
                    all_latencies_ms.append(dwres.latency_ms)
                    all_rtfs.append(rtf_val)
                    diar_latencies.append(dwres.latency_ms)
                    diar_rtfs.append(rtf_val)

                assert last_dwres is not None
                dw_schema = validate_diarization_response(
                    body=last_dwres.body,
                    content_type=last_dwres.content_type,
                    status_code=last_dwres.status_code,
                    expect_text=True,
                )
                dw_status = "pass" if dw_schema.valid else "fail"
                dw_mean_lat = round(float(statistics.mean(dw_lats)), 2)
                dw_mean_rtf = round(float(statistics.mean(dw_rtfs)), 4)
                dw_body = last_dwres.body if isinstance(last_dwres.body, dict) else {}
                dw_num_spk = int(dw_body.get("num_speakers", 0) or 0) if isinstance(dw_body.get("num_speakers"), int) else 0
                dw_segs = dw_body.get("segments", []) if isinstance(dw_body.get("segments"), list) else []

                result.add_case(
                    case_id=case_id_words,
                    suite="diarization",
                    status=dw_status,
                    metrics={
                        "latency_ms": {"value": dw_mean_lat, "unit": "ms", "better": "lower"},
                        "rtf": {"value": dw_mean_rtf, "unit": "x", "better": "lower"},
                        "audio_duration_sec": {"value": dialog_fix.duration_sec, "unit": "s"},
                        "num_speakers": {"value": dw_num_spk, "unit": "speakers"},
                        "segment_count": {"value": len(dw_segs), "unit": "segments"},
                    },
                    detail={
                        "endpoint": "/v1/audio/diarization",
                        "model": m_info.id,
                        "engine": m_info.engine,
                        "mode": "words_alignment",
                        "fixture_id": dialog_fix.id,
                        "audio_duration_sec": dialog_fix.duration_sec,
                        "words_supplied": len(words_payload),
                        "status_code": last_dwres.status_code,
                        "content_type": last_dwres.content_type,
                        "num_speakers": dw_num_spk,
                        "segment_count": len(dw_segs),
                        "response_body": last_dwres.body,
                        "passed": dw_schema.valid,
                        "eval_message": dw_schema.summary_message,
                        "checks": dw_schema.checks,
                        "errors": dw_schema.errors,
                    },
                )
                completed_cases += 1
                ctx.log(
                    f"  {case_id_words} -> {dw_status.upper()} | "
                    f"Latency={dw_mean_lat:.2f}ms, RTF={dw_mean_rtf:.4f}x | {dw_schema.summary_message}"
                )
                ctx.progress(completed_cases, planned_cases, f"Completed {case_id_words}")

        # -------------------------------------------------------------------
        # Suite 4: POST /v1/audio/speech (TTS batch + streaming)
        # -------------------------------------------------------------------
        if "tts" in suites_to_run:
            tts_batch_fix = BUILTIN_TTS_FIXTURES["tts_batch"]
            tts_stream_fix = BUILTIN_TTS_FIXTURES["tts_stream"]

            for m_info in tts_models:
                tested_engines.add(m_info.engine)
                tested_models.add(m_info.id)

                voice = str(tts_voice_override).strip() if tts_voice_override else (m_info.default_voice or "af_heart")

                # Case 4a: Non-streaming TTS synthesis
                ctx.check_cancelled()
                case_id_tts_batch = f"tts:{m_info.id}:{tts_format}"
                ctx.log(
                    f"Running [{completed_cases + 1}/{planned_cases}] {case_id_tts_batch} "
                    f"(engine={m_info.engine}, voice={voice}, format={tts_format})..."
                )

                tb_lats: list[float] = []
                tb_ttfbs: list[float] = []
                tb_rtfs: list[float] = []
                last_tbres = None
                last_out_dur = 0.0
                last_tb_schema = None

                for _ in range(repetitions):
                    ctx.check_cancelled()
                    try:
                        tbres = client.synthesize_speech(
                            input_text=tts_batch_fix.text,
                            model=m_info.id,
                            voice=voice,
                            response_format=tts_format,
                            speed=1.0,
                            stream=False,
                        )
                    except LiveSTTClientError as e:
                        ctx.log(f"Error calling {client.speech_url}: {e}", level="error")
                        raise TesterError(f"live-stt API error: {e}") from e

                    last_tbres = tbres
                    tb_schema, out_dur = validate_tts_response(
                        audio_bytes=tbres.audio_bytes,
                        content_type=tbres.content_type,
                        status_code=tbres.status_code,
                        response_format=tts_format,
                        expected_sample_rate=m_info.sample_rate,
                        is_streaming=False,
                        chunk_count=tbres.chunk_count,
                    )
                    last_tb_schema = tb_schema
                    last_out_dur = out_dur
                    rtf_val = compute_rtf(tbres.latency_ms, out_dur) if out_dur > 0 else 0.0

                    tb_lats.append(tbres.latency_ms)
                    tb_ttfbs.append(tbres.ttfb_ms)
                    tb_rtfs.append(rtf_val)

                    all_latencies_ms.append(tbres.latency_ms)
                    if out_dur > 0:
                        all_rtfs.append(rtf_val)
                    tts_latencies.append(tbres.latency_ms)
                    tts_ttfbs.append(tbres.ttfb_ms)
                    if out_dur > 0:
                        tts_rtfs.append(rtf_val)

                assert last_tbres is not None and last_tb_schema is not None
                tb_status = "pass" if last_tb_schema.valid else "fail"
                tb_mean_lat = round(float(statistics.mean(tb_lats)), 2)
                tb_mean_ttfb = round(float(statistics.mean(tb_ttfbs)), 2)
                tb_mean_rtf = round(float(statistics.mean(tb_rtfs)), 4)

                result.add_case(
                    case_id=case_id_tts_batch,
                    suite="tts",
                    status=tb_status,
                    metrics={
                        "latency_ms": {"value": tb_mean_lat, "unit": "ms", "better": "lower"},
                        "ttfb_ms": {"value": tb_mean_ttfb, "unit": "ms", "better": "lower"},
                        "rtf": {"value": tb_mean_rtf, "unit": "x", "better": "lower"},
                        "output_audio_duration_sec": {"value": round(last_out_dur, 3), "unit": "s", "better": "higher"},
                        "audio_bytes": {"value": len(last_tbres.audio_bytes), "unit": "bytes", "better": "higher"},
                    },
                    detail={
                        "endpoint": "/v1/audio/speech",
                        "model": m_info.id,
                        "engine": m_info.engine,
                        "voice": voice,
                        "response_format": tts_format,
                        "stream": False,
                        "input_text": tts_batch_fix.text,
                        "status_code": last_tbres.status_code,
                        "content_type": last_tbres.content_type,
                        "audio_bytes": len(last_tbres.audio_bytes),
                        "output_audio_duration_sec": round(last_out_dur, 4),
                        "error_body": last_tbres.error_body,
                        "passed": last_tb_schema.valid,
                        "eval_message": last_tb_schema.summary_message,
                        "checks": last_tb_schema.checks,
                        "errors": last_tb_schema.errors,
                    },
                )
                completed_cases += 1
                ctx.log(
                    f"  {case_id_tts_batch} -> {tb_status.upper()} | "
                    f"Latency={tb_mean_lat:.2f}ms, OutAudio={last_out_dur:.2f}s, RTF={tb_mean_rtf:.4f}x | "
                    f"{last_tb_schema.summary_message}"
                )
                ctx.progress(completed_cases, planned_cases, f"Completed {case_id_tts_batch}")

                # Case 4b: Streaming TTS synthesis (if supported)
                if m_info.supports_streaming:
                    ctx.check_cancelled()
                    case_id_tts_stream = f"tts:{m_info.id}:stream"
                    ctx.log(
                        f"Running [{completed_cases + 1}/{planned_cases}] {case_id_tts_stream} "
                        f"(engine={m_info.engine}, voice={voice}, format={tts_stream_format}, stream=True)..."
                    )

                    ts_lats: list[float] = []
                    ts_ttfbs: list[float] = []
                    ts_rtfs: list[float] = []
                    last_tsres = None
                    last_ts_dur = 0.0
                    last_ts_schema = None

                    for _ in range(repetitions):
                        ctx.check_cancelled()
                        try:
                            tsres = client.synthesize_speech(
                                input_text=tts_stream_fix.text,
                                model=m_info.id,
                                voice=voice,
                                response_format=tts_stream_format,
                                speed=1.0,
                                stream=True,
                            )
                        except LiveSTTClientError as e:
                            ctx.log(f"Error calling {client.speech_url} (stream): {e}", level="error")
                            raise TesterError(f"live-stt API error: {e}") from e

                        last_tsres = tsres
                        ts_schema, ts_dur = validate_tts_response(
                            audio_bytes=tsres.audio_bytes,
                            content_type=tsres.content_type,
                            status_code=tsres.status_code,
                            response_format=tts_stream_format,
                            expected_sample_rate=m_info.sample_rate,
                            is_streaming=True,
                            chunk_count=tsres.chunk_count,
                        )
                        last_ts_schema = ts_schema
                        last_ts_dur = ts_dur
                        rtf_val = compute_rtf(tsres.latency_ms, ts_dur) if ts_dur > 0 else 0.0

                        ts_lats.append(tsres.latency_ms)
                        ts_ttfbs.append(tsres.ttfb_ms)
                        ts_rtfs.append(rtf_val)

                        all_latencies_ms.append(tsres.latency_ms)
                        if ts_dur > 0:
                            all_rtfs.append(rtf_val)
                        tts_latencies.append(tsres.latency_ms)
                        tts_ttfbs.append(tsres.ttfb_ms)
                        if ts_dur > 0:
                            tts_rtfs.append(rtf_val)

                    assert last_tsres is not None and last_ts_schema is not None
                    ts_status = "pass" if last_ts_schema.valid else "fail"
                    ts_mean_lat = round(float(statistics.mean(ts_lats)), 2)
                    ts_mean_ttfb = round(float(statistics.mean(ts_ttfbs)), 2)
                    ts_mean_rtf = round(float(statistics.mean(ts_rtfs)), 4)

                    result.add_case(
                        case_id=case_id_tts_stream,
                        suite="tts",
                        status=ts_status,
                        metrics={
                            "ttfb_ms": {"value": ts_mean_ttfb, "unit": "ms", "better": "lower"},
                            "latency_ms": {"value": ts_mean_lat, "unit": "ms", "better": "lower"},
                            "rtf": {"value": ts_mean_rtf, "unit": "x", "better": "lower"},
                            "output_audio_duration_sec": {"value": round(last_ts_dur, 3), "unit": "s", "better": "higher"},
                            "audio_bytes": {"value": len(last_tsres.audio_bytes), "unit": "bytes", "better": "higher"},
                            "chunk_count": {"value": last_tsres.chunk_count, "unit": "chunks"},
                        },
                        detail={
                            "endpoint": "/v1/audio/speech",
                            "model": m_info.id,
                            "engine": m_info.engine,
                            "voice": voice,
                            "response_format": tts_stream_format,
                            "stream": True,
                            "input_text": tts_stream_fix.text,
                            "status_code": last_tsres.status_code,
                            "content_type": last_tsres.content_type,
                            "audio_bytes": len(last_tsres.audio_bytes),
                            "chunk_count": last_tsres.chunk_count,
                            "output_audio_duration_sec": round(last_ts_dur, 4),
                            "error_body": last_tsres.error_body,
                            "passed": last_ts_schema.valid,
                            "eval_message": last_ts_schema.summary_message,
                            "checks": last_ts_schema.checks,
                            "errors": last_ts_schema.errors,
                        },
                    )
                    completed_cases += 1
                    ctx.log(
                        f"  {case_id_tts_stream} -> {ts_status.upper()} | "
                        f"TTFB={ts_mean_ttfb:.2f}ms, Latency={ts_mean_lat:.2f}ms, "
                        f"Chunks={last_tsres.chunk_count}, RTF={ts_mean_rtf:.4f}x | "
                        f"{last_ts_schema.summary_message}"
                    )
                    ctx.progress(completed_cases, planned_cases, f"Completed {case_id_tts_stream}")

        # -------------------------------------------------------------------
        # Suite 5: Read-Only Admin & Contract Endpoints (No Mutating Calls)
        # -------------------------------------------------------------------
        if "admin" in suites_to_run:
            completed_cases = self._run_admin_suite(
                client=client,
                ctx=ctx,
                result=result,
                repetitions=repetitions,
                include_extended_admin=include_extended_admin,
                all_latencies_ms=all_latencies_ms,
                admin_latencies=admin_latencies,
                completed_cases=completed_cases,
                planned_cases=planned_cases,
            )

        # -------------------------------------------------------------------
        # Suite 6: Error-Path Contract Tests (Bad Audio, Unknown Model, Missing Fields)
        # -------------------------------------------------------------------
        if "error_handling" in suites_to_run:
            default_asr_id = asr_models[0].id if asr_models else DEFAULT_ASR_MODEL.id
            default_diar_id = diar_models[0].id if diar_models else DEFAULT_DIARIZATION_MODEL.id
            default_tts_id = tts_models[0].id if tts_models else DEFAULT_TTS_MODEL.id
            completed_cases = self._run_error_handling_suite(
                client=client,
                ctx=ctx,
                result=result,
                asr_model_id=default_asr_id,
                diar_model_id=default_diar_id,
                tts_model_id=default_tts_id,
                all_latencies_ms=all_latencies_ms,
                error_latencies=error_latencies,
                completed_cases=completed_cases,
                planned_cases=planned_cases,
            )

        # -------------------------------------------------------------------
        # Suite 7: Concurrency & Admission Budget Behavior (503 Over Budget)
        # -------------------------------------------------------------------
        if "concurrency" in suites_to_run:
            default_asr_id = asr_models[0].id if asr_models else DEFAULT_ASR_MODEL.id
            completed_cases = self._run_concurrency_suite(
                client=client,
                ctx=ctx,
                result=result,
                cfg=cfg,
                asr_model_id=default_asr_id,
                language=language,
                all_latencies_ms=all_latencies_ms,
                concurrency_latencies=concurrency_latencies,
                completed_cases=completed_cases,
                planned_cases=planned_cases,
            )

        total_wall_ms = round((time.perf_counter() - t_run_start) * 1000.0, 2)

        # -------------------------------------------------------------------
        # Aggregate Metrics
        # -------------------------------------------------------------------
        overall_stats = compute_latency_and_rtf_summary(all_latencies_ms, all_rtfs)
        result.set_metric("latency_mean_ms", overall_stats["latency_mean_ms"], unit="ms", better="lower")
        result.set_metric("latency_min_ms", overall_stats["latency_min_ms"], unit="ms", better="lower")
        result.set_metric("latency_max_ms", overall_stats["latency_max_ms"], unit="ms", better="lower")
        result.set_metric("latency_p50_ms", overall_stats["latency_p50_ms"], unit="ms", better="lower")
        result.set_metric("latency_p95_ms", overall_stats["latency_p95_ms"], unit="ms", better="lower")
        result.set_metric("latency_p99_ms", overall_stats["latency_p99_ms"], unit="ms", better="lower")
        result.set_metric("p50_ms", overall_stats["latency_p50_ms"], unit="ms", better="lower")
        result.set_metric("p95_ms", overall_stats["latency_p95_ms"], unit="ms", better="lower")
        result.set_metric("p99_ms", overall_stats["latency_p99_ms"], unit="ms", better="lower")

        result.set_metric("rtf", overall_stats["rtf_mean"], unit="x", better="lower")
        result.set_metric("rtf_mean", overall_stats["rtf_mean"], unit="x", better="lower")
        result.set_metric("rtf_min", overall_stats["rtf_min"], unit="x", better="lower")
        result.set_metric("rtf_max", overall_stats["rtf_max"], unit="x", better="lower")
        result.set_metric("rtf_p50", overall_stats["rtf_p50"], unit="x", better="lower")
        result.set_metric("rtf_p95", overall_stats["rtf_p95"], unit="x", better="lower")
        result.set_metric("rtf_p99", overall_stats["rtf_p99"], unit="x", better="lower")
        result.set_metric("total_wall_ms", total_wall_ms, unit="ms", better="lower")

        # Per-suite aggregate metrics
        if trans_json_latencies:
            tj_pct = compute_percentiles(trans_json_latencies)
            result.set_metric(
                "transcription_json_latency_mean_ms",
                round(float(statistics.mean(trans_json_latencies)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "transcription_json_latency_p95_ms",
                tj_pct[95],
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "transcription_json_rtf_mean",
                round(float(statistics.mean(trans_json_rtfs)), 4),
                unit="x",
                better="lower",
            )

        if trans_sse_latencies:
            ts_pct = compute_percentiles(trans_sse_latencies)
            result.set_metric(
                "transcription_sse_latency_mean_ms",
                round(float(statistics.mean(trans_sse_latencies)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "transcription_sse_latency_p95_ms",
                ts_pct[95],
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "transcription_sse_ttft_mean_ms",
                round(float(statistics.mean(trans_sse_ttfts)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "transcription_sse_rtf_mean",
                round(float(statistics.mean(trans_sse_rtfs)), 4),
                unit="x",
                better="lower",
            )

        combined_trans_lats = trans_json_latencies + trans_sse_latencies
        combined_trans_rtfs = trans_json_rtfs + trans_sse_rtfs
        if combined_trans_lats:
            result.set_metric(
                "transcription_latency_mean_ms",
                round(float(statistics.mean(combined_trans_lats)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "transcription_rtf_mean",
                round(float(statistics.mean(combined_trans_rtfs)), 4),
                unit="x",
                better="lower",
            )

        if diar_latencies:
            d_pct = compute_percentiles(diar_latencies)
            result.set_metric(
                "diarization_latency_mean_ms",
                round(float(statistics.mean(diar_latencies)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "diarization_latency_p95_ms",
                d_pct[95],
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "diarization_rtf_mean",
                round(float(statistics.mean(diar_rtfs)), 4),
                unit="x",
                better="lower",
            )

        if tts_latencies:
            tts_pct = compute_percentiles(tts_latencies)
            result.set_metric(
                "tts_latency_mean_ms",
                round(float(statistics.mean(tts_latencies)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "tts_latency_p95_ms",
                tts_pct[95],
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "tts_ttfb_mean_ms",
                round(float(statistics.mean(tts_ttfbs)), 2),
                unit="ms",
                better="lower",
            )
            if tts_rtfs:
                result.set_metric(
                    "tts_rtf_mean",
                    round(float(statistics.mean(tts_rtfs)), 4),
                    unit="x",
                    better="lower",
                )

        if admin_latencies:
            adm_pct = compute_percentiles(admin_latencies)
            result.set_metric(
                "admin_latency_mean_ms",
                round(float(statistics.mean(admin_latencies)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "admin_latency_p95_ms",
                adm_pct[95],
                unit="ms",
                better="lower",
            )

        if error_latencies:
            err_pct = compute_percentiles(error_latencies)
            result.set_metric(
                "error_handling_latency_mean_ms",
                round(float(statistics.mean(error_latencies)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "error_handling_latency_p95_ms",
                err_pct[95],
                unit="ms",
                better="lower",
            )

        if concurrency_latencies:
            conc_pct = compute_percentiles(concurrency_latencies)
            result.set_metric(
                "concurrency_latency_mean_ms",
                round(float(statistics.mean(concurrency_latencies)), 2),
                unit="ms",
                better="lower",
            )
            result.set_metric(
                "concurrency_latency_p95_ms",
                conc_pct[95],
                unit="ms",
                better="lower",
            )

        discovered_count = discovered.total_count if discovered else 0
        result.set_metric("models_discovered", discovered_count, unit="models", better="higher")
        result.set_metric("engines_tested", len(tested_engines), unit="engines", better="higher")
        result.set_metric("models_tested", len(tested_models), unit="models", better="higher")

        passed_cases = sum(1 for c in result.cases if c.status == "pass")
        total_cases = len(result.cases)
        pass_rate = round(passed_cases / total_cases * 100.0, 2) if total_cases > 0 else 0.0
        result.set_metric("cases_passed", passed_cases, unit="cases", better="higher")
        result.set_metric("cases_total", total_cases, unit="cases", better="higher")
        result.set_metric("pass_rate", pass_rate, unit="%", better="higher")

        result.status = "fail" if passed_cases < total_cases else "pass"
        result.summary = (
            f"live-stt HTTP test: {passed_cases}/{total_cases} cases passed ({pass_rate}%), "
            f"engines={len(tested_engines)}, models={len(tested_models)}, "
            f"mean_latency={overall_stats['latency_mean_ms']}ms, "
            f"mean_RTF={overall_stats['rtf_mean']}x"
        )

        models_summary = {
            "discovered_count": discovered_count,
            "models_schema_valid": models_schema_valid,
            "parametrization_mode": param_mode,
            "asr_models": [m.id for m in asr_models] if ("transcriptions_json" in suites_to_run or "transcriptions_sse" in suites_to_run) else [],
            "diarization_models": [m.id for m in diar_models] if "diarization" in suites_to_run else [],
            "tts_models": [m.id for m in tts_models] if "tts" in suites_to_run else [],
            "engines_tested": sorted(tested_engines),
            "models_tested": sorted(tested_models),
        }

        report_text = format_live_stt_report(
            base_url=client.root_url,
            models_summary=models_summary,
            metrics=result.metrics,
            cases=[c.to_dict() for c in result.cases],
        )
        report_file = ctx.artifacts_dir / "live_stt_report.txt"
        report_file.write_text(report_text, encoding="utf-8")
        ctx.add_artifact("live_stt_report.txt", report_file, mime="text/plain")

        detailed_json = {
            "tester": self.name,
            "base_url": client.root_url,
            "status": result.status,
            "summary": result.summary,
            "models_summary": models_summary,
            "suites_executed": suites_to_run,
            "metrics": result.metrics,
            "cases": [c.to_dict() for c in result.cases],
        }
        json_file = ctx.artifacts_dir / "live_stt_results.json"
        json_file.write_text(json.dumps(detailed_json, indent=2), encoding="utf-8")
        ctx.add_artifact("live_stt_results.json", json_file, mime="application/json")

        for art in ctx.artifacts:
            if art not in result.artifacts:
                result.artifacts.append(art)

        ctx.log(f"live-stt HTTP test finished with status: {result.status.upper()}")
        return result

    def _run_admin_suite(
        self,
        client: LiveSTTHTTPClient,
        ctx: TestContext,
        result: TesterResult,
        repetitions: int,
        include_extended_admin: bool,
        all_latencies_ms: list[float],
        admin_latencies: list[float],
        completed_cases: int,
        planned_cases: int,
    ) -> int:
        """Execute read-only GET checks against /v1/models, /api/health, /api/stats, and admin endpoints."""
        admin_specs: list[tuple[str, str, Any, Any]] = [
            (
                "admin:v1_models",
                "/v1/models",
                client.get_models_endpoint,
                lambda r: validate_models_response(
                    r.body,
                    status_code=r.status_code,
                    content_type=r.content_type,
                    strict_catalog=True,
                ),
            ),
            (
                "admin:api_health",
                "/api/health",
                client.get_health,
                lambda r: validate_health_response(
                    r.body,
                    status_code=r.status_code,
                    content_type=r.content_type,
                ),
            ),
            (
                "admin:api_stats",
                "/api/stats",
                client.get_stats,
                lambda r: validate_stats_response(
                    r.body,
                    status_code=r.status_code,
                    content_type=r.content_type,
                ),
            ),
        ]

        if include_extended_admin:
            admin_specs.extend([
                (
                    "admin:api_version",
                    "/api/version",
                    client.get_version,
                    lambda r: validate_version_response(
                        r.body,
                        status_code=r.status_code,
                        content_type=r.content_type,
                    ),
                ),
                (
                    "admin:api_config",
                    "/api/config",
                    client.get_config,
                    lambda r: validate_config_response(
                        r.body,
                        status_code=r.status_code,
                        content_type=r.content_type,
                    ),
                ),
                (
                    "admin:api_settings",
                    "/api/settings",
                    client.get_settings,
                    lambda r: validate_settings_response(
                        r.body,
                        status_code=r.status_code,
                        content_type=r.content_type,
                    ),
                ),
                (
                    "admin:api_backends",
                    "/api/backends",
                    client.get_backends,
                    lambda r: validate_backends_response(
                        r.body,
                        status_code=r.status_code,
                        content_type=r.content_type,
                    ),
                ),
                (
                    "admin:api_diarization_engines",
                    "/api/diarization-engines",
                    client.get_diarization_engines,
                    lambda r: validate_engine_status_response(
                        r.body,
                        status_code=r.status_code,
                        content_type=r.content_type,
                        expected_engines=("pyannote", "nemo_sortformer"),
                    ),
                ),
                (
                    "admin:api_tts_engines",
                    "/api/tts-engines",
                    client.get_tts_engines,
                    lambda r: validate_engine_status_response(
                        r.body,
                        status_code=r.status_code,
                        content_type=r.content_type,
                        expected_engines=("kokoro", "piper"),
                    ),
                ),
                (
                    "admin:metrics",
                    "/metrics",
                    client.get_metrics,
                    lambda r: validate_prometheus_metrics_response(
                        r.raw_text,
                        status_code=r.status_code,
                        content_type=r.content_type,
                    ),
                ),
                (
                    "admin:dashboard",
                    "/dashboard",
                    client.get_dashboard,
                    lambda r: validate_dashboard_html_response(
                        r.raw_text,
                        status_code=r.status_code,
                        content_type=r.content_type,
                    ),
                ),
            ])

        for case_id, endpoint_path, fetch_fn, validator_fn in admin_specs:
            ctx.check_cancelled()
            ctx.log(f"Running [{completed_cases + 1}/{planned_cases}] {case_id} (GET {endpoint_path}, read-only)...")

            lats: list[float] = []
            last_res = None
            for _ in range(repetitions):
                ctx.check_cancelled()
                try:
                    res = fetch_fn()
                except LiveSTTClientError as e:
                    ctx.log(f"Error calling GET {endpoint_path}: {e}", level="error")
                    raise TesterError(f"live-stt API error: {e}") from e
                last_res = res
                lats.append(res.latency_ms)
                all_latencies_ms.append(res.latency_ms)
                admin_latencies.append(res.latency_ms)

            assert last_res is not None
            schema_res: SchemaCheckResult = validator_fn(last_res)
            case_status = "pass" if schema_res.valid else "fail"
            mean_lat = round(float(statistics.mean(lats)), 2)

            case_metrics: dict[str, Any] = {
                "latency_ms": {"value": mean_lat, "unit": "ms", "better": "lower"},
                "status_code": {"value": last_res.status_code, "unit": "code"},
            }
            if endpoint_path == "/v1/models" and isinstance(last_res.body, dict):
                data_list = last_res.body.get("data")
                if isinstance(data_list, list):
                    case_metrics["models_count"] = {"value": len(data_list), "unit": "models"}
            elif endpoint_path == "/api/health" and isinstance(last_res.body, dict):
                cap = last_res.body.get("capacity")
                if isinstance(cap, dict):
                    if isinstance(cap.get("used"), int):
                        case_metrics["capacity_used"] = {"value": cap["used"], "unit": "slots"}
                    if isinstance(cap.get("total"), int):
                        case_metrics["capacity_total"] = {"value": cap["total"], "unit": "slots"}
            elif endpoint_path == "/api/stats" and isinstance(last_res.body, dict):
                if isinstance(last_res.body.get("active_calls"), int):
                    case_metrics["active_calls"] = {"value": last_res.body["active_calls"], "unit": "calls"}
                if isinstance(last_res.body.get("max_concurrent_calls"), int):
                    case_metrics["max_concurrent_calls"] = {
                        "value": last_res.body["max_concurrent_calls"],
                        "unit": "calls",
                    }

            result.add_case(
                case_id=case_id,
                suite="admin",
                status=case_status,
                metrics=case_metrics,
                detail={
                    "endpoint": endpoint_path,
                    "method": "GET",
                    "read_only": True,
                    "mutating": False,
                    "status_code": last_res.status_code,
                    "content_type": last_res.content_type,
                    "response_body": last_res.body if last_res.body is not None else last_res.raw_text[:500],
                    "passed": schema_res.valid,
                    "eval_message": schema_res.summary_message,
                    "checks": schema_res.checks,
                    "errors": schema_res.errors,
                },
            )
            completed_cases += 1
            ctx.log(
                f"  {case_id} -> {case_status.upper()} | "
                f"HTTP {last_res.status_code}, Latency={mean_lat:.2f}ms | {schema_res.summary_message}"
            )
            ctx.progress(completed_cases, planned_cases, f"Completed {case_id}")

        return completed_cases

    def _run_error_handling_suite(
        self,
        client: LiveSTTHTTPClient,
        ctx: TestContext,
        result: TesterResult,
        asr_model_id: str,
        diar_model_id: str,
        tts_model_id: str,
        all_latencies_ms: list[float],
        error_latencies: list[float],
        completed_cases: int,
        planned_cases: int,
    ) -> int:
        """Execute error-path contract checks: missing fields, bad audio, unknown models, invalid params, and 404."""
        valid_wav = get_audio_fixture("short_speech").wav_bytes
        corrupt_wav = get_bad_audio_fixture("corrupt_non_wav").wav_bytes
        wav_8khz = get_bad_audio_fixture("wrong_sample_rate_8khz").wav_bytes
        wav_stereo = get_bad_audio_fixture("stereo_2ch_wav").wav_bytes
        wav_8bit = get_bad_audio_fixture("wrong_bit_depth_8bit").wav_bytes

        error_specs: list[tuple[str, str, str, str, Any, int, str, str | None]] = [
            # 1-3: Missing required fields
            (
                "error_handling:transcriptions:missing_file",
                "/v1/audio/transcriptions",
                "POST",
                "Missing 'file' multipart field in /v1/audio/transcriptions",
                lambda: client.post_multipart_raw(
                    "/v1/audio/transcriptions",
                    fields={"model": asr_model_id, "response_format": "json"},
                    files=None,
                ),
                400,
                "nested",
                "file",
            ),
            (
                "error_handling:diarization:missing_file",
                "/v1/audio/diarization",
                "POST",
                "Missing 'file' multipart field in /v1/audio/diarization",
                lambda: client.post_multipart_raw(
                    "/v1/audio/diarization",
                    fields={"model": diar_model_id},
                    files=None,
                ),
                400,
                "nested",
                "file",
            ),
            (
                "error_handling:tts:missing_input",
                "/v1/audio/speech",
                "POST",
                "Missing or empty 'input' text field in /v1/audio/speech",
                lambda: client.post_json_raw(
                    "/v1/audio/speech",
                    {"model": tts_model_id, "input": "   "},
                ),
                400,
                "any",
                "input",
            ),
            # 4-8: Bad audio (corrupt non-WAV, 8kHz, stereo, 8-bit)
            (
                "error_handling:transcriptions:bad_audio_corrupt_non_wav",
                "/v1/audio/transcriptions",
                "POST",
                "Corrupt non-WAV bytes sent to /v1/audio/transcriptions",
                lambda: client.transcribe_json(
                    wav_bytes=corrupt_wav,
                    model=asr_model_id,
                    response_format="json",
                    filename="corrupt.wav",
                ),
                400,
                "nested",
                "WAV",
            ),
            (
                "error_handling:transcriptions:bad_audio_wrong_sample_rate_8khz",
                "/v1/audio/transcriptions",
                "POST",
                "8kHz WAV sent to /v1/audio/transcriptions (16kHz required)",
                lambda: client.transcribe_json(
                    wav_bytes=wav_8khz,
                    model=asr_model_id,
                    response_format="json",
                    filename="8khz.wav",
                ),
                400,
                "nested",
                "16kHz",
            ),
            (
                "error_handling:transcriptions:bad_audio_stereo_2ch_wav",
                "/v1/audio/transcriptions",
                "POST",
                "Stereo 2-channel WAV sent to /v1/audio/transcriptions (mono required)",
                lambda: client.transcribe_json(
                    wav_bytes=wav_stereo,
                    model=asr_model_id,
                    response_format="json",
                    filename="stereo.wav",
                ),
                400,
                "nested",
                "mono",
            ),
            (
                "error_handling:transcriptions:bad_audio_wrong_bit_depth_8bit",
                "/v1/audio/transcriptions",
                "POST",
                "8-bit unsigned PCM WAV sent to /v1/audio/transcriptions (16-bit required)",
                lambda: client.transcribe_json(
                    wav_bytes=wav_8bit,
                    model=asr_model_id,
                    response_format="json",
                    filename="8bit.wav",
                ),
                400,
                "nested",
                "16-bit",
            ),
            (
                "error_handling:diarization:bad_audio_corrupt_non_wav",
                "/v1/audio/diarization",
                "POST",
                "Corrupt non-WAV bytes sent to /v1/audio/diarization",
                lambda: client.diarize(
                    wav_bytes=corrupt_wav,
                    model=diar_model_id,
                    filename="corrupt.wav",
                ),
                400,
                "nested",
                "WAV",
            ),
            # 9-11: Unknown models across ASR, Diarization, and TTS
            (
                "error_handling:transcriptions:unknown_model",
                "/v1/audio/transcriptions",
                "POST",
                "Unknown ASR model ID in /v1/audio/transcriptions",
                lambda: client.transcribe_json(
                    wav_bytes=valid_wav,
                    model="unknown-asr-model-xyz",
                    response_format="json",
                ),
                400,
                "nested",
                "unknown",
            ),
            (
                "error_handling:diarization:unknown_model",
                "/v1/audio/diarization",
                "POST",
                "Unknown diarization model ID in /v1/audio/diarization",
                lambda: client.diarize(
                    wav_bytes=valid_wav,
                    model="unknown-diarization-model-xyz",
                ),
                400,
                "nested",
                "unknown",
            ),
            (
                "error_handling:tts:unknown_model",
                "/v1/audio/speech",
                "POST",
                "Unknown TTS model ID in /v1/audio/speech",
                lambda: client.post_json_raw(
                    "/v1/audio/speech",
                    {"model": "unknown-tts-model-xyz", "input": "Hello world"},
                ),
                400,
                "any",
                "unknown",
            ),
            # 12-15: Invalid parameters & 404 route
            (
                "error_handling:transcriptions:unsupported_response_format",
                "/v1/audio/transcriptions",
                "POST",
                "Unsupported response_format='srt' in /v1/audio/transcriptions",
                lambda: client.transcribe_json(
                    wav_bytes=valid_wav,
                    model=asr_model_id,
                    response_format="srt",
                ),
                400,
                "nested",
                "response_format",
            ),
            (
                "error_handling:diarization:invalid_words_json",
                "/v1/audio/diarization",
                "POST",
                "Malformed 'words' JSON in /v1/audio/diarization",
                lambda: client.post_multipart_raw(
                    "/v1/audio/diarization",
                    fields={"model": diar_model_id, "words": "{not-valid-json"},
                    files={"file": ("fixture.wav", valid_wav, "audio/wav")},
                ),
                400,
                "nested",
                "words",
            ),
            (
                "error_handling:diarization:invalid_num_speakers",
                "/v1/audio/diarization",
                "POST",
                "Non-integer 'num_speakers' field in /v1/audio/diarization",
                lambda: client.post_multipart_raw(
                    "/v1/audio/diarization",
                    fields={"model": diar_model_id, "num_speakers": "not_an_integer"},
                    files={"file": ("fixture.wav", valid_wav, "audio/wav")},
                ),
                400,
                "nested",
                "num_speakers",
            ),
            (
                "error_handling:admin:unknown_route_404",
                "/api/nonexistent-contract-route",
                "GET",
                "Unknown route returns HTTP 404 with JSON error body",
                lambda: client.get_endpoint("/api/nonexistent-contract-route"),
                404,
                "any",
                "not found",
            ),
        ]

        for (
            case_id,
            endpoint_path,
            method,
            description,
            invoke_fn,
            expected_status,
            expected_shape,
            expected_sub,
        ) in error_specs:
            ctx.check_cancelled()
            ctx.log(
                f"Running [{completed_cases + 1}/{planned_cases}] {case_id} "
                f"({method} {endpoint_path}, expect HTTP {expected_status})..."
            )
            try:
                res = invoke_fn()
            except LiveSTTClientError as e:
                ctx.log(f"Error calling {method} {endpoint_path}: {e}", level="error")
                raise TesterError(f"live-stt API error: {e}") from e

            all_latencies_ms.append(res.latency_ms)
            error_latencies.append(res.latency_ms)

            err_schema = validate_error_response(
                body=res.body,
                status_code=res.status_code,
                content_type=res.content_type,
                expected_status=expected_status,
                shape=expected_shape,
                expected_substring=expected_sub,
            )
            case_status = "pass" if err_schema.valid else "fail"
            err_msg = extract_error_message(res.body) or ""

            result.add_case(
                case_id=case_id,
                suite="error_handling",
                status=case_status,
                metrics={
                    "latency_ms": {"value": res.latency_ms, "unit": "ms", "better": "lower"},
                    "status_code": {"value": res.status_code, "unit": "code"},
                    "expected_status_code": {"value": expected_status, "unit": "code"},
                },
                detail={
                    "endpoint": endpoint_path,
                    "method": method,
                    "description": description,
                    "status_code": res.status_code,
                    "expected_status_code": expected_status,
                    "expected_shape": expected_shape,
                    "expected_substring": expected_sub,
                    "content_type": res.content_type,
                    "error_message": err_msg,
                    "response_body": res.body,
                    "passed": err_schema.valid,
                    "eval_message": err_schema.summary_message,
                    "checks": err_schema.checks,
                    "errors": err_schema.errors,
                },
            )
            completed_cases += 1
            ctx.log(
                f"  {case_id} -> {case_status.upper()} | "
                f"HTTP {res.status_code} ({err_msg!r}), Latency={res.latency_ms:.2f}ms | "
                f"{err_schema.summary_message}"
            )
            ctx.progress(completed_cases, planned_cases, f"Completed {case_id}")

        return completed_cases

    def _run_concurrency_suite(
        self,
        client: LiveSTTHTTPClient,
        ctx: TestContext,
        result: TesterResult,
        cfg: dict[str, Any],
        asr_model_id: str,
        language: str,
        all_latencies_ms: list[float],
        concurrency_latencies: list[float],
        completed_cases: int,
        planned_cases: int,
    ) -> int:
        """Execute concurrency and admission limit tests (503 when over budget, slot release, stats counters)."""
        short_fix = get_audio_fixture("short_speech")

        # -------------------------------------------------------------------
        # Case 7a: Pre-check admission budget via GET /api/stats & /api/health
        # -------------------------------------------------------------------
        ctx.check_cancelled()
        case_pre = "concurrency:pre_check_budget"
        ctx.log(f"Running [{completed_cases + 1}/{planned_cases}] {case_pre} (GET /api/stats & /api/health)...")
        try:
            stats_before = client.get_stats()
            health_before = client.get_health()
        except LiveSTTClientError as e:
            ctx.log(f"Error querying pre-burst stats/health: {e}", level="error")
            raise TesterError(f"live-stt API error: {e}") from e

        pre_lat = round(stats_before.latency_ms + health_before.latency_ms, 2)
        all_latencies_ms.append(pre_lat)
        concurrency_latencies.append(pre_lat)

        stats_schema = validate_stats_response(
            stats_before.body,
            status_code=stats_before.status_code,
            content_type=stats_before.content_type,
        )
        health_schema = validate_health_response(
            health_before.body,
            status_code=health_before.status_code,
            content_type=health_before.content_type,
        )

        stats_body = stats_before.body if isinstance(stats_before.body, dict) else {}
        health_body = health_before.body if isinstance(health_before.body, dict) else {}
        max_concurrent_calls = int(
            stats_body.get("max_concurrent_calls")
            or (health_body.get("capacity", {}) if isinstance(health_body.get("capacity"), dict) else {}).get("total")
            or 2
        )
        trans_before = stats_body.get("transcribe") if isinstance(stats_body.get("transcribe"), dict) else {}
        rej_cap_before = int(trans_before.get("rejected_capacity_total", 0) or 0)
        rej_vram_before = int(trans_before.get("rejected_vram_total", 0) or 0)

        pre_checks = stats_schema.checks + health_schema.checks
        pre_errors = stats_schema.errors + health_schema.errors
        pre_valid = len(pre_errors) == 0
        pre_status = "pass" if pre_valid else "fail"

        result.add_case(
            case_id=case_pre,
            suite="concurrency",
            status=pre_status,
            metrics={
                "latency_ms": {"value": pre_lat, "unit": "ms", "better": "lower"},
                "max_concurrent_calls": {"value": max_concurrent_calls, "unit": "calls"},
                "rejected_capacity_before": {"value": rej_cap_before, "unit": "calls"},
            },
            detail={
                "endpoint": "/api/stats + /api/health",
                "method": "GET",
                "max_concurrent_calls": max_concurrent_calls,
                "rejected_capacity_before": rej_cap_before,
                "rejected_vram_before": rej_vram_before,
                "stats_body": stats_before.body,
                "health_body": health_before.body,
                "passed": pre_valid,
                "eval_message": (
                    f"Budget max_concurrent_calls={max_concurrent_calls} verified"
                    if pre_valid
                    else "; ".join(pre_errors)
                ),
                "checks": pre_checks,
                "errors": pre_errors,
            },
        )
        completed_cases += 1
        ctx.log(
            f"  {case_pre} -> {pre_status.upper()} | "
            f"max_concurrent_calls={max_concurrent_calls}, Latency={pre_lat:.2f}ms"
        )
        ctx.progress(completed_cases, planned_cases, f"Completed {case_pre}")

        # -------------------------------------------------------------------
        # Case 7b: Concurrent burst over budget -> 503 admission rejection
        # -------------------------------------------------------------------
        ctx.check_cancelled()
        burst_size = int(cfg.get("concurrency_burst_size") or (max_concurrent_calls + 2))
        case_burst = "concurrency:over_budget_burst"
        ctx.log(
            f"Running [{completed_cases + 1}/{planned_cases}] {case_burst} "
            f"(launching {burst_size} concurrent POST /v1/audio/transcriptions vs budget={max_concurrent_calls})..."
        )
        try:
            burst_res = client.run_concurrent_transcriptions(
                wav_bytes=short_fix.wav_bytes,
                model=asr_model_id,
                concurrency=burst_size,
                response_format="json",
                language=language,
            )
        except LiveSTTClientError as e:
            ctx.log(f"Error during concurrent transcription burst: {e}", level="error")
            raise TesterError(f"live-stt API error: {e}") from e

        all_latencies_ms.append(burst_res.wall_ms)
        concurrency_latencies.append(burst_res.wall_ms)

        burst_checks: list[dict[str, Any]] = []
        burst_errors: list[str] = []

        only_200_503 = burst_res.other_status_count == 0
        burst_checks.append({
            "name": "only_200_or_503_statuses",
            "passed": only_200_503,
            "detail": (
                f"200={burst_res.ok_200_count}, 503={burst_res.rejected_503_count}, "
                f"other={burst_res.other_status_count}"
            ),
        })
        if not only_200_503:
            burst_errors.append(
                f"only_200_or_503_statuses: got {burst_res.other_status_count} unexpected status code(s)"
            )

        has_admitted = burst_res.ok_200_count >= 1
        burst_checks.append({
            "name": "at_least_one_request_admitted_200",
            "passed": has_admitted,
            "detail": f"Admitted {burst_res.ok_200_count}/{burst_res.total_requests} requests with HTTP 200",
        })
        if not has_admitted:
            burst_errors.append("at_least_one_request_admitted_200: 0 requests succeeded with HTTP 200")

        ok_responses = [r for r in burst_res.responses if r.status_code == 200]
        ok_schemas_valid = all(
            validate_transcription_json_response(r.body, r.content_type, r.status_code).valid
            for r in ok_responses
        )
        burst_checks.append({
            "name": "admitted_200_responses_valid_schema",
            "passed": ok_schemas_valid,
            "detail": f"Validated {len(ok_responses)} HTTP 200 response(s)",
        })
        if not ok_schemas_valid:
            burst_errors.append("admitted_200_responses_valid_schema: one or more 200 responses failed schema check")

        if burst_size > max_concurrent_calls:
            saw_503 = burst_res.rejected_503_count >= 1
            burst_checks.append({
                "name": "over_budget_requests_rejected_503",
                "passed": saw_503,
                "detail": (
                    f"Rejected {burst_res.rejected_503_count} over-budget request(s) with HTTP 503 "
                    f"(burst={burst_size}, max_concurrent_calls={max_concurrent_calls})"
                ),
            })
            if not saw_503:
                burst_errors.append(
                    f"over_budget_requests_rejected_503: expected >=1 HTTP 503 when burst={burst_size} > "
                    f"max_concurrent_calls={max_concurrent_calls}, got 0"
                )

        rej_responses = [r for r in burst_res.responses if r.status_code == 503]
        if rej_responses:
            rej_schemas = [
                validate_error_response(
                    body=r.body,
                    status_code=r.status_code,
                    content_type=r.content_type,
                    expected_status=503,
                    shape="nested",
                )
                for r in rej_responses
            ]
            all_503_valid = all(s.valid for s in rej_schemas)
            sample_503_msg = extract_error_message(rej_responses[0].body) or ""
            msg_matches_admission = any(
                token in sample_503_msg.lower()
                for token in ("capacity", "vram", "draining", "busy", "limit")
            )
            burst_checks.append({
                "name": "rejected_503_error_body_contract",
                "passed": all_503_valid and msg_matches_admission,
                "detail": f"503 error.message={sample_503_msg!r}",
            })
            if not (all_503_valid and msg_matches_admission):
                burst_errors.append(
                    f"rejected_503_error_body_contract: invalid 503 body or unexpected message {sample_503_msg!r}"
                )

        burst_valid = len(burst_errors) == 0
        burst_status = "pass" if burst_valid else "fail"
        result.add_case(
            case_id=case_burst,
            suite="concurrency",
            status=burst_status,
            metrics={
                "latency_ms": {"value": burst_res.wall_ms, "unit": "ms", "better": "lower"},
                "burst_size": {"value": burst_res.total_requests, "unit": "requests"},
                "admitted_200_count": {"value": burst_res.ok_200_count, "unit": "requests"},
                "rejected_503_count": {"value": burst_res.rejected_503_count, "unit": "requests"},
            },
            detail={
                "endpoint": "/v1/audio/transcriptions",
                "method": "POST",
                "model": asr_model_id,
                "max_concurrent_calls": max_concurrent_calls,
                "burst_size": burst_res.total_requests,
                "ok_200_count": burst_res.ok_200_count,
                "rejected_503_count": burst_res.rejected_503_count,
                "other_status_count": burst_res.other_status_count,
                "passed": burst_valid,
                "eval_message": (
                    f"Burst {burst_res.total_requests} reqs -> {burst_res.ok_200_count} HTTP 200, "
                    f"{burst_res.rejected_503_count} HTTP 503"
                    if burst_valid
                    else "; ".join(burst_errors)
                ),
                "checks": burst_checks,
                "errors": burst_errors,
            },
        )
        completed_cases += 1
        ctx.log(
            f"  {case_burst} -> {burst_status.upper()} | "
            f"200={burst_res.ok_200_count}, 503={burst_res.rejected_503_count}, Wall={burst_res.wall_ms:.2f}ms"
        )
        ctx.progress(completed_cases, planned_cases, f"Completed {case_burst}")

        # -------------------------------------------------------------------
        # Case 7c: Post-burst slot release recovery & /api/stats rejection counter
        # -------------------------------------------------------------------
        ctx.check_cancelled()
        case_post = "concurrency:post_burst_recovery_and_stats"
        ctx.log(
            f"Running [{completed_cases + 1}/{planned_cases}] {case_post} "
            "(verifying slot release and /api/stats rejection telemetry)..."
        )
        try:
            rec_res = client.transcribe_json(
                wav_bytes=short_fix.wav_bytes,
                model=asr_model_id,
                response_format="json",
                language=language,
                filename="recovery.wav",
            )
            stats_after = client.get_stats()
            health_after = client.get_health()
        except LiveSTTClientError as e:
            ctx.log(f"Error during post-burst recovery check: {e}", level="error")
            raise TesterError(f"live-stt API error: {e}") from e

        post_lat = round(rec_res.latency_ms + stats_after.latency_ms + health_after.latency_ms, 2)
        all_latencies_ms.append(post_lat)
        concurrency_latencies.append(post_lat)

        post_checks: list[dict[str, Any]] = []
        post_errors: list[str] = []

        rec_schema = validate_transcription_json_response(
            rec_res.body,
            rec_res.content_type,
            rec_res.status_code,
        )
        post_checks.append({
            "name": "recovery_transcription_200",
            "passed": rec_schema.valid,
            "detail": f"HTTP {rec_res.status_code}, {rec_schema.summary_message}",
        })
        if not rec_schema.valid:
            post_errors.append(f"recovery_transcription_200: {rec_schema.summary_message}")

        stats_after_body = stats_after.body if isinstance(stats_after.body, dict) else {}
        health_after_body = health_after.body if isinstance(health_after.body, dict) else {}
        active_after = stats_after_body.get("active_calls")
        trans_after = (
            stats_after_body.get("transcribe")
            if isinstance(stats_after_body.get("transcribe"), dict)
            else {}
        )
        trans_active_after = trans_after.get("active")
        slots_released = active_after == 0 and trans_active_after == 0
        post_checks.append({
            "name": "active_calls_released_to_zero",
            "passed": slots_released,
            "detail": f"active_calls={active_after!r}, transcribe.active={trans_active_after!r}",
        })
        if not slots_released:
            post_errors.append(
                f"active_calls_released_to_zero: expected 0 active calls after burst, "
                f"got active_calls={active_after!r}, transcribe.active={trans_active_after!r}"
            )

        cap_after = (
            health_after_body.get("capacity")
            if isinstance(health_after_body.get("capacity"), dict)
            else {}
        )
        health_admitting = (
            health_after.status_code == 200
            and health_after_body.get("status") == "ok"
            and cap_after.get("admitting") is True
        )
        post_checks.append({
            "name": "health_status_ok_and_admitting",
            "passed": health_admitting,
            "detail": f"HTTP {health_after.status_code}, status={health_after_body.get('status')!r}, capacity={cap_after!r}",
        })
        if not health_admitting:
            post_errors.append("health_status_ok_and_admitting: /api/health not ok or not admitting after burst")

        rej_cap_after = int(trans_after.get("rejected_capacity_total", 0) or 0)
        rej_vram_after = int(trans_after.get("rejected_vram_total", 0) or 0)
        total_rej_delta = (rej_cap_after + rej_vram_after) - (rej_cap_before + rej_vram_before)
        if burst_res.rejected_503_count > 0:
            counter_incremented = total_rej_delta >= burst_res.rejected_503_count
            post_checks.append({
                "name": "stats_rejection_counter_incremented",
                "passed": counter_incremented,
                "detail": (
                    f"Rejection counter delta={total_rej_delta} "
                    f"(expected >={burst_res.rejected_503_count})"
                ),
            })
            if not counter_incremented:
                post_errors.append(
                    f"stats_rejection_counter_incremented: delta={total_rej_delta} < "
                    f"rejected_503_count={burst_res.rejected_503_count}"
                )

        post_valid = len(post_errors) == 0
        post_status = "pass" if post_valid else "fail"
        result.add_case(
            case_id=case_post,
            suite="concurrency",
            status=post_status,
            metrics={
                "latency_ms": {"value": post_lat, "unit": "ms", "better": "lower"},
                "active_calls_after": {"value": int(active_after or 0), "unit": "calls", "better": "lower"},
                "rejected_delta": {"value": total_rej_delta, "unit": "calls"},
            },
            detail={
                "endpoint": "/v1/audio/transcriptions + /api/stats + /api/health",
                "recovery_status_code": rec_res.status_code,
                "active_calls_after": active_after,
                "rejected_capacity_before": rej_cap_before,
                "rejected_capacity_after": rej_cap_after,
                "rejected_delta": total_rej_delta,
                "passed": post_valid,
                "eval_message": (
                    f"Post-burst recovery HTTP 200, active_calls=0, rejected_delta={total_rej_delta}"
                    if post_valid
                    else "; ".join(post_errors)
                ),
                "checks": post_checks,
                "errors": post_errors,
            },
        )
        completed_cases += 1
        ctx.log(
            f"  {case_post} -> {post_status.upper()} | "
            f"active_calls={active_after}, rejected_delta={total_rej_delta}, Latency={post_lat:.2f}ms"
        )
        ctx.progress(completed_cases, planned_cases, f"Completed {case_post}")

        return completed_cases

