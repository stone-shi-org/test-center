"""HTTP client and /v1/models parametrization for live-stt endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Sequence
import httpx


class LiveSTTClientError(Exception):
    """Exception raised for transport or authentication errors communicating with live-stt."""
    pass


@dataclass
class ModelInfo:
    """Metadata for a model discovered via GET /v1/models or configured explicitly."""
    id: str
    type: str  # "asr" | "diarization" | "tts"
    engine: str
    is_default: bool = False
    streaming_capable: bool = True
    supports_num_speakers_hint: bool = True
    sample_rate: int | None = None
    default_voice: str | None = None
    supported_voices: list[str] = field(default_factory=list)
    supports_streaming: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "engine": self.engine,
            "is_default": self.is_default,
            "streaming_capable": self.streaming_capable,
            "supports_num_speakers_hint": self.supports_num_speakers_hint,
            "sample_rate": self.sample_rate,
            "default_voice": self.default_voice,
            "supported_voices": self.supported_voices,
            "supports_streaming": self.supports_streaming,
        }


@dataclass
class DiscoveredModels:
    """Parsed catalog of ASR, diarization, and TTS models from GET /v1/models."""
    status_code: int | None
    latency_ms: float
    raw_body: dict[str, Any] | None
    asr_models: list[ModelInfo] = field(default_factory=list)
    diarization_models: list[ModelInfo] = field(default_factory=list)
    tts_models: list[ModelInfo] = field(default_factory=list)
    error: str | None = None

    @property
    def total_count(self) -> int:
        return len(self.asr_models) + len(self.diarization_models) + len(self.tts_models)


@dataclass
class TranscribeJsonResult:
    """Telemetry and response from POST /v1/audio/transcriptions (JSON/verbose_json)."""
    status_code: int
    content_type: str
    body: Any
    raw_text: str
    latency_ms: float


@dataclass
class TranscribeSSEResult:
    """Telemetry and parsed SSE events from POST /v1/audio/transcriptions (stream=true)."""
    status_code: int
    content_type: str
    text: str
    events: list[dict[str, Any]]
    saw_done: bool
    ttft_ms: float
    latency_ms: float
    raw_body: str


@dataclass
class DiarizeResult:
    """Telemetry and response from POST /v1/audio/diarization."""
    status_code: int
    content_type: str
    body: Any
    raw_text: str
    latency_ms: float


@dataclass
class SpeechResult:
    """Telemetry and audio bytes from POST /v1/audio/speech (batch or streaming)."""
    status_code: int
    content_type: str
    headers: dict[str, str]
    audio_bytes: bytes
    chunk_count: int
    ttfb_ms: float
    latency_ms: float
    error_body: Any = None


DEFAULT_ASR_MODEL = ModelInfo(
    id="realtime_eou_120m-v1",
    type="asr",
    engine="parakeet",
    is_default=True,
    streaming_capable=True,
)

DEFAULT_DIARIZATION_MODEL = ModelInfo(
    id="pyannote/speaker-diarization-community-1",
    type="diarization",
    engine="pyannote",
    is_default=True,
    supports_num_speakers_hint=True,
)

DEFAULT_TTS_MODEL = ModelInfo(
    id="kokoro-82m-v1",
    type="tts",
    engine="kokoro",
    is_default=True,
    sample_rate=24000,
    default_voice="af_heart",
    supported_voices=["af_heart", "af_bella", "am_adam", "bf_emma"],
    supports_streaming=True,
)


def infer_model_info(item: dict[str, Any]) -> ModelInfo:
    """Parse a raw model dict from GET /v1/models into a structured ModelInfo."""
    model_id = str(item.get("id", "")).strip()
    id_lower = model_id.lower()

    raw_type = item.get("type")
    if raw_type in ("asr", "diarization", "tts"):
        model_type = str(raw_type)
    elif any(k in id_lower for k in ("diar", "pyannote", "sortformer")):
        model_type = "diarization"
    elif any(k in id_lower for k in ("tts", "kokoro", "piper")):
        model_type = "tts"
    else:
        model_type = "asr"

    raw_engine = item.get("engine")
    if isinstance(raw_engine, str) and raw_engine.strip():
        engine = raw_engine.strip()
    elif model_type == "diarization":
        engine = "nemo_sortformer" if ("sortformer" in id_lower or "nvidia" in id_lower) else "pyannote"
    elif model_type == "tts":
        engine = "piper" if "piper" in id_lower else "kokoro"
    else:
        if "whisper" in id_lower:
            engine = "whisper"
        elif "tdt" in id_lower:
            engine = "parakeet_tdt"
        else:
            engine = "parakeet"

    is_default = bool(item.get("default", False))
    streaming_capable = bool(item.get("streaming_capable", True))
    supports_hint = bool(item.get("supports_num_speakers_hint", engine == "pyannote"))

    sample_rate = item.get("sample_rate")
    if isinstance(sample_rate, (int, float)) and not isinstance(sample_rate, bool):
        sample_rate = int(sample_rate)
    elif model_type == "tts":
        sample_rate = 22050 if engine == "piper" else 24000
    else:
        sample_rate = None

    default_voice = item.get("default_voice")
    if not isinstance(default_voice, str) or not default_voice.strip():
        if model_type == "tts":
            default_voice = "en_US-lessac-medium" if engine == "piper" else "af_heart"
        else:
            default_voice = None

    raw_voices = item.get("supported_voices")
    supported_voices = [str(v) for v in raw_voices] if isinstance(raw_voices, list) else []
    if model_type == "tts" and not supported_voices and default_voice:
        supported_voices = [default_voice]

    supports_streaming = bool(item.get("supports_streaming", True))

    return ModelInfo(
        id=model_id,
        type=model_type,
        engine=engine,
        is_default=is_default,
        streaming_capable=streaming_capable,
        supports_num_speakers_hint=supports_hint,
        sample_rate=sample_rate,
        default_voice=default_voice,
        supported_voices=supported_voices,
        supports_streaming=supports_streaming,
        raw=dict(item),
    )


def select_parametrized_models(
    discovered: Sequence[ModelInfo],
    model_type: str,
    mode: str = "per_engine",
    explicit_models: Sequence[str] | None = None,
    engines_filter: Sequence[str] | None = None,
    fallback: ModelInfo | None = None,
) -> list[ModelInfo]:
    """Select models for testing according to parametrization mode and optional filters.

    Modes:
    - 'per_engine': one model per distinct engine (prefers model with is_default=True)
    - 'all': all discovered models for the service
    - 'default': only the default (or first) model for the service
    """
    by_id = {m.id: m for m in discovered}

    # Explicit model list takes highest priority
    if explicit_models:
        selected: list[ModelInfo] = []
        for mid in explicit_models:
            clean_id = str(mid).strip()
            if not clean_id:
                continue
            if clean_id in by_id:
                selected.append(by_id[clean_id])
            else:
                selected.append(infer_model_info({"id": clean_id, "type": model_type}))
        if selected:
            return selected

    candidates = list(discovered)
    if engines_filter:
        norm_filter = {str(e).strip().lower() for e in engines_filter if str(e).strip()}
        if norm_filter:
            filtered = [
                m for m in candidates
                if m.engine.lower() in norm_filter or m.id.lower() in norm_filter
            ]
            if filtered:
                candidates = filtered

    if not candidates:
        return [fallback] if fallback is not None else []

    norm_mode = (mode or "per_engine").strip().lower()
    if norm_mode == "all":
        return candidates

    if norm_mode == "default":
        defaults = [m for m in candidates if m.is_default]
        return [defaults[0]] if defaults else [candidates[0]]

    # Default: "per_engine" — group by engine preserving discovery order
    engine_groups: dict[str, list[ModelInfo]] = {}
    for m in candidates:
        engine_groups.setdefault(m.engine, []).append(m)

    chosen: list[ModelInfo] = []
    for group in engine_groups.values():
        default_in_group = next((m for m in group if m.is_default), None)
        chosen.append(default_in_group if default_in_group is not None else group[0])
    return chosen


class LiveSTTHTTPClient:
    """HTTP client for live-stt /v1/models, /v1/audio/transcriptions, /v1/audio/diarization, and /v1/audio/speech."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.raw_base_url = base_url.strip()
        self.api_key = api_key.strip() if api_key else None
        self.timeout = timeout
        self.transport = transport
        self.default_headers = default_headers or {}
        self.root_url = self._normalize_root_url(self.raw_base_url)
        self.models_url = f"{self.root_url}/v1/models"
        self.transcriptions_url = f"{self.root_url}/v1/audio/transcriptions"
        self.diarization_url = f"{self.root_url}/v1/audio/diarization"
        self.speech_url = f"{self.root_url}/v1/audio/speech"

    @staticmethod
    def _normalize_root_url(base_url: str) -> str:
        url = base_url.rstrip("/")
        for suffix in (
            "/v1/audio/transcriptions",
            "/v1/audio/diarization",
            "/v1/audio/speech",
            "/v1/models",
            "/v1",
        ):
            if url.endswith(suffix):
                url = url[: -len(suffix)].rstrip("/")
                break
        return url

    def _build_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {**self.default_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            headers.update(extra)
        return headers

    def discover_models(self) -> DiscoveredModels:
        """Query GET /v1/models and partition models into ASR, diarization, and TTS lists."""
        headers = self._build_headers({"Accept": "application/json"})
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=min(self.timeout, 15.0), transport=self.transport) as client:
                resp = client.get(self.models_url, headers=headers)
        except httpx.RequestError as e:
            raise LiveSTTClientError(f"HTTP request to {self.models_url} failed: {e}") from e

        elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)

        if resp.status_code in (401, 403):
            raise LiveSTTClientError(
                f"Authentication failed at {self.models_url} (status {resp.status_code}): {resp.text}"
            )

        if resp.status_code != 200:
            return DiscoveredModels(
                status_code=resp.status_code,
                latency_ms=elapsed_ms,
                raw_body=None,
                error=f"GET {self.models_url} returned status {resp.status_code}: {resp.text}",
            )

        try:
            body = resp.json()
        except Exception as e:
            return DiscoveredModels(
                status_code=resp.status_code,
                latency_ms=elapsed_ms,
                raw_body=None,
                error=f"Invalid JSON from {self.models_url}: {e}",
            )

        asr_models: list[ModelInfo] = []
        diar_models: list[ModelInfo] = []
        tts_models: list[ModelInfo] = []

        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list):
            for raw_item in data:
                if not isinstance(raw_item, dict) or not raw_item.get("id"):
                    continue
                info = infer_model_info(raw_item)
                if info.type == "diarization":
                    diar_models.append(info)
                elif info.type == "tts":
                    tts_models.append(info)
                else:
                    asr_models.append(info)

        return DiscoveredModels(
            status_code=resp.status_code,
            latency_ms=elapsed_ms,
            raw_body=body if isinstance(body, dict) else None,
            asr_models=asr_models,
            diarization_models=diar_models,
            tts_models=tts_models,
        )

    def transcribe_json(
        self,
        wav_bytes: bytes,
        model: str,
        response_format: str = "json",
        language: str | None = None,
        filename: str = "fixture.wav",
    ) -> TranscribeJsonResult:
        """Send non-streaming POST /v1/audio/transcriptions request."""
        headers = self._build_headers({"Accept": "application/json"})
        data: dict[str, str] = {
            "model": model,
            "response_format": response_format,
            "stream": "false",
        }
        if language:
            data["language"] = language

        files = {"file": (filename, wav_bytes, "audio/wav")}

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                resp = client.post(
                    self.transcriptions_url,
                    data=data,
                    files=files,
                    headers=headers,
                )
        except httpx.RequestError as e:
            raise LiveSTTClientError(
                f"HTTP request to {self.transcriptions_url} failed: {e}"
            ) from e

        latency_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)

        if resp.status_code in (401, 403):
            raise LiveSTTClientError(
                f"Authentication failed at {self.transcriptions_url} (status {resp.status_code}): {resp.text}"
            )

        content_type = resp.headers.get("content-type", "")
        raw_text = resp.text
        try:
            body = resp.json()
        except Exception:
            body = None

        return TranscribeJsonResult(
            status_code=resp.status_code,
            content_type=content_type,
            body=body,
            raw_text=raw_text,
            latency_ms=latency_ms,
        )

    def transcribe_sse(
        self,
        wav_bytes: bytes,
        model: str,
        language: str | None = None,
        filename: str = "fixture.wav",
    ) -> TranscribeSSEResult:
        """Send streaming POST /v1/audio/transcriptions (stream=true) request and parse SSE events."""
        headers = self._build_headers({"Accept": "text/event-stream, application/json"})
        data: dict[str, str] = {
            "model": model,
            "stream": "true",
        }
        if language:
            data["language"] = language

        files = {"file": (filename, wav_bytes, "audio/wav")}

        t0 = time.perf_counter()
        first_event_time: float | None = None
        events: list[dict[str, Any]] = []
        text_parts: list[str] = []
        done_text: str | None = None
        saw_done = False
        raw_lines: list[str] = []

        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                with client.stream(
                    "POST",
                    self.transcriptions_url,
                    data=data,
                    files=files,
                    headers=headers,
                ) as resp:
                    status_code = resp.status_code
                    content_type = resp.headers.get("content-type", "")

                    if status_code in (401, 403):
                        err_body = resp.read().decode("utf-8", errors="replace")
                        raise LiveSTTClientError(
                            f"Authentication failed at {self.transcriptions_url} (status {status_code}): {err_body}"
                        )

                    if status_code != 200:
                        err_body = resp.read().decode("utf-8", errors="replace")
                        latency_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
                        return TranscribeSSEResult(
                            status_code=status_code,
                            content_type=content_type,
                            text="",
                            events=[],
                            saw_done=False,
                            ttft_ms=latency_ms,
                            latency_ms=latency_ms,
                            raw_body=err_body,
                        )

                    for raw_line in resp.iter_lines():
                        raw_lines.append(raw_line)
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            now = time.perf_counter()
                            if first_event_time is None:
                                first_event_time = now
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                saw_done = True
                                break
                            try:
                                evt = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(evt, dict):
                                events.append(evt)
                                evt_type = evt.get("type")
                                evt_text = evt.get("text")
                                if evt_type == "transcript.text.done" and isinstance(evt_text, str):
                                    done_text = evt_text
                                elif isinstance(evt_text, str):
                                    text_parts.append(evt_text)
        except httpx.RequestError as e:
            raise LiveSTTClientError(
                f"HTTP request to {self.transcriptions_url} failed: {e}"
            ) from e

        t_end = time.perf_counter()
        latency_ms = round(max(0.01, (t_end - t0) * 1000.0), 2)
        if first_event_time is not None:
            ttft_ms = round(max(0.01, min((first_event_time - t0) * 1000.0, latency_ms)), 2)
        else:
            ttft_ms = latency_ms

        final_text = done_text if done_text is not None else "".join(text_parts)

        return TranscribeSSEResult(
            status_code=status_code,
            content_type=content_type,
            text=final_text,
            events=events,
            saw_done=saw_done,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
            raw_body="\n".join(raw_lines),
        )

    def diarize(
        self,
        wav_bytes: bytes,
        model: str | None = None,
        words: list[dict[str, Any]] | None = None,
        include_text: bool = False,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        device: str | None = None,
        response_format: str = "verbose_json",
        filename: str = "fixture.wav",
    ) -> DiarizeResult:
        """Send POST /v1/audio/diarization request."""
        headers = self._build_headers({"Accept": "application/json"})
        data: dict[str, str] = {
            "response_format": response_format,
            "include_text": "true" if include_text else "false",
        }
        if model:
            data["model"] = model
        if words is not None:
            data["words"] = json.dumps(words)
        if num_speakers is not None:
            data["num_speakers"] = str(int(num_speakers))
        if min_speakers is not None:
            data["min_speakers"] = str(int(min_speakers))
        if max_speakers is not None:
            data["max_speakers"] = str(int(max_speakers))
        if device:
            data["device"] = device

        files = {"file": (filename, wav_bytes, "audio/wav")}

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                resp = client.post(
                    self.diarization_url,
                    data=data,
                    files=files,
                    headers=headers,
                )
        except httpx.RequestError as e:
            raise LiveSTTClientError(
                f"HTTP request to {self.diarization_url} failed: {e}"
            ) from e

        latency_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)

        if resp.status_code in (401, 403):
            raise LiveSTTClientError(
                f"Authentication failed at {self.diarization_url} (status {resp.status_code}): {resp.text}"
            )

        content_type = resp.headers.get("content-type", "")
        raw_text = resp.text
        try:
            body = resp.json()
        except Exception:
            body = None

        return DiarizeResult(
            status_code=resp.status_code,
            content_type=content_type,
            body=body,
            raw_text=raw_text,
            latency_ms=latency_ms,
        )

    def synthesize_speech(
        self,
        input_text: str,
        model: str | None = None,
        voice: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
        stream: bool = False,
    ) -> SpeechResult:
        """Send POST /v1/audio/speech request (batch or chunked streaming)."""
        headers = self._build_headers({"Content-Type": "application/json"})
        payload: dict[str, Any] = {
            "input": input_text,
            "response_format": response_format,
            "speed": speed,
            "stream": bool(stream),
        }
        if model:
            payload["model"] = model
        if voice:
            payload["voice"] = voice

        t0 = time.perf_counter()

        if stream:
            first_byte_time: float | None = None
            chunks: list[bytes] = []
            try:
                with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                    with client.stream(
                        "POST",
                        self.speech_url,
                        json=payload,
                        headers=headers,
                    ) as resp:
                        status_code = resp.status_code
                        content_type = resp.headers.get("content-type", "")
                        resp_headers = {k.lower(): v for k, v in resp.headers.items()}

                        if status_code in (401, 403):
                            err_text = resp.read().decode("utf-8", errors="replace")
                            raise LiveSTTClientError(
                                f"Authentication failed at {self.speech_url} (status {status_code}): {err_text}"
                            )

                        if status_code != 200:
                            err_bytes = resp.read()
                            latency_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
                            try:
                                err_body = json.loads(err_bytes.decode("utf-8", errors="replace"))
                            except Exception:
                                err_body = err_bytes.decode("utf-8", errors="replace")
                            return SpeechResult(
                                status_code=status_code,
                                content_type=content_type,
                                headers=resp_headers,
                                audio_bytes=b"",
                                chunk_count=0,
                                ttfb_ms=latency_ms,
                                latency_ms=latency_ms,
                                error_body=err_body,
                            )

                        for chunk in resp.iter_bytes():
                            if chunk:
                                now = time.perf_counter()
                                if first_byte_time is None:
                                    first_byte_time = now
                                chunks.append(chunk)
            except httpx.RequestError as e:
                raise LiveSTTClientError(
                    f"HTTP request to {self.speech_url} failed: {e}"
                ) from e

            t_end = time.perf_counter()
            latency_ms = round(max(0.01, (t_end - t0) * 1000.0), 2)
            ttfb_ms = (
                round(max(0.01, min((first_byte_time - t0) * 1000.0, latency_ms)), 2)
                if first_byte_time is not None
                else latency_ms
            )
            audio_bytes = b"".join(chunks)
            return SpeechResult(
                status_code=status_code,
                content_type=content_type,
                headers=resp_headers,
                audio_bytes=audio_bytes,
                chunk_count=len(chunks),
                ttfb_ms=ttfb_ms,
                latency_ms=latency_ms,
            )

        # Non-streaming request
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                resp = client.post(
                    self.speech_url,
                    json=payload,
                    headers=headers,
                )
        except httpx.RequestError as e:
            raise LiveSTTClientError(
                f"HTTP request to {self.speech_url} failed: {e}"
            ) from e

        latency_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)

        if resp.status_code in (401, 403):
            raise LiveSTTClientError(
                f"Authentication failed at {self.speech_url} (status {resp.status_code}): {resp.text}"
            )

        content_type = resp.headers.get("content-type", "")
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}

        if resp.status_code != 200:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            return SpeechResult(
                status_code=resp.status_code,
                content_type=content_type,
                headers=resp_headers,
                audio_bytes=b"",
                chunk_count=0,
                ttfb_ms=latency_ms,
                latency_ms=latency_ms,
                error_body=err_body,
            )

        audio_bytes = resp.content
        return SpeechResult(
            status_code=resp.status_code,
            content_type=content_type,
            headers=resp_headers,
            audio_bytes=audio_bytes,
            chunk_count=1 if len(audio_bytes) > 0 else 0,
            ttfb_ms=latency_ms,
            latency_ms=latency_ms,
        )
