"""gRPC client and streaming harness for live-stt (livestt.v1.StreamingASR).

Vendors only ``proto/livestt/v1/asr.proto`` (zero dependency on ``live-stt``
application code), generates Python stubs in ``backend.app.testers.live_stt.pb``,
feeds 16kHz mono 16-bit PCM audio in real time (``speed=1.0``), faster than real
time (``speed > 1.0``, e.g. ``2.0``, ``4.0``), or unpaced burst mode
(``speed=0`` / ``realtime=False``), and records ``Ready``, partial
(``TranscriptDelta``), ``EndOfUtterance``, ``EndOfBoundary``, ``Warning``,
``Recycled``, and ``Final`` events with both monotonic and wall-clock timestamps.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any
from urllib.parse import urlparse
import uuid
import wave

os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")

import grpc

from backend.app.testers.live_stt.client import LiveSTTClientError
from backend.app.testers.live_stt.fixtures import (
    AudioFixture,
    BUILTIN_AUDIO_FIXTURES,
    WordTimestamp,
    get_audio_fixture,
)
from backend.app.testers.live_stt.metrics import compute_rtf
from backend.app.testers.live_stt.pb import asr_pb2, asr_pb2_grpc


DEFAULT_GRPC_PORT = 50051
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_CHUNK_MS = 20.0


class LiveSTTGRPCError(LiveSTTClientError):
    """Exception raised for gRPC status errors or transport failures when ``raise_on_grpc_error=True``."""

    def __init__(
        self,
        message: str,
        *,
        status_code: str = "UNKNOWN",
        status_int: int = -1,
        details: str | None = None,
        partial_result: StreamingSessionResult | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.status_int = status_int
        self.details = details
        self.partial_result = partial_result


@dataclass(frozen=True)
class ExtractedAudio:
    """Validated or raw PCM16 mono audio buffer ready for gRPC streaming."""

    pcm_bytes: bytes
    sample_rate_hz: int
    channels: int
    sample_width: int
    duration_sec: float
    source_format: str  # "wav" | "pcm" | "fixture"
    fixture_id: str | None = None
    reference_text: str | None = None


@dataclass(frozen=True)
class RecordedWord:
    """Word-level timestamp and confidence entry from a partial (TranscriptDelta) or Final event."""

    text: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_sec": round(float(self.start_sec), 3),
            "end_sec": round(float(self.end_sec), 3),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass(frozen=True)
class RecordedChunkSend:
    """Timestamp record for a single outgoing PCM audio chunk sent on the gRPC stream."""

    chunk_index: int
    byte_length: int
    audio_duration_sec: float
    cumulative_audio_sec: float
    sent_at_unix: float
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "byte_length": self.byte_length,
            "audio_duration_sec": round(self.audio_duration_sec, 4),
            "cumulative_audio_sec": round(self.cumulative_audio_sec, 4),
            "sent_at_unix": round(self.sent_at_unix, 6),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class RecordedStreamEvent:
    """Timestamped event received from the server on a ``StreamingASR.Transcribe`` stream."""

    seq: int
    kind: str  # "ready" | "delta" | "eou" | "eob" | "warning" | "recycled" | "final"
    event_type: str  # "ready" | "partial" | "eou" | "eob" | "warning" | "recycled" | "final"
    timestamp_unix: float
    timestamp_iso: str
    elapsed_ms: float
    since_first_audio_ms: float | None = None
    since_half_close_ms: float | None = None
    audio_sent_sec_at_receipt: float = 0.0
    text: str = ""
    words: list[RecordedWord] = field(default_factory=list)
    audio_offset_sec: float | None = None
    at_sec: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_partial(self) -> bool:
        """Return True if this event is an incremental partial (``TranscriptDelta``)."""
        return self.kind == "delta"

    @property
    def is_final(self) -> bool:
        """Return True if this event is the terminal ``Final`` event."""
        return self.kind == "final"

    @property
    def is_ready(self) -> bool:
        """Return True if this event is the initial ``Ready`` handshake event."""
        return self.kind == "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "event_type": self.event_type,
            "is_partial": self.is_partial,
            "is_final": self.is_final,
            "timestamp_unix": round(self.timestamp_unix, 6),
            "timestamp_iso": self.timestamp_iso,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "since_first_audio_ms": (
                round(self.since_first_audio_ms, 2)
                if self.since_first_audio_ms is not None
                else None
            ),
            "since_half_close_ms": (
                round(self.since_half_close_ms, 2)
                if self.since_half_close_ms is not None
                else None
            ),
            "audio_sent_sec_at_receipt": round(self.audio_sent_sec_at_receipt, 4),
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "audio_offset_sec": (
                round(self.audio_offset_sec, 4)
                if self.audio_offset_sec is not None
                else None
            ),
            "at_sec": round(self.at_sec, 4) if self.at_sec is not None else None,
            "detail": self.detail,
        }


@dataclass
class ServerInfoResult:
    """Telemetry and response from ``StreamingASR.GetServerInfo``."""

    ok: bool
    grpc_status_code: str
    grpc_status_int: int
    latency_ms: float
    version: str = ""
    built_at: str = ""
    parakeet_ref: str = ""
    parakeet_abi_version: int = 0
    default_model: str = ""
    backend: str = ""
    ggml_cpu_features: str = ""
    max_concurrent_calls: int = 0
    active_calls: int = 0
    warm_spares: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "grpc_status_code": self.grpc_status_code,
            "grpc_status_int": self.grpc_status_int,
            "latency_ms": round(self.latency_ms, 2),
            "version": self.version,
            "built_at": self.built_at,
            "parakeet_ref": self.parakeet_ref,
            "parakeet_abi_version": self.parakeet_abi_version,
            "default_model": self.default_model,
            "backend": self.backend,
            "ggml_cpu_features": self.ggml_cpu_features,
            "max_concurrent_calls": self.max_concurrent_calls,
            "active_calls": self.active_calls,
            "warm_spares": self.warm_spares,
            "error_message": self.error_message,
        }


@dataclass
class StreamingSessionResult:
    """Full recorded trace and computed metrics from one ``StreamingASR.Transcribe`` session."""

    ok: bool
    grpc_status_code: str
    grpc_status_int: int
    call_id: str
    requested_model: str
    resolved_model: str
    language: str
    sample_rate_hz: int
    audio_duration_sec: float
    chunk_ms: float
    speed: float | None
    realtime: bool
    bytes_sent: int
    chunks_sent: int
    started_at_unix: float
    started_at_iso: str
    finished_at_unix: float
    finished_at_iso: str
    text: str
    partial_texts: list[str] = field(default_factory=list)
    final_text: str = ""
    words: list[RecordedWord] = field(default_factory=list)
    events: list[RecordedStreamEvent] = field(default_factory=list)
    chunk_sends: list[RecordedChunkSend] = field(default_factory=list)
    ready_latency_ms: float | None = None
    first_partial_ms: float | None = None
    ttfp_ms: float | None = None
    ttfp_from_first_audio_ms: float | None = None
    first_audio_sent_ms: float | None = None
    last_audio_sent_ms: float | None = None
    half_close_ms: float | None = None
    send_duration_ms: float = 0.0
    final_latency_ms: float | None = None
    total_latency_ms: float = 0.0
    rtf: float = 0.0
    effective_feed_speed: float | None = None
    inter_partial_latencies_ms: list[float] = field(default_factory=list)
    worker_generations: int = 0
    supports_turn_detection: bool | None = None
    has_punctuation: bool | None = None
    model_chunk_ms: int | None = None
    accepted_sample_rate_hz: int | None = None
    total_audio_sec_reported: float | None = None
    error_message: str | None = None

    @property
    def model(self) -> str:
        return self.resolved_model or self.requested_model

    @property
    def time_to_first_partial_ms(self) -> float | None:
        return self.ttfp_ms

    @property
    def finalization_latency_ms(self) -> float | None:
        return self.final_latency_ms

    @property
    def latency_ms(self) -> float:
        return self.total_latency_ms

    @property
    def partials(self) -> list[RecordedStreamEvent]:
        """All incremental partial (``TranscriptDelta``) events in receipt order."""
        return [e for e in self.events if e.kind == "delta"]

    @property
    def final_event(self) -> RecordedStreamEvent | None:
        """The terminal ``Final`` event, or None if the stream did not reach Final."""
        return next((e for e in reversed(self.events) if e.kind == "final"), None)

    @property
    def ready_event(self) -> RecordedStreamEvent | None:
        """The initial ``Ready`` event, or None if the stream failed before Ready."""
        return next((e for e in self.events if e.kind == "ready"), None)

    @property
    def warnings(self) -> list[RecordedStreamEvent]:
        return [e for e in self.events if e.kind == "warning"]

    @property
    def recycled_events(self) -> list[RecordedStreamEvent]:
        return [e for e in self.events if e.kind == "recycled"]

    @property
    def eou_events(self) -> list[RecordedStreamEvent]:
        return [e for e in self.events if e.kind == "eou"]

    @property
    def eob_events(self) -> list[RecordedStreamEvent]:
        return [e for e in self.events if e.kind == "eob"]

    @property
    def partial_count(self) -> int:
        return len(self.partials)

    @property
    def word_count(self) -> int:
        if self.words:
            return len(self.words)
        return len(self.text.split()) if self.text.strip() else 0

    def to_dict(self, include_chunk_sends: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "grpc_status_code": self.grpc_status_code,
            "grpc_status_int": self.grpc_status_int,
            "error_message": self.error_message,
            "call_id": self.call_id,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "language": self.language,
            "sample_rate_hz": self.sample_rate_hz,
            "audio_duration_sec": round(self.audio_duration_sec, 4),
            "chunk_ms": self.chunk_ms,
            "speed": self.speed,
            "realtime": self.realtime,
            "bytes_sent": self.bytes_sent,
            "chunks_sent": self.chunks_sent,
            "started_at_unix": round(self.started_at_unix, 6),
            "started_at_iso": self.started_at_iso,
            "finished_at_unix": round(self.finished_at_unix, 6),
            "finished_at_iso": self.finished_at_iso,
            "text": self.text,
            "partial_texts": list(self.partial_texts),
            "final_text": self.final_text,
            "partial_count": self.partial_count,
            "word_count": self.word_count,
            "words": [w.to_dict() for w in self.words],
            "events": [e.to_dict() for e in self.events],
            "ready_latency_ms": (
                round(self.ready_latency_ms, 2)
                if self.ready_latency_ms is not None
                else None
            ),
            "first_partial_ms": (
                round(self.first_partial_ms, 2)
                if self.first_partial_ms is not None
                else None
            ),
            "ttfp_ms": round(self.ttfp_ms, 2) if self.ttfp_ms is not None else None,
            "ttfp_from_first_audio_ms": (
                round(self.ttfp_from_first_audio_ms, 2)
                if self.ttfp_from_first_audio_ms is not None
                else None
            ),
            "first_audio_sent_ms": (
                round(self.first_audio_sent_ms, 2)
                if self.first_audio_sent_ms is not None
                else None
            ),
            "last_audio_sent_ms": (
                round(self.last_audio_sent_ms, 2)
                if self.last_audio_sent_ms is not None
                else None
            ),
            "half_close_ms": (
                round(self.half_close_ms, 2)
                if self.half_close_ms is not None
                else None
            ),
            "send_duration_ms": round(self.send_duration_ms, 2),
            "final_latency_ms": (
                round(self.final_latency_ms, 2)
                if self.final_latency_ms is not None
                else None
            ),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "rtf": round(self.rtf, 4),
            "effective_feed_speed": (
                round(self.effective_feed_speed, 4)
                if self.effective_feed_speed is not None
                else None
            ),
            "inter_partial_latencies_ms": [
                round(v, 2) for v in self.inter_partial_latencies_ms
            ],
            "worker_generations": self.worker_generations,
            "supports_turn_detection": self.supports_turn_detection,
            "has_punctuation": self.has_punctuation,
            "model_chunk_ms": self.model_chunk_ms,
            "accepted_sample_rate_hz": self.accepted_sample_rate_hz,
            "total_audio_sec_reported": (
                round(self.total_audio_sec_reported, 4)
                if self.total_audio_sec_reported is not None
                else None
            ),
            "warning_count": len(self.warnings),
            "recycled_count": len(self.recycled_events),
            "eou_count": len(self.eou_events),
            "eob_count": len(self.eob_events),
        }
        if include_chunk_sends:
            payload["chunk_sends"] = [c.to_dict() for c in self.chunk_sends]
        return payload


@dataclass
class ConcurrentStreamingResult:
    """Aggregated telemetry from running multiple concurrent ``Transcribe`` sessions."""

    total_sessions: int
    ok_count: int
    resource_exhausted_count: int
    unavailable_count: int
    invalid_argument_count: int
    other_error_count: int
    wall_ms: float
    sessions: list[StreamingSessionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_sessions": self.total_sessions,
            "ok_count": self.ok_count,
            "resource_exhausted_count": self.resource_exhausted_count,
            "unavailable_count": self.unavailable_count,
            "invalid_argument_count": self.invalid_argument_count,
            "other_error_count": self.other_error_count,
            "wall_ms": round(self.wall_ms, 2),
            "sessions": [s.to_dict() for s in self.sessions],
        }


def _utc_iso_from_unix(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_grpc_target(
    target: str | dict[str, Any],
    config: dict[str, Any] | None = None,
    default_grpc_port: int = DEFAULT_GRPC_PORT,
) -> str:
    """Resolve a ``host:port`` gRPC target address from a target dict or URL string.

    Precedence:
    1. ``config["grpc_target"]`` or ``config["grpc_url"]``
    2. ``target["grpc_target"]``, ``target["grpc_url"]``, ``target["extra"]["grpc_target"]``, or ``target["extra"]["grpc_url"]``
    3. ``grpc_host`` + port in ``config`` / ``target`` / ``target["extra"]``
    4. Host from ``base_url`` (or raw string) combined with the port, the URL's port, or ``default_grpc_port`` (50051).

    Port precedence: the target's ``grpc_port`` (``target`` / ``target["extra"]``, the target form's
    gRPC Port field) wins over ``config["grpc_port"]``, which only applies when the target sets none.
    Saved run configs and templates can carry ``grpc_port: 50051`` from an old schema default, and
    that must not override the port configured on the target.
    """
    cfg = config if isinstance(config, dict) else {}
    tgt_dict = target if isinstance(target, dict) else {}
    extra = tgt_dict.get("extra") if isinstance(tgt_dict.get("extra"), dict) else {}

    explicit_port: int | None = None
    for port_candidate in (
        tgt_dict.get("grpc_port"),
        extra.get("grpc_port"),
        cfg.get("grpc_port"),
    ):
        if port_candidate is not None and str(port_candidate).strip():
            try:
                explicit_port = int(port_candidate)
                break
            except (TypeError, ValueError):
                pass

    for direct in (
        cfg.get("grpc_target"),
        cfg.get("grpc_url"),
        tgt_dict.get("grpc_target"),
        tgt_dict.get("grpc_url"),
        extra.get("grpc_target"),
        extra.get("grpc_url"),
    ):
        if isinstance(direct, str) and direct.strip():
            return _normalize_grpc_endpoint_str(
                direct.strip(),
                explicit_port=None,
                default_port=default_grpc_port,
            )

    for host_candidate in (
        cfg.get("grpc_host"),
        tgt_dict.get("grpc_host"),
        extra.get("grpc_host"),
    ):
        if isinstance(host_candidate, str) and host_candidate.strip():
            return _normalize_grpc_endpoint_str(
                host_candidate.strip(),
                explicit_port=explicit_port,
                default_port=default_grpc_port,
            )

    raw_base = (
        target
        if isinstance(target, str)
        else (tgt_dict.get("base_url") or cfg.get("base_url") or "")
    )
    raw_base_str = str(raw_base).strip()
    if not raw_base_str:
        raise ValueError("Target is missing 'base_url' or 'grpc_target'")

    return _normalize_grpc_endpoint_str(
        raw_base_str,
        explicit_port=explicit_port,
        default_port=default_grpc_port,
    )


def _normalize_grpc_endpoint_str(
    endpoint: str,
    explicit_port: int | None = None,
    default_port: int = DEFAULT_GRPC_PORT,
) -> str:
    raw = endpoint.strip()
    if not raw:
        raise ValueError("Empty gRPC target endpoint")

    if "://" in raw:
        parsed = urlparse(raw)
        host = parsed.hostname or "127.0.0.1"
        url_port = parsed.port
        port = explicit_port if explicit_port is not None else (url_port or default_port)
        if ":" in host and not host.startswith("["):
            return f"[{host}]:{port}"
        return f"{host}:{port}"

    # Handle bare host, host:port, or [ipv6]:port
    parsed = urlparse(f"grpc://{raw}")
    host = parsed.hostname or raw
    url_port = parsed.port
    port = explicit_port if explicit_port is not None else (url_port or default_port)
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def is_wav_bytes(data: bytes) -> bool:
    """Return True if ``data`` begins with a RIFF/WAVE header."""
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def extract_pcm16_audio(
    audio: AudioFixture | bytes | bytearray | memoryview | str | Path,
    *,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    channels: int = 1,
    sample_width: int = 2,
    validate_wav: bool = True,
) -> ExtractedAudio:
    """Extract raw int16 little-endian PCM bytes and metadata from a fixture, WAV buffer, WAV path, or raw PCM bytes."""
    if isinstance(audio, AudioFixture):
        extracted = extract_pcm16_audio(
            audio.wav_bytes,
            sample_rate_hz=audio.sample_rate,
            channels=audio.channels,
            sample_width=audio.sample_width,
            validate_wav=validate_wav,
        )
        return ExtractedAudio(
            pcm_bytes=extracted.pcm_bytes,
            sample_rate_hz=extracted.sample_rate_hz,
            channels=extracted.channels,
            sample_width=extracted.sample_width,
            duration_sec=audio.duration_sec,
            source_format="fixture",
            fixture_id=audio.id,
            reference_text=audio.reference_text,
        )

    if isinstance(audio, (str, Path)):
        path_obj = Path(audio)
        if path_obj.is_file():
            raw_bytes = path_obj.read_bytes()
            return extract_pcm16_audio(
                raw_bytes,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                sample_width=sample_width,
                validate_wav=validate_wav,
            )
        fixture_key = str(audio).strip()
        if fixture_key in BUILTIN_AUDIO_FIXTURES:
            return extract_pcm16_audio(
                BUILTIN_AUDIO_FIXTURES[fixture_key],
                validate_wav=validate_wav,
            )
        raise FileNotFoundError(
            f"Audio file or built-in fixture '{audio}' not found"
        )

    raw = bytes(audio)
    if is_wav_bytes(raw):
        try:
            with wave.open(io.BytesIO(raw), "rb") as wf:
                wav_sr = int(wf.getframerate())
                wav_ch = int(wf.getnchannels())
                wav_sw = int(wf.getsampwidth())
                n_frames = int(wf.getnframes())
                pcm_bytes = wf.readframes(n_frames)
        except (wave.Error, EOFError, RuntimeError) as exc:
            if validate_wav:
                raise ValueError(f"Failed to parse WAV container: {exc}") from exc
            pcm_bytes = raw
            wav_sr = sample_rate_hz
            wav_ch = channels
            wav_sw = sample_width
            n_frames = len(pcm_bytes) // max(1, wav_ch * wav_sw)

        if validate_wav:
            if wav_sr != 16000:
                raise ValueError(
                    f"WAV sample rate must be 16000Hz for live-stt gRPC streaming, got {wav_sr}Hz"
                )
            if wav_ch != 1:
                raise ValueError(
                    f"WAV channel count must be 1 (mono) for live-stt gRPC streaming, got {wav_ch}"
                )
            if wav_sw != 2:
                raise ValueError(
                    f"WAV sample width must be 2 bytes (16-bit PCM) for live-stt gRPC streaming, got {wav_sw}"
                )

        bytes_per_sec = max(1, wav_sr * wav_ch * wav_sw)
        duration_sec = len(pcm_bytes) / float(bytes_per_sec)
        return ExtractedAudio(
            pcm_bytes=pcm_bytes,
            sample_rate_hz=wav_sr,
            channels=wav_ch,
            sample_width=wav_sw,
            duration_sec=duration_sec,
            source_format="wav",
        )

    bytes_per_sec = max(1, sample_rate_hz * channels * sample_width)
    duration_sec = len(raw) / float(bytes_per_sec)
    return ExtractedAudio(
        pcm_bytes=raw,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width=sample_width,
        duration_sec=duration_sec,
        source_format="pcm",
    )


def slice_pcm16_chunks(
    pcm_bytes: bytes,
    *,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    chunk_ms: float = DEFAULT_CHUNK_MS,
    channels: int = 1,
    sample_width: int = 2,
) -> list[bytes]:
    """Slice raw PCM bytes into frames of ``chunk_ms`` milliseconds."""
    if chunk_ms <= 0:
        raise ValueError(f"chunk_ms must be positive, got {chunk_ms}")
    if not pcm_bytes:
        return []
    frame_bytes = max(1, channels * sample_width)
    samples_per_chunk = max(1, int(round(sample_rate_hz * (chunk_ms / 1000.0))))
    chunk_size = samples_per_chunk * frame_bytes
    return [
        pcm_bytes[offset : offset + chunk_size]
        for offset in range(0, len(pcm_bytes), chunk_size)
    ]


async def iter_paced_pcm_chunks(
    chunks: Sequence[bytes],
    *,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    channels: int = 1,
    sample_width: int = 2,
    speed: float | None = 1.0,
    realtime: bool = True,
    cancel_check: Callable[[], None] | None = None,
    sleep_fn: Callable[[float], Any] = asyncio.sleep,
    clock_fn: Callable[[], float] = time.perf_counter,
) -> AsyncIterator[tuple[int, bytes, float, float]]:
    """Yield ``(chunk_index, chunk_bytes, chunk_duration_sec, cumulative_audio_sec)`` paced at ``speed``.

    - ``realtime=True, speed=1.0``: real-time pacing (1.0s of audio fed over 1.0s wall time).
    - ``realtime=True, speed=2.0``: 2x faster than real-time (1.0s of audio fed over 0.5s wall time).
    - ``realtime=False`` or ``speed is None`` or ``speed <= 0`` or ``math.isinf(speed)``:
      unpaced burst mode (chunks yielded immediately with zero sleep).
    """
    bytes_per_sec = float(max(1, sample_rate_hz * channels * sample_width))
    paced = (
        bool(realtime)
        and speed is not None
        and speed > 0.0
        and not math.isinf(speed)
    )
    effective_speed = float(speed) if paced and speed is not None else 0.0

    t_origin: float | None = None
    cumulative_audio_sec = 0.0

    for idx, chunk in enumerate(chunks):
        if cancel_check is not None:
            cancel_check()

        if idx == 0:
            t_origin = clock_fn()
        elif paced and t_origin is not None and effective_speed > 0.0:
            target_elapsed_sec = cumulative_audio_sec / effective_speed
            now = clock_fn()
            delay_sec = (t_origin + target_elapsed_sec) - now
            if delay_sec > 0.0:
                await sleep_fn(delay_sec)
                if cancel_check is not None:
                    cancel_check()

        chunk_dur_sec = len(chunk) / bytes_per_sec
        cumulative_audio_sec += chunk_dur_sec
        yield idx, chunk, chunk_dur_sec, cumulative_audio_sec


def parse_transcription_event(
    event: Any,
    *,
    seq: int,
    t_recv_mono: float,
    t_recv_unix: float,
    t_stream_start_mono: float,
    t_first_audio_mono: float | None,
    t_half_close_mono: float | None,
    audio_sent_sec: float,
) -> RecordedStreamEvent:
    """Convert a protobuf ``TranscriptionEvent`` into a structured ``RecordedStreamEvent`` with timestamps."""
    kind = event.WhichOneof("event") or "unknown"
    elapsed_ms = max(0.0, (t_recv_mono - t_stream_start_mono) * 1000.0)
    since_first_audio_ms = (
        max(0.0, (t_recv_mono - t_first_audio_mono) * 1000.0)
        if t_first_audio_mono is not None
        else None
    )
    since_half_close_ms = (
        max(0.0, (t_recv_mono - t_half_close_mono) * 1000.0)
        if t_half_close_mono is not None
        else None
    )
    ts_iso = _utc_iso_from_unix(t_recv_unix)

    if kind == "ready":
        r = event.ready
        detail = {
            "model": str(r.model),
            "supports_turn_detection": bool(r.supports_turn_detection),
            "has_punctuation": bool(r.has_punctuation),
            "model_chunk_ms": int(r.model_chunk_ms),
            "accepted_sample_rate_hz": int(r.accepted_sample_rate_hz),
        }
        return RecordedStreamEvent(
            seq=seq,
            kind="ready",
            event_type="ready",
            timestamp_unix=t_recv_unix,
            timestamp_iso=ts_iso,
            elapsed_ms=elapsed_ms,
            since_first_audio_ms=since_first_audio_ms,
            since_half_close_ms=since_half_close_ms,
            audio_sent_sec_at_receipt=audio_sent_sec,
            detail=detail,
        )

    if kind == "delta":
        d = event.delta
        words = [
            RecordedWord(
                text=str(w.text),
                start_sec=float(w.start_sec),
                end_sec=float(w.end_sec),
                confidence=float(w.confidence),
            )
            for w in d.words
        ]
        offset_sec = float(d.audio_offset_sec)
        detail = {
            "text": str(d.text),
            "words": [w.to_dict() for w in words],
            "audio_offset_sec": round(offset_sec, 4),
        }
        return RecordedStreamEvent(
            seq=seq,
            kind="delta",
            event_type="partial",
            timestamp_unix=t_recv_unix,
            timestamp_iso=ts_iso,
            elapsed_ms=elapsed_ms,
            since_first_audio_ms=since_first_audio_ms,
            since_half_close_ms=since_half_close_ms,
            audio_sent_sec_at_receipt=audio_sent_sec,
            text=str(d.text),
            words=words,
            audio_offset_sec=offset_sec,
            detail=detail,
        )

    if kind == "final":
        f = event.final
        words = [
            RecordedWord(
                text=str(w.text),
                start_sec=float(w.start_sec),
                end_sec=float(w.end_sec),
                confidence=float(w.confidence),
            )
            for w in f.words
        ]
        total_sec = float(f.total_audio_sec)
        generations = int(f.worker_generations)
        detail = {
            "text": str(f.text),
            "words": [w.to_dict() for w in words],
            "total_audio_sec": round(total_sec, 4),
            "worker_generations": generations,
        }
        return RecordedStreamEvent(
            seq=seq,
            kind="final",
            event_type="final",
            timestamp_unix=t_recv_unix,
            timestamp_iso=ts_iso,
            elapsed_ms=elapsed_ms,
            since_first_audio_ms=since_first_audio_ms,
            since_half_close_ms=since_half_close_ms,
            audio_sent_sec_at_receipt=audio_sent_sec,
            text=str(f.text),
            words=words,
            audio_offset_sec=total_sec,
            detail=detail,
        )

    if kind == "eou":
        at_sec = float(event.eou.at_sec)
        return RecordedStreamEvent(
            seq=seq,
            kind="eou",
            event_type="eou",
            timestamp_unix=t_recv_unix,
            timestamp_iso=ts_iso,
            elapsed_ms=elapsed_ms,
            since_first_audio_ms=since_first_audio_ms,
            since_half_close_ms=since_half_close_ms,
            audio_sent_sec_at_receipt=audio_sent_sec,
            at_sec=at_sec,
            detail={"at_sec": round(at_sec, 4)},
        )

    if kind == "eob":
        at_sec = float(event.eob.at_sec)
        return RecordedStreamEvent(
            seq=seq,
            kind="eob",
            event_type="eob",
            timestamp_unix=t_recv_unix,
            timestamp_iso=ts_iso,
            elapsed_ms=elapsed_ms,
            since_first_audio_ms=since_first_audio_ms,
            since_half_close_ms=since_half_close_ms,
            audio_sent_sec_at_receipt=audio_sent_sec,
            at_sec=at_sec,
            detail={"at_sec": round(at_sec, 4)},
        )

    if kind == "warning":
        w = event.warning
        code_num = int(w.code)
        try:
            code_name = asr_pb2.WarningCode.Name(code_num)
        except ValueError:
            code_name = f"WARNING_CODE_{code_num}"
        at_sec = float(w.at_sec)
        detail = {
            "code": code_name,
            "code_num": code_num,
            "message": str(w.message),
            "at_sec": round(at_sec, 4),
            "behind_sec": round(float(w.behind_sec), 4),
            "dropped_sec": round(float(w.dropped_sec), 4),
            "deadline_sec": round(float(w.deadline_sec), 4),
        }
        return RecordedStreamEvent(
            seq=seq,
            kind="warning",
            event_type="warning",
            timestamp_unix=t_recv_unix,
            timestamp_iso=ts_iso,
            elapsed_ms=elapsed_ms,
            since_first_audio_ms=since_first_audio_ms,
            since_half_close_ms=since_half_close_ms,
            audio_sent_sec_at_receipt=audio_sent_sec,
            at_sec=at_sec,
            detail=detail,
        )

    if kind == "recycled":
        rec = event.recycled
        reason_num = int(rec.reason)
        try:
            reason_name = asr_pb2.RecycleReason.Name(reason_num)
        except ValueError:
            reason_name = f"RECYCLE_REASON_{reason_num}"
        at_audio_sec = float(rec.at_audio_sec)
        detail = {
            "reason": reason_name,
            "reason_num": reason_num,
            "gap_sec": round(float(rec.gap_sec), 4),
            "at_audio_sec": round(at_audio_sec, 4),
            "warm": bool(rec.warm),
        }
        return RecordedStreamEvent(
            seq=seq,
            kind="recycled",
            event_type="recycled",
            timestamp_unix=t_recv_unix,
            timestamp_iso=ts_iso,
            elapsed_ms=elapsed_ms,
            since_first_audio_ms=since_first_audio_ms,
            since_half_close_ms=since_half_close_ms,
            audio_sent_sec_at_receipt=audio_sent_sec,
            audio_offset_sec=at_audio_sec,
            at_sec=at_audio_sec,
            detail=detail,
        )

    return RecordedStreamEvent(
        seq=seq,
        kind=kind,
        event_type=kind,
        timestamp_unix=t_recv_unix,
        timestamp_iso=ts_iso,
        elapsed_ms=elapsed_ms,
        since_first_audio_ms=since_first_audio_ms,
        since_half_close_ms=since_half_close_ms,
        audio_sent_sec_at_receipt=audio_sent_sec,
        detail={},
    )


def _run_coro_sync(coro: Any) -> Any:
    """Execute an async coroutine synchronously, even if an event loop is already active on this thread."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


class LiveSTTGRPCClient:
    """Async and sync gRPC client for ``livestt.v1.StreamingASR``."""

    def __init__(
        self,
        target: str | dict[str, Any],
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        channel: grpc.aio.Channel | None = None,
        metadata: Sequence[tuple[str, str]] | None = None,
        channel_options: Sequence[tuple[str, Any]] | None = None,
    ) -> None:
        self.target = (
            resolve_grpc_target(target)
            if isinstance(target, dict) or "://" in str(target) or ":" not in str(target)
            else str(target).strip()
        )
        self.api_key = api_key.strip() if api_key else None
        self.timeout = float(timeout)
        self._external_channel = channel
        self._channel: grpc.aio.Channel | None = channel
        self._stub: Any | None = (
            asr_pb2_grpc.StreamingASRStub(channel) if channel is not None else None
        )
        self.default_metadata = list(metadata) if metadata else []
        self.channel_options = list(channel_options) if channel_options else []

    def _ensure_stub(self) -> Any:
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(
                self.target,
                options=self.channel_options or None,
            )
            self._stub = asr_pb2_grpc.StreamingASRStub(self._channel)
        return self._stub

    def _build_metadata(
        self, extra: Sequence[tuple[str, str]] | None = None
    ) -> list[tuple[str, str]]:
        md = list(self.default_metadata)
        if self.api_key:
            md.append(("authorization", f"Bearer {self.api_key}"))
        if extra:
            md.extend(extra)
        return md

    async def close(self) -> None:
        """Close the underlying gRPC channel if owned by this client."""
        if self._channel is not None and self._external_channel is None:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def __aenter__(self) -> "LiveSTTGRPCClient":
        self._ensure_stub()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def get_server_info(
        self,
        *,
        timeout: float | None = None,
        raise_on_grpc_error: bool = False,
    ) -> ServerInfoResult:
        """Call ``StreamingASR.GetServerInfo`` and return structured telemetry."""
        stub = self._ensure_stub()
        effective_timeout = timeout if timeout is not None else min(self.timeout, 15.0)
        metadata = self._build_metadata()
        t0 = time.perf_counter()

        try:
            resp = await stub.GetServerInfo(
                asr_pb2.ServerInfoRequest(),
                timeout=effective_timeout,
                metadata=metadata or None,
            )
            latency_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
            return ServerInfoResult(
                ok=True,
                grpc_status_code="OK",
                grpc_status_int=0,
                latency_ms=latency_ms,
                version=str(resp.version),
                built_at=str(resp.built_at),
                parakeet_ref=str(resp.parakeet_ref),
                parakeet_abi_version=int(resp.parakeet_abi_version),
                default_model=str(resp.default_model),
                backend=str(resp.backend),
                ggml_cpu_features=str(resp.ggml_cpu_features),
                max_concurrent_calls=int(resp.max_concurrent_calls),
                active_calls=int(resp.active_calls),
                warm_spares=int(resp.warm_spares),
            )
        except grpc.RpcError as exc:
            latency_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
            code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.UNKNOWN
            details = exc.details() if hasattr(exc, "details") else str(exc)
            code_name = code.name if hasattr(code, "name") else str(code)
            code_int = (
                int(code.value[0])
                if hasattr(code, "value") and isinstance(code.value, tuple)
                else -1
            )
            if raise_on_grpc_error:
                raise LiveSTTGRPCError(
                    f"GetServerInfo failed with {code_name}: {details}",
                    status_code=code_name,
                    status_int=code_int,
                    details=details,
                ) from exc
            return ServerInfoResult(
                ok=False,
                grpc_status_code=code_name,
                grpc_status_int=code_int,
                latency_ms=latency_ms,
                error_message=details,
            )

    def get_server_info_sync(
        self,
        *,
        timeout: float | None = None,
        raise_on_grpc_error: bool = False,
    ) -> ServerInfoResult:
        """Synchronous wrapper for ``get_server_info()``."""
        async def _run() -> ServerInfoResult:
            async with LiveSTTGRPCClient(
                self.target,
                api_key=self.api_key,
                timeout=self.timeout,
                metadata=self.default_metadata,
                channel_options=self.channel_options,
            ) as temp_client:
                return await temp_client.get_server_info(
                    timeout=timeout,
                    raise_on_grpc_error=raise_on_grpc_error,
                )

        if self._external_channel is not None or self._channel is not None:
            return _run_coro_sync(
                self.get_server_info(
                    timeout=timeout,
                    raise_on_grpc_error=raise_on_grpc_error,
                )
            )
        return _run_coro_sync(_run())

    def transcribe_events(
        self,
        config: Any,
        audio_chunks: AsyncIterator[bytes],
        *,
        timeout: float | None = None,
        metadata: Sequence[tuple[str, str]] | None = None,
    ) -> AsyncIterator[Any]:
        """Low-level bidirectional ``Transcribe`` stream yielding raw ``asr_pb2.TranscriptionEvent`` messages."""
        stub = self._ensure_stub()
        effective_timeout = timeout if timeout is not None else self.timeout
        md = self._build_metadata(metadata)

        async def _request_iter() -> AsyncIterator[Any]:
            yield asr_pb2.TranscriptionRequest(config=config)
            async for chunk in audio_chunks:
                yield asr_pb2.TranscriptionRequest(audio=chunk)

        return stub.Transcribe(
            _request_iter(),
            timeout=effective_timeout,
            metadata=md or None,
        )

    async def stream_transcribe(
        self,
        audio: AudioFixture | bytes | bytearray | memoryview | str | Path,
        *,
        model: str = "",
        language: str = "",
        call_id: str | None = None,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        encoding: int | None = None,
        enable_word_timestamps: bool = True,
        chunk_ms: float = DEFAULT_CHUNK_MS,
        speed: float | None = 1.0,
        realtime: bool = True,
        wait_for_ready: bool = True,
        validate_wav: bool = True,
        send_config: bool = True,
        send_duplicate_config: bool = False,
        custom_requests: Sequence[Any] | None = None,
        timeout: float | None = None,
        cancel_check: Callable[[], None] | None = None,
        on_event: Callable[[RecordedStreamEvent], None] | None = None,
        raise_on_grpc_error: bool = False,
        sleep_fn: Callable[[float], Any] = asyncio.sleep,
        clock_fn: Callable[[], float] = time.perf_counter,
    ) -> StreamingSessionResult:
        """Stream audio over ``StreamingASR.Transcribe``, pacing chunks and recording timestamped events.

        Args:
            audio: ``AudioFixture``, WAV bytes, raw PCM16 bytes, WAV file path, or fixture ID.
            model: ASR model key (empty string uses server default ``realtime_eou_120m-v1``).
            language: Optional language code (e.g. ``"en-US"``).
            call_id: Opaque call identifier (defaults to a generated UUID4 prefix).
            sample_rate_hz: Sample rate declared in ``StreamConfig`` (default 16000).
            encoding: Protobuf ``AudioEncoding`` enum value (default ``AUDIO_ENCODING_LINEAR16``).
            enable_word_timestamps: Request word-level timestamps on ``TranscriptDelta``/``Final``.
            chunk_ms: Outgoing PCM frame duration in milliseconds (default 20ms).
            speed: Feed rate multiplier relative to real time:
                - ``1.0``: real-time pacing (default)
                - ``> 1.0`` (e.g. ``2.0``, ``4.0``): faster than real-time
                - ``0.0`` or ``None``: unpaced burst mode (immediate send)
            realtime: When False, disables pacing regardless of ``speed``.
            wait_for_ready: Wait for server's ``Ready`` event before starting the audio chunk clock.
            validate_wav: Validate WAV header format (16kHz mono 16-bit PCM) before streaming.
            send_config: Send initial ``StreamConfig`` message (set False to test protocol errors).
            send_duplicate_config: Send a second ``StreamConfig`` after the first (for negative tests).
            custom_requests: Optional explicit sequence of ``TranscriptionRequest`` messages to send.
            timeout: Optional RPC timeout override in seconds.
            cancel_check: Optional callback invoked between chunks and events to abort on cancellation.
            on_event: Optional callback invoked immediately on each ``RecordedStreamEvent``.
            raise_on_grpc_error: Raise ``LiveSTTGRPCError`` on non-OK gRPC status instead of returning ``ok=False``.
        """
        stub = self._ensure_stub()
        effective_call_id = call_id if call_id is not None else f"tc-{uuid.uuid4().hex[:12]}"
        effective_encoding = (
            int(encoding)
            if encoding is not None
            else int(asr_pb2.AUDIO_ENCODING_LINEAR16)
        )

        extracted = extract_pcm16_audio(
            audio,
            sample_rate_hz=sample_rate_hz if sample_rate_hz > 0 else DEFAULT_SAMPLE_RATE_HZ,
            validate_wav=validate_wav,
        )
        pcm_chunks = slice_pcm16_chunks(
            extracted.pcm_bytes,
            sample_rate_hz=extracted.sample_rate_hz if extracted.sample_rate_hz > 0 else DEFAULT_SAMPLE_RATE_HZ,
            chunk_ms=chunk_ms,
            channels=extracted.channels,
            sample_width=extracted.sample_width,
        )

        stream_config = asr_pb2.StreamConfig(
            call_id=effective_call_id,
            encoding=effective_encoding,
            sample_rate_hz=int(sample_rate_hz),
            language=str(language or ""),
            model=str(model or ""),
            enable_word_timestamps=bool(enable_word_timestamps),
        )

        paced = (
            bool(realtime)
            and speed is not None
            and speed > 0.0
            and not math.isinf(speed)
        )
        effective_speed = float(speed) if paced and speed is not None else None

        started_at_unix = time.time()
        started_at_iso = _utc_iso_from_unix(started_at_unix)
        t_stream_start = clock_fn()

        t_first_audio_mono: float | None = None
        t_last_audio_mono: float | None = None
        t_half_close_mono: float | None = None
        audio_sent_sec = 0.0
        bytes_sent = 0
        chunks_sent = 0
        chunk_sends: list[RecordedChunkSend] = []

        ready_gate = asyncio.Event()
        stream_closed = False
        captured_cancel_exc: BaseException | None = None

        def _wrapped_cancel_check() -> None:
            nonlocal captured_cancel_exc, stream_closed
            if cancel_check is None:
                return
            try:
                cancel_check()
            except BaseException as exc:
                captured_cancel_exc = exc
                stream_closed = True
                ready_gate.set()
                raise

        async def _request_iterator() -> AsyncIterator[Any]:
            nonlocal t_first_audio_mono, t_last_audio_mono, t_half_close_mono
            nonlocal audio_sent_sec, bytes_sent, chunks_sent

            try:
                if custom_requests is not None:
                    for req in custom_requests:
                        if stream_closed:
                            return
                        _wrapped_cancel_check()
                        if req.WhichOneof("payload") == "audio":
                            now_m = clock_fn()
                            now_u = time.time()
                            if t_first_audio_mono is None:
                                t_first_audio_mono = now_m
                            t_last_audio_mono = now_m
                            c_len = len(req.audio)
                            c_dur = c_len / float(max(1, extracted.sample_rate_hz * 2))
                            audio_sent_sec += c_dur
                            bytes_sent += c_len
                            chunk_sends.append(
                                RecordedChunkSend(
                                    chunk_index=chunks_sent,
                                    byte_length=c_len,
                                    audio_duration_sec=c_dur,
                                    cumulative_audio_sec=audio_sent_sec,
                                    sent_at_unix=now_u,
                                    elapsed_ms=max(0.0, (now_m - t_stream_start) * 1000.0),
                                )
                            )
                            chunks_sent += 1
                        yield req
                    return

                if send_config:
                    yield asr_pb2.TranscriptionRequest(config=stream_config)
                    if wait_for_ready:
                        await ready_gate.wait()
                    if stream_closed:
                        return
                    if send_duplicate_config:
                        yield asr_pb2.TranscriptionRequest(config=stream_config)
                        return

                async for idx, chunk_bytes, chunk_dur_sec, cum_sec in iter_paced_pcm_chunks(
                    pcm_chunks,
                    sample_rate_hz=extracted.sample_rate_hz,
                    channels=extracted.channels,
                    sample_width=extracted.sample_width,
                    speed=speed,
                    realtime=realtime,
                    cancel_check=_wrapped_cancel_check,
                    sleep_fn=sleep_fn,
                    clock_fn=clock_fn,
                ):
                    if stream_closed:
                        return
                    now_m = clock_fn()
                    now_u = time.time()
                    if t_first_audio_mono is None:
                        t_first_audio_mono = now_m
                    t_last_audio_mono = now_m
                    audio_sent_sec = cum_sec
                    bytes_sent += len(chunk_bytes)
                    chunks_sent += 1
                    chunk_sends.append(
                        RecordedChunkSend(
                            chunk_index=idx,
                            byte_length=len(chunk_bytes),
                            audio_duration_sec=chunk_dur_sec,
                            cumulative_audio_sec=cum_sec,
                            sent_at_unix=now_u,
                            elapsed_ms=max(0.0, (now_m - t_stream_start) * 1000.0),
                        )
                    )
                    yield asr_pb2.TranscriptionRequest(audio=chunk_bytes)
                    if not send_config and idx == 0 and wait_for_ready:
                        await ready_gate.wait()
                        if stream_closed:
                            return
            finally:
                t_half_close_mono = clock_fn()

        events: list[RecordedStreamEvent] = []
        partial_texts: list[str] = []
        final_text = ""
        all_words: list[RecordedWord] = []
        resolved_model = str(model or "")
        supports_turn_detection: bool | None = None
        has_punctuation: bool | None = None
        model_chunk_ms: int | None = None
        accepted_sample_rate_hz: int | None = None
        worker_generations = 0
        total_audio_sec_reported: float | None = None

        grpc_status_code = "OK"
        grpc_status_int = 0
        error_message: str | None = None

        effective_timeout = timeout if timeout is not None else self.timeout
        metadata = self._build_metadata()

        try:
            call = stub.Transcribe(
                _request_iterator(),
                timeout=effective_timeout,
                metadata=metadata or None,
            )
            seq = 0
            async for raw_event in call:
                _wrapped_cancel_check()
                t_recv_mono = clock_fn()
                t_recv_unix = time.time()
                recorded = parse_transcription_event(
                    raw_event,
                    seq=seq,
                    t_recv_mono=t_recv_mono,
                    t_recv_unix=t_recv_unix,
                    t_stream_start_mono=t_stream_start,
                    t_first_audio_mono=t_first_audio_mono,
                    t_half_close_mono=t_half_close_mono,
                    audio_sent_sec=audio_sent_sec,
                )
                seq += 1
                events.append(recorded)

                if recorded.kind == "ready":
                    ready_gate.set()
                    if recorded.detail.get("model"):
                        resolved_model = str(recorded.detail["model"])
                    supports_turn_detection = recorded.detail.get("supports_turn_detection")
                    has_punctuation = recorded.detail.get("has_punctuation")
                    model_chunk_ms = recorded.detail.get("model_chunk_ms")
                    accepted_sample_rate_hz = recorded.detail.get("accepted_sample_rate_hz")
                elif recorded.kind == "delta":
                    partial_texts.append(recorded.text)
                    all_words.extend(recorded.words)
                elif recorded.kind == "final":
                    final_text = recorded.text
                    all_words.extend(recorded.words)
                    worker_generations = int(recorded.detail.get("worker_generations", 0))
                    total_audio_sec_reported = recorded.audio_offset_sec

                if on_event is not None:
                    on_event(recorded)
        except grpc.RpcError as exc:
            stream_closed = True
            ready_gate.set()
            if captured_cancel_exc is not None:
                raise captured_cancel_exc from exc
            code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.UNKNOWN
            details = exc.details() if hasattr(exc, "details") else str(exc)
            grpc_status_code = code.name if hasattr(code, "name") else str(code)
            grpc_status_int = (
                int(code.value[0])
                if hasattr(code, "value") and isinstance(code.value, tuple)
                else -1
            )
            error_message = details
        except asyncio.CancelledError as exc:
            stream_closed = True
            ready_gate.set()
            if captured_cancel_exc is not None:
                raise captured_cancel_exc from exc
            raise
        except BaseException:
            stream_closed = True
            ready_gate.set()
            raise
        finally:
            stream_closed = True
            ready_gate.set()

        t_end_mono = clock_fn()
        finished_at_unix = time.time()
        finished_at_iso = _utc_iso_from_unix(finished_at_unix)

        total_latency_ms = round(max(0.01, (t_end_mono - t_stream_start) * 1000.0), 2)
        first_audio_sent_ms = (
            round(max(0.0, (t_first_audio_mono - t_stream_start) * 1000.0), 2)
            if t_first_audio_mono is not None
            else None
        )
        last_audio_sent_ms = (
            round(max(0.0, (t_last_audio_mono - t_stream_start) * 1000.0), 2)
            if t_last_audio_mono is not None
            else None
        )
        half_close_ms = (
            round(max(0.0, (t_half_close_mono - t_stream_start) * 1000.0), 2)
            if t_half_close_mono is not None
            else None
        )
        send_duration_ms = (
            round(max(0.0, (t_last_audio_mono - t_first_audio_mono) * 1000.0), 2)
            if (t_first_audio_mono is not None and t_last_audio_mono is not None)
            else 0.0
        )

        ready_evt = next((e for e in events if e.kind == "ready"), None)
        ready_latency_ms = (
            round(ready_evt.elapsed_ms, 2) if ready_evt is not None else None
        )

        delta_events = [e for e in events if e.kind == "delta"]
        final_evt = next((e for e in reversed(events) if e.kind == "final"), None)

        first_partial_ms = (
            round(delta_events[0].elapsed_ms, 2) if delta_events else None
        )
        if delta_events:
            ttfp_ms: float | None = round(delta_events[0].elapsed_ms, 2)
            ttfp_from_first_audio_ms: float | None = (
                round(delta_events[0].since_first_audio_ms, 2)
                if delta_events[0].since_first_audio_ms is not None
                else ttfp_ms
            )
        elif final_evt is not None and final_evt.text:
            ttfp_ms = round(final_evt.elapsed_ms, 2)
            ttfp_from_first_audio_ms = (
                round(final_evt.since_first_audio_ms, 2)
                if final_evt.since_first_audio_ms is not None
                else ttfp_ms
            )
        else:
            ttfp_ms = None
            ttfp_from_first_audio_ms = None

        final_latency_ms: float | None = None
        if final_evt is not None:
            if final_evt.since_half_close_ms is not None:
                final_latency_ms = round(final_evt.since_half_close_ms, 2)
            elif half_close_ms is not None:
                final_latency_ms = round(max(0.0, final_evt.elapsed_ms - half_close_ms), 2)
            else:
                final_latency_ms = round(final_evt.elapsed_ms, 2)

        inter_partial_latencies_ms: list[float] = []
        for idx in range(1, len(delta_events)):
            gap = max(0.0, delta_events[idx].elapsed_ms - delta_events[idx - 1].elapsed_ms)
            inter_partial_latencies_ms.append(round(gap, 2))

        full_text = "".join(partial_texts) + final_text
        rtf_val = compute_rtf(total_latency_ms, extracted.duration_sec)
        effective_feed_speed: float | None = None
        if send_duration_ms > 0.0 and extracted.duration_sec > 0.0:
            effective_feed_speed = round(
                extracted.duration_sec / (send_duration_ms / 1000.0), 4
            )

        session_ok = grpc_status_code == "OK" and final_evt is not None

        result = StreamingSessionResult(
            ok=session_ok,
            grpc_status_code=grpc_status_code,
            grpc_status_int=grpc_status_int,
            call_id=effective_call_id,
            requested_model=str(model or ""),
            resolved_model=resolved_model,
            language=str(language or ""),
            sample_rate_hz=int(sample_rate_hz),
            audio_duration_sec=extracted.duration_sec,
            chunk_ms=float(chunk_ms),
            speed=effective_speed,
            realtime=paced,
            bytes_sent=bytes_sent,
            chunks_sent=chunks_sent,
            started_at_unix=started_at_unix,
            started_at_iso=started_at_iso,
            finished_at_unix=finished_at_unix,
            finished_at_iso=finished_at_iso,
            text=full_text,
            partial_texts=partial_texts,
            final_text=final_text,
            words=all_words,
            events=events,
            chunk_sends=chunk_sends,
            ready_latency_ms=ready_latency_ms,
            first_partial_ms=first_partial_ms,
            ttfp_ms=ttfp_ms,
            ttfp_from_first_audio_ms=ttfp_from_first_audio_ms,
            first_audio_sent_ms=first_audio_sent_ms,
            last_audio_sent_ms=last_audio_sent_ms,
            half_close_ms=half_close_ms,
            send_duration_ms=send_duration_ms,
            final_latency_ms=final_latency_ms,
            total_latency_ms=total_latency_ms,
            rtf=rtf_val,
            effective_feed_speed=effective_feed_speed,
            inter_partial_latencies_ms=inter_partial_latencies_ms,
            worker_generations=worker_generations,
            supports_turn_detection=supports_turn_detection,
            has_punctuation=has_punctuation,
            model_chunk_ms=model_chunk_ms,
            accepted_sample_rate_hz=accepted_sample_rate_hz,
            total_audio_sec_reported=total_audio_sec_reported,
            error_message=error_message,
        )

        if not session_ok and raise_on_grpc_error:
            raise LiveSTTGRPCError(
                f"gRPC Transcribe failed ({grpc_status_code}): {error_message or 'missing Final event'}",
                status_code=grpc_status_code,
                status_int=grpc_status_int,
                details=error_message,
                partial_result=result,
            )

        return result

    def stream_transcribe_sync(
        self,
        audio: AudioFixture | bytes | bytearray | memoryview | str | Path,
        **kwargs: Any,
    ) -> StreamingSessionResult:
        """Synchronous wrapper for ``stream_transcribe()``."""
        async def _run() -> StreamingSessionResult:
            async with LiveSTTGRPCClient(
                self.target,
                api_key=self.api_key,
                timeout=self.timeout,
                metadata=self.default_metadata,
                channel_options=self.channel_options,
            ) as temp_client:
                return await temp_client.stream_transcribe(audio, **kwargs)

        if self._external_channel is not None or self._channel is not None:
            return _run_coro_sync(self.stream_transcribe(audio, **kwargs))
        return _run_coro_sync(_run())


class LiveSTTStreamingHarness:
    """High-level streaming harness for running paced single or concurrent gRPC ``Transcribe`` calls.

    Used by TES-30 and TES-31 to feed audio in real time (``speed=1.0``), faster than
    real time (``speed > 1.0``), or unpaced burst mode (``speed=0`` / ``realtime=False``),
    record timestamped partial/final events, and measure TTFP, finalization latency,
    and real-time factor.
    """

    def __init__(
        self,
        target: str | dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        default_chunk_ms: float = DEFAULT_CHUNK_MS,
        default_speed: float | None = 1.0,
        default_realtime: bool = True,
        metadata: Sequence[tuple[str, str]] | None = None,
        channel_options: Sequence[tuple[str, Any]] | None = None,
    ) -> None:
        self.raw_target = target
        self.config = config or {}
        self.grpc_target = resolve_grpc_target(target, self.config)
        self.api_key = api_key
        self.timeout = float(timeout)
        self.default_chunk_ms = float(default_chunk_ms)
        self.default_speed = default_speed
        self.default_realtime = bool(default_realtime)
        self.metadata = list(metadata) if metadata else []
        self.channel_options = list(channel_options) if channel_options else []

    def create_client(
        self,
        *,
        channel: grpc.aio.Channel | None = None,
        timeout: float | None = None,
    ) -> LiveSTTGRPCClient:
        """Instantiate a ``LiveSTTGRPCClient`` bound to this harness's target."""
        return LiveSTTGRPCClient(
            self.grpc_target,
            api_key=self.api_key,
            timeout=timeout if timeout is not None else self.timeout,
            channel=channel,
            metadata=self.metadata,
            channel_options=self.channel_options,
        )

    async def get_server_info(
        self,
        *,
        timeout: float | None = None,
        raise_on_grpc_error: bool = False,
    ) -> ServerInfoResult:
        """Query ``StreamingASR.GetServerInfo`` on the target."""
        async with self.create_client(timeout=timeout) as client:
            return await client.get_server_info(
                timeout=timeout,
                raise_on_grpc_error=raise_on_grpc_error,
            )

    def get_server_info_sync(
        self,
        *,
        timeout: float | None = None,
        raise_on_grpc_error: bool = False,
    ) -> ServerInfoResult:
        """Synchronously query ``StreamingASR.GetServerInfo``."""
        return _run_coro_sync(
            self.get_server_info(
                timeout=timeout,
                raise_on_grpc_error=raise_on_grpc_error,
            )
        )

    async def run_session(
        self,
        audio: AudioFixture | bytes | bytearray | memoryview | str | Path = "short_speech",
        *,
        model: str = "",
        language: str = "",
        call_id: str | None = None,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        encoding: int | None = None,
        enable_word_timestamps: bool = True,
        chunk_ms: float | None = None,
        speed: float | None = ...,  # type: ignore[assignment]
        realtime: bool | None = None,
        wait_for_ready: bool = True,
        validate_wav: bool = True,
        send_config: bool = True,
        send_duplicate_config: bool = False,
        custom_requests: Sequence[Any] | None = None,
        timeout: float | None = None,
        cancel_check: Callable[[], None] | None = None,
        on_event: Callable[[RecordedStreamEvent], None] | None = None,
        raise_on_grpc_error: bool = False,
        sleep_fn: Callable[[float], Any] = asyncio.sleep,
        clock_fn: Callable[[], float] = time.perf_counter,
    ) -> StreamingSessionResult:
        """Run a single streaming ``Transcribe`` session and return its timestamped trace and metrics."""
        effective_chunk_ms = float(chunk_ms if chunk_ms is not None else self.default_chunk_ms)
        effective_speed = self.default_speed if speed is ... else speed
        effective_realtime = self.default_realtime if realtime is None else bool(realtime)

        async with self.create_client(timeout=timeout) as client:
            return await client.stream_transcribe(
                audio,
                model=model,
                language=language,
                call_id=call_id,
                sample_rate_hz=sample_rate_hz,
                encoding=encoding,
                enable_word_timestamps=enable_word_timestamps,
                chunk_ms=effective_chunk_ms,
                speed=effective_speed,
                realtime=effective_realtime,
                wait_for_ready=wait_for_ready,
                validate_wav=validate_wav,
                send_config=send_config,
                send_duplicate_config=send_duplicate_config,
                custom_requests=custom_requests,
                timeout=timeout,
                cancel_check=cancel_check,
                on_event=on_event,
                raise_on_grpc_error=raise_on_grpc_error,
                sleep_fn=sleep_fn,
                clock_fn=clock_fn,
            )

    def run_session_sync(
        self,
        audio: AudioFixture | bytes | bytearray | memoryview | str | Path = "short_speech",
        **kwargs: Any,
    ) -> StreamingSessionResult:
        """Synchronously run a single streaming ``Transcribe`` session."""
        return _run_coro_sync(self.run_session(audio, **kwargs))

    async def run_concurrent_sessions(
        self,
        concurrency: int,
        audio: AudioFixture | bytes | bytearray | memoryview | str | Path = "short_speech",
        *,
        model: str = "",
        language: str = "",
        chunk_ms: float | None = None,
        speed: float | None = ...,  # type: ignore[assignment]
        realtime: bool | None = None,
        wait_for_ready: bool = True,
        enable_word_timestamps: bool = True,
        timeout: float | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> ConcurrentStreamingResult:
        """Run ``concurrency`` simultaneous ``Transcribe`` streams and aggregate outcomes."""
        n = max(1, int(concurrency))
        t0 = time.perf_counter()

        async def _one(idx: int) -> StreamingSessionResult:
            return await self.run_session(
                audio,
                model=model,
                language=language,
                call_id=f"concurrent-{idx}-{uuid.uuid4().hex[:8]}",
                chunk_ms=chunk_ms,
                speed=speed,
                realtime=realtime,
                wait_for_ready=wait_for_ready,
                enable_word_timestamps=enable_word_timestamps,
                timeout=timeout,
                cancel_check=cancel_check,
                raise_on_grpc_error=False,
            )

        sessions = list(await asyncio.gather(*(_one(i) for i in range(n))))
        wall_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)

        ok_count = sum(1 for s in sessions if s.ok)
        res_exhausted = sum(
            1 for s in sessions if s.grpc_status_code == "RESOURCE_EXHAUSTED"
        )
        unavailable = sum(
            1 for s in sessions if s.grpc_status_code == "UNAVAILABLE"
        )
        invalid_arg = sum(
            1 for s in sessions if s.grpc_status_code == "INVALID_ARGUMENT"
        )
        other_err = n - ok_count - res_exhausted - unavailable - invalid_arg

        return ConcurrentStreamingResult(
            total_sessions=n,
            ok_count=ok_count,
            resource_exhausted_count=res_exhausted,
            unavailable_count=unavailable,
            invalid_argument_count=invalid_arg,
            other_error_count=other_err,
            wall_ms=wall_ms,
            sessions=sessions,
        )

    def run_concurrent_sessions_sync(
        self,
        concurrency: int,
        audio: AudioFixture | bytes | bytearray | memoryview | str | Path = "short_speech",
        **kwargs: Any,
    ) -> ConcurrentStreamingResult:
        """Synchronously run ``concurrency`` simultaneous ``Transcribe`` streams."""
        return _run_coro_sync(
            self.run_concurrent_sessions(concurrency, audio, **kwargs)
        )


# ===========================================================================
# In-process Mock/Fake StreamingASR gRPC server for offline testing
# ===========================================================================


class MockStreamingASRServicer(asr_pb2_grpc.StreamingASRServicer):
    """Configurable in-process ``StreamingASRServicer`` speaking the real ``livestt.v1`` wire protocol.

    Exercises real ``grpc.aio`` bidirectional streaming, ``StreamConfig`` validation,
    chunk coalescing to ``model_chunk_ms``, incremental ``TranscriptDelta`` partial
    events with word timestamps, optional ``EndOfUtterance``/``EndOfBoundary``/``Warning``/
    ``Recycled`` events, ``Final`` tail emission on half-close, concurrency admission
    (``RESOURCE_EXHAUSTED``), and drain mode (``UNAVAILABLE``).
    """

    def __init__(
        self,
        *,
        default_model: str = "realtime_eou_120m-v1",
        allowed_models: set[str] | None = None,
        batch_only_models: set[str] | None = None,
        backend: str = "cpu",
        version: str = "mock-0.1.0",
        built_at: str = "2026-10-16T00:00:00Z",
        parakeet_ref: str = "0123456789abcdef",
        parakeet_abi_version: int = 6,
        ggml_cpu_features: str = "AVX2 FMA",
        max_concurrent_calls: int = 3,
        warm_spares: int = 0,
        model_chunk_ms: int = 160,
        supports_turn_detection: bool = True,
        has_punctuation: bool = False,
        words_sequence: Sequence[WordTimestamp] | None = None,
        emit_eou_every_n_deltas: int = 0,
        emit_eob_on_first_delta: bool = False,
        emit_warning: dict[str, Any] | None = None,
        emit_recycled_at_sec: float | None = None,
        recycle_every_sec: float | None = None,
        repeat_words_for_long_audio: bool = True,
        worker_generations: int = 1,
        decode_delay_sec: float = 0.0,
        final_delay_sec: float = 0.0,
        draining: bool = False,
    ) -> None:
        self.default_model = default_model
        self.allowed_models = allowed_models or {
            "realtime_eou_120m-v1",
            "parakeet-ctc-0.6b",
            "parakeet-ctc-1.1b",
            "nemotron-3.5-asr-streaming-0.6b-q8_0",
        }
        self.batch_only_models = batch_only_models or {
            "whisper-large-v3-turbo",
            "whisper-large-v3-turbo-q8_0",
            "parakeet-tdt-0.6b-v3",
        }
        self.backend = backend
        self.version = version
        self.built_at = built_at
        self.parakeet_ref = parakeet_ref
        self.parakeet_abi_version = parakeet_abi_version
        self.ggml_cpu_features = ggml_cpu_features
        self.max_concurrent_calls = max_concurrent_calls
        self.warm_spares = warm_spares
        self.model_chunk_ms = model_chunk_ms
        self.supports_turn_detection = supports_turn_detection
        self.has_punctuation = has_punctuation
        self.words_sequence = list(
            words_sequence
            if words_sequence is not None
            else get_audio_fixture("short_speech").words
        )
        self.emit_eou_every_n_deltas = emit_eou_every_n_deltas
        self.emit_eob_on_first_delta = emit_eob_on_first_delta
        self.emit_warning = emit_warning
        self.emit_recycled_at_sec = emit_recycled_at_sec
        self.recycle_every_sec = recycle_every_sec
        self.repeat_words_for_long_audio = repeat_words_for_long_audio
        self.worker_generations = worker_generations
        self.decode_delay_sec = decode_delay_sec
        self.final_delay_sec = final_delay_sec
        self.draining = draining

        self.active_calls = 0
        self.peak_active_calls = 0
        self.total_calls = 0
        self.completed_calls = 0
        self.aborted_calls = 0
        self.received_Metadata: list[dict[str, str]] = []

    async def GetServerInfo(  # noqa: N802
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Any:
        if self.draining:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "draining")
            return asr_pb2.ServerInfoResponse()

        return asr_pb2.ServerInfoResponse(
            version=self.version,
            built_at=self.built_at,
            parakeet_ref=self.parakeet_ref,
            parakeet_abi_version=self.parakeet_abi_version,
            default_model=self.default_model,
            backend=self.backend,
            ggml_cpu_features=self.ggml_cpu_features,
            max_concurrent_calls=self.max_concurrent_calls,
            active_calls=self.active_calls,
            warm_spares=self.warm_spares,
        )

    async def Transcribe(  # noqa: N802
        self,
        request_iterator: AsyncIterator[Any],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Any]:
        md = {k.lower(): v for k, v in (context.invocation_metadata() or [])}
        self.received_Metadata.append(md)

        if self.draining:
            try:
                await asyncio.wait_for(request_iterator.__anext__(), timeout=0.05)
            except Exception:
                pass
            await context.abort(grpc.StatusCode.UNAVAILABLE, "draining")
            return

        if self.active_calls >= self.max_concurrent_calls:
            try:
                await asyncio.wait_for(request_iterator.__anext__(), timeout=0.05)
            except Exception:
                pass
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "at capacity, try again shortly",
            )
            return

        self.active_calls += 1
        self.total_calls += 1
        if self.active_calls > self.peak_active_calls:
            self.peak_active_calls = self.active_calls

        completed_cleanly = False
        try:
            try:
                first = await request_iterator.__anext__()
            except StopAsyncIteration:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "stream closed before any message",
                )
                return

            if first.WhichOneof("payload") != "config":
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "first message must be StreamConfig",
                )
                return

            config = first.config
            if config.encoding not in (
                asr_pb2.AUDIO_ENCODING_UNSPECIFIED,
                asr_pb2.AUDIO_ENCODING_LINEAR16,
            ):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unsupported encoding {asr_pb2.AudioEncoding.Name(config.encoding)}; "
                    "only AUDIO_ENCODING_LINEAR16 is accepted",
                )
                return

            if config.sample_rate_hz not in (0, 16000):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unsupported sample_rate_hz={config.sample_rate_hz}; only 16000 is accepted",
                )
                return

            chosen_model = config.model or self.default_model
            if chosen_model in self.batch_only_models:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"model {chosen_model!r} is batch-only (no incremental streaming support); "
                    "use POST /v1/audio/transcriptions instead of the Transcribe RPC",
                )
                return

            if chosen_model not in self.allowed_models:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unknown model {chosen_model!r}",
                )
                return

            is_nemotron = "nemotron" in chosen_model.lower()
            model_chunk_ms = 320 if is_nemotron else self.model_chunk_ms
            supports_eou = False if is_nemotron else self.supports_turn_detection

            yield asr_pb2.TranscriptionEvent(
                ready=asr_pb2.Ready(
                    model=chosen_model,
                    supports_turn_detection=supports_eou,
                    has_punctuation=self.has_punctuation,
                    model_chunk_ms=model_chunk_ms,
                    accepted_sample_rate_hz=16000,
                )
            )

            if self.emit_warning is not None:
                w_cfg = self.emit_warning
                yield asr_pb2.TranscriptionEvent(
                    warning=asr_pb2.Warning(
                        code=int(w_cfg.get("code", asr_pb2.WARNING_CODE_FALLING_BEHIND)),
                        message=str(w_cfg.get("message", "decode is behind real-time")),
                        at_sec=float(w_cfg.get("at_sec", 0.0)),
                        behind_sec=float(w_cfg.get("behind_sec", 1.2)),
                        dropped_sec=float(w_cfg.get("dropped_sec", 0.0)),
                        deadline_sec=float(w_cfg.get("deadline_sec", 0.0)),
                    )
                )

            bytes_per_model_chunk = int(16000 * (model_chunk_ms / 1000.0) * 2)
            pending_bytes = 0
            total_bytes = 0
            emitted_word_idx = 0
            delta_count = 0
            recycled_emitted = False
            recycled_count = 0
            completed_cleanly = False

            # Reserve the last word for Final (matching real parakeet stream_finalize behavior)
            stream_words = list(self.words_sequence[:-1]) if len(self.words_sequence) > 1 else []
            final_words = list(self.words_sequence[-1:]) if self.words_sequence else []
            base_cycle_sec = (
                max(4.5, round(float(self.words_sequence[-1].end_sec) + 1.0, 2))
                if self.words_sequence
                else 4.5
            )
            next_cycle_offset_sec = base_cycle_sec
            next_periodic_recycle_sec = (
                float(self.recycle_every_sec)
                if (self.recycle_every_sec is not None and self.recycle_every_sec > 0.0)
                else None
            )

            async for req in request_iterator:
                which = req.WhichOneof("payload")
                if which == "config":
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "config sent twice",
                    )
                    return
                if which != "audio":
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "empty request payload",
                    )
                    return

                chunk_len = len(req.audio)
                pending_bytes += chunk_len
                total_bytes += chunk_len

                while pending_bytes >= bytes_per_model_chunk:
                    pending_bytes -= bytes_per_model_chunk
                    audio_offset_sec = (total_bytes - pending_bytes) / 32000.0

                    if self.decode_delay_sec > 0.0:
                        await asyncio.sleep(self.decode_delay_sec)

                    if (
                        self.repeat_words_for_long_audio
                        and self.words_sequence
                        and audio_offset_sec >= next_cycle_offset_sec
                    ):
                        while audio_offset_sec >= next_cycle_offset_sec:
                            stream_words.extend(final_words)
                            shifted_stream = [
                                WordTimestamp(
                                    text=w.text,
                                    start_sec=round(w.start_sec + next_cycle_offset_sec, 3),
                                    end_sec=round(w.end_sec + next_cycle_offset_sec, 3),
                                )
                                for w in (
                                    self.words_sequence[:-1]
                                    if len(self.words_sequence) > 1
                                    else []
                                )
                            ]
                            stream_words.extend(shifted_stream)
                            final_words = [
                                WordTimestamp(
                                    text=w.text,
                                    start_sec=round(w.start_sec + next_cycle_offset_sec, 3),
                                    end_sec=round(w.end_sec + next_cycle_offset_sec, 3),
                                )
                                for w in self.words_sequence[-1:]
                            ]
                            next_cycle_offset_sec = round(next_cycle_offset_sec + base_cycle_sec, 3)

                    if (
                        next_periodic_recycle_sec is not None
                        and audio_offset_sec >= next_periodic_recycle_sec
                    ):
                        recycled_emitted = True
                        recycled_count += 1
                        next_periodic_recycle_sec = round(
                            next_periodic_recycle_sec + float(self.recycle_every_sec or 0.0), 3
                        )
                        yield asr_pb2.TranscriptionEvent(
                            recycled=asr_pb2.Recycled(
                                reason=asr_pb2.RECYCLE_REASON_RSS_THRESHOLD,
                                gap_sec=0.0,
                                at_audio_sec=audio_offset_sec,
                                warm=True,
                            )
                        )
                    elif (
                        self.emit_recycled_at_sec is not None
                        and not recycled_emycled_check(recycled_emitted)
                        and audio_offset_sec >= self.emit_recycled_at_sec
                    ):
                        recycled_emitted = True
                        recycled_count += 1
                        yield asr_pb2.TranscriptionEvent(
                            recycled=asr_pb2.Recycled(
                                reason=asr_pb2.RECYCLE_REASON_RSS_THRESHOLD,
                                gap_sec=0.0,
                                at_audio_sec=audio_offset_sec,
                                warm=True,
                            )
                        )

                    # Emit any words whose end_sec <= current audio_offset_sec
                    due_words: list[WordTimestamp] = []
                    while (
                        emitted_word_idx < len(stream_words)
                        and stream_words[emitted_word_idx].end_sec <= audio_offset_sec + 1e-6
                    ):
                        due_words.append(stream_words[emitted_word_idx])
                        emitted_word_idx += 1

                    if due_words:
                        delta_count += 1
                        prefix = " " if delta_count > 1 else ""
                        delta_text = prefix + " ".join(w.text for w in due_words)
                        pb_words = (
                            [
                                asr_pb2.Word(
                                    text=w.text,
                                    start_sec=w.start_sec,
                                    end_sec=w.end_sec,
                                    confidence=0.98,
                                )
                                for w in due_words
                            ]
                            if config.enable_word_timestamps
                            else []
                        )
                        yield asr_pb2.TranscriptionEvent(
                            delta=asr_pb2.TranscriptDelta(
                                text=delta_text,
                                words=pb_words,
                                audio_offset_sec=audio_offset_sec,
                            )
                        )
                        if self.emit_eob_on_first_delta and delta_count == 1:
                            yield asr_pb2.TranscriptionEvent(
                                eob=asr_pb2.EndOfBoundary(at_sec=audio_offset_sec)
                            )
                        if (
                            self.emit_eou_every_n_deltas > 0
                            and supports_eou
                            and delta_count % self.emit_eou_every_n_deltas == 0
                        ):
                            yield asr_pb2.TranscriptionEvent(
                                eou=asr_pb2.EndOfUtterance(at_sec=audio_offset_sec)
                            )

            # Half-close -> finalize()
            if self.final_delay_sec > 0.0:
                await asyncio.sleep(self.final_delay_sec)

            total_audio_sec = total_bytes / 32000.0
            remaining_words = list(stream_words[emitted_word_idx:]) + list(final_words)
            if total_bytes == 0:
                remaining_words = []

            if remaining_words:
                prefix = " " if delta_count > 0 else ""
                final_text = prefix + " ".join(w.text for w in remaining_words)
                pb_final_words = (
                    [
                        asr_pb2.Word(
                            text=w.text,
                            start_sec=w.start_sec,
                            end_sec=w.end_sec,
                            confidence=0.99,
                        )
                        for w in remaining_words
                    ]
                    if config.enable_word_timestamps
                    else []
                )
            else:
                final_text = ""
                pb_final_words = []

            generations = (1 + recycled_count) if recycled_count > 0 else self.worker_generations
            yield asr_pb2.TranscriptionEvent(
                final=asr_pb2.Final(
                    text=final_text,
                    words=pb_final_words,
                    total_audio_sec=total_audio_sec,
                    worker_generations=generations,
                )
            )
            completed_cleanly = True
        finally:
            if completed_cleanly:
                self.completed_calls += 1
            else:
                self.aborted_calls += 1
            self.active_calls = max(0, self.active_calls - 1)


def recycled_emycled_check(already_emitted: bool) -> bool:
    return already_emitted


class MockLiveSTTGRPCServer:
    """Async and sync context manager that runs ``MockStreamingASRServicer`` on an ephemeral loopback port."""

    def __init__(self, servicer: MockStreamingASRServicer | None = None, **kwargs: Any) -> None:
        self.servicer = servicer or MockStreamingASRServicer(**kwargs)
        self._server: grpc.aio.Server | None = None
        self.port: int = 0
        self._bg_loop: asyncio.AbstractEventLoop | None = None
        self._bg_thread: Any = None
        self._bg_stop_event: asyncio.Event | None = None

    @property
    def target(self) -> str:
        return f"127.0.0.1:{self.port}"

    async def __aenter__(self) -> "MockLiveSTTGRPCServer":
        self._server = grpc.aio.server()
        asr_pb2_grpc.add_StreamingASRServicer_to_server(self.servicer, self._server)
        self.port = self._server.add_insecure_port("127.0.0.1:0")
        await self._server.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._server is not None:
            await self._server.stop(grace=0.5)
            await self._server.wait_for_termination()
            self._server = None

    def __enter__(self) -> "MockLiveSTTGRPCServer":
        import threading

        ready = threading.Event()

        def _thread_runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_loop_for_thread = loop
            asyncio.set_event_loop(loop)
            self._bg_loop = loop

            async def _serve() -> None:
                self._bg_stop_event = asyncio.Event()
                await self.__aenter__()
                ready.set()
                await self._bg_stop_event.wait()
                await self.__aexit__(None, None, None)

            loop.run_until_complete(_serve())
            loop.close()

        self._bg_thread = threading.Thread(target=_thread_runner, daemon=True)
        self._bg_thread.start()
        ready.wait(timeout=10.0)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._bg_loop is not None and self._bg_stop_event is not None:
            self._bg_loop.call_soon_threadsafe(self._bg_stop_event.set)
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=5.0)
            self._bg_thread = None
            self._bg_loop = None
            self._bg_stop_event = None


# ===========================================================================
# CLI Entrypoint for ad-hoc streaming harness execution
# ===========================================================================


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="live-stt gRPC streaming harness: feed audio in real time (or faster) and record partial/final events with timestamps."
    )
    parser.add_argument(
        "--target",
        default="127.0.0.1:50051",
        help="gRPC target host:port or URL (default: 127.0.0.1:50051)",
    )
    parser.add_argument(
        "--audio",
        default="short_speech",
        help="Path to 16kHz mono 16-bit WAV file or built-in fixture ID (short_speech, medium_speech, two_speaker_dialog)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="ASR model name (empty uses server default)",
    )
    parser.add_argument(
        "--language",
        default="",
        help="Language code (e.g. en-US)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=DEFAULT_CHUNK_MS,
        help="Audio frame size in milliseconds (default: 20.0)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Feed speed multiplier: 1.0 = real-time, 2.0 = 2x faster, 0 = unpaced burst (default: 1.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full JSON session trace to stdout",
    )
    return parser


async def _cli_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    harness = LiveSTTStreamingHarness(
        args.target,
        default_chunk_ms=args.chunk_ms,
        default_speed=args.speed if args.speed > 0 else None,
        default_realtime=args.speed > 0,
    )
    res = await harness.run_session(
        args.audio,
        model=args.model,
        language=args.language,
    )
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(
            f"Session {res.call_id} -> status={res.grpc_status_code} ok={res.ok} "
            f"model={res.model!r} audio={res.audio_duration_sec:.2f}s"
        )
        for evt in res.events:
            print(
                f"  [{evt.elapsed_ms:7.2f}ms | {evt.timestamp_iso}] "
                f"{evt.event_type.upper():8s} text={evt.text!r} offset={evt.audio_offset_sec}"
            )
        print(
            f"Metrics: TTFP={res.ttfp_ms}ms, FinalLatency={res.final_latency_ms}ms, "
            f"Total={res.total_latency_ms:.2f}ms, RTF={res.rtf:.4f}x"
        )
        print(f"Transcript: {res.text!r}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_cli_main()))
