"""gRPC StreamingASR.Transcribe test suite and stability evaluators (TES-31).

Covers:
- Time to First Partial (TTFP: from stream start and from first audio chunk)
- Finalization latency (client half-close EOF to terminal Final event)
- Real-Time Factor (RTF) and inter-partial cadence across real-time and fast-paced streams
- Concurrent gRPC Transcribe sessions (within capacity and over-budget RESOURCE_EXHAUSTED + slot recovery)
- Reconnect after mid-stream client abort/cancellation, clean half-close shutdown, channel reuse,
  server drain (UNAVAILABLE / WARNING_CODE_SERVER_DRAINING), and INVALID_ARGUMENT protocol error contract
- Opt-in long-call stability suite (grpc_long_call): sustained multi-utterance stream, continuous
  TranscriptDelta emission, word timestamp rebasing across Recycled worker rotations, and latency/RTF drift
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import statistics
import time
from typing import Any

from backend.app.testers.base import (
    TestContext,
    TesterCancelledError,
    TesterResult,
)
from backend.app.testers.live_stt.client import (
    KNOWN_BATCH_ONLY_PREFIXES,
    ModelInfo,
    infer_model_info,
    is_batch_only_grpc_model,
)
from backend.app.testers.live_stt.fixtures import (
    AudioFixture,
    WordTimestamp,
    get_audio_fixture,
    make_pcm_wav_bytes,
)
from backend.app.testers.live_stt.grpc_client import (
    ConcurrentStreamingResult,
    DEFAULT_CHUNK_MS,
    DEFAULT_SAMPLE_RATE_HZ,
    LiveSTTGRPCClient,
    LiveSTTStreamingHarness,
    ServerInfoResult,
    StreamingSessionResult,
    _run_coro_sync,
    extract_pcm16_audio,
)
from backend.app.testers.live_stt.accuracy import (
    DEFAULT_SPEECH_WER_THRESHOLD,
    speech_accuracy_check,
)
from backend.app.testers.live_stt.metrics import (
    compute_percentiles,
    compute_rtf,
)
from backend.app.testers.live_stt.pb import asr_pb2


DEFAULT_GRPC_STREAMING_MODEL = "realtime_eou_120m-v1"


class _IntentionalMidStreamAbort(Exception):
    """Internal sentinel raised by cancel_check to simulate a mid-stream client disconnect."""


@dataclass
class LongCallStabilityReport:
    """Detailed evaluation of a long-duration StreamingASR.Transcribe session."""

    ok: bool
    passed: bool
    grpc_status_code: str
    model: str
    audio_duration_sec: float
    reported_audio_sec: float
    duration_drift_sec: float
    ttfp_ms: float | None
    ttfp_from_first_audio_ms: float | None
    final_latency_ms: float | None
    ready_latency_ms: float | None
    total_latency_ms: float
    rtf: float
    partial_count: int
    word_count: int
    worker_generations: int
    recycled_count: int
    warm_recycle_count: int
    cold_recycle_count: int
    max_recycle_gap_sec: float
    warning_count: int
    dropped_audio_sec: float
    offsets_monotonic: bool
    word_timestamps_monotonic: bool
    recycled_rebased_properly: bool
    first_half_inter_partial_mean_ms: float
    second_half_inter_partial_mean_ms: float
    max_inter_partial_latency_ms: float
    inter_partial_drift_ratio: float
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "passed": self.passed,
            "grpc_status_code": self.grpc_status_code,
            "model": self.model,
            "audio_duration_sec": round(self.audio_duration_sec, 3),
            "reported_audio_sec": round(self.reported_audio_sec, 3),
            "duration_drift_sec": round(self.duration_drift_sec, 4),
            "ttfp_ms": round(self.ttfp_ms, 2) if self.ttfp_ms is not None else None,
            "ttfp_from_first_audio_ms": (
                round(self.ttfp_from_first_audio_ms, 2)
                if self.ttfp_from_first_audio_ms is not None
                else None
            ),
            "final_latency_ms": (
                round(self.final_latency_ms, 2)
                if self.final_latency_ms is not None
                else None
            ),
            "ready_latency_ms": (
                round(self.ready_latency_ms, 2)
                if self.ready_latency_ms is not None
                else None
            ),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "rtf": round(self.rtf, 4),
            "partial_count": self.partial_count,
            "word_count": self.word_count,
            "worker_generations": self.worker_generations,
            "recycled_count": self.recycled_count,
            "warm_recycle_count": self.warm_recycle_count,
            "cold_recycle_count": self.cold_recycle_count,
            "max_recycle_gap_sec": round(self.max_recycle_gap_sec, 4),
            "warning_count": self.warning_count,
            "dropped_audio_sec": round(self.dropped_audio_sec, 4),
            "offsets_monotonic": self.offsets_monotonic,
            "word_timestamps_monotonic": self.word_timestamps_monotonic,
            "recycled_rebased_properly": self.recycled_rebased_properly,
            "first_half_inter_partial_mean_ms": round(self.first_half_inter_partial_mean_ms, 2),
            "second_half_inter_partial_mean_ms": round(self.second_half_inter_partial_mean_ms, 2),
            "max_inter_partial_latency_ms": round(self.max_inter_partial_latency_ms, 2),
            "inter_partial_drift_ratio": round(self.inter_partial_drift_ratio, 4),
            "checks": self.checks,
            "errors": self.errors,
            "summary_message": self.summary_message,
        }


@dataclass
class ReconnectShutdownResult:
    """Telemetry and contract checks for mid-stream client abort, slot release, and reconnect."""

    passed: bool
    abort_duration_ms: float
    aborted_chunks_sent: int
    active_calls_after_abort: int
    reconnect_session: StreamingSessionResult
    active_calls_after_reconnect: int
    total_wall_ms: float
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "abort_duration_ms": round(self.abort_duration_ms, 2),
            "aborted_chunks_sent": self.aborted_chunks_sent,
            "active_calls_after_abort": self.active_calls_after_abort,
            "reconnect_ok": self.reconnect_session.ok,
            "reconnect_grpc_status": self.reconnect_session.grpc_status_code,
            "reconnect_ttfp_ms": self.reconnect_session.ttfp_ms,
            "reconnect_final_latency_ms": self.reconnect_session.final_latency_ms,
            "reconnect_latency_ms": round(self.reconnect_session.total_latency_ms, 2),
            "reconnect_text": self.reconnect_session.text,
            "active_calls_after_reconnect": self.active_calls_after_reconnect,
            "total_wall_ms": round(self.total_wall_ms, 2),
            "checks": self.checks,
            "errors": self.errors,
            "summary_message": self.summary_message,
        }


def compute_grpc_streaming_summary(
    sessions: Sequence[StreamingSessionResult],
) -> dict[str, float]:
    """Compute aggregate statistics across gRPC StreamingASR.Transcribe sessions.

    Includes percentiles (p50, p95, p99), mean, min, and max for:
    - Time to First Partial (`ttfp_ms` and `ttfp_from_first_audio_ms`)
    - Finalization latency (`final_latency_ms` from half-close to Final event)
    - Ready handshake latency (`ready_latency_ms`)
    - Inter-partial latency (`inter_partial_latencies_ms`)
    - Total stream latency (`total_latency_ms`)
    - Real-Time Factor (`rtf`)
    """
    ttfps = [float(s.ttfp_ms) for s in sessions if s.ttfp_ms is not None]
    ttfps_audio = [
        float(s.ttfp_from_first_audio_ms)
        for s in sessions
        if s.ttfp_from_first_audio_ms is not None
    ]
    final_lats = [
        float(s.final_latency_ms) for s in sessions if s.final_latency_ms is not None
    ]
    ready_lats = [
        float(s.ready_latency_ms) for s in sessions if s.ready_latency_ms is not None
    ]
    inter_partials = [
        float(gap) for s in sessions for gap in s.inter_partial_latencies_ms
    ]
    total_lats = [float(s.total_latency_ms) for s in sessions]
    rtfs = [float(s.rtf) for s in sessions if s.audio_duration_sec > 0]

    ttfp_pct = compute_percentiles(ttfps, (50, 95, 99))
    ttfp_audio_pct = compute_percentiles(ttfps_audio, (50, 95, 99))
    final_pct = compute_percentiles(final_lats, (50, 95, 99))
    ready_pct = compute_percentiles(ready_lats, (50, 95, 99))
    inter_pct = compute_percentiles(inter_partials, (50, 95, 99))
    lat_pct = compute_percentiles(total_lats, (50, 95, 99))

    import numpy as np

    rtf_pct = (
        {int(p): round(float(np.percentile(rtfs, p)), 4) for p in (50, 95, 99)}
        if rtfs
        else {50: 0.0, 95: 0.0, 99: 0.0}
    )

    def _mean(vals: list[float], ndigits: int = 2) -> float:
        return round(float(statistics.mean(vals)), ndigits) if vals else 0.0

    def _min(vals: list[float], ndigits: int = 2) -> float:
        return round(float(min(vals)), ndigits) if vals else 0.0

    def _max(vals: list[float], ndigits: int = 2) -> float:
        return round(float(max(vals)), ndigits) if vals else 0.0

    return {
        "ttfp_mean_ms": _mean(ttfps),
        "ttfp_min_ms": _min(ttfps),
        "ttfp_max_ms": _max(ttfps),
        "ttfp_p50_ms": ttfp_pct[50],
        "ttfp_p95_ms": ttfp_pct[95],
        "ttfp_p99_ms": ttfp_pct[99],
        "ttfp_from_first_audio_mean_ms": _mean(ttfps_audio),
        "ttfp_from_first_audio_p50_ms": ttfp_audio_pct[50],
        "ttfp_from_first_audio_p95_ms": ttfp_audio_pct[95],
        "ttfp_from_first_audio_p99_ms": ttfp_audio_pct[99],
        "final_latency_mean_ms": _mean(final_lats),
        "final_latency_min_ms": _min(final_lats),
        "final_latency_max_ms": _max(final_lats),
        "final_latency_p50_ms": final_pct[50],
        "final_latency_p95_ms": final_pct[95],
        "final_latency_p99_ms": final_pct[99],
        "ready_latency_mean_ms": _mean(ready_lats),
        "ready_latency_min_ms": _min(ready_lats),
        "ready_latency_max_ms": _max(ready_lats),
        "ready_latency_p50_ms": ready_pct[50],
        "ready_latency_p95_ms": ready_pct[95],
        "ready_latency_p99_ms": ready_pct[99],
        "inter_partial_latency_mean_ms": _mean(inter_partials),
        "inter_partial_latency_p50_ms": inter_pct[50],
        "inter_partial_latency_p95_ms": inter_pct[95],
        "inter_partial_latency_p99_ms": inter_pct[99],
        "latency_mean_ms": _mean(total_lats),
        "latency_min_ms": _min(total_lats),
        "latency_max_ms": _max(total_lats),
        "latency_p50_ms": lat_pct[50],
        "latency_p95_ms": lat_pct[95],
        "latency_p99_ms": lat_pct[99],
        "rtf_mean": _mean(rtfs, 4),
        "rtf_min": _min(rtfs, 4),
        "rtf_max": _max(rtfs, 4),
        "rtf_p50": rtf_pct[50],
        "rtf_p95": rtf_pct[95],
        "rtf_p99": rtf_pct[99],
        "sessions_total": float(len(sessions)),
        "sessions_ok": float(sum(1 for s in sessions if s.ok)),
        "total_partials": float(sum(s.partial_count for s in sessions)),
        "total_words": float(sum(s.word_count for s in sessions)),
        "total_recycled_events": float(sum(len(s.recycled_events) for s in sessions)),
        "total_warnings": float(sum(len(s.warnings) for s in sessions)),
    }


def select_grpc_streaming_models(
    asr_models: Sequence[ModelInfo],
    server_info: ServerInfoResult | None = None,
    config: dict[str, Any] | None = None,
) -> list[ModelInfo]:
    """Select streaming-capable ASR models for gRPC Transcribe evaluation."""
    cfg = config or {}
    explicit: list[str] = []
    if isinstance(cfg.get("grpc_models"), list) and cfg["grpc_models"]:
        explicit = [str(m).strip() for m in cfg["grpc_models"] if str(m).strip()]
    elif cfg.get("grpc_model"):
        explicit = [str(cfg["grpc_model"]).strip()]

    candidates = (
        [infer_model_info({"id": m, "type": "asr", "supports_streaming": True}) for m in explicit]
        if explicit
        else list(asr_models)
    )

    streaming_models = [
        m
        for m in candidates
        if m.grpc_streaming
    ]

    default_from_server = (
        server_info.default_model.strip()
        if (server_info is not None and server_info.ok and server_info.default_model.strip())
        else DEFAULT_GRPC_STREAMING_MODEL
    )

    if not streaming_models:
        return [infer_model_info({"id": default_from_server, "type": "asr", "supports_streaming": True})]

    return streaming_models


def build_long_call_audio_fixture(
    duration_sec: float = 30.0,
    base_fixture_id: str = "two_speaker_dialog",
    base_fixture: AudioFixture | None = None,
) -> AudioFixture:
    """Build a long-call 16kHz mono 16-bit PCM WAV fixture by looping a base speech fixture.

    ``base_fixture`` (e.g. a real NOTSOFAR-1 excerpt) takes precedence over the built-in
    ``base_fixture_id``. Rebases word-level timestamps across each loop cycle so that ``words`` and
    ``reference_text`` span the full ``[0, duration_sec]`` timeline.
    """
    target_sec = max(1.0, float(duration_sec))
    base_fix = base_fixture if base_fixture is not None else get_audio_fixture(base_fixture_id)
    extracted = extract_pcm16_audio(base_fix)
    sample_rate = extracted.sample_rate_hz
    bytes_per_sec = sample_rate * 2  # 16kHz mono 16-bit = 32,000 bytes/s
    target_bytes = int(round(target_sec * bytes_per_sec))
    if target_bytes % 2 != 0:
        target_bytes += 1

    base_pcm = extracted.pcm_bytes
    base_dur = max(0.1, extracted.duration_sec)
    if not base_pcm:
        return make_pcm_wav_bytes_fixture(target_sec)

    chunks: list[bytes] = []
    rebased_words: list[WordTimestamp] = []
    accum_bytes = 0
    cycle_idx = 0

    while accum_bytes < target_bytes:
        needed = target_bytes - accum_bytes
        take = min(len(base_pcm), needed)
        if take % 2 != 0:
            take -= 1
        if take <= 0:
            break
        chunks.append(base_pcm[:take])
        offset_sec = round(cycle_idx * base_dur, 4)
        slice_dur_sec = take / float(bytes_per_sec)

        for w in base_fix.words:
            if w.end_sec <= slice_dur_sec + 1e-6:
                rebased_words.append(
                    WordTimestamp(
                        text=w.text,
                        start_sec=round(offset_sec + w.start_sec, 3),
                        end_sec=round(offset_sec + w.end_sec, 3),
                    )
                )

        accum_bytes += take
        cycle_idx += 1

    full_pcm = b"".join(chunks)
    actual_duration_sec = round(len(full_pcm) / float(bytes_per_sec), 3)
    wav_bytes = _wrap_pcm16_in_wav(full_pcm, sample_rate=sample_rate)
    ref_text = " ".join(w.text for w in rebased_words)

    return AudioFixture(
        id=f"long_call_{int(round(actual_duration_sec))}s"
        + (f":{base_fix.id}" if base_fixture is not None else ""),
        name=f"Long Call ({actual_duration_sec:.1f}s)",
        description=(
            f"{actual_duration_sec:.1f}s 16kHz mono 16-bit PCM long-call fixture looping '{base_fix.id}'"
        ),
        wav_bytes=wav_bytes,
        duration_sec=actual_duration_sec,
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        words=tuple(rebased_words),
        reference_text=ref_text,
        expected_speakers=base_fix.expected_speakers,
    )


def _wrap_pcm16_in_wav(pcm_bytes: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE_HZ) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def make_pcm_wav_bytes_fixture(duration_sec: float) -> AudioFixture:
    wav_bytes = make_pcm_wav_bytes(duration_sec=duration_sec, sample_rate=DEFAULT_SAMPLE_RATE_HZ)
    return AudioFixture(
        id=f"long_call_{int(round(duration_sec))}s",
        name=f"Long Call ({duration_sec:.1f}s)",
        description=f"Synthesized {duration_sec:.1f}s WAV fixture",
        wav_bytes=wav_bytes,
        duration_sec=duration_sec,
        sample_rate=DEFAULT_SAMPLE_RATE_HZ,
        channels=1,
        sample_width=2,
        words=(),
        reference_text="",
        expected_speakers=1,
    )


def evaluate_streaming_session_contract(
    session: StreamingSessionResult,
    *,
    expected_duration_sec: float | None = None,
    require_partials: bool = True,
    require_word_timestamps: bool = True,
) -> tuple[bool, list[dict[str, Any]], list[str], str]:
    """Validate a single StreamingASR.Transcribe session's events, timestamps, and metrics."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def _add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    _add(
        "grpc_status_ok",
        session.ok and session.grpc_status_code == "OK",
        f"status={session.grpc_status_code}, ok={session.ok}, error={session.error_message!r}",
    )

    ready_evt = session.ready_event
    _add(
        "ready_event_first",
        ready_evt is not None and ready_evt.seq == 0 and session.accepted_sample_rate_hz == 16000,
        (
            f"seq={ready_evt.seq if ready_evt else None}, "
            f"model={session.resolved_model!r}, "
            f"sample_rate={session.accepted_sample_rate_hz}, "
            f"model_chunk_ms={session.model_chunk_ms}"
        ),
    )

    if require_partials:
        _add(
            "partial_events_emitted",
            session.partial_count >= 1 or bool(session.text.strip()),
            f"partial_count={session.partial_count}, text_len={len(session.text)}",
        )

    _add(
        "ttfp_recorded",
        session.ttfp_ms is not None and session.ttfp_ms >= 0.0,
        f"ttfp_ms={session.ttfp_ms}, ttfp_from_first_audio_ms={session.ttfp_from_first_audio_ms}",
    )

    final_evt = session.final_event
    _add(
        "final_event_last_with_latency",
        (
            final_evt is not None
            and session.events[-1].kind == "final"
            and session.final_latency_ms is not None
            and session.final_latency_ms >= 0.0
        ),
        (
            f"has_final={final_evt is not None}, "
            f"final_latency_ms={session.final_latency_ms}, "
            f"worker_generations={session.worker_generations}"
        ),
    )

    target_dur = expected_duration_sec if expected_duration_sec is not None else session.audio_duration_sec
    rep_dur = session.total_audio_sec_reported
    dur_ok = rep_dur is not None and abs(rep_dur - target_dur) <= max(0.25, target_dur * 0.05)
    _add(
        "reported_audio_duration_matches",
        dur_ok,
        f"reported={rep_dur}s vs expected={target_dur:.3f}s",
    )

    if require_word_timestamps:
        words_ok = len(session.words) > 0 and all(
            w.end_sec >= w.start_sec >= 0.0 for w in session.words
        )
        monotonic_words = True
        for i in range(1, len(session.words)):
            if session.words[i].start_sec + 1e-4 < session.words[i - 1].start_sec:
                monotonic_words = False
                break
        _add(
            "word_timestamps_valid_and_monotonic",
            words_ok and monotonic_words,
            f"words={len(session.words)}, monotonic={monotonic_words}",
        )

    passed = len(errors) == 0
    summary = (
        f"gRPC Transcribe OK: TTFP={session.ttfp_ms}ms, "
        f"FinalLat={session.final_latency_ms}ms, RTF={session.rtf:.4f}x, "
        f"Partials={session.partial_count}, Words={session.word_count}"
        if passed
        else "; ".join(errors)
    )
    return passed, checks, errors, summary


def evaluate_long_call_stability(
    session: StreamingSessionResult,
    *,
    expected_duration_sec: float | None = None,
    max_rtf: float = 2.0,
    max_drift_ratio: float = 4.0,
) -> LongCallStabilityReport:
    """Evaluate long-call stability over a sustained StreamingASR.Transcribe stream."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def _add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    target_dur = (
        float(expected_duration_sec)
        if expected_duration_sec is not None
        else float(session.audio_duration_sec)
    )
    reported_dur = float(session.total_audio_sec_reported or 0.0)
    duration_drift_sec = abs(reported_dur - target_dur)

    _add(
        "long_call_completed_ok",
        session.ok and session.grpc_status_code == "OK" and session.final_event is not None,
        f"status={session.grpc_status_code}, ok={session.ok}, error={session.error_message!r}",
    )

    tol_sec = max(0.5, target_dur * 0.05)
    _add(
        "audio_duration_drift_within_bounds",
        session.total_audio_sec_reported is not None and duration_drift_sec <= tol_sec,
        f"reported={reported_dur:.3f}s, expected={target_dur:.3f}s, drift={duration_drift_sec:.4f}s (tol={tol_sec:.2f}s)",
    )

    partials = session.partials
    _add(
        "continuous_partials_emitted",
        len(partials) >= 2 and bool(session.text.strip()),
        f"partial_count={len(partials)}, word_count={session.word_count}",
    )

    offsets = [
        float(p.audio_offset_sec)
        for p in partials
        if p.audio_offset_sec is not None
    ]
    offsets_monotonic = all(
        offsets[i] + 1e-6 >= offsets[i - 1] for i in range(1, len(offsets))
    )
    _add(
        "partial_audio_offsets_monotonic",
        offsets_monotonic and len(offsets) > 0,
        f"offsets_count={len(offsets)}, monotonic={offsets_monotonic}",
    )

    word_timestamps_monotonic = True
    for i, w in enumerate(session.words):
        if w.end_sec < w.start_sec or w.start_sec < 0.0:
            word_timestamps_monotonic = False
            break
        if i > 0 and w.start_sec + 1e-4 < session.words[i - 1].start_sec:
            word_timestamps_monotonic = False
            break

    _add(
        "word_timestamps_monotonic_across_call",
        word_timestamps_monotonic and len(session.words) > 0,
        f"word_count={len(session.words)}, monotonic={word_timestamps_monotonic}",
    )

    recycled_evts = session.recycled_events
    recycled_count = len(recycled_evts)
    warm_recycle_count = sum(
        1 for e in recycled_evts if bool(e.detail.get("warm", False))
    )
    cold_recycle_count = recycled_count - warm_recycle_count
    max_recycle_gap_sec = max(
        (float(e.detail.get("gap_sec", 0.0) or 0.0) for e in recycled_evts),
        default=0.0,
    )

    generations_ok = session.worker_generations >= (1 + recycled_count if recycled_count > 0 else 1)
    recycled_rebased_properly = word_timestamps_monotonic and generations_ok
    _add(
        "worker_generations_and_recycle_rebasing",
        recycled_rebased_properly,
        (
            f"worker_generations={session.worker_generations}, "
            f"recycled_count={recycled_count} (warm={warm_recycle_count}, cold={cold_recycle_count}), "
            f"max_gap_sec={max_recycle_gap_sec:.3f}s"
        ),
    )

    dropped_audio_sec = sum(
        float(w.detail.get("dropped_sec", 0.0) or 0.0) for w in session.warnings
    )
    _add(
        "zero_dropped_audio",
        dropped_audio_sec == 0.0,
        f"dropped_audio_sec={dropped_audio_sec:.4f}s, warning_count={len(session.warnings)}",
    )

    gaps = session.inter_partial_latencies_ms
    if len(gaps) >= 2:
        mid = len(gaps) // 2
        first_half_mean = float(statistics.mean(gaps[:mid]))
        second_half_mean = float(statistics.mean(gaps[mid:]))
        baseline_den = max(first_half_mean, 5.0)
        drift_ratio = round(second_half_mean / baseline_den, 4)
    elif len(gaps) == 1:
        first_half_mean = float(gaps[0])
        second_half_mean = float(gaps[0])
        drift_ratio = 1.0
    else:
        first_half_mean = 0.0
        second_half_mean = 0.0
        drift_ratio = 1.0

    max_inter_partial = max(gaps, default=0.0)
    _add(
        "inter_partial_latency_drift_bounded",
        drift_ratio <= max_drift_ratio,
        (
            f"first_half_mean={first_half_mean:.2f}ms, second_half_mean={second_half_mean:.2f}ms, "
            f"drift_ratio={drift_ratio:.2f}x <= max={max_drift_ratio:.2f}x"
        ),
    )

    _add(
        "rtf_within_stability_budget",
        session.rtf <= max_rtf,
        f"rtf={session.rtf:.4f}x <= max_rtf={max_rtf:.2f}x",
    )

    passed = len(errors) == 0
    summary = (
        f"Long call ({target_dur:.1f}s) stable: TTFP={session.ttfp_ms}ms, "
        f"FinalLat={session.final_latency_ms}ms, RTF={session.rtf:.4f}x, "
        f"Partials={session.partial_count}, Words={session.word_count}, "
        f"Generations={session.worker_generations}, Recycled={recycled_count}, "
        f"Drift={drift_ratio:.2f}x"
        if passed
        else "; ".join(errors)
    )

    return LongCallStabilityReport(
        ok=session.ok,
        passed=passed,
        grpc_status_code=session.grpc_status_code,
        model=session.model,
        audio_duration_sec=target_dur,
        reported_audio_sec=reported_dur,
        duration_drift_sec=duration_drift_sec,
        ttfp_ms=session.ttfp_ms,
        ttfp_from_first_audio_ms=session.ttfp_from_first_audio_ms,
        final_latency_ms=session.final_latency_ms,
        ready_latency_ms=session.ready_latency_ms,
        total_latency_ms=session.total_latency_ms,
        rtf=session.rtf,
        partial_count=session.partial_count,
        word_count=session.word_count,
        worker_generations=session.worker_generations,
        recycled_count=recycled_count,
        warm_recycle_count=warm_recycle_count,
        cold_recycle_count=cold_recycle_count,
        max_recycle_gap_sec=max_recycle_gap_sec,
        warning_count=len(session.warnings),
        dropped_audio_sec=dropped_audio_sec,
        offsets_monotonic=offsets_monotonic,
        word_timestamps_monotonic=word_timestamps_monotonic,
        recycled_rebased_properly=recycled_rebased_properly,
        first_half_inter_partial_mean_ms=first_half_mean,
        second_half_inter_partial_mean_ms=second_half_mean,
        max_inter_partial_latency_ms=max_inter_partial,
        inter_partial_drift_ratio=drift_ratio,
        checks=checks,
        errors=errors,
        summary_message=summary,
    )


async def _wait_for_zero_active_calls(
    harness: LiveSTTStreamingHarness,
    *,
    max_wait_sec: float = 0.5,
) -> ServerInfoResult:
    """Poll GetServerInfo briefly until active_calls reaches 0 after an abort/burst."""
    deadline = time.perf_counter() + max_wait_sec
    last_info = await harness.get_server_info()
    while last_info.ok and last_info.active_calls > 0 and time.perf_counter() < deadline:
        await asyncio.sleep(0.02)
        last_info = await harness.get_server_info()
    return last_info


async def run_reconnect_after_abort_check_async(
    harness: LiveSTTStreamingHarness,
    *,
    model: str = "",
    language: str = "en",
    abort_after_chunks: int = 4,
    audio: AudioFixture | bytes | str = "medium_speech",
    chunk_ms: float = DEFAULT_CHUNK_MS,
    speed: float | None = 2.0,
    realtime: bool = True,
    cancel_check: Callable[[], None] | None = None,
) -> ReconnectShutdownResult:
    """Abort a gRPC Transcribe session mid-stream, verify slot release, and immediately reconnect."""
    t0 = time.perf_counter()
    chunks_seen = 0
    aborted_cleanly = False

    def _abort_hook() -> None:
        nonlocal chunks_seen
        if cancel_check is not None:
            cancel_check()
        chunks_seen += 1
        if chunks_seen > abort_after_chunks:
            raise _IntentionalMidStreamAbort("Simulated client mid-stream disconnect")

    t_abort_start = time.perf_counter()
    try:
        await harness.run_session(
            audio,
            model=model,
            language=language,
            call_id="abort-before-reconnect",
            chunk_ms=chunk_ms,
            speed=speed,
            realtime=realtime,
            cancel_check=_abort_hook,
        )
    except _IntentionalMidStreamAbort:
        aborted_cleanly = True
    abort_duration_ms = round(max(0.01, (time.perf_counter() - t_abort_start) * 1000.0), 2)

    if cancel_check is not None:
        cancel_check()

    info_after_abort = await _wait_for_zero_active_calls(harness, max_wait_sec=0.5)
    active_after_abort = info_after_abort.active_calls if info_after_abort.ok else -1

    # Immediately reconnect on a new Transcribe stream
    reconnect_res = await harness.run_session(
        audio,
        model=model,
        language=language,
        call_id="reconnected-after-abort",
        chunk_ms=chunk_ms,
        speed=speed,
        realtime=realtime,
        cancel_check=cancel_check,
    )

    info_after_reconnect = await _wait_for_zero_active_calls(harness, max_wait_sec=0.5)
    active_after_reconnect = (
        info_after_reconnect.active_calls if info_after_reconnect.ok else -1
    )
    total_wall_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)

    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def _add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    _add(
        "mid_stream_abort_terminated_promptly",
        aborted_cleanly,
        f"aborted={aborted_cleanly} after {abort_after_chunks} checks in {abort_duration_ms:.2f}ms",
    )
    _add(
        "slot_released_after_client_abort",
        info_after_abort.ok and active_after_abort == 0,
        f"GetServerInfo.ok={info_after_abort.ok}, active_calls={active_after_abort}",
    )
    _add(
        "reconnected_stream_succeeded",
        reconnect_res.ok and reconnect_res.grpc_status_code == "OK" and bool(reconnect_res.text.strip()),
        (
            f"status={reconnect_res.grpc_status_code}, ok={reconnect_res.ok}, "
            f"ttfp_ms={reconnect_res.ttfp_ms}, final_latency_ms={reconnect_res.final_latency_ms}, "
            f"words={reconnect_res.word_count}"
        ),
    )
    _add(
        "slot_released_after_reconnect_completion",
        info_after_reconnect.ok and active_after_reconnect == 0,
        f"GetServerInfo.ok={info_after_reconnect.ok}, active_calls={active_after_reconnect}",
    )

    passed = len(errors) == 0
    summary = (
        f"Client abort + reconnect OK: slot released (active_calls=0), "
        f"reconnected TTFP={reconnect_res.ttfp_ms}ms, FinalLat={reconnect_res.final_latency_ms}ms"
        if passed
        else "; ".join(errors)
    )

    return ReconnectShutdownResult(
        passed=passed,
        abort_duration_ms=abort_duration_ms,
        aborted_chunks_sent=abort_after_chunks,
        active_calls_after_abort=active_after_abort,
        reconnect_session=reconnect_res,
        active_calls_after_reconnect=active_after_reconnect,
        total_wall_ms=total_wall_ms,
        checks=checks,
        errors=errors,
        summary_message=summary,
    )


async def run_clean_shutdown_and_channel_reuse_check_async(
    harness: LiveSTTStreamingHarness,
    *,
    model: str = "",
    language: str = "en",
    audio: AudioFixture | bytes | str = "short_speech",
    chunk_ms: float = DEFAULT_CHUNK_MS,
    speed: float | None = 4.0,
    realtime: bool = True,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Verify clean half-close Final tail emission, sequential channel reuse, and client.close()."""
    t0 = time.perf_counter()
    async with harness.create_client() as shared_client:
        s1 = await shared_client.stream_transcribe(
            audio,
            model=model,
            language=language,
            call_id="channel-reuse-1",
            chunk_ms=chunk_ms,
            speed=speed,
            realtime=realtime,
            cancel_check=cancel_check,
        )
        s2 = await shared_client.stream_transcribe(
            audio,
            model=model,
            language=language,
            call_id="channel-reuse-2",
            chunk_ms=chunk_ms,
            speed=speed,
            realtime=realtime,
            cancel_check=cancel_check,
        )
        info_after = await shared_client.get_server_info()

    wall_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def _add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    _add(
        "first_stream_clean_half_close_final",
        s1.ok and s1.final_event is not None and s1.final_latency_ms is not None,
        f"s1.ok={s1.ok}, final_latency_ms={s1.final_latency_ms}, total_audio_sec={s1.total_audio_sec_reported}",
    )
    _add(
        "second_stream_channel_reuse_ok",
        s2.ok and s2.final_event is not None and s2.text == s1.text,
        f"s2.ok={s2.ok}, final_latency_ms={s2.final_latency_ms}, text_matches={s2.text == s1.text}",
    )
    _add(
        "zero_active_calls_before_channel_close",
        info_after.ok and info_after.active_calls == 0,
        f"active_calls={info_after.active_calls}",
    )

    passed = len(errors) == 0
    return {
        "passed": passed,
        "wall_ms": wall_ms,
        "sessions": [s1, s2],
        "active_calls_after": info_after.active_calls if info_after.ok else -1,
        "checks": checks,
        "errors": errors,
        "summary_message": (
            f"Clean half-close & channel reuse OK (2 sequential streams, "
            f"final_latency={s1.final_latency_ms}ms / {s2.final_latency_ms}ms, active_calls=0)"
            if passed
            else "; ".join(errors)
        ),
    }


async def run_protocol_and_drain_contract_check_async(
    harness: LiveSTTStreamingHarness,
    *,
    model: str = "",
    batch_only_model: str = "whisper-large-v3-turbo",
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Verify gRPC Transcribe INVALID_ARGUMENT protocol contract checks."""
    t0 = time.perf_counter()
    short_fix = get_audio_fixture("short_speech")

    if cancel_check is not None:
        cancel_check()

    # 1. Missing initial StreamConfig (audio frame sent first)
    res_no_config = await harness.run_session(
        short_fix,
        model=model,
        send_config=False,
        realtime=False,
        raise_on_grpc_error=False,
    )

    # 2. Duplicate StreamConfig sent on the same stream
    res_dup_config = await harness.run_session(
        short_fix,
        model=model,
        send_duplicate_config=True,
        realtime=False,
        raise_on_grpc_error=False,
    )

    # 3. Unsupported sample_rate_hz=8000 in StreamConfig
    res_bad_sr = await harness.run_session(
        b"\x00\x00" * 1600,
        model=model,
        sample_rate_hz=8000,
        validate_wav=False,
        realtime=False,
        raise_on_grpc_error=False,
    )

    # 4. Unsupported encoding=AUDIO_ENCODING_MULAW
    res_bad_enc = await harness.run_session(
        short_fix,
        model=model,
        encoding=int(asr_pb2.AUDIO_ENCODING_MULAW),
        realtime=False,
        raise_on_grpc_error=False,
    )

    # 5. Batch-only model on StreamingASR.Transcribe
    res_batch_model = await harness.run_session(
        short_fix,
        model=batch_only_model,
        realtime=False,
        raise_on_grpc_error=False,
    )

    # 6. Unknown model on StreamingASR.Transcribe
    res_unknown_model = await harness.run_session(
        short_fix,
        model="unknown-nonexistent-streaming-model-999",
        realtime=False,
        raise_on_grpc_error=False,
    )

    wall_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def _add(name: str, res: StreamingSessionResult) -> None:
        passed = (not res.ok) and res.grpc_status_code == "INVALID_ARGUMENT"
        detail = f"status={res.grpc_status_code}, ok={res.ok}, msg={res.error_message!r}"
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            errors.append(f"{name}: expected INVALID_ARGUMENT, got {detail}")

    _add("missing_stream_config_rejected", res_no_config)
    _add("duplicate_stream_config_rejected", res_dup_config)
    _add("unsupported_sample_rate_8khz_rejected", res_bad_sr)
    _add("unsupported_encoding_mulaw_rejected", res_bad_enc)
    _add("batch_only_model_rejected", res_batch_model)
    _add("unknown_model_rejected", res_unknown_model)

    passed = len(errors) == 0
    return {
        "passed": passed,
        "wall_ms": wall_ms,
        "checks": checks,
        "errors": errors,
        "summary_message": (
            f"All {len(checks)} gRPC protocol error checks rejected with INVALID_ARGUMENT"
            if passed
            else "; ".join(errors)
        ),
    }


def run_reconnect_after_abort_check(
    harness: LiveSTTStreamingHarness,
    **kwargs: Any,
) -> ReconnectShutdownResult:
    """Synchronous wrapper for ``run_reconnect_after_abort_check_async``."""
    return _run_coro_sync(run_reconnect_after_abort_check_async(harness, **kwargs))


def run_clean_shutdown_and_channel_reuse_check(
    harness: LiveSTTStreamingHarness,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synchronous wrapper for ``run_clean_shutdown_and_channel_reuse_check_async``."""
    return _run_coro_sync(
        run_clean_shutdown_and_channel_reuse_check_async(harness, **kwargs)
    )


def run_protocol_and_drain_contract_check(
    harness: LiveSTTStreamingHarness,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synchronous wrapper for ``run_protocol_and_drain_contract_check_async``."""
    return _run_coro_sync(
        run_protocol_and_drain_contract_check_async(harness, **kwargs)
    )


def classify_within_budget_rejections(sessions: Sequence[StreamingSessionResult]) -> dict[str, Any]:
    """Split failed sessions of a within-budget burst into VRAM-limited rejections (acceptable),
    call-limit rejections and other failures, grouped by status and message for reporting."""
    vram_limited = capacity_rejected = other_failed = 0
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for sess in sessions:
        if sess.ok:
            continue
        code = str(sess.grpc_status_code or "UNKNOWN")
        msg = str(sess.error_message or "").strip()
        if code == "RESOURCE_EXHAUSTED" and "vram" in msg.lower():
            kind = "vram_limited"
            vram_limited += 1
        elif code == "RESOURCE_EXHAUSTED":
            kind = "capacity_rejected"
            capacity_rejected += 1
        else:
            kind = "other_failed"
            other_failed += 1
        g = groups.setdefault((code, kind), {"grpc_status_code": code, "kind": kind, "count": 0, "sample_message": msg[:200]})
        g["count"] += 1
    return {
        "vram_limited": vram_limited,
        "capacity_rejected": capacity_rejected,
        "other_failed": other_failed,
        "groups": sorted(groups.values(), key=lambda g: (g["kind"], g["grpc_status_code"])),
    }


def execute_grpc_streaming_suites(
    *,
    ctx: TestContext,
    result: TesterResult,
    cfg: dict[str, Any],
    target: dict[str, Any],
    suites_to_run: list[str],
    asr_models: list[ModelInfo],
    language: str,
    repetitions: int,
    tested_engines: set[str],
    tested_models: set[str],
    all_latencies_ms: list[float],
    all_rtfs: list[float],
    completed_cases: int,
    planned_cases: int,
    api_key: str | None = None,
    speech_fixtures: dict[str, AudioFixture] | None = None,
    speech_source: str = "synthetic",
    speech_wer_threshold: float = DEFAULT_SPEECH_WER_THRESHOLD,
) -> int:
    """Execute `grpc_streaming` and/or `grpc_long_call` suites using `LiveSTTStreamingHarness`.

    ``speech_fixtures`` ('short_speech', 'medium_speech', 'dialog') replace the generated tone fixtures
    when real speech is available (``speech_source='notsofar'``); a loose WER check is then added to the
    per-model Transcribe cases.
    """
    run_streaming = "grpc_streaming" in suites_to_run
    run_long_call = ("grpc_long_call" in suites_to_run) or (
        run_streaming and bool(cfg.get("include_long_call", False))
    )
    if not run_streaming and not run_long_call:
        return completed_cases

    chunk_ms = float(cfg.get("grpc_chunk_ms", cfg.get("chunk_ms", DEFAULT_CHUNK_MS)))
    speed = float(cfg.get("grpc_speed", 1.0))
    burst_speed = float(cfg.get("grpc_burst_speed", 4.0))
    realtime = bool(cfg.get("grpc_realtime", True))
    timeout = float(cfg.get("timeout_seconds", 120.0))

    harness = LiveSTTStreamingHarness(
        target,
        config=cfg,
        api_key=api_key,
        timeout=timeout,
        default_chunk_ms=chunk_ms,
        default_speed=speed if (realtime and speed > 0) else None,
        default_realtime=realtime and speed > 0,
    )

    server_info = harness.get_server_info_sync(raise_on_grpc_error=False)
    streaming_models = select_grpc_streaming_models(asr_models, server_info, cfg)
    max_concurrent = (
        int(server_info.max_concurrent_calls)
        if (server_info.ok and server_info.max_concurrent_calls > 0)
        else 2
    )

    recorded_sessions: list[StreamingSessionResult] = []

    if run_streaming:
        # -------------------------------------------------------------------
        # Case 1: GetServerInfo readiness & server metadata
        # -------------------------------------------------------------------
        ctx.check_cancelled()
        case_info_id = "grpc_streaming:get_server_info"
        ctx.log(f"Running [{completed_cases + 1}/{planned_cases}] {case_info_id} on {harness.grpc_target}...")

        info_checks: list[dict[str, Any]] = []
        info_errors: list[str] = []

        def _add_info_check(name: str, passed: bool, detail: str) -> None:
            info_checks.append({"name": name, "passed": passed, "detail": detail})
            if not passed:
                info_errors.append(f"{name}: {detail}")

        _add_info_check(
            "grpc_status_ok",
            server_info.ok and server_info.grpc_status_code == "OK",
            f"status={server_info.grpc_status_code}, error={server_info.error_message!r}",
        )
        _add_info_check(
            "version_and_backend_populated",
            bool(server_info.version.strip()) and bool(server_info.backend.strip()),
            f"version={server_info.version!r}, backend={server_info.backend!r}, default_model={server_info.default_model!r}",
        )
        _add_info_check(
            "capacity_telemetry_valid",
            server_info.max_concurrent_calls >= 1 and server_info.active_calls >= 0,
            (
                f"max_concurrent_calls={server_info.max_concurrent_calls}, "
                f"active_calls={server_info.active_calls}, warm_spares={server_info.warm_spares}"
            ),
        )

        info_passed = len(info_errors) == 0
        info_status = "pass" if info_passed else "fail"
        all_latencies_ms.append(server_info.latency_ms)

        result.add_case(
            case_id=case_info_id,
            suite="grpc_streaming",
            status=info_status,
            metrics={
                "latency_ms": {"value": server_info.latency_ms, "unit": "ms", "better": "lower"},
                "max_concurrent_calls": {"value": server_info.max_concurrent_calls, "unit": "calls"},
                "active_calls": {"value": server_info.active_calls, "unit": "calls", "better": "lower"},
                "warm_spares": {"value": server_info.warm_spares, "unit": "workers"},
            },
            detail={
                "endpoint": "StreamingASR.GetServerInfo",
                "grpc_target": harness.grpc_target,
                "server_info": server_info.to_dict(),
                "passed": info_passed,
                "eval_message": (
                    f"GetServerInfo OK: model={server_info.default_model}, "
                    f"backend={server_info.backend}, max_concurrent={server_info.max_concurrent_calls}"
                    if info_passed
                    else "; ".join(info_errors)
                ),
                "checks": info_checks,
                "errors": info_errors,
            },
        )
        completed_cases += 1
        ctx.progress(completed_cases, planned_cases, f"Completed {case_info_id}")

        # -------------------------------------------------------------------
        # Case 2: Per-model paced & burst Transcribe sessions (TTFP, Final Latency, RTF)
        # -------------------------------------------------------------------
        fx = speech_fixtures or {}
        short_fix = fx.get("short_speech") or get_audio_fixture("short_speech")
        medium_fix = fx.get("medium_speech") or get_audio_fixture("medium_speech")
        dialog_fix = fx.get("dialog") or get_audio_fixture("two_speaker_dialog")

        fixture_specs = [
            ("short_speech", short_fix, speed if realtime else 0.0, realtime),
            ("medium_speech", medium_fix, speed if realtime else 0.0, realtime),
            (
                "fast_paced_burst",
                dialog_fix,
                burst_speed if burst_speed > 0 else None,
                burst_speed > 0,
            ),
        ]

        for m_info in streaming_models:
            tested_engines.add(m_info.engine)
            tested_models.add(m_info.id)

            for case_suffix, fix, case_speed, case_rt in fixture_specs:
                ctx.check_cancelled()
                case_id = f"grpc_streaming:{m_info.id}:{case_suffix}"
                ctx.log(
                    f"Running [{completed_cases + 1}/{planned_cases}] {case_id} "
                    f"(fixture={fix.id}, {fix.duration_sec}s, chunk_ms={chunk_ms}, speed={case_speed})..."
                )

                rep_sessions: list[StreamingSessionResult] = []
                for _ in range(repetitions):
                    ctx.check_cancelled()
                    sess = harness.run_session_sync(
                        fix,
                        model=m_info.id,
                        language=language,
                        chunk_ms=chunk_ms,
                        speed=case_speed,
                        realtime=case_rt,
                        enable_word_timestamps=True,
                        cancel_check=ctx.check_cancelled,
                    )
                    rep_sessions.append(sess)
                    recorded_sessions.append(sess)
                    all_latencies_ms.append(sess.total_latency_ms)
                    all_rtfs.append(sess.rtf)

                last_sess = rep_sessions[-1]
                passed, checks, errors, eval_msg = evaluate_streaming_session_contract(
                    last_sess,
                    expected_duration_sec=fix.duration_sec,
                    require_partials=True,
                    require_word_timestamps=True,
                )
                acc_metrics: dict[str, Any] = {}
                acc_detail: dict[str, Any] = {"fixture_source": speech_source}
                if speech_source == "notsofar" and fix.reference_text:
                    acc_ok, acc_check, acc_err, acc_metrics, acc_extra = speech_accuracy_check(
                        fix.reference_text, last_sess.text, speech_wer_threshold
                    )
                    acc_detail.update(acc_extra)
                    checks = [*checks, acc_check]
                    if acc_err:
                        errors = [*errors, acc_err]
                        eval_msg = f"{eval_msg}; {acc_err}" if not passed else acc_err
                    passed = passed and acc_ok
                case_status = "pass" if passed else "fail"

                rep_summary = compute_grpc_streaming_summary(rep_sessions)
                case_metrics: dict[str, Any] = {
                    "ttfp_ms": {
                        "value": rep_summary["ttfp_mean_ms"],
                        "unit": "ms",
                        "better": "lower",
                    },
                    "ttfp_from_first_audio_ms": {
                        "value": rep_summary["ttfp_from_first_audio_mean_ms"],
                        "unit": "ms",
                        "better": "lower",
                    },
                    "final_latency_ms": {
                        "value": rep_summary["final_latency_mean_ms"],
                        "unit": "ms",
                        "better": "lower",
                    },
                    "ready_latency_ms": {
                        "value": rep_summary["ready_latency_mean_ms"],
                        "unit": "ms",
                        "better": "lower",
                    },
                    "latency_ms": {
                        "value": rep_summary["latency_mean_ms"],
                        "unit": "ms",
                        "better": "lower",
                    },
                    "rtf": {
                        "value": rep_summary["rtf_mean"],
                        "unit": "x",
                        "better": "lower",
                    },
                    "partial_count": {
                        "value": last_sess.partial_count,
                        "unit": "events",
                    },
                    "word_count": {
                        "value": last_sess.word_count,
                        "unit": "words",
                    },
                    "audio_duration_sec": {
                        "value": fix.duration_sec,
                        "unit": "s",
                    },
                    **acc_metrics,
                }
                if last_sess.inter_partial_latencies_ms:
                    case_metrics["inter_partial_latency_mean_ms"] = {
                        "value": rep_summary["inter_partial_latency_mean_ms"],
                        "unit": "ms",
                        "better": "lower",
                    }

                result.add_case(
                    case_id=case_id,
                    suite="grpc_streaming",
                    status=case_status,
                    metrics=case_metrics,
                    detail={
                        "endpoint": "StreamingASR.Transcribe",
                        "grpc_target": harness.grpc_target,
                        "model": m_info.id,
                        "resolved_model": last_sess.resolved_model,
                        "engine": m_info.engine,
                        "fixture_id": fix.id,
                        **acc_detail,
                        "audio_duration_sec": fix.duration_sec,
                        "chunk_ms": chunk_ms,
                        "speed": case_speed,
                        "realtime": case_rt,
                        "grpc_status_code": last_sess.grpc_status_code,
                        "text": last_sess.text,
                        "partial_count": last_sess.partial_count,
                        "word_count": last_sess.word_count,
                        "worker_generations": last_sess.worker_generations,
                        "session": last_sess.to_dict(),
                        "passed": passed,
                        "eval_message": eval_msg,
                        "checks": checks,
                        "errors": errors,
                    },
                )
                completed_cases += 1
                ctx.log(f"  {case_id} -> {case_status.upper()} | {eval_msg}")
                ctx.progress(completed_cases, planned_cases, f"Completed {case_id}")

        # -------------------------------------------------------------------
        # Case 3: Concurrent Transcribe sessions (within budget & over-budget recovery)
        # -------------------------------------------------------------------
        primary_model = streaming_models[0].id
        ctx.check_cancelled()
        within_n = max(1, min(max_concurrent, int(cfg.get("grpc_concurrent_within_budget") or max_concurrent)))
        case_conc_ok = "grpc_streaming:concurrent_within_budget"
        ctx.log(
            f"Running [{completed_cases + 1}/{planned_cases}] {case_conc_ok} "
            f"(concurrency={within_n} <= max_concurrent_calls={max_concurrent})..."
        )

        conc_within: ConcurrentStreamingResult = harness.run_concurrent_sessions_sync(
            within_n,
            short_fix,
            model=primary_model,
            language=language,
            chunk_ms=chunk_ms,
            speed=max(speed, 1.0) if realtime else 2.0,
            realtime=True,
            cancel_check=ctx.check_cancelled,
        )
        for s in conc_within.sessions:
            if s.ok:
                recorded_sessions.append(s)
        all_latencies_ms.append(conc_within.wall_ms)

        within_summary = compute_grpc_streaming_summary([s for s in conc_within.sessions if s.ok])
        within_breakdown = classify_within_budget_rejections(conc_within.sessions)
        vram_limited = within_breakdown["vram_limited"]
        capacity_rejected = within_breakdown["capacity_rejected"]
        other_failed = within_breakdown["other_failed"]
        rejection_text = ", ".join(
            f"{g['grpc_status_code']} x{g['count']} ({g['sample_message']!r})"
            for g in within_breakdown["groups"]
        )
        # Below max_concurrent_calls, a clean RESOURCE_EXHAUSTED for insufficient VRAM is correct
        # behaviour (LS-17): the GPU, not the call limit, is the real budget. Only call-limit
        # rejections and other errors are failures.
        within_passed = (
            conc_within.ok_count >= 1
            and capacity_rejected == 0
            and other_failed == 0
            and conc_within.ok_count + vram_limited == within_n
        )
        within_detail_text = (
            f"ok={conc_within.ok_count}/{within_n}, vram_limited={vram_limited}, "
            f"capacity_rejected={capacity_rejected}, other_errors={other_failed}"
            + (f": {rejection_text}" if rejection_text else "")
        )
        within_checks = [
            {
                "name": "within_budget_sessions_ok_or_vram_limited",
                "passed": within_passed,
                "detail": within_detail_text,
            }
        ]
        if vram_limited:
            within_checks.append(
                {
                    "name": "vram_limited_sessions_rejected_cleanly",
                    "passed": True,
                    "detail": (
                        f"{vram_limited} session(s) rejected with RESOURCE_EXHAUSTED for insufficient VRAM "
                        f"below max_concurrent_calls={max_concurrent}; effective GPU capacity is about "
                        f"{conc_within.ok_count} concurrent {primary_model} stream(s)"
                    ),
                }
            )
        within_errors = (
            []
            if within_passed
            else [f"within_budget_sessions_ok_or_vram_limited: {within_detail_text}"]
        )

        result.add_case(
            case_id=case_conc_ok,
            suite="grpc_streaming",
            status="pass" if within_passed else "fail",
            metrics={
                "latency_ms": {"value": conc_within.wall_ms, "unit": "ms", "better": "lower"},
                "ttfp_ms": {"value": within_summary["ttfp_mean_ms"], "unit": "ms", "better": "lower"},
                "final_latency_ms": {
                    "value": within_summary["final_latency_mean_ms"],
                    "unit": "ms",
                    "better": "lower",
                },
                "rtf": {"value": within_summary["rtf_mean"], "unit": "x", "better": "lower"},
                "concurrent_sessions": {"value": within_n, "unit": "sessions"},
                "sessions_ok": {"value": conc_within.ok_count, "unit": "sessions", "better": "higher"},
                "vram_limited_sessions": {"value": vram_limited, "unit": "sessions"},
            },
            detail={
                "endpoint": "StreamingASR.Transcribe",
                "model": primary_model,
                "concurrency": within_n,
                "max_concurrent_calls": max_concurrent,
                "ok_count": conc_within.ok_count,
                "resource_exhausted_count": conc_within.resource_exhausted_count,
                "vram_limited_count": vram_limited,
                "capacity_rejected_count": capacity_rejected,
                "other_failed_count": other_failed,
                "rejections": within_breakdown["groups"],
                "wall_ms": conc_within.wall_ms,
                "passed": within_passed,
                "eval_message": (
                    f"Concurrent within-budget ({conc_within.ok_count}/{within_n} OK"
                    + (f", {vram_limited} limited by VRAM" if vram_limited else "")
                    + "): "
                    f"TTFP={within_summary['ttfp_mean_ms']}ms, "
                    f"FinalLat={within_summary['final_latency_mean_ms']}ms, "
                    f"RTF={within_summary['rtf_mean']:.4f}x"
                    if within_passed
                    else "; ".join(within_errors)
                ),
                "checks": within_checks,
                "errors": within_errors,
            },
        )
        completed_cases += 1
        ctx.progress(completed_cases, planned_cases, f"Completed {case_conc_ok}")

        # Case 3b: Over-budget concurrent burst -> RESOURCE_EXHAUSTED + post-burst slot recovery
        ctx.check_cancelled()
        burst_n = int(cfg.get("grpc_concurrency_burst_size") or (max_concurrent + 2))
        case_conc_burst = "grpc_streaming:concurrent_over_budget_and_recovery"
        ctx.log(
            f"Running [{completed_cases + 1}/{planned_cases}] {case_conc_burst} "
            f"(burst={burst_n} vs max_concurrent_calls={max_concurrent})..."
        )

        # Use real-time pacing so all streams overlap in flight on the server
        conc_burst: ConcurrentStreamingResult = harness.run_concurrent_sessions_sync(
            burst_n,
            short_fix,
            model=primary_model,
            language=language,
            chunk_ms=chunk_ms,
            speed=2.0,
            realtime=True,
            cancel_check=ctx.check_cancelled,
        )
        info_after_burst = harness.get_server_info_sync(raise_on_grpc_error=False)
        recovery_sess = harness.run_session_sync(
            short_fix,
            model=primary_model,
            language=language,
            chunk_ms=chunk_ms,
            speed=4.0,
            realtime=True,
            cancel_check=ctx.check_cancelled,
        )
        if recovery_sess.ok:
            recorded_sessions.append(recovery_sess)
        all_latencies_ms.append(conc_burst.wall_ms)

        burst_checks: list[dict[str, Any]] = []
        burst_errors: list[str] = []

        def _add_burst_check(name: str, passed: bool, detail: str) -> None:
            burst_checks.append({"name": name, "passed": passed, "detail": detail})
            if not passed:
                burst_errors.append(f"{name}: {detail}")

        _add_burst_check(
            "only_ok_or_resource_exhausted",
            conc_burst.other_error_count == 0 and conc_burst.invalid_argument_count == 0,
            (
                f"ok={conc_burst.ok_count}, resource_exhausted={conc_burst.resource_exhausted_count}, "
                f"unavailable={conc_burst.unavailable_count}, other={conc_burst.other_error_count}"
            ),
        )
        _add_burst_check(
            "admitted_sessions_succeeded",
            1 <= conc_burst.ok_count <= max_concurrent,
            f"admitted_ok={conc_burst.ok_count} (max_concurrent_calls={max_concurrent})",
        )
        if burst_n > max_concurrent:
            _add_burst_check(
                "excess_sessions_rejected_resource_exhausted",
                conc_burst.resource_exhausted_count >= 1,
                (
                    f"resource_exhausted={conc_burst.resource_exhausted_count} "
                    f"(burst={burst_n} > max_concurrent_calls={max_concurrent})"
                ),
            )
        _add_burst_check(
            "post_burst_slots_released_and_recovery_ok",
            info_after_burst.ok and info_after_burst.active_calls == 0 and recovery_sess.ok,
            (
                f"active_calls_after={info_after_burst.active_calls}, "
                f"recovery_ok={recovery_sess.ok} ({recovery_sess.grpc_status_code})"
            ),
        )

        burst_passed = len(burst_errors) == 0
        result.add_case(
            case_id=case_conc_burst,
            suite="grpc_streaming",
            status="pass" if burst_passed else "fail",
            metrics={
                "latency_ms": {"value": conc_burst.wall_ms, "unit": "ms", "better": "lower"},
                "burst_size": {"value": burst_n, "unit": "sessions"},
                "admitted_ok_count": {"value": conc_burst.ok_count, "unit": "sessions"},
                "resource_exhausted_count": {
                    "value": conc_burst.resource_exhausted_count,
                    "unit": "sessions",
                },
                "active_calls_after": {
                    "value": info_after_burst.active_calls if info_after_burst.ok else -1,
                    "unit": "calls",
                    "better": "lower",
                },
            },
            detail={
                "endpoint": "StreamingASR.Transcribe + GetServerInfo",
                "model": primary_model,
                "burst_size": burst_n,
                "max_concurrent_calls": max_concurrent,
                "ok_count": conc_burst.ok_count,
                "resource_exhausted_count": conc_burst.resource_exhausted_count,
                "active_calls_after": info_after_burst.active_calls,
                "recovery_ok": recovery_sess.ok,
                "passed": burst_passed,
                "eval_message": (
                    f"Burst {burst_n} streams -> {conc_burst.ok_count} OK, "
                    f"{conc_burst.resource_exhausted_count} RESOURCE_EXHAUSTED; "
                    f"recovery OK (active_calls=0)"
                    if burst_passed
                    else "; ".join(burst_errors)
                ),
                "checks": burst_checks,
                "errors": burst_errors,
            },
        )
        completed_cases += 1
        ctx.progress(completed_cases, planned_cases, f"Completed {case_conc_burst}")

        # -------------------------------------------------------------------
        # Case 4: Reconnect after mid-stream client abort
        # -------------------------------------------------------------------
        ctx.check_cancelled()
        case_reconn_id = "grpc_streaming:reconnect_after_client_abort"
        ctx.log(f"Running [{completed_cases + 1}/{planned_cases}] {case_reconn_id}...")
        reconn_res = run_reconnect_after_abort_check(
            harness,
            model=primary_model,
            language=language,
            abort_after_chunks=4,
            audio=medium_fix,
            chunk_ms=chunk_ms,
            speed=max(speed, 2.0) if realtime else 2.0,
            realtime=True,
            cancel_check=ctx.check_cancelled,
        )
        if reconn_res.reconnect_session.ok:
            recorded_sessions.append(reconn_res.reconnect_session)
        all_latencies_ms.append(reconn_res.total_wall_ms)

        result.add_case(
            case_id=case_reconn_id,
            suite="grpc_streaming",
            status="pass" if reconn_res.passed else "fail",
            metrics={
                "latency_ms": {"value": reconn_res.total_wall_ms, "unit": "ms", "better": "lower"},
                "abort_duration_ms": {
                    "value": reconn_res.abort_duration_ms,
                    "unit": "ms",
                    "better": "lower",
                },
                "ttfp_ms": {
                    "value": float(reconn_res.reconnect_session.ttfp_ms or 0.0),
                    "unit": "ms",
                    "better": "lower",
                },
                "final_latency_ms": {
                    "value": float(reconn_res.reconnect_session.final_latency_ms or 0.0),
                    "unit": "ms",
                    "better": "lower",
                },
                "active_calls_after_abort": {
                    "value": reconn_res.active_calls_after_abort,
                    "unit": "calls",
                    "better": "lower",
                },
            },
            detail={
                "endpoint": "StreamingASR.Transcribe",
                "model": primary_model,
                **reconn_res.to_dict(),
                "eval_message": reconn_res.summary_message,
            },
        )
        completed_cases += 1
        ctx.progress(completed_cases, planned_cases, f"Completed {case_reconn_id}")

        # -------------------------------------------------------------------
        # Case 5: Clean half-close shutdown & sequential channel reuse
        # -------------------------------------------------------------------
        ctx.check_cancelled()
        case_shutdown_id = "grpc_streaming:clean_half_close_shutdown"
        ctx.log(f"Running [{completed_cases + 1}/{planned_cases}] {case_shutdown_id}...")
        shutdown_res = run_clean_shutdown_and_channel_reuse_check(
            harness,
            model=primary_model,
            language=language,
            audio=short_fix,
            chunk_ms=chunk_ms,
            speed=max(speed, 2.0) if realtime else 2.0,
            realtime=True,
            cancel_check=ctx.check_cancelled,
        )
        for s in shutdown_res["sessions"]:
            if s.ok:
                recorded_sessions.append(s)
        all_latencies_ms.append(shutdown_res["wall_ms"])
        sh_summary = compute_grpc_streaming_summary(shutdown_res["sessions"])

        result.add_case(
            case_id=case_shutdown_id,
            suite="grpc_streaming",
            status="pass" if shutdown_res["passed"] else "fail",
            metrics={
                "latency_ms": {"value": shutdown_res["wall_ms"], "unit": "ms", "better": "lower"},
                "final_latency_ms": {
                    "value": sh_summary["final_latency_mean_ms"],
                    "unit": "ms",
                    "better": "lower",
                },
                "ttfp_ms": {
                    "value": sh_summary["ttfp_mean_ms"],
                    "unit": "ms",
                    "better": "lower",
                },
                "active_calls_after": {
                    "value": shutdown_res["active_calls_after"],
                    "unit": "calls",
                    "better": "lower",
                },
            },
            detail={
                "endpoint": "StreamingASR.Transcribe",
                "model": primary_model,
                "passed": shutdown_res["passed"],
                "wall_ms": shutdown_res["wall_ms"],
                "active_calls_after": shutdown_res["active_calls_after"],
                "checks": shutdown_res["checks"],
                "errors": shutdown_res["errors"],
                "eval_message": shutdown_res["summary_message"],
            },
        )
        completed_cases += 1
        ctx.progress(completed_cases, planned_cases, f"Completed {case_shutdown_id}")

        # -------------------------------------------------------------------
        # Case 6: Protocol error contract (INVALID_ARGUMENT checks)
        # -------------------------------------------------------------------
        ctx.check_cancelled()
        case_proto_err_id = "grpc_streaming:protocol_error_contract"
        ctx.log(f"Running [{completed_cases + 1}/{planned_cases}] {case_proto_err_id}...")
        proto_err_res = run_protocol_and_drain_contract_check(
            harness,
            model=primary_model,
            cancel_check=ctx.check_cancelled,
        )
        all_latencies_ms.append(proto_err_res["wall_ms"])

        result.add_case(
            case_id=case_proto_err_id,
            suite="grpc_streaming",
            status="pass" if proto_err_res["passed"] else "fail",
            metrics={
                "latency_ms": {"value": proto_err_res["wall_ms"], "unit": "ms", "better": "lower"},
                "checks_passed": {
                    "value": sum(1 for c in proto_err_res["checks"] if c["passed"]),
                    "unit": "checks",
                    "better": "higher",
                },
            },
            detail={
                "endpoint": "StreamingASR.Transcribe",
                "model": primary_model,
                "passed": proto_err_res["passed"],
                "wall_ms": proto_err_res["wall_ms"],
                "checks": proto_err_res["checks"],
                "errors": proto_err_res["errors"],
                "eval_message": proto_err_res["summary_message"],
            },
        )
        completed_cases += 1
        ctx.progress(completed_cases, planned_cases, f"Completed {case_proto_err_id}")

    # ===================================================================
    # Opt-in Slow Suite: grpc_long_call (Long-Call Stability)
    # ===================================================================
    if run_long_call:
        long_dur_sec = float(cfg.get("long_call_duration_sec", 30.0))
        long_speed = float(cfg.get("long_call_speed", 4.0))
        long_max_rtf = float(cfg.get("long_call_max_rtf", 2.0))
        long_max_drift = float(cfg.get("long_call_max_drift_ratio", 4.0))
        # Loop real speech when available: the generated tone fixture yields no partials or words,
        # which fails continuous_partials_emitted and the word-timestamp checks (run 36).
        long_base = (speech_fixtures or {}).get("dialog") if speech_source == "notsofar" else None
        long_fix = build_long_call_audio_fixture(
            duration_sec=long_dur_sec,
            base_fixture_id="two_speaker_dialog",
            base_fixture=long_base,
        )
        long_models = (
            streaming_models[:1]
            if streaming_models
            else [infer_model_info({"id": DEFAULT_GRPC_STREAMING_MODEL, "type": "asr", "supports_streaming": True})]
        )

        for m_info in long_models:
            tested_engines.add(m_info.engine)
            tested_models.add(m_info.id)

            ctx.check_cancelled()
            case_long_id = f"grpc_long_call:{m_info.id}:sustained_stream"
            ctx.log(
                f"Running [{completed_cases + 1}/{planned_cases}] {case_long_id} "
                f"(duration={long_fix.duration_sec:.1f}s, speed={long_speed}, chunk_ms={chunk_ms})..."
            )

            long_sess = harness.run_session_sync(
                long_fix,
                model=m_info.id,
                language=language,
                chunk_ms=chunk_ms,
                speed=long_speed if long_speed > 0 else None,
                realtime=long_speed > 0,
                enable_word_timestamps=True,
                timeout=max(timeout, long_fix.duration_sec * 2.5 + 30.0),
                cancel_check=ctx.check_cancelled,
            )
            info_after_long = harness.get_server_info_sync(raise_on_grpc_error=False)
            recorded_sessions.append(long_sess)
            all_latencies_ms.append(long_sess.total_latency_ms)
            all_rtfs.append(long_sess.rtf)

            report = evaluate_long_call_stability(
                long_sess,
                expected_duration_sec=long_fix.duration_sec,
                max_rtf=long_max_rtf,
                max_drift_ratio=long_max_drift,
            )
            slot_clean = info_after_long.ok and info_after_long.active_calls == 0
            report.checks.append({
                "name": "post_long_call_slot_released",
                "passed": slot_clean,
                "detail": f"active_calls_after={info_after_long.active_calls}",
            })
            if not slot_clean:
                report.errors.append(
                    f"post_long_call_slot_released: active_calls={info_after_long.active_calls}"
                )
                report.passed = False

            long_acc_metrics: dict[str, Any] = {}
            long_acc_detail: dict[str, Any] = {"fixture_source": speech_source if long_base is not None else "synthetic"}
            if long_base is not None and long_fix.reference_text:
                acc_ok, acc_check, acc_err, long_acc_metrics, acc_extra = speech_accuracy_check(
                    long_fix.reference_text, long_sess.text, speech_wer_threshold
                )
                long_acc_detail.update(acc_extra)
                report.checks.append(acc_check)
                if acc_err:
                    report.errors.append(acc_err)
                    report.passed = False
            if not report.passed:
                # Checks appended after evaluate_long_call_stability() must show in the summary too
                report.summary_message = "; ".join(report.errors)

            long_status = "pass" if report.passed else "fail"
            long_metrics: dict[str, Any] = {
                "audio_duration_sec": {"value": report.audio_duration_sec, "unit": "s"},
                "reported_audio_sec": {"value": report.reported_audio_sec, "unit": "s"},
                "duration_drift_sec": {
                    "value": round(report.duration_drift_sec, 4),
                    "unit": "s",
                    "better": "lower",
                },
                "ttfp_ms": {
                    "value": round(float(report.ttfp_ms or 0.0), 2),
                    "unit": "ms",
                    "better": "lower",
                },
                "final_latency_ms": {
                    "value": round(float(report.final_latency_ms or 0.0), 2),
                    "unit": "ms",
                    "better": "lower",
                },
                "latency_ms": {
                    "value": report.total_latency_ms,
                    "unit": "ms",
                    "better": "lower",
                },
                "rtf": {
                    "value": report.rtf,
                    "unit": "x",
                    "better": "lower",
                },
                "partial_count": {
                    "value": report.partial_count,
                    "unit": "events",
                    "better": "higher",
                },
                "word_count": {
                    "value": report.word_count,
                    "unit": "words",
                    "better": "higher",
                },
                "worker_generations": {
                    "value": report.worker_generations,
                    "unit": "generations",
                },
                "recycled_count": {
                    "value": report.recycled_count,
                    "unit": "rotations",
                },
                "inter_partial_drift_ratio": {
                    "value": report.inter_partial_drift_ratio,
                    "unit": "x",
                    "better": "lower",
                },
                **long_acc_metrics,
            }

            result.add_case(
                case_id=case_long_id,
                suite="grpc_long_call",
                status=long_status,
                metrics=long_metrics,
                detail={
                    "endpoint": "StreamingASR.Transcribe",
                    "grpc_target": harness.grpc_target,
                    "model": m_info.id,
                    "engine": m_info.engine,
                    "fixture_id": long_fix.id,
                    **long_acc_detail,
                    "audio_duration_sec": long_fix.duration_sec,
                    "stability": report.to_dict(),
                    "passed": report.passed,
                    "eval_message": report.summary_message,
                    "checks": report.checks,
                    "errors": report.errors,
                },
            )
            completed_cases += 1
            ctx.log(f"  {case_long_id} -> {long_status.upper()} | {report.summary_message}")
            ctx.progress(completed_cases, planned_cases, f"Completed {case_long_id}")

            # Set top-level long-call metrics on TesterResult
            result.set_metric("grpc_long_call_duration_sec", report.audio_duration_sec, unit="s")
            result.set_metric("grpc_long_call_rtf", report.rtf, unit="x", better="lower")
            result.set_metric("grpc_long_call_latency_ms", report.total_latency_ms, unit="ms", better="lower")
            if report.ttfp_ms is not None:
                result.set_metric("grpc_long_call_ttfp_ms", round(report.ttfp_ms, 2), unit="ms", better="lower")
            if report.final_latency_ms is not None:
                result.set_metric(
                    "grpc_long_call_final_latency_ms",
                    round(report.final_latency_ms, 2),
                    unit="ms",
                    better="lower",
                )
            result.set_metric("grpc_long_call_partial_count", report.partial_count, unit="events", better="higher")
            result.set_metric("grpc_long_call_word_count", report.word_count, unit="words", better="higher")
            result.set_metric("grpc_long_call_worker_generations", report.worker_generations, unit="generations")
            result.set_metric("grpc_long_call_recycled_count", report.recycled_count, unit="rotations")
            result.set_metric(
                "grpc_long_call_inter_partial_drift_ratio",
                report.inter_partial_drift_ratio,
                unit="x",
                better="lower",
            )

    # Set top-level aggregate gRPC streaming metrics on TesterResult
    if recorded_sessions:
        s_summary = compute_grpc_streaming_summary(recorded_sessions)
        result.set_metric("grpc_ttfp_mean_ms", s_summary["ttfp_mean_ms"], unit="ms", better="lower")
        result.set_metric("grpc_ttfp_min_ms", s_summary["ttfp_min_ms"], unit="ms", better="lower")
        result.set_metric("grpc_ttfp_max_ms", s_summary["ttfp_max_ms"], unit="ms", better="lower")
        result.set_metric("grpc_ttfp_p50_ms", s_summary["ttfp_p50_ms"], unit="ms", better="lower")
        result.set_metric("grpc_ttfp_p95_ms", s_summary["ttfp_p95_ms"], unit="ms", better="lower")
        result.set_metric("grpc_ttfp_p99_ms", s_summary["ttfp_p99_ms"], unit="ms", better="lower")
        result.set_metric(
            "grpc_ttfp_from_first_audio_mean_ms",
            s_summary["ttfp_from_first_audio_mean_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_ttfp_from_first_audio_p95_ms",
            s_summary["ttfp_from_first_audio_p95_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric("ttfp_mean_ms", s_summary["ttfp_mean_ms"], unit="ms", better="lower")
        result.set_metric("ttfp_p50_ms", s_summary["ttfp_p50_ms"], unit="ms", better="lower")
        result.set_metric("ttfp_p95_ms", s_summary["ttfp_p95_ms"], unit="ms", better="lower")

        result.set_metric(
            "grpc_final_latency_mean_ms",
            s_summary["final_latency_mean_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_final_latency_min_ms",
            s_summary["final_latency_min_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_final_latency_max_ms",
            s_summary["final_latency_max_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_final_latency_p50_ms",
            s_summary["final_latency_p50_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_final_latency_p95_ms",
            s_summary["final_latency_p95_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_final_latency_p99_ms",
            s_summary["final_latency_p99_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "final_latency_mean_ms",
            s_summary["final_latency_mean_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "final_latency_p50_ms",
            s_summary["final_latency_p50_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "final_latency_p95_ms",
            s_summary["final_latency_p95_ms"],
            unit="ms",
            better="lower",
        )

        result.set_metric(
            "grpc_ready_latency_mean_ms",
            s_summary["ready_latency_mean_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_ready_latency_p95_ms",
            s_summary["ready_latency_p95_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_inter_partial_latency_mean_ms",
            s_summary["inter_partial_latency_mean_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_inter_partial_latency_p95_ms",
            s_summary["inter_partial_latency_p95_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_latency_mean_ms",
            s_summary["latency_mean_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric(
            "grpc_latency_p95_ms",
            s_summary["latency_p95_ms"],
            unit="ms",
            better="lower",
        )
        result.set_metric("grpc_rtf_mean", s_summary["rtf_mean"], unit="x", better="lower")
        result.set_metric("grpc_rtf_p50", s_summary["rtf_p50"], unit="x", better="lower")
        result.set_metric("grpc_rtf_p95", s_summary["rtf_p95"], unit="x", better="lower")
        result.set_metric("grpc_rtf_p99", s_summary["rtf_p99"], unit="x", better="lower")

    return completed_cases
