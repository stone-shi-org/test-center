"""Latency, percentile, and Real-Time Factor (RTF) metric helpers for live-stt HTTP tester."""

from __future__ import annotations

import statistics
from typing import Sequence
import numpy as np


def compute_rtf(latency_ms: float, audio_duration_sec: float) -> float:
    """Compute Real-Time Factor (RTF = processing_time_sec / audio_duration_sec).

    Lower is better (< 1.0 means faster than real-time).
    """
    if audio_duration_sec <= 0:
        return 0.0
    processing_sec = max(0.0, latency_ms / 1000.0)
    return round(processing_sec / audio_duration_sec, 4)


def compute_percentiles(
    values: Sequence[float],
    percentiles: Sequence[float] = (50, 95, 99),
) -> dict[int, float]:
    """Compute percentile values for a sequence of numeric measurements."""
    if not values:
        return {int(p): 0.0 for p in percentiles}
    return {int(p): round(float(np.percentile(values, p)), 2) for p in percentiles}


def compute_latency_and_rtf_summary(
    latencies_ms: Sequence[float],
    rtfs: Sequence[float],
) -> dict[str, float]:
    """Compute aggregate statistics for latency (ms) and real-time factor (RTF)."""
    lat_pct = compute_percentiles(latencies_ms, (50, 95, 99))
    rtf_pct = (
        {int(p): round(float(np.percentile(rtfs, p)), 4) for p in (50, 95, 99)}
        if rtfs
        else {50: 0.0, 95: 0.0, 99: 0.0}
    )

    lat_mean = round(float(statistics.mean(latencies_ms)), 2) if latencies_ms else 0.0
    lat_min = round(float(min(latencies_ms)), 2) if latencies_ms else 0.0
    lat_max = round(float(max(latencies_ms)), 2) if latencies_ms else 0.0

    rtf_mean = round(float(statistics.mean(rtfs)), 4) if rtfs else 0.0
    rtf_min = round(float(min(rtfs)), 4) if rtfs else 0.0
    rtf_max = round(float(max(rtfs)), 4) if rtfs else 0.0

    return {
        "latency_mean_ms": lat_mean,
        "latency_min_ms": lat_min,
        "latency_max_ms": lat_max,
        "latency_p50_ms": lat_pct[50],
        "latency_p95_ms": lat_pct[95],
        "latency_p99_ms": lat_pct[99],
        "rtf_mean": rtf_mean,
        "rtf_min": rtf_min,
        "rtf_max": rtf_max,
        "rtf_p50": rtf_pct[50],
        "rtf_p95": rtf_pct[95],
        "rtf_p99": rtf_pct[99],
    }
