"""Speed telemetry and statistical metric calculations for the LLM tester."""

from __future__ import annotations

import statistics
from typing import Sequence
import numpy as np


def estimate_tokens(text: str) -> int:
    """Estimate token count from text when API usage statistics are unavailable."""
    if not text or not text.strip():
        return 0
    words = text.strip().split()
    return max(1, int(round(len(words) * 1.33)))


def compute_percentiles(
    values: Sequence[float],
    percentiles: Sequence[float] = (50, 95, 99),
) -> dict[int, float]:
    """Compute percentile values for a sequence of numeric measurements."""
    if not values:
        return {int(p): 0.0 for p in percentiles}
    return {int(p): round(float(np.percentile(values, p)), 2) for p in percentiles}


def compute_speed_summary(
    ttfts_ms: Sequence[float],
    latencies_ms: Sequence[float],
    decode_tps_list: Sequence[float],
    prefill_tps_list: Sequence[float],
) -> dict[str, float]:
    """Compute aggregate speed statistics across streaming chat calls.

    Returns a flat dictionary with TTFT, latency percentiles, and tokens/s stats.
    """
    ttft_pct = compute_percentiles(ttfts_ms)
    lat_pct = compute_percentiles(latencies_ms)
    decode_pct = compute_percentiles(decode_tps_list)

    ttft_mean = round(float(statistics.mean(ttfts_ms)), 2) if ttfts_ms else 0.0
    ttft_min = round(float(min(ttfts_ms)), 2) if ttfts_ms else 0.0
    ttft_max = round(float(max(ttfts_ms)), 2) if ttfts_ms else 0.0

    lat_mean = round(float(statistics.mean(latencies_ms)), 2) if latencies_ms else 0.0
    lat_min = round(float(min(latencies_ms)), 2) if latencies_ms else 0.0
    lat_max = round(float(max(latencies_ms)), 2) if latencies_ms else 0.0

    decode_mean = round(float(statistics.mean(decode_tps_list)), 2) if decode_tps_list else 0.0
    prefill_mean = round(float(statistics.mean(prefill_tps_list)), 2) if prefill_tps_list else 0.0

    return {
        "ttft_mean_ms": ttft_mean,
        "ttft_min_ms": ttft_min,
        "ttft_max_ms": ttft_max,
        "ttft_p50_ms": ttft_pct[50],
        "ttft_p95_ms": ttft_pct[95],
        "ttft_p99_ms": ttft_pct[99],
        "latency_mean_ms": lat_mean,
        "latency_min_ms": lat_min,
        "latency_max_ms": lat_max,
        "latency_p50_ms": lat_pct[50],
        "latency_p95_ms": lat_pct[95],
        "latency_p99_ms": lat_pct[99],
        "decode_tps_mean": decode_mean,
        "decode_tps_p50": decode_pct[50],
        "decode_tps_p95": decode_pct[95],
        "prefill_tps_mean": prefill_mean,
    }
