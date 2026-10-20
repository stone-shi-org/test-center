"""Per-problem and run-level composite scoring for ``llm_coding``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DEFAULT_ATTEMPT_FACTORS: list[float] = [1.0, 0.7, 0.5]
DEFAULT_WEIGHT_CORRECTNESS: float = 0.8
DEFAULT_WEIGHT_JUDGE: float = 0.2


def parse_attempt_factors(raw: Any) -> list[float]:
    """Parse ``attempt_factors`` from config (accepts list of floats or comma-separated string)."""
    if raw is None:
        return list(DEFAULT_ATTEMPT_FACTORS)
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        parsed: list[float] = []
        for p in parts:
            try:
                parsed.append(max(0.0, min(1.0, float(p))))
            except ValueError:
                pass
        return parsed or list(DEFAULT_ATTEMPT_FACTORS)
    if isinstance(raw, (list, tuple)):
        parsed = []
        for item in raw:
            try:
                parsed.append(max(0.0, min(1.0, float(item))))
            except (TypeError, ValueError):
                pass
        return parsed or list(DEFAULT_ATTEMPT_FACTORS)
    return list(DEFAULT_ATTEMPT_FACTORS)


def get_attempt_factor(
    attempt_number: int | None,
    attempt_factors: list[float] | None = None,
) -> float:
    """Return the attempt multiplier for a 1-based ``attempt_number``."""
    if attempt_number is None or attempt_number <= 0:
        return 0.0
    factors = attempt_factors or DEFAULT_ATTEMPT_FACTORS
    if not factors:
        return 1.0
    idx = min(attempt_number - 1, len(factors) - 1)
    return float(factors[idx])


@dataclass
class CaseScoreBreakdown:
    correct: int
    solved_attempt: int | None
    attempt_factor: float
    judge_overall: float | None
    judge_norm: float | None
    case_score: float  # 0.0 .. 1.0
    case_score_100: float  # 0.0 .. 100.0


def compute_case_score(
    correct: int | bool,
    solved_attempt: int | None,
    attempt_factors: list[float] | None = None,
    judge_enabled: bool = False,
    judge_overall: float | None = None,
    weight_correctness: float = DEFAULT_WEIGHT_CORRECTNESS,
    weight_judge: float = DEFAULT_WEIGHT_JUDGE,
) -> CaseScoreBreakdown:
    """Compute a problem's composite ``case_score`` in ``[0.0, 1.0]``.

    Formula (TES-46 Section 7):
    - ``correct`` = 1 if all hidden tests pass, else 0.
    - ``attempt_factor`` = 1.0 / 0.7 / 0.5 for success on attempt 1 / 2 / 3.
    - Without judge: ``case_score = correct * attempt_factor``
    - With judge: ``case_score = correct * attempt_factor * (w_c + w_j * judge_norm)``
      where ``judge_norm = (overall - 1) / 4``.
    """
    is_correct = 1 if bool(correct) else 0
    if not is_correct:
        judge_norm = (
            max(0.0, min(1.0, (float(judge_overall) - 1.0) / 4.0))
            if judge_overall is not None
            else None
        )
        return CaseScoreBreakdown(
            correct=0,
            solved_attempt=None,
            attempt_factor=0.0,
            judge_overall=float(judge_overall) if judge_overall is not None else None,
            judge_norm=judge_norm,
            case_score=0.0,
            case_score_100=0.0,
        )

    eff_attempt = solved_attempt if (solved_attempt is not None and solved_attempt >= 1) else 1
    factor = get_attempt_factor(eff_attempt, attempt_factors)

    if judge_enabled and judge_overall is not None:
        j_overall = max(1.0, min(5.0, float(judge_overall)))
        j_norm = max(0.0, min(1.0, (j_overall - 1.0) / 4.0))
        w_c = float(weight_correctness)
        w_j = float(weight_judge)
        raw_score = float(is_correct) * factor * (w_c + w_j * j_norm)
        case_score = max(0.0, min(1.0, raw_score))
        return CaseScoreBreakdown(
            correct=1,
            solved_attempt=eff_attempt,
            attempt_factor=factor,
            judge_overall=j_overall,
            judge_norm=j_norm,
            case_score=round(case_score, 6),
            case_score_100=round(case_score * 100.0, 2),
        )

    case_score = max(0.0, min(1.0, float(is_correct) * factor))
    return CaseScoreBreakdown(
        correct=1,
        solved_attempt=eff_attempt,
        attempt_factor=factor,
        judge_overall=None,
        judge_norm=None,
        case_score=round(case_score, 6),
        case_score_100=round(case_score * 100.0, 2),
    )


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def aggregate_run_metrics(
    case_records: list[dict[str, Any]],
    judge_enabled: bool = False,
) -> dict[str, dict[str, Any]]:
    """Aggregate per-problem records into run-level metrics with ``unit`` and ``better`` metadata."""
    n = len(case_records)
    if n == 0:
        return {
            "score": {"value": 0.0, "unit": "pts", "better": "higher"},
            "solve_at_1": {"value": 0.0, "unit": "%", "better": "higher"},
            "solve_at_k": {"value": 0.0, "unit": "%", "better": "higher"},
            "mean_attempts_to_solve": {"value": 0.0, "unit": "attempts", "better": "lower"},
            "test_case_pass_rate": {"value": 0.0, "unit": "%", "better": "higher"},
        }

    mean_score_100 = (sum(float(r.get("case_score", 0.0)) for r in case_records) / n) * 100.0
    solved_at_1_count = sum(
        1 for r in case_records if r.get("correct") == 1 and r.get("solved_attempt") == 1
    )
    solved_at_k_count = sum(1 for r in case_records if r.get("correct") == 1)

    solve_at_1_pct = (solved_at_1_count / n) * 100.0
    solve_at_k_pct = (solved_at_k_count / n) * 100.0

    solved_attempts = [
        int(r["solved_attempt"])
        for r in case_records
        if r.get("correct") == 1 and r.get("solved_attempt") is not None
    ]
    mean_attempts = (
        sum(solved_attempts) / len(solved_attempts) if solved_attempts else 0.0
    )

    mean_tc_pass_rate_pct = (
        sum(float(r.get("test_case_pass_rate", 0.0)) for r in case_records) / n
    ) * 100.0

    judge_scores = [
        float(r["judge_overall"])
        for r in case_records
        if r.get("judge_overall") is not None
    ]
    judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    prompt_tokens_total = sum(int(r.get("prompt_tokens", 0)) for r in case_records)
    completion_tokens_total = sum(int(r.get("completion_tokens", 0)) for r in case_records)
    total_tokens = prompt_tokens_total + completion_tokens_total

    wall_times = sorted(float(r.get("wall_time_ms", 0.0)) for r in case_records)
    wall_mean = sum(wall_times) / n
    p50_ms = _percentile(wall_times, 50.0)
    p95_ms = _percentile(wall_times, 95.0)

    tool_calls_mean = sum(int(r.get("tool_call_count", 0)) for r in case_records) / n

    metrics: dict[str, dict[str, Any]] = {
        "score": {"value": round(mean_score_100, 2), "unit": "pts", "better": "higher"},
        "solve_at_1": {"value": round(solve_at_1_pct, 2), "unit": "%", "better": "higher"},
        "solve_at_k": {"value": round(solve_at_k_pct, 2), "unit": "%", "better": "higher"},
        "mean_attempts_to_solve": {
            "value": round(mean_attempts, 2),
            "unit": "attempts",
            "better": "lower",
        },
        "test_case_pass_rate": {
            "value": round(mean_tc_pass_rate_pct, 2),
            "unit": "%",
            "better": "higher",
        },
        "judge_mean": {"value": round(judge_mean, 2), "unit": "/5", "better": "higher"},
        "tool_calls_mean": {
            "value": round(tool_calls_mean, 2),
            "unit": "calls",
            "better": "lower",
        },
        "prompt_tokens_total": {
            "value": prompt_tokens_total,
            "unit": "tokens",
            "better": "lower",
        },
        "completion_tokens_total": {
            "value": completion_tokens_total,
            "unit": "tokens",
            "better": "lower",
        },
        "total_tokens": {
            "value": total_tokens,
            "unit": "tokens",
            "better": "lower",
        },
        "latency_p50_ms": {"value": round(p50_ms, 1), "unit": "ms", "better": "lower"},
        "latency_p95_ms": {"value": round(p95_ms, 1), "unit": "ms", "better": "lower"},
        "wall_time_mean_ms": {"value": round(wall_mean, 1), "unit": "ms", "better": "lower"},
    }

    # Per-difficulty score and solve_at_1 breakdown
    for diff in ("easy", "medium", "hard"):
        diff_cases = [
            r for r in case_records if str(r.get("difficulty") or "").lower() == diff
        ]
        if diff_cases:
            dn = len(diff_cases)
            d_score = (sum(float(r.get("case_score", 0.0)) for r in diff_cases) / dn) * 100.0
            d_s1 = (
                sum(
                    1
                    for r in diff_cases
                    if r.get("correct") == 1 and r.get("solved_attempt") == 1
                )
                / dn
            ) * 100.0
            metrics[f"score_{diff}"] = {
                "value": round(d_score, 2),
                "unit": "pts",
                "better": "higher",
            }
            metrics[f"solve_at_1_{diff}"] = {
                "value": round(d_s1, 2),
                "unit": "%",
                "better": "higher",
            }

    return metrics
