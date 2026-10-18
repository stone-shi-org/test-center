"""Formatted ASCII report generator for LLM chat benchmark and capability results."""

from __future__ import annotations

from typing import Any


def format_llm_report(
    model: str,
    base_url: str,
    metrics: dict[str, Any],
    cases: list[dict[str, Any]],
    judge_info: dict[str, Any] | None = None,
    effort: str | None = None,
) -> str:
    """Format a human-readable ASCII summary report for an LLM test run."""
    lines: list[str] = [
        "=" * 78,
        "LLM CHAT BENCHMARK & CAPABILITY REPORT",
        "=" * 78,
        f"Model                 : {model}",
        f"Endpoint URL          : {base_url}",
        f"Reasoning Effort      : {effort if effort else 'unset'}",
    ]

    if judge_info and judge_info.get("enabled"):
        lines.extend([
            f"LLM Judge Model       : {judge_info.get('model', 'N/A')}",
            f"LLM Judge Endpoint    : {judge_info.get('endpoint_url', 'N/A')}",
        ])
    else:
        lines.append("LLM Judge             : Disabled")

    lines.extend([
        "-" * 78,
        "Speed & Streaming Telemetry:",
        f"  TTFT (mean)         : {metrics.get('ttft_mean_ms', {}).get('value', 0.0):.2f} ms "
        f"(min {metrics.get('ttft_min_ms', {}).get('value', 0.0):.2f} / "
        f"max {metrics.get('ttft_max_ms', {}).get('value', 0.0):.2f} ms)",
        f"  TTFT p50/p95/p99    : {metrics.get('ttft_p50_ms', {}).get('value', 0.0):.2f} / "
        f"{metrics.get('ttft_p95_ms', {}).get('value', 0.0):.2f} / "
        f"{metrics.get('ttft_p99_ms', {}).get('value', 0.0):.2f} ms",
        f"  Latency (mean)      : {metrics.get('latency_mean_ms', {}).get('value', 0.0):.2f} ms "
        f"(min {metrics.get('latency_min_ms', {}).get('value', 0.0):.2f} / "
        f"max {metrics.get('latency_max_ms', {}).get('value', 0.0):.2f} ms)",
        f"  Latency p50/p95/p99 : {metrics.get('latency_p50_ms', {}).get('value', 0.0):.2f} / "
        f"{metrics.get('latency_p95_ms', {}).get('value', 0.0):.2f} / "
        f"{metrics.get('latency_p99_ms', {}).get('value', 0.0):.2f} ms",
        f"  Decode Speed (mean) : {metrics.get('tokens_per_sec', {}).get('value', 0.0):.2f} tokens/s "
        f"(p50 {metrics.get('decode_tps_p50', {}).get('value', 0.0):.2f} / "
        f"p95 {metrics.get('decode_tps_p95', {}).get('value', 0.0):.2f})",
        f"  Prefill Speed (mean): {metrics.get('prefill_tps_mean', {}).get('value', 0.0):.2f} tokens/s",
        f"  Aggregate Throughput: {metrics.get('aggregate_tps', {}).get('value', 0.0):.2f} tokens/s",
        "-" * 78,
        "Token Usage & Capability Summary:",
        f"  Prompt / Completion : {metrics.get('prompt_tokens', {}).get('value', 0)} / "
        f"{metrics.get('completion_tokens', {}).get('value', 0)} "
        f"(total {metrics.get('total_tokens', {}).get('value', 0)})",
        f"  Cases Passed        : {metrics.get('cases_passed', {}).get('value', 0)} / "
        f"{metrics.get('cases_total', {}).get('value', 0)} "
        f"({metrics.get('pass_rate', {}).get('value', 0.0):.1f}%)",
    ])

    if "judge_score_mean" in metrics:
        lines.append(
            f"  Judge Mean Score    : {metrics.get('judge_score_mean', {}).get('value', 0.0):.2f} / 5.0 "
            f"(Pass Rate: {metrics.get('judge_pass_rate', {}).get('value', 0.0):.1f}%)"
        )

    if cases:
        lines.extend([
            "-" * 78,
            f"Capability Cases Breakdown ({len(cases)} cases):",
            f"{'Case ID':<22} | {'Suite':<13} | {'Status':<6} | {'TTFT ms':<8} | {'TPS':<7} | {'Judge':<5} | Detail",
            "-" * 78,
        ])
        for c in cases:
            cid = str(c.get("case_id", ""))
            suite = str(c.get("suite", ""))
            status = str(c.get("status", ""))
            m = c.get("metrics", {})
            ttft = float(m.get("ttft_ms", {}).get("value", 0.0))
            tps = float(m.get("tokens_per_sec", {}).get("value", 0.0))
            j_score = m.get("judge_score", {}).get("value")
            j_str = f"{j_score}/5" if j_score is not None else "-"
            detail = c.get("detail", {})
            eval_msg = str(detail.get("eval_message", ""))
            short_msg = (eval_msg[:36] + "...") if len(eval_msg) > 36 else eval_msg
            lines.append(
                f"{cid:<22} | {suite:<13} | {status:<6} | {ttft:<8.2f} | {tps:<7.2f} | {j_str:<5} | {short_msg}"
            )

    lines.append("=" * 78)
    return "\n".join(lines) + "\n"
