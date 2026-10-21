"""ASCII report and JSON artifact formatter for ``llm_coding`` runs."""

from __future__ import annotations

from typing import Any


def format_llm_coding_report(
    model: str,
    endpoint_url: str,
    runner_url: str,
    dataset_version: str,
    manifest_sha256: str,
    metrics: dict[str, Any],
    cases: list[dict[str, Any]],
    judge_info: dict[str, Any] | None = None,
    effort: str | None = None,
    tools_enabled: bool = False,
    max_attempts: int = 1,
) -> str:
    """Format a human-readable ASCII summary report (``llm_coding_report.txt``)."""
    score_val = metrics.get("score", {}).get("value", 0.0)
    s1_val = metrics.get("solve_at_1", {}).get("value", 0.0)
    sk_val = metrics.get("solve_at_k", {}).get("value", 0.0)
    mean_att = metrics.get("mean_attempts_to_solve", {}).get("value", 0.0)
    tc_rate = metrics.get("test_case_pass_rate", {}).get("value", 0.0)
    judge_mean = metrics.get("judge_mean", {}).get("value", 0.0)
    tool_mean = metrics.get("tool_calls_mean", {}).get("value", 0.0)
    tps_val = metrics.get("tokens_per_sec", {}).get("value", 0.0)

    lines: list[str] = [
        "=" * 90,
        "LLM CODING BENCHMARK REPORT (iqb-java / Piston)",
        "=" * 90,
        f"Model                 : {model}",
        f"Endpoint URL          : {endpoint_url}",
        f"Piston Runner URL     : {runner_url}",
        f"Dataset Version       : {dataset_version} (sha256: {manifest_sha256[:16]}...)",
        f"Reasoning Effort      : {effort if effort else 'unset'}",
        f"Max Attempts          : {max_attempts} (pass with feedback, not statistical pass@k)",
        f"Agent Tool Mode       : {'Enabled' if tools_enabled else 'Disabled'}",
    ]

    if judge_info and judge_info.get("enabled"):
        lines.extend(
            [
                f"LLM Judge Model       : {judge_info.get('model', 'N/A')}",
                f"LLM Judge Endpoint    : {judge_info.get('endpoint_url', 'N/A')}",
            ]
        )
    else:
        lines.append("LLM Judge             : Disabled")

    lines.extend(
        [
            "-" * 90,
            "Headline Scores & Solve Rates:",
            f"  Composite Score     : {score_val:.2f} / 100.0 pts",
            f"  solve_at_1          : {s1_val:.2f}%",
            f"  solve_at_k          : {sk_val:.2f}% (k={max_attempts}, pass with feedback)",
            f"  Mean Attempts Solved: {mean_att:.2f}",
            f"  Test Case Pass Rate : {tc_rate:.2f}%",
            f"  Judge Mean Score    : {judge_mean:.2f} / 5.0",
            f"  Tool Calls (mean)   : {tool_mean:.2f}",
            f"  Generation Speed    : {tps_val:.2f} tokens/s",
            f"  Total Tokens        : {metrics.get('total_tokens', {}).get('value', 0)} "
            f"(prompt {metrics.get('prompt_tokens_total', {}).get('value', 0)} / "
            f"completion {metrics.get('completion_tokens_total', {}).get('value', 0)})",
            f"  Latency p50 / p95   : {metrics.get('latency_p50_ms', {}).get('value', 0.0):.1f} ms / "
            f"{metrics.get('latency_p95_ms', {}).get('value', 0.0):.1f} ms",
        ]
    )

    diff_parts: list[str] = []
    for diff in ("easy", "medium", "hard"):
        if f"score_{diff}" in metrics:
            ds = metrics[f"score_{diff}"]["value"]
            ds1 = metrics.get(f"solve_at_1_{diff}", {}).get("value", 0.0)
            diff_parts.append(f"{diff.capitalize()}: score={ds:.1f}, solve_at_1={ds1:.1f}%")
    if diff_parts:
        lines.append("  By Difficulty       : " + " | ".join(diff_parts))

    if cases:
        lines.extend(
            [
                "-" * 90,
                f"Problem Breakdown ({len(cases)} problems):",
                f"{'Problem ID':<32} | {'Diff':<6} | {'Verdict':<18} | {'Att':<3} | {'Tests':<7} | {'Judge':<5} | {'Score':<6}",
                "-" * 90,
            ]
        )
        for c in cases:
            cid = str(c.get("case_id", ""))[:32]
            detail = c.get("detail", {})
            diff = str(detail.get("difficulty", ""))[:6]
            verdict = str(detail.get("verdict", c.get("status", "")))[:18]
            att = str(detail.get("attempts_used", 1))
            passed_tc = detail.get("passed_cases", 0)
            total_tc = detail.get("total_cases", 0)
            tc_str = f"{passed_tc}/{total_tc}"
            judge_out = detail.get("judge")
            j_str = (
                f"{judge_out.get('overall', 0):.1f}"
                if isinstance(judge_out, dict) and judge_out.get("overall") is not None
                else "-"
            )
            c_metrics = c.get("metrics", {})
            c_score = float(c_metrics.get("case_score", {}).get("value", 0.0))
            lines.append(
                f"{cid:<32} | {diff:<6} | {verdict:<18} | {att:<3} | {tc_str:<7} | {j_str:<5} | {c_score:<6.2f}"
            )

    lines.append("=" * 90)
    return "\n".join(lines) + "\n"
