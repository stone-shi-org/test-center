"""Formatted ASCII report generator for live-stt HTTP functional and latency test runs."""

from __future__ import annotations

from typing import Any


def format_live_stt_report(
    base_url: str,
    models_summary: dict[str, Any],
    metrics: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    """Format a human-readable ASCII summary report for a live-stt HTTP test run."""
    lines: list[str] = [
        "=" * 84,
        "LIVE-STT HTTP ENDPOINT FUNCTIONAL & LATENCY REPORT",
        "=" * 84,
        f"Base URL              : {base_url}",
        f"Models Discovered     : {models_summary.get('discovered_count', 0)} "
        f"(Parametrization: {models_summary.get('parametrization_mode', 'per_engine')})",
        f"ASR Models Tested     : {', '.join(models_summary.get('asr_models', [])) or 'None'}",
        f"Diarization Models    : {', '.join(models_summary.get('diarization_models', [])) or 'None'}",
        f"TTS Models Tested     : {', '.join(models_summary.get('tts_models', [])) or 'None'}",
        "-" * 84,
        "Aggregate Latency & Real-Time Factor (RTF):",
        f"  Latency (mean)      : {metrics.get('latency_mean_ms', {}).get('value', 0.0):.2f} ms "
        f"(min {metrics.get('latency_min_ms', {}).get('value', 0.0):.2f} / "
        f"max {metrics.get('latency_max_ms', {}).get('value', 0.0):.2f} ms)",
        f"  Latency p50/p95/p99 : {metrics.get('latency_p50_ms', {}).get('value', 0.0):.2f} / "
        f"{metrics.get('latency_p95_ms', {}).get('value', 0.0):.2f} / "
        f"{metrics.get('latency_p99_ms', {}).get('value', 0.0):.2f} ms",
        f"  RTF (mean)          : {metrics.get('rtf_mean', {}).get('value', 0.0):.4f}x "
        f"(p50 {metrics.get('rtf_p50', {}).get('value', 0.0):.4f}x / "
        f"p95 {metrics.get('rtf_p95', {}).get('value', 0.0):.4f}x)",
        f"  Cases Passed        : {metrics.get('cases_passed', {}).get('value', 0)} / "
        f"{metrics.get('cases_total', {}).get('value', 0)} "
        f"({metrics.get('pass_rate', {}).get('value', 0.0):.1f}%)",
    ]

    # Per-endpoint section
    endpoint_lines: list[str] = []
    if "transcription_json_latency_mean_ms" in metrics:
        endpoint_lines.append(
            f"  Transcriptions JSON : {metrics['transcription_json_latency_mean_ms']['value']:.2f} ms "
            f"(RTF {metrics.get('transcription_json_rtf_mean', {}).get('value', 0.0):.4f}x)"
        )
    if "transcription_sse_latency_mean_ms" in metrics:
        endpoint_lines.append(
            f"  Transcriptions SSE  : {metrics['transcription_sse_latency_mean_ms']['value']:.2f} ms "
            f"(TTFT {metrics.get('transcription_sse_ttft_mean_ms', {}).get('value', 0.0):.2f} ms, "
            f"RTF {metrics.get('transcription_sse_rtf_mean', {}).get('value', 0.0):.4f}x)"
        )
    if "diarization_latency_mean_ms" in metrics:
        endpoint_lines.append(
            f"  Diarization         : {metrics['diarization_latency_mean_ms']['value']:.2f} ms "
            f"(RTF {metrics.get('diarization_rtf_mean', {}).get('value', 0.0):.4f}x)"
        )
    if "tts_latency_mean_ms" in metrics:
        endpoint_lines.append(
            f"  TTS Speech          : {metrics['tts_latency_mean_ms']['value']:.2f} ms "
            f"(TTFB {metrics.get('tts_ttfb_mean_ms', {}).get('value', 0.0):.2f} ms, "
            f"RTF {metrics.get('tts_rtf_mean', {}).get('value', 0.0):.4f}x)"
        )
    if "admin_latency_mean_ms" in metrics:
        endpoint_lines.append(
            f"  Admin / Contract    : {metrics['admin_latency_mean_ms']['value']:.2f} ms "
            f"(p95 {metrics.get('admin_latency_p95_ms', {}).get('value', 0.0):.2f} ms, read-only)"
        )
    if "error_handling_latency_mean_ms" in metrics:
        endpoint_lines.append(
            f"  Error Handling      : {metrics['error_handling_latency_mean_ms']['value']:.2f} ms "
            f"(p95 {metrics.get('error_handling_latency_p95_ms', {}).get('value', 0.0):.2f} ms)"
        )
    if "concurrency_latency_mean_ms" in metrics:
        endpoint_lines.append(
            f"  Concurrency Limits  : {metrics['concurrency_latency_mean_ms']['value']:.2f} ms "
            f"(p95 {metrics.get('concurrency_latency_p95_ms', {}).get('value', 0.0):.2f} ms)"
        )
    if "grpc_ttfp_mean_ms" in metrics:
        endpoint_lines.append(
            f"  gRPC Transcribe     : TTFP {metrics['grpc_ttfp_mean_ms']['value']:.2f} ms "
            f"(p95 {metrics.get('grpc_ttfp_p95_ms', {}).get('value', 0.0):.2f} ms), "
            f"Final {metrics.get('grpc_final_latency_mean_ms', {}).get('value', 0.0):.2f} ms, "
            f"RTF {metrics.get('grpc_rtf_mean', {}).get('value', 0.0):.4f}x"
        )
    if "grpc_long_call_duration_sec" in metrics:
        endpoint_lines.append(
            f"  gRPC Long-Call      : {metrics['grpc_long_call_duration_sec']['value']:.1f} s audio, "
            f"RTF {metrics.get('grpc_long_call_rtf', {}).get('value', 0.0):.4f}x, "
            f"gens={int(metrics.get('grpc_long_call_worker_generations', {}).get('value', 1))}, "
            f"recycled={int(metrics.get('grpc_long_call_recycled_count', {}).get('value', 0))}"
        )
    if "wer" in metrics:
        endpoint_lines.append(
            f"  NOTSOFAR-1 WER      : {metrics['wer']['value'] * 100.0:.2f}% "
            f"(ratio {metrics['wer']['value']:.4f}, mean {metrics.get('wer_mean', {}).get('value', 0.0) * 100.0:.2f}%)"
        )
    if "http_wer" in metrics:
        endpoint_lines.append(
            f"  Batch HTTP WER      : {metrics['http_wer']['value'] * 100.0:.2f}% "
            f"(ratio {metrics['http_wer']['value']:.4f})"
        )
    if "streaming_wer" in metrics:
        endpoint_lines.append(
            f"  Streaming WER       : {metrics['streaming_wer']['value'] * 100.0:.2f}% "
            f"(ratio {metrics['streaming_wer']['value']:.4f})"
        )
    if "cpwer" in metrics:
        endpoint_lines.append(
            f"  Speaker cpWER       : {metrics['cpwer']['value'] * 100.0:.2f}% "
            f"(ratio {metrics['cpwer']['value']:.4f})"
        )
    if "der" in metrics:
        endpoint_lines.append(
            f"  Diarization DER     : {metrics['der']['value'] * 100.0:.2f}% "
            f"(ratio {metrics['der']['value']:.4f}, "
            f"spk_err {metrics.get('speaker_count_error', {}).get('value', 0.0):.2f})"
        )
    if "accuracy_cache_hits" in metrics or "accuracy_cache_misses" in metrics:
        endpoint_lines.append(
            f"  Accuracy Eval Cache : {int(metrics.get('accuracy_cache_hits', {}).get('value', 0))} hits / "
            f"{int(metrics.get('accuracy_cache_misses', {}).get('value', 0))} misses"
        )

    if endpoint_lines:
        lines.append("-" * 84)
        lines.append("Per-Endpoint Telemetry:")
        lines.extend(endpoint_lines)

    if cases:
        lines.extend([
            "-" * 84,
            f"Cases Breakdown ({len(cases)} cases):",
            f"{'Case ID':<38} | {'Suite':<18} | {'Status':<6} | {'Lat ms':<8} | {'RTF':<8} | Detail",
            "-" * 84,
        ])
        for c in cases:
            cid = str(c.get("case_id", ""))
            if len(cid) > 38:
                cid = cid[:35] + "..."
            suite = str(c.get("suite", ""))
            status = str(c.get("status", ""))
            m = c.get("metrics", {})
            lat = float(m.get("latency_ms", {}).get("value", 0.0))
            rtf = float(m.get("rtf", {}).get("value", 0.0))
            detail = c.get("detail", {})
            eval_msg = str(detail.get("eval_message", ""))
            short_msg = (eval_msg[:32] + "...") if len(eval_msg) > 32 else eval_msg
            lines.append(
                f"{cid:<38} | {suite:<18} | {status:<6} | {lat:<8.2f} | {rtf:<8.4f} | {short_msg}"
            )

    lines.append("=" * 84)
    return "\n".join(lines) + "\n"
