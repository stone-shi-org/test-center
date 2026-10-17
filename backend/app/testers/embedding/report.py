"""Formatted ASCII report generator for embedding test results."""

from __future__ import annotations

from typing import Any


def format_embedding_report(
    model: str,
    base_url: str,
    n_docs: int,
    n_queries: int,
    dim: int,
    metrics: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    """Format a detailed human-readable ASCII report for embedding test run."""
    lines: list[str] = [
        "=" * 72,
        "EMBEDDING BENCHMARK REPORT",
        "=" * 72,
        f"Model                 : {model}",
        f"Base URL              : {base_url}",
        f"Embedding Dimension   : {dim}",
        f"Corpus Documents      : {n_docs}",
        f"Retrieval Queries     : {n_queries}",
        "-" * 72,
        "Latency (ms):",
        f"  doc embed (total)   : {metrics.get('doc_embed_total_ms', {}).get('value', 0.0):.2f}",
        f"  doc embed (mean)    : {metrics.get('latency_mean_ms', {}).get('value', 0.0):.2f} "
        f"(min {metrics.get('latency_min_ms', {}).get('value', 0.0):.2f} / "
        f"max {metrics.get('latency_max_ms', {}).get('value', 0.0):.2f})",
        f"  doc embed p50/p95/99: {metrics.get('latency_p50_ms', {}).get('value', 0.0):.2f} / "
        f"{metrics.get('latency_p95_ms', {}).get('value', 0.0):.2f} / "
        f"{metrics.get('latency_p99_ms', {}).get('value', 0.0):.2f}",
    ]

    if "query_latency_mean_ms" in metrics:
        lines.extend([
            f"  query embed (mean)  : {metrics.get('query_latency_mean_ms', {}).get('value', 0.0):.2f}",
            f"  query p50/p95/p99   : {metrics.get('query_latency_p50_ms', {}).get('value', 0.0):.2f} / "
            f"{metrics.get('query_latency_p95_ms', {}).get('value', 0.0):.2f} / "
            f"{metrics.get('query_latency_p99_ms', {}).get('value', 0.0):.2f}",
        ])
    if "search_total_ms" in metrics:
        lines.append(f"  cosine search (tot) : {metrics.get('search_total_ms', {}).get('value', 0.0):.2f}")

    lines.extend([
        "-" * 72,
        "Throughput:",
        f"  sequential docs/sec : {metrics.get('throughput_seq_docs_per_sec', {}).get('value', 0.0):.2f}",
    ])

    if "throughput_batched_docs_per_sec" in metrics:
        lines.append(
            f"  batched docs/sec    : {metrics.get('throughput_batched_docs_per_sec', {}).get('value', 0.0):.2f} "
            f"(single call, {metrics.get('batched_latency_ms', {}).get('value', 0.0):.2f} ms)"
        )
    if "throughput_concurrent_docs_per_sec" in metrics:
        lines.append(
            f"  concurrent docs/sec : {metrics.get('throughput_concurrent_docs_per_sec', {}).get('value', 0.0):.2f} "
            f"({metrics.get('concurrent_wall_ms', {}).get('value', 0.0):.2f} ms wall)"
        )

    lines.extend([
        "-" * 72,
        "Token Usage:",
        f"  prompt tokens       : {metrics.get('prompt_tokens', {}).get('value', 0)}",
        f"  total tokens        : {metrics.get('total_tokens', {}).get('value', 0)}",
    ])

    if "consistency_l2_max" in metrics:
        lines.extend([
            "-" * 72,
            "Consistency (re-embed same text, L2 distance should be ~0):",
            f"  max L2 distance     : {metrics.get('consistency_l2_max', {}).get('value', 0.0):.6f}",
        ])

    if "norm_mean" in metrics:
        lines.extend([
            "-" * 72,
            "Embedding Space:",
            f"  norm mean / std     : {metrics.get('norm_mean', {}).get('value', 0.0):.4f} / "
            f"{metrics.get('norm_std', {}).get('value', 0.0):.4f}",
            f"  norm min / max      : {metrics.get('norm_min', {}).get('value', 0.0):.4f} / "
            f"{metrics.get('norm_max', {}).get('value', 0.0):.4f}",
            f"  mean pairwise sim   : {metrics.get('mean_pairwise_similarity', {}).get('value', 0.0):.4f} "
            "(lower = more discriminative embeddings)",
        ])

    if "recall_at_1" in metrics:
        lines.extend([
            "-" * 72,
            "Retrieval Quality:",
            f"  Recall@1            : {metrics.get('recall_at_1', {}).get('value', 0.0):.2f}%",
            f"  Recall@3            : {metrics.get('recall_at_3', {}).get('value', 0.0):.2f}%",
            f"  Recall@5            : {metrics.get('recall_at_5', {}).get('value', 0.0):.2f}%",
            f"  MRR                 : {metrics.get('mrr', {}).get('value', 0.0):.4f}",
            f"  mean rank of expect : {metrics.get('mean_rank', {}).get('value', 0.0):.2f}",
            f"  mean top similarity : {metrics.get('mean_top_similarity', {}).get('value', 0.0):.4f}",
        ])

    if cases:
        lines.extend([
            "-" * 72,
            f"Retrieval Query Breakdown ({len(cases)} cases):",
            f"{'Case ID':<22} | {'Status':<6} | {'Rank':<4} | {'Top Sim':<7} | Query Text",
            "-" * 72,
        ])
        for c in cases:
            case_id = c.get("case_id", "")
            status = c.get("status", "")
            m = c.get("metrics", {})
            rank = m.get("rank", {}).get("value", "-")
            top_sim = m.get("top_similarity", {}).get("value", 0.0)
            detail = c.get("detail", {})
            query_preview = (detail.get("query", "")[:35] + "...") if len(detail.get("query", "")) > 35 else detail.get("query", "")
            lines.append(f"{case_id:<22} | {status:<6} | {rank:<4} | {top_sim:<7.3f} | {query_preview}")

    lines.append("=" * 72)
    return "\n".join(lines) + "\n"
