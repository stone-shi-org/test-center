"""OpenAI-compatible embedding tester plugin implementation."""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from backend.app.testers.base import (
    TesterPlugin,
    TestContext,
    TesterResult,
    TesterConfigError,
    TesterError,
)
from backend.app.testers.embedding.corpus import (
    get_default_corpus,
    DocumentItem,
    QueryItem,
)
from backend.app.testers.embedding.client import (
    OpenAIEmbeddingClient,
    EmbeddingClientError,
)
from backend.app.testers.embedding.metrics import (
    compute_vector_norms,
    compute_anisotropy,
    compute_l2_distance,
    compute_percentiles,
    compute_retrieval_metrics,
)
from backend.app.testers.embedding.report import format_embedding_report


class EmbeddingTester(TesterPlugin):
    """Tester benchmarking OpenAI-compatible embedding endpoints.

    Evaluates:
    - Latency: p50, p95, p99, mean, min, max for document and query embedding.
    - Throughput: sequential, batched, and concurrent requests per second.
    - Consistency: L2 distance upon re-embedding identical text.
    - Vector Space Shape: norm mean/std/min/max and mean pairwise cosine similarity (anisotropy).
    - Retrieval Quality: top-k Recall@1/3/5, MRR, and per-query case evaluations.
    """

    name = "embedding"
    description = (
        "Benchmarks OpenAI-compatible embedding endpoints for latency (p50/p95/p99), "
        "sequential/batched/concurrent throughput, consistency, vector norms, and Recall@k/MRR."
    )
    suites = ["latency", "throughput", "consistency", "retrieval"]

    config_schema = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Embedding model identifier (e.g. text-embedding-3-small)",
            },
            "suites": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["latency", "throughput", "consistency", "retrieval"],
                },
                "default": ["latency", "throughput", "consistency", "retrieval"],
                "description": "Selected sub-suites to execute",
            },
            "concurrency": {
                "type": "integer",
                "default": 8,
                "minimum": 1,
                "maximum": 64,
                "description": "Worker count for concurrent throughput benchmark",
            },
            "batch_size": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Batch size for batched embedding (0 = full corpus in single request)",
            },
            "timeout_seconds": {
                "type": "number",
                "default": 60.0,
                "minimum": 1.0,
                "description": "HTTP request timeout in seconds",
            },
            "consistency_samples": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 10,
                "description": "Number of sample documents to re-embed for consistency check",
            },
            "pass_rank_threshold": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 30,
                "description": "Expected rank threshold considered a 'pass' for individual query cases",
            },
            "dimensions": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional dimensionality parameter requested from embedding API",
            },
        },
    }

    def _resolve_target_and_client(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
    ) -> tuple[OpenAIEmbeddingClient, str, int | None]:
        """Resolve endpoint URL, auth credentials, model name, and client instance."""
        base_url = target.get("base_url")
        if not base_url:
            raise TesterConfigError("Target definition is missing required 'base_url'")

        # Resolve API key from target auth_env_var or environment
        auth_env_var = target.get("auth_env_var")
        api_key: str | None = None
        if auth_env_var:
            api_key = os.environ.get(auth_env_var)
        if not api_key:
            api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

        # Resolve model name: config override > target extra > default
        model = (
            config.get("model")
            or target.get("extra", {}).get("model")
            or target.get("model")
            or "text-embedding-3-small"
        )

        timeout = float(config.get("timeout_seconds", 60.0))
        transport = config.get("_transport")  # For testing mock transport injection
        dimensions = config.get("dimensions")

        client = OpenAIEmbeddingClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )

        return client, model, dimensions

    def run(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
        ctx: TestContext,
    ) -> TesterResult:
        cfg = self.validate_and_apply_config(config)
        client, model, dimensions = self._resolve_target_and_client(cfg, target)

        # Determine suites
        selected_suites = cfg.get("suites") or target.get("suites") or self.suites
        suites_to_run = [s for s in self.suites if s in selected_suites]
        if not suites_to_run:
            suites_to_run = list(self.suites)

        concurrency = int(cfg.get("concurrency", 8))
        consistency_samples = int(cfg.get("consistency_samples", 3))
        pass_rank_threshold = int(cfg.get("pass_rank_threshold", 1))

        # Built-in evaluation corpus
        docs, queries = get_default_corpus()
        n_docs = len(docs)
        n_queries = len(queries)

        ctx.log(
            f"Starting embedding benchmark for target='{target.get('name', 'default')}', "
            f"model='{model}', suites={suites_to_run}"
        )
        ctx.log(f"Endpoint: {client.endpoint_url}, Corpus: {n_docs} documents, {n_queries} queries")

        result = TesterResult(status="pass")
        total_prompt_tokens = 0
        total_tokens = 0

        # Phase 1: Sequential Document Embeddings
        # (Needed for latency, consistency, norms, and retrieval suites)
        ctx.log(f"Phase 1: Embedding {n_docs} corpus documents sequentially...")
        doc_latencies: list[float] = []
        doc_vectors: list[np.ndarray] = []

        for i, doc in enumerate(docs):
            ctx.check_cancelled()
            try:
                vecs, usage, dt = client.embed([doc.text], model=model, dimensions=dimensions)
            except EmbeddingClientError as e:
                ctx.log(f"Error embedding document {doc.id}: {e}", level="error")
                raise TesterError(f"Embedding API error: {e}") from e

            doc_latencies.append(dt)
            total_prompt_tokens += usage["prompt_tokens"]
            total_tokens += usage["total_tokens"]
            doc_vectors.append(vecs[0])

            ctx.progress(i + 1, n_docs, f"Embedding document {i + 1}/{n_docs}")

        doc_matrix = np.vstack(doc_vectors).astype(np.float32)
        dim = int(doc_matrix.shape[1])
        result.set_metric("embedding_dim", dim, unit="dims")

        # Document latency metrics
        doc_total_ms = sum(doc_latencies)
        doc_pct = compute_percentiles(doc_latencies)
        result.set_metric("doc_embed_total_ms", round(doc_total_ms, 2), unit="ms", better="lower")
        result.set_metric("latency_p50_ms", doc_pct[50], unit="ms", better="lower")
        result.set_metric("latency_p95_ms", doc_pct[95], unit="ms", better="lower")
        result.set_metric("latency_p99_ms", doc_pct[99], unit="ms", better="lower")
        result.set_metric("latency_mean_ms", round(statistics.mean(doc_latencies), 2), unit="ms", better="lower")
        result.set_metric("latency_min_ms", round(min(doc_latencies), 2), unit="ms", better="lower")
        result.set_metric("latency_max_ms", round(max(doc_latencies), 2), unit="ms", better="lower")

        # Sequential throughput
        seq_throughput = n_docs / (doc_total_ms / 1000.0) if doc_total_ms > 0 else 0.0
        result.set_metric("throughput_seq_docs_per_sec", round(seq_throughput, 2), unit="docs/s", better="higher")

        # Vector norms & anisotropy
        norms_stats = compute_vector_norms(doc_matrix)
        result.set_metric("norm_mean", norms_stats["mean"], unit=None)
        result.set_metric("norm_std", norms_stats["std"], unit=None)
        result.set_metric("norm_min", norms_stats["min"], unit=None)
        result.set_metric("norm_max", norms_stats["max"], unit=None)

        mean_pairwise_sim = compute_anisotropy(doc_matrix)
        result.set_metric("mean_pairwise_similarity", mean_pairwise_sim, unit=None, better="lower")

        # Phase 2: Consistency Benchmark
        if "consistency" in suites_to_run:
            ctx.log(f"Phase 2: Checking consistency on {consistency_samples} samples...")
            max_l2 = 0.0
            num_samples = min(consistency_samples, n_docs)
            for s_idx in range(num_samples):
                ctx.check_cancelled()
                sample_doc = docs[s_idx]
                try:
                    rep_vecs, rep_usage, _ = client.embed([sample_doc.text], model=model, dimensions=dimensions)
                except EmbeddingClientError as e:
                    ctx.log(f"Error in consistency check for {sample_doc.id}: {e}", level="error")
                    raise TesterError(f"Embedding consistency check failed: {e}") from e

                total_prompt_tokens += rep_usage["prompt_tokens"]
                total_tokens += rep_usage["total_tokens"]

                l2_dist = compute_l2_distance(doc_matrix[s_idx], rep_vecs[0])
                if l2_dist > max_l2:
                    max_l2 = l2_dist

            result.set_metric("consistency_l2_max", round(max_l2, 6), unit=None, better="lower")
            ctx.log(f"Consistency check complete. Max L2 distance = {max_l2:.6f}")
            if max_l2 > 1e-3:
                ctx.log(f"Warning: Non-negligible L2 drift observed ({max_l2:.6f})", level="warning")

        # Phase 3: Throughput Benchmarks (Batched & Concurrent)
        if "throughput" in suites_to_run:
            ctx.log("Phase 3a: Testing batched throughput (all documents in single request)...")
            ctx.check_cancelled()
            try:
                _, batch_usage, batch_ms = client.embed(
                    [d.text for d in docs],
                    model=model,
                    dimensions=dimensions,
                )
            except EmbeddingClientError as e:
                ctx.log(f"Batched embedding error: {e}", level="error")
                raise TesterError(f"Batched embedding failed: {e}") from e

            total_prompt_tokens += batch_usage["prompt_tokens"]
            total_tokens += batch_usage["total_tokens"]
            batched_throughput = n_docs / (batch_ms / 1000.0) if batch_ms > 0 else 0.0
            result.set_metric("batched_latency_ms", round(batch_ms, 2), unit="ms", better="lower")
            result.set_metric("throughput_batched_docs_per_sec", round(batched_throughput, 2), unit="docs/s", better="higher")
            ctx.log(f"Batched throughput: {batched_throughput:.2f} docs/sec ({batch_ms:.2f} ms)")

            ctx.log(f"Phase 3b: Testing concurrent throughput with {concurrency} parallel workers...")
            ctx.check_cancelled()
            t_concurrent_start = time.perf_counter()
            pool = cf.ThreadPoolExecutor(max_workers=concurrency)
            try:
                futures = [
                    pool.submit(client.embed, [d.text], model, dimensions)
                    for d in docs
                ]
                for fut in cf.as_completed(futures):
                    ctx.check_cancelled()
                    try:
                        _, c_usage, _ = fut.result()
                        total_prompt_tokens += c_usage["prompt_tokens"]
                        total_tokens += c_usage["total_tokens"]
                    except Exception as e:
                        ctx.log(f"Concurrent embed request failed: {e}", level="error")
                        raise TesterError(f"Concurrent embedding worker failed: {e}") from e
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)

            concurrent_wall_ms = (time.perf_counter() - t_concurrent_start) * 1000.0
            concurrent_throughput = n_docs / (concurrent_wall_ms / 1000.0) if concurrent_wall_ms > 0 else 0.0
            result.set_metric("concurrent_wall_ms", round(concurrent_wall_ms, 2), unit="ms", better="lower")
            result.set_metric(
                "throughput_concurrent_docs_per_sec",
                round(concurrent_throughput, 2),
                unit="docs/s",
                better="higher",
            )
            ctx.log(f"Concurrent throughput: {concurrent_throughput:.2f} docs/sec ({concurrent_wall_ms:.2f} ms wall)")

        # Phase 4: Retrieval Quality & Query Evaluation
        retrieval_stats: dict[str, float] = {}
        if "retrieval" in suites_to_run:
            ctx.log(f"Phase 4: Evaluating {n_queries} retrieval queries against corpus...")
            query_latencies: list[float] = []
            ranks: list[int] = []
            top_similarities: list[float] = []
            total_search_ms: float = 0.0

            # Normalize document matrix for fast cosine similarity search
            doc_norm = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-12)

            for q_idx, q in enumerate(queries):
                ctx.check_cancelled()
                try:
                    q_vec, q_usage, q_dt = client.embed([q.text], model=model, dimensions=dimensions)
                except EmbeddingClientError as e:
                    ctx.log(f"Error embedding query {q.id}: {e}", level="error")
                    raise TesterError(f"Query embedding failed: {e}") from e

                query_latencies.append(q_dt)
                total_prompt_tokens += q_usage["prompt_tokens"]
                total_tokens += q_usage["total_tokens"]

                ctx.progress(q_idx + 1, n_queries, f"Evaluating query {q_idx + 1}/{n_queries}")

                # Search against doc matrix
                t_search = time.perf_counter()
                q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)
                sims = np.dot(q_norm, doc_norm.T)[0]
                total_search_ms += (time.perf_counter() - t_search) * 1000.0

                sorted_order = np.argsort(-sims)
                expected_idx = q.expected_index
                rank = int(np.where(sorted_order == expected_idx)[0][0]) + 1  # 1-indexed rank
                top1_idx = int(sorted_order[0])
                top1_sim = float(sims[top1_idx])
                expected_sim = float(sims[expected_idx])

                ranks.append(rank)
                top_similarities.append(top1_sim)

                case_status = "pass" if rank <= pass_rank_threshold else "fail"

                top_matches = [
                    {
                        "rank": r + 1,
                        "doc_id": docs[idx].id,
                        "similarity": round(float(sims[idx]), 4),
                        "text": docs[idx].text,
                    }
                    for r, idx in enumerate(sorted_order[:5])
                ]

                result.add_case(
                    case_id=q.id,
                    suite="retrieval",
                    status=case_status,
                    metrics={
                        "rank": {"value": rank, "better": "lower"},
                        "reciprocal_rank": {"value": round(1.0 / rank, 4), "better": "higher"},
                        "top_similarity": {"value": round(top1_sim, 4), "better": "higher"},
                        "expected_similarity": {"value": round(expected_sim, 4), "better": "higher"},
                        "latency_ms": {"value": round(q_dt, 2), "unit": "ms", "better": "lower"},
                    },
                    detail={
                        "query": q.text,
                        "topic": q.topic,
                        "expected_doc_id": q.expected_doc_id,
                        "expected_doc": docs[expected_idx].text,
                        "expected_rank": rank,
                        "top1_doc_id": docs[top1_idx].id,
                        "top1_doc": docs[top1_idx].text,
                        "top1_similarity": round(top1_sim, 4),
                        "expected_similarity": round(expected_sim, 4),
                        "top_matches": top_matches,
                    },
                )

            # Query latency percentiles & search time
            q_pct = compute_percentiles(query_latencies)
            result.set_metric("query_latency_p50_ms", q_pct[50], unit="ms", better="lower")
            result.set_metric("query_latency_p95_ms", q_pct[95], unit="ms", better="lower")
            result.set_metric("query_latency_p99_ms", q_pct[99], unit="ms", better="lower")
            result.set_metric("query_latency_mean_ms", round(statistics.mean(query_latencies), 2), unit="ms", better="lower")
            result.set_metric("search_total_ms", round(total_search_ms, 2), unit="ms", better="lower")

            # Retrieval quality metrics
            retrieval_stats = compute_retrieval_metrics(ranks, top_similarities)
            result.set_metric("recall_at_1", retrieval_stats["recall_at_1"], unit="%", better="higher")
            result.set_metric("recall@1", retrieval_stats["recall_at_1"], unit="%", better="higher")
            result.set_metric("recall_at_3", retrieval_stats["recall_at_3"], unit="%", better="higher")
            result.set_metric("recall@3", retrieval_stats["recall_at_3"], unit="%", better="higher")
            result.set_metric("recall_at_5", retrieval_stats["recall_at_5"], unit="%", better="higher")
            result.set_metric("recall@5", retrieval_stats["recall_at_5"], unit="%", better="higher")
            result.set_metric("mrr", retrieval_stats["mrr"], unit=None, better="higher")
            result.set_metric("mean_rank", retrieval_stats["mean_rank"], unit=None, better="lower")
            result.set_metric("mean_top_similarity", retrieval_stats["mean_top_similarity"], unit=None, better="higher")

            ctx.log(
                f"Retrieval complete: Recall@1={retrieval_stats['recall_at_1']}%, "
                f"Recall@3={retrieval_stats['recall_at_3']}%, "
                f"Recall@5={retrieval_stats['recall_at_5']}%, "
                f"MRR={retrieval_stats['mrr']:.4f}"
            )

        # Token usage metrics
        result.set_metric("prompt_tokens", total_prompt_tokens, unit="tokens", better="lower")
        result.set_metric("total_tokens", total_tokens, unit="tokens", better="lower")

        # Determine overall result status
        has_failed_cases = any(c.status == "fail" for c in result.cases)
        result.status = "fail" if has_failed_cases else "pass"

        # Summary text
        recall_1_str = f"{retrieval_stats.get('recall_at_1', 'N/A')}%" if "retrieval" in suites_to_run else "N/A"
        mrr_str = f"{retrieval_stats.get('mrr', 'N/A')}" if "retrieval" in suites_to_run else "N/A"
        result.summary = (
            f"Embedding benchmark ({model}): Recall@1={recall_1_str}, MRR={mrr_str}, "
            f"p50={result.metrics.get('latency_p50_ms', {}).get('value')}ms, "
            f"seq_throughput={result.metrics.get('throughput_seq_docs_per_sec', {}).get('value')} docs/s"
        )

        # Generate artifacts
        report_text = format_embedding_report(
            model=model,
            base_url=client.endpoint_url,
            n_docs=n_docs,
            n_queries=n_queries if "retrieval" in suites_to_run else 0,
            dim=dim,
            metrics=result.metrics,
            cases=[c.to_dict() for c in result.cases],
        )

        report_file = ctx.artifacts_dir / "embedding_report.txt"
        report_file.write_text(report_text, encoding="utf-8")
        ctx.add_artifact("embedding_report.txt", report_file, mime="text/plain")

        detailed_json = {
            "tester": self.name,
            "model": model,
            "endpoint_url": client.endpoint_url,
            "status": result.status,
            "summary": result.summary,
            "metrics": result.metrics,
            "suites_executed": suites_to_run,
            "retrieval_cases": [c.to_dict() for c in result.cases],
        }
        json_file = ctx.artifacts_dir / "embedding_results.json"
        json_file.write_text(json.dumps(detailed_json, indent=2), encoding="utf-8")
        ctx.add_artifact("embedding_results.json", json_file, mime="application/json")

        for art in ctx.artifacts:
            if art not in result.artifacts:
                result.artifacts.append(art)

        ctx.log(f"Embedding test finished with status: {result.status.upper()}")
        return result
