"""Comprehensive test suite for OpenAI-compatible Embedding Tester."""

import hashlib
import json
from pathlib import Path
import re
from typing import Any
import httpx
import numpy as np
import pytest

from backend.app.testers.base import TestContext, TesterCancelledError, TesterConfigError, TesterError
from backend.app.testers.registry import get_tester, list_testers
from backend.app.testers.embedding import (
    EmbeddingTester,
    OpenAIEmbeddingClient,
    EmbeddingClientError,
    BUILTIN_DOCUMENTS,
    BUILTIN_QUERIES,
    get_default_corpus,
)
from backend.app.testers.embedding.metrics import (
    cosine_similarity,
    compute_vector_norms,
    compute_anisotropy,
    compute_l2_distance,
    compute_percentiles,
    compute_retrieval_metrics,
)
from backend.app.testers.embedding.report import format_embedding_report


# ===========================================================================
# 1. Corpus Tests
# ===========================================================================

def test_corpus_structure():
    """Verify built-in corpus contains 30 documents and 30 queries with valid references."""
    docs, queries = get_default_corpus()
    assert len(docs) == 30
    assert len(queries) == 30

    doc_ids = {d.id for d in docs}
    assert len(doc_ids) == 30

    for idx, d in enumerate(docs):
        assert d.id == f"doc_{idx+1:02d}"
        assert len(d.text) > 10
        assert d.topic
        assert d.index == idx

    for idx, q in enumerate(queries):
        assert q.id == f"retrieval_query_{idx+1:02d}"
        assert len(q.text) > 5
        assert q.expected_doc_id in doc_ids
        assert 0 <= q.expected_index < 30
        assert docs[q.expected_index].id == q.expected_doc_id


# ===========================================================================
# 2. Client & URL Normalization Tests
# ===========================================================================

def test_client_url_normalization():
    """Verify endpoint URL normalization handles various base URL formats."""
    # Standard /v1
    c1 = OpenAIEmbeddingClient(base_url="https://api.openai.com/v1")
    assert c1.endpoint_url == "https://api.openai.com/v1/embeddings"

    # Trailing slash
    c2 = OpenAIEmbeddingClient(base_url="https://api.openai.com/v1/")
    assert c2.endpoint_url == "https://api.openai.com/v1/embeddings"

    # Already has /embeddings
    c3 = OpenAIEmbeddingClient(base_url="https://api.openai.com/v1/embeddings")
    assert c3.endpoint_url == "https://api.openai.com/v1/embeddings"

    # Root localhost
    c4 = OpenAIEmbeddingClient(base_url="http://localhost:8000")
    assert c4.endpoint_url == "http://localhost:8000/embeddings"


def test_client_headers():
    """Verify headers include Authorization Bearer when API key is provided."""
    client_with_key = OpenAIEmbeddingClient(
        base_url="https://api.openai.com/v1",
        api_key="sk-test-secret-12345",
    )
    headers = client_with_key._build_headers()
    assert headers["Authorization"] == "Bearer sk-test-secret-12345"
    assert headers["Content-Type"] == "application/json"

    client_no_key = OpenAIEmbeddingClient(base_url="https://api.openai.com/v1")
    headers_no_key = client_no_key._build_headers()
    assert "Authorization" not in headers_no_key


def test_client_empty_texts():
    """Verify embedding empty texts returns empty array without making HTTP call."""
    client = OpenAIEmbeddingClient(base_url="https://api.openai.com/v1")
    vectors, usage, dt = client.embed([], model="test-model")
    assert vectors.shape == (0, 0)
    assert usage["total_tokens"] == 0
    assert dt == 0.0


# ===========================================================================
# 3. Vector Math & Metric Calculation Tests
# ===========================================================================

def test_cosine_similarity():
    """Verify cosine similarity calculation for identical, orthogonal, and opposing vectors."""
    v1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    v2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    sims = cosine_similarity(v1, v2)
    assert sims.shape == (2, 2)
    assert np.isclose(sims[0, 0], 1.0, atol=1e-5)  # identical
    assert np.isclose(sims[0, 1], 0.0, atol=1e-5)  # orthogonal
    assert np.isclose(sims[1, 0], 0.0, atol=1e-5)  # orthogonal
    assert np.isclose(sims[1, 1], -1.0, atol=1e-5) # opposite


def test_vector_norms_and_anisotropy():
    """Verify norm statistics and anisotropy mean pairwise similarity."""
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    stats = compute_vector_norms(vectors)
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["mean"] == 2.0

    # Orthogonal vectors have pairwise similarity 0.0
    anisotropy = compute_anisotropy(vectors)
    assert np.isclose(anisotropy, 0.0, atol=1e-4)

    # Identical vectors have pairwise similarity 1.0
    identical_vecs = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    assert np.isclose(compute_anisotropy(identical_vecs), 1.0, atol=1e-4)


def test_retrieval_metrics():
    """Verify Recall@1, Recall@3, Recall@5, and MRR calculations."""
    # 4 queries: ranks = [1, 2, 4, 10]
    # Hits@1: rank 1 -> 1/4 = 25%
    # Hits@3: ranks 1, 2 -> 2/4 = 50%
    # Hits@5: ranks 1, 2, 4 -> 3/4 = 75%
    # MRR: (1/1 + 1/2 + 1/4 + 1/10) / 4 = (1.0 + 0.5 + 0.25 + 0.1) / 4 = 1.85 / 4 = 0.4625
    ranks = [1, 2, 4, 10]
    top_sims = [0.95, 0.85, 0.75, 0.65]
    metrics = compute_retrieval_metrics(ranks, top_sims)
    assert metrics["recall_at_1"] == 25.0
    assert metrics["recall_at_3"] == 50.0
    assert metrics["recall_at_5"] == 75.0
    assert np.isclose(metrics["mrr"], 0.4625, atol=1e-4)
    assert metrics["mean_rank"] == 4.25
    assert np.isclose(metrics["mean_top_similarity"], 0.80, atol=1e-4)


# ===========================================================================
# 4. Mock Helpers for End-to-End Testing
# ===========================================================================

def _create_deterministic_embedding(text: str, dim: int = 64) -> list[float]:
    """Generate a deterministic pseudo-embedding vector from text hash."""
    # Use MD5 to generate a seed
    h = hashlib.md5(text.encode("utf-8")).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.tolist()


def _make_mock_transport(dim: int = 64, semantic_bias: bool = True) -> httpx.MockTransport:
    """Create a mock transport simulating OpenAI /embeddings endpoint.

    If semantic_bias is True: queries generate vectors close to their ground truth document,
    ensuring high retrieval Recall@1.
    """
    docs, queries = get_default_corpus()
    doc_text_to_idx = {d.text: idx for idx, d in enumerate(docs)}
    query_text_to_target = {q.text: q.expected_index for q in queries}

    # Precompute document vectors
    doc_vectors = [_create_deterministic_embedding(d.text, dim=dim) for d in docs]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        body = json.loads(request.content.decode("utf-8"))
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]

        data = []
        for idx, text in enumerate(inputs):
            if text in doc_text_to_idx:
                d_idx = doc_text_to_idx[text]
                vec = list(doc_vectors[d_idx])
            elif semantic_bias and text in query_text_to_target:
                # Target doc vector + tiny noise so it matches top 1
                target_idx = query_text_to_target[text]
                base_vec = np.array(doc_vectors[target_idx], dtype=np.float32)
                noise = np.random.RandomState(42).randn(dim).astype(np.float32) * 0.01
                q_vec = base_vec + noise
                q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
                vec = q_vec.tolist()
            else:
                vec = _create_deterministic_embedding(text, dim=dim)

            data.append({
                "object": "embedding",
                "index": idx,
                "embedding": vec,
            })

        resp_body = {
            "object": "list",
            "data": data,
            "model": body.get("model", "text-embedding-3-small"),
            "usage": {
                "prompt_tokens": len(inputs) * 10,
                "total_tokens": len(inputs) * 10,
            },
        }
        return httpx.Response(200, json=resp_body)

    return httpx.MockTransport(handler)


# ===========================================================================
# 5. Full EmbeddingTester Execution & Metrics Verification
# ===========================================================================

def test_embedding_tester_registered():
    """Verify EmbeddingTester is registered in tester_registry and discoverable."""
    testers = list_testers()
    assert any(t.name == "embedding" for t in testers)

    plugin = get_tester("embedding")
    assert isinstance(plugin, EmbeddingTester)
    assert plugin.suites == ["latency", "throughput", "consistency", "retrieval"]
    assert "properties" in plugin.config_schema


def test_embedding_tester_full_run(tmp_path: Path):
    """Run full benchmark with all suites and verify all required metrics and per-case rows."""
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=101, artifacts_dir=artifacts_dir, emit_stdout=False)

    tester = EmbeddingTester()
    mock_transport = _make_mock_transport(dim=64, semantic_bias=True)

    config = {
        "_transport": mock_transport,
        "model": "text-embedding-3-small",
        "concurrency": 4,
        "consistency_samples": 2,
        "pass_rank_threshold": 1,
    }
    target = {
        "name": "mock-openai",
        "base_url": "https://api.openai.com/v1",
        "auth_env_var": "MOCK_KEY",
    }

    result = tester.run(config, target, ctx)

    # 1. Overall status
    assert result.status == "pass"
    assert "Embedding benchmark" in (result.summary or "")

    # 2. Latency metrics (p50/p95/p99)
    assert "latency_p50_ms" in result.metrics
    assert "latency_p95_ms" in result.metrics
    assert "latency_p99_ms" in result.metrics
    assert "latency_mean_ms" in result.metrics
    assert "latency_min_ms" in result.metrics
    assert "latency_max_ms" in result.metrics
    assert result.metrics["latency_p50_ms"]["unit"] == "ms"
    assert result.metrics["latency_p50_ms"]["better"] == "lower"
    assert "query_latency_p50_ms" in result.metrics
    assert "query_latency_p95_ms" in result.metrics
    assert "query_latency_p99_ms" in result.metrics
    assert "search_total_ms" in result.metrics

    # 3. Throughput metrics (sequential/batched/concurrent)
    assert "throughput_seq_docs_per_sec" in result.metrics
    assert "throughput_batched_docs_per_sec" in result.metrics
    assert "throughput_concurrent_docs_per_sec" in result.metrics
    assert "batched_latency_ms" in result.metrics
    assert "concurrent_wall_ms" in result.metrics
    assert result.metrics["throughput_batched_docs_per_sec"]["better"] == "higher"
    assert result.metrics["throughput_concurrent_docs_per_sec"]["better"] == "higher"

    # 4. Token usage
    assert "prompt_tokens" in result.metrics
    assert "total_tokens" in result.metrics
    assert result.metrics["total_tokens"]["value"] > 0

    # 5. Embedding consistency & dimension
    assert "consistency_l2_max" in result.metrics
    assert result.metrics["consistency_l2_max"]["value"] < 1e-4  # Deterministic mock
    assert result.metrics["embedding_dim"]["value"] == 64

    # 6. Vector norms & anisotropy
    assert "norm_mean" in result.metrics
    assert "norm_std" in result.metrics
    assert "norm_min" in result.metrics
    assert "norm_max" in result.metrics
    assert "mean_pairwise_similarity" in result.metrics

    # 7. Retrieval Recall@1/3/5 and MRR
    assert "recall_at_1" in result.metrics
    assert "recall@1" in result.metrics
    assert "recall_at_3" in result.metrics
    assert "recall@3" in result.metrics
    assert "recall_at_5" in result.metrics
    assert "recall@5" in result.metrics
    assert "mrr" in result.metrics
    assert "mean_rank" in result.metrics
    assert "mean_top_similarity" in result.metrics

    assert result.metrics["recall_at_1"]["value"] == 100.0
    assert result.metrics["mrr"]["value"] == 1.0

    # 8. Per-case rows for retrieval queries
    assert len(result.cases) == 30
    for case in result.cases:
        assert case.suite == "retrieval"
        assert case.status == "pass"
        assert case.metrics["rank"]["value"] == 1
        assert case.metrics["reciprocal_rank"]["value"] == 1.0
        assert "latency_ms" in case.metrics
        assert "query" in case.detail
        assert "expected_doc_id" in case.detail
        assert "top1_doc_id" in case.detail
        assert len(case.detail["top_matches"]) == 5

    # 9. Artifacts verification
    assert len(result.artifacts) >= 2
    report_artifact = next(a for a in result.artifacts if a.name == "embedding_report.txt")
    json_artifact = next(a for a in result.artifacts if a.name == "embedding_results.json")

    report_path = Path(report_artifact.path)
    assert report_path.exists()
    report_content = report_path.read_text(encoding="utf-8")
    assert "EMBEDDING BENCHMARK REPORT" in report_content
    assert "Latency (ms):" in report_content
    assert "Throughput:" in report_content
    assert "Retrieval Quality:" in report_content
    assert "Retrieval Query Breakdown" in report_content

    json_path = Path(json_artifact.path)
    assert json_path.exists()
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_data["tester"] == "embedding"
    assert len(json_data["retrieval_cases"]) == 30


def test_embedding_tester_partial_suites(tmp_path: Path):
    """Verify running a restricted subset of suites (e.g. latency only)."""
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=102, artifacts_dir=artifacts_dir, emit_stdout=False)

    tester = EmbeddingTester()
    mock_transport = _make_mock_transport(dim=32)

    config = {
        "_transport": mock_transport,
        "suites": ["latency"],
    }
    target = {
        "name": "mock-openai",
        "base_url": "https://api.openai.com/v1",
    }

    result = tester.run(config, target, ctx)
    assert result.status == "pass"
    assert "latency_p50_ms" in result.metrics
    # Throughput and retrieval suites were not executed
    assert "throughput_batched_docs_per_sec" not in result.metrics
    assert "recall_at_1" not in result.metrics
    assert len(result.cases) == 0


def test_embedding_tester_cancellation(tmp_path: Path):
    """Verify execution halts immediately and raises TesterCancelledError when context is cancelled."""
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=103, artifacts_dir=artifacts_dir, emit_stdout=False)
    ctx.set_cancelled()

    tester = EmbeddingTester()
    mock_transport = _make_mock_transport(dim=32)

    config = {"_transport": mock_transport}
    target = {"base_url": "https://api.openai.com/v1"}

    with pytest.raises(TesterCancelledError):
        tester.run(config, target, ctx)


def test_embedding_tester_api_error_handling(tmp_path: Path):
    """Verify TesterError is raised with descriptive context when API returns 401."""
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key provided"}})

    transport = httpx.MockTransport(error_handler)
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=104, artifacts_dir=artifacts_dir, emit_stdout=False)

    tester = EmbeddingTester()
    config = {"_transport": transport}
    target = {"base_url": "https://api.openai.com/v1"}

    with pytest.raises(TesterError, match="Embedding API error"):
        tester.run(config, target, ctx)


def test_embedding_tester_failing_retrieval_case(tmp_path: Path):
    """Verify result status is 'fail' when a query fails the pass_rank_threshold."""
    # semantic_bias=False creates random orthogonal-like embeddings, so rank will be > 1
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=105, artifacts_dir=artifacts_dir, emit_stdout=False)

    tester = EmbeddingTester()
    mock_transport = _make_mock_transport(dim=64, semantic_bias=False)

    config = {
        "_transport": mock_transport,
        "suites": ["retrieval"],
        "pass_rank_threshold": 1,
    }
    target = {"base_url": "https://api.openai.com/v1"}

    result = tester.run(config, target, ctx)
    # With random vectors, rank of expected doc won't be 1 for all queries
    assert result.status == "fail"
    assert any(c.status == "fail" for c in result.cases)


def test_embedding_tester_missing_base_url():
    """Verify TesterConfigError when target lacks base_url."""
    tester = EmbeddingTester()
    ctx = TestContext(run_id=106, artifacts_dir=Path("/tmp"), emit_stdout=False)
    with pytest.raises(TesterConfigError, match="missing required 'base_url'"):
        tester.run({}, {}, ctx)


def test_embedding_api_endpoints(app_and_client):
    """Verify embedding tester discovery and test definition creation via REST API."""
    client, _, _ = app_and_client

    # 1. GET /api/testers
    res = client.get("/api/testers")
    assert res.status_code == 200
    testers = res.json()
    embedding_info = next((t for t in testers if t["name"] == "embedding"), None)
    assert embedding_info is not None
    assert set(embedding_info["suites"]) == {"latency", "throughput", "consistency", "retrieval"}

    # 2. GET /api/testers/embedding
    res_details = client.get("/api/testers/embedding")
    assert res_details.status_code == 200
    details = res_details.json()
    assert details["name"] == "embedding"
    assert "concurrency" in details["config_schema"]["properties"]

    # 3. Create target
    res_t = client.post(
        "/api/targets",
        json={
            "name": "OpenAI Embedding Target",
            "kind": "openai-compatible",
            "base_url": "https://api.openai.com/v1",
            "auth_env_var": "OPENAI_API_KEY",
            "extra": {"model": "text-embedding-3-small"},
        },
    )
    assert res_t.status_code == 201
    target_id = res_t.json()["id"]

    # 4. Create test definition
    res_def = client.post(
        "/api/test-definitions",
        json={
            "name": "Embedding Standard Benchmark",
            "tester": "embedding",
            "suites": ["latency", "retrieval"],
            "config": {
                "concurrency": 8,
                "pass_rank_threshold": 1,
            },
            "target_id": target_id,
        },
    )
    assert res_def.status_code == 201
    def_data = res_def.json()
    assert def_data["tester"] == "embedding"
    assert def_data["suites"] == ["latency", "retrieval"]
    assert def_data["config"]["concurrency"] == 8

    # 5. Trigger run
    def_id = def_data["id"]
    res_run = client.post(f"/api/test-definitions/{def_id}/run")
    assert res_run.status_code == 201
    run_data = res_run.json()
    assert run_data["status"] == "queued"
    assert run_data["config_snapshot"]["tester"] == "embedding"


def test_embedding_tester_worker_subprocess_execution(app_and_client):
    """End-to-end integration: run embedding tester via worker subprocess against mock server."""
    import asyncio
    import http.server
    import threading
    from backend.app.runner.worker import get_worker

    client, session_factory, artifacts_dir = app_and_client

    # 1. Start mock HTTP embedding server
    docs, queries = get_default_corpus()
    doc_text_to_idx = {d.text: idx for idx, d in enumerate(docs)}
    query_text_to_target = {q.text: q.expected_index for q in queries}
    dim = 32

    # Pre-generate document vectors
    doc_vectors = [_create_deterministic_embedding(d.text, dim=dim) for d in docs]

    class MockEmbeddingServerHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]

            data = []
            for idx, text in enumerate(inputs):
                if text in doc_text_to_idx:
                    d_idx = doc_text_to_idx[text]
                    vec = list(doc_vectors[d_idx])
                elif text in query_text_to_target:
                    target_idx = query_text_to_target[text]
                    base_vec = np.array(doc_vectors[target_idx], dtype=np.float32)
                    noise = np.random.RandomState(42).randn(dim).astype(np.float32) * 0.01
                    q_vec = base_vec + noise
                    vec = (q_vec / (np.linalg.norm(q_vec) + 1e-12)).tolist()
                else:
                    vec = _create_deterministic_embedding(text, dim=dim)

                data.append({"object": "embedding", "index": idx, "embedding": vec})

            resp = {
                "object": "list",
                "data": data,
                "model": body.get("model", "text-embedding-3-small"),
                "usage": {"prompt_tokens": len(inputs) * 10, "total_tokens": len(inputs) * 10},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockEmbeddingServerHandler)
    server_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # 2. Create Target pointing to mock HTTP server
        res_t = client.post(
            "/api/targets",
            json={
                "name": "Live Mock Embedding Endpoint",
                "kind": "openai-compatible",
                "base_url": f"http://127.0.0.1:{server_port}/v1",
                "auth_env_var": "TEST_KEY",
                "extra": {"model": "text-embedding-3-small"},
            },
        )
        assert res_t.status_code == 201
        target_id = res_t.json()["id"]

        # 3. Create TestDefinition with embedding tester
        res_def = client.post(
            "/api/test-definitions",
            json={
                "name": "Live Embedding Worker Suite",
                "tester": "embedding",
                "suites": ["latency", "retrieval", "consistency"],
                "config": {
                    "concurrency": 2,
                    "consistency_samples": 2,
                },
                "target_id": target_id,
            },
        )
        assert res_def.status_code == 201
        def_id = res_def.json()["id"]

        # 4. Trigger Run
        res_run = client.post(f"/api/test-definitions/{def_id}/run")
        assert res_run.status_code == 201
        run_id = res_run.json()["id"]

        # 5. Execute run via worker
        worker = get_worker()
        loop = asyncio.new_event_loop()
        executed = loop.run_until_complete(worker.poll_and_execute_once())
        loop.close()
        assert executed is True

        # 6. Verify Run completed
        res_get_run = client.get(f"/api/runs/{run_id}")
        assert res_get_run.status_code == 200
        run_info = res_get_run.json()
        assert run_info["status"] == "completed"
        assert run_info["finished_at"] is not None

        # 7. Check Results API
        res_results = client.get(f"/api/runs/{run_id}/results")
        assert res_results.status_code == 200
        results_data = res_results.json()
        assert len(results_data) == 1
        assert results_data[0]["status"] == "pass"
        result_id = results_data[0]["id"]
        metrics = results_data[0]["metrics"]

        assert "latency_p50_ms" in metrics
        assert "recall_at_1" in metrics
        assert metrics["recall_at_1"]["value"] == 100.0
        assert "mrr" in metrics
        assert metrics["mrr"]["value"] == 1.0

        # 8. Check Case Results API
        res_cases = client.get(f"/api/results/{result_id}/cases")
        assert res_cases.status_code == 200
        cases = res_cases.json()
        assert len(cases) == 30
        assert all(c["status"] == "pass" for c in cases)
        assert all(c["suite"] == "retrieval" for c in cases)

        # 9. Check Artifacts API
        res_artifacts = client.get(f"/api/results/{result_id}/artifacts")
        assert res_artifacts.status_code == 200
        artifacts = res_artifacts.json()
        artifact_names = {a["name"] for a in artifacts}
        assert "embedding_report.txt" in artifact_names
        assert "embedding_results.json" in artifact_names

    finally:
        server.shutdown()
        server.server_close()


