"""Comprehensive test suite for the OpenAI-compatible LLM Tester plugin."""

from __future__ import annotations

import asyncio
import http.server
import json
from pathlib import Path
import threading
from typing import Any
import httpx
import pytest

from backend.app.runner.worker import get_worker
from backend.app.testers.base import (
    TestContext,
    TesterCancelledError,
    TesterConfigError,
    TesterError,
)
from backend.app.testers.registry import get_tester, list_testers
from backend.app.testers.llm import (
    BUILTIN_CAPABILITY_CASES,
    ChatClientError,
    LLMTester,
    OpenAIChatClient,
    coerce_judge_score,
    compute_percentiles,
    compute_speed_summary,
    estimate_tokens,
    evaluate_coding,
    evaluate_constraint_following,
    evaluate_reasoning,
    evaluate_summarization,
    evaluate_tool_call_json,
    extract_json_object,
    format_llm_report,
    get_builtin_cases,
    parse_judge_response,
    strip_think_blocks,
)


# ===========================================================================
# 1. Capability Evaluators & Sandbox Security Tests
# ===========================================================================

def test_builtin_cases_catalog():
    """Verify all 5 required capability cases exist and their sample responses pass."""
    cases = get_builtin_cases()
    assert len(cases) == 5
    case_ids = [c.id for c in cases]
    assert case_ids == [
        "coding",
        "tool_call_json",
        "constraint_following",
        "reasoning",
        "summarization",
    ]

    for case in cases:
        outcome = case.evaluator(case.sample_passing_response)
        assert outcome.passed is True, f"Sample response for {case.id} failed: {outcome.message}"
        assert len(outcome.checks) > 0


def test_evaluate_coding_executed_checks():
    """Verify coding evaluator executes valid RLE code and catches broken implementations."""
    coding_case = next(c for c in BUILTIN_CAPABILITY_CASES if c.id == "coding")
    good_outcome = evaluate_coding(coding_case.sample_passing_response)
    assert good_outcome.passed is True
    assert good_outcome.extra["passed_assertions"] == 10
    assert good_outcome.extra["total_assertions"] == 10

    # Wrong output implementation
    bad_code = (
        "```python\n"
        "def rle_encode(s: str) -> str:\n"
        "    return 'WRONG'\n"
        "def rle_decode(s: str) -> str:\n"
        "    return s\n"
        "```"
    )
    bad_outcome = evaluate_coding(bad_code)
    assert bad_outcome.passed is False
    assert "Failed coding checks" in bad_outcome.message

    # Missing functions
    missing_fn_outcome = evaluate_coding("```python\ndef foo(): pass\n```")
    assert missing_fn_outcome.passed is False
    assert "not defined" in missing_fn_outcome.message

    # Syntax error
    syntax_err_outcome = evaluate_coding(
        "```python\ndef rle_encode(s):\n    return (\ndef rle_decode(s): pass\n```"
    )
    assert syntax_err_outcome.passed is False
    assert "SyntaxError" in syntax_err_outcome.message


def test_evaluate_coding_sandbox_restrictions():
    """Verify sandbox blocks dangerous imports (os/subprocess) and file access."""
    malicious_import = (
        "```python\n"
        "import os\n"
        "def rle_encode(s): return os.getcwd()\n"
        "def rle_decode(s): return s\n"
        "```"
    )
    outcome = evaluate_coding(malicious_import)
    assert outcome.passed is False
    assert "ImportError" in outcome.message or "not permitted" in outcome.message

    malicious_open = (
        "```python\n"
        "def rle_encode(s):\n"
        "    open('/etc/hosts', 'r')\n"
        "    return s\n"
        "def rle_decode(s): return s\n"
        "```"
    )
    outcome_open = evaluate_coding(malicious_open)
    assert outcome_open.passed is False
    assert any(not c["passed"] for c in outcome_open.checks)


def test_evaluate_tool_call_json():
    """Verify tool-call JSON evaluator checks prerequisite tool and argument bindings."""
    valid_json = (
        '```json\n'
        '{"thought": "Need forecast first", "tool_to_call": "get_weather", "arguments": {"city": "Chicago"}}\n'
        '```'
    )
    assert evaluate_tool_call_json(valid_json).passed is True

    # Premature tool call (set_thermostat instead of get_weather)
    wrong_tool = (
        '{"thought": "Set temp now", "tool_to_call": "set_thermostat", "arguments": {"temp": 72}}'
    )
    res_wrong_tool = evaluate_tool_call_json(wrong_tool)
    assert res_wrong_tool.passed is False
    assert "correct_prerequisite_tool" in res_wrong_tool.message

    # Wrong city
    wrong_city = (
        '{"thought": "Check weather", "tool_to_call": "get_weather", "arguments": {"city": "Boston"}}'
    )
    res_wrong_city = evaluate_tool_call_json(wrong_city)
    assert res_wrong_city.passed is False
    assert "arguments_city" in res_wrong_city.message

    # Invalid JSON
    assert evaluate_tool_call_json("Not a json object at all").passed is False


def test_evaluate_constraint_following():
    """Verify 4-line A-poem constraint checks."""
    valid_poem = (
        "Apples fall toward the ground,\n"
        "Attracting moon and distant star,\n"
        "Across the cosmos forces bound,\n"
        "Anchoring galaxies afar."
    )
    assert evaluate_constraint_following(valid_poem).passed is True

    # Only 3 lines
    three_lines = "All things fall,\nAcross the sky,\nAlways down."
    assert evaluate_constraint_following(three_lines).passed is False

    # Line 2 starts with 'B'
    wrong_letter = (
        "Apples fall toward the ground,\n"
        "Beneath the bright and starry sky,\n"
        "Across the cosmos forces bound,\n"
        "Anchoring galaxies on high."
    )
    assert evaluate_constraint_following(wrong_letter).passed is False


def test_evaluate_reasoning():
    """Verify river-crossing reasoning evaluator."""
    valid_reasoning = (
        "Step 1: Take the chicken across the river.\n"
        "Step 2: Return alone and take the fox across.\n"
        "Step 3: Bring the chicken back, then take the grain across.\n"
        "Step 4: Return alone and bring the chicken across.\n"
        "DONE"
    )
    assert evaluate_reasoning(valid_reasoning).passed is True

    # Missing DONE at end
    missing_done = valid_reasoning[:-4] + "FINISHED"
    assert evaluate_reasoning(missing_done).passed is False

    # Missing grain
    missing_entity = "Take the chicken and fox across the river, then bring the chicken back. DONE"
    assert evaluate_reasoning(missing_entity).passed is False


def test_evaluate_summarization():
    """Verify single-sentence <=20-word summarization evaluator."""
    valid_summary = (
        "Artificial intelligence studies machine agents that perceive their environment "
        "and act to achieve goals."
    )
    assert evaluate_summarization(valid_summary).passed is True

    # Too many words (>20)
    too_long = (
        "Artificial intelligence is a very broad and fascinating field of computer science "
        "where intelligent machine agents perceive their complex environment and take actions "
        "to maximize their chance of achieving goals."
    )
    assert evaluate_summarization(too_long).passed is False

    # Multiple sentences
    two_sentences = "Artificial intelligence uses machine agents. They perceive their environment."
    assert evaluate_summarization(two_sentences).passed is False


# ===========================================================================
# 2. Client, Streaming SSE, & Metric Helper Tests
# ===========================================================================

def test_chat_client_url_normalization_and_headers():
    """Verify OpenAIChatClient normalizes /chat/completions and /models URLs."""
    c1 = OpenAIChatClient(base_url="https://api.openai.com/v1", api_key="sk-test-123")
    assert c1.endpoint_url == "https://api.openai.com/v1/chat/completions"
    assert c1.models_url == "https://api.openai.com/v1/models"
    assert c1._build_headers()["Authorization"] == "Bearer sk-test-123"

    c2 = OpenAIChatClient(base_url="http://localhost:1234/v1/chat/completions/")
    assert c2.endpoint_url == "http://localhost:1234/v1/chat/completions"
    assert c2.models_url == "http://localhost:1234/v1/models"
    assert "Authorization" not in c2._build_headers()


def test_chat_client_streaming_sse_and_think_stripping():
    """Verify streaming SSE parsing, reasoning delta handling, <think> stripping, and usage stats."""
    sse_lines = [
        'data: {"choices": [{"index": 0, "delta": {"reasoning_content": "Thinking step..."}}]}\n\n',
        'data: {"choices": [{"index": 0, "delta": {"content": "<think>internal</think>Hello "}}]}\n\n',
        'data: {"choices": [{"index": 0, "delta": {"content": "world!"}}]}\n\n',
        'data: {"choices": [], "usage": {"prompt_tokens": 18, "completion_tokens": 12, "total_tokens": 30}}\n\n',
        'data: [DONE]\n\n',
    ]
    sse_payload = "".join(sse_lines).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "auto-detected-llm"}]})
        payload = json.loads(request.content.decode("utf-8"))
        assert "temperature" not in payload
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_payload,
        )

    client = OpenAIChatClient(
        base_url="http://localhost:8000/v1",
        transport=httpx.MockTransport(handler),
    )
    assert client.detect_model() == "auto-detected-llm"

    res = client.stream_chat(
        messages=[{"role": "user", "content": "Say hello"}],
        model="auto-detected-llm",
    )
    assert res.content == "Hello world!"
    assert res.reasoning == "Thinking step..."
    assert res.prompt_tokens == 18
    assert res.completion_tokens == 12
    assert res.total_tokens == 30
    assert res.ttft_ms > 0
    assert res.latency_ms >= res.ttft_ms
    assert res.tokens_per_sec > 0
    assert res.chunk_count == 3


def test_judge_helpers_and_score_coercion():
    """Verify coerce_judge_score and parse_judge_response handle edge cases safely."""
    assert coerce_judge_score(5) == 5
    assert coerce_judge_score("4") == 4
    assert coerce_judge_score(3.6) == 4
    assert coerce_judge_score(0) is None
    assert coerce_judge_score(6) is None
    assert coerce_judge_score(True) is None
    assert coerce_judge_score(None) is None
    assert coerce_judge_score("not-a-number") is None

    parsed = parse_judge_response(
        '```json\n{"score": 4, "passed": true, "justification": "Clear and accurate."}\n```',
        min_score=3,
    )
    assert parsed["score"] == 4
    assert parsed["passed"] is True
    assert parsed["justification"] == "Clear and accurate."
    assert parsed["error"] is None

    parsed_low = parse_judge_response(
        '{"score": 2, "passed": true, "justification": "Below threshold."}',
        min_score=3,
    )
    assert parsed_low["score"] == 2
    assert parsed_low["passed"] is False

    parsed_bad = parse_judge_response("Unparseable judge output")
    assert parsed_bad["score"] is None
    assert parsed_bad["passed"] is None
    assert parsed_bad["error"] is not None


# ===========================================================================
# 3. Mock Streaming Transport Helpers for LLMTester
# ===========================================================================

def _build_sse_response_for_text(text: str, prompt_tokens: int = 40, completion_tokens: int = 35) -> bytes:
    """Split text into multiple SSE chunks plus a usage chunk and [DONE]."""
    chunks: list[str] = []
    step = max(1, len(text) // 4)
    for i in range(0, len(text), step):
        piece = text[i : i + step]
        payload = {"choices": [{"index": 0, "delta": {"content": piece}}]}
        chunks.append(f"data: {json.dumps(payload)}\n\n")
    usage_payload = {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    chunks.append(f"data: {json.dumps(usage_payload)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


def _make_llm_mock_transport(all_pass: bool = True) -> httpx.MockTransport:
    """Create a mock transport simulating an OpenAI-compatible streaming chat server."""
    cases_by_prompt = {c.prompt: c for c in BUILTIN_CAPABILITY_CASES}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "mock-chat-v1"}]})

        body = json.loads(request.content.decode("utf-8"))
        assert "temperature" not in body
        messages = body.get("messages", [])
        user_content = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        matched_case = cases_by_prompt.get(user_content)
        if matched_case and all_pass:
            reply_text = matched_case.sample_passing_response
        elif matched_case and not all_pass:
            reply_text = "Invalid response that fails constraints."
        else:
            reply_text = "Default mock response."

        sse_bytes = _build_sse_response_for_text(reply_text)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_bytes,
        )

    return httpx.MockTransport(handler)


def _make_judge_mock_transport(score: int = 5, passed: bool = True) -> httpx.MockTransport:
    """Create a mock transport simulating a secondary LLM Judge target."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "mock-judge-v1"}]})

        body = json.loads(request.content.decode("utf-8"))
        assert "temperature" not in body
        judge_reply = json.dumps({
            "score": score,
            "passed": passed,
            "justification": f"Rated {score}/5 by mock judge target.",
        })
        resp_body = {
            "id": "chatcmpl-judge-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": judge_reply}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }
        return httpx.Response(200, json=resp_body)

    return httpx.MockTransport(handler)


# ===========================================================================
# 4. LLMTester Full Run, Partial Suites, Judge, & Error Handling Tests
# ===========================================================================

def test_llm_tester_registered():
    """Verify LLMTester is registered in tester_registry and discoverable."""
    testers = list_testers()
    assert any(t.name == "llm" for t in testers)

    plugin = get_tester("llm")
    assert isinstance(plugin, LLMTester)
    assert plugin.suites == ["coding", "tool_call", "constraints", "reasoning", "summarization"]
    assert "properties" in plugin.config_schema
    assert "temperature" not in plugin.config_schema["properties"]


def test_llm_tester_full_run_without_judge(tmp_path: Path):
    """Run all 5 capability cases and verify speed metrics, per-case rows, and artifacts."""
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=201, artifacts_dir=artifacts_dir, emit_stdout=False)

    tester = LLMTester()
    config = {
        "_transport": _make_llm_mock_transport(all_pass=True),
        "model": "qwen-test-7b",
        "repetitions": 2,
    }
    target = {
        "name": "local-llm-target",
        "base_url": "http://localhost:1234/v1",
    }

    result = tester.run(config, target, ctx)

    assert result.status == "pass"
    assert "LLM benchmark (qwen-test-7b)" in (result.summary or "")

    # Speed metrics: TTFT, tokens/s, latency percentiles
    for key in (
        "ttft_ms",
        "ttft_mean_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "tps",
        "tokens_per_sec",
        "decode_tps_mean",
        "decode_tps_p50",
        "decode_tps_p95",
        "prefill_tps_mean",
        "aggregate_tps",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cases_passed",
        "cases_total",
        "pass_rate",
    ):
        assert key in result.metrics, f"Missing expected metric: {key}"

    assert result.metrics["ttft_ms"]["better"] == "lower"
    assert result.metrics["tps"]["better"] == "higher"
    assert result.metrics["latency_p95_ms"]["better"] == "lower"
    assert result.metrics["cases_passed"]["value"] == 5
    assert result.metrics["cases_total"]["value"] == 5
    assert result.metrics["pass_rate"]["value"] == 100.0
    assert "judge_score_mean" not in result.metrics

    # Verify all 5 capability cases
    assert len(result.cases) == 5
    for case in result.cases:
        assert case.status == "pass"
        assert "ttft_ms" in case.metrics
        assert "latency_ms" in case.metrics
        assert "tokens_per_sec" in case.metrics
        assert case.detail["passed"] is True
        assert case.detail["local_passed"] is True
        assert len(case.detail["checks"]) > 0
        assert case.detail["judge"]["enabled"] is False

    # Verify artifacts
    report_artifact = next(a for a in result.artifacts if a.name == "llm_report.txt")
    json_artifact = next(a for a in result.artifacts if a.name == "llm_results.json")
    assert Path(report_artifact.path).exists()
    assert Path(json_artifact.path).exists()

    report_text = Path(report_artifact.path).read_text(encoding="utf-8")
    assert "LLM CHAT BENCHMARK & CAPABILITY REPORT" in report_text
    assert "Speed & Streaming Telemetry:" in report_text
    assert "Capability Cases Breakdown" in report_text


def test_llm_tester_with_second_target_judge(tmp_path: Path):
    """Verify optional LLM-as-a-Judge using a second target rates cases and records judge metrics."""
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=202, artifacts_dir=artifacts_dir, emit_stdout=False)

    tester = LLMTester()
    config = {
        "_transport": _make_llm_mock_transport(all_pass=True),
        "_judge_transport": _make_judge_mock_transport(score=4, passed=True),
        "judge_enabled": True,
        "judge_model": "deepseek-judge-v4",
    }
    target = {
        "name": "primary-llm",
        "base_url": "http://localhost:1234/v1",
        "extra": {"model": "local-qwen-7b"},
        "judge_target": {
            "id": 2,
            "name": "judge-llm-target",
            "base_url": "http://localhost:4000/v1",
            "extra": {"model": "deepseek-judge-v4"},
        },
    }

    result = tester.run(config, target, ctx)

    assert result.status == "pass"
    assert "judge_score_mean" in result.metrics
    assert result.metrics["judge_score_mean"]["value"] == 4.0
    assert result.metrics["judge_pass_rate"]["value"] == 100.0
    assert result.metrics["judge_cases_evaluated"]["value"] == 5

    for case in result.cases:
        assert case.status == "pass"
        assert case.metrics["judge_score"]["value"] == 4
        assert case.detail["judge"]["enabled"] is True
        assert case.detail["judge"]["score"] == 4
        assert case.detail["judge"]["passed"] is True
        assert "Rated 4/5" in case.detail["judge"]["justification"]


def test_llm_tester_judge_rejection_fails_case(tmp_path: Path):
    """Verify that when judge rejects a response and judge_require_pass=True, case status is fail."""
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=203, artifacts_dir=artifacts_dir, emit_stdout=False)

    tester = LLMTester()
    config = {
        "_transport": _make_llm_mock_transport(all_pass=True),
        "_judge_transport": _make_judge_mock_transport(score=1, passed=False),
        "suites": ["summarization"],
        "judge_enabled": True,
        "judge_min_score": 3,
        "judge_require_pass": True,
    }
    target = {
        "name": "primary-llm",
        "base_url": "http://localhost:1234/v1",
        "judge_target": {
            "name": "strict-judge",
            "base_url": "http://localhost:4000/v1",
            "extra": {"model": "judge-pro"},
        },
    }

    result = tester.run(config, target, ctx)
    assert result.status == "fail"
    assert len(result.cases) == 1
    assert result.cases[0].status == "fail"
    assert result.cases[0].detail["local_passed"] is True
    assert result.cases[0].detail["judge"]["passed"] is False
    assert result.cases[0].detail["judge"]["score"] == 1


def test_llm_tester_partial_suites_and_aliases(tmp_path: Path):
    """Verify running a subset of suites and suite alias normalization."""
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=204, artifacts_dir=artifacts_dir, emit_stdout=False)

    tester = LLMTester()
    config = {
        "_transport": _make_llm_mock_transport(all_pass=True),
        "suites": ["coding", "tool_call_json"],
    }
    target = {"base_url": "http://localhost:1234/v1"}

    result = tester.run(config, target, ctx)
    assert result.status == "pass"
    assert len(result.cases) == 2
    assert [c.case_id for c in result.cases] == ["coding", "tool_call_json"]


def test_llm_tester_cancellation_and_errors(tmp_path: Path):
    """Verify cancellation, missing base_url, and HTTP error handling."""
    tester = LLMTester()

    # 1. Missing base_url
    ctx1 = TestContext(run_id=205, artifacts_dir=tmp_path / "a1", emit_stdout=False)
    with pytest.raises(TesterConfigError, match="missing required 'base_url'"):
        tester.run({}, {}, ctx1)

    # 2. Pre-cancelled context
    ctx2 = TestContext(run_id=206, artifacts_dir=tmp_path / "a2", emit_stdout=False)
    ctx2.set_cancelled()
    with pytest.raises(TesterCancelledError):
        tester.run(
            {"_transport": _make_llm_mock_transport()},
            {"base_url": "http://localhost:1234/v1"},
            ctx2,
        )

    # 3. API 401 Unauthorized error
    def err_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Unauthorized key"}})

    ctx3 = TestContext(run_id=207, artifacts_dir=tmp_path / "a3", emit_stdout=False)
    with pytest.raises(TesterError, match="LLM API error"):
        tester.run(
            {"_transport": httpx.MockTransport(err_handler)},
            {"base_url": "http://localhost:1234/v1"},
            ctx3,
        )


# ===========================================================================
# 5. REST API & Worker Subprocess End-to-End Integration with Second Judge Target
# ===========================================================================

def test_llm_tester_worker_subprocess_with_judge_target(app_and_client, monkeypatch):
    """End-to-end integration: run LLM tester via worker subprocess with primary + judge targets."""
    client, session_factory, artifacts_dir = app_and_client

    primary_secret = "sk-primary-llm-confidential-token-111"
    judge_secret = "sk-judge-llm-confidential-token-222"
    monkeypatch.setenv("PRIMARY_LLM_KEY", primary_secret)
    monkeypatch.setenv("JUDGE_LLM_KEY", judge_secret)

    cases_by_prompt = {c.prompt: c for c in BUILTIN_CAPABILITY_CASES}
    received_primary_auth: list[str] = []
    received_judge_auth: list[str] = []

    class MockPrimaryLLMHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received_primary_auth.append(self.headers.get("Authorization", ""))
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))
            messages = body.get("messages", [])
            user_prompt = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            matched = cases_by_prompt.get(user_prompt)
            reply_text = matched.sample_passing_response if matched else "DONE"
            sse_bytes = _build_sse_response_for_text(reply_text, prompt_tokens=32, completion_tokens=28)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(sse_bytes)

        def log_message(self, *args):
            pass

    class MockJudgeLLMHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received_judge_auth.append(self.headers.get("Authorization", ""))
            judge_json = json.dumps({
                "score": 5,
                "passed": True,
                "justification": "Excellent adherence to instructions and constraints.",
            })
            resp = {
                "id": "chatcmpl-judge",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": judge_json}}],
                "usage": {"prompt_tokens": 60, "completion_tokens": 18, "total_tokens": 78},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        def log_message(self, *args):
            pass

    primary_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockPrimaryLLMHandler)
    primary_port = primary_server.server_address[1]
    threading.Thread(target=primary_server.serve_forever, daemon=True).start()

    judge_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockJudgeLLMHandler)
    judge_port = judge_server.server_address[1]
    threading.Thread(target=judge_server.serve_forever, daemon=True).start()

    try:
        # 1. Verify /api/testers/llm endpoint
        res_tester = client.get("/api/testers/llm")
        assert res_tester.status_code == 200
        tester_info = res_tester.json()
        assert tester_info["name"] == "llm"
        assert set(tester_info["suites"]) == {
            "coding", "tool_call", "constraints", "reasoning", "summarization"
        }

        # 2. Create Primary Target
        res_t1 = client.post(
            "/api/targets",
            json={
                "name": "Primary Local LLM",
                "kind": "openai-compatible",
                "base_url": f"http://127.0.0.1:{primary_port}/v1",
                "auth_env_var": "PRIMARY_LLM_KEY",
                "extra": {"model": "qwen3-8b"},
            },
        )
        assert res_t1.status_code == 201
        primary_target_id = res_t1.json()["id"]

        # 3. Create Judge Target (second target)
        res_t2 = client.post(
            "/api/targets",
            json={
                "name": "Judge LLM Endpoint",
                "kind": "openai-compatible",
                "base_url": f"http://127.0.0.1:{judge_port}/v1",
                "auth_env_var": "JUDGE_LLM_KEY",
                "extra": {"model": "deepseek-v4-judge"},
            },
        )
        assert res_t2.status_code == 201
        judge_target_id = res_t2.json()["id"]

        # 4. Create TestDefinition referencing primary target and judge_target_id
        res_def = client.post(
            "/api/test-definitions",
            json={
                "name": "Full LLM Capability + Judge Suite",
                "tester": "llm",
                "suites": ["coding", "tool_call", "constraints", "reasoning", "summarization"],
                "config": {
                    "judge_enabled": True,
                    "judge_target_id": judge_target_id,
                    "judge_min_score": 4,
                },
                "target_id": primary_target_id,
            },
        )
        assert res_def.status_code == 201
        def_id = res_def.json()["id"]

        # 5. Trigger Run
        res_run = client.post(f"/api/test-definitions/{def_id}/run")
        assert res_run.status_code == 201
        run_id = res_run.json()["id"]
        assert res_run.json()["target_snapshot"]["judge_target"]["id"] == judge_target_id

        # 6. Execute run via worker subprocess
        worker = get_worker()
        loop = asyncio.new_event_loop()
        executed = loop.run_until_complete(worker.poll_and_execute_once())
        loop.close()
        assert executed is True

        # Verify both mock servers received their expected Bearer tokens
        assert len(received_primary_auth) == 5
        assert all(h == f"Bearer {primary_secret}" for h in received_primary_auth)
        assert len(received_judge_auth) == 5
        assert all(h == f"Bearer {judge_secret}" for h in received_judge_auth)

        # 7. Verify Run & Results via REST API
        res_get_run = client.get(f"/api/runs/{run_id}")
        assert res_get_run.status_code == 200
        run_data = res_get_run.json()
        assert run_data["status"] == "completed"
        assert primary_secret not in res_get_run.text
        assert judge_secret not in res_get_run.text

        results = run_data["results"]
        assert len(results) == 1
        res_item = results[0]
        assert res_item["status"] == "pass"
        metrics = res_item["metrics"]
        assert metrics["cases_passed"]["value"] == 5
        assert metrics["pass_rate"]["value"] == 100.0
        assert metrics["judge_score_mean"]["value"] == 5.0
        assert metrics["ttft_ms"]["value"] > 0
        assert metrics["tps"]["value"] > 0
        assert metrics["latency_p95_ms"]["value"] > 0

        # 8. Verify per-case results via /api/results/{id}/cases
        result_id = res_item["id"]
        res_cases = client.get(f"/api/results/{result_id}/cases")
        assert res_cases.status_code == 200
        cases = res_cases.json()
        assert len(cases) == 5
        assert all(c["status"] == "pass" for c in cases)
        assert all(c["metrics"]["judge_score"]["value"] == 5 for c in cases)

        # 9. Verify artifacts via /api/results/{id}/artifacts
        res_artifacts = client.get(f"/api/results/{result_id}/artifacts")
        assert res_artifacts.status_code == 200
        artifact_names = {a["name"] for a in res_artifacts.json()}
        assert "llm_report.txt" in artifact_names
        assert "llm_results.json" in artifact_names

    finally:
        primary_server.shutdown()
        primary_server.server_close()
        judge_server.shutdown()
        judge_server.server_close()


def test_llm_reasoning_effort_schema_and_validation():
    """Verify effort schema definition, enum values, normalization, and validation errors."""
    tester = LLMTester()
    props = tester.config_schema["properties"]
    assert "effort" in props
    assert props["effort"]["type"] == "string"
    assert props["effort"]["enum"] == ["none", "low", "medium", "high", "xhigh", "max"]
    assert "x-advanced" not in props["effort"] or props["effort"]["x-advanced"] is False

    # Valid effort values pass
    for valid_val in ["none", "low", "medium", "high", "xhigh", "max"]:
        validated = tester.validate_and_apply_config({"effort": valid_val})
        assert validated["effort"] == valid_val

    # Empty string normalizes to unset
    empty_val = tester.validate_and_apply_config({"effort": ""})
    assert "effort" not in empty_val or empty_val["effort"] is None

    # Invalid effort raises TesterConfigError
    with pytest.raises(TesterConfigError, match="effort"):
        tester.validate_and_apply_config({"effort": "super_maximum"})


def test_llm_reasoning_effort_request_payload_and_artifacts(tmp_path: Path):
    """Verify reasoning_effort in chat payload when set, absent when unset, and not sent to judge."""
    recorded_primary_payloads: list[dict[str, Any]] = []
    recorded_judge_payloads: list[dict[str, Any]] = []

    def primary_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "mock-chat-v1"}]})
        body = json.loads(request.content.decode("utf-8"))
        recorded_primary_payloads.append(body)
        reply = "```python\ndef rle_encode(s: str) -> str:\n    if not s: return ''\n    res, cnt = [], 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]: cnt += 1\n        else:\n            res.append(f'{cnt}{s[i-1]}')\n            cnt = 1\n    res.append(f'{cnt}{s[-1]}')\n    return ''.join(res)\n\ndef rle_decode(s: str) -> str:\n    import re\n    return ''.join(int(cnt) * ch for cnt, ch in re.findall(r'(\\d+)([A-Za-z])', s))\n```"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=_build_sse_response_for_text(reply))

    def judge_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        recorded_judge_payloads.append(body)
        judge_json = json.dumps({"score": 5, "passed": True, "justification": "Good"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": judge_json}}]})

    # Case A: effort="high"
    tester = LLMTester()
    ctx1 = TestContext(run_id=701, artifacts_dir=tmp_path / "a701", emit_stdout=False)
    res1 = tester.run(
        {
            "_transport": httpx.MockTransport(primary_handler),
            "_judge_transport": httpx.MockTransport(judge_handler),
            "suites": ["coding"],
            "effort": "high",
            "judge_enabled": True,
        },
        {
            "base_url": "http://primary:8000/v1",
            "judge_target": {"base_url": "http://judge:8000/v1"},
        },
        ctx1,
    )
    assert res1.status == "pass"
    assert len(recorded_primary_payloads) == 1
    assert recorded_primary_payloads[0].get("reasoning_effort") == "high"
    # Judge request must NOT have reasoning_effort
    assert len(recorded_judge_payloads) == 1
    assert "reasoning_effort" not in recorded_judge_payloads[0]

    # Verify report and case detail
    assert "effort=high" in (res1.summary or "").lower()
    assert res1.cases[0].detail["effort"] == "high"
    report_text = Path(next(a for a in res1.artifacts if a.name == "llm_report.txt").path).read_text(encoding="utf-8")
    assert "Reasoning Effort      : high" in report_text

    # Case B: effort="none"
    recorded_primary_payloads.clear()
    recorded_judge_payloads.clear()
    ctx2 = TestContext(run_id=702, artifacts_dir=tmp_path / "a702", emit_stdout=False)
    res2 = tester.run(
        {
            "_transport": httpx.MockTransport(primary_handler),
            "suites": ["coding"],
            "effort": "none",
        },
        {"base_url": "http://primary:8000/v1"},
        ctx2,
    )
    assert res2.status == "pass"
    assert len(recorded_primary_payloads) == 1
    assert recorded_primary_payloads[0].get("reasoning_effort") == "none"
    assert res2.cases[0].detail["effort"] == "none"

    # Case C: effort unset (omitted)
    recorded_primary_payloads.clear()
    ctx3 = TestContext(run_id=703, artifacts_dir=tmp_path / "a703", emit_stdout=False)
    res3 = tester.run(
        {
            "_transport": httpx.MockTransport(primary_handler),
            "suites": ["coding"],
        },
        {"base_url": "http://primary:8000/v1"},
        ctx3,
    )
    assert res3.status == "pass"
    assert len(recorded_primary_payloads) == 1
    assert "reasoning_effort" not in recorded_primary_payloads[0]
    assert res3.cases[0].detail["effort"] is None
