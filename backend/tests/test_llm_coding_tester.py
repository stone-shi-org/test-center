"""Unit, tester, and Piston integration tests for the ``llm_coding`` tester plugin (TES-46)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from backend.app.testers.base import (
    TesterCancelledError,
    TesterConfigError,
    TesterResult,
    TestContext,
    set_result_checkpoint_callback,
)
from backend.app.testers.llm_coding import LLMCodingTester
from backend.app.testers.llm_coding.dataset import (
    DEFAULT_JUDGE_PROMPT_MD,
    SOLUTION_HEADING_RE,
    compute_problem_sha256,
    compute_problems_Rollup_sha256,
    load_iqb_java_dataset,
    pick_sample_case_indices,
    resolve_iqb_java_dir,
    split_solutions,
)
from backend.app.testers.llm_coding.feedback import build_attempt_feedback
from backend.app.testers.llm_coding.harness import (
    annotate_runtime_output,
    extract_java_code,
    merge_java_solution_with_harness,
)
from backend.app.testers.llm_coding.judge import (
    parse_coding_judge_json,
    render_coding_judge_prompt,
)
from backend.app.testers.llm_coding.piston import (
    STATUS_ACCEPTED,
    STATUS_COMPILE_ERROR,
    STATUS_NO_CODE,
    STATUS_PERFORMANCE_LIMIT,
    STATUS_RUNTIME_ERROR,
    STATUS_TLE,
    STATUS_WRONG_ANSWER,
    PistonClient,
    clamp_piston_timeout_ms,
    judge_java_solution,
    resolve_runner_url,
)
from backend.app.testers.llm_coding.scoring import (
    aggregate_run_metrics,
    compute_case_score,
)
from backend.app.testers.registry import get_tester, list_testers


# ==============================================================================
# Fixtures & Fake Helpers
# ==============================================================================


SAMPLE_HARNESS_JAVA = """import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        if (!sc.hasNextInt()) return;
        int a = sc.nextInt();
        int b = sc.nextInt();
        System.out.println(new Solution().add(a, b));
    }
}
"""

SAMPLE_TEMPLATE_JAVA = """public class Solution {
    public int add(int a, int b) {
        
    }
}
"""

SAMPLE_REFERENCE_JAVA = """import java.util.*;

public class Solution {
    public int add(int a, int b) {
        return a + b;
    }
}
"""


def _write_synthetic_dataset(root: Path, num_problems: int = 2) -> Path:
    """Create a self-contained synthetic ``iqb-java`` dataset with valid SHA-256 hashes."""
    problems_dir = root / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)

    judge_prompt_path = root / "judge_prompt.md"
    judge_prompt_path.write_text(DEFAULT_JUDGE_PROMPT_MD, encoding="utf-8")
    judge_prompt_sha = hashlib.sha256(
        judge_prompt_path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()

    problems_map: dict[str, str] = {}
    diffs = ["easy", "medium", "hard"]

    for i in range(1, num_problems + 1):
        pid = f"prob-{i}"
        pdir = problems_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)
        diff = diffs[(i - 1) % len(diffs)]

        tests = [
            {"input": "2 3\n", "expected_output": "5\n"},
            {"input": "10 20\n", "expected_output": "30\n"},
            {"input": "SECRET_IN_1 100\n", "expected_output": "SECRET_OUT_101\n"},
            {"input": "SECRET_IN_2 200\n", "expected_output": "SECRET_OUT_202\n"},
        ]
        meta = {
            "id": pid,
            "title": f"Problem {i}: Sum Two Integers",
            "difficulty": diff,
            "topics": ["math", "array"] if i % 2 == 1 else ["dp"],
            "source": "interviewqb",
            "leetcode_id": str(1000 + i),
            "statement_md": f"Given two integers `a` and `b`, return `a + b`.\n\nExample 1:\nInput: 2 3\nOutput: 5",
            "timeout_ms": 3000,
            "cpu_limit_ms": 3000,
            "style_rubric": "Prefer O(1) time and space.",
            "sample_case_idx": [0, 1],
        }
        (pdir / "problem.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (pdir / "Solution.template.java").write_text(SAMPLE_TEMPLATE_JAVA, encoding="utf-8")
        (pdir / "Harness.java").write_text(SAMPLE_HARNESS_JAVA, encoding="utf-8")
        (pdir / "Reference.java").write_text(SAMPLE_REFERENCE_JAVA, encoding="utf-8")
        (pdir / "tests.json").write_text(json.dumps(tests, indent=2) + "\n", encoding="utf-8")

        problems_map[pid] = compute_problem_sha256(pdir)

    manifest_sha = compute_problems_Rollup_sha256(problems_map)
    manifest = {
        "version": "1.0-test",
        "exported_at": "2026-10-20T00:00:00Z",
        "source": "synthetic-test",
        "problem_count": len(problems_map),
        "manifest_sha256": manifest_sha,
        "judge_prompt_sha256": judge_prompt_sha,
        "problems": problems_map,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


@dataclass
class FakeStreamResp:
    content: str
    prompt_tokens: int = 120
    completion_tokens: int = 80


class FakeChatClient:
    """Controllable fake LLM client supporting ``stream_chat``, ``chat``, and ``chat_with_tools``."""

    def __init__(
        self,
        stream_responses: list[str] | None = None,
        chat_responses: list[str] | None = None,
        tool_turns: list[dict[str, Any]] | None = None,
        on_call: Any | None = None,
    ) -> None:
        self.endpoint_url = "http://fake-llm:8000/v1/chat/completions"
        self.stream_responses = list(stream_responses or [])
        self.chat_responses = list(chat_responses or [])
        self.tool_turns = list(tool_turns or [])
        self.stream_calls: list[list[dict[str, Any]]] = []
        self.chat_calls: list[list[dict[str, Any]]] = []
        self.tool_calls_history: list[list[dict[str, Any]]] = []
        self.on_call = on_call

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        effort: str | None = None,
        **kwargs: Any,
    ) -> FakeStreamResp:
        self.stream_calls.append([dict(m) for m in messages])
        if self.on_call:
            self.on_call(len(self.stream_calls), messages)
        text = (
            self.stream_responses.pop(0)
            if self.stream_responses
            else "```java\nclass Solution { public int add(int a, int b) { return a + b; } }\n```"
        )
        return FakeStreamResp(content=text, prompt_tokens=100, completion_tokens=50)

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int], float]:
        self.chat_calls.append([dict(m) for m in messages])
        text = (
            self.chat_responses.pop(0)
            if self.chat_responses
            else json.dumps(
                {
                    "approach_correctness": 5,
                    "complexity_vs_reference": 5,
                    "readability_idiom": 4,
                    "overall": 5,
                    "justification": "Optimal O(1) solution with clean formatting.",
                }
            )
        )
        return text, {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260}, 15.0

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
        effort: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.tool_calls_history.append([dict(m) for m in messages])
        if self.tool_turns:
            turn = self.tool_turns.pop(0)
        else:
            turn = {
                "content": "```java\nclass Solution { public int add(int a, int b) { return a + b; } }\n```",
                "tool_calls": [],
            }
        content = turn.get("content", "")
        tcs = turn.get("tool_calls", [])
        usage = turn.get(
            "usage",
            {"prompt_tokens": 150, "completion_tokens": 70, "total_tokens": 220},
        )
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tcs:
            msg["tool_calls"] = tcs
        return {
            "content": content,
            "tool_calls": tcs,
            "message": msg,
            "usage": usage,
            "latency_ms": 20.0,
        }


# ==============================================================================
# 1. Unit Tests: Java Harness Merge & Runtime Output Annotation
# ==============================================================================


def test_java_harness_merge_hoists_imports_and_preserves_line_numbers():
    solution = (
        "package com.example.leetcode;\n"
        "import java.util.*;\n"
        "import java.math.BigInteger;\n"
        "import java.util.*; // duplicate should be deduped\n"
        "import static java.lang.Math.max;\n"
        "\n"
        "public final class Solution {\n"
        "    public int add(int a, int b) {\n"
        "        return max(a, b) + Math.min(a, b);\n"
        "    }\n"
        "}"
    )
    merged = merge_java_solution_with_harness(solution, SAMPLE_HARNESS_JAVA)

    # Hoisted imports appear at the very top, deduplicated
    assert merged.code.startswith(
        "import java.util.*;\nimport java.math.BigInteger;\nimport static java.lang.Math.max;\n\n"
    )
    # Package statement is stripped
    assert "package com.example" not in merged.code
    # `public final class Solution` is demoted to `final class Solution`
    assert "public final class Solution" not in merged.code
    assert "final class Solution" in merged.code

    # Check line number alignment: line 8 in original `solution` is `    public int add(int a, int b) {`
    merged_lines = merged.code.split("\n")
    target_merged_line_1based = merged.solution_start_line + 8 - 1
    assert "public int add(int a, int b)" in merged_lines[target_merged_line_1based - 1]

    # Annotate runtime output maps Main.java line numbers in Solution back to Solution.java
    raw_err = (
        f"Main.java:5: warning in harness\n"
        f"Main.java:{target_merged_line_1based}: error: incompatible types\n"
    )
    annotated = annotate_runtime_output(raw_err, merged)
    assert "Main.java:5: warning in harness" in annotated
    assert "Main.java:5 (" not in annotated
    assert f"Main.java:{target_merged_line_1based} (Solution.java:8): error: incompatible types" in annotated


# ==============================================================================
# 2. Unit Tests: Code Extraction from LLM Responses
# ==============================================================================


def test_extract_java_code_variants():
    # Single fenced java block
    resp1 = "Here is the solution:\n```java\nimport java.util.*;\npublic class Solution {\n    int f() { return 1; }\n}\n```\nDone!"
    code1 = extract_java_code(resp1)
    assert code1 is not None
    assert "class Solution" in code1
    assert "```" not in code1

    # Multiple fenced blocks: prefers the final block defining class Solution
    resp2 = (
        "First a helper:\n```java\n// draft snippet\nint x = 1;\n```\n"
        "Final answer:\n```java\nclass Solution {\n    public int solve() { return 42; }\n}\n```"
    )
    code2 = extract_java_code(resp2)
    assert code2 is not None
    assert "return 42;" in code2

    # Unfenced raw code containing class Solution
    resp3 = "import java.util.*;\nclass Solution {\n    public int solve() { return 7; }\n}\n"
    code3 = extract_java_code(resp3)
    assert code3 is not None
    assert "return 7;" in code3

    # No Java code at all -> None
    assert extract_java_code("I cannot solve this problem right now.") is None
    assert extract_java_code("") is None
    assert extract_java_code(None) is None


# ==============================================================================
# 3. Unit Tests: Status Mapping from Piston Responses
# ==============================================================================


def test_piston_status_mapping_all_statuses():
    tests = [
        {"input": "2 3\n", "expected_output": "5\n"},
        {"input": "10 20\n", "expected_output": "30\n"},
    ]

    # 3a. no_code_extracted
    empty_client = PistonClient(execute_hook=lambda p: {"run": {"stdout": "5\n", "stderr": "", "code": 0}})
    res_no_code = judge_java_solution(
        client=empty_client,
        code="No code in this text",
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=tests,
        sample_case_idx=[0],
        extract_from_markdown=True,
    )
    assert res_no_code.status == STATUS_NO_CODE
    assert res_no_code.passed is False
    assert res_no_code.correct == 0

    # 3b. Accepted
    outputs = ["5\n", "30\n"]
    acc_client = PistonClient(
        execute_hook=lambda p: {
            "run": {"stdout": outputs.pop(0), "stderr": "", "code": 0, "cpu_time": 120}
        }
    )
    res_acc = judge_java_solution(
        client=acc_client,
        code=SAMPLE_REFERENCE_JAVA,
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=tests,
        sample_case_idx=[0],
        cpu_limit_ms=1000,
    )
    assert res_acc.status == STATUS_ACCEPTED
    assert res_acc.passed is True
    assert res_acc.correct == 1
    assert res_acc.test_case_pass_rate == 1.0

    # 3c. Partial Wrong Answer (runs every case without stopping at first failure!)
    wa_outputs = ["999\n", "30\n"]
    wa_client = PistonClient(
        execute_hook=lambda p: {
            "run": {"stdout": wa_outputs.pop(0), "stderr": "", "code": 0, "cpu_time": 100}
        }
    )
    res_wa = judge_java_solution(
        client=wa_client,
        code=SAMPLE_REFERENCE_JAVA,
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=tests,
        sample_case_idx=[0, 1],
        stop_on_first_failure=False,
    )
    assert res_wa.status == STATUS_WRONG_ANSWER
    assert res_wa.passed_cases == 1
    assert res_wa.total_cases == 2
    assert res_wa.test_case_pass_rate == 0.5

    # 3d. Compile Error via JEP 330 source launch (`error: compilation failed` on run.stderr)
    ce_client = PistonClient(
        execute_hook=lambda p: {
            "run": {
                "stdout": "",
                "stderr": "Main.java:16: error: ';' expected\n1 error\nerror: compilation failed\n",
                "code": 1,
            }
        }
    )
    res_ce = judge_java_solution(
        client=ce_client,
        code=SAMPLE_REFERENCE_JAVA,
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=tests,
        sample_case_idx=[0],
    )
    assert res_ce.status == STATUS_COMPILE_ERROR
    assert res_ce.compilation_failed is True
    assert "Solution.java:" in res_ce.compile_error_output
    assert res_ce.passed_cases == 0

    # 3e. Runtime Error
    re_client = PistonClient(
        execute_hook=lambda p: {
            "run": {
                "stdout": "",
                "stderr": "Exception in thread \"main\" java.lang.ArithmeticException: / by zero\n\tat Solution.add(Main.java:14)\n",
                "code": 1,
            }
        }
    )
    res_re = judge_java_solution(
        client=re_client,
        code=SAMPLE_REFERENCE_JAVA,
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=tests,
        sample_case_idx=[0],
    )
    assert res_re.status == STATUS_RUNTIME_ERROR
    assert res_re.passed is False

    # 3f. Time Limit Exceeded (SIGKILL)
    tle_client = PistonClient(
        execute_hook=lambda p: {
            "run": {"stdout": "", "stderr": "", "code": 137, "signal": "SIGKILL"}
        }
    )
    res_tle = judge_java_solution(
        client=tle_client,
        code=SAMPLE_REFERENCE_JAVA,
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=tests,
        sample_case_idx=[0],
    )
    assert res_tle.status == STATUS_TLE
    assert res_tle.passed is False

    # 3g. Performance Limit Exceeded (stdout matches, but cpu_time > cpu_limit_ms)
    ple_outputs = ["5\n", "30\n"]
    ple_client = PistonClient(
        execute_hook=lambda p: {
            "run": {"stdout": ple_outputs.pop(0), "stderr": "", "code": 0, "cpu_time": 1800}
        }
    )
    res_ple = judge_java_solution(
        client=ple_client,
        code=SAMPLE_REFERENCE_JAVA,
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=tests,
        sample_case_idx=[0],
        cpu_limit_ms=500,
    )
    assert res_ple.status == STATUS_PERFORMANCE_LIMIT
    assert res_ple.passed is False
    assert res_ple.correct == 0


def test_runner_url_resolution_and_timeout_clamp(monkeypatch):
    monkeypatch.setenv("PISTON_URL", "http://env-piston:2000")
    assert resolve_runner_url() == "http://env-piston:2000"

    target = {"base_url": "http://llm:8000", "extra": {"runner_url": "http://target-piston:2000/"}}
    assert resolve_runner_url(target=target, config={}) == "http://target-piston:2000"

    # Config override takes precedence when explicitly set
    assert (
        resolve_runner_url(target=target, config={"runner_url": "http://cfg-piston:2000/"})
        == "http://cfg-piston:2000"
    )

    # Verify LLMCodingTester.validate_and_apply_config({}) does NOT inject a default runner_url
    tester = LLMCodingTester()
    validated = tester.validate_and_apply_config({})
    assert "runner_url" not in validated
    assert resolve_runner_url(target=target, config=validated) == "http://target-piston:2000"

    # Timeout clamping to min(test_timeout_value, 3000)
    assert clamp_piston_timeout_ms(4000) == 3000
    assert clamp_piston_timeout_ms(1500) == 1500


# ==============================================================================
# 4. Unit Tests: Visible/Hidden Split, Solution Stripping & Zero Hidden Leakage
# ==============================================================================


def test_split_solutions_strips_embedded_solutions():
    raw_md = (
        "# Two Sum\n\n"
        "Given an array of integers, return indices.\n\n"
        "<details><summary>Click for Solution hint</summary>Use a hash map.</details>\n\n"
        "---\n\n"
        "## 1. Optimal Solution\n"
        "```java\nclass Solution { /* secret reference */ }\n```\n"
        "## Approach 2: Brute Force\n"
        "Try all pairs.\n"
    )
    res = split_solutions(raw_md)
    assert "secret reference" not in res.body
    assert "Use a hash map" not in res.body
    assert "## 1. Optimal Solution" not in res.body
    assert "## Approach 2" not in res.body
    assert len(res.solutions) == 3
    for line in res.body.splitlines():
        assert not SOLUTION_HEADING_RE.match(line)


def test_visible_hidden_split_and_no_hidden_leakage_in_feedback():
    tests = [
        {"input": "SAMPLE_IN_1\n", "expected_output": "SAMPLE_EXP_1\n"},
        {"input": "SAMPLE_IN_2\n", "expected_output": "SAMPLE_EXP_2\n"},
        {"input": "TOP_SECRET_HIDDEN_INPUT_XYZ\n", "expected_output": "TOP_SECRET_HIDDEN_EXP_ABC\n"},
    ]
    sample_idx = pick_sample_case_indices("Example:\nSAMPLE_IN_1\nSAMPLE_EXP_1", tests)
    # For 3 test cases, cap is min(3, max(1, 3 // 2)) = 1 sample so 2 remain hidden
    assert sample_idx == [0]

    # Run where sample case 0 fails AND hidden cases 1 and 2 fail with secret strings
    outputs = ["SAMPLE_WRONG_1\n", "HIDDEN_WRONG_2\n", "TOP_SECRET_HIDDEN_ACTUAL_DEF\n"]
    client = PistonClient(
        execute_hook=lambda p: {
            "run": {
                "stdout": outputs.pop(0),
                "stderr": "TOP_SECRET_STDERR_LEAK" if len(outputs) == 0 else "",
                "code": 0,
            }
        }
    )
    judge_res = judge_java_solution(
        client=client,
        code=SAMPLE_REFERENCE_JAVA,
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=tests,
        sample_case_idx=[0],
        stop_on_first_failure=False,
    )
    assert judge_res.sample_total == 1
    assert judge_res.hidden_total == 2

    for level in ("verdict", "first_failure", "full"):
        fb = build_attempt_feedback(
            judge_result=judge_res,
            feedback_level=level,
            attempt_number=1,
            max_attempts=3,
        )
        # Hidden case inputs, expected outputs, actual outputs, and stderr MUST NEVER appear
        assert "TOP_SECRET_HIDDEN_INPUT_XYZ" not in fb
        assert "TOP_SECRET_HIDDEN_EXP_ABC" not in fb
        assert "TOP_SECRET_HIDDEN_ACTUAL_DEF" not in fb
        assert "TOP_SECRET_STDERR_LEAK" not in fb
        assert "Hidden tests: 2/2 failed" in fb

        if level == "verdict":
            assert "SAMPLE_IN_1" not in fb
            assert "SAMPLE_EXP_1" not in fb
        else:
            # Sample failure details ARE shown in first_failure and full
            assert "SAMPLE_IN_1" in fb
            assert "SAMPLE_EXP_1" in fb
            assert "SAMPLE_WRONG_1" in fb


# ==============================================================================
# 5. Unit Tests: Scoring Formula & Judge JSON Parsing
# ==============================================================================


def test_scoring_formula_and_attempt_factors():
    # Failed problem -> 0 regardless of attempt or judge
    s_fail = compute_case_score(correct=0, solved_attempt=None, judge_enabled=True, judge_overall=5.0)
    assert s_fail.case_score == 0.0
    assert s_fail.case_score_100 == 0.0

    # Solved on attempt 1, 2, 3 without judge
    s1 = compute_case_score(correct=1, solved_attempt=1, judge_enabled=False)
    s2 = compute_case_score(correct=1, solved_attempt=2, judge_enabled=False)
    s3 = compute_case_score(correct=1, solved_attempt=3, judge_enabled=False)
    assert s1.case_score == pytest.approx(1.0)
    assert s2.case_score == pytest.approx(0.7)
    assert s3.case_score == pytest.approx(0.5)

    # Solved on attempt 1 with judge overall = 3.0 -> judge_norm = (3-1)/4 = 0.5
    # case_score = 1 * 1.0 * (0.8 + 0.2 * 0.5) = 0.90
    sj1 = compute_case_score(
        correct=1,
        solved_attempt=1,
        judge_enabled=True,
        judge_overall=3.0,
        weight_correctness=0.8,
        weight_judge=0.2,
    )
    assert sj1.judge_norm == pytest.approx(0.5)
    assert sj1.case_score == pytest.approx(0.9)
    assert sj1.case_score_100 == pytest.approx(90.0)

    # Solved on attempt 2 with judge overall = 5.0 -> judge_norm = 1.0 -> 0.7 * (0.8 + 0.2*1.0) = 0.7
    sj2 = compute_case_score(
        correct=1,
        solved_attempt=2,
        judge_enabled=True,
        judge_overall=5.0,
    )
    assert sj2.case_score == pytest.approx(0.7)


def test_judge_json_parsing_and_prompt_rendering(tmp_path):
    raw_json = """Here is my evaluation:
```json
{
  "approach_correctness": 5,
  "complexity_vs_reference": 4,
  "readability_idiom": 3,
  "overall": 4,
  "justification": "Good hash map approach, minor naming nitpicks."
}
```"""
    parsed = parse_coding_judge_json(raw_json)
    assert parsed.approach_correctness == 5.0
    assert parsed.complexity_vs_reference == 4.0
    assert parsed.readability_idiom == 3.0
    assert parsed.overall == 4.0
    assert parsed.judge_norm == pytest.approx(0.75)
    assert "Good hash map approach" in parsed.justification

    ds_dir = _write_synthetic_dataset(tmp_path / "ds", num_problems=1)
    loaded = load_iqb_java_dataset(ds_dir)
    prob = loaded.problems[0]
    dummy_run = judge_java_solution(
        client=PistonClient(execute_hook=lambda p: {"run": {"stdout": "5\n", "stderr": "", "code": 0}}),
        code=SAMPLE_REFERENCE_JAVA,
        harness=SAMPLE_HARNESS_JAVA,
        test_cases=prob.tests,
        sample_case_idx=prob.sample_case_idx,
    )
    rendered = render_coding_judge_prompt(loaded.judge_prompt_md, prob, SAMPLE_REFERENCE_JAVA, dummy_run)
    assert prob.statement_md in rendered
    assert prob.style_rubric in rendered
    assert "return a + b;" in rendered


# ==============================================================================
# 6. Unit Tests: Dataset Loader & Manifest Hash Checking
# ==============================================================================


def test_dataset_loader_and_manifest_hash_verification(tmp_path):
    ds_dir = _write_synthetic_dataset(tmp_path / "iqb-java", num_problems=3)
    loaded = load_iqb_java_dataset(ds_dir, verify_hash=True)
    assert len(loaded.problems) == 3
    assert loaded.manifest.problem_count == 3

    # Filter by difficulty and limit
    easy_only = load_iqb_java_dataset(ds_dir, difficulty=["easy"])
    assert all(p.difficulty == "easy" for p in easy_only.problems)

    # Tamper with Reference.java in prob-1 and verify hash check fails
    ref_file = ds_dir / "problems" / "prob-1" / "Reference.java"
    ref_file.write_text("// tampered\n" + ref_file.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_iqb_java_dataset(ds_dir, verify_hash=True)


# ==============================================================================
# 7. Tester Tests: Single-Attempt, Retry, Tool Mode, Judge & Cancel Checkpointing
# ==============================================================================


def test_tester_registered_and_single_attempt_run(tmp_path):
    assert any(t.name == "llm_coding" for t in list_testers())
    tester = get_tester("llm_coding")
    assert isinstance(tester, LLMCodingTester)

    ds_dir = _write_synthetic_dataset(tmp_path / "ds", num_problems=2)
    fake_llm = FakeChatClient()

    def _piston_hook(payload: dict[str, Any]) -> dict[str, Any]:
        stdin = payload.get("stdin", "")
        if "2 3" in stdin:
            return {"run": {"stdout": "5\n", "stderr": "", "code": 0, "cpu_time": 50}}
        if "10 20" in stdin:
            return {"run": {"stdout": "30\n", "stderr": "", "code": 0, "cpu_time": 50}}
        if "SECRET_IN_1" in stdin:
            return {"run": {"stdout": "SECRET_OUT_101\n", "stderr": "", "code": 0, "cpu_time": 50}}
        return {"run": {"stdout": "SECRET_OUT_202\n", "stderr": "", "code": 0, "cpu_time": 50}}

    ctx = TestContext(run_id=101, artifacts_dir=tmp_path / "artifacts", emit_stdout=False)
    res = tester.run(
        config={
            "dataset_dir": str(ds_dir),
            "model": "test-coder-7b",
            "_chat_client": fake_llm,
            "_piston_execute_hook": _piston_hook,
        },
        target={"name": "local-llm", "kind": "llm", "base_url": "http://fake-llm:8000/v1"},
        ctx=ctx,
    )

    assert res.status == "pass"
    assert len(res.cases) == 2
    assert [c.case_id for c in res.cases] == ["prob-1", "prob-2"]
    assert res.metrics["score"]["value"] == 100.0
    assert res.metrics["solve_at_1"]["value"] == 100.0
    assert res.metrics["solve_at_k"]["value"] == 100.0
    assert res.metrics["test_case_pass_rate"]["value"] == 100.0
    assert "score_easy" in res.metrics
    assert "solve_at_1_easy" in res.metrics

    artifact_names = {a.name for a in res.artifacts}
    assert "llm_coding_report.txt" in artifact_names
    assert "llm_coding_results.json" in artifact_names


def test_tester_multi_attempt_retry_and_judge_composite_score(tmp_path):
    ds_dir = _write_synthetic_dataset(tmp_path / "ds", num_problems=1)
    # Attempt 1 returns broken code, Attempt 2 returns working code
    fake_llm = FakeChatClient(
        stream_responses=[
            "```java\nclass Solution { public int add(int a, int b) { return a - b; } }\n```",
            "```java\nclass Solution { public int add(int a, int b) { return a + b; } }\n```",
        ],
        chat_responses=[
            json.dumps(
                {
                    "approach_correctness": 5,
                    "complexity_vs_reference": 3,
                    "readability_idiom": 4,
                    "overall": 3,
                    "justification": "Correct on second try.",
                }
            )
        ],
    )

    def _piston_hook(payload: dict[str, Any]) -> dict[str, Any]:
        src = payload["files"][0]["content"]
        stdin = payload.get("stdin", "")
        if "return a - b;" in src:
            return {"run": {"stdout": "-1\n", "stderr": "", "code": 0, "cpu_time": 40}}
        if "2 3" in stdin:
            return {"run": {"stdout": "5\n", "stderr": "", "code": 0, "cpu_time": 40}}
        if "10 20" in stdin:
            return {"run": {"stdout": "30\n", "stderr": "", "code": 0, "cpu_time": 40}}
        if "SECRET_IN_1" in stdin:
            return {"run": {"stdout": "SECRET_OUT_101\n", "stderr": "", "code": 0, "cpu_time": 40}}
        return {"run": {"stdout": "SECRET_OUT_202\n", "stderr": "", "code": 0, "cpu_time": 40}}

    tester = LLMCodingTester()
    ctx = TestContext(run_id=102, artifacts_dir=tmp_path / "artifacts", emit_stdout=False)
    res = tester.run(
        config={
            "dataset_dir": str(ds_dir),
            "max_attempts": 3,
            "feedback_level": "first_failure",
            "judge_enabled": True,
            "_chat_client": fake_llm,
            "_piston_execute_hook": _piston_hook,
        },
        target={"name": "local-llm", "kind": "llm", "base_url": "http://fake-llm:8000/v1"},
        ctx=ctx,
    )

    assert res.status == "pass"
    assert res.metrics["solve_at_1"]["value"] == 0.0
    assert res.metrics["solve_at_k"]["value"] == 100.0
    assert res.metrics["mean_attempts_to_solve"]["value"] == 2.0
    assert res.metrics["judge_mean"]["value"] == 3.0

    # Solved on attempt 2 (factor 0.7) with judge overall 3.0 (judge_norm = 0.5):
    # expected score = 0.7 * (0.8 + 0.2 * 0.5) * 100 = 63.0
    assert res.metrics["score"]["value"] == pytest.approx(63.0)

    # Verify the feedback sent in prompt 2 did NOT leak hidden test inputs/outputs
    second_turn_messages = fake_llm.stream_calls[1]
    feedback_msg = second_turn_messages[-1]["content"]
    assert "SECRET_IN_1" not in feedback_msg
    assert "SECRET_OUT_101" not in feedback_msg
    assert "Sample Case #1" in feedback_msg


def test_tester_agent_tool_mode(tmp_path):
    ds_dir = _write_synthetic_dataset(tmp_path / "ds", num_problems=1)
    good_code = "class Solution { public int add(int a, int b) { return a + b; } }"

    fake_llm = FakeChatClient(
        tool_turns=[
            {
                "content": "Let me compile and run samples first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "compile",
                            "arguments": json.dumps({"code": good_code}),
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "run_samples",
                            "arguments": json.dumps({"code": good_code}),
                        },
                    },
                ],
            },
            {
                "content": "Let me test a custom input and then submit.",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "run_custom",
                            "arguments": json.dumps({"code": good_code, "stdin": "7 8\n"}),
                        },
                    },
                    {
                        "id": "call_4",
                        "type": "function",
                        "function": {
                            "name": "submit",
                            "arguments": json.dumps({"code": good_code}),
                        },
                    },
                ],
            },
        ]
    )

    def _piston_hook(payload: dict[str, Any]) -> dict[str, Any]:
        stdin = payload.get("stdin", "")
        if "2 3" in stdin:
            return {"run": {"stdout": "5\n", "stderr": "", "code": 0, "cpu_time": 30}}
        if "10 20" in stdin:
            return {"run": {"stdout": "30\n", "stderr": "", "code": 0, "cpu_time": 30}}
        if "7 8" in stdin:
            return {"run": {"stdout": "15\n", "stderr": "", "code": 0, "cpu_time": 30}}
        if "SECRET_IN_1" in stdin:
            return {"run": {"stdout": "SECRET_OUT_101\n", "stderr": "", "code": 0, "cpu_time": 30}}
        return {"run": {"stdout": "SECRET_OUT_202\n", "stderr": "", "code": 0, "cpu_time": 30}}

    tester = LLMCodingTester()
    ctx = TestContext(run_id=103, artifacts_dir=tmp_path / "artifacts", emit_stdout=False)
    res = tester.run(
        config={
            "dataset_dir": str(ds_dir),
            "tools_enabled": True,
            "max_attempts": 3,
            "max_tool_calls": 10,
            "_chat_client": fake_llm,
            "_piston_execute_hook": _piston_hook,
        },
        target={"name": "local-llm", "kind": "llm", "base_url": "http://fake-llm:8000/v1"},
        ctx=ctx,
    )

    assert res.status == "pass"
    assert res.metrics["score"]["value"] == 100.0
    assert res.metrics["solve_at_1"]["value"] == 100.0
    assert res.metrics["tool_calls_mean"]["value"] == 4.0
    case_detail = res.cases[0].detail
    assert case_detail["tool_call_count"] == 4
    assert [e["name"] for e in case_detail["tool_events"]] == [
        "compile",
        "run_samples",
        "run_custom",
        "submit",
    ]


def test_tester_partial_results_checkpoint_on_cancel(tmp_path):
    ds_dir = _write_synthetic_dataset(tmp_path / "ds", num_problems=3)
    ctx = TestContext(run_id=104, artifacts_dir=tmp_path / "artifacts", emit_stdout=False)

    # Cancel right after problem 1's LLM call finishes, before problem 2 starts
    def _on_llm_call(call_num: int, _messages: Any) -> None:
        if call_num >= 1:
            # Let problem 1 finish its Piston check, then cancel before problem 2
            pass

    fake_llm = FakeChatClient(on_call=_on_llm_call)
    piston_calls = 0

    def _piston_hook(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal piston_calls
        piston_calls += 1
        # Problem 1 has 4 test cases; after the 4th test case, mark context cancelled
        if piston_calls == 4:
            ctx.set_cancelled()
        stdin = payload.get("stdin", "")
        if "2 3" in stdin:
            return {"run": {"stdout": "5\n", "stderr": "", "code": 0, "cpu_time": 20}}
        if "10 20" in stdin:
            return {"run": {"stdout": "30\n", "stderr": "", "code": 0, "cpu_time": 20}}
        if "SECRET_IN_1" in stdin:
            return {"run": {"stdout": "SECRET_OUT_101\n", "stderr": "", "code": 0, "cpu_time": 20}}
        return {"run": {"stdout": "SECRET_OUT_202\n", "stderr": "", "code": 0, "cpu_time": 20}}

    checkpoints: list[TesterResult] = []
    set_result_checkpoint_callback(lambda r: checkpoints.append(TesterResult.from_dict(r.to_dict())))
    try:
        tester = LLMCodingTester()
        with pytest.raises(TesterCancelledError):
            tester.run(
                config={
                    "dataset_dir": str(ds_dir),
                    "_chat_client": fake_llm,
                    "_piston_execute_hook": _piston_hook,
                },
                target={"name": "local-llm", "kind": "llm", "base_url": "http://fake-llm:8000/v1"},
                ctx=ctx,
            )
    finally:
        set_result_checkpoint_callback(None)

    assert len(checkpoints) >= 1
    last_cp = checkpoints[-1]
    assert len(last_cp.cases) == 1
    assert last_cp.cases[0].case_id == "prob-1"
    assert last_cp.metrics["score"]["value"] == 100.0
    assert last_cp.metrics["problems_evaluated"]["value"] == 1


def test_tester_concurrency_execution(tmp_path):
    """Verify that concurrency > 1 executes problems in parallel across worker threads."""
    tester = get_tester("llm_coding")
    ds_dir = _write_synthetic_dataset(tmp_path / "ds", num_problems=4)
    fake_llm = FakeChatClient()

    active_threads = 0
    max_concurrent = 0
    lock = threading.Lock()

    def _piston_hook(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal active_threads, max_concurrent
        with lock:
            active_threads += 1
            if active_threads > max_concurrent:
                max_concurrent = active_threads
        time.sleep(0.04)
        with lock:
            active_threads -= 1

        stdin = str(payload.get("stdin") or "")
        if "2 3" in stdin:
            out = "5\n"
        elif "10 20" in stdin:
            out = "30\n"
        elif "SECRET_IN_1" in stdin:
            out = "SECRET_OUT_101\n"
        elif "SECRET_IN_2" in stdin:
            out = "SECRET_OUT_202\n"
        else:
            out = "0\n"
        return {"run": {"stdout": out, "stderr": "", "code": 0, "cpu_time": 50}}

    ctx = TestContext(run_id=202, artifacts_dir=tmp_path / "artifacts", emit_stdout=False)
    res = tester.run(
        config={
            "dataset_dir": str(ds_dir),
            "model": "test-coder-7b",
            "concurrency": 4,
            "_chat_client": fake_llm,
            "_piston_execute_hook": _piston_hook,
        },
        target={"name": "local-llm", "kind": "llm", "base_url": "http://fake-llm:8000/v1"},
        ctx=ctx,
    )

    assert res.status == "pass"
    assert len(res.cases) == 4
    assert [c.case_id for c in res.cases] == ["prob-1", "prob-2", "prob-3", "prob-4"]
    assert max_concurrent >= 2, f"Expected concurrent execution, but max concurrency was {max_concurrent}"


def test_llm_coding_compare_and_baselines_integration(app_and_client, tmp_path):
    """Verify two llm_coding runs line up by stable problem case_id in /api/compare and /api/baselines."""
    client, session_factory, _artifacts_dir = app_and_client
    from backend.app.db.base import utc_now
    from backend.app.db.models.case_result import CaseResult
    from backend.app.db.models.result import Result
    from backend.app.db.models.run import Run

    ds_dir = _write_synthetic_dataset(tmp_path / "ds", num_problems=2)
    tester = LLMCodingTester()

    def _piston_all_pass(payload: dict[str, Any]) -> dict[str, Any]:
        stdin = payload.get("stdin", "")
        if "2 3" in stdin:
            return {"run": {"stdout": "5\n", "stderr": "", "code": 0, "cpu_time": 20}}
        if "10 20" in stdin:
            return {"run": {"stdout": "30\n", "stderr": "", "code": 0, "cpu_time": 20}}
        if "SECRET_IN_1" in stdin:
            return {"run": {"stdout": "SECRET_OUT_101\n", "stderr": "", "code": 0, "cpu_time": 20}}
        return {"run": {"stdout": "SECRET_OUT_202\n", "stderr": "", "code": 0, "cpu_time": 20}}

    # Run 1 (baseline model): solves prob-1, fails prob-2
    fake_llm_1 = FakeChatClient(
        stream_responses=[
            "```java\nclass Solution { public int add(int a, int b) { return a + b; } }\n```",
            "```java\nclass Solution { public int add(int a, int b) { return 0; } }\n```",
        ]
    )

    def _piston_run1(payload: dict[str, Any]) -> dict[str, Any]:
        if "return 0;" in payload["files"][0]["content"]:
            return {"run": {"stdout": "0\n", "stderr": "", "code": 0, "cpu_time": 20}}
        return _piston_all_pass(payload)

    res1 = tester.run(
        config={
            "dataset_dir": str(ds_dir),
            "model": "model-v1",
            "_chat_client": fake_llm_1,
            "_piston_execute_hook": _piston_run1,
        },
        target={"name": "vllm-1", "kind": "llm", "base_url": "http://vllm:8000/v1"},
        ctx=TestContext(run_id=201, artifacts_dir=tmp_path / "art1", emit_stdout=False),
    )

    # Run 2 (candidate model): solves both prob-1 and prob-2
    res2 = tester.run(
        config={
            "dataset_dir": str(ds_dir),
            "model": "model-v2",
            "_chat_client": FakeChatClient(),
            "_piston_execute_hook": _piston_all_pass,
        },
        target={"name": "vllm-2", "kind": "llm", "base_url": "http://vllm:8000/v1"},
        ctx=TestContext(run_id=202, artifacts_dir=tmp_path / "art2", emit_stdout=False),
    )

    run_ids: list[int] = []
    db = session_factory()
    try:
        for idx, tr in enumerate((res1, res2), start=1):
            now = utc_now()
            run_row = Run(
                status="completed",
                queued_at=now,
                started_at=now,
                finished_at=now,
                config_snapshot={"tester": "llm_coding", "config": {"model": f"model-v{idx}"}},
                target_snapshot={"name": f"vllm-{idx}", "base_url": "http://vllm:8000/v1"},
            )
            db.add(run_row)
            db.flush()
            res_row = Result(
                run_id=run_row.id,
                tester="llm_coding",
                status=tr.status,
                metrics=tr.metrics,
                summary=tr.summary,
            )
            db.add(res_row)
            db.flush()
            for c in tr.cases:
                db.add(
                    CaseResult(
                        result_id=res_row.id,
                        suite=c.suite,
                        case_id=c.case_id,
                        status=c.status,
                        metrics=c.metrics,
                        detail=c.detail,
                    )
                )
            db.commit()
            run_ids.append(run_row.id)
    finally:
        db.close()

    cmp_resp = client.get(f"/api/compare?run_ids={run_ids[0]},{run_ids[1]}")
    assert cmp_resp.status_code == 200
    cmp_data = cmp_resp.json()
    assert cmp_data["tester"] == "llm_coding"
    mmap = cmp_data["metric_map"]
    assert "score" in mmap
    assert "solve_at_1" in mmap
    assert "test_case_pass_rate" in mmap
    score_m = mmap["score"]
    assert score_m["baseline_value"] == 50.0
    assert score_m["target_value"] == 100.0
    assert score_m["improvement"] is True
    case_ids_compared = [c["case_id"] for c in cmp_data["cases"]]
    assert "prob-1" in case_ids_compared
    assert "prob-2" in case_ids_compared


# ==============================================================================
# 8. Integration Test: Exported Dataset & Reference Solutions on Sidecar Piston
# ==============================================================================


def test_exported_iqb_java_dataset_and_sidecar_piston():
    """Verify the exported dataset in ``test-data/iqb-java`` and run Reference.java on Piston if reachable."""
    ds_path = resolve_iqb_java_dir()
    if not (ds_path / "manifest.json").is_file():
        pytest.skip(f"Exported iqb-java dataset not mounted at {ds_path}")

    dataset = load_iqb_java_dataset(ds_path, verify_hash=True)
    # Acceptance criterion: ~80-90 verified problems exported, statements contain no solution text
    assert len(dataset.problems) >= 75
    for prob in dataset.problems:
        assert len(prob.sample_tests) >= 1, f"{prob.id} missing visible sample tests"
        assert len(prob.hidden_tests) >= 1, f"{prob.id} missing hidden tests"
        for line in prob.statement_md.splitlines():
            assert not SOLUTION_HEADING_RE.match(
                line
            ), f"{prob.id} leaked solution heading: {line}"

    # Check if dedicated Piston sidecar is reachable for live integration verification
    candidate_urls = []
    if os.getenv("PISTON_URL"):
        candidate_urls.append(os.environ["PISTON_URL"])
    candidate_urls.extend(["http://piston:2000", "http://10.100.1.65:2000"])

    live_client: PistonClient | None = None
    for url in candidate_urls:
        c = PistonClient(base_url=url, request_timeout_s=3.0)
        try:
            runtimes = c.get_runtimes()
            if any(r.get("language") == "java" for r in runtimes):
                live_client = PistonClient(base_url=url, request_timeout_s=30.0)
                break
        except Exception:
            continue

    if live_client is None:
        return

    # When RUN_FULL_PISTON_VERIFICATION=1 is set, verify every exported problem;
    # otherwise verify a representative sample across easy/medium/hard to keep unit test suite fast.
    if os.getenv("RUN_FULL_PISTON_VERIFICATION") == "1":
        to_verify = dataset.problems
    else:
        by_diff: dict[str, list[Any]] = {"easy": [], "medium": [], "hard": []}
        for p in dataset.problems:
            by_diff.setdefault(p.difficulty, []).append(p)
        to_verify = (
            by_diff["easy"][:2] + by_diff["medium"][:2] + by_diff["hard"][:1]
        )

    for prob in to_verify:
        res = judge_java_solution(
            client=live_client,
            code=prob.reference_java,
            harness=prob.harness_java,
            test_cases=prob.tests,
            sample_case_idx=prob.sample_case_idx,
            timeout_ms=prob.timeout_ms,
            cpu_limit_ms=prob.cpu_limit_ms,
            stop_on_first_failure=False,
        )
        assert res.passed is True, (
            f"Reference solution for {prob.id} failed on Piston: "
            f"status={res.status}, passed={res.passed_cases}/{res.total_cases}"
        )
        assert res.passed_cases == res.total_cases
