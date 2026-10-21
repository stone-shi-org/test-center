"""Agentic tool-calling mode for ``llm_coding`` (``compile``, ``run_samples``, ``run_custom``, ``submit``)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.app.testers.llm_coding.dataset import ProblemSpec
from backend.app.testers.llm_coding.feedback import build_attempt_feedback
from backend.app.testers.llm_coding.harness import extract_java_code
from backend.app.testers.llm_coding.piston import (
    JudgeRunResult,
    PistonClient,
    compile_java_code,
    judge_java_solution,
    run_java_custom,
)

CODING_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "compile",
            "description": (
                "Compile your Java `class Solution` alongside the problem test harness on Piston "
                "and return compiler diagnostics only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Java source code containing `class Solution` (imports allowed, no `Main`).",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_samples",
            "description": (
                "Compile and execute your Java `class Solution` against the problem's visible sample test cases only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Java source code containing `class Solution` (imports allowed, no `Main`).",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_custom",
            "description": (
                "Compile and execute your Java `class Solution` through the problem's test harness using custom stdin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Java source code containing `class Solution` (imports allowed, no `Main`).",
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Custom standard input text formatted for the problem's test harness.",
                    },
                },
                "required": ["code", "stdin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit your final Java `class Solution` for evaluation against the hidden test suite. "
                "Each `submit` call consumes one attempt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Java source code containing `class Solution` (imports allowed, no `Main`).",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


@dataclass
class ToolLoopOutcome:
    attempts: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_time_ms: float = 0.0
    solved: bool = False
    solved_attempt: int | None = None
    final_code: str = ""
    final_judge_result: JudgeRunResult | None = None
    budget_exhausted_reason: str | None = None


def execute_run_samples(
    piston_client: PistonClient,
    problem: ProblemSpec,
    code: str,
    timeout_ms: int | None = None,
    cpu_limit_ms: int | None = None,
) -> dict[str, Any]:
    """Run candidate code on visible sample test cases ONLY (never runs or reveals hidden tests)."""
    sample_cases = problem.sample_tests
    sample_indices = list(range(len(sample_cases)))
    eff_timeout_ms = timeout_ms if timeout_ms is not None else max(problem.timeout_ms or 3000, 3000)
    eff_cpu_limit_ms = (
        cpu_limit_ms
        if cpu_limit_ms is not None
        else (max(int(problem.cpu_limit_ms), 3000) if problem.cpu_limit_ms is not None else None)
    )
    res = judge_java_solution(
        client=piston_client,
        code=code,
        harness=problem.harness_java,
        test_cases=sample_cases,
        sample_case_idx=sample_indices,
        timeout_ms=eff_timeout_ms,
        cpu_limit_ms=eff_cpu_limit_ms,
        stop_on_first_failure=False,
        extract_from_markdown=False,
    )
    if res.compilation_failed:
        return {
            "status": res.status,
            "passed": False,
            "sample_passed": 0,
            "sample_total": len(sample_cases),
            "compile_error": res.compile_error_output,
            "cases": [],
        }

    cases_payload = []
    for r in res.test_results:
        cases_payload.append(
            {
                "case_number": r.case_number,
                "status": r.status_detail or r.status,
                "passed": r.passed,
                "input": r.input,
                "expected": r.expected,
                "actual": r.actual,
                "stderr": r.stderr,
                "time_ms": r.time_ms,
            }
        )
    return {
        "status": res.status,
        "passed": res.passed,
        "sample_passed": res.sample_passed,
        "sample_total": res.sample_total,
        "cases": cases_payload,
    }


def run_agent_tool_loop(
    chat_client: Any,
    piston_client: PistonClient,
    problem: ProblemSpec,
    model: str,
    initial_messages: list[dict[str, Any]],
    max_attempts: int = 1,
    feedback_level: str = "first_failure",
    max_tool_calls: int = 12,
    max_total_tokens: int = 32768,
    max_tokens_per_turn: int = 4096,
    effort: str | None = None,
    timeout_ms: int | None = None,
    cpu_limit_ms: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ToolLoopOutcome:
    """Execute the OpenAI tool-calling loop for a coding problem with tool/token/attempt budgets."""
    t0 = time.perf_counter()
    eff_timeout_ms = timeout_ms if timeout_ms is not None else max(problem.timeout_ms or 3000, 3000)
    eff_cpu_limit_ms = (
        cpu_limit_ms
        if cpu_limit_ms is not None
        else (max(int(problem.cpu_limit_ms), 3000) if problem.cpu_limit_ms is not None else None)
    )
    messages: list[dict[str, Any]] = [dict(m) for m in initial_messages]
    outcome = ToolLoopOutcome(transcript=messages)
    last_seen_code: str = ""

    while True:
        if should_cancel and should_cancel():
            break

        if outcome.tool_call_count >= max_tool_calls:
            outcome.budget_exhausted_reason = f"max_tool_calls ({max_tool_calls}) reached"
            break

        if outcome.total_tokens >= max_total_tokens:
            outcome.budget_exhausted_reason = f"max_total_tokens ({max_total_tokens}) reached"
            break

        turn = chat_client.chat_with_tools(
            messages=messages,
            model=model,
            tools=CODING_TOOLS_SCHEMA,
            max_tokens=max_tokens_per_turn,
            effort=effort,
        )
        usage = turn.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens", 0) or 0)
        c_tok = int(usage.get("completion_tokens", 0) or 0)
        t_tok = int(usage.get("total_tokens", 0) or (p_tok + c_tok))
        outcome.prompt_tokens += p_tok
        outcome.completion_tokens += c_tok
        outcome.total_tokens += t_tok

        assistant_msg = turn.get("message") or {
            "role": "assistant",
            "content": turn.get("content") or "",
        }
        messages.append(assistant_msg)

        content_str = str(turn.get("content") or "")
        extracted_from_text = extract_java_code(content_str)
        if extracted_from_text:
            last_seen_code = extracted_from_text

        tool_calls = turn.get("tool_calls") or []
        if not tool_calls:
            # Model stopped calling tools without a final submit (or after finishing).
            break

        should_stop_loop = False
        for tc in tool_calls:
            if outcome.tool_call_count >= max_tool_calls:
                outcome.budget_exhausted_reason = f"max_tool_calls ({max_tool_calls}) reached"
                should_stop_loop = True
                break

            outcome.tool_call_count += 1
            tc_id = str(tc.get("id") or f"call_{outcome.tool_call_count}")
            fn = tc.get("function") or {}
            fn_name = str(fn.get("name") or "").strip()
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {}

            raw_code = str(args.get("code") or "")
            cleaned_code = extract_java_code(raw_code) or raw_code.strip()
            if cleaned_code:
                last_seen_code = cleaned_code

            tool_t0 = time.perf_counter()
            if fn_name == "compile":
                tool_res = compile_java_code(
                    client=piston_client,
                    code=cleaned_code,
                    harness=problem.harness_java,
                    timeout_ms=eff_timeout_ms,
                )
            elif fn_name == "run_samples":
                tool_res = execute_run_samples(
                    piston_client=piston_client,
                    problem=problem,
                    code=cleaned_code,
                    timeout_ms=eff_timeout_ms,
                    cpu_limit_ms=eff_cpu_limit_ms,
                )
            elif fn_name == "run_custom":
                custom_stdin = str(args.get("stdin") or "")
                tool_res = run_java_custom(
                    client=piston_client,
                    code=cleaned_code,
                    harness=problem.harness_java,
                    stdin=custom_stdin,
                    timeout_ms=eff_timeout_ms,
                )
            elif fn_name == "submit":
                attempt_num = len(outcome.attempts) + 1
                judge_res = judge_java_solution(
                    client=piston_client,
                    code=cleaned_code,
                    harness=problem.harness_java,
                    test_cases=problem.tests,
                    sample_case_idx=problem.sample_case_idx,
                    timeout_ms=eff_timeout_ms,
                    cpu_limit_ms=eff_cpu_limit_ms,
                    stop_on_first_failure=False,
                    extract_from_markdown=False,
                )
                outcome.final_code = cleaned_code
                outcome.final_judge_result = judge_res
                attempt_record = {
                    "attempt": attempt_num,
                    "code": cleaned_code,
                    "status": judge_res.status,
                    "passed": judge_res.passed,
                    "correct": judge_res.correct,
                    "passed_cases": judge_res.passed_cases,
                    "total_cases": judge_res.total_cases,
                    "test_case_pass_rate": judge_res.test_case_pass_rate,
                    "sample_passed": judge_res.sample_passed,
                    "sample_total": judge_res.sample_total,
                    "hidden_passed": judge_res.hidden_passed,
                    "hidden_total": judge_res.hidden_total,
                    "compile_error_output": judge_res.compile_error_output,
                    "run_time_ms": judge_res.run_time_ms,
                    "max_cpu_time_ms": judge_res.max_cpu_time_ms,
                }
                outcome.attempts.append(attempt_record)

                if judge_res.passed:
                    outcome.solved = True
                    outcome.solved_attempt = attempt_num
                    tool_res = {
                        "status": judge_res.status,
                        "passed": True,
                        "attempt": attempt_num,
                        "max_attempts": max_attempts,
                        "message": "Accepted! All hidden test cases passed.",
                    }
                    should_stop_loop = True
                elif attempt_num >= max_attempts:
                    tool_res = {
                        "status": judge_res.status,
                        "passed": False,
                        "attempt": attempt_num,
                        "max_attempts": max_attempts,
                        "message": "Maximum submission attempts reached.",
                    }
                    should_stop_loop = True
                else:
                    fb_text = build_attempt_feedback(
                        judge_result=judge_res,
                        feedback_level=feedback_level,
                        attempt_number=attempt_num,
                        max_attempts=max_attempts,
                    )
                    tool_res = {
                        "status": judge_res.status,
                        "passed": False,
                        "attempt": attempt_num,
                        "max_attempts": max_attempts,
                        "feedback": fb_text,
                    }
            else:
                tool_res = {"error": f"Unknown tool '{fn_name}'"}

            tool_ms = (time.perf_counter() - tool_t0) * 1000.0
            outcome.tool_events.append(
                {
                    "tool_call_id": tc_id,
                    "name": fn_name,
                    "duration_ms": round(tool_ms, 2),
                    "status": tool_res.get("status") if isinstance(tool_res, dict) else "done",
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": fn_name,
                    "content": json.dumps(tool_res, ensure_ascii=False),
                }
            )
            if should_stop_loop:
                break

        if should_stop_loop:
            break

    # If the agent never called `submit` (or had remaining attempts when stopping) and
    # produced candidate code in text or earlier tool calls, grade the latest code once
    # if no submission was made yet.
    if not outcome.attempts and len(outcome.attempts) < max_attempts:
        candidate = last_seen_code
        judge_res = judge_java_solution(
            client=piston_client,
            code=candidate if candidate else None,
            harness=problem.harness_java,
            test_cases=problem.tests,
            sample_case_idx=problem.sample_case_idx,
            timeout_ms=eff_timeout_ms,
            cpu_limit_ms=eff_cpu_limit_ms,
            stop_on_first_failure=False,
            extract_from_markdown=False,
        )
        attempt_num = len(outcome.attempts) + 1
        outcome.final_code = candidate
        outcome.final_judge_result = judge_res
        outcome.attempts.append(
            {
                "attempt": attempt_num,
                "code": candidate,
                "status": judge_res.status,
                "passed": judge_res.passed,
                "correct": judge_res.correct,
                "passed_cases": judge_res.passed_cases,
                "total_cases": judge_res.total_cases,
                "test_case_pass_rate": judge_res.test_case_pass_rate,
                "sample_passed": judge_res.sample_passed,
                "sample_total": judge_res.sample_total,
                "hidden_passed": judge_res.hidden_passed,
                "hidden_total": judge_res.hidden_total,
                "compile_error_output": judge_res.compile_error_output,
                "run_time_ms": judge_res.run_time_ms,
                "max_cpu_time_ms": judge_res.max_cpu_time_ms,
                "auto_submitted_at_end": True,
            }
        )
        if judge_res.passed:
            outcome.solved = True
            outcome.solved_attempt = attempt_num

    outcome.wall_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return outcome
