"""Multi-attempt feedback builder for ``llm_coding``.

Strictly enforces the rule that feedback ONLY reveals test case inputs, expected
outputs, actual outputs, or stderr for visible sample cases (``is_sample=True``).
For hidden test cases, feedback reports only the verdict and failure count.
"""

from __future__ import annotations

from backend.app.testers.llm_coding.piston import (
    STATUS_COMPILE_ERROR,
    STATUS_NO_CODE,
    JudgeRunResult,
    TestCaseResult,
)

FEEDBACK_LEVELS = ("verdict", "first_failure", "full")


def _format_sample_case_failure(case: TestCaseResult) -> str:
    lines = [
        f"- Sample Case #{case.case_number}: {case.status_detail or case.status}",
        f"  Input:\n    {case.input.strip()}",
        f"  Expected Output:\n    {case.expected.strip()}",
        f"  Actual Output:\n    {case.actual.strip()}",
    ]
    if case.stderr and case.stderr.strip():
        lines.append(f"  Stderr:\n    {case.stderr.strip()}")
    return "\n".join(lines)


def build_attempt_feedback(
    judge_result: JudgeRunResult,
    feedback_level: str = "first_failure",
    attempt_number: int = 1,
    max_attempts: int = 1,
) -> str:
    """Construct retry feedback for the model after a failed attempt.

    ``feedback_level`` options:
    - ``verdict``: status (and hidden/sample failure counts) only.
    - ``first_failure``: compile error, or the first failing visible sample case's
      input/expected/actual; for hidden failures, verdict and failure count only.
    - ``full``: details for every failing visible sample case; for hidden failures,
      verdict and failure count only.
    """
    level = (feedback_level or "first_failure").strip().lower()
    if level not in FEEDBACK_LEVELS:
        level = "first_failure"

    header = (
        f"Attempt {attempt_number} of {max_attempts} failed with verdict: "
        f"**{judge_result.status}**."
    )

    if judge_result.status == STATUS_NO_CODE:
        return (
            f"{header}\n\n"
            "No valid Java code block containing `class Solution` could be extracted from your response. "
            "Please return your complete solution inside a single fenced ```java block containing `class Solution` (no `Main`)."
        )

    sample_failures = [
        r for r in judge_result.test_results if r.is_sample and not r.passed
    ]
    hidden_failures = [
        r for r in judge_result.test_results if (not r.is_sample) and not r.passed
    ]

    parts: list[str] = [header]

    if level == "verdict":
        if judge_result.sample_total > 0:
            parts.append(
                f"Sample tests passed: {judge_result.sample_passed}/{judge_result.sample_total}."
            )
        if judge_result.hidden_total > 0:
            parts.append(
                f"Hidden tests: {len(hidden_failures)}/{judge_result.hidden_total} failed "
                f"(verdict: {hidden_failures[0].status if hidden_failures else judge_result.status})."
            )
        parts.append(
            "\nPlease fix the issue and return your updated `class Solution` in a single fenced ```java code block."
        )
        return "\n".join(parts)

    # first_failure or full
    if judge_result.compilation_failed or judge_result.status == STATUS_COMPILE_ERROR:
        err_text = (judge_result.compile_error_output or "Compilation failed.").strip()
        parts.extend(
            [
                "",
                "Compiler Diagnostics:",
                "```text",
                err_text,
                "```",
                "",
                "Please fix the compilation error(s) and return your complete `class Solution` in a single fenced ```java code block.",
            ]
        )
        return "\n".join(parts)

    if sample_failures:
        if level == "first_failure":
            parts.extend(
                [
                    "",
                    f"Sample tests passed: {judge_result.sample_passed}/{judge_result.sample_total}. First failing sample case:",
                    _format_sample_case_failure(sample_failures[0]),
                ]
            )
        else:  # full
            parts.extend(
                [
                    "",
                    f"Sample tests passed: {judge_result.sample_passed}/{judge_result.sample_total}. Failing sample cases:",
                    *(_format_sample_case_failure(c) for c in sample_failures),
                ]
            )
    elif judge_result.sample_total > 0:
        parts.append(
            f"All {judge_result.sample_total} visible sample test cases passed."
        )

    if hidden_failures:
        hidden_verdict = hidden_failures[0].status
        parts.append(
            f"Hidden tests: {len(hidden_failures)}/{judge_result.hidden_total} failed "
            f"(verdict: {hidden_verdict}). Hidden test inputs and outputs are not shown."
        )

    parts.append(
        "\nPlease analyze the failure, fix your implementation, and return the complete updated `class Solution` in a single fenced ```java code block."
    )
    return "\n".join(parts)
