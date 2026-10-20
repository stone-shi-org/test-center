"""LLM code-quality judge for ``llm_coding``."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from backend.app.testers.llm_coding.dataset import DEFAULT_JUDGE_PROMPT_MD, ProblemSpec
from backend.app.testers.llm_coding.piston import JudgeRunResult

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class CodingJudgeResult:
    approach_correctness: float
    complexity_vs_reference: float
    readability_idiom: float
    overall: float
    justification: str
    judge_norm: float
    raw_response: str = ""
    judge_model: str = ""
    judge_target: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp_1_to_5(val: Any, default: float = 3.0) -> float:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return default
    return max(1.0, min(5.0, num))


def render_coding_judge_prompt(
    template_md: str | None,
    problem: ProblemSpec,
    candidate_code: str,
    judge_run: JudgeRunResult,
) -> str:
    """Fill ``judge_prompt.md`` placeholders with problem, rubric, reference, candidate, and outcome."""
    tmpl = (template_md or DEFAULT_JUDGE_PROMPT_MD).strip()
    outcome_lines = [
        f"- Verdict: {judge_run.status}",
        f"- Passed: {judge_run.passed} ({judge_run.passed_cases}/{judge_run.total_cases} total test cases; "
        f"{judge_run.hidden_passed}/{judge_run.hidden_total} hidden test cases)",
        f"- Max CPU Time: {judge_run.max_cpu_time_ms:.1f} ms"
        + (f" (limit: {problem.cpu_limit_ms} ms)" if problem.cpu_limit_ms else ""),
    ]
    if judge_run.compilation_failed and judge_run.compile_error_output:
        outcome_lines.append(f"- Compile Error:\n{judge_run.compile_error_output[:500]}")

    outcome_str = "\n".join(outcome_lines)
    rubric = problem.style_rubric.strip() or "Write clean, idiomatic, optimal Java code."

    replacements = {
        "{{statement}}": problem.statement_md.strip(),
        "{{style_rubric}}": rubric,
        "{{reference_solution}}": problem.reference_java.strip(),
        "{{candidate_code}}": (candidate_code or "").strip(),
        "{{test_outcome}}": outcome_str,
    }
    rendered = tmpl
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def parse_coding_judge_json(raw_text: str | None) -> CodingJudgeResult:
    """Parse the LLM judge's JSON output into a ``CodingJudgeResult``."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty response from LLM judge")

    candidate_json = text
    fence_match = JSON_FENCE_RE.search(text)
    if fence_match:
        candidate_json = fence_match.group(1).strip()
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate_json = text[first_brace : last_brace + 1]

    data = json.loads(candidate_json)
    if not isinstance(data, dict):
        raise ValueError("Judge JSON root is not an object")

    ac = _clamp_1_to_5(data.get("approach_correctness"), default=3.0)
    cvr = _clamp_1_to_5(data.get("complexity_vs_reference"), default=3.0)
    ri = _clamp_1_to_5(data.get("readability_idiom"), default=3.0)

    raw_overall = data.get("overall")
    if raw_overall is not None:
        overall = _clamp_1_to_5(raw_overall, default=(ac + cvr + ri) / 3.0)
    else:
        overall = round((ac + cvr + ri) / 3.0, 2)

    justification = str(data.get("justification") or "").strip()
    judge_norm = max(0.0, min(1.0, (overall - 1.0) / 4.0))

    return CodingJudgeResult(
        approach_correctness=ac,
        complexity_vs_reference=cvr,
        readability_idiom=ri,
        overall=overall,
        justification=justification,
        judge_norm=judge_norm,
        raw_response=text,
    )


def evaluate_with_coding_judge(
    judge_client: Any,
    judge_model: str,
    judge_target_name: str,
    judge_prompt_md: str,
    problem: ProblemSpec,
    candidate_code: str,
    judge_run: JudgeRunResult,
    max_tokens: int = 1024,
) -> CodingJudgeResult:
    """Invoke the configured LLM judge for a candidate Java solution."""
    prompt = render_coding_judge_prompt(
        template_md=judge_prompt_md,
        problem=problem,
        candidate_code=candidate_code,
        judge_run=judge_run,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict, objective Java code reviewer. "
                "Respond with valid JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    raw_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    err_msg: str | None = None

    try:
        if hasattr(judge_client, "chat"):
            content, usage, _elapsed = judge_client.chat(
                messages=messages,
                model=judge_model,
                max_tokens=max_tokens,
            )
            raw_text = str(content or "")
            prompt_tokens = int((usage or {}).get("prompt_tokens", 0) or 0)
            completion_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
        else:
            resp = judge_client.stream_chat(
                messages=messages,
                model=judge_model,
                max_tokens=max_tokens,
            )
            raw_text = str(getattr(resp, "content", None) or getattr(resp, "text", None) or "")
            prompt_tokens = int(getattr(resp, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(resp, "completion_tokens", 0) or 0)
            err_msg = getattr(resp, "error", None)
    except Exception as exc:
        err_msg = str(exc)

    if err_msg:
        return CodingJudgeResult(
            approach_correctness=1.0,
            complexity_vs_reference=1.0,
            readability_idiom=1.0,
            overall=1.0,
            justification=f"Judge request failed: {err_msg}",
            judge_norm=0.0,
            raw_response=raw_text,
            judge_model=judge_model,
            judge_target=judge_target_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=err_msg,
        )

    try:
        parsed = parse_coding_judge_json(raw_text)
        parsed.judge_model = judge_model
        parsed.judge_target = judge_target_name
        parsed.prompt_tokens = prompt_tokens
        parsed.completion_tokens = completion_tokens
        return parsed
    except Exception as exc:
        return CodingJudgeResult(
            approach_correctness=3.0,
            complexity_vs_reference=3.0,
            readability_idiom=3.0,
            overall=3.0,
            justification=f"Judge output parse fallback ({exc})",
            judge_norm=0.5,
            raw_response=raw_text,
            judge_model=judge_model,
            judge_target=judge_target_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=f"parse_error: {exc}",
        )
