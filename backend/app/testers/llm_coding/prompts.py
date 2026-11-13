"""Prompt construction for ``llm_coding`` problems (standard and tool-calling modes)."""

from __future__ import annotations

import re

from backend.app.testers.llm_coding.dataset import ProblemSpec

_TOP_LEVEL_TYPE_RE = re.compile(
    r"^(?:(?:public|final|abstract|static)\s+)*(?:class|interface|enum|record)\s+(\w+)", re.MULTILINE
)


def _top_level_types(java: str) -> list[str]:
    """Top-level (column 0) type names, ignoring comments and string literals."""
    masked = re.sub(
        r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\\n])*"',
        lambda m: re.sub(r"[^\n]", " ", m.group(0)),
        java or "",
        flags=re.DOTALL,
    )
    seen: list[str] = []
    for name in _TOP_LEVEL_TYPE_RE.findall(masked):
        if name not in seen:
            seen.append(name)
    return seen


def template_answer_classes(problem: ProblemSpec) -> tuple[bool, list[str]]:
    """Return ``(has_solution_class, classes_to_submit)`` for a problem's starter template.

    ``classes_to_submit`` are the template's top-level types that the harness does not define
    (so the answer must contain them), other than ``Solution``.
    """
    template_types = _top_level_types(problem.template_java)
    harness_types = set(_top_level_types(problem.harness_java))
    others = [t for t in template_types if t != "Solution" and t not in harness_types]
    return "Solution" in template_types, others


def _fmt(names: list[str]) -> str:
    return ", ".join(f"`{n}`" for n in names)

SYSTEM_PROMPT_STANDARD = (
    "You are an expert Java programmer solving coding interview problems. "
    "Follow the template signature and requirements strictly. "
    "Return your final code in a single fenced ```java code block containing `class Solution`."
)

SYSTEM_PROMPT_TOOLS = (
    "You are an expert Java programmer solving coding interview problems with access to execution tools. "
    "You may use `compile`, `run_samples`, and `run_custom` to verify and debug your `class Solution` implementation. "
    "When ready, call `submit(code)` with the complete Java source containing `class Solution` (imports allowed, no `class Main`)."
)


def build_problem_prompt(problem: ProblemSpec, tools_enabled: bool = False) -> str:
    """Build the initial user prompt for a coding problem."""
    parts: list[str] = [
        f"# {problem.title}",
        "",
        problem.statement_md.strip(),
        "",
        "## Starter Java Template (`Solution.template.java`)",
        "```java",
        problem.template_java.strip(),
        "```",
        "",
        "## Rules",
        "- Implement the solution in Java using the exact class and method signature from the starter template.",
        "- Standard library `import` statements (e.g. `import java.util.*;`) are allowed at the top of your code.",
        "- Do **NOT** define `class Main` or `public static void main` — a separate test harness provides `Main`.",
    ]
    # Templates without ``class Solution`` (e.g. ``LRUCache``) or with helper classes the harness
    # does not define (e.g. ``Node``): say so explicitly. Other problems keep the exact prompt
    # text, so their result-cache keys do not change.
    has_solution, others = template_answer_classes(problem)
    if not has_solution and others:
        parts.append(
            f"- This template has **no** `class Solution`: implement and return its {_fmt(others)} "
            f"class{'es' if len(others) > 1 else ''} with the exact names and signatures shown. "
            "Do **not** add a `class Solution`."
        )
    elif others:
        parts.append(
            f"- Your code block must also contain the template's {_fmt(others)} "
            f"class{'es' if len(others) > 1 else ''} as given (the test harness does not define "
            f"{'them' if len(others) > 1 else 'it'})."
        )
    solution_word = "`class Solution`" if has_solution or not others else f"the {_fmt(others)} class{'es' if len(others) > 1 else ''}"
    if tools_enabled:
        parts.extend(
            [
                "- You can use `compile(code)`, `run_samples(code)`, and `run_custom(code, stdin)` to test your code.",
                "- Call `submit(code)` to submit your final solution for evaluation against the hidden test suite (each `submit` uses one attempt).",
                f"- If you finish without calling `submit`, return your final answer as a single fenced ```java block containing {solution_word}.",
            ]
        )
    else:
        parts.append(f"- Return **exactly one** fenced ```java code block containing {solution_word}.")

    return "\n".join(parts)
