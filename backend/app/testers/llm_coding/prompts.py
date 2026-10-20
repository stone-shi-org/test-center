"""Prompt construction for ``llm_coding`` problems (standard and tool-calling modes)."""

from __future__ import annotations

from backend.app.testers.llm_coding.dataset import ProblemSpec

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
    if tools_enabled:
        parts.extend(
            [
                "- You can use `compile(code)`, `run_samples(code)`, and `run_custom(code, stdin)` to test your code.",
                "- Call `submit(code)` to submit your final solution for evaluation against the hidden test suite (each `submit` uses one attempt).",
                "- If you finish without calling `submit`, return your final answer as a single fenced ```java block containing `class Solution`.",
            ]
        )
    else:
        parts.append(
            "- Return **exactly one** fenced ```java code block containing `class Solution`."
        )

    return "\n".join(parts)
