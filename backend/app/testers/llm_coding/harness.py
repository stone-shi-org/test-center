"""Java harness merging and code extraction utilities.

Ported from ``interview-question-bank/src/lib/harness-merge.ts`` so that
test-center merges a candidate ``class Solution`` with a problem's
``class Main`` test harness identically to PrepLeet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

IMPORT_LINE_RE = re.compile(r"^\s*import\s+(?:static\s+)?[\w.*]+\s*;\s*(?://.*)?$")
PACKAGE_LINE_RE = re.compile(r"^\s*package\s+[\w.]+\s*;\s*(?://.*)?$")

# Matches ``public class Solution`` (with optional ``final`` / ``abstract`` modifiers)
# and strips ``public`` so the class can sit alongside ``public class Main`` in a
# single ``Main.java`` compilation unit.
PUBLIC_SOLUTION_RE = re.compile(
    r"(^|\s)public\s+((?:final\s+|abstract\s+)?class\s+Solution\b)",
    re.MULTILINE,
)

MAIN_JAVA_LINE_RE = re.compile(r"Main\.java:(\d+)")

FENCED_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
CLASS_SOLUTION_RE = re.compile(r"\bclass\s+Solution\b")


@dataclass(frozen=True)
class MergedJavaSource:
    code: str
    # 1-based line in ``code`` where line 1 of the user's solution lands.
    solution_start_line: int


def merge_java_solution_with_harness(solution: str, harness: str) -> MergedJavaSource:
    """Combine a candidate Java ``Solution`` with a ``Main`` test harness.

    - Hoists user ``import`` lines above the harness (deduplicated, order-preserved)
      and blanks their original lines so line numbers in ``Solution`` stay aligned.
    - Blanks any ``package ...;`` declaration in the user code.
    - Demotes ``public class Solution`` to package-private ``class Solution``.
    """
    normalized_solution = (solution or "").replace("\r\n", "\n")
    normalized_harness = (harness or "").replace("\r\n", "\n")

    solution_lines = normalized_solution.split("\n")
    hoisted_imports: list[str] = []
    stripped_lines: list[str] = []

    for line in solution_lines:
        if IMPORT_LINE_RE.match(line):
            clean_import = re.sub(r"\s*//.*$", "", line).strip()
            hoisted_imports.append(clean_import)
            stripped_lines.append("")
        elif PACKAGE_LINE_RE.match(line):
            stripped_lines.append("")
        else:
            stripped_lines.append(line)

    seen_imports: set[str] = set()
    unique_imports: list[str] = []
    for imp in hoisted_imports:
        if imp not in seen_imports:
            seen_imports.add(imp)
            unique_imports.append(imp)

    cleaned_solution = PUBLIC_SOLUTION_RE.sub(r"\1\2", "\n".join(stripped_lines))
    harness_trimmed = normalized_harness.rstrip()

    prefix_parts: list[str] = []
    if unique_imports:
        prefix_parts.append("\n".join(unique_imports))
    prefix_parts.append(harness_trimmed)

    prefix = "\n\n".join(prefix_parts)
    prefix_line_count = len(prefix.split("\n"))
    # ``prefix`` + "\n\n" + ``cleaned_solution`` -> one blank separator line, so
    # ``cleaned_solution`` starts at ``prefix_line_count + 2`` (1-indexed).
    solution_start_line = prefix_line_count + 2

    return MergedJavaSource(
        code=f"{prefix}\n\n{cleaned_solution}\n",
        solution_start_line=solution_start_line,
    )


def annotate_runtime_output(output: str | None, merged: MergedJavaSource) -> str:
    """Rewrite ``Main.java:<line>`` references in compiler/runtime output.

    Lines inside the user's solution region are annotated with
    ``(Solution.java:<userLine>)`` so error messages refer to the candidate's
    own line numbers.
    """
    if not output:
        return ""

    def _replace(match: re.Match[str]) -> str:
        line_num = int(match.group(1))
        if line_num >= merged.solution_start_line:
            user_line = line_num - merged.solution_start_line + 1
            return f"{match.group(0)} (Solution.java:{user_line})"
        return match.group(0)

    return MAIN_JAVA_LINE_RE.sub(_replace, output)


def extract_java_code(response_text: str | None) -> str | None:
    """Extract Java candidate code from an LLM response.

    Prefers fenced code blocks containing ``class Solution``. Falls back to any
    ``java`` fenced block, or raw text containing ``class Solution``.
    Returns ``None`` if no Java code can be extracted.
    """
    if not response_text or not response_text.strip():
        return None

    text = response_text.strip()
    matches = FENCED_BLOCK_RE.findall(text)

    if matches:
        # 1. Prefer blocks that contain `class Solution` (take the last one in case
        #    the model reasoned through an earlier snippet before the final block).
        solution_blocks = [
            body.strip()
            for _lang, body in matches
            if body.strip() and CLASS_SOLUTION_RE.search(body)
        ]
        if solution_blocks:
            return solution_blocks[-1]

        # 2. Otherwise check explicitly tagged ```java blocks.
        java_blocks = [
            body.strip()
            for lang, body in matches
            if lang.strip().lower() == "java" and body.strip()
        ]
        if java_blocks:
            return java_blocks[-1]

        # 3. Any non-empty code block if it looks like Java code (contains braces/semicolons).
        generic_blocks = [
            body.strip()
            for lang, body in matches
            if lang.strip() == "" and body.strip() and "{" in body and "}" in body
        ]
        if generic_blocks:
            return generic_blocks[-1]

        return None

    # No fenced block: check if raw response directly provides `class Solution`.
    if CLASS_SOLUTION_RE.search(text) and "{" in text and "}" in text:
        lines = text.splitlines()
        start_idx = 0
        for idx, line in enumerate(lines):
            if (
                IMPORT_LINE_RE.match(line)
                or PACKAGE_LINE_RE.match(line)
                or CLASS_SOLUTION_RE.search(line)
            ):
                start_idx = idx
                break
        # Trim trailing prose after the last closing brace
        end_idx = len(lines) - 1
        while end_idx > start_idx and "}" not in lines[end_idx]:
            end_idx -= 1
        candidate = "\n".join(lines[start_idx : end_idx + 1]).strip()
        if candidate:
            return candidate

    return None
