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
# Any other public top-level type (column 0), e.g. ``public class LRUCache`` for a template
# without ``Solution``: single-file source launch only allows the public class ``Main``.
PUBLIC_TOP_LEVEL_TYPE_RE = re.compile(
    r"^public\s+((?:final\s+|abstract\s+)?(?:class|interface|enum|record)\s+(?!Main\b)\w+)",
    re.MULTILINE,
)

MAIN_JAVA_LINE_RE = re.compile(r"Main\.java:(\d+)")

FENCED_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
CLASS_SOLUTION_RE = re.compile(r"\bclass\s+Solution\b")

# Unfenced fallback: a line that *declares* ``class Solution`` (not prose that
# merely mentions it), and other top-level type declarations that may follow it.
_ANNOTATIONS = r"(?:@\w+(?:\([^)]*\))?\s+)*"
SOLUTION_DECL_LINE_RE = re.compile(
    r"^\s*" + _ANNOTATIONS + r"(?:(?:public|final|abstract|static)\s+)*class\s+Solution\b"
    r"(?:\s+(?:extends|implements)\s+[\w<>,.\s]+?)?\s*(?:\{.*)?$"
)
TOP_LEVEL_DECL_LINE_RE = re.compile(
    r"^\s*" + _ANNOTATIONS
    + r"(?:(?:public|private|protected|final|abstract|static|sealed|non-sealed)\s+)*"
    + r"(?:class|interface|enum|record)\s+\w+"
)
# A real solution is a few KB. Anything far larger from the unfenced path is
# reasoning prose, and would also exceed Piston's 100 KiB request-body limit.
MAX_UNFENCED_CODE_CHARS = 64_000


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
    - Demotes ``public class Solution`` (and any other public top-level type except ``Main``) to package-private.
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
    cleaned_solution = PUBLIC_TOP_LEVEL_TYPE_RE.sub(r"\1", cleaned_solution)
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

    # No fenced block: accept raw text only if it holds a real, brace-balanced
    # ``class Solution`` declaration (see ``_extract_unfenced_solution``).
    return _extract_unfenced_solution(text)


def _block_end(lines: list[str], start: int) -> int | None:
    """Index of the line closing the brace block opened at/after ``lines[start]``.

    Skips braces inside ``//`` and ``/* */`` comments and string/char literals.
    Returns ``None`` if the block never opens or never closes (e.g. a response cut
    off mid-code, or prose that only looks like a declaration).
    """
    depth = 0
    opened = False
    in_block_comment = False
    for i in range(start, len(lines)):
        line = lines[i]
        j = 0
        quote: str | None = None
        while j < len(line):
            ch = line[j]
            nxt = line[j + 1] if j + 1 < len(line) else ""
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    j += 1
            elif quote:
                if ch == "\\":
                    j += 1
                elif ch == quote:
                    quote = None
            elif ch == "/" and nxt == "/":
                break
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                j += 1
            elif ch in "\"'":
                quote = ch
            elif ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    return i
                if depth < 0:
                    return None
            j += 1
        if not opened and i > start + 2:
            return None
    return None


def _is_preamble_line(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or bool(IMPORT_LINE_RE.match(line))
        or bool(PACKAGE_LINE_RE.match(line))
        or stripped.startswith("//")
        or stripped.startswith("@")
    )


def _extract_unfenced_solution(text: str) -> str | None:
    """Extract an unfenced ``class Solution`` (plus imports and trailing helper types).

    Only a line that actually declares ``class Solution`` starts a candidate, and the
    candidate ends at its matching closing brace (extended over any top-level helper
    types right after it). The last balanced candidate wins, mirroring the fenced
    path. Truncated or oversized candidates are rejected, so unfinished reasoning
    that merely mentions ``class Solution`` yields ``None`` instead of a huge blob.
    """
    lines = text.splitlines()
    best: str | None = None
    for decl_idx, line in enumerate(lines):
        if not SOLUTION_DECL_LINE_RE.match(line):
            continue
        end = _block_end(lines, decl_idx)
        if end is None:
            continue
        start = decl_idx
        while start > 0 and _is_preamble_line(lines[start - 1]):
            start -= 1
        nxt = end + 1
        while nxt < len(lines):
            if not lines[nxt].strip():
                nxt += 1
                continue
            if not TOP_LEVEL_DECL_LINE_RE.match(lines[nxt]):
                break
            helper_end = _block_end(lines, nxt)
            if helper_end is None:
                break
            end = helper_end
            nxt = helper_end + 1
        candidate = "\n".join(lines[start : end + 1]).strip()
        if candidate and len(candidate) <= MAX_UNFENCED_CODE_CHARS:
            best = candidate
    return best
