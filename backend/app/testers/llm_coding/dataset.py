"""Dataset loader, solution stripper, and manifest verifier for ``test-data/iqb-java/``."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from backend.app.config import BASE_DIR, get_settings
except ImportError:  # pragma: no cover - host script fallback when pydantic is not installed
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    get_settings = None  # type: ignore[assignment]

SOLUTION_HEADING_RE = re.compile(
    r"^##\s+(?:(?:\d+\.\s*)?(?:optimal\s+)?solutions?|approach(?:es)?|reference\s+solutions?)\b",
    re.IGNORECASE,
)
ANY_H2_RE = re.compile(r"^##\s+\S")
DETAILS_BLOCK_RE = re.compile(r"<details\b[^>]*>[\s\S]*?</details>", re.IGNORECASE)
SUMMARY_SOLUTION_RE = re.compile(
    r"<summary\b[^>]*>[^<]*solutions?[^<]*</summary>", re.IGNORECASE
)
TRAILING_HR_RE = re.compile(r"\n---\s*$")

DEFAULT_JUDGE_PROMPT_MD = """You are an expert Java software engineer and technical interviewer evaluating a candidate's Java solution to a coding problem.

## Problem Statement
{{statement}}

## Coding Style & Quality Rubric
{{style_rubric}}

## Reference Solution
```java
{{reference_solution}}
```

## Candidate Solution
```java
{{candidate_code}}
```

## Automated Test Outcome
{{test_outcome}}

## Evaluation Instructions
Score the candidate's Java solution on a 1 to 5 integer scale (where 1 = poor, 3 = acceptable, 5 = excellent) across three dimensions, plus an overall score (1 to 5):
1. `approach_correctness`: Soundness of the algorithm, handling of edge cases, and alignment with the test outcome.
2. `complexity_vs_reference`: Time and space complexity compared to the reference solution and problem constraints.
3. `readability_idiom`: Java idioms, naming clarity, structure, and adherence to the style rubric.
4. `overall`: Overall engineering quality (1 to 5).

Respond with ONLY a valid JSON object matching this schema:
```json
{
  "approach_correctness": 5,
  "complexity_vs_reference": 5,
  "readability_idiom": 5,
  "overall": 5,
  "justification": "Concise 1-3 sentence explanation of the scores."
}
```
"""


@dataclass
class SplitSolutionsResult:
    body: str
    solutions: list[str] = field(default_factory=list)


def split_solutions(markdown: str | None) -> SplitSolutionsResult:
    """Separate embedded solution sections from a problem statement markdown.

    Ported from ``interview-question-bank/src/lib/solutions.ts`` so exported
    statements never leak reference solutions or approaches into the prompt.
    """
    if not markdown:
        return SplitSolutionsResult(body="", solutions=[])

    normalized = markdown.replace("\r\n", "\n")
    solutions: list[str] = []

    def _replace_details(match: re.Match[str]) -> str:
        block = match.group(0)
        if SUMMARY_SOLUTION_RE.search(block):
            solutions.append(block.strip())
            return ""
        return block

    without_details = DETAILS_BLOCK_RE.sub(_replace_details, normalized)

    lines = without_details.split("\n")
    body_lines: list[str] = []
    current_solution: list[str] | None = None
    in_fence = False
    saw_solution_heading = False
    body_line_count_before_first_solution = -1

    for line in lines:
        if re.match(r"^\s*```", line):
            in_fence = not in_fence

        if not in_fence and ANY_H2_RE.match(line):
            if SOLUTION_HEADING_RE.match(line):
                if current_solution is not None:
                    solutions.append("\n".join(current_solution).strip())
                if not saw_solution_heading:
                    saw_solution_heading = True
                    body_line_count_before_first_solution = len(body_lines)
                current_solution = [line]
                continue
            elif current_solution is not None:
                solutions.append("\n".join(current_solution).strip())
                current_solution = None

        if current_solution is not None:
            current_solution.append(line)
        else:
            body_lines.append(line)

    if current_solution is not None:
        solutions.append("\n".join(current_solution).strip())

    if not solutions:
        return SplitSolutionsResult(body=markdown, solutions=[])

    if saw_solution_heading and body_line_count_before_first_solution > 0:
        before = "\n".join(body_lines[:body_line_count_before_first_solution]).rstrip()
        after = "\n".join(body_lines[body_line_count_before_first_solution:])
        cleaned_before = TRAILING_HR_RE.sub("", before)
        body = (
            f"{cleaned_before}\n{after}".strip()
            if after.strip()
            else cleaned_before.strip()
        )
        return SplitSolutionsResult(
            body=body,
            solutions=[s for s in solutions if len(s) > 0],
        )

    return SplitSolutionsResult(
        body="\n".join(body_lines).strip(),
        solutions=[s for s in solutions if len(s) > 0],
    )


def pick_sample_case_indices(
    statement_md: str,
    test_cases: list[dict[str, Any]],
    max_samples: int = 3,
) -> list[int]:
    """Choose 0-based indices of visible sample test cases for a problem.

    Guarantees that at least 1 test case remains hidden whenever ``len(test_cases) >= 2``,
    because only hidden test cases count toward scoring.
    Prefers test cases whose input or output matches an example in the statement,
    falling back to the leading test cases ``0, 1, ...``.
    """
    n = len(test_cases)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Ensure at least one hidden case remains for scoring:
    # n=2 -> 1 sample, 1 hidden; n=3 -> 1 sample, 2 hidden; n=4..5 -> 2 samples; n>=6 -> up to 3 samples.
    cap = min(max_samples, max(1, n // 2))

    matched: list[int] = []
    stmt = statement_md or ""
    if stmt:
        for idx, tc in enumerate(test_cases):
            raw_in = str(tc.get("input") or "").strip()
            raw_out = str(tc.get("expected_output") or tc.get("expected") or "").strip()
            # Check if non-trivial expected output and input fragments appear in statement examples
            first_in_line = raw_in.splitlines()[0].strip() if raw_in else ""
            if (
                first_in_line
                and len(first_in_line) >= 2
                and first_in_line in stmt
                and raw_out
                and raw_out in stmt
            ):
                matched.append(idx)
                if len(matched) >= cap:
                    break

    selected = list(matched[:cap])
    for idx in range(n):
        if len(selected) >= cap:
            break
        if idx not in selected:
            selected.append(idx)

    return sorted(selected)


@dataclass
class ProblemSpec:
    id: str
    title: str
    difficulty: str
    topics: list[str]
    source: str
    leetcode_id: str | int | None
    statement_md: str
    timeout_ms: int
    cpu_limit_ms: int | None
    style_rubric: str
    sample_case_idx: list[int]
    template_java: str
    harness_java: str
    reference_java: str
    tests: list[dict[str, str]]
    sha256: str = ""

    @property
    def sample_tests(self) -> list[dict[str, str]]:
        sample_set = set(self.sample_case_idx)
        return [tc for i, tc in enumerate(self.tests) if i in sample_set]

    @property
    def hidden_tests(self) -> list[dict[str, str]]:
        sample_set = set(self.sample_case_idx)
        return [tc for i, tc in enumerate(self.tests) if i not in sample_set]

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "difficulty": self.difficulty,
            "topics": self.topics,
            "source": self.source,
            "leetcode_id": self.leetcode_id,
            "statement_md": self.statement_md,
            "timeout_ms": self.timeout_ms,
            "cpu_limit_ms": self.cpu_limit_ms,
            "style_rubric": self.style_rubric,
            "sample_case_idx": self.sample_case_idx,
        }


@dataclass
class DatasetManifest:
    version: str
    exported_at: str
    source: str
    problem_count: int
    problems: dict[str, str]
    manifest_sha256: str = ""
    judge_prompt_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedDataset:
    root_dir: Path
    manifest: DatasetManifest
    judge_prompt_md: str
    problems: list[ProblemSpec]


PROBLEM_HASH_FILES = (
    "problem.json",
    "Solution.template.java",
    "Harness.java",
    "Reference.java",
    "tests.json",
)


def compute_problem_sha256(problem_dir: Path) -> str:
    """Compute a deterministic SHA-256 across all canonical files of a problem directory."""
    hasher = hashlib.sha256()
    for fname in PROBLEM_HASH_FILES:
        fpath = problem_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing required problem file: {fpath}")
        content = fpath.read_bytes().replace(b"\r\n", b"\n")
        hasher.update(fname.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(content)
        hasher.update(b"\0")
    return hasher.hexdigest()


def compute_problems_Rollup_sha256(problems_map: dict[str, str]) -> str:
    """Compute an overall dataset SHA-256 from the sorted per-problem SHA-256 map."""
    hasher = hashlib.sha256()
    for pid in sorted(problems_map.keys()):
        hasher.update(f"{pid}:{problems_map[pid]}\n".encode("utf-8"))
    return hasher.hexdigest()


def resolve_iqb_java_dir(dataset_dir: str | Path | None = None) -> Path:
    """Locate the ``iqb-java`` dataset directory across container, repo, and worktree paths."""
    candidates: list[Path] = []
    if dataset_dir:
        candidates.append(Path(dataset_dir))

    env_dir = os.getenv("IQB_JAVA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    if get_settings is not None:
        settings = get_settings()
        candidates.append(settings.TEST_DATA_DIR / "iqb-java")
    elif os.getenv("TEST_DATA_DIR"):
        candidates.append(Path(os.environ["TEST_DATA_DIR"]) / "iqb-java")

    candidates.extend(
        [
            BASE_DIR / "test-data" / "iqb-java",
            BASE_DIR.parent.parent / "test-data" / "iqb-java",
        ]
    )

    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()

    # Return first candidate so error messages show the expected path
    return candidates[0]


def _normalize_str_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [item.strip() for item in val.split(",") if item.strip()]
    if isinstance(val, (list, tuple, set)):
        return [str(item).strip() for item in val if str(item).strip()]
    return []


def load_problem_from_dir(problem_dir: Path) -> ProblemSpec:
    """Load a single problem from ``problems/<id>/`` and compute its SHA-256."""
    meta_path = problem_dir / "problem.json"
    template_path = problem_dir / "Solution.template.java"
    harness_path = problem_dir / "Harness.java"
    ref_path = problem_dir / "Reference.java"
    tests_path = problem_dir / "tests.json"

    for req in (meta_path, template_path, harness_path, ref_path, tests_path):
        if not req.is_file():
            raise FileNotFoundError(f"Missing file {req} in problem directory {problem_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tests_raw = json.loads(tests_path.read_text(encoding="utf-8"))
    if not isinstance(tests_raw, list) or not tests_raw:
        raise ValueError(f"Problem {problem_dir.name} has empty or invalid tests.json")

    tests: list[dict[str, str]] = []
    for tc in tests_raw:
        tests.append(
            {
                "input": str(tc.get("input") or ""),
                "expected_output": str(tc.get("expected_output") or tc.get("expected") or ""),
            }
        )

    difficulty = str(meta.get("difficulty") or "medium").strip().lower()
    raw_cpu = meta.get("cpu_limit_ms")
    cpu_limit_ms = int(raw_cpu) if isinstance(raw_cpu, (int, float)) and raw_cpu > 0 else None
    raw_timeout = meta.get("timeout_ms")
    timeout_ms = int(raw_timeout) if isinstance(raw_timeout, (int, float)) and raw_timeout > 0 else 2000

    sample_idx = meta.get("sample_case_idx")
    if isinstance(sample_idx, list):
        sample_case_idx = [int(i) for i in sample_idx if isinstance(i, (int, float)) and 0 <= int(i) < len(tests)]
    else:
        sample_case_idx = pick_sample_case_indices(str(meta.get("statement_md") or ""), tests)

    sha = compute_problem_sha256(problem_dir)

    return ProblemSpec(
        id=str(meta.get("id") or problem_dir.name),
        title=str(meta.get("title") or problem_dir.name),
        difficulty=difficulty,
        topics=_normalize_str_list(meta.get("topics")),
        source=str(meta.get("source") or "interviewqb"),
        leetcode_id=meta.get("leetcode_id"),
        statement_md=str(meta.get("statement_md") or ""),
        timeout_ms=timeout_ms,
        cpu_limit_ms=cpu_limit_ms,
        style_rubric=str(meta.get("style_rubric") or ""),
        sample_case_idx=sample_case_idx,
        template_java=template_path.read_text(encoding="utf-8"),
        harness_java=harness_path.read_text(encoding="utf-8"),
        reference_java=ref_path.read_text(encoding="utf-8"),
        tests=tests,
        sha256=sha,
    )


def load_iqb_java_dataset(
    dataset_dir: str | Path | None = None,
    verify_hash: bool = True,
    suite_id: str = "all",
    ids: list[str] | str | None = None,
    difficulty: list[str] | str | None = None,
    topics: list[str] | str | None = None,
    limit: int | None = None,
) -> LoadedDataset:
    """Load the ``iqb-java`` dataset, verify per-problem SHA-256 hashes against ``manifest.json``,
    and apply suite/config filters.
    """
    root = resolve_iqb_java_dir(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"iqb-java dataset manifest not found at {manifest_path}. "
            "Run scripts/export_iqb_java.py or set dataset_dir / IQB_JAVA_DIR."
        )

    raw_manifest_bytes = manifest_path.read_bytes()
    raw_manifest = json.loads(raw_manifest_bytes.decode("utf-8"))

    raw_problems = raw_manifest.get("problems") or {}
    problems_sha_map: dict[str, str] = {}
    if isinstance(raw_problems, dict):
        for pid, val in raw_problems.items():
            if isinstance(val, str):
                problems_sha_map[pid] = val
            elif isinstance(val, dict) and isinstance(val.get("sha256"), str):
                problems_sha_map[pid] = val["sha256"]

    computed_rollup = compute_problems_Rollup_sha256(problems_sha_map)
    recorded_manifest_sha = str(raw_manifest.get("manifest_sha256") or "")
    if verify_hash and recorded_manifest_sha and recorded_manifest_sha != computed_rollup:
        raise ValueError(
            f"Manifest rollup SHA-256 mismatch in {manifest_path}: "
            f"expected {recorded_manifest_sha}, got {computed_rollup}"
        )

    manifest = DatasetManifest(
        version=str(raw_manifest.get("version") or "1.0"),
        exported_at=str(raw_manifest.get("exported_at") or ""),
        source=str(raw_manifest.get("source") or "interviewqb"),
        problem_count=int(raw_manifest.get("problem_count") or len(problems_sha_map)),
        problems=problems_sha_map,
        manifest_sha256=recorded_manifest_sha or computed_rollup,
        judge_prompt_sha256=str(raw_manifest.get("judge_prompt_sha256") or ""),
    )

    if verify_hash and manifest.problem_count != len(problems_sha_map):
        raise ValueError(
            f"Manifest problem_count ({manifest.problem_count}) does not match "
            f"problems map length ({len(problems_sha_map)})"
        )

    judge_prompt_path = root / "judge_prompt.md"
    if judge_prompt_path.is_file():
        judge_prompt_md = judge_prompt_path.read_text(encoding="utf-8")
        if verify_hash and manifest.judge_prompt_sha256:
            actual_jp_sha = hashlib.sha256(
                judge_prompt_path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            if actual_jp_sha != manifest.judge_prompt_sha256:
                raise ValueError(
                    f"judge_prompt.md SHA-256 mismatch: expected {manifest.judge_prompt_sha256}, got {actual_jp_sha}"
                )
    else:
        judge_prompt_md = DEFAULT_JUDGE_PROMPT_MD

    problems_dir = root / "problems"
    id_filter = set(_normalize_str_list(ids))
    diff_filter = {d.lower() for d in _normalize_str_list(difficulty)}
    suite_norm = (suite_id or "all").strip().lower()
    if suite_norm in ("easy", "medium", "hard"):
        diff_filter.add(suite_norm)
    topic_filter = {t.lower() for t in _normalize_str_list(topics)}

    loaded_problems: list[ProblemSpec] = []
    problem_ids_ordered = sorted(problems_sha_map.keys())
    for pid in problem_ids_ordered:
        if id_filter and pid not in id_filter:
            continue
        pdir = problems_dir / pid
        if not pdir.is_dir():
            if verify_hash:
                raise FileNotFoundError(f"Problem directory missing for {pid}: {pdir}")
            continue

        prob = load_problem_from_dir(pdir)
        expected_sha = problems_sha_map.get(pid)
        if verify_hash and expected_sha and prob.sha256 != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for problem '{pid}': "
                f"manifest has {expected_sha}, computed {prob.sha256}"
            )

        if diff_filter and prob.difficulty.lower() not in diff_filter:
            continue

        if topic_filter:
            prob_topics_lower = {t.lower() for t in prob.topics}
            if not (prob_topics_lower & topic_filter):
                continue

        loaded_problems.append(prob)

    if limit is not None and int(limit) > 0:
        loaded_problems = loaded_problems[: int(limit)]

    return LoadedDataset(
        root_dir=root,
        manifest=manifest,
        judge_prompt_md=judge_prompt_md,
        problems=loaded_problems,
    )
