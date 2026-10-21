#!/usr/bin/env python3
"""Read-only export and Piston re-verification script for the ``iqb-java`` dataset.

Queries the ``interviewqb`` Postgres database in strictly read-only mode for coding
problems with an accepted Java submission by ``stonexp@gmail.com``, strips embedded
solution sections from the markdown statements, selects visible sample indices,
re-verifies each reference solution against all test cases on the dedicated Piston
sidecar, and writes ``test-data/iqb-java/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import types

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Allow running on host Python without triggering backend/app/testers/__init__.py
# (which imports SQLAlchemy/Pydantic models only installed inside the container).
for pkg_name, pkg_path in (
    ("backend", BACKEND_DIR),
    ("backend.app", BACKEND_DIR / "app"),
    ("backend.app.testers", BACKEND_DIR / "app" / "testers"),
    ("backend.app.testers.llm_coding", BACKEND_DIR / "app" / "testers" / "llm_coding"),
):
    if pkg_name not in sys.modules:
        mod = types.ModuleType(pkg_name)
        mod.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = mod

from backend.app.testers.llm_coding.dataset import (  # noqa: E402
    DEFAULT_JUDGE_PROMPT_MD,
    compute_problem_sha256,
    compute_problems_Rollup_sha256,
    pick_sample_case_indices,
    split_solutions,
)
from backend.app.testers.llm_coding.piston import (  # noqa: E402
    PistonClient,
    judge_java_solution,
)


READ_ONLY_SQL = """
WITH latest_accepted AS (
    SELECT DISTINCT ON (s.question_id)
        s.question_id,
        s.id AS submission_id,
        s.code AS reference_code,
        s.created_at AS submitted_at
    FROM submissions s
    WHERE LOWER(s.user_email) = LOWER($1)
      AND s.language = 'java'
      AND s.passed = true
    ORDER BY s.question_id, s.created_at DESC, s.id DESC
)
SELECT
    q.id,
    q.title,
    q.difficulty,
    q.topics,
    q.source,
    q.leetcode_id,
    q.content,
    q.is_coding,
    q.template_code,
    q.test_harness,
    q.test_cases,
    q.test_timeout_value,
    q.coding_execution_time_expectation,
    q.coding_style_expectation,
    la.submission_id,
    la.reference_code,
    la.submitted_at
FROM latest_accepted la
JOIN questions q ON q.id = la.question_id
ORDER BY q.id ASC;
"""


def fetch_candidates_readonly(
    pg_host: str,
    pg_port: int,
    pg_db: str,
    pg_user: str,
    pg_password: str,
    user_email: str,
) -> list[dict[str, Any]]:
    """Fetch candidate rows in strictly read-only transaction mode."""
    try:
        import psycopg2  # type: ignore[import-untyped]
        import psycopg2.extras  # type: ignore[import-untyped]

        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            dbname=pg_db,
            user=pg_user,
            password=pg_password,
        )
        try:
            conn.set_session(readonly=True, autocommit=True)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(READ_ONLY_SQL.replace("$1", "%s"), (user_email,))
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    except ImportError:
        pass

    node_script = """
const { Client } = require("pg");
(async () => {
  const client = new Client({
    host: process.env.EXPORT_PG_HOST,
    port: Number(process.env.EXPORT_PG_PORT || 5432),
    database: process.env.EXPORT_PG_DB,
    user: process.env.EXPORT_PG_USER,
    password: process.env.EXPORT_PG_PASSWORD,
  });
  await client.connect();
  try {
    await client.query("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY");
    await client.query("SET default_transaction_read_only = on");
    const res = await client.query(process.env.EXPORT_SQL, [process.env.EXPORT_EMAIL]);
    process.stdout.write(JSON.stringify(res.rows));
  } finally {
    await client.end();
  }
})().catch(err => {
  console.error(err);
  process.exit(1);
});
"""
    env = os.environ.copy()
    iqb_node_modules = "/data/homes/stoneshi/src/interview-question-bank/node_modules"
    if os.path.isdir(iqb_node_modules):
        env["NODE_PATH"] = iqb_node_modules
    env["EXPORT_PG_HOST"] = pg_host
    env["EXPORT_PG_PORT"] = str(pg_port)
    env["EXPORT_PG_DB"] = pg_db
    env["EXPORT_PG_USER"] = pg_user
    env["EXPORT_PG_PASSWORD"] = pg_password
    env["EXPORT_EMAIL"] = user_email
    env["EXPORT_SQL"] = READ_ONLY_SQL

    proc = subprocess.run(
        ["node", "-e", node_script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _parse_json_field(val: Any, default: Any) -> Any:
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return default
    return default


def export_dataset(
    output_dir: Path,
    piston_url: str,
    pg_host: str,
    pg_port: int,
    pg_db: str,
    pg_user: str,
    pg_password: str,
    user_email: str = "stonexp@gmail.com",
    dataset_version: str = "1.0",
) -> dict[str, Any]:
    print(f"[export] Fetching solved Java candidates for {user_email} from {pg_host}:{pg_port}/{pg_db} (read-only)...")
    rows = fetch_candidates_readonly(
        pg_host=pg_host,
        pg_port=pg_port,
        pg_db=pg_db,
        pg_user=pg_user,
        pg_password=pg_password,
        user_email=user_email,
    )
    print(f"[export] Found {len(rows)} candidate questions with accepted Java submissions.")

    client = PistonClient(base_url=piston_url)
    runtime_info = client.ensure_java_runtime()
    print(f"[export] Piston sidecar at {piston_url} ready (Java runtime: {runtime_info}).")

    problems_root = output_dir / "problems"
    if problems_root.exists():
        shutil.rmtree(problems_root)
    problems_root.mkdir(parents=True, exist_ok=True)

    judge_prompt_path = output_dir / "judge_prompt.md"
    if not judge_prompt_path.is_file():
        judge_prompt_path.write_text(DEFAULT_JUDGE_PROMPT_MD, encoding="utf-8")
    judge_prompt_sha256 = hashlib.sha256(
        judge_prompt_path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()

    exported_problems: dict[str, str] = {}
    problem_summaries: list[dict[str, Any]] = []
    dropped_problems: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("id") or "").strip()
        title = str(row.get("title") or qid).strip()
        is_coding = int(row.get("is_coding") or 0)
        template_map = _parse_json_field(row.get("template_code"), {})
        harness_map = _parse_json_field(row.get("test_harness"), {})
        raw_tests = _parse_json_field(row.get("test_cases"), [])

        template_java = str((template_map or {}).get("java") or "").strip()
        harness_java = str((harness_map or {}).get("java") or "").strip()
        reference_java = str(row.get("reference_code") or "").strip()

        if is_coding != 1:
            dropped_problems.append({"id": qid, "title": title, "reason": "is_coding != 1"})
            continue
        if not template_java:
            dropped_problems.append({"id": qid, "title": title, "reason": "Missing template_code.java"})
            continue
        if not harness_java:
            dropped_problems.append({"id": qid, "title": title, "reason": "Missing test_harness.java"})
            continue
        if not isinstance(raw_tests, list) or len(raw_tests) == 0:
            dropped_problems.append({"id": qid, "title": title, "reason": "Empty test_cases"})
            continue
        if not reference_java:
            dropped_problems.append({"id": qid, "title": title, "reason": "Empty reference Java submission"})
            continue

        normalized_tests: list[dict[str, str]] = []
        for tc in raw_tests:
            if isinstance(tc, dict):
                normalized_tests.append(
                    {
                        "input": str(tc.get("input") or ""),
                        "expected_output": str(tc.get("expected_output") or tc.get("expected") or ""),
                    }
                )
        if not normalized_tests:
            dropped_problems.append({"id": qid, "title": title, "reason": "No valid test cases after normalization"})
            continue

        raw_content = str(row.get("content") or "")
        split_res = split_solutions(raw_content)
        statement_md = split_res.body.strip()

        # Verify no solution heading remains in statement_md
        if "\n## Solution" in statement_md or "\n## Optimal Solution" in statement_md:
            dropped_problems.append({"id": qid, "title": title, "reason": "Solution section could not be cleanly stripped"})
            continue

        timeout_ms = max(int(row.get("test_timeout_value") or 3000), 3000)
        cpu_expectations = _parse_json_field(row.get("coding_execution_time_expectation"), {})
        raw_java_cpu = (cpu_expectations or {}).get("java") if isinstance(cpu_expectations, dict) else None
        cpu_limit_ms = int(raw_java_cpu) if isinstance(raw_java_cpu, (int, float)) and raw_java_cpu > 0 else None
        if cpu_limit_ms is not None and cpu_limit_ms < 3000:
            cpu_limit_ms = 3000

        sample_case_idx = pick_sample_case_indices(statement_md, normalized_tests)

        # Re-verify Reference.java on Piston against ALL test cases
        judge_res = judge_java_solution(
            client=client,
            code=reference_java,
            harness=harness_java,
            test_cases=normalized_tests,
            sample_case_idx=sample_case_idx,
            timeout_ms=timeout_ms,
            cpu_limit_ms=cpu_limit_ms,
            stop_on_first_failure=False,
        )

        # Check that every single test case (both sample and hidden) passed
        all_cases_passed = (
            not judge_res.compilation_failed
            and judge_res.total_cases == len(normalized_tests)
            and judge_res.passed_cases == len(normalized_tests)
        )
        if not all_cases_passed:
            if judge_res.compilation_failed and judge_res.compile_error_output:
                first_err = " | ".join(
                    line.strip()
                    for line in judge_res.compile_error_output.strip().splitlines()[:3]
                    if line.strip()
                )
                detail_str = first_err
            else:
                failed_details = [
                    f"case #{r.case_number} ({r.status_detail or r.status})"
                    for r in judge_res.test_results
                    if not r.passed
                ]
                detail_str = ", ".join(failed_details[:3])
            reason = (
                f"Reference failed Piston re-verification: status={judge_res.status}, "
                f"passed={judge_res.passed_cases}/{judge_res.total_cases}; {detail_str}"
            )
            print(f"  [{idx:02d}/{len(rows):02d}] DROP {qid}: {reason}")
            dropped_problems.append(
                {
                    "id": qid,
                    "title": title,
                    "reason": reason,
                    "status": judge_res.status,
                    "passed_cases": judge_res.passed_cases,
                    "total_cases": judge_res.total_cases,
                }
            )
            continue

        difficulty = str(row.get("difficulty") or "medium").strip().lower()
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"

        topics_raw = _parse_json_field(row.get("topics"), [])
        topics = [str(t).strip() for t in topics_raw if str(t).strip()] if isinstance(topics_raw, list) else []
        source = str(row.get("source") or "interviewqb").strip()
        leetcode_id = row.get("leetcode_id")
        style_rubric = str(row.get("coding_style_expectation") or "").strip()

        pdir = problems_root / qid
        pdir.mkdir(parents=True, exist_ok=True)

        problem_json = {
            "id": qid,
            "title": title,
            "difficulty": difficulty,
            "topics": topics,
            "source": source,
            "leetcode_id": leetcode_id,
            "statement_md": statement_md,
            "timeout_ms": timeout_ms,
            "cpu_limit_ms": cpu_limit_ms,
            "style_rubric": style_rubric,
            "sample_case_idx": sample_case_idx,
        }

        (pdir / "problem.json").write_text(
            json.dumps(problem_json, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (pdir / "Solution.template.java").write_text(
            template_java + "\n", encoding="utf-8"
        )
        (pdir / "Harness.java").write_text(harness_java + "\n", encoding="utf-8")
        (pdir / "Reference.java").write_text(reference_java + "\n", encoding="utf-8")
        (pdir / "tests.json").write_text(
            json.dumps(normalized_tests, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        prob_sha = compute_problem_sha256(pdir)
        exported_problems[qid] = prob_sha
        problem_summaries.append(
            {
                "id": qid,
                "title": title,
                "difficulty": difficulty,
                "total_cases": len(normalized_tests),
                "sample_cases": len(sample_case_idx),
                "hidden_cases": len(normalized_tests) - len(sample_case_idx),
                "max_cpu_time_ms": judge_res.max_cpu_time_ms,
                "sha256": prob_sha,
                "stripped_solutions_count": len(split_res.solutions),
            }
        )
        print(
            f"  [{idx:02d}/{len(rows):02d}] OK   {qid} ({difficulty}, {len(normalized_tests)} cases, "
            f"samples={sample_case_idx}, max_cpu={judge_res.max_cpu_time_ms:.0f}ms)"
        )

    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest_sha256 = compute_problems_Rollup_sha256(exported_problems)
    manifest_data = {
        "version": dataset_version,
        "exported_at": exported_at,
        "source": f"interviewqb ({pg_host}:{pg_port}/{pg_db}, user={user_email})",
        "problem_count": len(exported_problems),
        "manifest_sha256": manifest_sha256,
        "judge_prompt_sha256": judge_prompt_sha256,
        "problems": exported_problems,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    by_diff: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for s in problem_summaries:
        d = s["difficulty"]
        by_diff[d] = by_diff.get(d, 0) + 1

    report_data = {
        "exported_at": exported_at,
        "source_db": f"{pg_host}:{pg_port}/{pg_db}",
        "solved_by": user_email,
        "piston_url": piston_url,
        "candidates_total": len(rows),
        "exported_total": len(exported_problems),
        "dropped_total": len(dropped_problems),
        "by_difficulty": by_diff,
        "manifest_sha256": manifest_sha256,
        "dropped_problems": dropped_problems,
        "exported_problems": problem_summaries,
    }
    (output_dir / "export_report.json").write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "IQB-JAVA DATASET EXPORT & PISTON RE-VERIFICATION REPORT",
        "=======================================================",
        f"Exported At    : {exported_at}",
        f"Source DB      : {pg_host}:{pg_port}/{pg_db} (read-only)",
        f"Solved By      : {user_email}",
        f"Piston Runner  : {piston_url}",
        f"Candidates     : {len(rows)}",
        f"Verified/Kept  : {len(exported_problems)} (easy={by_diff.get('easy', 0)}, medium={by_diff.get('medium', 0)}, hard={by_diff.get('hard', 0)})",
        f"Dropped        : {len(dropped_problems)}",
        f"Manifest SHA256: {manifest_sha256}",
        "",
    ]
    if dropped_problems:
        lines.append("Dropped Problems:")
        for dp in dropped_problems:
            lines.append(f"  - {dp['id']} ({dp.get('title', '')}): {dp['reason']}")
    else:
        lines.append("Dropped Problems: None (all candidate reference solutions passed 100% of test cases on Piston).")

    (output_dir / "export_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return report_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and re-verify iqb-java dataset from interviewqb.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "test-data" / "iqb-java"),
        help="Destination directory (default: test-data/iqb-java)",
    )
    parser.add_argument(
        "--piston-url",
        default=os.getenv("PISTON_URL", "http://10.100.1.65:2000"),
        help="Dedicated Piston sidecar URL for re-verification",
    )
    parser.add_argument("--pg-host", default=os.getenv("POSTGRES_HOST", "10.100.0.50"))
    parser.add_argument("--pg-port", type=int, default=int(os.getenv("POSTGRES_PORT", "5432")))
    parser.add_argument("--pg-db", default=os.getenv("POSTGRES_DB", "interviewqb"))
    parser.add_argument("--pg-user", default=os.getenv("POSTGRES_USER", "postgres"))
    parser.add_argument(
        "--pg-password",
        default=os.getenv("POSTGRES_PASSWORD", "736fea418879e23b24c09570d4f4c557"),
    )
    parser.add_argument("--email", default="stonexp@gmail.com")
    args = parser.parse_args()

    export_dataset(
        output_dir=Path(args.output_dir),
        piston_url=args.piston_url,
        pg_host=args.pg_host,
        pg_port=args.pg_port,
        pg_db=args.pg_db,
        pg_user=args.pg_user,
        pg_password=args.pg_password,
        user_email=args.email,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
