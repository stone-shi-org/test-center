#!/usr/bin/env python3
"""Build ``test-data/iqb-java/subsets.json`` from the latest llm_coding run of each model.

Subsets (problems present in every selected run and in the dataset):

- ``baseline``       solved by every model. Failures here are flagged in reports
                     (``expect_all_solved``): a capable model failing one is usually a setup problem.
- ``discriminating`` not solved by every model: where the ranking actually comes from.
- ``hard_core``      failed by at least ``--hard-min-failures`` models (default 2).

``subsets.json`` is not part of the manifest or any problem hash, so rebuilding it never changes
cache keys. Re-run this after adding models or changing the dataset.

Usage:
  python scripts/build_iqb_java_subsets.py [--api URL] [--runs 109,111,...] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API = "https://test-center.local.shifamily.com"
DEFAULT_DATASET = REPO_ROOT / "test-data" / "iqb-java"


def _get(api: str, path: str):
    with urllib.request.urlopen(f"{api.rstrip('/')}{path}", timeout=120) as resp:
        return json.load(resp)


def _model(run: dict) -> str | None:
    cs = run.get("config_snapshot") or {}
    return (cs.get("config") or {}).get("model") if isinstance(cs, dict) else None


def latest_run_per_model(api: str) -> list[int]:
    runs = _get(api, "/api/runs?tester=llm_coding&limit=500&view=summary")
    latest: dict[str, dict] = {}
    for r in runs:
        model = _model(r)
        if r.get("status") != "completed" or not model:
            continue
        if model not in latest or r["id"] > latest[model]["id"]:
            latest[model] = r
    return sorted(r["id"] for r in latest.values())


def build(api: str, run_ids: list[int], dataset_dir: Path, hard_min_failures: int) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_ids = set(manifest.get("problems") or {})
    passed: dict[int, dict[str, bool]] = {}
    sources = []
    for run_id in run_ids:
        results = _get(api, f"/api/runs/{run_id}/results")
        if not results:
            raise SystemExit(f"run {run_id} has no results")
        res = results[0]
        cases = res.get("cases") or []
        passed[run_id] = {c["case_id"]: c.get("status") == "pass" for c in cases}
        run = _get(api, f"/api/runs/{run_id}")
        sources.append({
            "run_id": run_id,
            "model": _model(run),
            "score": (res.get("metrics") or {}).get("score", {}).get("value"),
            "dataset_versions": sorted({str((c.get("detail") or {}).get("dataset_version")) for c in cases}),
        })
    common = set(dataset_ids)
    for p in passed.values():
        common &= set(p)
    n_models = len(run_ids)
    solved_by = {pid: sum(1 for p in passed.values() if p[pid]) for pid in common}
    baseline = sorted(pid for pid, n in solved_by.items() if n == n_models)
    discriminating = sorted(pid for pid, n in solved_by.items() if n < n_models)
    hard_core = sorted(pid for pid, n in solved_by.items() if n_models - n >= hard_min_failures)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_version": manifest.get("version"),
        "source_runs": sources,
        "problems_considered": len(common),
        "problems_not_considered": sorted(dataset_ids - common),
        "subsets": {
            "baseline": {
                "description": f"Solved by all {n_models} models in the source runs; failures here are red flags.",
                "expect_all_solved": True,
                "problems": baseline,
            },
            "discriminating": {
                "description": f"Not solved by every one of the {n_models} models; the ranking comes from these.",
                "problems": discriminating,
            },
            "hard_core": {
                "description": f"Failed by at least {hard_min_failures} of the {n_models} models.",
                "problems": hard_core,
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--runs", default="", help="Comma-separated run ids (default: latest completed run per model)")
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    ap.add_argument("--output", default="", help="Default: <dataset-dir>/subsets.json")
    ap.add_argument("--hard-min-failures", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_ids = [int(x) for x in args.runs.split(",") if x.strip()] or latest_run_per_model(args.api)
    data = build(args.api, run_ids, Path(args.dataset_dir), args.hard_min_failures)
    for name, sub in data["subsets"].items():
        print(f"{name:15} {len(sub['problems']):4}  {sub['description']}")
    print(f"source runs: {[(s['run_id'], s['model']) for s in data['source_runs']]}")
    if data["problems_not_considered"]:
        print(f"not in every run (unassigned): {data['problems_not_considered']}")
    if not args.dry_run:
        out = Path(args.output) if args.output else Path(args.dataset_dir) / "subsets.json"
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
