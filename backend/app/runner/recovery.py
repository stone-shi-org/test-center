"""Recovery utilities for stale runs after backend restarts."""

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.base import utc_now
from backend.app.db.models.artifact import Artifact
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run

logger = logging.getLogger(__name__)

RESTART_SUMMARY = "Run interrupted by backend restart"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _checkpoint_dirs(run_dir: Path) -> list[Path]:
    """Directories holding a tester checkpoint (``result.json``) for this run.

    Single-tester runs write ``<run>/result.json``; multi-step jobs write one per
    ``<run>/step_<n>_<tester>/`` directory.
    """
    if (run_dir / "result.json").is_file():
        return [run_dir]
    if not run_dir.is_dir():
        return []
    steps = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step_") and (d / "result.json").is_file()]

    def _step_no(d: Path) -> int:
        try:
            return int(d.name.split("_")[1])
        except (IndexError, ValueError):
            return 1_000_000

    return sorted(steps, key=_step_no)


def _snapshot_tester(run: Run) -> tuple[str, Any]:
    tester_name, test_def_id = "unknown", None
    if isinstance(run.config_snapshot, dict):
        tester_name = run.config_snapshot.get("tester", "unknown")
        test_def_id = run.config_snapshot.get("test_definition_id")
    elif isinstance(run.config_snapshot, list) and run.config_snapshot:
        first = run.config_snapshot[0]
        if isinstance(first, dict):
            tester_name = first.get("tester", "unknown")
            test_def_id = first.get("id") or first.get("test_definition_id")
    return tester_name, test_def_id


def _add_file_artifact(db: Session, result: Result, path: Path, name: str, mime: str, known: set[str]) -> None:
    if not path.exists() or str(path) in known:
        return
    known.add(str(path))
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    db.add(Artifact(result_id=result.id, name=name, path=str(path), mime=mime, size_bytes=size))


def _attach_checkpoint(db: Session, result: Result, ckpt_dir: Path, data: dict[str, Any], log_file: Path) -> int:
    """Copy checkpointed cases/artifacts into ``result``. Returns the number of cases added."""
    added = 0
    if not result.cases:
        for c in data.get("cases") or []:
            if not isinstance(c, dict):
                continue
            db.add(
                CaseResult(
                    result_id=result.id,
                    suite=c.get("suite", "default"),
                    case_id=c.get("case_id", "unknown"),
                    status=c.get("status", "pass"),
                    metrics=c.get("metrics", {}),
                    detail=c.get("detail", {}),
                )
            )
            added += 1
    known = {a.path for a in (result.artifacts or [])}
    for a in data.get("artifacts") or []:
        if isinstance(a, dict) and a.get("path"):
            p = Path(a["path"])
            if str(p) not in known:
                known.add(str(p))
                db.add(
                    Artifact(
                        result_id=result.id,
                        name=a.get("name", p.name),
                        path=str(p),
                        mime=a.get("mime", "application/octet-stream"),
                        size_bytes=a.get("size_bytes"),
                    )
                )
    step_log = ckpt_dir / "run.log"
    _add_file_artifact(db, result, step_log if step_log.exists() else log_file, "run.log", "text/plain", known)
    _add_file_artifact(db, result, ckpt_dir / "result.json", "result.json", "application/json", known)
    return added


def _recovered_summary(data: dict[str, Any]) -> str:
    cases = [c for c in data.get("cases") or [] if isinstance(c, dict)]
    if not cases:
        return RESTART_SUMMARY
    passed = sum(1 for c in cases if c.get("status") == "pass")
    return f"{RESTART_SUMMARY}; {len(cases)} completed cases recovered ({passed} passed)"


def recover_stale_runs(db: Session, artifacts_dir: Path) -> list[int]:
    """Find any runs left in 'running' or 'cancelling' state and mark them as error.

    Results the tester had already checkpointed to ``result.json`` (written after every
    completed case) are loaded as partial results, the same way a cancelled run keeps them,
    so a restart mid-run does not discard hours of completed work.
    """
    stmt = select(Run).where(Run.status.in_(["running", "cancelling"]))
    stale_runs: Sequence[Run] = db.execute(stmt).scalars().all()

    recovered_ids: list[int] = []
    now = utc_now()

    for run in stale_runs:
        run.status = "error"
        run.finished_at = now
        recovered_ids.append(run.id)

        run_dir = artifacts_dir / str(run.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "run.log"
        checkpoints = [(d, _load_json(d / "result.json")) for d in _checkpoint_dirs(run_dir)]
        checkpoints = [(d, data) for d, data in checkpoints if data]
        n_cases = sum(len(data.get("cases") or []) for _, data in checkpoints)

        note = "Run marked as error due to backend restart."
        if n_cases:
            note += f" Recovered {n_cases} completed cases from checkpoint."
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{now.isoformat()}] [ERROR] {note}\n")
        except OSError:
            pass

        if not run.results:
            tester_name, test_def_id = _snapshot_tester(run)
            if not checkpoints:
                err_result = Result(
                    run_id=run.id,
                    test_definition_id=test_def_id,
                    tester=tester_name,
                    status="error",
                    summary=RESTART_SUMMARY,
                    metrics={},
                )
                db.add(err_result)
                db.flush()
                _add_file_artifact(db, err_result, log_file, "run.log", "text/plain", set())
                continue

            for ckpt_dir, data in checkpoints:
                payload = _load_json(ckpt_dir / "payload.json")
                result = Result(
                    run_id=run.id,
                    test_definition_id=test_def_id,
                    tester=str(payload.get("tester") or tester_name),
                    status="error",
                    summary=_recovered_summary(data),
                    metrics=data.get("metrics") or {},
                )
                db.add(result)
                db.flush()
                _attach_checkpoint(db, result, ckpt_dir, data, log_file)
        else:
            for r in run.results:
                if r.status in ("running", "pending"):
                    r.status = "error"
                    r.summary = (r.summary or "") + " [Interrupted by backend restart]"
            # Single-result runs: fill in cases from the checkpoint if none were persisted.
            if len(run.results) == 1 and len(checkpoints) == 1:
                result = run.results[0]
                if not result.cases:
                    ckpt_dir, data = checkpoints[0]
                    if not result.metrics and data.get("metrics"):
                        result.metrics = data["metrics"]
                    if _attach_checkpoint(db, result, ckpt_dir, data, log_file):
                        result.summary = _recovered_summary(data)

    if recovered_ids:
        db.commit()
        try:
            from backend.app.runner.junit import write_run_junit_artifact

            for run_id in recovered_ids:
                write_run_junit_artifact(db, run_id, artifacts_dir)
        except Exception as exc:  # junit export is best-effort
            logger.warning(f"Failed to write junit.xml for recovered runs: {exc}")

    return recovered_ids
