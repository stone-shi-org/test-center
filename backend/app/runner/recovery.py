"""Recovery utilities for stale runs after backend restarts."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.base import utc_now
from backend.app.db.models.run import Run
from backend.app.db.models.result import Result
from backend.app.db.models.artifact import Artifact


def recover_stale_runs(db: Session, artifacts_dir: Path) -> list[int]:
    """Find any runs left in 'running' or 'cancelling' state and mark them as error.
    
    This ensures that after a backend restart, orphaned running runs don't stay
    stuck in 'running' indefinitely.
    """
    stmt = select(Run).where(Run.status.in_(["running", "cancelling"]))
    stale_runs: Sequence[Run] = db.execute(stmt).scalars().all()

    recovered_ids: list[int] = []
    now = utc_now()

    for run in stale_runs:
        run.status = "error"
        run.finished_at = now
        recovered_ids.append(run.id)

        # Append restart notice to run.log
        run_dir = artifacts_dir / str(run.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "run.log"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{now.isoformat()}] [ERROR] Run marked as error due to backend restart.\n")
        except OSError:
            pass

        # Ensure a Result record exists indicating error
        if not run.results:
            tester_name = "unknown"
            if isinstance(run.config_snapshot, dict):
                tester_name = run.config_snapshot.get("tester", "unknown")
            elif isinstance(run.config_snapshot, list) and run.config_snapshot:
                first = run.config_snapshot[0]
                if isinstance(first, dict):
                    tester_name = first.get("tester", "unknown")

            err_result = Result(
                run_id=run.id,
                tester=tester_name,
                status="error",
                summary="Run interrupted by backend restart",
                metrics={},
            )
            db.add(err_result)
            db.flush()

            if log_file.exists():
                size = None
                try:
                    size = log_file.stat().st_size
                except OSError:
                    pass
                db.add(
                    Artifact(
                        result_id=err_result.id,
                        name="run.log",
                        path=str(log_file),
                        mime="text/plain",
                        size_bytes=size,
                    )
                )
        else:
            for r in run.results:
                if r.status in ("running", "pending"):
                    r.status = "error"
                    r.summary = (r.summary or "") + " [Interrupted by backend restart]"

    if recovered_ids:
        db.commit()

    return recovered_ids
