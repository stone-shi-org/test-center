"""Background worker that polls queued runs and executes them in subprocesses."""

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import get_settings
from backend.app.db.base import utc_now
from backend.app.db.models.run import Run
from backend.app.db.models.result import Result
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.artifact import Artifact
from backend.app.db.session import SessionLocal
from backend.app.runner.events import get_broadcaster
from backend.app.runner.recovery import recover_stale_runs

logger = logging.getLogger(__name__)


class RunWorker:
    """Worker polling DB for queued runs and executing them in isolated subprocesses."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] = SessionLocal,
        artifacts_dir: Path | None = None,
        poll_interval: float = 1.0,
        default_timeout: float = 300.0,
    ) -> None:
        self.session_factory = session_factory
        self.settings = get_settings()
        self.artifacts_dir = Path(artifacts_dir or self.settings.ARTIFACTS_DIR)
        self.poll_interval = poll_interval
        self.default_timeout = default_timeout

        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._active_processes: dict[int, asyncio.subprocess.Process] = {}
        self._cancel_requests: set[int] = set()

    def trigger(self) -> None:
        """Wake up worker loop immediately when a new run is queued."""
        self._wake_event.set()

    def start(self) -> None:
        """Start the background worker polling task."""
        if self._loop_task is None or self._loop_task.done():
            self._stop_event.clear()
            self._loop_task = asyncio.create_task(self._worker_loop())
            logger.info("RunWorker started")

    async def stop(self) -> None:
        """Stop background worker cleanly."""
        self._stop_event.set()
        self._wake_event.set()
        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._loop_task.cancel()
            self._loop_task = None
        logger.info("RunWorker stopped")

    async def _worker_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            executed = False
            try:
                executed = await self.poll_and_execute_once()
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)

            if not executed and not self._stop_event.is_set():
                # Wait for wake event or timeout
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
                self._wake_event.clear()

    async def poll_and_execute_once(self) -> bool:
        """Poll for next queued run and execute it. Returns True if a run was executed."""
        run_id = self._pick_next_queued_run()
        if run_id is None:
            return False

        await self.execute_run(run_id)
        return True

    def _pick_next_queued_run(self) -> int | None:
        """Find the oldest queued/pending run and atomically mark it as running."""
        db: Session = self.session_factory()
        try:
            stmt = (
                select(Run)
                .where(Run.status.in_(["queued", "pending"]))
                .order_by(Run.queued_at.asc().nulls_last(), Run.id.asc())
                .limit(1)
            )
            run = db.execute(stmt).scalars().first()
            if not run:
                return None

            run.status = "running"
            run.started_at = utc_now()
            db.commit()
            return run.id
        finally:
            db.close()

    async def cancel_run(self, run_id: int) -> bool:
        """Cancel a queued or currently executing run."""
        db: Session = self.session_factory()
        try:
            run = db.get(Run, run_id)
            if not run:
                return False

            if run.status in ("queued", "pending"):
                run.status = "cancelled"
                run.finished_at = utc_now()
                db.commit()

                # Notify broadcaster
                broadcaster = get_broadcaster()
                broadcaster.publish(run_id, {
                    "type": "completed",
                    "run_id": run_id,
                    "status": "cancelled",
                    "message": "Cancelled while queued",
                })
                return True

            if run.status == "running":
                self._cancel_requests.add(run_id)
                # If process is running, send SIGTERM to process group
                proc = self._active_processes.get(run_id)
                if proc and proc.returncode is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except OSError:
                        pass
                return True

            return False
        finally:
            db.close()

    async def execute_run(self, run_id: int) -> None:
        """Execute a specific run in a subprocess and record results."""
        broadcaster = get_broadcaster()
        db: Session = self.session_factory()

        run: Run | None = None
        try:
            run = db.get(Run, run_id)
            if not run:
                return

            run_artifacts_dir = self.artifacts_dir / str(run_id)
            run_artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_file = run_artifacts_dir / "run.log"
            result_file = run_artifacts_dir / "result.json"
            payload_file = run_artifacts_dir / "payload.json"

            # Parse config snapshot
            tester_name = "dummy"
            config_data: dict[str, Any] = {}
            suites: list[str] | None = None
            timeout = self.default_timeout

            if isinstance(run.config_snapshot, dict):
                tester_name = run.config_snapshot.get("tester", "dummy")
                config_data = run.config_snapshot.get("config", {})
                suites = run.config_snapshot.get("suites")
                if "timeout" in run.config_snapshot:
                    timeout = float(run.config_snapshot["timeout"])
            elif isinstance(run.config_snapshot, list) and run.config_snapshot:
                first = run.config_snapshot[0]
                if isinstance(first, dict):
                    tester_name = first.get("tester", "dummy")
                    config_data = first.get("config", {})
                    suites = first.get("suites")

            target_data = run.target_snapshot if isinstance(run.target_snapshot, dict) else {}

            payload = {
                "run_id": run_id,
                "tester": tester_name,
                "config": config_data,
                "target": target_data,
                "suites": suites,
                "artifacts_dir": str(run_artifacts_dir),
                "log_file": str(log_file),
                "result_file": str(result_file),
            }
            payload_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        finally:
            db.close()

        # Check if cancellation was requested before process started
        if run_id in self._cancel_requests:
            self._cancel_requests.discard(run_id)
            await self._finalize_run(run_id, status="cancelled", summary="Cancelled before start")
            return

        # Prepare subprocess environment
        env = dict(os.environ)
        # Ensure project root is in PYTHONPATH
        root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root_dir}:{current_pythonpath}" if current_pythonpath else root_dir

        cmd = [
            sys.executable,
            "-m",
            "backend.app.runner.executor",
            "--payload-file",
            str(payload_file),
        ]

        cancelled = False
        timed_out = False
        exit_code: int | None = None

        try:
            # start_new_session=True creates a new process group (setsid)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            self._active_processes[run_id] = process

            async def read_stream():
                assert process.stdout is not None
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_text = line.decode("utf-8", errors="replace").strip()
                    if not line_text:
                        continue
                    try:
                        event = json.loads(line_text)
                        if isinstance(event, dict) and "type" in event:
                            broadcaster.publish(run_id, event)
                    except json.JSONDecodeError:
                        broadcaster.publish(
                            run_id,
                            {
                                "type": "log",
                                "run_id": run_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "level": "INFO",
                                "message": line_text,
                            },
                        )

            try:
                # Run stream reader and process wait concurrently under timeout
                await asyncio.wait_for(
                    asyncio.gather(read_stream(), process.wait()),
                    timeout=timeout,
                )
                exit_code = process.returncode
            except asyncio.TimeoutError:
                timed_out = True
                logger.warning(f"Run {run_id} timed out after {timeout}s")
                await self._kill_process_group(process)
            except asyncio.CancelledError:
                cancelled = True
                await self._kill_process_group(process)
                raise

        finally:
            self._active_processes.pop(run_id, None)

        if run_id in self._cancel_requests or exit_code == 2:
            self._cancel_requests.discard(run_id)
            cancelled = True

        # Finalize run in DB
        if cancelled:
            await self._finalize_run(run_id, status="cancelled", summary="Run was cancelled")
        elif timed_out:
            await self._finalize_run(run_id, status="error", summary=f"Execution timed out after {timeout} seconds")
        else:
            await self._process_completed_run(run_id, exit_code or 0)

    async def _kill_process_group(self, process: asyncio.subprocess.Process) -> None:
        """Send SIGTERM, then SIGKILL if process doesn't exit promptly."""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        # Give it up to 2 seconds to terminate gracefully
        for _ in range(20):
            if process.returncode is not None:
                return
            await asyncio.sleep(0.1)

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _finalize_run(self, run_id: int, status: str, summary: str) -> None:
        """Mark run finished with given status and summary."""
        db: Session = self.session_factory()
        now = utc_now()
        try:
            run = db.get(Run, run_id)
            if not run:
                return

            run.status = status
            run.finished_at = now

            run_dir = self.artifacts_dir / str(run_id)
            log_file = run_dir / "run.log"
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{now.isoformat()}] [{status.upper()}] {summary}\n")
            except OSError:
                pass

            # Create or update Result record
            result = None
            if run.results:
                result = run.results[0]
                result.status = "cancelled" if status == "cancelled" else "error"
                result.summary = summary
            else:
                tester_name = "unknown"
                test_def_id = None
                if isinstance(run.config_snapshot, dict):
                    tester_name = run.config_snapshot.get("tester", "unknown")
                    test_def_id = run.config_snapshot.get("test_definition_id")
                elif isinstance(run.config_snapshot, list) and run.config_snapshot:
                    first = run.config_snapshot[0]
                    if isinstance(first, dict):
                        tester_name = first.get("tester", "unknown")
                        test_def_id = first.get("id") or first.get("test_definition_id")

                result = Result(
                    run_id=run_id,
                    test_definition_id=test_def_id,
                    tester=tester_name,
                    status="cancelled" if status == "cancelled" else "error",
                    summary=summary,
                    metrics={},
                )
                db.add(result)
                db.flush()

            # Ensure run.log is linked as an artifact
            if log_file.exists():
                size = None
                try:
                    size = log_file.stat().st_size
                except OSError:
                    pass
                db.add(
                    Artifact(
                        result_id=result.id,
                        name="run.log",
                        path=str(log_file),
                        mime="text/plain",
                        size_bytes=size,
                    )
                )

            db.commit()
        finally:
            db.close()

        get_broadcaster().publish(
            run_id,
            {"type": "completed", "run_id": run_id, "status": status, "summary": summary},
        )

    async def _process_completed_run(self, run_id: int, exit_code: int) -> None:
        """Process successful or non-zero exit from subprocess executor."""
        db: Session = self.session_factory()
        now = utc_now()
        try:
            run = db.get(Run, run_id)
            if not run:
                return

            run_dir = self.artifacts_dir / str(run_id)
            result_file = run_dir / "result.json"
            log_file = run_dir / "run.log"

            if exit_code == 0 and result_file.exists():
                try:
                    res_data = json.loads(result_file.read_text(encoding="utf-8"))
                except Exception as e:
                    res_data = {
                        "status": "error",
                        "summary": f"Failed to parse result.json: {e}",
                        "metrics": {},
                        "cases": [],
                        "artifacts": [],
                    }
            else:
                res_data = {
                    "status": "error",
                    "summary": f"Subprocess exited with return code {exit_code}",
                    "metrics": {},
                    "cases": [],
                    "artifacts": [],
                }

            run.status = "completed" if res_data.get("status") in ("pass", "fail") else res_data.get("status", "error")
            run.finished_at = now

            tester_name = "unknown"
            test_def_id = None
            if isinstance(run.config_snapshot, dict):
                tester_name = run.config_snapshot.get("tester", "unknown")
                test_def_id = run.config_snapshot.get("test_definition_id")
            elif isinstance(run.config_snapshot, list) and run.config_snapshot:
                first = run.config_snapshot[0]
                if isinstance(first, dict):
                    tester_name = first.get("tester", "unknown")
                    test_def_id = first.get("id") or first.get("test_definition_id")

            result = Result(
                run_id=run_id,
                test_definition_id=test_def_id,
                tester=tester_name,
                status=res_data.get("status", "pass"),
                metrics=res_data.get("metrics", {}),
                summary=res_data.get("summary"),
            )
            db.add(result)
            db.flush()

            # Insert CaseResults
            for c in res_data.get("cases", []):
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

            # Insert Artifacts
            known_paths = set()
            for a in res_data.get("artifacts", []):
                p = Path(a["path"])
                known_paths.add(str(p))
                db.add(
                    Artifact(
                        result_id=result.id,
                        name=a.get("name", p.name),
                        path=str(p),
                        mime=a.get("mime", "application/octet-stream"),
                        size_bytes=a.get("size_bytes"),
                    )
                )

            # Link run.log as artifact if not already linked
            if log_file.exists() and str(log_file) not in known_paths:
                db.add(
                    Artifact(
                        result_id=result.id,
                        name="run.log",
                        path=str(log_file),
                        mime="text/plain",
                        size_bytes=log_file.stat().st_size,
                    )
                )

            # Link result.json as artifact if exists
            if result_file.exists() and str(result_file) not in known_paths:
                db.add(
                    Artifact(
                        result_id=result.id,
                        name="result.json",
                        path=str(result_file),
                        mime="application/json",
                        size_bytes=result_file.stat().st_size,
                    )
                )

            db.commit()

            get_broadcaster().publish(
                run_id,
                {
                    "type": "completed",
                    "run_id": run_id,
                    "status": run.status,
                    "summary": result.summary,
                },
            )

        finally:
            db.close()


# Global worker instance
_global_worker: RunWorker | None = None


def get_worker() -> RunWorker:
    """Retrieve or initialize the global RunWorker instance."""
    global _global_worker
    if _global_worker is None:
        _global_worker = RunWorker()
    return _global_worker
