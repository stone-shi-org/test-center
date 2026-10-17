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
        self._active_run_processes: dict[int, set[asyncio.subprocess.Process]] = {}
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
            running_job_ids = set(
                db.execute(
                    select(Run.job_id).where(
                        Run.status.in_(["running", "cancelling"]),
                        Run.job_id.is_not(None),
                    )
                )
                .scalars()
                .all()
            )
            stmt = (
                select(Run)
                .where(Run.status.in_(["queued", "pending"]))
                .order_by(Run.queued_at.asc().nulls_last(), Run.id.asc())
            )
            if running_job_ids:
                stmt = stmt.where(
                    (Run.job_id.is_(None)) | (Run.job_id.not_in(running_job_ids))
                )
            run = db.execute(stmt.limit(1)).scalars().first()
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
                # Send SIGTERM to all active subprocesses for this run
                procs = list(self._active_run_processes.get(run_id, set()))
                single_proc = self._active_processes.get(run_id)
                if single_proc and single_proc not in procs:
                    procs.append(single_proc)
                for proc in procs:
                    if proc and proc.returncode is None:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except OSError:
                            pass
                return True

            return False
        finally:
            db.close()

    def _resolve_judge_target(
        self,
        db: Session,
        config_data: dict[str, Any],
        target_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Populate judge_target snapshot in target_data if judge_target_id is set."""
        judge_target_id = config_data.get("judge_target_id") if isinstance(config_data, dict) else None
        if (
            judge_target_id is not None
            and not isinstance(target_data.get("judge_target"), dict)
            and not isinstance(config_data.get("judge_target"), dict)
        ):
            from backend.app.db.models.target import Target
            try:
                judge_t = db.get(Target, int(judge_target_id))
                if judge_t is not None:
                    target_data = dict(target_data)
                    target_data["judge_target"] = judge_t.to_snapshot()
            except Exception:
                pass
        return target_data

    def _extract_execution_steps(
        self,
        db: Session,
        run: Run,
    ) -> tuple[list[dict[str, Any]], str]:
        """Extract ordered execution steps and run_mode ('sequential' | 'parallel') from Run."""
        run_mode = "sequential"
        run_timeout: float | None = None
        default_target = dict(run.target_snapshot) if isinstance(run.target_snapshot, dict) else {}

        raw_defs: list[Any] | None = None
        if isinstance(run.config_snapshot, dict):
            if "run_mode" in run.config_snapshot and isinstance(run.config_snapshot["run_mode"], str):
                run_mode = run.config_snapshot["run_mode"].lower()
            if "timeout" in run.config_snapshot and run.config_snapshot["timeout"] is not None:
                run_timeout = float(run.config_snapshot["timeout"])
            if isinstance(run.config_snapshot.get("definitions"), list) and len(run.config_snapshot["definitions"]) > 0:
                raw_defs = run.config_snapshot["definitions"]
        elif isinstance(run.config_snapshot, list) and len(run.config_snapshot) > 0:
            raw_defs = run.config_snapshot

        if raw_defs:
            steps: list[dict[str, Any]] = []
            for idx, item in enumerate(raw_defs):
                if not isinstance(item, dict):
                    continue
                tester_name = item.get("tester", "dummy")
                config_data = dict(item.get("config") or {}) if isinstance(item.get("config"), dict) else {}
                suites = item.get("suites")
                test_def_id = (
                    item.get("test_definition_id")
                    if item.get("test_definition_id") is not None
                    else item.get("id")
                )
                step_name = item.get("name") or f"Step {idx + 1} ({tester_name})"
                step_timeout = float(
                    item.get("timeout")
                    if item.get("timeout") is not None
                    else (run_timeout if run_timeout is not None else self.default_timeout)
                )

                # Resolve target for this step
                step_target: dict[str, Any] = {}
                if isinstance(item.get("target_snapshot"), dict) and item.get("target_snapshot"):
                    step_target = dict(item["target_snapshot"])
                elif isinstance(item.get("target"), dict) and item.get("target"):
                    step_target = dict(item["target"])
                elif item.get("target_id") is not None:
                    from backend.app.db.models.target import Target
                    try:
                        t_obj = db.get(Target, int(item["target_id"]))
                        if t_obj is not None:
                            step_target = t_obj.to_snapshot()
                    except Exception:
                        step_target = dict(default_target)
                else:
                    step_target = dict(default_target)

                step_target = self._resolve_judge_target(db, config_data, step_target)

                steps.append({
                    "step_index": idx,
                    "name": step_name,
                    "test_definition_id": test_def_id,
                    "tester": tester_name,
                    "config": config_data,
                    "suites": suites,
                    "target": step_target,
                    "timeout": step_timeout,
                })
            if steps:
                return steps, (run_mode if run_mode in ("sequential", "parallel") else "sequential")

        # Fallback: single step run
        tester_name = "dummy"
        config_data: dict[str, Any] = {}
        suites: list[str] | None = None
        test_def_id: int | None = None
        timeout = run_timeout if run_timeout is not None else self.default_timeout

        if isinstance(run.config_snapshot, dict):
            tester_name = run.config_snapshot.get("tester", "dummy")
            config_data = dict(run.config_snapshot.get("config") or {})
            suites = run.config_snapshot.get("suites")
            test_def_id = run.config_snapshot.get("test_definition_id") or run.config_snapshot.get("id")

        target_data = self._resolve_judge_target(db, config_data, default_target)

        return [
            {
                "step_index": 0,
                "name": tester_name,
                "test_definition_id": test_def_id,
                "tester": tester_name,
                "config": config_data,
                "suites": suites,
                "target": target_data,
                "timeout": timeout,
            }
        ], "sequential"

    async def execute_run(self, run_id: int) -> None:
        """Execute a specific run (single tester or multi-definition job) and record results."""
        db: Session = self.session_factory()
        try:
            run = db.get(Run, run_id)
            if not run:
                return

            run_artifacts_dir = self.artifacts_dir / str(run_id)
            run_artifacts_dir.mkdir(parents=True, exist_ok=True)
            steps, run_mode = self._extract_execution_steps(db, run)
        finally:
            db.close()

        # Check if cancellation was requested before process started
        if run_id in self._cancel_requests:
            self._cancel_requests.discard(run_id)
            await self._finalize_run(run_id, status="cancelled", summary="Cancelled before start")
            return

        if len(steps) == 1:
            # Single-step execution (preserves exact top-level artifact paths)
            step = steps[0]
            outcome = await self._execute_step_subprocess(
                run_id=run_id,
                step=step,
                total_steps=1,
                step_dir=run_artifacts_dir,
                main_log_file=run_artifacts_dir / "run.log",
            )
            if outcome["cancelled"]:
                await self._finalize_run(run_id, status="cancelled", summary="Run was cancelled")
            elif outcome["timed_out"]:
                await self._finalize_run(
                    run_id,
                    status="error",
                    summary=f"Execution timed out after {step['timeout']} seconds",
                )
            else:
                await self._process_completed_run(run_id, outcome["exit_code"] or 0)
            return

        # Multi-step execution (Job with 2+ test definitions)
        main_log_file = run_artifacts_dir / "run.log"
        broadcaster = get_broadcaster()
        total_steps = len(steps)

        header_msg = (
            f"Starting multi-definition job run #{run_id} "
            f"({total_steps} definitions, mode={run_mode})"
        )
        self._append_main_log(main_log_file, "INFO", header_msg)
        broadcaster.publish(
            run_id,
            {
                "type": "log",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": header_msg,
            },
        )

        step_outcomes: list[dict[str, Any]] = []
        try:
            if run_mode == "parallel":
                coros = [
                    self._execute_step_subprocess(
                        run_id=run_id,
                        step=step,
                        total_steps=total_steps,
                        step_dir=run_artifacts_dir / f"step_{idx + 1}_{step['tester']}",
                        main_log_file=main_log_file,
                    )
                    for idx, step in enumerate(steps)
                ]
                step_outcomes = list(await asyncio.gather(*coros))
            else:
                # Sequential execution in exact order
                for idx, step in enumerate(steps):
                    if run_id in self._cancel_requests:
                        step_outcomes.append({
                            "step": step,
                            "step_dir": run_artifacts_dir / f"step_{idx + 1}_{step['tester']}",
                            "cancelled": True,
                            "timed_out": False,
                            "exit_code": 2,
                        })
                        continue

                    outcome = await self._execute_step_subprocess(
                        run_id=run_id,
                        step=step,
                        total_steps=total_steps,
                        step_dir=run_artifacts_dir / f"step_{idx + 1}_{step['tester']}",
                        main_log_file=main_log_file,
                    )
                    step_outcomes.append(outcome)
                    if outcome["cancelled"]:
                        # Mark remaining steps as cancelled
                        for rem_idx in range(idx + 1, total_steps):
                            rem_step = steps[rem_idx]
                            step_outcomes.append({
                                "step": rem_step,
                                "step_dir": run_artifacts_dir / f"step_{rem_idx + 1}_{rem_step['tester']}",
                                "cancelled": True,
                                "timed_out": False,
                                "exit_code": 2,
                            })
                        break
        finally:
            self._cancel_requests.discard(run_id)
            self._active_run_processes.pop(run_id, None)
            self._active_processes.pop(run_id, None)

        await self._persist_multi_step_outcomes(run_id, step_outcomes, main_log_file)

    def _append_main_log(self, log_file: Path, level: str, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] [{level.upper()}] {message}\n")
                f.flush()
        except OSError:
            pass

    async def _execute_step_subprocess(
        self,
        run_id: int,
        step: dict[str, Any],
        total_steps: int,
        step_dir: Path,
        main_log_file: Path,
    ) -> dict[str, Any]:
        """Run a single definition step in its own subprocess."""
        broadcaster = get_broadcaster()
        step_dir.mkdir(parents=True, exist_ok=True)
        step_idx = step["step_index"]
        tester_name = step["tester"]
        timeout = float(step["timeout"])

        step_log_file = step_dir / "run.log"
        step_result_file = step_dir / "result.json"
        step_payload_file = step_dir / "payload.json"

        payload = {
            "run_id": run_id,
            "tester": tester_name,
            "config": step["config"],
            "target": step["target"],
            "suites": step["suites"],
            "artifacts_dir": str(step_dir),
            "log_file": str(step_log_file),
            "result_file": str(step_result_file),
        }
        step_payload_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if run_id in self._cancel_requests:
            return {
                "step": step,
                "step_dir": step_dir,
                "cancelled": True,
                "timed_out": False,
                "exit_code": 2,
            }

        env = dict(os.environ)
        root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root_dir}:{current_pythonpath}" if current_pythonpath else root_dir

        cmd = [
            sys.executable,
            "-m",
            "backend.app.runner.executor",
            "--payload-file",
            str(step_payload_file),
        ]

        cancelled = False
        timed_out = False
        exit_code: int | None = None
        process: asyncio.subprocess.Process | None = None

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            self._active_processes[run_id] = process
            self._active_run_processes.setdefault(run_id, set()).add(process)

            async def read_stream():
                assert process is not None and process.stdout is not None
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
                            if event["type"] == "completed":
                                # Final run completion event is emitted after DB commit
                                continue
                            if total_steps > 1:
                                event["step_index"] = step_idx
                                event["tester"] = tester_name
                                if event["type"] == "log":
                                    prefix = f"[Step {step_idx + 1}/{total_steps} {tester_name}] "
                                    msg = str(event.get("message", ""))
                                    event["message"] = prefix + msg
                                    self._append_main_log(
                                        main_log_file,
                                        str(event.get("level", "INFO")),
                                        event["message"],
                                    )
                            broadcaster.publish(run_id, event)
                    except json.JSONDecodeError:
                        msg_text = (
                            f"[Step {step_idx + 1}/{total_steps} {tester_name}] {line_text}"
                            if total_steps > 1
                            else line_text
                        )
                        if total_steps > 1:
                            self._append_main_log(main_log_file, "INFO", msg_text)
                        broadcaster.publish(
                            run_id,
                            {
                                "type": "log",
                                "run_id": run_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "level": "INFO",
                                "message": msg_text,
                            },
                        )

            try:
                await asyncio.wait_for(
                    asyncio.gather(read_stream(), process.wait()),
                    timeout=timeout,
                )
                exit_code = process.returncode
            except asyncio.TimeoutError:
                timed_out = True
                logger.warning(f"Run {run_id} step {step_idx + 1} timed out after {timeout}s")
                await self._kill_process_group(process)
            except asyncio.CancelledError:
                cancelled = True
                await self._kill_process_group(process)
                raise

        finally:
            if process is not None:
                run_procs = self._active_run_processes.get(run_id)
                if run_procs:
                    run_procs.discard(process)
                    if not run_procs:
                        self._active_run_processes.pop(run_id, None)
                if self._active_processes.get(run_id) is process:
                    self._active_processes.pop(run_id, None)

        if run_id in self._cancel_requests or exit_code == 2:
            if total_steps == 1:
                self._cancel_requests.discard(run_id)
            cancelled = True

        return {
            "step": step,
            "step_dir": step_dir,
            "cancelled": cancelled,
            "timed_out": timed_out,
            "exit_code": exit_code,
        }

    async def _persist_multi_step_outcomes(
        self,
        run_id: int,
        step_outcomes: list[dict[str, Any]],
        main_log_file: Path,
    ) -> None:
        """Persist Result, CaseResult, and Artifact rows for all steps of a multi-step run."""
        db: Session = self.session_factory()
        now = utc_now()
        try:
            run = db.get(Run, run_id)
            if not run:
                return

            result_statuses: list[str] = []
            summaries: list[str] = []

            for outcome in step_outcomes:
                step = outcome["step"]
                step_dir: Path = outcome["step_dir"]
                step_result_file = step_dir / "result.json"
                step_log_file = step_dir / "run.log"
                tester_name = step["tester"]
                test_def_id = step.get("test_definition_id")

                if outcome["cancelled"]:
                    res_data = {
                        "status": "cancelled",
                        "summary": "Run was cancelled",
                        "metrics": {},
                        "cases": [],
                        "artifacts": [],
                    }
                elif outcome["timed_out"]:
                    res_data = {
                        "status": "error",
                        "summary": f"Execution timed out after {step['timeout']} seconds",
                        "metrics": {},
                        "cases": [],
                        "artifacts": [],
                    }
                elif (outcome["exit_code"] or 0) == 0 and step_result_file.exists():
                    try:
                        res_data = json.loads(step_result_file.read_text(encoding="utf-8"))
                    except Exception as e:
                        res_data = {
                            "status": "error",
                            "summary": f"Failed to parse result.json: {e}",
                            "metrics": {},
                            "cases": [],
                            "artifacts": [],
                        }
                elif step_result_file.exists():
                    try:
                        res_data = json.loads(step_result_file.read_text(encoding="utf-8"))
                    except Exception:
                        res_data = {
                            "status": "error",
                            "summary": f"Subprocess exited with return code {outcome['exit_code']}",
                            "metrics": {},
                            "cases": [],
                            "artifacts": [],
                        }
                else:
                    res_data = {
                        "status": "error",
                        "summary": f"Subprocess exited with return code {outcome['exit_code']}",
                        "metrics": {},
                        "cases": [],
                        "artifacts": [],
                    }

                r_status = res_data.get("status", "pass")
                result_statuses.append(r_status)
                if res_data.get("summary"):
                    summaries.append(f"{tester_name}: {res_data['summary']}")

                result = Result(
                    run_id=run_id,
                    test_definition_id=test_def_id,
                    tester=tester_name,
                    status=r_status,
                    metrics=res_data.get("metrics", {}),
                    summary=res_data.get("summary"),
                )
                db.add(result)
                db.flush()

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

                if step_log_file.exists() and str(step_log_file) not in known_paths:
                    db.add(
                        Artifact(
                            result_id=result.id,
                            name="run.log",
                            path=str(step_log_file),
                            mime="text/plain",
                            size_bytes=step_log_file.stat().st_size,
                        )
                    )

                if step_result_file.exists() and str(step_result_file) not in known_paths:
                    db.add(
                        Artifact(
                            result_id=result.id,
                            name="result.json",
                            path=str(step_result_file),
                            mime="application/json",
                            size_bytes=step_result_file.stat().st_size,
                        )
                    )

            if any(s == "cancelled" for s in result_statuses):
                run.status = "cancelled"
            elif result_statuses and all(s == "error" for s in result_statuses):
                run.status = "error"
            else:
                run.status = "completed"

            run.finished_at = now
            overall_summary = " | ".join(summaries) if summaries else f"Completed {len(step_outcomes)} steps"
            self._append_main_log(main_log_file, run.status.upper(), overall_summary)
            db.commit()

            get_broadcaster().publish(
                run_id,
                {
                    "type": "completed",
                    "run_id": run_id,
                    "status": run.status,
                    "summary": overall_summary,
                },
            )
        finally:
            db.close()

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
