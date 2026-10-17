"""In-process cron scheduler for Jobs feeding the DB-backed run queue.

Features:
- Evaluates enabled jobs with cron schedules against UTC time
- Persists `next_run_at` and `last_scheduled_at` on the `jobs` table in SQLite
- Enforces no overlapping runs of the same job (skips if a run for the job is already active)
- Defined behavior after downtime via per-job `missed_run_policy`:
  - 'skip': skip missed occurrences during downtime and advance `next_run_at` to future
  - 'run_once' (alias 'coalesce'): coalesce 1+ missed occurrences during downtime into a single
    immediate catch-up run and advance `next_run_at` to future
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.base import utc_now
from backend.app.db.models.job import Job
from backend.app.db.models.run import Run
from backend.app.db.models.target import Target
from backend.app.db.models.test_definition import TestDefinition
from backend.app.db.session import SessionLocal
from backend.app.runner.cron import (
    _ensure_utc,
    count_missed_cron_runs,
    get_next_cron_time,
    is_manual_schedule,
    parse_cron_expression,
)
from backend.app.runner.worker import get_worker

logger = logging.getLogger(__name__)

ACTIVE_RUN_STATUSES = ("pending", "queued", "running", "cancelling")

MISSED_POLICY_ALIASES: dict[str, Literal["skip", "run_once"]] = {
    "skip": "skip",
    "ignore": "skip",
    "none": "skip",
    "run_once": "run_once",
    "coalesce": "run_once",
    "once": "run_once",
    "fire_once": "run_once",
    "catch_up": "run_once",
    "catchup": "run_once",
}


def normalize_missed_run_policy(value: str | None) -> Literal["skip", "run_once"]:
    """Normalize missed_run_policy string into 'skip' or 'run_once'."""
    if value is None:
        return "skip"
    key = value.strip().lower().replace("-", "_")
    if key not in MISSED_POLICY_ALIASES:
        raise ValueError(
            "missed_run_policy must be 'skip' or 'run_once' (or 'coalesce')"
        )
    return MISSED_POLICY_ALIASES[key]


def get_active_job_run(db: Session, job_id: int) -> Run | None:
    """Return an active (pending/queued/running/cancelling) Run for `job_id`, if any."""
    stmt = (
        select(Run)
        .where(Run.job_id == job_id, Run.status.in_(ACTIVE_RUN_STATUSES))
        .order_by(Run.id.desc())
        .limit(1)
    )
    return db.execute(stmt).scalars().first()


def sync_job_schedule_fields(job: Job, now: datetime | None = None) -> datetime | None:
    """Compute and update `job.next_run_at` based on `job.enabled` and `job.schedule`."""
    ref_now = _ensure_utc(now)
    if not job.enabled or is_manual_schedule(job.schedule):
        job.next_run_at = None
        return None

    parsed = parse_cron_expression(job.schedule)
    job.schedule = parsed.raw
    job.next_run_at = get_next_cron_time(parsed.raw, after=ref_now)
    return job.next_run_at


def build_and_enqueue_job_run(
    db: Session,
    job: Job,
    *,
    trigger_source: str = "cron",
    scheduled_for: datetime | None = None,
    coalesced_missed_count: int = 0,
    queued_at: datetime | None = None,
    wake_worker: bool = True,
) -> Run:
    """Create a snapshot Run for a Job and insert it into the run queue."""
    from backend.app.api.jobs import _extract_step_override
    from backend.app.api.test_definitions import _validate_definition_config

    def_ids = list(job.definition_ids or [])
    if not def_ids:
        raise ValueError(f"Job {job.id} has no test definitions configured to run")

    effective_run_mode = (job.run_mode or "sequential").strip().lower()
    saved_overrides = dict(job.overrides or {})

    step_snapshots: list[dict[str, Any]] = []
    for idx, def_id in enumerate(def_ids):
        def_obj = db.get(TestDefinition, def_id)
        if not def_obj:
            raise ValueError(f"TestDefinition {def_id} not found for job {job.id}")

        saved_step_ovr = _extract_step_override(saved_overrides, def_id, idx)
        eff_suites = (
            saved_step_ovr["suites"]
            if "suites" in saved_step_ovr and saved_step_ovr["suites"] is not None
            else list(def_obj.suites or [])
        )
        eff_target_id = (
            saved_step_ovr["target_id"]
            if "target_id" in saved_step_ovr
            else def_obj.target_id
        )
        eff_timeout = saved_step_ovr.get("timeout")

        merged_cfg: dict[str, Any] = dict(def_obj.config or {})
        if isinstance(saved_step_ovr.get("config"), dict):
            merged_cfg.update(saved_step_ovr["config"])

        validated_cfg = _validate_definition_config(def_obj.tester, merged_cfg, eff_suites)

        step_target_snap: dict[str, Any] = {}
        if eff_target_id is not None:
            target_obj = db.get(Target, int(eff_target_id))
            if target_obj:
                step_target_snap = target_obj.to_snapshot()

        judge_target_id = validated_cfg.get("judge_target_id")
        if judge_target_id is not None and not isinstance(step_target_snap.get("judge_target"), dict):
            judge_target = db.get(Target, int(judge_target_id))
            if judge_target:
                step_target_snap = dict(step_target_snap)
                step_target_snap["judge_target"] = judge_target.to_snapshot()

        step_snap: dict[str, Any] = {
            "order": idx,
            "id": def_obj.id,
            "test_definition_id": def_obj.id,
            "name": def_obj.name,
            "tester": def_obj.tester,
            "suites": eff_suites,
            "config": validated_cfg,
            "target_id": eff_target_id,
            "target_snapshot": step_target_snap,
        }
        if eff_timeout is not None:
            step_snap["timeout"] = float(eff_timeout)
        step_snapshots.append(step_snap)

    first_step = step_snapshots[0]
    testers_list = [s["tester"] for s in step_snapshots]

    from backend.app.runner.notifications import normalize_notification_settings

    config_snapshot: dict[str, Any] = {
        "job_id": job.id,
        "job_name": job.name,
        "run_mode": effective_run_mode,
        "trigger": trigger_source,
        "schedule": job.schedule,
        "tester": first_step["tester"],
        "testers": testers_list,
        "suites": first_step["suites"],
        "config": first_step["config"],
        "test_definition_id": first_step["test_definition_id"],
        "definition_ids": def_ids,
        "definitions": step_snapshots,
        "overrides": saved_overrides,
        "notifications": normalize_notification_settings(job.notifications),
    }
    if scheduled_for is not None:
        config_snapshot["scheduled_for"] = _ensure_utc(scheduled_for).isoformat()
    if coalesced_missed_count > 0:
        config_snapshot["missed_run_coalesced"] = True
        config_snapshot["coalesced_missed_count"] = coalesced_missed_count
    if first_step.get("timeout") is not None and len(step_snapshots) == 1:
        config_snapshot["timeout"] = float(first_step["timeout"])

    target_snapshot: dict[str, Any] = (
        dict(first_step["target_snapshot"]) if first_step["target_snapshot"] else {}
    )
    if len(step_snapshots) > 1:
        target_snapshot["targets"] = [s["target_snapshot"] for s in step_snapshots]

    run = Run(
        job_id=job.id,
        status="queued",
        queued_at=_ensure_utc(queued_at) if queued_at is not None else utc_now(),
        config_snapshot=config_snapshot,
        target_snapshot=target_snapshot,
    )
    db.add(run)
    db.flush()

    if wake_worker:
        get_worker().trigger()

    return run


class CronScheduler:
    """In-process scheduler polling SQLite jobs and queueing due runs without overlap."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] = SessionLocal,
        poll_interval: float = 1.0,
        misfire_grace_seconds: float = 60.0,
    ) -> None:
        self.session_factory = session_factory
        self.poll_interval = poll_interval
        self.misfire_grace_seconds = misfire_grace_seconds

        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self.last_tick_at: datetime | None = None
        self.last_downtime_recovery: dict[str, Any] | None = None

    @property
    def is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def trigger(self) -> None:
        """Wake up the scheduler loop immediately when a job schedule changes."""
        self._wake_event.set()

    def start(self, reconcile_startup: bool = True) -> None:
        """Start the background cron scheduler loop."""
        if reconcile_startup:
            try:
                self.recover_after_downtime()
            except Exception as exc:
                logger.warning(f"Failed to reconcile job schedules on startup: {exc}")

        if self._loop_task is None or self._loop_task.done():
            self._stop_event.clear()
            self._loop_task = asyncio.create_task(self._scheduler_loop())
            logger.info("CronScheduler started")

    async def stop(self) -> None:
        """Stop the background cron scheduler loop cleanly."""
        self._stop_event.set()
        self._wake_event.set()
        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._loop_task.cancel()
            self._loop_task = None
        logger.info("CronScheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Background loop checking for due cron jobs."""
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                logger.error(f"Unexpected error in CronScheduler loop: {exc}", exc_info=True)

            if not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
                self._wake_event.clear()

    def recover_after_downtime(self, now: datetime | None = None) -> dict[str, Any]:
        """Reconcile all scheduled jobs after process downtime according to missed_run_policy."""
        result = self.tick(now=now, is_startup=True)
        self.last_downtime_recovery = result
        return result

    def tick(
        self,
        now: datetime | None = None,
        *,
        is_startup: bool = False,
    ) -> dict[str, Any]:
        """Evaluate all jobs against `now` (defaults to UTC now) and enqueue due runs.

        Returns a summary dictionary of triggered runs, skipped overlaps, and skipped missed runs.
        """
        ref_now = _ensure_utc(now)
        self.last_tick_at = ref_now

        triggered_runs: list[dict[str, Any]] = []
        skipped_overlap: list[dict[str, Any]] = []
        skipped_missed: list[dict[str, Any]] = []
        initialized_jobs: list[int] = []

        db: Session = self.session_factory()
        try:
            jobs = db.execute(select(Job).order_by(Job.id.asc())).scalars().all()
            dirty = False

            for job in jobs:
                if not job.enabled or is_manual_schedule(job.schedule):
                    if job.next_run_at is not None:
                        job.next_run_at = None
                        dirty = True
                    continue

                try:
                    parsed = parse_cron_expression(job.schedule)
                except ValueError as exc:
                    logger.warning(
                        f"Job #{job.id} ('{job.name}') has invalid cron schedule '{job.schedule}': {exc}"
                    )
                    if job.next_run_at is not None:
                        job.next_run_at = None
                        dirty = True
                    continue

                if job.next_run_at is None:
                    job.next_run_at = get_next_cron_time(parsed.raw, after=ref_now)
                    initialized_jobs.append(job.id)
                    dirty = True
                    continue

                due_at = _ensure_utc(job.next_run_at)
                if due_at > ref_now:
                    continue

                # Job is due (due_at <= ref_now)
                missed_count = count_missed_cron_runs(parsed.raw, due_at, ref_now)
                delay_seconds = (ref_now - due_at).total_seconds()
                is_downtime_misfire = (
                    is_startup
                    or (delay_seconds > self.misfire_grace_seconds)
                    or (missed_count > 1)
                )
                next_future = get_next_cron_time(parsed.raw, after=ref_now)
                policy = normalize_missed_run_policy(job.missed_run_policy)

                if is_downtime_misfire and policy == "skip":
                    logger.info(
                        f"Skipping {missed_count} missed run(s) for job #{job.id} ('{job.name}') "
                        f"due at {due_at.isoformat()} (policy='skip'); next run at {next_future.isoformat()}"
                    )
                    job.next_run_at = next_future
                    dirty = True
                    skipped_missed.append({
                        "job_id": job.id,
                        "job_name": job.name,
                        "policy": policy,
                        "missed_count": missed_count,
                        "scheduled_for": due_at.isoformat(),
                        "next_run_at": next_future.isoformat(),
                    })
                    continue

                if not job.definition_ids:
                    logger.warning(
                        f"Scheduled job #{job.id} ('{job.name}') has no definitions; advancing next_run_at"
                    )
                    job.next_run_at = next_future
                    dirty = True
                    continue

                # Prevent overlapping runs of the same job
                active_run = get_active_job_run(db, job.id)
                if active_run is not None:
                    logger.info(
                        f"Skipping scheduled run for job #{job.id} ('{job.name}'): "
                        f"run #{active_run.id} is still {active_run.status}"
                    )
                    job.next_run_at = next_future
                    dirty = True
                    skipped_overlap.append({
                        "job_id": job.id,
                        "job_name": job.name,
                        "active_run_id": active_run.id,
                        "active_status": active_run.status,
                        "scheduled_for": due_at.isoformat(),
                        "next_run_at": next_future.isoformat(),
                    })
                    continue

                try:
                    run = build_and_enqueue_job_run(
                        db,
                        job,
                        trigger_source="cron",
                        scheduled_for=due_at,
                        coalesced_missed_count=missed_count if is_downtime_misfire else 0,
                        queued_at=ref_now,
                        wake_worker=False,
                    )
                    job.last_scheduled_at = ref_now
                    job.next_run_at = next_future
                    dirty = True
                    triggered_runs.append({
                        "job_id": job.id,
                        "job_name": job.name,
                        "run_id": run.id,
                        "scheduled_for": due_at.isoformat(),
                        "next_run_at": next_future.isoformat(),
                        "coalesced_missed_count": missed_count if is_downtime_misfire else 0,
                    })
                except Exception as exc:
                    logger.error(
                        f"Failed to enqueue scheduled run for job #{job.id} ('{job.name}'): {exc}",
                        exc_info=True,
                    )
                    job.next_run_at = next_future
                    dirty = True

            if dirty:
                db.commit()

            if triggered_runs:
                get_worker().trigger()

            return {
                "checked_at": ref_now.isoformat(),
                "is_startup": is_startup,
                "triggered_runs": triggered_runs,
                "skipped_overlap": skipped_overlap,
                "skipped_missed": skipped_missed,
                "initialized_jobs": initialized_jobs,
            }
        finally:
            db.close()

    def get_status(self, db: Session | None = None) -> dict[str, Any]:
        """Return current scheduler status and summary of scheduled jobs."""
        own_db = db is None
        session = db or self.session_factory()
        try:
            jobs = session.execute(select(Job).order_by(Job.id.asc())).scalars().all()
            scheduled_jobs = [
                j for j in jobs if j.enabled and not is_manual_schedule(j.schedule)
            ]
            next_times = [
                _ensure_utc(j.next_run_at)
                for j in scheduled_jobs
                if j.next_run_at is not None
            ]
            next_due = min(next_times).isoformat() if next_times else None
            return {
                "running": self.is_running,
                "poll_interval": self.poll_interval,
                "misfire_grace_seconds": self.misfire_grace_seconds,
                "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
                "total_jobs": len(jobs),
                "scheduled_enabled_jobs": len(scheduled_jobs),
                "next_due_at": next_due,
            }
        finally:
            if own_db:
                session.close()


_scheduler_instance: CronScheduler | None = None


def get_scheduler() -> CronScheduler:
    """Get the global singleton CronScheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = CronScheduler()
    return _scheduler_instance
