"""API routes for managing jobs and triggering bundled test runs.

A Job is a named set of test definitions with ordering, run mode (sequential/parallel),
and optional per-definition or job-wide overrides.
"""

from datetime import datetime
from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.api.targets import TargetResponse
from backend.app.api.test_definitions import _validate_definition_config
from backend.app.db.base import utc_now
from backend.app.db.models.job import Job
from backend.app.db.models.run import Run
from backend.app.db.models.target import Target
from backend.app.db.models.test_definition import TestDefinition
from backend.app.db.session import get_db
from backend.app.runner.cron import (
    _ensure_utc,
    describe_cron_expression,
    get_next_cron_time,
    get_upcoming_cron_times,
    is_manual_schedule,
    normalize_schedule,
    validate_and_preview_schedule,
)
from backend.app.runner.scheduler import (
    get_active_job_run,
    get_scheduler,
    normalize_missed_run_policy,
    sync_job_schedule_fields,
)
from backend.app.runner.notifications import (
    build_notification_payload,
    dispatch_notifications,
    evaluate_run_for_notifications,
    normalize_notification_settings,
    process_run_notifications,
)
from backend.app.runner.worker import get_worker

router = APIRouter(prefix="/jobs", tags=["jobs"])

_ENVELOPE_OVERRIDE_KEYS = {
    "config",
    "suites",
    "target_id",
    "timeout",
    "test_definition_id",
    "definition_id",
    "id",
    "overrides",
    "order",
    "name",
    "tester",
}

_GLOBAL_META_OVERRIDE_KEYS = {
    "config",
    "suites",
    "target_id",
    "timeout",
    "definitions",
    "per_definition",
    "steps",
}


def _normalize_run_mode(value: str | None) -> Literal["sequential", "parallel"]:
    if value is None:
        return "sequential"
    mode = value.strip().lower()
    if mode not in ("sequential", "parallel"):
        raise ValueError("run_mode must be 'sequential' or 'parallel'")
    return mode  # type: ignore[return-value]


def _normalize_and_validate_schedule(value: str | None) -> str:
    return normalize_schedule(value)



def _parse_single_override_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single override dictionary into {suites?, config?, target_id?, timeout?}."""
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out

    # Unwrap nested 'overrides' dict if present
    inner = raw.get("overrides")
    if isinstance(inner, dict):
        merged_raw = {**inner, **{k: v for k, v in raw.items() if k != "overrides"}}
    else:
        merged_raw = raw

    if "suites" in merged_raw and merged_raw["suites"] is not None:
        if isinstance(merged_raw["suites"], list):
            out["suites"] = [str(s) for s in merged_raw["suites"]]

    if "target_id" in merged_raw:
        out["target_id"] = merged_raw["target_id"]

    if "timeout" in merged_raw and merged_raw["timeout"] is not None:
        out["timeout"] = float(merged_raw["timeout"])

    cfg: dict[str, Any] = {}
    if isinstance(merged_raw.get("config"), dict):
        cfg.update(merged_raw["config"])

    # Also treat any non-envelope keys as direct config overrides (unless they are definition IDs)
    for k, v in merged_raw.items():
        if k in _ENVELOPE_OVERRIDE_KEYS or k in _GLOBAL_META_OVERRIDE_KEYS:
            continue
        if str(k).isdigit():
            continue
        cfg[k] = v

    if cfg:
        out["config"] = cfg

    return out


def _extract_step_override(
    overrides: dict[str, Any] | None,
    def_id: int,
    step_index: int = 0,
) -> dict[str, Any]:
    """Resolve effective overrides for a specific definition/step from a job or request overrides dict."""
    if not overrides or not isinstance(overrides, dict):
        return {}

    resolved: dict[str, Any] = {}
    resolved_cfg: dict[str, Any] = {}

    # 1. Job-level global overrides (target_id, timeout, suites, config)
    if "suites" in overrides and isinstance(overrides["suites"], list):
        resolved["suites"] = [str(s) for s in overrides["suites"]]
    if "target_id" in overrides and overrides["target_id"] is not None:
        resolved["target_id"] = overrides["target_id"]
    if "timeout" in overrides and overrides["timeout"] is not None:
        resolved["timeout"] = float(overrides["timeout"])
    if isinstance(overrides.get("config"), dict):
        resolved_cfg.update(overrides["config"])

    # Also check if top-level overrides has direct config keys when no per-def keys are used
    has_per_def_keys = any(
        str(k).isdigit() or k in ("definitions", "per_definition", "steps")
        for k in overrides.keys()
    )
    if not has_per_def_keys:
        for k, v in overrides.items():
            if k not in _ENVELOPE_OVERRIDE_KEYS and k not in _GLOBAL_META_OVERRIDE_KEYS:
                resolved_cfg[k] = v

    # 2. Per-definition overrides (under definitions/per_definition or top-level str(def_id)/int(def_id))
    candidate_dicts: list[dict[str, Any]] = []
    for container_key in ("definitions", "per_definition"):
        container = overrides.get(container_key)
        if isinstance(container, dict):
            for key_variant in (str(def_id), def_id):
                if key_variant in container and isinstance(container[key_variant], dict):
                    candidate_dicts.append(container[key_variant])

    for key_variant in (str(def_id), def_id):
        if key_variant in overrides and isinstance(overrides[key_variant], dict):
            candidate_dicts.append(overrides[key_variant])

    # 3. Per-step index overrides (under 'steps')
    steps_container = overrides.get("steps")
    if isinstance(steps_container, list) and 0 <= step_index < len(steps_container):
        if isinstance(steps_container[step_index], dict):
            candidate_dicts.append(steps_container[step_index])
    elif isinstance(steps_container, dict):
        for key_variant in (str(step_index), step_index):
            if key_variant in steps_container and isinstance(steps_container[key_variant], dict):
                candidate_dicts.append(steps_container[key_variant])

    for cand in candidate_dicts:
        parsed = _parse_single_override_dict(cand)
        if "suites" in parsed:
            resolved["suites"] = parsed["suites"]
        if "target_id" in parsed:
            resolved["target_id"] = parsed["target_id"]
        if "timeout" in parsed:
            resolved["timeout"] = parsed["timeout"]
        if "config" in parsed and isinstance(parsed["config"], dict):
            resolved_cfg.update(parsed["config"])

    if resolved_cfg:
        resolved["config"] = resolved_cfg

    return resolved


def _extract_definitions_and_overrides_from_payload(
    definition_ids: list[int] | None,
    overrides: dict[str, Any] | None,
    definitions_input: list[Any] | None,
    steps_input: list[Any] | None,
) -> tuple[list[int] | None, dict[str, Any] | None]:
    """Normalize definition_ids and overrides from flexible payload formats."""
    items_list = definitions_input if definitions_input is not None else steps_input
    norm_overrides: dict[str, Any] | None = dict(overrides) if overrides is not None else None
    norm_def_ids: list[int] | None = list(definition_ids) if definition_ids is not None else None

    if items_list is not None:
        extracted_ids: list[int] = []
        if norm_overrides is None:
            norm_overrides = {}

        for item in items_list:
            if isinstance(item, int) and not isinstance(item, bool):
                extracted_ids.append(item)
            elif isinstance(item, dict):
                raw_id = (
                    item.get("test_definition_id")
                    if item.get("test_definition_id") is not None
                    else (
                        item.get("definition_id")
                        if item.get("definition_id") is not None
                        else item.get("id")
                    )
                )
                if raw_id is None:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail="Each definition item must include 'test_definition_id' (or 'id')",
                    )
                def_id = int(raw_id)
                extracted_ids.append(def_id)
                step_ovr = _parse_single_override_dict(item)
                if step_ovr:
                    existing_ovr = norm_overrides.get(str(def_id))
                    if isinstance(existing_ovr, dict):
                        merged_cfg = {
                            **(existing_ovr.get("config") or {}),
                            **(step_ovr.get("config") or {}),
                        }
                        merged = {**existing_ovr, **step_ovr}
                        if merged_cfg:
                            merged["config"] = merged_cfg
                        norm_overrides[str(def_id)] = merged
                    else:
                        norm_overrides[str(def_id)] = step_ovr
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Invalid definition item: {item!r}",
                )

        if norm_def_ids is None:
            norm_def_ids = extracted_ids

    return norm_def_ids, norm_overrides


def _validate_job_definitions_and_overrides(
    db: Session,
    definition_ids: list[int],
    overrides: dict[str, Any],
) -> list[TestDefinition]:
    """Verify all referenced test definitions and targets exist, and validate merged configs."""
    resolved_defs: list[TestDefinition] = []

    # Check global target_id override if present
    if isinstance(overrides, dict) and overrides.get("target_id") is not None:
        global_tid = overrides["target_id"]
        if isinstance(global_tid, int) and not isinstance(global_tid, bool):
            if not db.get(Target, global_tid):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Target {global_tid} not found",
                )

    for idx, def_id in enumerate(definition_ids):
        if not isinstance(def_id, int) or isinstance(def_id, bool):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid definition_id: {def_id!r}",
            )
        stmt = (
            select(TestDefinition)
            .options(selectinload(TestDefinition.target))
            .where(TestDefinition.id == def_id)
        )
        def_obj = db.execute(stmt).scalars().first()
        if not def_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"TestDefinition {def_id} not found",
            )

        step_ovr = _extract_step_override(overrides, def_id, idx)
        ovr_target_id = step_ovr.get("target_id")
        if ovr_target_id is not None:
            if not db.get(Target, int(ovr_target_id)):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Target {ovr_target_id} not found",
                )

        merged_suites = (
            step_ovr["suites"]
            if "suites" in step_ovr and step_ovr["suites"] is not None
            else list(def_obj.suites or [])
        )
        merged_config = dict(def_obj.config or {})
        if isinstance(step_ovr.get("config"), dict):
            merged_config.update(step_ovr["config"])

        judge_tid = merged_config.get("judge_target_id")
        if judge_tid is not None and isinstance(judge_tid, int) and not isinstance(judge_tid, bool):
            if not db.get(Target, judge_tid):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Judge target {judge_tid} not found",
                )

        # Validate merged config and suites against tester plugin schema
        _validate_definition_config(def_obj.tester, merged_config, merged_suites)
        resolved_defs.append(def_obj)

    return resolved_defs


def _format_job(job: Job, db: Session) -> dict[str, Any]:
    """Serialize a Job with its ordered definitions and effective overrides."""
    def_ids = list(job.definition_ids or [])
    overrides = dict(job.overrides or {})

    # Preload referenced TestDefinitions and Targets
    defs_by_id: dict[int, TestDefinition] = {}
    if def_ids:
        stmt = (
            select(TestDefinition)
            .options(selectinload(TestDefinition.target))
            .where(TestDefinition.id.in_(set(def_ids)))
        )
        for d in db.execute(stmt).scalars().all():
            defs_by_id[d.id] = d

    definitions_list: list[dict[str, Any]] = []
    for idx, def_id in enumerate(def_ids):
        def_obj = defs_by_id.get(def_id)
        if not def_obj:
            continue

        step_ovr = _extract_step_override(overrides, def_id, idx)
        eff_suites = (
            step_ovr["suites"]
            if "suites" in step_ovr and step_ovr["suites"] is not None
            else list(def_obj.suites or [])
        )
        eff_config = dict(def_obj.config or {})
        if isinstance(step_ovr.get("config"), dict):
            eff_config.update(step_ovr["config"])

        eff_target_id = (
            step_ovr["target_id"]
            if "target_id" in step_ovr
            else def_obj.target_id
        )
        eff_target_obj: Target | None = None
        if eff_target_id is not None:
            if def_obj.target and def_obj.target.id == eff_target_id:
                eff_target_obj = def_obj.target
            else:
                eff_target_obj = db.get(Target, int(eff_target_id))

        base_target_dict = (
            TargetResponse.model_validate(def_obj.target).model_dump(mode="json")
            if def_obj.target
            else None
        )
        eff_target_dict = (
            TargetResponse.model_validate(eff_target_obj).model_dump(mode="json")
            if eff_target_obj
            else None
        )

        definitions_list.append({
            "order": idx,
            "id": def_obj.id,
            "test_definition_id": def_obj.id,
            "name": def_obj.name,
            "tester": def_obj.tester,
            "suites": list(def_obj.suites or []),
            "config": dict(def_obj.config or {}),
            "target_id": def_obj.target_id,
            "target": base_target_dict,
            "overrides": step_ovr,
            "effective_suites": eff_suites,
            "effective_config": eff_config,
            "effective_target_id": eff_target_id,
            "effective_target": eff_target_dict,
            "timeout": step_ovr.get("timeout"),
        })

    is_manual = is_manual_schedule(job.schedule)
    next_run_iso: str | None = None
    upcoming_runs: list[str] = []

    if job.enabled and not is_manual:
        try:
            if job.next_run_at is not None:
                first_dt = _ensure_utc(job.next_run_at)
                next_run_iso = first_dt.isoformat()
                rest = get_upcoming_cron_times(job.schedule, count=4, after=first_dt)
                upcoming_runs = [next_run_iso] + [dt.isoformat() for dt in rest]
            else:
                upcoming = get_upcoming_cron_times(job.schedule, count=5)
                if upcoming:
                    next_run_iso = upcoming[0].isoformat()
                    upcoming_runs = [dt.isoformat() for dt in upcoming]
        except ValueError:
            next_run_iso = None
            upcoming_runs = []

    active_run = get_active_job_run(db, job.id)
    try:
        norm_notifications = normalize_notification_settings(job.notifications)
    except ValueError:
        norm_notifications = dict(job.notifications or {})

    return {
        "id": job.id,
        "name": job.name,
        "description": job.description,
        "schedule": job.schedule,
        "is_scheduled": not is_manual,
        "schedule_description": describe_cron_expression(job.schedule),
        "enabled": job.enabled,
        "run_mode": job.run_mode or "sequential",
        "missed_run_policy": job.missed_run_policy or "skip",
        "next_run_at": next_run_iso,
        "upcoming_runs": upcoming_runs,
        "last_scheduled_at": (
            _ensure_utc(job.last_scheduled_at).isoformat() if job.last_scheduled_at else None
        ),
        "is_running": active_run is not None,
        "active_run_id": active_run.id if active_run else None,
        "definition_ids": def_ids,
        "overrides": overrides,
        "notifications": norm_notifications,
        "notification_settings": norm_notifications,
        "definitions": definitions_list,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


class JobCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Unique job name")
    description: str | None = Field(default=None, description="Optional job description")
    schedule: str = Field(default="manual", max_length=128, description="Schedule ('manual' or 5-field cron expression)")
    enabled: bool = Field(default=True, description="Whether job is enabled")
    run_mode: str = Field(default="sequential", description="Execution mode: 'sequential' or 'parallel'")
    missed_run_policy: str = Field(default="skip", description="Missed-run policy after downtime: 'skip' or 'run_once'")
    definition_ids: list[int] | None = Field(default=None, description="Ordered list of TestDefinition IDs")
    overrides: dict[str, Any] | None = Field(default=None, description="Optional per-definition or global overrides")
    definitions: list[Any] | None = Field(default=None, description="Optional ordered list of definition items with inline overrides")
    steps: list[Any] | None = Field(default=None, description="Optional alias for definitions")
    notifications: dict[str, Any] | None = Field(default=None, description="Per-job notification settings (webhook/email on failure/regression)")
    notification_settings: dict[str, Any] | None = Field(default=None, description="Optional alias for notifications")
    webhook_url: str | None = Field(default=None, description="Optional webhook URL shortcut")
    notify_on_failure: bool | None = Field(default=None, description="Optional trigger on failed run shortcut")
    notify_on_regression: bool | None = Field(default=None, description="Optional trigger on regression shortcut")
    regression_threshold: float | None = Field(default=None, description="Optional regression percentage threshold shortcut")
    email_to: list[str] | str | None = Field(default=None, description="Optional email recipients shortcut")

    @field_validator("schedule")
    @classmethod
    def check_schedule(cls, v: str) -> str:
        return _normalize_and_validate_schedule(v)

    @field_validator("run_mode")
    @classmethod
    def check_run_mode(cls, v: str) -> str:
        return _normalize_run_mode(v)

    @field_validator("missed_run_policy")
    @classmethod
    def check_missed_run_policy(cls, v: str) -> str:
        return normalize_missed_run_policy(v)


class JobUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None)
    schedule: str | None = Field(default=None, max_length=128)
    enabled: bool | None = Field(default=None)
    run_mode: str | None = Field(default=None)
    missed_run_policy: str | None = Field(default=None)
    definition_ids: list[int] | None = Field(default=None)
    overrides: dict[str, Any] | None = Field(default=None)
    definitions: list[Any] | None = Field(default=None)
    steps: list[Any] | None = Field(default=None)
    notifications: dict[str, Any] | None = Field(default=None)
    notification_settings: dict[str, Any] | None = Field(default=None)
    webhook_url: str | None = Field(default=None)
    notify_on_failure: bool | None = Field(default=None)
    notify_on_regression: bool | None = Field(default=None)
    regression_threshold: float | None = Field(default=None)
    email_to: list[str] | str | None = Field(default=None)

    @field_validator("schedule")
    @classmethod
    def check_schedule(cls, v: str | None) -> str | None:
        if v is not None:
            return _normalize_and_validate_schedule(v)
        return v

    @field_validator("run_mode")
    @classmethod
    def check_run_mode(cls, v: str | None) -> str | None:
        if v is not None:
            return _normalize_run_mode(v)
        return v

    @field_validator("missed_run_policy")
    @classmethod
    def check_missed_run_policy(cls, v: str | None) -> str | None:
        if v is not None:
            return normalize_missed_run_policy(v)
        return v


class JobOrderUpdate(BaseModel):
    definition_ids: list[int] = Field(..., description="New ordered list of TestDefinition IDs")


class CronValidateRequest(BaseModel):
    schedule: str | None = Field(default=None, description="Schedule string ('manual' or cron expression)")
    expression: str | None = Field(default=None, description="Alias for schedule")
    count: int = Field(default=5, ge=1, le=20, description="Number of upcoming run timestamps to preview")


class TriggerJobRunRequest(BaseModel):
    run_mode: str | None = Field(default=None, description="Optional run_mode override ('sequential' or 'parallel')")
    overrides: dict[str, Any] | None = Field(default=None, description="Optional runtime overrides")
    config: dict[str, Any] | None = Field(default=None, description="Optional global config override")
    suites: list[str] | None = Field(default=None, description="Optional global suites override")
    target_id: int | None = Field(default=None, description="Optional global target ID override")
    timeout: float | None = Field(default=None, description="Optional timeout in seconds")
    definition_ids: list[int] | None = Field(default=None, description="Optional subset/reordered definition IDs")
    prevent_overlap: bool = Field(default=False, description="If true, return 409 when a run for this job is already active")

    @field_validator("run_mode")
    @classmethod
    def check_run_mode(cls, v: str | None) -> str | None:
        if v is not None:
            return _normalize_run_mode(v)
        return v


class TestJobNotificationRequest(BaseModel):
    run_id: int | None = Field(default=None, description="Optional Run ID to evaluate and send notifications for")
    webhook_url: str | None = Field(default=None, description="Optional webhook URL override for testing")
    email_to: list[str] | str | None = Field(default=None, description="Optional email recipients override for testing")
    notifications: dict[str, Any] | None = Field(default=None, description="Optional notification settings override")


@router.post("/validate-cron")
def validate_cron_post(req: CronValidateRequest) -> dict[str, Any]:
    """Validate a cron expression or schedule string and preview upcoming run timestamps."""
    expr = req.schedule if req.schedule is not None else req.expression
    return validate_and_preview_schedule(expr, upcoming_count=req.count)


@router.get("/validate-cron")
def validate_cron_get(
    schedule: str | None = Query(None, description="Schedule or cron expression"),
    expression: str | None = Query(None, description="Alias for schedule"),
    count: int = Query(5, ge=1, le=20, description="Number of upcoming run timestamps"),
) -> dict[str, Any]:
    """Validate a cron expression via query parameters and preview upcoming runs."""
    expr = schedule if schedule is not None else expression
    return validate_and_preview_schedule(expr, upcoming_count=count)


@router.get("/scheduler/status")
def get_scheduler_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get current status of the in-process cron scheduler."""
    return get_scheduler().get_status(db)


@router.post("/scheduler/tick")
def trigger_scheduler_tick() -> dict[str, Any]:
    """Manually evaluate due cron jobs immediately."""
    return get_scheduler().tick()


@router.post("", status_code=status.HTTP_201_CREATED)
def create_job(
    req: JobCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Create a new job bundling test definitions with ordering, run mode, schedule, notifications, and optional overrides."""
    existing = db.execute(select(Job).where(Job.name == req.name)).scalars().first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job with name '{req.name}' already exists",
        )

    def_ids, norm_overrides = _extract_definitions_and_overrides_from_payload(
        definition_ids=req.definition_ids,
        overrides=req.overrides,
        definitions_input=req.definitions,
        steps_input=req.steps,
    )
    final_def_ids = def_ids if def_ids is not None else []
    final_overrides = norm_overrides if norm_overrides is not None else {}

    _validate_job_definitions_and_overrides(db, final_def_ids, final_overrides)

    raw_notif = (
        req.notifications
        if req.notifications is not None
        else req.notification_settings
    )
    try:
        final_notifications = normalize_notification_settings(
            raw_notif,
            webhook_url=req.webhook_url,
            notify_on_failure=req.notify_on_failure,
            notify_on_regression=req.notify_on_regression,
            regression_threshold=req.regression_threshold,
            email_to=req.email_to,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    job = Job(
        name=req.name,
        description=req.description,
        schedule=req.schedule,
        enabled=req.enabled,
        run_mode=req.run_mode,
        missed_run_policy=req.missed_run_policy,
        definition_ids=final_def_ids,
        overrides=final_overrides,
        notifications=final_notifications,
    )
    sync_job_schedule_fields(job)
    db.add(job)
    db.commit()
    db.refresh(job)
    get_scheduler().trigger()
    return _format_job(job, db)


@router.get("")
def list_jobs(
    enabled: bool | None = Query(None, description="Filter by enabled status"),
    run_mode: str | None = Query(None, description="Filter by run_mode ('sequential' or 'parallel')"),
    scheduled: bool | None = Query(None, description="Filter by whether job has a cron schedule"),
    limit: int = Query(50, ge=1, le=200, description="Max jobs to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List jobs ordered by ID ascending."""
    stmt = select(Job).order_by(Job.id.asc()).offset(offset).limit(limit)
    if enabled is not None:
        stmt = stmt.where(Job.enabled == enabled)
    if run_mode:
        stmt = stmt.where(Job.run_mode == run_mode.lower())
    if scheduled is True:
        stmt = stmt.where(Job.schedule != "manual")
    elif scheduled is False:
        stmt = stmt.where(Job.schedule == "manual")

    jobs = db.execute(stmt).scalars().all()
    return [_format_job(j, db) for j in jobs]


@router.get("/{job_id}")
def get_job(
    job_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get details of a specific job by ID."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    return _format_job(job, db)


@router.put("/{job_id}")
@router.patch("/{job_id}")
def update_job(
    job_id: int,
    req: JobUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Update job fields, definition ordering, run mode, schedule, missed-run policy, notifications, or overrides."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    update_data = req.model_dump(exclude_unset=True)

    if "name" in update_data and update_data["name"] is not None and update_data["name"] != job.name:
        existing = db.execute(select(Job).where(Job.name == update_data["name"])).scalars().first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job with name '{update_data['name']}' already exists",
            )

    prev_schedule = job.schedule
    prev_enabled = job.enabled

    def_ids_in = update_data.pop("definition_ids", None)
    overrides_in = update_data.pop("overrides", None)
    definitions_in = update_data.pop("definitions", None)
    steps_in = update_data.pop("steps", None)

    has_notif_update = any(
        k in update_data
        for k in (
            "notifications",
            "notification_settings",
            "webhook_url",
            "notify_on_failure",
            "notify_on_regression",
            "regression_threshold",
            "email_to",
        )
    )
    notif_in = update_data.pop("notifications", None)
    notif_alias_in = update_data.pop("notification_settings", None)
    webhook_url_in = update_data.pop("webhook_url", None)
    notify_fail_in = update_data.pop("notify_on_failure", None)
    notify_reg_in = update_data.pop("notify_on_regression", None)
    reg_thr_in = update_data.pop("regression_threshold", None)
    email_to_in = update_data.pop("email_to", None)

    if has_notif_update:
        raw_notif_update = notif_in if notif_in is not None else notif_alias_in
        try:
            job.notifications = normalize_notification_settings(
                raw_notif_update,
                base=dict(job.notifications or {}),
                webhook_url=webhook_url_in,
                notify_on_failure=notify_fail_in,
                notify_on_regression=notify_reg_in,
                regression_threshold=reg_thr_in,
                email_to=email_to_in,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    extracted_def_ids, extracted_overrides = _extract_definitions_and_overrides_from_payload(
        definition_ids=def_ids_in,
        overrides=overrides_in,
        definitions_input=definitions_in,
        steps_input=steps_in,
    )

    next_def_ids = extracted_def_ids if extracted_def_ids is not None else list(job.definition_ids or [])
    next_overrides = extracted_overrides if extracted_overrides is not None else dict(job.overrides or {})

    if (
        extracted_def_ids is not None
        or extracted_overrides is not None
    ):
        _validate_job_definitions_and_overrides(db, next_def_ids, next_overrides)
        job.definition_ids = next_def_ids
        job.overrides = next_overrides

    for field, val in update_data.items():
        if val is not None or field == "description":
            setattr(job, field, val)

    schedule_or_enabled_changed = (job.schedule != prev_schedule) or (job.enabled != prev_enabled)
    if schedule_or_enabled_changed or (
        job.enabled and not is_manual_schedule(job.schedule) and job.next_run_at is None
    ) or not job.enabled or is_manual_schedule(job.schedule):
        sync_job_schedule_fields(job)

    db.commit()
    db.refresh(job)
    get_scheduler().trigger()
    return _format_job(job, db)


@router.post("/{job_id}/enable")
def enable_job(
    job_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Enable a job and compute its next scheduled run time if it has a cron schedule."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    job.enabled = True
    sync_job_schedule_fields(job)
    db.commit()
    db.refresh(job)
    get_scheduler().trigger()
    return _format_job(job, db)


@router.post("/{job_id}/disable")
def disable_job(
    job_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Disable a job and clear its next scheduled run time."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    job.enabled = False
    job.next_run_at = None
    db.commit()
    db.refresh(job)
    get_scheduler().trigger()
    return _format_job(job, db)


@router.post("/{job_id}/toggle")
def toggle_job_enabled(
    job_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Toggle a job's enabled state and update its next scheduled run time."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    job.enabled = not job.enabled
    sync_job_schedule_fields(job)
    db.commit()
    db.refresh(job)
    get_scheduler().trigger()
    return _format_job(job, db)


@router.put("/{job_id}/order")
@router.post("/{job_id}/reorder", include_in_schema=False)
def reorder_job_definitions(
    job_id: int,
    req: JobOrderUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Update the ordering of test definitions in a job."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    _validate_job_definitions_and_overrides(db, req.definition_ids, dict(job.overrides or {}))
    job.definition_ids = list(req.definition_ids)
    db.commit()
    db.refresh(job)
    return _format_job(job, db)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(
    job_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a job. Associated runs remain in history with job_id set to NULL."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    db.delete(job)
    db.commit()
    get_scheduler().trigger()


@router.post("/{job_id}/run", status_code=status.HTTP_201_CREATED)
@router.post("/{job_id}/trigger", status_code=status.HTTP_201_CREATED, include_in_schema=False)
def trigger_job_run(
    job_id: int,
    req: TriggerJobRunRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Trigger a manual run ('Run now') of a job.

    Snapshots all ordered test definitions with their merged job and runtime overrides,
    and queues a single Run that groups the per-tester results.
    """
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    req_data = req or TriggerJobRunRequest()
    if req_data.prevent_overlap:
        active_run = get_active_job_run(db, job.id)
        if active_run is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job {job_id} already has an active run (#{active_run.id} is {active_run.status})",
            )

    def_ids = (
        list(req_data.definition_ids)
        if req_data.definition_ids is not None
        else list(job.definition_ids or [])
    )
    if not def_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Job {job_id} has no test definitions configured to run",
        )

    effective_run_mode = _normalize_run_mode(req_data.run_mode or job.run_mode or "sequential")
    saved_overrides = dict(job.overrides or {})
    runtime_overrides = dict(req_data.overrides or {})

    step_snapshots: list[dict[str, Any]] = []
    for idx, def_id in enumerate(def_ids):
        def_obj = db.get(TestDefinition, def_id)
        if not def_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"TestDefinition {def_id} not found",
            )

        saved_step_ovr = _extract_step_override(saved_overrides, def_id, idx)
        rt_step_ovr = _extract_step_override(runtime_overrides, def_id, idx)

        if req_data.suites is not None:
            rt_step_ovr["suites"] = list(req_data.suites)
        if req_data.target_id is not None:
            rt_step_ovr["target_id"] = req_data.target_id
        if req_data.timeout is not None:
            rt_step_ovr["timeout"] = float(req_data.timeout)
        if req_data.config:
            rt_step_ovr["config"] = {**(rt_step_ovr.get("config") or {}), **req_data.config}

        eff_suites = (
            rt_step_ovr["suites"]
            if "suites" in rt_step_ovr and rt_step_ovr["suites"] is not None
            else (
                saved_step_ovr["suites"]
                if "suites" in saved_step_ovr and saved_step_ovr["suites"] is not None
                else list(def_obj.suites or [])
            )
        )
        eff_target_id = (
            rt_step_ovr["target_id"]
            if "target_id" in rt_step_ovr
            else (
                saved_step_ovr["target_id"]
                if "target_id" in saved_step_ovr
                else def_obj.target_id
            )
        )
        eff_timeout = (
            rt_step_ovr.get("timeout")
            if rt_step_ovr.get("timeout") is not None
            else saved_step_ovr.get("timeout")
        )

        merged_cfg: dict[str, Any] = dict(def_obj.config or {})
        if isinstance(saved_step_ovr.get("config"), dict):
            merged_cfg.update(saved_step_ovr["config"])
        if isinstance(rt_step_ovr.get("config"), dict):
            merged_cfg.update(rt_step_ovr["config"])

        validated_cfg = _validate_definition_config(def_obj.tester, merged_cfg, eff_suites)

        step_target_snap: dict[str, Any] = {}
        if eff_target_id is not None:
            target_obj = db.get(Target, int(eff_target_id))
            if not target_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Target {eff_target_id} not found",
                )
            step_target_snap = target_obj.to_snapshot()

        judge_target_id = validated_cfg.get("judge_target_id")
        if judge_target_id is not None and not isinstance(step_target_snap.get("judge_target"), dict):
            judge_target = db.get(Target, int(judge_target_id))
            if not judge_target:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Judge target {judge_target_id} not found",
                )
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

    combined_overrides = dict(saved_overrides)
    if runtime_overrides:
        combined_overrides["runtime_overrides"] = runtime_overrides

    config_snapshot: dict[str, Any] = {
        "job_id": job.id,
        "job_name": job.name,
        "run_mode": effective_run_mode,
        "tester": first_step["tester"],
        "testers": testers_list,
        "suites": first_step["suites"],
        "config": first_step["config"],
        "test_definition_id": first_step["test_definition_id"],
        "definition_ids": def_ids,
        "definitions": step_snapshots,
        "overrides": combined_overrides,
        "notifications": normalize_notification_settings(job.notifications),
    }
    if req_data.timeout is not None:
        config_snapshot["timeout"] = float(req_data.timeout)
    elif first_step.get("timeout") is not None and len(step_snapshots) == 1:
        config_snapshot["timeout"] = float(first_step["timeout"])

    target_snapshot: dict[str, Any] = (
        dict(first_step["target_snapshot"]) if first_step["target_snapshot"] else {}
    )
    if len(step_snapshots) > 1:
        target_snapshot["targets"] = [s["target_snapshot"] for s in step_snapshots]

    run = Run(
        job_id=job.id,
        status="queued",
        queued_at=utc_now(),
        config_snapshot=config_snapshot,
        target_snapshot=target_snapshot,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Wake up worker to process the queued job run
    get_worker().trigger()

    from backend.app.api.runs import _format_run_detail
    return _format_run_detail(run)


@router.get("/{job_id}/notifications")
def get_job_notifications(
    job_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get per-job notification settings."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    norm = normalize_notification_settings(job.notifications)
    return {**norm, "job_id": job.id, "notifications": norm}


@router.put("/{job_id}/notifications")
@router.patch("/{job_id}/notifications")
def update_job_notifications(
    job_id: int,
    payload: dict[str, Any],
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Update per-job notification settings."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    raw_input = (
        payload.get("notifications")
        if isinstance(payload.get("notifications"), dict)
        else (
            payload.get("notification_settings")
            if isinstance(payload.get("notification_settings"), dict)
            else payload
        )
    )
    try:
        norm = normalize_notification_settings(
            raw_input,
            base=dict(job.notifications or {}),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    job.notifications = norm
    db.commit()
    db.refresh(job)
    return {**norm, "job_id": job.id, "notifications": norm}


@router.post("/{job_id}/notifications/test")
async def test_job_notifications(
    job_id: int,
    req: TestJobNotificationRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Send a test notification (or evaluate and notify for a specific run) for a job."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    req_data = req or TestJobNotificationRequest()
    try:
        eff_settings = normalize_notification_settings(
            req_data.notifications,
            base=dict(job.notifications or {}),
            webhook_url=req_data.webhook_url,
            email_to=req_data.email_to,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    if req_data.run_id is not None:
        run_obj = db.get(Run, req_data.run_id)
        if not run_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run {req_data.run_id} not found",
            )
        report = await process_run_notifications(
            db,
            req_data.run_id,
            force=True,
            settings_override=eff_settings,
        )
        return {
            "job_id": job.id,
            "run_id": req_data.run_id,
            **report,
        }

    # Otherwise dispatch a synthetic test notification payload
    evaluation = {
        "should_notify": True,
        "reasons": ["test"],
        "failed": False,
        "regressed": False,
        "settings": eff_settings,
        "failure": {
            "failed": False,
            "run_status": "completed",
            "failed_results": [],
            "failed_cases": [],
            "summary": None,
        },
        "regression": {
            "has_regression": False,
            "compared": False,
            "baseline_run_id": None,
            "current_run_id": None,
            "threshold": eff_settings.get("regression_threshold", 0.0),
            "metrics_regressed": 0,
            "cases_regressed": 0,
            "regressions": [],
            "compare_url": None,
            "comparisons": [],
        },
        "baseline_run_id": None,
    }
    payload = build_notification_payload(run=None, job=job, evaluation=evaluation, is_test=True)
    deliveries = await dispatch_notifications(settings=eff_settings, payload=payload)
    return {
        "job_id": job.id,
        "triggered": len(deliveries) > 0,
        "ok": any(d.get("ok") for d in deliveries),
        "deliveries": deliveries,
        "payload": payload,
    }


@router.get("/{job_id}/runs")
def list_job_runs(
    job_id: int,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List runs triggered for a specific job."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    stmt = (
        select(Run)
        .where(Run.job_id == job_id)
        .order_by(Run.id.desc())
        .offset(offset)
        .limit(limit)
    )
    runs = db.execute(stmt).scalars().all()
    from backend.app.api.runs import _format_run_detail
    return [_format_run_detail(r) for r in runs]


@router.get("/{job_id}/bamboo")
def get_job_bamboo_info(
    job_id: int,
    base_url: str = Query("http://localhost:8000", description="Base URL of Test Center for snippet generation"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get Bamboo / CI integration configuration and ready-to-use plan script for a job."""
    from backend.app.api.bamboo import (
        build_bamboo_plan_script,
        get_configured_bamboo_token_env_vars,
        is_bamboo_token_configured,
    )

    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    env_vars = get_configured_bamboo_token_env_vars(job)
    primary_env_var = env_vars[0] if env_vars else "BAMBOO_API_TOKEN"
    return {
        "job_id": job.id,
        "job_name": job.name,
        "token_configured": is_bamboo_token_configured(job),
        "token_env_var": primary_env_var,
        "candidate_token_env_vars": env_vars,
        "trigger_url": f"/api/bamboo/jobs/{job.id}/trigger",
        "poll_url_template": "/api/bamboo/runs/{run_id}",
        "junit_url_template": "/api/bamboo/runs/{run_id}/junit.xml",
        "public_junit_url_template": "/api/runs/{run_id}/junit.xml",
        "script": build_bamboo_plan_script(
            job,
            base_url=base_url,
            token_env_var=primary_env_var,
        ),
    }


@router.post("/{job_id}/bamboo/trigger", status_code=status.HTTP_201_CREATED)
@router.post("/{job_id}/bamboo/run", status_code=status.HTTP_201_CREATED, include_in_schema=False)
@router.post("/{job_id}/ci/trigger", status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def trigger_job_from_bamboo_route(
    job_id: int,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    _token: str = Depends(
        __import__(
            "backend.app.api.bamboo",
            fromlist=["require_bamboo_token"],
        ).require_bamboo_token
    ),
) -> dict[str, Any]:
    """Token-protected alias under /api/jobs/{job_id}/bamboo/trigger to trigger a job from CI."""
    from backend.app.api.bamboo import (
        BambooTriggerRequest,
        _trigger_bamboo_job_impl,
        verify_bamboo_token_value,
    )

    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    verify_bamboo_token_value(_token, job=job)

    body_dict: dict[str, Any] = {}
    try:
        raw_bytes = await request.body()
        if raw_bytes and raw_bytes.strip():
            import json
            parsed = json.loads(raw_bytes)
            if isinstance(parsed, dict):
                body_dict = parsed
    except Exception:
        pass

    req_data = BambooTriggerRequest(**body_dict)
    return await _trigger_bamboo_job_impl(job, req_data, request, response, db)


