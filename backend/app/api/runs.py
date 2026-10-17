"""API routes for managing and executing test runs, filtering history, metric trends, retention, streaming logs, and retrieving artifacts."""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any, AsyncIterator, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.api.artifacts import (
    build_artifact_content_payload,
    build_artifact_download_response,
    build_artifact_preview_response,
    format_artifact_metadata,
    resolve_run_artifact,
    sync_run_disk_artifacts,
)
from backend.app.api.compare import ComparedItem, _extract_item_metadata, _parse_metric_entry
from backend.app.config import get_settings
from backend.app.db.base import utc_now
from backend.app.db.models.run import Run
from backend.app.db.models.target import Target
from backend.app.db.models.result import Result
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.artifact import Artifact
from backend.app.db.session import get_db
from backend.app.runner.events import get_broadcaster
from backend.app.runner.retention import (
    ACTIVE_RUN_STATUSES,
    apply_retention_policy,
    get_retention_status,
    save_retention_settings,
)
from backend.app.runner.worker import get_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])
trends_alias_router = APIRouter(prefix="/trends", tags=["runs"])
retention_alias_router = APIRouter(prefix="/retention", tags=["runs"])

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class CreateRunRequest(BaseModel):
    test_definition_id: int | None = Field(default=None, description="Optional TestDefinition ID to run")
    tester: str | None = Field(default=None, description="Tester plugin name (defaults to 'dummy' or definition's tester)")
    suites: list[str] | None = Field(default=None, description="Sub-suites to run")
    config: dict[str, Any] = Field(default_factory=dict, description="Tester configuration dictionary")
    target_id: int | None = Field(default=None, description="Target ID")
    target: dict[str, Any] | None = Field(default=None, description="Inline target configuration dictionary")
    job_id: int | None = Field(default=None, description="Job ID if triggered by job")
    timeout: float | None = Field(default=None, description="Execution timeout in seconds")


class RetentionUpdateRequest(BaseModel):
    enabled: bool | None = Field(default=None, description="Enable or disable automatic retention cleanup")
    max_age_days: int | None = Field(default=None, description="Delete runs/artifacts older than this many days (0 or null = unlimited)")
    retention_days: int | None = Field(default=None, description="Alias for max_age_days")
    days: int | None = Field(default=None, description="Alias for max_age_days")
    max_runs: int | None = Field(default=None, description="Keep at most this many recent terminal runs (0 or null = unlimited)")
    keep_last_n: int | None = Field(default=None, description="Alias for max_runs")
    artifacts_only: bool | None = Field(
        default=None,
        description="When true, delete only artifact files while retaining Run and Result rows in the database",
    )
    keep_failed: bool | None = Field(
        default=None,
        description="When true, exclude failed/errored runs from retention cleanup",
    )
    apply_now: bool = Field(
        default=False,
        description="When true, immediately execute retention cleanup after saving settings",
    )


class RetentionCleanupRequest(BaseModel):
    max_age_days: int | None = Field(default=None, description="Optional override for max age in days")
    retention_days: int | None = Field(default=None, description="Alias for max_age_days")
    days: int | None = Field(default=None, description="Alias for max_age_days")
    max_runs: int | None = Field(default=None, description="Optional override for max terminal runs to keep")
    keep_last_n: int | None = Field(default=None, description="Alias for max_runs")
    artifacts_only: bool | None = Field(default=None, description="Optional override for artifacts_only mode")
    keep_failed: bool | None = Field(default=None, description="Optional override for keep_failed flag")
    dry_run: bool = Field(default=False, description="When true, preview eligible deletions without modifying DB or disk")


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_run_timestamp(run: Run) -> datetime | None:
    """Return the primary execution timestamp of a run in UTC."""
    for candidate in (run.started_at, run.queued_at, run.finished_at, run.created_at):
        if candidate is not None:
            return _ensure_utc(candidate)
    return None


def _parse_date_bound(raw: str | None, *, is_end: bool = False, param_name: str = "date") -> datetime | None:
    """Parse an ISO-8601 date ('YYYY-MM-DD') or datetime string into a UTC datetime."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    if _DATE_ONLY_RE.match(text):
        try:
            year, month, day = (int(p) for p in text.split("-"))
            if is_end:
                return datetime(year, month, day, 23, 59, 59, 999999, tzinfo=timezone.utc)
            return datetime(year, month, day, 0, 0, 0, 0, tzinfo=timezone.utc)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid date {raw!r} for '{param_name}': {e}",
            ) from e

    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(normalized)
        return _ensure_utc(dt)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ISO date/datetime {raw!r} for '{param_name}'",
        ) from e


def _extract_run_testers(run: Run) -> list[str]:
    """Collect all tester names associated with a Run."""
    testers: list[str] = []

    def _add(val: Any) -> None:
        if isinstance(val, str):
            for part in val.split(","):
                p = part.strip()
                if p and p not in testers:
                    testers.append(p)

    cfg = run.config_snapshot
    if isinstance(cfg, dict):
        if isinstance(cfg.get("testers"), list):
            for item in cfg["testers"]:
                _add(item)
        _add(cfg.get("tester"))
        if isinstance(cfg.get("definitions"), list):
            for d in cfg["definitions"]:
                if isinstance(d, dict):
                    _add(d.get("tester"))
    elif isinstance(cfg, list):
        for d in cfg:
            if isinstance(d, dict):
                _add(d.get("tester"))

    for res in run.results or []:
        _add(res.tester)

    return testers


def _extract_run_targets(run: Run) -> tuple[list[int], list[str]]:
    """Collect all target IDs and target names associated with a Run."""
    target_ids: list[int] = []
    target_names: list[str] = []

    def _add_target_dict(d: Any) -> None:
        if not isinstance(d, dict):
            return
        tid = d.get("id") if d.get("id") is not None else d.get("target_id")
        if isinstance(tid, int) and not isinstance(tid, bool) and tid not in target_ids:
            target_ids.append(tid)
        tname = d.get("name") or d.get("target_name")
        if isinstance(tname, str) and tname.strip() and tname.strip() not in target_names:
            target_names.append(tname.strip())

    tsnap = run.target_snapshot
    if isinstance(tsnap, dict):
        _add_target_dict(tsnap)
        if isinstance(tsnap.get("targets"), list):
            for t in tsnap["targets"]:
                _add_target_dict(t)
    elif isinstance(tsnap, list):
        for t in tsnap:
            _add_target_dict(t)

    cfg = run.config_snapshot
    if isinstance(cfg, dict) and isinstance(cfg.get("definitions"), list):
        for d in cfg["definitions"]:
            if isinstance(d, dict):
                _add_target_dict(d)
                _add_target_dict(d.get("target_snapshot"))
                _add_target_dict(d.get("target"))
    elif isinstance(cfg, list):
        for d in cfg:
            if isinstance(d, dict):
                _add_target_dict(d)
                _add_target_dict(d.get("target_snapshot"))
                _add_target_dict(d.get("target"))

    return target_ids, target_names


def _extract_run_models(run: Run) -> list[str]:
    """Collect all model names associated with a Run."""
    models: list[str] = []

    def _add_model(val: Any) -> None:
        if isinstance(val, str) and val.strip() and val.strip() not in models:
            models.append(val.strip())

    def _scan_config_and_target(cfg_dict: Any, tgt_dict: Any) -> None:
        if isinstance(cfg_dict, dict):
            _add_model(cfg_dict.get("model"))
            _add_model(cfg_dict.get("transcription_model"))
            inner = cfg_dict.get("config")
            if isinstance(inner, dict):
                _add_model(inner.get("model"))
                _add_model(inner.get("transcription_model"))
        if isinstance(tgt_dict, dict):
            _add_model(tgt_dict.get("model"))
            extra = tgt_dict.get("extra")
            if isinstance(extra, dict):
                _add_model(extra.get("model"))
                _add_model(extra.get("transcription_model"))

    _scan_config_and_target(run.config_snapshot, run.target_snapshot)
    if isinstance(run.config_snapshot, dict) and isinstance(run.config_snapshot.get("definitions"), list):
        for d in run.config_snapshot["definitions"]:
            if isinstance(d, dict):
                _scan_config_and_target(d, d.get("target_snapshot") or d.get("target"))
    elif isinstance(run.config_snapshot, list):
        for d in run.config_snapshot:
            if isinstance(d, dict):
                _scan_config_and_target(d, d.get("target_snapshot") or d.get("target"))

    for res in run.results or []:
        for c in res.cases or []:
            if isinstance(c.detail, dict):
                _add_model(c.detail.get("model"))

    return models


def _extract_run_job_info(run: Run) -> tuple[int | None, str | None]:
    """Extract (job_id, job_name) from a Run and its config_snapshot."""
    job_id = run.job_id
    job_name: str | None = None
    if isinstance(run.config_snapshot, dict):
        if job_id is None and isinstance(run.config_snapshot.get("job_id"), int):
            job_id = run.config_snapshot["job_id"]
        if isinstance(run.config_snapshot.get("job_name"), str):
            job_name = run.config_snapshot["job_name"]
    if not job_name and run.job is not None:
        job_name = run.job.name
    return job_id, job_name


def _matches_run_filters(
    run: Run,
    *,
    status_filter: str | None = None,
    tester: str | None = None,
    target_id: int | None = None,
    target: str | None = None,
    model: str | None = None,
    job_id: int | None = None,
    job: str | None = None,
    dt_from: datetime | None = None,
    dt_to: datetime | None = None,
) -> bool:
    """Evaluate whether a Run matches all provided history filter criteria."""
    if status_filter:
        sf = status_filter.strip().lower()
        if sf == "pass":
            if run.status != "pass" and not (
                run.results and all(r.status == "pass" for r in run.results)
            ):
                return False
        elif sf == "fail":
            if run.status not in ("fail", "failed") and not any(
                r.status == "fail" for r in (run.results or [])
            ):
                return False
        elif sf == "failed":
            if run.status != "failed" and not any(
                r.status == "fail" for r in (run.results or [])
            ):
                return False
        else:
            if run.status.lower() != sf and not any(
                r.status.lower() == sf for r in (run.results or [])
            ):
                return False

    if tester:
        tf = tester.strip().lower()
        run_testers = [t.lower() for t in _extract_run_testers(run)]
        if tf not in run_testers:
            return False

    if target_id is not None:
        t_ids, _ = _extract_run_targets(run)
        if target_id not in t_ids:
            return False

    if target:
        tg_f = target.strip().lower()
        t_ids, t_names = _extract_run_targets(run)
        if tg_f in ("none", "no-target", "unassigned"):
            if t_ids or t_names:
                return False
        else:
            matched_id = tg_f.isdigit() and int(tg_f) in t_ids
            matched_name = any(n.lower() == tg_f or tg_f in n.lower() for n in t_names)
            if not (matched_id or matched_name):
                return False

    if model:
        mf = model.strip().lower()
        run_models = [m.lower() for m in _extract_run_models(run)]
        if not any(m == mf or mf in m for m in run_models):
            return False

    r_job_id, r_job_name = _extract_run_job_info(run)
    if job_id is not None:
        if r_job_id != job_id:
            return False

    if job:
        jf = job.strip().lower()
        if jf in ("none", "manual", "ad-hoc", "adhoc", "null"):
            if r_job_id is not None:
                return False
        elif jf.isdigit():
            if r_job_id != int(jf):
                return False
        else:
            if not r_job_name or (r_job_name.lower() != jf and jf not in r_job_name.lower()):
                return False

    if dt_from is not None or dt_to is not None:
        r_dt = _extract_run_timestamp(run)
        if r_dt is None:
            return False
        if dt_from is not None and r_dt < dt_from:
            return False
        if dt_to is not None and r_dt > dt_to:
            return False

    return True


def _format_run_detail(run: Run) -> dict[str, Any]:
    """Helper to serialize a Run with its nested results, cases, and artifacts."""
    def_names_by_id: dict[int, str] = {}
    job_id, job_name = _extract_run_job_info(run)
    if isinstance(run.config_snapshot, dict):
        raw_defs = run.config_snapshot.get("definitions")
        if isinstance(raw_defs, list):
            for d in raw_defs:
                if isinstance(d, dict):
                    did = d.get("test_definition_id") or d.get("id")
                    dname = d.get("name")
                    if isinstance(did, int) and isinstance(dname, str):
                        def_names_by_id[did] = dname

    results_list = []
    for r in run.results:
        item_meta = _extract_item_metadata(ComparedItem(run=run, result=r), is_baseline=False)
        cases_list = [
            {
                "id": c.id,
                "suite": c.suite,
                "case_id": c.case_id,
                "status": c.status,
                "metrics": c.metrics,
                "detail": c.detail,
            }
            for c in r.cases
        ]
        artifacts_list = [
            format_artifact_metadata(a, run_id=run.id, result_id=r.id)
            for a in r.artifacts
        ]
        def_name = (
            r.test_definition.name
            if r.test_definition is not None
            else (def_names_by_id.get(r.test_definition_id) if r.test_definition_id is not None else None)
        )
        results_list.append({
            "id": r.id,
            "test_definition_id": r.test_definition_id,
            "definition_name": def_name,
            "tester": r.tester,
            "status": r.status,
            "target_id": item_meta.get("target_id"),
            "target_name": item_meta.get("target_name"),
            "model": item_meta.get("model"),
            "metrics": r.metrics,
            "summary": r.summary,
            "cases": cases_list,
            "artifacts": artifacts_list,
        })

    notif_report = (
        run.config_snapshot.get("notification_report")
        if isinstance(run.config_snapshot, dict)
        else None
    )
    notifs_sent = (
        run.config_snapshot.get("notifications_sent")
        if isinstance(run.config_snapshot, dict)
        else None
    )
    bamboo_meta = (
        run.config_snapshot.get("bamboo")
        if isinstance(run.config_snapshot, dict) and isinstance(run.config_snapshot.get("bamboo"), dict)
        else (
            run.git_info
            if isinstance(run.git_info, dict) and run.git_info.get("source") == "bamboo"
            else None
        )
    )
    bamboo_report = (
        run.config_snapshot.get("bamboo_report")
        if isinstance(run.config_snapshot, dict) and isinstance(run.config_snapshot.get("bamboo_report"), dict)
        else None
    )

    return {
        "id": run.id,
        "job_id": run.job_id,
        "job_name": job_name,
        "status": run.status,
        "queued_at": run.queued_at.isoformat() if run.queued_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "config_snapshot": run.config_snapshot,
        "target_snapshot": run.target_snapshot,
        "git_info": run.git_info,
        "bamboo": bamboo_meta,
        "bamboo_report": bamboo_report,
        "junit_url": f"/api/runs/{run.id}/junit.xml",
        "notification_report": notif_report,
        "notifications_sent": notifs_sent,
        "results": results_list,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_run(
    req: CreateRunRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Queue a new test run manually, from a test definition, or from a job."""
    if req.job_id is not None and req.test_definition_id is None and req.tester is None:
        from backend.app.api.jobs import TriggerJobRunRequest, trigger_job_run
        return trigger_job_run(
            job_id=req.job_id,
            req=TriggerJobRunRequest(
                config=req.config if req.config else None,
                suites=req.suites,
                target_id=req.target_id,
                timeout=req.timeout,
            ),
            db=db,
        )

    test_def = None
    if req.test_definition_id is not None:
        from backend.app.db.models.test_definition import TestDefinition
        test_def = db.get(TestDefinition, req.test_definition_id)
        if not test_def:
            raise HTTPException(status_code=404, detail=f"TestDefinition {req.test_definition_id} not found")

    target_id = req.target_id
    if target_id is None and req.target is None and test_def and test_def.target_id:
        target_id = test_def.target_id

    target_snapshot: dict[str, Any] = {}
    if target_id:
        target = db.get(Target, target_id)
        if not target:
            raise HTTPException(status_code=404, detail=f"Target id {target_id} not found")
        target_snapshot = target.to_snapshot()
    elif req.target:
        target_snapshot = req.target

    tester_name = req.tester
    if tester_name is None:
        tester_name = test_def.tester if test_def else "dummy"

    suites = req.suites
    if suites is None and test_def:
        suites = test_def.suites

    merged_config: dict[str, Any] = {}
    if test_def:
        merged_config.update(test_def.config)
    if req.config:
        merged_config.update(req.config)

    judge_target_id = merged_config.get("judge_target_id")
    if judge_target_id is not None and not isinstance(target_snapshot.get("judge_target"), dict):
        judge_target = db.get(Target, int(judge_target_id))
        if not judge_target:
            raise HTTPException(status_code=404, detail=f"Judge target id {judge_target_id} not found")
        target_snapshot = dict(target_snapshot)
        target_snapshot["judge_target"] = judge_target.to_snapshot()

    config_snapshot: dict[str, Any] = {
        "tester": tester_name,
        "suites": suites,
        "config": merged_config,
    }
    if test_def:
        config_snapshot["test_definition_id"] = test_def.id
    if req.timeout:
        config_snapshot["timeout"] = req.timeout

    run = Run(
        job_id=req.job_id,
        status="queued",
        queued_at=utc_now(),
        config_snapshot=config_snapshot,
        target_snapshot=target_snapshot,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Wake up worker to process the new run
    get_worker().trigger()

    return _format_run_detail(run)


@router.get("")
def list_runs(
    status: str | None = Query(None, description="Filter by run status or result status"),
    tester: str | None = Query(None, description="Filter by tester plugin name"),
    target_id: int | None = Query(None, description="Filter by target ID"),
    target: str | None = Query(None, description="Filter by target name or ID"),
    model: str | None = Query(None, description="Filter by model name"),
    job_id: int | None = Query(None, description="Filter by job ID"),
    job: str | None = Query(None, description="Filter by job ID, job name, or 'manual'/'none'"),
    date_from: str | None = Query(None, description="Filter runs on or after date/datetime (YYYY-MM-DD or ISO-8601)"),
    date_to: str | None = Query(None, description="Filter runs on or before date/datetime (YYYY-MM-DD or ISO-8601)"),
    start_date: str | None = Query(None, description="Alias for date_from"),
    end_date: str | None = Query(None, description="Alias for date_to"),
    date: str | None = Query(None, description="Filter runs on a specific date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List runs ordered by creation time descending, with optional filtering by tester, target, job, status, and date."""
    eff_from_raw = date_from or start_date or date
    eff_to_raw = date_to or end_date or date
    dt_from = _parse_date_bound(eff_from_raw, is_end=False, param_name="date_from")
    dt_to = _parse_date_bound(eff_to_raw, is_end=True, param_name="date_to")

    has_memory_filters = any([
        tester is not None,
        target_id is not None,
        target is not None,
        model is not None,
        job is not None,
        dt_from is not None,
        dt_to is not None,
        status is not None and status.strip().lower() in ("pass", "fail", "failed"),
    ])

    stmt = (
        select(Run)
        .options(
            selectinload(Run.job),
            selectinload(Run.results).selectinload(Result.cases),
            selectinload(Run.results).selectinload(Result.artifacts),
            selectinload(Run.results).selectinload(Result.test_definition),
        )
        .order_by(Run.id.desc())
    )

    if status and not has_memory_filters:
        stmt = stmt.where(Run.status == status)
    if job_id is not None and not has_memory_filters:
        stmt = stmt.where(Run.job_id == job_id)

    if not has_memory_filters:
        stmt = stmt.offset(offset).limit(limit)
        runs = db.execute(stmt).scalars().all()
        return [_format_run_detail(r) for r in runs]

    all_runs = db.execute(stmt).scalars().all()
    filtered = [
        r
        for r in all_runs
        if _matches_run_filters(
            r,
            status_filter=status,
            tester=tester,
            target_id=target_id,
            target=target,
            model=model,
            job_id=job_id,
            job=job,
            dt_from=dt_from,
            dt_to=dt_to,
        )
    ]
    sliced = filtered[offset : offset + limit]
    return [_format_run_detail(r) for r in sliced]


@router.get("/trends")
@trends_alias_router.get("", include_in_schema=False)
def get_run_trends(
    metric: str | None = Query(None, description="Metric name to chart (defaults to first available metric)"),
    tester: str | None = Query(None, description="Filter by tester plugin name"),
    target_id: int | None = Query(None, description="Filter by target ID"),
    target: str | None = Query(None, description="Filter by target name or ID"),
    model: str | None = Query(None, description="Filter by model name"),
    job_id: int | None = Query(None, description="Filter by job ID"),
    job: str | None = Query(None, description="Filter by job ID, job name, or 'manual'/'none'"),
    status: str | None = Query(None, description="Filter by run or result status"),
    date_from: str | None = Query(None, description="Filter runs on or after date/datetime"),
    date_to: str | None = Query(None, description="Filter runs on or before date/datetime"),
    start_date: str | None = Query(None, description="Alias for date_from"),
    end_date: str | None = Query(None, description="Alias for date_to"),
    date: str | None = Query(None, description="Filter runs on a specific date"),
    limit: int = Query(200, ge=1, le=1000, description="Max runs to scan"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Compute time-series trend data for a chosen metric over time per target/model."""
    eff_from_raw = date_from or start_date or date
    eff_to_raw = date_to or end_date or date
    dt_from = _parse_date_bound(eff_from_raw, is_end=False, param_name="date_from")
    dt_to = _parse_date_bound(eff_to_raw, is_end=True, param_name="date_to")

    stmt = (
        select(Run)
        .options(
            selectinload(Run.job),
            selectinload(Run.results).selectinload(Result.cases),
            selectinload(Run.results).selectinload(Result.artifacts),
            selectinload(Run.results).selectinload(Result.test_definition),
        )
        .order_by(Run.id.desc())
    )
    all_runs = db.execute(stmt).scalars().all()
    matching_runs = [
        r
        for r in all_runs
        if _matches_run_filters(
            r,
            status_filter=status,
            tester=tester,
            target_id=target_id,
            target=target,
            model=model,
            job_id=job_id,
            job=job,
            dt_from=dt_from,
            dt_to=dt_to,
        )
    ][:limit]

    # Sort chronologically (oldest first) for trend visualization
    matching_runs.sort(
        key=lambda r: (
            (_extract_run_timestamp(r) or datetime.min.replace(tzinfo=timezone.utc)).timestamp(),
            r.id,
        )
    )

    metric_catalog: dict[str, dict[str, Any]] = {}
    extracted_items: list[dict[str, Any]] = []

    for run in matching_runs:
        r_job_id, r_job_name = _extract_run_job_info(run)
        r_dt = _extract_run_timestamp(run)
        ts_iso = r_dt.isoformat() if r_dt else None

        for res in run.results or []:
            if tester and res.tester.lower() != tester.strip().lower():
                continue

            meta = _extract_item_metadata(ComparedItem(run=run, result=res), is_baseline=False)
            res_target_id = meta.get("target_id")
            res_target_name = meta.get("target_name")
            res_model = meta.get("model")

            if target_id is not None and res_target_id != target_id:
                continue
            if target:
                tg_f = target.strip().lower()
                if tg_f not in ("none", "no-target", "unassigned"):
                    m_id = tg_f.isdigit() and res_target_id == int(tg_f)
                    m_name = bool(
                        res_target_name
                        and (res_target_name.lower() == tg_f or tg_f in res_target_name.lower())
                    )
                    if not (m_id or m_name):
                        continue
            if model:
                mf = model.strip().lower()
                res_models = [str(m).lower() for m in (meta.get("models") or [])]
                if res_model:
                    res_models.append(res_model.lower())
                if not any(m == mf or mf in m for m in res_models):
                    continue

            if res_target_name and res_model:
                series_label = f"{res_target_name} / {res_model}"
            elif res_target_name:
                series_label = str(res_target_name)
            elif res_model:
                series_label = str(res_model)
            else:
                series_label = f"{res.tester} (default)"

            parsed_metrics: dict[str, tuple[float | int, str | None, Literal["higher", "lower"] | None]] = {}
            for m_name, raw_entry in (res.metrics or {}).items():
                val, unit, better = _parse_metric_entry(raw_entry)
                if val is None:
                    continue
                parsed_metrics[m_name] = (val, unit, better)
                if m_name not in metric_catalog:
                    metric_catalog[m_name] = {
                        "name": m_name,
                        "unit": unit,
                        "better": better,
                        "testers": [res.tester],
                        "count": 1,
                    }
                else:
                    cat = metric_catalog[m_name]
                    if cat["unit"] is None and unit is not None:
                        cat["unit"] = unit
                    if cat["better"] is None and better is not None:
                        cat["better"] = better
                    if res.tester not in cat["testers"]:
                        cat["testers"].append(res.tester)
                    cat["count"] += 1

            extracted_items.append({
                "run_id": run.id,
                "result_id": res.id,
                "job_id": r_job_id,
                "job_name": r_job_name,
                "tester": res.tester,
                "status": res.status,
                "run_status": run.status,
                "timestamp": ts_iso,
                "target_id": res_target_id,
                "target_name": res_target_name,
                "model": res_model,
                "series_key": series_label,
                "label": series_label,
                "parsed_metrics": parsed_metrics,
            })

    available_metrics = list(metric_catalog.values())
    selected_metric = metric
    if not selected_metric and available_metrics:
        selected_metric = available_metrics[0]["name"]

    selected_meta = metric_catalog.get(selected_metric or "", {})
    selected_unit = selected_meta.get("unit")
    selected_better = selected_meta.get("better")

    series_map: dict[str, dict[str, Any]] = {}
    all_points: list[dict[str, Any]] = []

    if selected_metric:
        for item in extracted_items:
            if selected_metric not in item["parsed_metrics"]:
                continue
            val, pt_unit, pt_better = item["parsed_metrics"][selected_metric]
            eff_unit = pt_unit if pt_unit is not None else selected_unit
            eff_better = pt_better if pt_better is not None else selected_better

            point = {
                "run_id": item["run_id"],
                "result_id": item["result_id"],
                "job_id": item["job_id"],
                "job_name": item["job_name"],
                "tester": item["tester"],
                "status": item["status"],
                "run_status": item["run_status"],
                "timestamp": item["timestamp"],
                "metric": selected_metric,
                "value": val,
                "unit": eff_unit,
                "better": eff_better,
                "target_id": item["target_id"],
                "target_name": item["target_name"],
                "model": item["model"],
                "series_key": item["series_key"],
                "label": item["label"],
            }
            all_points.append(point)

            s_key = item["series_key"]
            if s_key not in series_map:
                series_map[s_key] = {
                    "series_key": s_key,
                    "label": item["label"],
                    "target_id": item["target_id"],
                    "target_name": item["target_name"],
                    "model": item["model"],
                    "tester": item["tester"],
                    "points": [],
                }
            series_map[s_key]["points"].append(point)

    return {
        "metric": selected_metric,
        "unit": selected_unit,
        "better": selected_better,
        "available_metrics": available_metrics,
        "series": list(series_map.values()),
        "points": all_points,
    }


@router.get("/retention")
@retention_alias_router.get("", include_in_schema=False)
def get_runs_retention(
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get current retention policy settings and storage statistics for old runs/artifacts."""
    return get_retention_status(db)


@router.put("/retention")
@router.post("/retention")
@retention_alias_router.put("", include_in_schema=False)
@retention_alias_router.post("", include_in_schema=False)
def update_runs_retention(
    req: RetentionUpdateRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Update retention policy settings for old runs/artifacts, optionally applying cleanup immediately."""
    payload = req.model_dump(exclude_unset=True)
    apply_now = bool(payload.pop("apply_now", False))
    try:
        save_retention_settings(payload)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    cleanup_result = None
    if apply_now:
        cleanup_result = apply_retention_policy(db)

    status_data = get_retention_status(db)
    if cleanup_result is not None:
        status_data["cleanup_result"] = cleanup_result
    return status_data


@router.post("/retention/cleanup")
@router.post("/cleanup")
@retention_alias_router.post("/cleanup", include_in_schema=False)
def trigger_runs_retention_cleanup(
    req: RetentionCleanupRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Execute retention cleanup immediately for old terminal runs and/or artifacts."""
    body = req or RetentionCleanupRequest()
    eff_days = (
        body.max_age_days
        if body.max_age_days is not None
        else (body.retention_days if body.retention_days is not None else body.days)
    )
    eff_runs = body.max_runs if body.max_runs is not None else body.keep_last_n
    try:
        return apply_retention_policy(
            db,
            max_age_days=eff_days,
            max_runs=eff_runs,
            artifacts_only=body.artifacts_only,
            keep_failed=body.keep_failed,
            dry_run=body.dry_run,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/{run_id}")
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get full details of a specific run."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    sync_run_disk_artifacts(db, run)
    return _format_run_detail(run)


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a finished run and remove its artifact files from disk."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run.status in ACTIVE_RUN_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete active run {run_id} in state '{run.status}'. Cancel it first.",
        )

    settings = get_settings()
    run_dir = settings.ARTIFACTS_DIR / str(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    db.delete(run)
    db.commit()


@router.get("/{run_id}/results")
def get_run_results(
    run_id: int,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get all results associated with a specific run."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    sync_run_disk_artifacts(db, run)
    detail = _format_run_detail(run)
    return detail["results"]


@router.get("/{run_id}/junit.xml")
@router.get("/{run_id}/junit")
@router.get("/{run_id}/export/junit.xml", include_in_schema=False)
def get_run_junit_xml(
    run_id: int,
    suite: str | None = Query(None, description="Optional filter by suite name"),
    tester: str | None = Query(None, description="Optional filter by tester plugin name"),
    result_id: int | None = Query(None, description="Optional filter by Result ID"),
    download: bool = Query(False, description="If true, force attachment download"),
    _token: str | None = Depends(
        __import__(
            "backend.app.api.bamboo",
            fromlist=["optional_verify_bamboo_token"],
        ).optional_verify_bamboo_token
    ),
    db: Session = Depends(get_db),
):
    """Export all test cases of a run as standard JUnit XML for Bamboo's JUnit parser."""
    from backend.app.api.bamboo import _load_run_with_relations, build_junit_xml_http_response
    from backend.app.runner.junit import build_run_junit_xml

    run = _load_run_with_relations(db, run_id)
    xml_str = build_run_junit_xml(
        run,
        suite_filter=suite,
        tester_filter=tester,
        result_id_filter=result_id,
    )
    return build_junit_xml_http_response(
        xml_str,
        filename=f"run-{run_id}-junit.xml",
        download=download,
    )


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Cancel a queued or currently executing run."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if run.status in ("completed", "cancelled", "error", "failed"):
        return {
            "status": "noop",
            "message": f"Run {run_id} is already in terminal state '{run.status}'",
            "run_status": run.status,
        }

    success = await get_worker().cancel_run(run_id)
    return {
        "status": "ok" if success else "failed",
        "message": f"Cancellation requested for run {run_id}",
    }


@router.get("/{run_id}/logs")
def get_run_logs(
    run_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve full logs for a run."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    settings = get_settings()
    log_file = settings.ARTIFACTS_DIR / str(run_id) / "run.log"
    content = ""
    lines: list[str] = []
    if log_file.exists():
        content = log_file.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in content.splitlines() if line]

    return {
        "run_id": run_id,
        "status": run.status,
        "lines": lines,
        "raw": content,
    }


@router.get("/{run_id}/logs/stream")
async def stream_run_logs_sse(
    run_id: int,
    db: Session = Depends(get_db),
):
    """Stream live logs and events for a run via Server-Sent Events (SSE)."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    settings = get_settings()
    log_file = settings.ARTIFACTS_DIR / str(run_id) / "run.log"

    async def sse_generator() -> AsyncIterator[str]:
        # 1. Stream existing log lines from disk
        if log_file.exists():
            try:
                content = log_file.read_text(encoding="utf-8", errors="replace")
                for line in content.splitlines():
                    if line.strip():
                        ev = {
                            "type": "log",
                            "run_id": run_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": line,
                        }
                        yield f"event: log\ndata: {json.dumps(ev)}\n\n"
            except OSError:
                pass

        # 2. Check if run is already finished
        if run.status in ("completed", "cancelled", "error", "failed"):
            yield f"event: completed\ndata: {json.dumps({'type': 'completed', 'run_id': run_id, 'status': run.status})}\n\n"
            return

        # 3. Stream live events from broadcaster
        broadcaster = get_broadcaster()
        async for event in broadcaster.subscribe(run_id):
            ev_type = event.get("type", "message")
            yield f"event: {ev_type}\ndata: {json.dumps(event)}\n\n"
            if ev_type in ("completed", "end"):
                break

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.websocket("/{run_id}/ws")
@router.websocket("/{run_id}/logs/ws")
async def websocket_run_logs(websocket: WebSocket, run_id: int):
    """Stream live logs and events for a run over WebSocket."""
    await websocket.accept()

    settings = get_settings()
    log_file = settings.ARTIFACTS_DIR / str(run_id) / "run.log"

    try:
        # Send historical logs
        if log_file.exists():
            try:
                content = log_file.read_text(encoding="utf-8", errors="replace")
                for line in content.splitlines():
                    if line.strip():
                        await websocket.send_text(
                            json.dumps({
                                "type": "log",
                                "run_id": run_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "message": line,
                            })
                        )
            except OSError:
                pass

        # Subscribe to broadcaster for live updates
        broadcaster = get_broadcaster()
        async for event in broadcaster.subscribe(run_id):
            await websocket.send_text(json.dumps(event))
            if event.get("type") in ("completed", "end"):
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for run {run_id}")
    except Exception as e:
        logger.warning(f"WebSocket error for run {run_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/{run_id}/artifacts")
def list_run_artifacts(
    run_id: int,
    kind: str | None = Query(None, description="Optional filter by artifact kind (html, json, audio, log, image, binary)"),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all artifacts stored under data/artifacts/<run_id>/ and associated with the run."""
    stmt = (
        select(Run)
        .options(selectinload(Run.results).selectinload(Result.artifacts))
        .where(Run.id == run_id)
    )
    run = db.execute(stmt).scalars().first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    artifacts = sync_run_disk_artifacts(db, run)
    if kind:
        k_lower = kind.strip().lower()
        artifacts = [a for a in artifacts if a.get("kind") == k_lower]
    return artifacts


@router.get("/{run_id}/artifacts/{artifact_ref:path}/download")
def download_artifact(
    run_id: int,
    artifact_ref: str,
    db: Session = Depends(get_db),
):
    """Download an artifact file associated with a run."""
    meta, file_path = resolve_run_artifact(db, run_id, artifact_ref)
    return build_artifact_download_response(meta, file_path)


@router.get("/{run_id}/artifacts/{artifact_ref:path}/preview")
@router.get("/{run_id}/artifacts/{artifact_ref:path}/view")
def preview_artifact(
    run_id: int,
    artifact_ref: str,
    db: Session = Depends(get_db),
):
    """Safely preview an artifact inline (sandboxed CSP for HTML, inline streaming for audio/log/JSON)."""
    meta, file_path = resolve_run_artifact(db, run_id, artifact_ref)
    return build_artifact_preview_response(meta, file_path)


@router.get("/{run_id}/artifacts/{artifact_ref:path}/content")
def get_artifact_content(
    run_id: int,
    artifact_ref: str,
    max_bytes: int = Query(5_000_000, ge=1024, le=50_000_000, description="Maximum text bytes to read"),
    search: str | None = Query(None, description="Optional substring filter for log lines"),
    level: str | None = Query(None, description="Optional log level filter (INFO, WARNING, ERROR)"),
    tail: int | None = Query(None, ge=1, description="Optional tail N lines for log artifacts"),
    offset: int = Query(0, ge=0, description="Line offset for log artifacts"),
    limit: int | None = Query(None, ge=1, description="Line limit for log artifacts"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve structured artifact content for in-page viewers (HTML, JSON tree, audio player, log viewer)."""
    meta, file_path = resolve_run_artifact(db, run_id, artifact_ref)
    return build_artifact_content_payload(
        meta,
        file_path,
        max_bytes=max_bytes,
        search=search,
        level=level,
        tail=tail,
        offset=offset,
        limit=limit,
    )


@router.get("/{run_id}/artifacts/{artifact_ref:path}")
def get_run_artifact_detail(
    run_id: int,
    artifact_ref: str,
    max_bytes: int = Query(5_000_000, ge=1024, le=50_000_000),
    search: str | None = Query(None),
    level: str | None = Query(None),
    tail: int | None = Query(None, ge=1),
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get artifact metadata and parsed content by artifact ID or relative path under data/artifacts/<run_id>/."""
    meta, file_path = resolve_run_artifact(db, run_id, artifact_ref)
    return build_artifact_content_payload(
        meta,
        file_path,
        max_bytes=max_bytes,
        search=search,
        level=level,
        tail=tail,
        offset=offset,
        limit=limit,
    )

