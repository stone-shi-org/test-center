"""API routes for managing named run baselines."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.db.base import utc_now
from backend.app.db.models.baseline import Baseline
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run
from backend.app.db.session import get_db
from backend.app.runner.retention import TERMINAL_RUN_STATUSES

router = APIRouter(prefix="/baselines", tags=["baselines"])


class CreateBaselineRequest(BaseModel):
    name: str = Field(..., description="Unique baseline name (case-insensitive)")
    description: str | None = Field(default=None, description="Optional description")
    move: bool = Field(
        default=False,
        description="When true and name already exists, move the baseline to this run and record the previous run in history",
    )


class UpdateBaselineRequest(BaseModel):
    name: str | None = Field(default=None, description="New unique baseline name (case-insensitive)")
    description: str | None = Field(default=None, description="Updated description")


def format_baseline(baseline: Baseline, run: Run | None = None) -> dict[str, Any]:
    """Serialize a Baseline along with its associated Run summary."""
    from backend.app.api.runs import (
        _extract_run_models,
        _extract_run_targets,
        _extract_run_testers,
    )

    eff_run = run or baseline.run
    testers: list[str] = []
    target_ids: list[int] = []
    target_names: list[str] = []
    models: list[str] = []
    headline_metrics: dict[str, Any] = {}
    run_status: str | None = None
    finished_at: str | None = None
    started_at: str | None = None
    queued_at: str | None = None

    if eff_run is not None:
        testers = _extract_run_testers(eff_run)
        target_ids, target_names = _extract_run_targets(eff_run)
        models = _extract_run_models(eff_run)
        run_status = eff_run.status
        finished_at = eff_run.finished_at.isoformat() if eff_run.finished_at else None
        started_at = eff_run.started_at.isoformat() if eff_run.started_at else None
        queued_at = eff_run.queued_at.isoformat() if eff_run.queued_at else None
        for res in eff_run.results or []:
            if isinstance(res.metrics, dict):
                for m_key, m_val in res.metrics.items():
                    if m_key not in headline_metrics:
                        headline_metrics[m_key] = m_val

    primary_tester = testers[0] if testers else None
    primary_target = target_names[0] if target_names else None
    primary_target_id = target_ids[0] if target_ids else None
    primary_model = models[0] if models else None

    return {
        "id": baseline.id,
        "name": baseline.name,
        "run_id": baseline.run_id,
        "description": baseline.description,
        "history": list(baseline.history or []),
        "created_at": baseline.created_at.isoformat() if baseline.created_at else None,
        "updated_at": baseline.updated_at.isoformat() if baseline.updated_at else None,
        "status": run_status,
        "run_status": run_status,
        "tester": primary_tester,
        "testers": testers,
        "target": primary_target,
        "target_name": primary_target,
        "target_id": primary_target_id,
        "model": primary_model,
        "models": models,
        "finished_at": finished_at,
        "started_at": started_at,
        "queued_at": queued_at,
        "metrics": headline_metrics,
        "run": {
            "id": baseline.run_id,
            "status": run_status,
            "tester": primary_tester,
            "testers": testers,
            "target": primary_target,
            "target_name": primary_target,
            "target_id": primary_target_id,
            "model": primary_model,
            "models": models,
            "finished_at": finished_at,
            "started_at": started_at,
            "queued_at": queued_at,
            "metrics": headline_metrics,
        },
    }


def get_previous_baselines_by_run_id(db: Session) -> dict[int, list[dict[str, Any]]]:
    """Build a mapping of run_id -> list of baselines that previously pointed to that run."""
    all_baselines = db.execute(select(Baseline).order_by(Baseline.id.asc())).scalars().all()
    by_run: dict[int, list[dict[str, Any]]] = {}
    for b in all_baselines:
        for entry in b.history or []:
            if not isinstance(entry, dict):
                continue
            hist_run_id = entry.get("run_id")
            if not isinstance(hist_run_id, int) or hist_run_id == b.run_id:
                continue
            existing_list = by_run.setdefault(hist_run_id, [])
            if not any(item["id"] == b.id for item in existing_list):
                existing_list.append({
                    "id": b.id,
                    "name": b.name,
                    "current_run_id": b.run_id,
                    "from": entry.get("from"),
                    "to": entry.get("to"),
                })
    return by_run


def save_or_move_run_baseline(
    db: Session,
    run: Run,
    req: CreateBaselineRequest,
) -> Baseline | JSONResponse:
    """Create a new baseline or move an existing baseline to `run`."""
    if run.status not in TERMINAL_RUN_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot save active run #{run.id} (status '{run.status}') as a baseline. Only finished runs can be baselined.",
        )

    clean_name = req.name.strip()
    if not clean_name:
        raise HTTPException(
            status_code=422,
            detail="Baseline name cannot be empty",
        )

    existing = (
        db.execute(
            select(Baseline).where(func.lower(Baseline.name) == clean_name.lower())
        )
        .scalars()
        .first()
    )

    if existing is not None:
        if not req.move:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "detail": f"Baseline '{existing.name}' already exists and points to run #{existing.run_id}",
                    "run_id": existing.run_id,
                    "current_run_id": existing.run_id,
                    "baseline_id": existing.id,
                    "name": existing.name,
                },
            )

        now = utc_now()
        now_iso = now.isoformat()
        old_history = list(existing.history or [])
        if existing.run_id != run.id:
            if old_history and isinstance(old_history[-1], dict) and old_history[-1].get("to"):
                from_iso = old_history[-1]["to"]
            elif existing.created_at is not None:
                from_iso = existing.created_at.isoformat()
            else:
                from_iso = now_iso
            old_history.append({
                "run_id": existing.run_id,
                "from": from_iso,
                "to": now_iso,
            })
            existing.history = old_history
            existing.run_id = run.id

        existing.name = clean_name
        if "description" in req.model_fields_set:
            existing.description = req.description
        existing.updated_at = now
        db.commit()
        db.refresh(existing)
        return existing

    now = utc_now()
    baseline = Baseline(
        name=clean_name,
        run_id=run.id,
        description=req.description,
        history=[],
        created_at=now,
        updated_at=now,
    )
    db.add(baseline)
    db.commit()
    db.refresh(baseline)
    return baseline


@router.get("")
def list_baselines(
    q: str | None = Query(None, description="Case-insensitive search by baseline name or description"),
    tester: str | None = Query(None, description="Filter by tester name"),
    target: str | None = Query(None, description="Filter by target name or ID"),
    model: str | None = Query(None, description="Filter by model name"),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all baselines with a summary of their associated run. Not limited to recent runs."""
    stmt = (
        select(Baseline)
        .options(
            selectinload(Baseline.run).selectinload(Run.results).selectinload(Result.cases),
        )
        .order_by(Baseline.updated_at.desc(), Baseline.id.desc())
    )
    baselines = db.execute(stmt).scalars().all()

    q_norm = q.strip().lower() if q and q.strip() else None
    tester_norm = tester.strip().lower() if tester and tester.strip() else None
    target_norm = target.strip().lower() if target and target.strip() else None
    model_norm = model.strip().lower() if model and model.strip() else None

    items: list[dict[str, Any]] = []
    for b in baselines:
        formatted = format_baseline(b, b.run)

        if q_norm:
            name_match = q_norm in b.name.lower()
            desc_match = bool(b.description and q_norm in b.description.lower())
            run_match = q_norm == str(b.run_id) or q_norm == f"#{b.run_id}"
            if not (name_match or desc_match or run_match):
                continue

        if tester_norm:
            run_testers = [t.lower() for t in (formatted.get("testers") or [])]
            if tester_norm not in run_testers:
                continue

        if target_norm:
            t_id = formatted.get("target_id")
            t_name = (formatted.get("target_name") or "").lower()
            matched_id = target_norm.isdigit() and t_id == int(target_norm)
            matched_name = bool(t_name and (t_name == target_norm or target_norm in t_name))
            if not (matched_id or matched_name):
                continue

        if model_norm:
            run_models = [m.lower() for m in (formatted.get("models") or [])]
            if not any(m == model_norm or model_norm in m for m in run_models):
                continue

        items.append(formatted)

    return items


@router.get("/{baseline_id}")
def get_baseline(
    baseline_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get details of a specific baseline, including its run history."""
    stmt = (
        select(Baseline)
        .options(
            selectinload(Baseline.run).selectinload(Run.results).selectinload(Result.cases),
        )
        .where(Baseline.id == baseline_id)
    )
    baseline = db.execute(stmt).scalars().first()
    if not baseline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Baseline {baseline_id} not found")
    return format_baseline(baseline, baseline.run)


@router.patch("/{baseline_id}")
@router.put("/{baseline_id}", include_in_schema=False)
def update_baseline(
    baseline_id: int,
    req: UpdateBaselineRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Rename a baseline or update its description while enforcing case-insensitive uniqueness."""
    stmt = (
        select(Baseline)
        .options(
            selectinload(Baseline.run).selectinload(Run.results).selectinload(Result.cases),
        )
        .where(Baseline.id == baseline_id)
    )
    baseline = db.execute(stmt).scalars().first()
    if not baseline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Baseline {baseline_id} not found")

    if "name" in req.model_fields_set and req.name is not None:
        clean_name = req.name.strip()
        if not clean_name:
            raise HTTPException(
                status_code=422,
                detail="Baseline name cannot be empty",
            )
        conflict = (
            db.execute(
                select(Baseline).where(
                    func.lower(Baseline.name) == clean_name.lower(),
                    Baseline.id != baseline_id,
                )
            )
            .scalars()
            .first()
        )
        if conflict is not None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "detail": f"Baseline '{conflict.name}' already exists and points to run #{conflict.run_id}",
                    "run_id": conflict.run_id,
                    "current_run_id": conflict.run_id,
                    "baseline_id": conflict.id,
                    "name": conflict.name,
                },
            )  # type: ignore[return-value]
        baseline.name = clean_name

    if "description" in req.model_fields_set:
        baseline.description = req.description

    baseline.updated_at = utc_now()
    db.commit()
    db.refresh(baseline)
    return format_baseline(baseline, baseline.run)


@router.delete("/{baseline_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_baseline(
    baseline_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Remove a baseline label without deleting the underlying run."""
    baseline = db.get(Baseline, baseline_id)
    if not baseline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Baseline {baseline_id} not found")
    db.delete(baseline)
    db.commit()
