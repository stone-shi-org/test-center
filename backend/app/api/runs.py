"""API routes for managing and executing test runs, streaming logs, and retrieving artifacts."""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.config import get_settings
from backend.app.db.base import utc_now
from backend.app.db.models.run import Run
from backend.app.db.models.target import Target
from backend.app.db.models.result import Result
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.artifact import Artifact
from backend.app.db.session import get_db
from backend.app.runner.events import get_broadcaster
from backend.app.runner.worker import get_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


class CreateRunRequest(BaseModel):
    test_definition_id: int | None = Field(default=None, description="Optional TestDefinition ID to run")
    tester: str | None = Field(default=None, description="Tester plugin name (defaults to 'dummy' or definition's tester)")
    suites: list[str] | None = Field(default=None, description="Sub-suites to run")
    config: dict[str, Any] = Field(default_factory=dict, description="Tester configuration dictionary")
    target_id: int | None = Field(default=None, description="Target ID")
    target: dict[str, Any] | None = Field(default=None, description="Inline target configuration dictionary")
    job_id: int | None = Field(default=None, description="Job ID if triggered by job")
    timeout: float | None = Field(default=None, description="Execution timeout in seconds")


def _format_run_detail(run: Run) -> dict[str, Any]:
    """Helper to serialize a Run with its nested results, cases, and artifacts."""
    results_list = []
    for r in run.results:
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
            {
                "id": a.id,
                "name": a.name,
                "path": a.path,
                "mime": a.mime,
                "size_bytes": a.size_bytes,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in r.artifacts
        ]
        results_list.append({
            "id": r.id,
            "tester": r.tester,
            "status": r.status,
            "metrics": r.metrics,
            "summary": r.summary,
            "cases": cases_list,
            "artifacts": artifacts_list,
        })

    return {
        "id": run.id,
        "job_id": run.job_id,
        "status": run.status,
        "queued_at": run.queued_at.isoformat() if run.queued_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "config_snapshot": run.config_snapshot,
        "target_snapshot": run.target_snapshot,
        "git_info": run.git_info,
        "results": results_list,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_run(
    req: CreateRunRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Queue a new test run manually or from a test definition."""
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
    status: str | None = Query(None, description="Filter by status"),
    job_id: int | None = Query(None, description="Filter by job ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List runs ordered by creation time descending."""
    stmt = select(Run).order_by(Run.id.desc()).offset(offset).limit(limit)
    if status:
        stmt = stmt.where(Run.status == status)
    if job_id:
        stmt = stmt.where(Run.job_id == job_id)

    runs = db.execute(stmt).scalars().all()
    return [_format_run_detail(r) for r in runs]


@router.get("/{run_id}")
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get full details of a specific run."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _format_run_detail(run)


@router.get("/{run_id}/results")
def get_run_results(
    run_id: int,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get all results associated with a specific run."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    detail = _format_run_detail(run)
    return detail["results"]


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


@router.get("/{run_id}/artifacts/{artifact_id}/download")
def download_artifact(
    run_id: int,
    artifact_id: int,
    db: Session = Depends(get_db),
):
    """Download an artifact file associated with a run."""
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    file_path = Path(artifact.path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file does not exist on disk")

    return FileResponse(
        path=str(file_path),
        filename=artifact.name,
        media_type=artifact.mime,
    )
