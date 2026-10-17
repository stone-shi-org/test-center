"""API routes for inspecting and querying test results, test cases, and artifacts."""

from datetime import datetime
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.db.models.result import Result
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.artifact import Artifact
from backend.app.db.session import get_db

router = APIRouter(prefix="/results", tags=["results"])


class CaseResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    result_id: int
    suite: str
    case_id: str
    status: str
    metrics: dict[str, Any]
    detail: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    result_id: int
    name: str
    path: str
    mime: str
    size_bytes: int | None
    created_at: datetime


class ResultSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    test_definition_id: int | None
    tester: str
    status: str
    metrics: dict[str, Any]
    summary: str | None
    case_count: int
    artifact_count: int
    created_at: datetime
    updated_at: datetime


class ResultDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    test_definition_id: int | None
    tester: str
    status: str
    metrics: dict[str, Any]
    summary: str | None
    created_at: datetime
    updated_at: datetime
    cases: list[CaseResultResponse]
    artifacts: list[ArtifactResponse]


@router.get("", response_model=list[ResultSummaryResponse])
def list_results(
    run_id: int | None = Query(None, description="Filter by Run ID"),
    test_definition_id: int | None = Query(None, description="Filter by TestDefinition ID"),
    tester: str | None = Query(None, description="Filter by tester name"),
    status: str | None = Query(None, description="Filter by result status (pass, fail, error, cancelled)"),
    limit: int = Query(50, ge=1, le=200, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List test execution results with optional filters."""
    stmt = (
        select(Result)
        .options(selectinload(Result.cases), selectinload(Result.artifacts))
        .order_by(Result.id.desc())
        .offset(offset)
        .limit(limit)
    )
    if run_id is not None:
        stmt = stmt.where(Result.run_id == run_id)
    if test_definition_id is not None:
        stmt = stmt.where(Result.test_definition_id == test_definition_id)
    if tester:
        stmt = stmt.where(Result.tester == tester)
    if status:
        stmt = stmt.where(Result.status == status)

    results = db.execute(stmt).scalars().all()
    output = []
    for r in results:
        output.append({
            "id": r.id,
            "run_id": r.run_id,
            "test_definition_id": r.test_definition_id,
            "tester": r.tester,
            "status": r.status,
            "metrics": r.metrics,
            "summary": r.summary,
            "case_count": len(r.cases),
            "artifact_count": len(r.artifacts),
            "created_at": r.created_at,
            "updated_at": r.updated_at,
        })
    return output


@router.get("/{result_id}", response_model=ResultDetailResponse)
def get_result(
    result_id: int,
    db: Session = Depends(get_db),
) -> Result:
    """Get complete details of a specific result including all test cases and artifacts."""
    stmt = (
        select(Result)
        .options(selectinload(Result.cases), selectinload(Result.artifacts))
        .where(Result.id == result_id)
    )
    res = db.execute(stmt).scalars().first()
    if not res:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result {result_id} not found")
    return res


@router.get("/{result_id}/cases", response_model=list[CaseResultResponse])
def list_result_cases(
    result_id: int,
    suite: str | None = Query(None, description="Filter cases by suite name"),
    status: str | None = Query(None, description="Filter cases by status"),
    db: Session = Depends(get_db),
) -> list[CaseResult]:
    """List test case outcomes for a specific result."""
    # Verify result exists
    if not db.get(Result, result_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result {result_id} not found")

    stmt = select(CaseResult).where(CaseResult.result_id == result_id).order_by(CaseResult.id.asc())
    if suite:
        stmt = stmt.where(CaseResult.suite == suite)
    if status:
        stmt = stmt.where(CaseResult.status == status)

    return list(db.execute(stmt).scalars().all())


@router.get("/{result_id}/artifacts", response_model=list[ArtifactResponse])
def list_result_artifacts(
    result_id: int,
    db: Session = Depends(get_db),
) -> list[Artifact]:
    """List artifacts associated with a result."""
    if not db.get(Result, result_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result {result_id} not found")

    stmt = select(Artifact).where(Artifact.result_id == result_id).order_by(Artifact.id.asc())
    return list(db.execute(stmt).scalars().all())


@router.get("/{result_id}/artifacts/{artifact_id}/download")
def download_result_artifact(
    result_id: int,
    artifact_id: int,
    db: Session = Depends(get_db),
):
    """Download a specific artifact from a result."""
    artifact = db.get(Artifact, artifact_id)
    if not artifact or artifact.result_id != result_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    file_path = Path(artifact.path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file does not exist on disk")

    return FileResponse(
        path=str(file_path),
        filename=artifact.name,
        media_type=artifact.mime,
    )
