"""API routes for managing test definitions.

A TestDefinition specifies a test configuration, selected suites, and optionally a target.
Manual runs can be directly triggered from a TestDefinition.
"""

from datetime import datetime
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.api.targets import TargetResponse
from backend.app.db.base import utc_now
from backend.app.db.models.target import Target
from backend.app.db.models.test_definition import TestDefinition
from backend.app.db.models.run import Run
from backend.app.db.session import get_db
from backend.app.runner.worker import get_worker
from backend.app.testers.registry import tester_registry, TesterNotFoundError
from backend.app.testers.base import TesterConfigError

router = APIRouter(prefix="/test-definitions", tags=["test-definitions"])
definitions_alias_router = APIRouter(prefix="/definitions", tags=["test-definitions"])


class TestDefinitionCreate(BaseModel):
    __test__ = False
    name: str = Field(..., min_length=1, max_length=255, description="Descriptive name of test definition")
    tester: str = Field(..., min_length=1, max_length=64, description="Registered tester plugin name")
    suites: list[str] = Field(default_factory=list, description="Sub-suites to execute")
    config: dict[str, Any] = Field(default_factory=dict, description="Tester configuration dictionary")
    target_id: int | None = Field(default=None, description="Optional target ID")


class TestDefinitionUpdate(BaseModel):
    __test__ = False
    name: str | None = Field(default=None, min_length=1, max_length=255)
    tester: str | None = Field(default=None, min_length=1, max_length=64)
    suites: list[str] | None = None
    config: dict[str, Any] | None = None
    target_id: int | None = None


class TestDefinitionResponse(BaseModel):
    __test__ = False
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    tester: str
    suites: list[str]
    config: dict[str, Any]
    target_id: int | None
    created_at: datetime
    updated_at: datetime
    target: TargetResponse | None = None


class TriggerDefinitionRunRequest(BaseModel):
    __test__ = False
    suites: list[str] | None = Field(default=None, description="Optional suite override")
    config: dict[str, Any] | None = Field(default=None, description="Optional config override / additions")
    target_id: int | None = Field(default=None, description="Optional target ID override")
    timeout: float | None = Field(default=None, description="Optional run timeout in seconds")


def _validate_definition_config(tester_name: str, config: dict[str, Any], suites: list[str] | None) -> dict[str, Any]:
    """Validate tester and config if tester plugin is registered."""
    try:
        plugin = tester_registry.get(tester_name)
    except TesterNotFoundError:
        # If tester plugin is not registered, return config as is (or fail if strict)
        return config

    if suites:
        allowed_suites = set(plugin.suites) | set(getattr(plugin, "suite_aliases", set()))
        unknown_suites = [s for s in suites if s not in allowed_suites]
        if unknown_suites and plugin.suites:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown suites {unknown_suites} for tester '{tester_name}'. Available: {plugin.suites}",
            )

    try:
        validated_config = plugin.validate_and_apply_config(config)
        return validated_config
    except TesterConfigError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )


@router.post("", response_model=TestDefinitionResponse, status_code=status.HTTP_201_CREATED)
@definitions_alias_router.post("", response_model=TestDefinitionResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
def create_test_definition(
    req: TestDefinitionCreate,
    db: Session = Depends(get_db),
) -> TestDefinition:
    """Create a new test definition."""
    if req.target_id is not None:
        target = db.get(Target, req.target_id)
        if not target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {req.target_id} not found")

    if isinstance(req.config, dict) and req.config.get("judge_target_id") is not None:
        judge_tid = req.config["judge_target_id"]
        if isinstance(judge_tid, int) and not isinstance(judge_tid, bool):
            judge_target = db.get(Target, judge_tid)
            if not judge_target:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Judge target {judge_tid} not found")

    validated_config = _validate_definition_config(req.tester, req.config, req.suites)

    definition = TestDefinition(
        name=req.name,
        tester=req.tester,
        suites=req.suites,
        config=validated_config,
        target_id=req.target_id,
    )
    db.add(definition)
    db.commit()
    db.refresh(definition)
    return definition


@router.get("", response_model=list[TestDefinitionResponse])
@definitions_alias_router.get("", response_model=list[TestDefinitionResponse], include_in_schema=False)
def list_test_definitions(
    tester: str | None = Query(None, description="Filter by tester plugin name"),
    target_id: int | None = Query(None, description="Filter by target ID"),
    limit: int = Query(50, ge=1, le=200, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> list[TestDefinition]:
    """List test definitions with optional filtering."""
    stmt = (
        select(TestDefinition)
        .options(selectinload(TestDefinition.target))
        .order_by(TestDefinition.id.asc())
        .offset(offset)
        .limit(limit)
    )
    if tester:
        stmt = stmt.where(TestDefinition.tester == tester)
    if target_id is not None:
        stmt = stmt.where(TestDefinition.target_id == target_id)

    return list(db.execute(stmt).scalars().all())


@router.get("/{definition_id}", response_model=TestDefinitionResponse)
@definitions_alias_router.get("/{definition_id}", response_model=TestDefinitionResponse, include_in_schema=False)
def get_test_definition(
    definition_id: int,
    db: Session = Depends(get_db),
) -> TestDefinition:
    """Get details of a specific test definition."""
    stmt = select(TestDefinition).options(selectinload(TestDefinition.target)).where(TestDefinition.id == definition_id)
    definition = db.execute(stmt).scalars().first()
    if not definition:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TestDefinition {definition_id} not found")
    return definition


@router.put("/{definition_id}", response_model=TestDefinitionResponse)
@router.patch("/{definition_id}", response_model=TestDefinitionResponse)
@definitions_alias_router.put("/{definition_id}", response_model=TestDefinitionResponse, include_in_schema=False)
@definitions_alias_router.patch("/{definition_id}", response_model=TestDefinitionResponse, include_in_schema=False)
def update_test_definition(
    definition_id: int,
    req: TestDefinitionUpdate,
    db: Session = Depends(get_db),
) -> TestDefinition:
    """Update test definition fields."""
    stmt = select(TestDefinition).options(selectinload(TestDefinition.target)).where(TestDefinition.id == definition_id)
    definition = db.execute(stmt).scalars().first()
    if not definition:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TestDefinition {definition_id} not found")

    update_data = req.model_dump(exclude_unset=True)

    if "target_id" in update_data and update_data["target_id"] is not None:
        target = db.get(Target, update_data["target_id"])
        if not target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {update_data['target_id']} not found")

    tester = update_data.get("tester", definition.tester)
    config = update_data.get("config", definition.config)
    suites = update_data.get("suites", definition.suites)

    if isinstance(config, dict) and config.get("judge_target_id") is not None:
        judge_tid = config["judge_target_id"]
        if isinstance(judge_tid, int) and not isinstance(judge_tid, bool):
            judge_target = db.get(Target, judge_tid)
            if not judge_target:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Judge target {judge_tid} not found")

    if "config" in update_data or "tester" in update_data or "suites" in update_data:
        validated_config = _validate_definition_config(tester, config, suites)
        update_data["config"] = validated_config

    for field, val in update_data.items():
        setattr(definition, field, val)

    db.commit()
    db.refresh(definition)
    return definition


@router.delete("/{definition_id}", status_code=status.HTTP_204_NO_CONTENT)
@definitions_alias_router.delete("/{definition_id}", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=False)
def delete_test_definition(
    definition_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a test definition."""
    definition = db.get(TestDefinition, definition_id)
    if not definition:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TestDefinition {definition_id} not found")

    db.delete(definition)
    db.commit()


@router.post("/{definition_id}/run", status_code=status.HTTP_201_CREATED)
@definitions_alias_router.post("/{definition_id}/run", status_code=status.HTTP_201_CREATED, include_in_schema=False)
def trigger_definition_run(
    definition_id: int,
    req: TriggerDefinitionRunRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Trigger a manual test run from an existing test definition.
    
    Optional overrides for suites, config, target_id, and timeout may be provided.
    """
    definition = db.get(TestDefinition, definition_id)
    if not definition:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TestDefinition {definition_id} not found")

    req_data = req or TriggerDefinitionRunRequest()

    target_id = req_data.target_id if req_data.target_id is not None else definition.target_id
    target_snapshot: dict[str, Any] = {}
    if target_id is not None:
        target = db.get(Target, target_id)
        if not target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")
        target_snapshot = target.to_snapshot()

    suites = req_data.suites if req_data.suites is not None else definition.suites
    config = dict(definition.config)
    if req_data.config:
        config.update(req_data.config)

    judge_target_id = config.get("judge_target_id")
    if judge_target_id is not None and not isinstance(target_snapshot.get("judge_target"), dict):
        judge_target = db.get(Target, int(judge_target_id))
        if not judge_target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Judge target {judge_target_id} not found")
        target_snapshot = dict(target_snapshot)
        target_snapshot["judge_target"] = judge_target.to_snapshot()

    config_snapshot: dict[str, Any] = {
        "tester": definition.tester,
        "suites": suites,
        "config": config,
        "test_definition_id": definition.id,
    }
    if req_data.timeout:
        config_snapshot["timeout"] = req_data.timeout

    run = Run(
        status="queued",
        queued_at=utc_now(),
        config_snapshot=config_snapshot,
        target_snapshot=target_snapshot,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Wake up runner worker
    get_worker().trigger()

    # Format response using standard run detail formatter
    from backend.app.api.runs import _format_run_detail
    return _format_run_detail(run)
