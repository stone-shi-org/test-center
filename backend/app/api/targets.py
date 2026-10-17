"""API routes for managing test targets.

Targets specify the endpoint, kind, and authentication (via environment variable reference only)
for external services being tested or evaluated.
Secrets are never stored in the database or returned by the API.
"""

from datetime import datetime
import re
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.target import Target
from backend.app.db.session import get_db

router = APIRouter(prefix="/targets", tags=["targets"])

_ENV_VAR_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DISALLOWED_SECRET_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "password",
    "token",
    "access_token",
    "auth_token",
    "bearer_token",
    "private_key",
}


def _validate_auth_env_var(value: str | None) -> str | None:
    if value is None:
        return None
    val = value.strip()
    if not val:
        return None
    # Explicitly check for tokens or secrets starting with common credential prefixes
    if val.startswith(("sk-", "ghp_", "xoxb-", "Bearer ", "token-")):
        raise ValueError(
            "API key must be provided by environment variable reference only (e.g. 'OPENAI_API_KEY'), "
            "never a raw secret or token."
        )
    if not _ENV_VAR_REGEX.match(val):
        raise ValueError(
            "auth_env_var must be a valid environment variable reference only (letters, numbers, underscores), "
            "not a raw API key, token, or secret value."
        )
    return val


def _validate_extra(extra_dict: dict[str, Any] | None) -> dict[str, Any]:
    if not extra_dict:
        return {}
    for key, val in extra_dict.items():
        k_lower = key.lower()
        if k_lower in _DISALLOWED_SECRET_KEYS:
            if isinstance(val, str) and len(val.strip()) > 0:
                raise ValueError(
                    f"Direct secrets are forbidden in target extra ('{key}'). "
                    "Use auth_env_var to reference secrets via environment variables."
                )
    return extra_dict


class TargetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Unique target name")
    kind: str = Field(..., min_length=1, max_length=64, description="Target kind (e.g. openai-compatible, live-stt, embedding)")
    base_url: str = Field(..., min_length=1, max_length=1024, description="Base URL of target service")
    auth_env_var: str | None = Field(default=None, description="Environment variable name holding API key (reference only)")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional non-secret target parameters")

    @field_validator("auth_env_var")
    @classmethod
    def check_auth_env_var(cls, v: str | None) -> str | None:
        return _validate_auth_env_var(v)

    @field_validator("extra")
    @classmethod
    def check_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_extra(v)


class TargetCreate(TargetBase):
    pass


class TargetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    kind: str | None = Field(default=None, min_length=1, max_length=64)
    base_url: str | None = Field(default=None, min_length=1, max_length=1024)
    auth_env_var: str | None = Field(default=None)
    extra: dict[str, Any] | None = Field(default=None)

    @field_validator("auth_env_var")
    @classmethod
    def check_auth_env_var(cls, v: str | None) -> str | None:
        return _validate_auth_env_var(v)

    @field_validator("extra")
    @classmethod
    def check_extra(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None:
            return _validate_extra(v)
        return v


class TargetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    kind: str
    base_url: str
    auth_env_var: str | None
    extra: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@router.post("", response_model=TargetResponse, status_code=status.HTTP_201_CREATED)
def create_target(
    req: TargetCreate,
    db: Session = Depends(get_db),
) -> Target:
    """Create a new test target.
    
    API keys must only be referenced by their environment variable name via `auth_env_var`.
    Secrets are never stored or returned by the API.
    """
    # Check uniqueness of name
    existing = db.execute(select(Target).where(Target.name == req.name)).scalars().first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Target with name '{req.name}' already exists",
        )

    target = Target(
        name=req.name,
        kind=req.kind,
        base_url=req.base_url,
        auth_env_var=req.auth_env_var,
        extra=req.extra,
    )
    db.add(target)
    db.commit()
    db.refresh(target)
    return target


@router.get("", response_model=list[TargetResponse])
def list_targets(
    kind: str | None = Query(None, description="Filter targets by kind"),
    limit: int = Query(50, ge=1, le=200, description="Max targets to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> list[Target]:
    """List test targets with optional filtering by kind."""
    stmt = select(Target).order_by(Target.id.asc()).offset(offset).limit(limit)
    if kind:
        stmt = stmt.where(Target.kind == kind)
    return list(db.execute(stmt).scalars().all())


@router.get("/{target_id}", response_model=TargetResponse)
def get_target(
    target_id: int,
    db: Session = Depends(get_db),
) -> Target:
    """Get details of a specific target by ID."""
    target = db.get(Target, target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")
    return target


@router.put("/{target_id}", response_model=TargetResponse)
@router.patch("/{target_id}", response_model=TargetResponse)
def update_target(
    target_id: int,
    req: TargetUpdate,
    db: Session = Depends(get_db),
) -> Target:
    """Update target fields."""
    target = db.get(Target, target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")

    update_data = req.model_dump(exclude_unset=True)

    # Check name collision if name is being changed
    if "name" in update_data and update_data["name"] != target.name:
        existing = db.execute(select(Target).where(Target.name == update_data["name"])).scalars().first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Target with name '{update_data['name']}' already exists",
            )

    for field, val in update_data.items():
        setattr(target, field, val)

    db.commit()
    db.refresh(target)
    return target


@router.delete("/{target_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_target(
    target_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a target. References from test definitions are automatically set to NULL."""
    target = db.get(Target, target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")

    db.delete(target)
    db.commit()
