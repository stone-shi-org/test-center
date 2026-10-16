"""API routes for tester plugins."""

from typing import Any
from fastapi import APIRouter, HTTPException
from backend.app.testers.registry import tester_registry, TesterNotFoundError

router = APIRouter(prefix="/testers", tags=["testers"])


@router.get("")
def list_available_testers() -> list[dict[str, Any]]:
    """List all registered tester plugins with metadata and configuration schemas."""
    return tester_registry.get_info_list()


@router.get("/{name}")
def get_tester_details(name: str) -> dict[str, Any]:
    """Get metadata, suites, and configuration schema for a specific tester plugin."""
    try:
        plugin = tester_registry.get(name)
        return {
            "name": plugin.name,
            "description": plugin.description,
            "suites": plugin.suites,
            "config_schema": plugin.config_schema,
        }
    except TesterNotFoundError:
        raise HTTPException(status_code=404, detail=f"Tester '{name}' not found")
