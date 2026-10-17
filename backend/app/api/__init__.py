"""API routers for Test Center."""

from backend.app.api.health import router as health_router
from backend.app.api.testers import router as testers_router
from backend.app.api.targets import router as targets_router
from backend.app.api.test_definitions import (
    router as test_definitions_router,
    definitions_alias_router,
)
from backend.app.api.runs import (
    router as runs_router,
    trends_alias_router,
    retention_alias_router,
)
from backend.app.api.results import router as results_router
from backend.app.api.compare import router as compare_router
from backend.app.api.jobs import router as jobs_router
from backend.app.api.bamboo import (
    router as bamboo_router,
    ci_alias_router,
)

__all__ = [
    "health_router",
    "testers_router",
    "targets_router",
    "test_definitions_router",
    "definitions_alias_router",
    "jobs_router",
    "runs_router",
    "trends_alias_router",
    "retention_alias_router",
    "results_router",
    "compare_router",
    "bamboo_router",
    "ci_alias_router",
]


