from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api import (
    health_router,
    testers_router,
    targets_router,
    test_definitions_router,
    definitions_alias_router,
    jobs_router,
    runs_router,
    trends_alias_router,
    retention_alias_router,
    results_router,
    compare_router,
    baselines_router,
    bamboo_router,
    ci_alias_router,
)
from backend.app.config import get_settings
from backend.app.db import apply_migrations
from backend.app.db.session import SessionLocal
from backend.app.runner.recovery import recover_stale_runs
from backend.app.runner.scheduler import get_scheduler
from backend.app.runner.worker import get_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Ensure data and artifact directories exist
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Bring the database schema up to date (creates tables on a fresh volume).
    # Failing here is fatal: running against a missing/stale schema only yields 500s.
    apply_migrations()

    # Recover any runs left in 'running' state from previous process crash/restart
    try:
        with SessionLocal() as db:
            recover_stale_runs(db, settings.ARTIFACTS_DIR)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to recover stale runs on startup: {e}")

    # Start runner worker and cron scheduler
    worker = get_worker()
    worker.start()

    scheduler = get_scheduler()
    scheduler.start()

    yield

    # Cleanly stop scheduler and worker on shutdown
    await scheduler.stop()
    await worker.stop()


OPENAPI_TAGS = [
    {"name": "health", "description": "System health and status endpoints."},
    {"name": "testers", "description": "Tester plugins discovery and configuration schemas."},
    {"name": "targets", "description": "Target service endpoints and authentication environment variable references."},
    {"name": "test-definitions", "description": "Test definitions, suite selections, and manual run triggers."},
    {"name": "jobs", "description": "Jobs bundling ordered test definitions, cron scheduling, enable/disable, missed-run policies, and manual triggers."},
    {"name": "runs", "description": "Test execution queueing, status, cancellation, and live streaming."},
    {"name": "results", "description": "Execution results, per-case details, metrics, and artifact downloads."},
    {"name": "compare", "description": "Side-by-side comparison of 2+ runs or results, aligned metrics, deltas, and per-case diffs."},
    {"name": "baselines", "description": "Named run baselines protected from retention and available in Run History and Compare."},
    {"name": "bamboo", "description": "Optional Bamboo / CI integration: token-protected job triggering, run status polling, status reporting, and JUnit XML export."},
]


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="Test Center REST API: targets, testers, test definitions, jobs, runs, results, compare, and Bamboo CI integration.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(health_router, prefix="/api")
    app.include_router(testers_router, prefix="/api")
    app.include_router(targets_router, prefix="/api")
    app.include_router(test_definitions_router, prefix="/api")
    # Also support /api/definitions as an alias for test definitions
    app.include_router(definitions_alias_router, prefix="/api", include_in_schema=False)
    app.include_router(jobs_router, prefix="/api")
    app.include_router(runs_router, prefix="/api")
    app.include_router(trends_alias_router, prefix="/api", include_in_schema=False)
    app.include_router(retention_alias_router, prefix="/api", include_in_schema=False)
    app.include_router(results_router, prefix="/api")
    app.include_router(compare_router, prefix="/api")
    app.include_router(baselines_router, prefix="/api")
    app.include_router(bamboo_router, prefix="/api")
    app.include_router(ci_alias_router, prefix="/api", include_in_schema=False)

    # Static assets and SPA handling for frontend
    static_dir = Path(settings.STATIC_DIR)
    if static_dir.exists() and (static_dir / "index.html").exists():
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            # Don't hijack undefined /api routes with the SPA index
            if full_path.startswith("api/") or full_path == "api":
                raise HTTPException(status_code=404, detail="Not Found")

            file_path = static_dir / full_path
            if full_path and file_path.is_file():
                return FileResponse(file_path)

            return FileResponse(static_dir / "index.html")
    else:
        @app.get("/")
        async def root():
            return {
                "app": settings.APP_NAME,
                "version": settings.VERSION,
                "message": "Backend API is running. Frontend static assets not built or located.",
                "docs_url": "/docs",
                "health_url": "/api/health",
            }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
