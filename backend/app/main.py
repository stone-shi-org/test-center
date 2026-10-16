from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.health import router as health_router
from backend.app.api.testers import router as testers_router
from backend.app.api.runs import router as runs_router
from backend.app.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.runner.recovery import recover_stale_runs
from backend.app.runner.worker import get_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Ensure data and artifact directories exist
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Recover any runs left in 'running' state from previous process crash/restart
    try:
        with SessionLocal() as db:
            recover_stale_runs(db, settings.ARTIFACTS_DIR)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to recover stale runs on startup: {e}")

    # Start runner worker
    worker = get_worker()
    worker.start()

    yield

    # Cleanly stop worker on shutdown
    await worker.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
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
    app.include_router(runs_router, prefix="/api")

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
