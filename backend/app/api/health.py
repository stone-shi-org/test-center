from fastapi import APIRouter
from backend.app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "data_dir": str(settings.DATA_DIR),
        "data_dir_exists": settings.DATA_DIR.exists(),
        "test_data_dir": str(settings.TEST_DATA_DIR),
        "test_data_dir_exists": settings.TEST_DATA_DIR.exists(),
    }
