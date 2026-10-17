import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    APP_NAME: str = "Test Center"
    VERSION: str = "0.1.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    TEST_DATA_DIR: Path = Path(os.getenv("TEST_DATA_DIR", str(BASE_DIR / "test-data")))
    STATIC_DIR: Path = Path(os.getenv("STATIC_DIR", str(BASE_DIR / "frontend" / "dist")))
    DATABASE_URL: str = ""
    RETENTION_ENABLED: bool = os.getenv("RETENTION_ENABLED", "false").lower() in ("1", "true", "yes")
    RETENTION_MAX_AGE_DAYS: int | None = (
        int(os.getenv("RETENTION_MAX_AGE_DAYS", "0"))
        if os.getenv("RETENTION_MAX_AGE_DAYS", "0").isdigit() and int(os.getenv("RETENTION_MAX_AGE_DAYS", "0")) > 0
        else None
    )
    RETENTION_MAX_RUNS: int | None = (
        int(os.getenv("RETENTION_MAX_RUNS", "0"))
        if os.getenv("RETENTION_MAX_RUNS", "0").isdigit() and int(os.getenv("RETENTION_MAX_RUNS", "0")) > 0
        else None
    )
    RETENTION_ARTIFACTS_ONLY: bool = os.getenv("RETENTION_ARTIFACTS_ONLY", "false").lower() in ("1", "true", "yes")
    RETENTION_KEEP_FAILED: bool = os.getenv("RETENTION_KEEP_FAILED", "false").lower() in ("1", "true", "yes")
    WEBHOOK_TIMEOUT_SECONDS: float = float(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "10.0"))
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD_ENV_VAR: str = os.getenv("SMTP_PASSWORD_ENV_VAR", "SMTP_PASSWORD")
    SMTP_FROM: str = os.getenv("SMTP_FROM", "test-center@localhost")
    SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")
    SMTP_USE_SSL: bool = os.getenv("SMTP_USE_SSL", "false").lower() in ("1", "true", "yes")
    BAMBOO_TOKEN_ENV_VAR: str = os.getenv("BAMBOO_TOKEN_ENV_VAR", "BAMBOO_API_TOKEN")
    BAMBOO_BASE_URL: str = os.getenv("BAMBOO_BASE_URL", os.getenv("BAMBOO_URL", ""))
    BAMBOO_USER: str = os.getenv("BAMBOO_USER", os.getenv("BAMBOO_USERNAME", ""))
    BAMBOO_CREDS_ENV_VAR: str = os.getenv("BAMBOO_CREDS_ENV_VAR", "BAMBOO_API_TOKEN")

    def model_post_init(self, __context):
        if not self.DATABASE_URL:
            env_db = os.getenv("DATABASE_URL")
            if env_db:
                self.DATABASE_URL = env_db
            else:
                self.DATABASE_URL = f"sqlite:///{self.DATA_DIR / 'test_center.db'}"

    @property
    def ARTIFACTS_DIR(self) -> Path:
        env_artifacts = os.getenv("ARTIFACTS_DIR")
        if env_artifacts:
            return Path(env_artifacts)
        return self.DATA_DIR / "artifacts"



@lru_cache()
def get_settings() -> Settings:
    return Settings()
