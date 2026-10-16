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

    def model_post_init(self, __context):
        if not self.DATABASE_URL:
            env_db = os.getenv("DATABASE_URL")
            if env_db:
                self.DATABASE_URL = env_db
            else:
                self.DATABASE_URL = f"sqlite:///{self.DATA_DIR / 'test_center.db'}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
