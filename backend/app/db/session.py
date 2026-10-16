from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from backend.app.config import get_settings


def configure_engine_pragmas(target_engine: Engine) -> None:
    """Configure SQLite pragmas (WAL mode and foreign keys)."""
    @event.listens_for(target_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        if "sqlite" in type(dbapi_connection).__module__:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
            except Exception:
                pass
            cursor.close()


def create_db_engine(database_url: str | None = None, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine configured for the database URL."""
    if database_url is None:
        settings = get_settings()
        database_url = settings.DATABASE_URL

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Ensure parent directory exists for file-based SQLite databases
        if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
            db_path = database_url.replace("sqlite:///", "")
            if db_path:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    new_engine = create_engine(
        database_url,
        connect_args=connect_args,
        echo=echo,
        future=True,
    )
    configure_engine_pragmas(new_engine)
    return new_engine


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
