import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.session import configure_engine_pragmas


@pytest.fixture(scope="function")
def test_engine():
    """Create a temporary in-memory database engine for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    configure_engine_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine) -> Session:
    """Yield a database session for testing."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def app_and_client(tmp_path):
    from pathlib import Path
    from fastapi.testclient import TestClient
    from backend.app.config import get_settings
    from backend.app.db.session import get_db
    from backend.app.main import create_app
    from backend.app.runner.worker import get_worker

    db_path = tmp_path / "test_api.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    configure_engine_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    app = create_app()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    worker = get_worker()
    worker.session_factory = TestingSessionLocal
    worker.artifacts_dir = tmp_path / "artifacts"

    settings = get_settings()
    settings.DATA_DIR = tmp_path
    settings.DATABASE_URL = db_url

    client = TestClient(app)
    return client, TestingSessionLocal, tmp_path / "artifacts"
