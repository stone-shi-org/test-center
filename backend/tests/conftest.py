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
    from backend.app.runner.scheduler import get_scheduler
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

    scheduler = get_scheduler()
    scheduler.session_factory = TestingSessionLocal

    settings = get_settings()
    settings.DATA_DIR = tmp_path
    settings.DATABASE_URL = db_url

    client = TestClient(app)
    return client, TestingSessionLocal, tmp_path / "artifacts"


@pytest.fixture(autouse=True)
def _hermetic_live_stt_speech_fixtures(request, monkeypatch):
    """Keep live-stt tests independent of a NOTSOFAR-1 dataset on the machine running them.

    With speech_fixture/diarization_fixture='auto', the tester looks for the dataset in default
    locations (TEST_DATA_DIR, BASE_DIR/test-data) and, if found, switches to real speech, which the
    mock live-stt servers in these tests do not transcribe. Default-location lookups therefore return
    None in tests; lookups with an explicit dataset_dir (hermetic fake datasets) still work.
    Opt out with @pytest.mark.real_dataset.
    """
    if request.node.get_closest_marker("real_dataset") is not None:
        return
    try:
        from backend.app.testers.live_stt import tester as live_stt_tester
    except Exception:
        return

    real_speech = live_stt_tester.build_notsofar_speech_fixture
    real_dialog = live_stt_tester.build_notsofar_dialog_fixture

    def speech(role="short_speech", *, dataset_dir=None, **kwargs):
        return real_speech(role, dataset_dir=dataset_dir, **kwargs) if dataset_dir else None

    def dialog(*, dataset_dir=None, **kwargs):
        return real_dialog(dataset_dir=dataset_dir, **kwargs) if dataset_dir else None

    monkeypatch.setattr(live_stt_tester, "build_notsofar_speech_fixture", speech)
    monkeypatch.setattr(live_stt_tester, "build_notsofar_dialog_fixture", dialog)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_dataset: allow live-stt tests to use a NOTSOFAR-1 dataset found in default locations"
    )
