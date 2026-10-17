from pathlib import Path
from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine, inspect


def test_alembic_migrations_lifecycle(tmp_path):
    """Test that migrations apply cleanly on an empty DB, can be downgraded, and re-applied."""
    db_file = tmp_path / "migration_test.db"
    db_url = f"sqlite:///{db_file}"

    # Locate backend root and alembic.ini
    backend_dir = Path(__file__).resolve().parent.parent
    alembic_ini_path = backend_dir / "alembic.ini"

    config = Config(str(alembic_ini_path))
    config.set_main_option("sqlalchemy.url", db_url)
    config.set_main_option("script_location", str(backend_dir / "alembic"))

    # 1. Upgrade from empty DB to head
    command.upgrade(config, "head")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    expected_tables = {
        "alembic_version",
        "targets",
        "test_definitions",
        "jobs",
        "runs",
        "results",
        "case_results",
        "artifacts",
    }
    assert expected_tables.issubset(tables), f"Missing tables in migration: {expected_tables - tables}"

    # Verify column existence on jobs, runs (snapshots) and results (metrics)
    job_columns = {col["name"] for col in inspector.get_columns("jobs")}
    assert "run_mode" in job_columns
    assert "overrides" in job_columns
    assert "description" in job_columns
    assert "missed_run_policy" in job_columns
    assert "next_run_at" in job_columns
    assert "last_scheduled_at" in job_columns

    run_columns = {col["name"] for col in inspector.get_columns("runs")}
    assert "config_snapshot" in run_columns
    assert "target_snapshot" in run_columns
    assert "git_info" in run_columns

    result_columns = {col["name"] for col in inspector.get_columns("results")}
    assert "metrics" in result_columns
    assert "tester" in result_columns

    engine.dispose()

    # 2. Downgrade back to base
    command.downgrade(config, "base")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables_after_downgrade = set(inspector.get_table_names())
    # All app tables should be dropped
    assert "targets" not in tables_after_downgrade
    assert "runs" not in tables_after_downgrade
    assert "results" not in tables_after_downgrade
    assert "case_results" not in tables_after_downgrade
    assert "artifacts" not in tables_after_downgrade
    engine.dispose()

    # 3. Upgrade to head again
    command.upgrade(config, "head")
    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables_reapplied = set(inspector.get_table_names())
    assert expected_tables.issubset(tables_reapplied)
    engine.dispose()


def test_apply_migrations_helper(tmp_path):
    """Test the programmatic apply_migrations helper."""
    from backend.app.db.migrations import apply_migrations

    db_file = tmp_path / "helper_test.db"
    db_url = f"sqlite:///{db_file}"

    apply_migrations(db_url)

    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert "targets" in tables
    assert "runs" in tables
    assert "results" in tables
    engine.dispose()
