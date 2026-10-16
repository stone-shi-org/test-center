from pathlib import Path
from alembic.config import Config
from alembic import command


def apply_migrations(database_url: str | None = None) -> None:
    """Apply Alembic migrations up to head programmatically."""
    # backend directory
    backend_dir = Path(__file__).resolve().parent.parent.parent
    alembic_ini = backend_dir / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    if database_url:
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")
