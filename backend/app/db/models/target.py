from typing import TYPE_CHECKING, Any
from sqlalchemy import Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.db.models.test_definition import TestDefinition


class Target(Base, TimestampMixin):
    __tablename__ = "targets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    base_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    auth_env_var: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    test_definitions: Mapped[list["TestDefinition"]] = relationship(
        "TestDefinition",
        back_populates="target",
    )

    def to_snapshot(self) -> dict[str, Any]:
        """Produce an immutable snapshot of target configuration for runs."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "auth_env_var": self.auth_env_var,
            "extra": self.extra,
        }
