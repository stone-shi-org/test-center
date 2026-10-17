from datetime import datetime
from typing import TYPE_CHECKING, Any
from sqlalchemy import Integer, String, Boolean, JSON, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.db.models.run import Run


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    schedule: Mapped[str] = mapped_column(String(128), default="manual", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    run_mode: Mapped[str] = mapped_column(
        String(32), default="sequential", server_default="sequential", nullable=False
    )
    missed_run_policy: Mapped[str] = mapped_column(
        String(32), default="skip", server_default="skip", nullable=False
    )
    next_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    last_scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    definition_ids: Mapped[list[int]] = mapped_column(JSON, default=list, nullable=False)
    overrides: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, server_default="{}", nullable=False
    )

    runs: Mapped[list["Run"]] = relationship(
        "Run",
        back_populates="job",
    )

    def to_snapshot(self) -> dict[str, Any]:
        """Produce an immutable snapshot of job configuration."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "schedule": self.schedule,
            "enabled": self.enabled,
            "run_mode": self.run_mode or "sequential",
            "missed_run_policy": self.missed_run_policy or "skip",
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "last_scheduled_at": (
                self.last_scheduled_at.isoformat() if self.last_scheduled_at else None
            ),
            "definition_ids": list(self.definition_ids or []),
            "overrides": dict(self.overrides or {}),
        }


