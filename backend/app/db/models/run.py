from datetime import datetime
from typing import TYPE_CHECKING, Any
from sqlalchemy import Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin, UTCDateTime

if TYPE_CHECKING:
    from backend.app.db.models.baseline import Baseline
    from backend.app.db.models.job import Job
    from backend.app.db.models.result import Result
    from backend.app.db.models.target import Target
    from backend.app.db.models.test_definition import TestDefinition


class Run(Base, TimestampMixin):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False, index=True)
    queued_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    # Snapshots ensure that historical run results remain comparable after edits or deletions
    config_snapshot: Mapped[dict[str, Any] | list[Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    target_snapshot: Mapped[dict[str, Any] | list[Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    git_info: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    job: Mapped["Job | None"] = relationship(
        "Job",
        back_populates="runs",
    )
    results: Mapped[list["Result"]] = relationship(
        "Result",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    baselines: Mapped[list["Baseline"]] = relationship(
        "Baseline",
        back_populates="run",
        passive_deletes="all",
    )

    def snapshot_target(self, target: "Target | None") -> None:
        """Capture an immutable snapshot of target data."""
        if target is None:
            self.target_snapshot = {}
        else:
            self.target_snapshot = target.to_snapshot()

    def snapshot_config(self, definitions: list["TestDefinition"]) -> None:
        """Capture an immutable snapshot of test definitions."""
        self.config_snapshot = [d.to_snapshot() for d in definitions]
