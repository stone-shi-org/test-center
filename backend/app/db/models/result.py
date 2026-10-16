from typing import TYPE_CHECKING, Any, Literal
from sqlalchemy import Integer, String, Text, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin
from backend.app.db.metrics import format_metric

if TYPE_CHECKING:
    from backend.app.db.models.run import Run
    from backend.app.db.models.test_definition import TestDefinition
    from backend.app.db.models.case_result import CaseResult
    from backend.app.db.models.artifact import Artifact


class Result(Base, TimestampMixin):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_definition_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("test_definitions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    tester: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # pass | fail | error

    # Metrics stored as JSON with unit and better=higher|lower
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped["Run"] = relationship(
        "Run",
        back_populates="results",
    )
    test_definition: Mapped["TestDefinition | None"] = relationship(
        "TestDefinition",
        back_populates="results",
    )
    cases: Mapped[list["CaseResult"]] = relationship(
        "CaseResult",
        back_populates="result",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        "Artifact",
        back_populates="result",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def add_metric(
        self,
        name: str,
        value: float | int,
        unit: str | None = None,
        better: Literal["higher", "lower"] | str | None = None,
    ) -> None:
        """Add or update a metric entry with optional unit and better direction."""
        metric_dict = dict(self.metrics or {})
        metric_dict[name] = format_metric(value=value, unit=unit, better=better)
        self.metrics = metric_dict
