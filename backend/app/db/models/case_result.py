from typing import TYPE_CHECKING, Any, Literal
from sqlalchemy import Integer, String, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin
from backend.app.db.metrics import format_metric

if TYPE_CHECKING:
    from backend.app.db.models.result import Result


class CaseResult(Base, TimestampMixin):
    __tablename__ = "case_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    result_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("results.id", ondelete="CASCADE"), nullable=False, index=True
    )
    suite: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # pass | fail | error

    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    result: Mapped["Result"] = relationship(
        "Result",
        back_populates="cases",
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
