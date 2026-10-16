from typing import TYPE_CHECKING, Any
from sqlalchemy import Integer, String, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.db.models.target import Target
    from backend.app.db.models.result import Result


class TestDefinition(Base, TimestampMixin):
    __tablename__ = "test_definitions"
    __test__ = False

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    tester: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    suites: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    target_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("targets.id", ondelete="SET NULL"), nullable=True, index=True
    )

    target: Mapped["Target | None"] = relationship(
        "Target",
        back_populates="test_definitions",
    )
    results: Mapped[list["Result"]] = relationship(
        "Result",
        back_populates="test_definition",
    )

    def to_snapshot(self) -> dict[str, Any]:
        """Produce an immutable snapshot of test definition configuration for runs."""
        return {
            "id": self.id,
            "name": self.name,
            "tester": self.tester,
            "suites": self.suites,
            "config": self.config,
            "target_id": self.target_id,
        }
