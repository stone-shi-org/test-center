from typing import TYPE_CHECKING, Any
from sqlalchemy import Integer, String, Text, JSON, ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.db.models.run import Run


class Baseline(Base, TimestampMixin):
    __tablename__ = "baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("runs.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, server_default="[]", nullable=False
    )

    __table_args__ = (
        Index("uq_baselines_name_lower", func.lower(name), unique=True),
    )

    run: Mapped["Run"] = relationship(
        "Run",
        back_populates="baselines",
    )
