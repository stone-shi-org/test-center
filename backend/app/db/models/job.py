from typing import TYPE_CHECKING
from sqlalchemy import Integer, String, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.db.models.run import Run


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    schedule: Mapped[str] = mapped_column(String(128), default="manual", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    definition_ids: Mapped[list[int]] = mapped_column(JSON, default=list, nullable=False)

    runs: Mapped[list["Run"]] = relationship(
        "Run",
        back_populates="job",
    )
