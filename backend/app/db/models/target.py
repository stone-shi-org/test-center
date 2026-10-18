from typing import TYPE_CHECKING, Any
from sqlalchemy import Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.db.models.test_definition import TestDefinition

_CAPABILITY_ALIASES: dict[str, str] = {
    "chat": "chat",
    "llm": "chat",
    "completion": "chat",
    "completions": "chat",
    "embedding": "embedding",
    "embeddings": "embedding",
    "embed": "embedding",
    "asr": "asr",
    "stt": "asr",
    "transcription": "asr",
    "transcriptions": "asr",
    "live-stt": "asr",
    "live_stt": "asr",
    "grpc": "grpc",
    "grpc-streaming": "grpc",
    "grpc_streaming": "grpc",
    "streaming-asr": "grpc",
    "tts": "tts",
    "speech": "tts",
    "diarization": "diarization",
    "diarize": "diarization",
}


def normalize_capabilities(capabilities: list[str] | None) -> list[str]:
    """Normalize and deduplicate capability identifiers while preserving order."""
    if not capabilities:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for raw in capabilities:
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip().lower()
        if not cleaned:
            continue
        norm = _CAPABILITY_ALIASES.get(cleaned, cleaned)
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def infer_target_capabilities(kind: str | None, extra: dict[str, Any] | None = None) -> list[str]:
    """Infer target capabilities from explicit extra.capabilities or target kind."""
    if isinstance(extra, dict):
        raw_caps = extra.get("capabilities")
        if isinstance(raw_caps, list):
            norm_caps = normalize_capabilities([str(c) for c in raw_caps])
            if norm_caps:
                return norm_caps

    k = (kind or "").strip().lower()
    if k in ("embedding", "embeddings"):
        return ["embedding"]
    if k in ("live-stt", "live_stt", "stt", "asr"):
        return ["asr", "tts", "diarization", "grpc"]
    if k in ("openai-compatible", "llm", "chat"):
        return ["chat"]
    return ["chat"]


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

    @property
    def capabilities(self) -> list[str]:
        """Return configured or inferred capabilities for this target."""
        return infer_target_capabilities(self.kind, self.extra)

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

