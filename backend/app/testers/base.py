"""Base contract and data structures for tester plugins."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Literal

from backend.app.db.metrics import format_metric, validate_metrics_dict
from backend.app.testers.schema import validate_schema, apply_schema_defaults


class TesterError(Exception):
    """Base exception for tester errors."""
    __test__ = False


class TesterCancelledError(TesterError):
    """Raised when a tester execution is cancelled."""
    __test__ = False


class TesterConfigError(TesterError):
    """Raised when tester configuration is invalid."""
    __test__ = False


@dataclass
class CaseResultData:
    """Individual test case result."""
    case_id: str
    suite: str
    status: Literal["pass", "fail", "error"]
    metrics: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseResultData":
        return cls(
            case_id=data["case_id"],
            suite=data["suite"],
            status=data["status"],
            metrics=data.get("metrics", {}),
            detail=data.get("detail", {}),
        )


@dataclass
class ArtifactData:
    """Artifact produced during test execution."""
    name: str
    path: str
    mime: str = "application/octet-stream"
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactData":
        return cls(
            name=data["name"],
            path=data["path"],
            mime=data.get("mime", "application/octet-stream"),
            size_bytes=data.get("size_bytes"),
        )


@dataclass
class TesterResult:
    """Structured result returned by a tester plugin."""
    __test__ = False
    status: Literal["pass", "fail", "error", "cancelled"]
    metrics: dict[str, Any] = field(default_factory=dict)
    cases: list[CaseResultData] = field(default_factory=list)
    artifacts: list[ArtifactData] = field(default_factory=list)
    summary: str | None = None

    def set_metric(
        self,
        name: str,
        value: float | int,
        unit: str | None = None,
        better: Literal["higher", "lower"] | str | None = None,
    ) -> None:
        """Add or update a metric with unit and direction."""
        self.metrics[name] = format_metric(value=value, unit=unit, better=better)

    def add_case(
        self,
        case_id: str,
        suite: str,
        status: Literal["pass", "fail", "error"],
        metrics: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> CaseResultData:
        """Add a test case result."""
        case = CaseResultData(
            case_id=case_id,
            suite=suite,
            status=status,
            metrics=metrics or {},
            detail=detail or {},
        )
        self.cases.append(case)
        return case

    def add_artifact(
        self,
        name: str,
        path: str | Path,
        mime: str = "application/octet-stream",
        size_bytes: int | None = None,
    ) -> ArtifactData:
        """Add an artifact to this result."""
        p = Path(path)
        if size_bytes is None and p.exists() and p.is_file():
            try:
                size_bytes = p.stat().st_size
            except OSError:
                size_bytes = None
        artifact = ArtifactData(
            name=name,
            path=str(p),
            mime=mime,
            size_bytes=size_bytes,
        )
        self.artifacts.append(artifact)
        return artifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metrics": self.metrics,
            "cases": [c.to_dict() for c in self.cases],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TesterResult":
        cases = [CaseResultData.from_dict(c) for c in data.get("cases", [])]
        artifacts = [ArtifactData.from_dict(a) for a in data.get("artifacts", [])]
        return cls(
            status=data["status"],
            metrics=data.get("metrics", {}),
            cases=cases,
            artifacts=artifacts,
            summary=data.get("summary"),
        )


class TestContext:
    """Execution context passed to tester plugins.
    
    Provides logging, progress updates, cancellation check, and artifact registration.
    """
    __test__ = False

    def __init__(
        self,
        run_id: int,
        artifacts_dir: Path,
        log_file: Path | None = None,
        emit_stdout: bool = True,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.artifacts_dir = Path(artifacts_dir)
        self.log_file = Path(log_file) if log_file else self.artifacts_dir / "run.log"
        self.emit_stdout = emit_stdout
        self.on_event = on_event
        self._cancelled = False
        self.artifacts: list[ArtifactData] = []

        # Ensure directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_cancelled(self) -> bool:
        """Return True if execution has been cancelled."""
        return self._cancelled

    def set_cancelled(self) -> None:
        """Mark context as cancelled."""
        self._cancelled = True

    def check_cancelled(self) -> None:
        """Raise TesterCancelledError if execution has been cancelled."""
        if self._cancelled:
            raise TesterCancelledError(f"Run {self.run_id} cancelled")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log(self, message: str, level: str = "info") -> None:
        """Log a message, write to disk, and stream event."""
        level = level.upper()
        ts = self._now_iso()
        formatted_line = f"[{ts}] [{level}] {message}\n"

        # Append to log file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(formatted_line)
                f.flush()
        except OSError:
            pass

        event_payload = {
            "type": "log",
            "run_id": self.run_id,
            "timestamp": ts,
            "level": level,
            "message": message,
        }

        if self.on_event:
            self.on_event(event_payload)

        if self.emit_stdout:
            sys.stdout.write(json.dumps(event_payload) + "\n")
            sys.stdout.flush()

    def progress(self, current: int, total: int, message: str | None = None) -> None:
        """Emit progress update event."""
        ts = self._now_iso()
        event_payload = {
            "type": "progress",
            "run_id": self.run_id,
            "timestamp": ts,
            "current": current,
            "total": total,
            "message": message or "",
        }

        if self.on_event:
            self.on_event(event_payload)

        if self.emit_stdout:
            sys.stdout.write(json.dumps(event_payload) + "\n")
            sys.stdout.flush()

    def add_artifact(
        self,
        name: str,
        path: str | Path,
        mime: str = "application/octet-stream",
    ) -> ArtifactData:
        """Register an artifact file created under the run directory."""
        p = Path(path)
        size_bytes = None
        if p.exists() and p.is_file():
            try:
                size_bytes = p.stat().st_size
            except OSError:
                pass

        artifact = ArtifactData(
            name=name,
            path=str(p),
            mime=mime,
            size_bytes=size_bytes,
        )
        self.artifacts.append(artifact)
        return artifact


class TesterPlugin(ABC):
    """Abstract base class for all tester plugins."""
    __test__ = False

    name: str = ""
    description: str = ""
    config_schema: dict[str, Any] = {}
    suites: list[str] = []

    def validate_and_apply_config(self, raw_config: dict[str, Any] | None) -> dict[str, Any]:
        """Validate config against JSON Schema and apply defaults."""
        config_with_defaults = apply_schema_defaults(raw_config or {}, self.config_schema)
        try:
            validate_schema(config_with_defaults, self.config_schema)
        except Exception as e:
            raise TesterConfigError(f"Invalid configuration for tester '{self.name}': {e}") from e
        return config_with_defaults

    @abstractmethod
    def run(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
        ctx: TestContext,
    ) -> TesterResult:
        """Execute test run.
        
        Args:
            config: Tester-specific validated config dictionary.
            target: Target snapshot dictionary (e.g. base_url, kind, extra).
            ctx: TestContext providing log(), progress(), cancellation checks, and artifact helpers.
            
        Returns:
            TesterResult containing status, metrics, cases, artifacts, summary.
        """
        pass
