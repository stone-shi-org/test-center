"""Retention policy management and cleanup for old runs and artifacts.

Supports:
- Persisting retention configuration (`enabled`, `max_age_days` / `retention_days`,
  `max_runs`, `artifacts_only`, `keep_failed`) in `DATA_DIR / "retention_settings.json"`.
- Inspecting live storage statistics and identifying eligible old runs/artifacts.
- Executing retention cleanup (`apply_retention_policy`) either deleting full old runs
  (and their on-disk artifact directories) or pruning only artifacts while retaining
  historical Run/Result/CaseResult records for long-term metric trend analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import shutil
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.config import get_settings
from backend.app.db.base import utc_now
from backend.app.db.models.artifact import Artifact
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run

logger = logging.getLogger(__name__)

TERMINAL_RUN_STATUSES = ("completed", "failed", "error", "cancelled")
ACTIVE_RUN_STATUSES = ("pending", "queued", "running", "cancelling")


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _run_effective_time(run: Run) -> datetime | None:
    """Resolve the best timestamp for evaluating run age."""
    for candidate in (run.finished_at, run.started_at, run.queued_at, run.created_at):
        if candidate is not None:
            return _ensure_utc(candidate)
    return None


def _is_failed_run(run: Run) -> bool:
    """Return True if a run or any of its results failed or errored."""
    if run.status in ("failed", "error"):
        return True
    for res in run.results or []:
        if res.status in ("fail", "error"):
            return True
    return False


def _normalize_positive_int(val: Any, field_name: str) -> int | None:
    """Normalize a limit value: None or 0 -> None; positive int -> int; negative -> ValueError."""
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        raise ValueError(f"{field_name} must be an integer, not boolean")
    try:
        num = int(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must be a non-negative integer") from e
    if num < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return num if num > 0 else None


def _retention_file_path(data_dir: Path | None = None) -> Path:
    settings = get_settings()
    base = Path(data_dir) if data_dir is not None else Path(settings.DATA_DIR)
    return base / "retention_settings.json"


def _default_retention_settings() -> dict[str, Any]:
    settings = get_settings()
    max_age_days = settings.RETENTION_MAX_AGE_DAYS
    max_runs = settings.RETENTION_MAX_RUNS
    return {
        "enabled": bool(settings.RETENTION_ENABLED),
        "max_age_days": max_age_days,
        "retention_days": max_age_days,
        "max_runs": max_runs,
        "artifacts_only": bool(settings.RETENTION_ARTIFACTS_ONLY),
        "keep_failed": bool(settings.RETENTION_KEEP_FAILED),
        "last_cleanup_at": None,
        "last_cleanup_summary": None,
    }


def load_retention_settings(data_dir: Path | None = None) -> dict[str, Any]:
    """Load persisted retention settings from disk or return defaults."""
    defaults = _default_retention_settings()
    path = _retention_file_path(data_dir)
    if not path.exists():
        return defaults

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return defaults
    except Exception as e:
        logger.warning(f"Failed to read retention settings from {path}: {e}")
        return defaults

    raw_days = raw.get("max_age_days") if "max_age_days" in raw else raw.get("retention_days")
    try:
        max_age_days = _normalize_positive_int(raw_days, "max_age_days")
    except ValueError:
        max_age_days = defaults["max_age_days"]

    try:
        max_runs = _normalize_positive_int(raw.get("max_runs"), "max_runs")
    except ValueError:
        max_runs = defaults["max_runs"]

    return {
        "enabled": bool(raw.get("enabled", defaults["enabled"])),
        "max_age_days": max_age_days,
        "retention_days": max_age_days,
        "max_runs": max_runs,
        "artifacts_only": bool(raw.get("artifacts_only", defaults["artifacts_only"])),
        "keep_failed": bool(raw.get("keep_failed", defaults["keep_failed"])),
        "last_cleanup_at": raw.get("last_cleanup_at"),
        "last_cleanup_summary": raw.get("last_cleanup_summary"),
    }


def save_retention_settings(
    updates: dict[str, Any],
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate and persist updated retention settings to disk."""
    current = load_retention_settings(data_dir)

    if "max_age_days" in updates and updates["max_age_days"] is not None:
        max_age_days = _normalize_positive_int(updates["max_age_days"], "max_age_days")
    elif "retention_days" in updates and updates["retention_days"] is not None:
        max_age_days = _normalize_positive_int(updates["retention_days"], "retention_days")
    elif "days" in updates and updates["days"] is not None:
        max_age_days = _normalize_positive_int(updates["days"], "days")
    elif any(k in updates for k in ("max_age_days", "retention_days", "days")):
        max_age_days = None
    else:
        max_age_days = current["max_age_days"]

    if "max_runs" in updates and updates["max_runs"] is not None:
        max_runs = _normalize_positive_int(updates["max_runs"], "max_runs")
    elif "keep_last_n" in updates and updates["keep_last_n"] is not None:
        max_runs = _normalize_positive_int(updates["keep_last_n"], "keep_last_n")
    elif any(k in updates for k in ("max_runs", "keep_last_n")):
        max_runs = None
    else:
        max_runs = current["max_runs"]

    enabled = bool(updates["enabled"]) if "enabled" in updates and updates["enabled"] is not None else current["enabled"]
    artifacts_only = (
        bool(updates["artifacts_only"])
        if "artifacts_only" in updates and updates["artifacts_only"] is not None
        else current["artifacts_only"]
    )
    keep_failed = (
        bool(updates["keep_failed"])
        if "keep_failed" in updates and updates["keep_failed"] is not None
        else current["keep_failed"]
    )
    last_cleanup_at = updates.get("last_cleanup_at", current.get("last_cleanup_at"))
    last_cleanup_summary = updates.get("last_cleanup_summary", current.get("last_cleanup_summary"))

    new_settings = {
        "enabled": enabled,
        "max_age_days": max_age_days,
        "retention_days": max_age_days,
        "max_runs": max_runs,
        "artifacts_only": artifacts_only,
        "keep_failed": keep_failed,
        "last_cleanup_at": last_cleanup_at,
        "last_cleanup_summary": last_cleanup_summary,
    }

    path = _retention_file_path(data_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(new_settings, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write retention settings to {path}: {e}")

    return new_settings


def _compute_dir_size_bytes(directory: Path) -> int:
    """Calculate total bytes of all files under a directory."""
    if not directory.exists():
        return 0
    total = 0
    try:
        for p in directory.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _compute_run_artifacts_bytes(run: Run, artifacts_dir: Path) -> int:
    """Compute disk usage for a specific run's artifacts."""
    run_dir = artifacts_dir / str(run.id)
    if run_dir.exists():
        return _compute_dir_size_bytes(run_dir)

    total = 0
    seen_paths: set[str] = set()
    for res in run.results or []:
        for art in res.artifacts or []:
            if art.path and art.path not in seen_paths:
                seen_paths.add(art.path)
                p = Path(art.path)
                if p.exists() and p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
                elif isinstance(art.size_bytes, int) and art.size_bytes > 0:
                    total += art.size_bytes
    return total


def find_eligible_runs_for_retention(
    db: Session,
    *,
    max_age_days: int | None,
    max_runs: int | None,
    artifacts_only: bool = False,
    keep_failed: bool = False,
    artifacts_dir: Path | None = None,
    now: datetime | None = None,
) -> tuple[list[Run], datetime | None]:
    """Identify terminal runs that exceed `max_runs` or are older than `max_age_days`."""
    ref_now = _ensure_utc(now) or utc_now()
    cutoff: datetime | None = (
        ref_now - timedelta(days=int(max_age_days))
        if (max_age_days is not None and int(max_age_days) > 0)
        else None
    )

    if cutoff is None and (max_runs is None or int(max_runs) <= 0):
        return [], None

    stmt = (
        select(Run)
        .options(selectinload(Run.results).selectinload(Result.artifacts))
        .where(Run.status.in_(TERMINAL_RUN_STATUSES))
        .order_by(Run.id.desc())
    )
    terminal_runs = list(db.execute(stmt).scalars().all())

    eligible: list[Run] = []
    eff_artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else get_settings().ARTIFACTS_DIR

    for idx, run in enumerate(terminal_runs):
        if keep_failed and _is_failed_run(run):
            continue

        exceeds_count = max_runs is not None and int(max_runs) > 0 and idx >= int(max_runs)
        exceeds_age = False
        if cutoff is not None:
            r_time = _run_effective_time(run)
            if r_time is not None and r_time < cutoff:
                exceeds_age = True

        if not (exceeds_count or exceeds_age):
            continue

        if artifacts_only:
            has_db_artifacts = any(len(res.artifacts or []) > 0 for res in (run.results or []))
            has_disk_dir = (eff_artifacts_dir / str(run.id)).exists()
            if not has_db_artifacts and not has_disk_dir:
                continue

        eligible.append(run)

    return eligible, cutoff


def get_retention_status(
    db: Session,
    artifacts_dir: Path | None = None,
    data_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return current retention configuration together with storage and eligibility statistics."""
    settings = get_settings()
    eff_artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else settings.ARTIFACTS_DIR
    eff_data_dir = Path(data_dir) if data_dir is not None else settings.DATA_DIR

    cfg = load_retention_settings(eff_data_dir)
    total_runs = db.scalar(select(func.count(Run.id))) or 0
    terminal_runs = (
        db.scalar(select(func.count(Run.id)).where(Run.status.in_(TERMINAL_RUN_STATUSES))) or 0
    )
    active_runs = (
        db.scalar(select(func.count(Run.id)).where(Run.status.in_(ACTIVE_RUN_STATUSES))) or 0
    )
    total_artifacts = db.scalar(select(func.count(Artifact.id))) or 0
    disk_bytes = _compute_dir_size_bytes(eff_artifacts_dir)

    eligible_runs, cutoff = find_eligible_runs_for_retention(
        db,
        max_age_days=cfg["max_age_days"],
        max_runs=cfg["max_runs"],
        artifacts_only=cfg["artifacts_only"],
        keep_failed=cfg["keep_failed"],
        artifacts_dir=eff_artifacts_dir,
        now=now,
    )

    return {
        **cfg,
        "cutoff_date": cutoff.isoformat() if cutoff else None,
        "stats": {
            "total_runs": total_runs,
            "terminal_runs": terminal_runs,
            "active_runs": active_runs,
            "total_artifacts": total_artifacts,
            "artifacts_size_bytes": disk_bytes,
            "eligible_runs_count": len(eligible_runs),
            "eligible_run_ids": [r.id for r in eligible_runs],
        },
    }


def apply_retention_policy(
    db: Session,
    artifacts_dir: Path | None = None,
    *,
    max_age_days: int | None = None,
    max_runs: int | None = None,
    artifacts_only: bool | None = None,
    keep_failed: bool | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute retention cleanup on old terminal runs and/or their artifacts."""
    settings = get_settings()
    eff_artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else settings.ARTIFACTS_DIR
    eff_data_dir = Path(data_dir) if data_dir is not None else settings.DATA_DIR

    saved = load_retention_settings(eff_data_dir)
    eff_max_age_days = (
        _normalize_positive_int(max_age_days, "max_age_days")
        if max_age_days is not None
        else saved["max_age_days"]
    )
    eff_max_runs = (
        _normalize_positive_int(max_runs, "max_runs")
        if max_runs is not None
        else saved["max_runs"]
    )
    eff_artifacts_only = (
        bool(artifacts_only) if artifacts_only is not None else bool(saved["artifacts_only"])
    )
    eff_keep_failed = (
        bool(keep_failed) if keep_failed is not None else bool(saved["keep_failed"])
    )

    eligible_runs, cutoff = find_eligible_runs_for_retention(
        db,
        max_age_days=eff_max_age_days,
        max_runs=eff_max_runs,
        artifacts_only=eff_artifacts_only,
        keep_failed=eff_keep_failed,
        artifacts_dir=eff_artifacts_dir,
        now=now,
    )

    deleted_run_ids: list[int] = []
    cleaned_run_ids: list[int] = []
    deleted_artifacts_count = 0
    freed_bytes = 0

    for run in eligible_runs:
        run_id = run.id
        run_bytes = _compute_run_artifacts_bytes(run, eff_artifacts_dir)
        run_artifact_rows = [
            art for res in (run.results or []) for art in (res.artifacts or [])
        ]
        freed_bytes += run_bytes
        deleted_artifacts_count += len(run_artifact_rows)
        cleaned_run_ids.append(run_id)
        if not eff_artifacts_only:
            deleted_run_ids.append(run_id)

        if not dry_run:
            # Remove artifact files on disk
            run_dir = eff_artifacts_dir / str(run_id)
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

            for art in run_artifact_rows:
                if art.path:
                    p = Path(art.path)
                    if p.exists() and p.is_file():
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass

            if eff_artifacts_only:
                for art in run_artifact_rows:
                    db.delete(art)
            else:
                db.delete(run)

    if not dry_run:
        db.commit()

        # Also clean up orphaned numeric run directories in artifacts_dir if not artifacts_only
        if eff_artifacts_dir.exists():
            existing_run_ids = set(db.execute(select(Run.id)).scalars().all())
            try:
                for child in eff_artifacts_dir.iterdir():
                    if child.is_dir() and child.name.isdigit():
                        dir_run_id = int(child.name)
                        if dir_run_id not in existing_run_ids:
                            freed_bytes += _compute_dir_size_bytes(child)
                            shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass

    summary = {
        "status": "ok",
        "dry_run": dry_run,
        "artifacts_only": eff_artifacts_only,
        "keep_failed": eff_keep_failed,
        "max_age_days": eff_max_age_days,
        "retention_days": eff_max_age_days,
        "max_runs": eff_max_runs,
        "cutoff_date": cutoff.isoformat() if cutoff else None,
        "deleted_runs": len(deleted_run_ids),
        "deleted_run_ids": deleted_run_ids,
        "cleaned_runs": len(cleaned_run_ids),
        "cleaned_run_ids": cleaned_run_ids,
        "deleted_artifacts": deleted_artifacts_count,
        "freed_bytes": freed_bytes,
    }

    if not dry_run:
        ref_now = _ensure_utc(now) or utc_now()
        save_retention_settings(
            {
                "last_cleanup_at": ref_now.isoformat(),
                "last_cleanup_summary": summary,
            },
            data_dir=eff_data_dir,
        )

    return summary
