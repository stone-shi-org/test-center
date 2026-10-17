"""Shared helpers and routes for safe artifact classification, path resolution, preview, content inspection, and download."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Literal
from fastapi import HTTPException, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.config import get_settings
from backend.app.db.models.artifact import Artifact
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run

ArtifactKind = Literal["html", "json", "audio", "log", "image", "binary"]

# Sandbox CSP for safe in-page HTML artifact rendering (blocks scripts, forms, same-origin, network)
HTML_SANDBOX_CSP = (
    "sandbox; "
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "img-src data: blob:; "
    "media-src 'self' data: blob:; "
    "font-src data:;"
)

_EXTENSION_MIME_MAP: dict[str, str] = {
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".ndjson": "application/x-ndjson",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".log": "text/plain",
    ".txt": "text/plain",
    ".out": "text/plain",
    ".err": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".xml": "application/xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}

_INTERNAL_EXCLUDED_FILES = {"payload.json"}


def classify_artifact(
    name: str,
    mime: str | None = None,
    path: str | None = None,
) -> tuple[str, ArtifactKind]:
    """Determine (effective_mime, kind) for an artifact from its filename, path, and declared MIME type."""
    raw_mime = (mime or "").strip().lower()
    if ";" in raw_mime:
        raw_mime = raw_mime.split(";", 1)[0].strip()

    candidate_name = (name or path or "").strip()
    ext = Path(candidate_name).suffix.lower()
    if not ext and path:
        ext = Path(path).suffix.lower()

    inferred_mime = _EXTENSION_MIME_MAP.get(ext)
    if not inferred_mime and candidate_name:
        guessed, _ = mimetypes.guess_type(candidate_name)
        if guessed:
            inferred_mime = guessed.lower()

    if not raw_mime or raw_mime == "application/octet-stream":
        effective_mime = inferred_mime or "application/octet-stream"
    else:
        effective_mime = raw_mime

    # 1. HTML
    if (
        ext in (".html", ".htm", ".xhtml")
        or effective_mime in ("text/html", "application/xhtml+xml")
    ):
        return (
            effective_mime if effective_mime in ("text/html", "application/xhtml+xml") else "text/html",
            "html",
        )

    # 2. JSON
    if (
        ext in (".json", ".jsonl", ".ndjson")
        or effective_mime == "application/json"
        or effective_mime.endswith("+json")
        or effective_mime == "application/x-ndjson"
    ):
        return (
            effective_mime if "json" in effective_mime else "application/json",
            "json",
        )

    # 3. Audio
    if (
        ext in (".wav", ".mp3", ".ogg", ".oga", ".opus", ".flac", ".m4a", ".aac")
        or effective_mime.startswith("audio/")
    ):
        if not effective_mime.startswith("audio/"):
            effective_mime = inferred_mime or "audio/wav"
        return effective_mime, "audio"

    # 4. Image
    if (
        ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
        or effective_mime.startswith("image/")
    ):
        if not effective_mime.startswith("image/"):
            effective_mime = inferred_mime or "image/png"
        return effective_mime, "image"

    # 5. Log / Text
    if (
        ext in (".log", ".txt", ".out", ".err", ".md", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".ini", ".cfg", ".conf")
        or effective_mime.startswith("text/")
        or effective_mime in ("application/xml", "application/x-yaml", "application/yaml")
    ):
        if effective_mime == "application/octet-stream":
            effective_mime = inferred_mime or "text/plain"
        return effective_mime, "log"

    return effective_mime, "binary"


def format_artifact_metadata(
    artifact: Artifact,
    *,
    run_id: int,
    result_id: int | None = None,
) -> dict[str, Any]:
    """Serialize an Artifact ORM object into an enriched dictionary with viewer metadata and URLs."""
    eff_result_id = result_id if result_id is not None else artifact.result_id
    effective_mime, kind = classify_artifact(artifact.name, artifact.mime, artifact.path)

    size_bytes = artifact.size_bytes
    if size_bytes is None and artifact.path:
        try:
            p = Path(artifact.path)
            if p.exists() and p.is_file():
                size_bytes = p.stat().st_size
        except OSError:
            pass

    return {
        "id": artifact.id,
        "result_id": eff_result_id,
        "run_id": run_id,
        "name": artifact.name,
        "path": artifact.path,
        "mime": effective_mime,
        "kind": kind,
        "viewer_type": kind,
        "size_bytes": size_bytes,
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
        "preview_url": f"/api/runs/{run_id}/artifacts/{artifact.id}/preview",
        "content_url": f"/api/runs/{run_id}/artifacts/{artifact.id}/content",
        "download_url": f"/api/runs/{run_id}/artifacts/{artifact.id}/download",
    }


def sync_run_disk_artifacts(db: Session, run: Run) -> list[dict[str, Any]]:
    """Discover any unindexed artifact files under data/artifacts/<run_id>/ and sync/serialize them."""
    settings = get_settings()
    run_dir = settings.ARTIFACTS_DIR / str(run.id)

    # Collect existing DB artifacts across all results of this run
    existing_paths: set[str] = set()
    existing_names: set[str] = set()
    for res in run.results or []:
        for a in res.artifacts or []:
            existing_paths.add(str(Path(a.path)))
            try:
                existing_paths.add(str(Path(a.path).resolve()))
            except OSError:
                pass
            existing_names.add(a.name)

    unlinked_virtual: list[dict[str, Any]] = []

    if run_dir.exists() and run_dir.is_dir():
        run_dir_resolved = run_dir.resolve()
        new_db_rows_added = False
        primary_result = run.results[0] if run.results else None

        for candidate in sorted(run_dir.rglob("*")):
            try:
                if candidate.is_symlink():
                    if not candidate.resolve().is_relative_to(run_dir_resolved):
                        continue
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if not resolved.is_relative_to(run_dir_resolved):
                    continue
            except OSError:
                continue

            rel_path = candidate.relative_to(run_dir).as_posix()
            if any(part.startswith(".") for part in Path(rel_path).parts):
                continue
            if candidate.name in _INTERNAL_EXCLUDED_FILES:
                continue

            if (
                str(candidate) in existing_paths
                or str(resolved) in existing_paths
                or rel_path in existing_names
                or candidate.name in existing_names
            ):
                continue

            effective_mime, kind = classify_artifact(rel_path, None, str(candidate))
            try:
                size_bytes = candidate.stat().st_size
            except OSError:
                size_bytes = None

            if primary_result is not None:
                new_art = Artifact(
                    result_id=primary_result.id,
                    name=rel_path,
                    path=str(candidate),
                    mime=effective_mime,
                    size_bytes=size_bytes,
                )
                db.add(new_art)
                existing_paths.add(str(candidate))
                existing_paths.add(str(resolved))
                existing_names.add(rel_path)
                new_db_rows_added = True
            else:
                unlinked_virtual.append({
                    "id": 0,
                    "result_id": None,
                    "run_id": run.id,
                    "name": rel_path,
                    "path": str(candidate),
                    "mime": effective_mime,
                    "kind": kind,
                    "viewer_type": kind,
                    "size_bytes": size_bytes,
                    "created_at": None,
                    "preview_url": f"/api/runs/{run.id}/artifacts/{rel_path}/preview",
                    "content_url": f"/api/runs/{run.id}/artifacts/{rel_path}/content",
                    "download_url": f"/api/runs/{run.id}/artifacts/{rel_path}/download",
                })

        if new_db_rows_added:
            db.commit()
            db.refresh(run)

    output: list[dict[str, Any]] = []
    for res in run.results or []:
        for a in res.artifacts or []:
            output.append(format_artifact_metadata(a, run_id=run.id, result_id=res.id))
    output.extend(unlinked_virtual)
    return output


def _validate_safe_file_path(
    raw_path: str | Path,
    run_id: int | None = None,
) -> Path:
    """Validate and resolve an artifact file path on disk, blocking path traversal and escaping symlinks."""
    path_obj = Path(raw_path)
    if ".." in path_obj.parts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact path: path traversal is not allowed",
        )

    settings = get_settings()
    artifacts_root = settings.ARTIFACTS_DIR.resolve()
    run_dir = (settings.ARTIFACTS_DIR / str(run_id)).resolve() if run_id is not None else None

    candidates: list[Path] = []
    if path_obj.is_absolute():
        candidates.append(path_obj)
        if run_id is not None:
            candidates.append(settings.ARTIFACTS_DIR / str(run_id) / path_obj.name)
    else:
        if run_id is not None:
            candidates.append(settings.ARTIFACTS_DIR / str(run_id) / path_obj)
            candidates.append(settings.ARTIFACTS_DIR / str(run_id) / path_obj.name)
        candidates.append(settings.ARTIFACTS_DIR / path_obj)
        candidates.append(settings.DATA_DIR / path_obj)
        candidates.append(path_obj)

    for candidate in candidates:
        try:
            if candidate.is_symlink():
                target_resolved = candidate.resolve()
                allowed_root = run_dir if run_dir is not None else artifacts_root
                if not target_resolved.is_relative_to(allowed_root):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid artifact path: symlink escapes artifact directory",
                    )

            if candidate.exists() and candidate.is_file():
                resolved = candidate.resolve()
                # If the path is inside ARTIFACTS_DIR and run_id is known, enforce run_dir isolation
                if run_dir is not None and resolved.is_relative_to(artifacts_root):
                    if not resolved.is_relative_to(run_dir):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid artifact path: cross-run path access is not allowed",
                        )
                return resolved
        except HTTPException:
            raise
        except OSError:
            continue

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Artifact file does not exist on disk",
    )


def resolve_run_artifact(
    db: Session,
    run_id: int,
    artifact_ref: str | int,
) -> tuple[dict[str, Any], Path]:
    """Resolve an artifact belonging to `run_id` by integer ID or relative filename under data/artifacts/<run_id>/."""
    ref_str = str(artifact_ref).strip()
    if not ref_str:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    # Block any traversal attempts immediately
    if ".." in Path(ref_str).parts or "\\" in ref_str or ref_str.startswith("/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact path: path traversal is not allowed",
        )

    # 1. Try integer Artifact ID lookup
    if ref_str.isdigit():
        artifact_id = int(ref_str)
        stmt = (
            select(Artifact)
            .options(selectinload(Artifact.result))
            .where(Artifact.id == artifact_id)
        )
        artifact = db.execute(stmt).scalars().first()
        if artifact is not None:
            if artifact.result is None or artifact.result.run_id != run_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Artifact not found for this run",
                )
            file_path = _validate_safe_file_path(artifact.path, run_id=run_id)
            meta = format_artifact_metadata(artifact, run_id=run_id, result_id=artifact.result_id)
            return meta, file_path

    # 2. Try matching by artifact name within the run's results or disk directory
    stmt_run = (
        select(Run)
        .options(selectinload(Run.results).selectinload(Result.artifacts))
        .where(Run.id == run_id)
    )
    run = db.execute(stmt_run).scalars().first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    # Sync disk artifacts first so newly created files get DB records if possible
    sync_run_disk_artifacts(db, run)

    for res in run.results or []:
        for a in res.artifacts or []:
            if a.name == ref_str or Path(a.name).name == ref_str:
                file_path = _validate_safe_file_path(a.path, run_id=run_id)
                meta = format_artifact_metadata(a, run_id=run_id, result_id=res.id)
                return meta, file_path

    # 3. Fallback: direct file under data/artifacts/<run_id>/
    settings = get_settings()
    run_dir = (settings.ARTIFACTS_DIR / str(run_id)).resolve()
    candidate = run_dir / ref_str
    if candidate.is_symlink() and not candidate.resolve().is_relative_to(run_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact path: symlink escapes run artifact directory",
        )
    resolved = candidate.resolve()
    if not resolved.is_relative_to(run_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact path: path traversal is not allowed",
        )
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found",
        )

    effective_mime, kind = classify_artifact(ref_str, None, str(resolved))
    meta = {
        "id": 0,
        "result_id": run.results[0].id if run.results else None,
        "run_id": run_id,
        "name": ref_str,
        "path": str(resolved),
        "mime": effective_mime,
        "kind": kind,
        "viewer_type": kind,
        "size_bytes": resolved.stat().st_size,
        "created_at": None,
        "preview_url": f"/api/runs/{run_id}/artifacts/{ref_str}/preview",
        "content_url": f"/api/runs/{run_id}/artifacts/{ref_str}/content",
        "download_url": f"/api/runs/{run_id}/artifacts/{ref_str}/download",
    }
    return meta, resolved


def resolve_result_artifact(
    db: Session,
    result_id: int,
    artifact_id: int,
) -> tuple[dict[str, Any], Path]:
    """Resolve an artifact belonging to `result_id` by its Artifact ID."""
    stmt = (
        select(Artifact)
        .options(selectinload(Artifact.result))
        .where(Artifact.id == artifact_id)
    )
    artifact = db.execute(stmt).scalars().first()
    if not artifact or artifact.result_id != result_id or artifact.result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    run_id = artifact.result.run_id
    file_path = _validate_safe_file_path(artifact.path, run_id=run_id)
    meta = format_artifact_metadata(artifact, run_id=run_id, result_id=result_id)
    return meta, file_path


def build_artifact_download_response(meta: dict[str, Any], file_path: Path) -> FileResponse:
    """Create a FileResponse for downloading an artifact as an attachment."""
    safe_filename = Path(meta["name"]).name or "artifact"
    return FileResponse(
        path=str(file_path),
        filename=safe_filename,
        media_type=meta["mime"],
        headers={
            "X-Content-Type-Options": "nosniff",
        },
    )


def build_artifact_preview_response(meta: dict[str, Any], file_path: Path) -> Response:
    """Create a safe inline Response/FileResponse for previewing an artifact in-page."""
    safe_filename = Path(meta["name"]).name.replace('"', "") or "artifact"
    headers: dict[str, str] = {
        "Content-Disposition": f'inline; filename="{safe_filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    if meta["kind"] == "html" or meta["mime"] in ("text/html", "application/xhtml+xml"):
        headers["Content-Security-Policy"] = HTML_SANDBOX_CSP
        headers["Referrer-Policy"] = "no-referrer"

    return FileResponse(
        path=str(file_path),
        media_type=meta["mime"],
        headers=headers,
    )


def build_artifact_content_payload(
    meta: dict[str, Any],
    file_path: Path,
    *,
    max_bytes: int = 5_000_000,
    search: str | None = None,
    level: str | None = None,
    tail: int | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build structured JSON payload for in-page artifact viewers (HTML, JSON tree, Audio, Log viewer)."""
    kind: ArtifactKind = meta["kind"]
    file_size = file_path.stat().st_size

    payload: dict[str, Any] = {
        **meta,
        "size_bytes": file_size,
        "truncated": False,
        "content": None,
        "json_data": None,
        "json_valid": None,
        "json_error": None,
        "lines": None,
        "total_lines": None,
        "filtered_lines": None,
    }

    if kind == "html":
        payload["sandbox_csp"] = HTML_SANDBOX_CSP

    if kind in ("html", "json", "log"):
        with open(file_path, "rb") as f:
            raw_bytes = f.read(max_bytes + 1)
        if len(raw_bytes) > max_bytes:
            payload["truncated"] = True
            raw_bytes = raw_bytes[:max_bytes]

        text = raw_bytes.decode("utf-8", errors="replace")
        payload["content"] = text

        if kind == "json":
            try:
                if meta["name"].lower().endswith((".jsonl", ".ndjson")):
                    payload["json_data"] = [
                        json.loads(line) for line in text.splitlines() if line.strip()
                    ]
                else:
                    payload["json_data"] = json.loads(text)
                payload["json_valid"] = True
            except Exception as e:
                payload["json_valid"] = False
                payload["json_error"] = str(e)

        elif kind == "log":
            all_lines = text.splitlines()
            payload["total_lines"] = len(all_lines)
            filtered = all_lines

            if level and level.strip().upper() != "ALL":
                lvl_upper = level.strip().upper()
                filtered = [
                    ln
                    for ln in filtered
                    if f"[{lvl_upper}]" in ln.upper() or f" {lvl_upper} " in ln.upper() or ln.upper().startswith(lvl_upper)
                ]

            if search and search.strip():
                q = search.strip().lower()
                filtered = [ln for ln in filtered if q in ln.lower()]

            payload["filtered_lines"] = len(filtered)

            if tail is not None and tail > 0:
                filtered = filtered[-tail:]
            elif offset > 0 or limit is not None:
                end = (offset + limit) if limit is not None else None
                filtered = filtered[offset:end]

            payload["lines"] = filtered

    return payload
