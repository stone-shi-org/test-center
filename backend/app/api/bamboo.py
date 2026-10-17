"""Bamboo / CI integration API routes and status reporting.

Provides:
- Token-protected endpoints to trigger a job (`POST /api/bamboo/jobs/{job_ref}/trigger`, `POST /api/bamboo/trigger`)
- Token-protected endpoints to poll a run's status and result (`GET /api/bamboo/runs/{run_id}`, `GET /api/bamboo/runs/{run_id}/status`)
- JUnit XML export of a run's cases (`GET /api/bamboo/runs/{run_id}/junit.xml`, `GET /api/runs/{run_id}/junit.xml`)
- Optional automatic or on-demand status report-back to a Bamboo build result or CI callback URL (`POST /api/bamboo/runs/{run_id}/report`)
- Ready-to-use Bamboo plan script snippet generator (`GET /api/jobs/{job_id}/bamboo`)
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
import httpx
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.config import get_settings
from backend.app.db.models.job import Job
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run
from backend.app.db.session import SessionLocal, get_db
from backend.app.runner.junit import build_run_junit_xml, write_run_junit_artifact

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bamboo", tags=["bamboo"])
ci_alias_router = APIRouter(prefix="/ci", tags=["bamboo"])

_TERMINAL_RUN_STATUSES = {"completed", "cancelled", "error", "failed"}

_CANDIDATE_TOKEN_ENV_VARS = (
    "BAMBOO_API_TOKEN",
    "BAMBOO_TOKEN",
    "CI_API_TOKEN",
    "CI_TOKEN",
    "TEST_CENTER_API_TOKEN",
    "API_TOKEN",
)


def get_configured_bamboo_token_env_vars(job: Job | None = None) -> list[str]:
    """Return ordered list of environment variable names checked for CI/Bamboo token auth."""
    settings = get_settings()
    env_vars: list[str] = []

    def _add(name: Any) -> None:
        if isinstance(name, str) and name.strip() and name.strip() not in env_vars:
            env_vars.append(name.strip())

    if job is not None:
        if isinstance(job.overrides, dict):
            _add(job.overrides.get("bamboo_token_env_var"))
            _add(job.overrides.get("ci_token_env_var"))
            bamboo_ovr = job.overrides.get("bamboo")
            if isinstance(bamboo_ovr, dict):
                _add(bamboo_ovr.get("token_env_var"))
        if isinstance(job.notifications, dict):
            _add(job.notifications.get("bamboo_token_env_var"))

    _add(getattr(settings, "BAMBOO_TOKEN_ENV_VAR", "BAMBOO_API_TOKEN"))
    for fallback_var in _CANDIDATE_TOKEN_ENV_VARS:
        _add(fallback_var)

    return env_vars


def get_configured_bamboo_tokens(job: Job | None = None) -> list[str]:
    """Collect all non-empty token values configured in environment variables (never returned by API)."""
    tokens: list[str] = []
    for var_name in get_configured_bamboo_token_env_vars(job):
        val = os.getenv(var_name)
        if val and val.strip():
            clean = val.strip()
            if clean not in tokens:
                tokens.append(clean)
    return tokens


def is_bamboo_token_configured(job: Job | None = None) -> bool:
    """Return True if at least one CI/Bamboo token environment variable is set and non-empty."""
    return len(get_configured_bamboo_tokens(job)) > 0


def extract_request_token(
    request: Request | None = None,
    *,
    authorization: str | None = None,
    x_bamboo_token: str | None = None,
    x_ci_token: str | None = None,
    x_api_token: str | None = None,
    x_api_key: str | None = None,
    token_query: str | None = None,
    api_token_query: str | None = None,
) -> str | None:
    """Extract CI/Bamboo authentication token from headers or query parameters."""
    # 1. Explicit header args or request headers
    auth_val = authorization
    if auth_val is None and request is not None:
        auth_val = request.headers.get("Authorization")

    if auth_val and auth_val.strip():
        raw_auth = auth_val.strip()
        parts = raw_auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token", "apikey"):
            if parts[1].strip():
                return parts[1].strip()
        elif len(parts) == 2 and parts[0].lower() == "basic":
            try:
                decoded = base64.b64decode(parts[1].strip()).decode("utf-8", errors="replace")
                if ":" in decoded:
                    _, pwd = decoded.split(":", 1)
                    if pwd.strip():
                        return pwd.strip()
                elif decoded.strip():
                    return decoded.strip()
            except Exception:
                pass
        elif len(parts) == 1 and parts[0]:
            return parts[0]

    for header_val, header_name in (
        (x_bamboo_token, "X-Bamboo-Token"),
        (x_ci_token, "X-CI-Token"),
        (x_api_token, "X-API-Token"),
        (x_api_key, "X-Api-Key"),
    ):
        val = header_val
        if val is None and request is not None:
            val = request.headers.get(header_name)
        if val and val.strip():
            return val.strip()

    for q_val, q_name in ((token_query, "token"), (api_token_query, "api_token")):
        val = q_val
        if val is None and request is not None:
            val = request.query_params.get(q_name)
        if val and val.strip():
            return val.strip()

    return None


def verify_bamboo_token_value(provided_token: str | None, *, job: Job | None = None) -> None:
    """Validate `provided_token` against configured environment variable tokens using constant-time comparison."""
    configured_tokens = get_configured_bamboo_tokens(job)
    if not configured_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="CI/Bamboo API token is not configured on the server (set BAMBOO_API_TOKEN env var)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not provided_token or not provided_token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing CI/Bamboo authentication token. Provide 'Authorization: Bearer <token>' or 'X-Bamboo-Token' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    candidate_bytes = provided_token.strip().encode("utf-8")
    matched = any(
        hmac.compare_digest(candidate_bytes, expected.encode("utf-8"))
        for expected in configured_tokens
    )
    if not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid CI/Bamboo authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_bamboo_token(
    request: Request,
    authorization: str | None = Header(default=None),
    x_bamboo_token: str | None = Header(default=None),
    x_ci_token: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    token: str | None = Query(default=None, description="Optional CI/Bamboo API token"),
    api_token: str | None = Query(default=None, include_in_schema=False),
) -> str:
    """FastAPI dependency that enforces a valid CI/Bamboo authentication token."""
    extracted = extract_request_token(
        request,
        authorization=authorization,
        x_bamboo_token=x_bamboo_token,
        x_ci_token=x_ci_token,
        x_api_token=x_api_token,
        x_api_key=x_api_key,
        token_query=token,
        api_token_query=api_token,
    )
    verify_bamboo_token_value(extracted)
    return extracted or ""


def optional_verify_bamboo_token(
    request: Request,
    authorization: str | None = Header(default=None),
    x_bamboo_token: str | None = Header(default=None),
    x_ci_token: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    token: str | None = Query(default=None, include_in_schema=False),
    api_token: str | None = Query(default=None, include_in_schema=False),
) -> str | None:
    """If the caller supplies a CI/Bamboo token on a shared route, validate it; otherwise allow internal access."""
    extracted = extract_request_token(
        request,
        authorization=authorization,
        x_bamboo_token=x_bamboo_token,
        x_ci_token=x_ci_token,
        x_api_token=x_api_token,
        x_api_key=x_api_key,
        token_query=token,
        api_token_query=api_token,
    )
    if extracted is not None:
        verify_bamboo_token_value(extracted)
    return extracted


class BambooTriggerRequest(BaseModel):
    job_id: int | None = Field(default=None, description="Job ID to trigger (when not specified in path)")
    job_name: str | None = Field(default=None, description="Job name to trigger (when not specified in path)")
    job: int | str | None = Field(default=None, description="Job ID or name alias")
    run_mode: str | None = Field(default=None, description="Optional run_mode override ('sequential' or 'parallel')")
    overrides: dict[str, Any] | None = Field(default=None, description="Optional runtime step overrides")
    config: dict[str, Any] | None = Field(default=None, description="Optional global config override")
    suites: list[str] | None = Field(default=None, description="Optional global suites override")
    target_id: int | None = Field(default=None, description="Optional global target ID override")
    timeout: float | None = Field(default=None, description="Optional execution timeout in seconds")
    definition_ids: list[int] | None = Field(default=None, description="Optional subset/reordered definition IDs")
    prevent_overlap: bool = Field(default=False, description="If true, return 409 when a run for this job is already active")

    # Bamboo / CI build metadata
    plan_key: str | None = Field(default=None, description="Bamboo plan key (e.g. 'TES-PLAN')")
    bamboo_plan_key: str | None = Field(default=None, description="Alias for plan_key")
    build_number: int | str | None = Field(default=None, description="Bamboo build number (e.g. 42)")
    bamboo_build_number: int | str | None = Field(default=None, description="Alias for build_number")
    build_key: str | None = Field(default=None, description="Bamboo job build key (e.g. 'TES-PLAN-JOB1')")
    bamboo_build_key: str | None = Field(default=None, description="Alias for build_key")
    build_result_key: str | None = Field(default=None, description="Bamboo build result key (e.g. 'TES-PLAN-42')")
    bamboo_build_result_key: str | None = Field(default=None, description="Alias for build_result_key")
    build_url: str | None = Field(default=None, description="Bamboo build results URL")
    bamboo_results_url: str | None = Field(default=None, description="Alias for build_url")
    branch: str | None = Field(default=None, description="Git branch name")
    git_branch: str | None = Field(default=None, description="Alias for branch")
    commit: str | None = Field(default=None, description="Git commit SHA / revision")
    git_commit: str | None = Field(default=None, description="Alias for commit")
    revision: str | None = Field(default=None, description="Alias for commit")
    triggered_by: str | None = Field(default=None, description="CI caller or user identifier")

    # Optional status callback / Bamboo server report-back
    callback_url: str | None = Field(
        default=None,
        description="Optional HTTP(S) callback URL to POST run status to when execution finishes",
    )
    bamboo_base_url: str | None = Field(
        default=None,
        description="Optional Bamboo base URL for posting build comments/labels back on completion",
    )

    # Optional synchronous wait mode
    wait: bool = Field(
        default=False,
        description="If true, wait up to wait_timeout seconds for the run to complete before returning",
    )
    wait_timeout: float = Field(
        default=30.0,
        ge=0.1,
        le=600.0,
        description="Max seconds to wait when wait=true",
    )


class BambooReportRequest(BaseModel):
    callback_url: str | None = Field(
        default=None,
        description="Optional HTTP(S) callback URL override to POST run status to",
    )
    bamboo_base_url: str | None = Field(
        default=None,
        description="Optional Bamboo server base URL override",
    )
    build_result_key: str | None = Field(
        default=None,
        description="Optional Bamboo build result key (e.g. 'TES-PLAN-42') for posting build comments/labels",
    )


def _resolve_job_by_ref(db: Session, job_ref: int | str | None) -> Job:
    """Resolve a Job by integer ID or unique name."""
    if job_ref is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A job_id or job name is required to trigger a CI run",
        )

    ref_str = str(job_ref).strip()
    if not ref_str:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A job_id or job name is required to trigger a CI run",
        )

    job: Job | None = None
    if isinstance(job_ref, int) and not isinstance(job_ref, bool):
        job = db.get(Job, job_ref)
    elif ref_str.isdigit():
        job = db.get(Job, int(ref_str))

    if job is None:
        job = db.execute(select(Job).where(Job.name == ref_str)).scalars().first()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_ref}' not found",
        )
    return job


def _extract_bamboo_metadata(
    req_data: BambooTriggerRequest,
    request: Request | None = None,
) -> dict[str, Any]:
    """Extract normalized Bamboo / CI metadata from trigger request body and headers."""
    headers = request.headers if request is not None else {}

    plan_key = (
        req_data.plan_key
        or req_data.bamboo_plan_key
        or headers.get("X-Bamboo-Plan-Key")
    )
    build_number_raw = (
        req_data.build_number
        if req_data.build_number is not None
        else (
            req_data.bamboo_build_number
            if req_data.bamboo_build_number is not None
            else headers.get("X-Bamboo-Build-Number")
        )
    )
    build_number: int | str | None = None
    if build_number_raw is not None and str(build_number_raw).strip() != "":
        bn_str = str(build_number_raw).strip()
        build_number = int(bn_str) if bn_str.isdigit() else bn_str

    build_key = (
        req_data.build_key
        or req_data.bamboo_build_key
        or headers.get("X-Bamboo-Build-Key")
    )
    build_result_key = (
        req_data.build_result_key
        or req_data.bamboo_build_result_key
        or headers.get("X-Bamboo-Build-Result-Key")
    )
    if not build_result_key and plan_key and build_number is not None:
        build_result_key = f"{plan_key}-{build_number}"

    build_url = (
        req_data.build_url
        or req_data.bamboo_results_url
        or headers.get("X-Bamboo-Results-Url")
    )
    branch = (
        req_data.branch
        or req_data.git_branch
        or headers.get("X-Bamboo-Branch")
    )
    commit = (
        req_data.commit
        or req_data.git_commit
        or req_data.revision
        or headers.get("X-Bamboo-Revision")
    )
    triggered_by = (
        req_data.triggered_by
        or headers.get("X-Bamboo-Triggered-By")
        or "bamboo"
    )
    callback_url = (
        req_data.callback_url
        or headers.get("X-Bamboo-Callback-Url")
    )
    bamboo_base_url = (
        req_data.bamboo_base_url
        or headers.get("X-Bamboo-Base-Url")
    )

    if callback_url:
        parsed = urlparse(callback_url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="callback_url must be a valid http:// or https:// URL",
            )
        callback_url = callback_url.strip()

    if bamboo_base_url:
        parsed_base = urlparse(bamboo_base_url.strip())
        if parsed_base.scheme not in ("http", "https") or not parsed_base.netloc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="bamboo_base_url must be a valid http:// or https:// URL",
            )
        bamboo_base_url = bamboo_base_url.strip().rstrip("/")

    meta: dict[str, Any] = {
        "source": "bamboo",
        "triggered_by": triggered_by,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }
    if plan_key:
        meta["plan_key"] = str(plan_key).strip()
    if build_number is not None:
        meta["build_number"] = build_number
    if build_key:
        meta["build_key"] = str(build_key).strip()
    if build_result_key:
        meta["build_result_key"] = str(build_result_key).strip()
    if build_url:
        meta["build_url"] = str(build_url).strip()
    if branch:
        meta["branch"] = str(branch).strip()
    if commit:
        meta["commit"] = str(commit).strip()
    if callback_url:
        meta["callback_url"] = callback_url
    if bamboo_base_url:
        meta["bamboo_base_url"] = bamboo_base_url

    return meta


def format_bamboo_run_status(run: Run, db: Session | None = None) -> dict[str, Any]:
    """Serialize a Run into a CI/Bamboo status and result payload."""
    from backend.app.api.runs import _ensure_utc, _extract_run_job_info, _format_run_detail

    detail = _format_run_detail(run)
    job_id, job_name = _extract_run_job_info(run)

    run_status = (run.status or "queued").strip().lower()
    is_finished = run_status in _TERMINAL_RUN_STATUSES

    results = list(run.results or [])
    total_results = len(results)
    passed_results = 0
    failed_results = 0
    errored_results = 0
    cancelled_results = 0

    total_cases = 0
    passed_cases = 0
    failed_cases = 0
    errored_cases = 0
    skipped_cases = 0

    summaries: list[str] = []

    for r in results:
        r_st = (r.status or "pass").strip().lower()
        if r_st == "pass":
            passed_results += 1
        elif r_st in ("fail", "failed"):
            failed_results += 1
        elif r_st == "cancelled":
            cancelled_results += 1
        else:
            errored_results += 1

        if r.summary:
            if total_results > 1:
                summaries.append(f"{r.tester}: {r.summary}")
            else:
                summaries.append(r.summary)

        for c in r.cases or []:
            total_cases += 1
            c_st = (c.status or "pass").strip().lower()
            if c_st == "pass":
                passed_cases += 1
            elif c_st in ("fail", "failed"):
                failed_cases += 1
            elif c_st in ("skip", "skipped", "cancelled"):
                skipped_cases += 1
            else:
                errored_cases += 1

    # Compute overall result_status & Bamboo state
    if not is_finished:
        result_status = "running" if run_status == "running" else "queued"
        bamboo_state = "InProgress" if run_status == "running" else "Queued"
        passed = False
        failed = False
        exit_code: int | None = None
    elif run_status == "cancelled" or (total_results > 0 and cancelled_results > 0):
        result_status = "cancelled"
        bamboo_state = "Cancelled"
        passed = False
        failed = False
        exit_code = 2
    elif (
        run_status == "error"
        or errored_results > 0
        or errored_cases > 0
    ):
        result_status = "error" if (run_status == "error" or errored_results > 0) else "fail"
        bamboo_state = "Failed"
        passed = False
        failed = True
        exit_code = 1
    elif (
        run_status in ("fail", "failed")
        or failed_results > 0
        or failed_cases > 0
    ):
        result_status = "fail"
        bamboo_state = "Failed"
        passed = False
        failed = True
        exit_code = 1
    elif total_results == 0:
        # Completed without result records
        result_status = "pass" if run_status == "completed" else run_status
        bamboo_state = "Successful" if result_status == "pass" else "Failed"
        passed = result_status == "pass"
        failed = not passed
        exit_code = 0 if passed else 1
    else:
        result_status = "pass"
        bamboo_state = "Successful"
        passed = True
        failed = False
        exit_code = 0

    duration_seconds: float | None = None
    if run.started_at and run.finished_at:
        s_dt = _ensure_utc(run.started_at)
        f_dt = _ensure_utc(run.finished_at)
        if s_dt and f_dt:
            duration_seconds = round(max(0.0, (f_dt - s_dt).total_seconds()), 3)

    bamboo_meta: dict[str, Any] | None = None
    bamboo_report: dict[str, Any] | None = None
    if isinstance(run.config_snapshot, dict):
        if isinstance(run.config_snapshot.get("bamboo"), dict):
            bamboo_meta = dict(run.config_snapshot["bamboo"])
        if isinstance(run.config_snapshot.get("bamboo_report"), dict):
            bamboo_report = dict(run.config_snapshot["bamboo_report"])
    if bamboo_meta is None and isinstance(run.git_info, dict) and run.git_info.get("source") == "bamboo":
        bamboo_meta = dict(run.git_info)

    summary_text = " | ".join(summaries) if summaries else None
    if not summary_text and is_finished:
        summary_text = (
            f"Run #{run.id} {result_status.upper()}: "
            f"{passed_cases}/{total_cases} cases passed across {total_results} result(s)"
        )

    failed_case_items: list[dict[str, Any]] = []
    for r in results:
        for c in r.cases or []:
            c_st = (c.status or "pass").strip().lower()
            if c_st in ("fail", "failed", "error"):
                failed_case_items.append(
                    {
                        "result_id": r.id,
                        "tester": r.tester,
                        "suite": c.suite,
                        "case_id": c.case_id,
                        "status": c.status,
                        "detail": c.detail,
                    }
                )

    counts = {
        "total": total_cases,
        "passed": passed_cases,
        "failed": failed_cases,
        "errors": errored_cases,
        "errored": errored_cases,
        "skipped": skipped_cases,
        "total_results": total_results,
        "passed_results": passed_results,
        "failed_results": failed_results,
        "errored_results": errored_results,
        "cancelled_results": cancelled_results,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "errored_cases": errored_cases,
        "skipped_cases": skipped_cases,
    }

    return {
        **detail,
        "id": run.id,
        "run_id": run.id,
        "job_id": job_id,
        "job_name": job_name,
        "status": run.status,
        "run_status": run.status,
        "result_status": result_status,
        "outcome": result_status,
        "state": bamboo_state,
        "finished": is_finished,
        "is_terminal": is_finished,
        "completed": is_finished,
        "passed": passed,
        "success": passed,
        "failed": failed,
        "exit_code": exit_code,
        "summary": summary_text,
        "duration_seconds": duration_seconds,
        "counts": counts,
        "case_counts": {
            "total": total_cases,
            "passed": passed_cases,
            "failed": failed_cases,
            "errored": errored_cases,
            "skipped": skipped_cases,
        },
        "failed_cases": failed_case_items,
        "bamboo": bamboo_meta,
        "bamboo_report": bamboo_report,
        "poll_url": f"/api/bamboo/runs/{run.id}",
        "status_url": f"/api/bamboo/runs/{run.id}/status",
        "junit_url": f"/api/bamboo/runs/{run.id}/junit.xml",
        "public_junit_url": f"/api/runs/{run.id}/junit.xml",
        "run_url": f"/runs/{run.id}",
        "logs_url": f"/api/runs/{run.id}/logs",
        "artifacts_url": f"/api/runs/{run.id}/artifacts",
    }


def _load_run_with_relations(db: Session, run_id: int) -> Run:
    stmt = (
        select(Run)
        .options(
            selectinload(Run.job),
            selectinload(Run.results).selectinload(Result.cases),
            selectinload(Run.results).selectinload(Result.artifacts),
            selectinload(Run.results).selectinload(Result.test_definition),
        )
        .where(Run.id == run_id)
    )
    run = db.execute(stmt).scalars().first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )
    return run


async def _wait_for_run_terminal(
    run_id: int,
    *,
    timeout_seconds: float = 30.0,
    poll_interval: float = 0.15,
) -> None:
    """Wait asynchronously until a run reaches a terminal status or timeout_seconds elapses."""
    from backend.app.runner.worker import get_worker

    deadline = time.monotonic() + max(0.05, float(timeout_seconds))
    worker = get_worker()
    session_factory = getattr(worker, "session_factory", None) or SessionLocal

    while time.monotonic() < deadline:
        with session_factory() as check_db:
            r = check_db.get(Run, run_id)
            if r is None or (r.status or "").lower() in _TERMINAL_RUN_STATUSES:
                return
        await asyncio.sleep(min(poll_interval, max(0.01, deadline - time.monotonic())))


async def dispatch_bamboo_status_report(
    run_status_payload: dict[str, Any],
    *,
    callback_url: str | None = None,
    bamboo_base_url: str | None = None,
    build_result_key: str | None = None,
) -> dict[str, Any]:
    """Report run completion status back to a callback URL and/or Bamboo build result REST API."""
    settings = get_settings()
    bamboo_meta = run_status_payload.get("bamboo") if isinstance(run_status_payload.get("bamboo"), dict) else {}

    eff_callback_url = callback_url or bamboo_meta.get("callback_url")
    eff_bamboo_base = (
        bamboo_base_url
        or bamboo_meta.get("bamboo_base_url")
        or getattr(settings, "BAMBOO_BASE_URL", "")
    )
    if isinstance(eff_bamboo_base, str):
        eff_bamboo_base = eff_bamboo_base.strip().rstrip("/")

    eff_result_key = build_result_key or bamboo_meta.get("build_result_key")
    if not eff_result_key and bamboo_meta.get("plan_key") and bamboo_meta.get("build_number") is not None:
        eff_result_key = f"{bamboo_meta['plan_key']}-{bamboo_meta['build_number']}"

    deliveries: list[dict[str, Any]] = []
    timeout_sec = float(getattr(settings, "WEBHOOK_TIMEOUT_SECONDS", 10.0) or 10.0)

    # 1. Callback URL POST (webhook-style status callback for CI / Bamboo listener)
    if eff_callback_url:
        sent_at = datetime.now(timezone.utc).isoformat()
        report_body = {
            "event": "test_center.bamboo.run_completed",
            "run_id": run_status_payload.get("run_id"),
            "job_id": run_status_payload.get("job_id"),
            "job_name": run_status_payload.get("job_name"),
            "status": run_status_payload.get("status"),
            "result_status": run_status_payload.get("result_status"),
            "state": run_status_payload.get("state"),
            "passed": run_status_payload.get("passed"),
            "exit_code": run_status_payload.get("exit_code"),
            "summary": run_status_payload.get("summary"),
            "duration_seconds": run_status_payload.get("duration_seconds"),
            "counts": run_status_payload.get("counts"),
            "bamboo": bamboo_meta,
            "junit_url": run_status_payload.get("junit_url"),
            "public_junit_url": run_status_payload.get("public_junit_url"),
            "run_url": run_status_payload.get("run_url"),
            "sent_at": sent_at,
        }
        raw_bytes = json.dumps(report_body, sort_keys=True).encode("utf-8")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "TestCenter-BambooIntegration/1.0",
            "X-TestCenter-Run-Id": str(run_status_payload.get("run_id", "")),
            "X-TestCenter-Result-Status": str(run_status_payload.get("result_status", "")),
        }
        tokens = get_configured_bamboo_tokens()
        if tokens:
            sig = hmac.new(tokens[0].encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()
            headers["X-TestCenter-Signature-256"] = f"sha256={sig}"
            headers["X-TestCenter-Signature"] = f"sha256={sig}"
            headers["X-Test-Center-Signature"] = f"sha256={sig}"

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                resp = await client.post(eff_callback_url, content=raw_bytes, headers=headers)
            ok = 200 <= resp.status_code < 300
            deliveries.append({
                "channel": "callback_url",
                "target": eff_callback_url,
                "status": "sent" if ok else "failed",
                "ok": ok,
                "status_code": resp.status_code,
                "error": None if ok else f"HTTP {resp.status_code}",
                "sent_at": sent_at,
            })
        except Exception as exc:
            deliveries.append({
                "channel": "callback_url",
                "target": eff_callback_url,
                "status": "failed",
                "ok": False,
                "status_code": None,
                "error": str(exc),
                "sent_at": sent_at,
            })

    # 2. Bamboo Server REST API build result comment & label
    if eff_bamboo_base and eff_result_key:
        sent_at = datetime.now(timezone.utc).isoformat()
        comment_url = f"{eff_bamboo_base}/rest/api/latest/result/{eff_result_key}/comment"
        label_url = f"{eff_bamboo_base}/rest/api/latest/result/{eff_result_key}/label"

        passed = bool(run_status_payload.get("passed"))
        counts = run_status_payload.get("counts") or {}
        comment_text = (
            f"[Test Center] Run #{run_status_payload.get('run_id')} "
            f"({run_status_payload.get('job_name') or 'manual'}): "
            f"{'PASSED' if passed else 'FAILED'} "
            f"[{counts.get('passed_cases', 0)}/{counts.get('total_cases', 0)} cases passed]. "
            f"{run_status_payload.get('summary') or ''} "
            f"(JUnit XML: {run_status_payload.get('public_junit_url')})"
        ).strip()

        req_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Atlassian-Token": "no-check",
        }
        creds_env = getattr(settings, "BAMBOO_CREDS_ENV_VAR", "BAMBOO_API_TOKEN")
        auth_token = os.getenv(creds_env) or (get_configured_bamboo_tokens()[0] if get_configured_bamboo_tokens() else None)
        bamboo_user = getattr(settings, "BAMBOO_USER", "")
        if auth_token:
            if bamboo_user:
                b64 = base64.b64encode(f"{bamboo_user}:{auth_token}".encode("utf-8")).decode("ascii")
                req_headers["Authorization"] = f"Basic {b64}"
            else:
                req_headers["Authorization"] = f"Bearer {auth_token}"

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                resp_comment = await client.post(
                    comment_url,
                    content=json.dumps({"content": comment_text}).encode("utf-8"),
                    headers=req_headers,
                )
                label_name = "test-center-passed" if passed else "test-center-failed"
                await client.post(
                    label_url,
                    content=json.dumps({"name": label_name}).encode("utf-8"),
                    headers=req_headers,
                )
            ok = 200 <= resp_comment.status_code < 300
            deliveries.append({
                "channel": "bamboo_rest_api",
                "target": comment_url,
                "build_result_key": eff_result_key,
                "status": "sent" if ok else "failed",
                "ok": ok,
                "status_code": resp_comment.status_code,
                "error": None if ok else f"HTTP {resp_comment.status_code}",
                "sent_at": sent_at,
            })
        except Exception as exc:
            deliveries.append({
                "channel": "bamboo_rest_api",
                "target": comment_url,
                "build_result_key": eff_result_key,
                "status": "failed",
                "ok": False,
                "status_code": None,
                "error": str(exc),
                "sent_at": sent_at,
            })

    return {
        "run_id": run_status_payload.get("run_id"),
        "job_id": run_status_payload.get("job_id"),
        "triggered": len(deliveries) > 0,
        "reported": len(deliveries) > 0,
        "ok": any(d.get("ok") for d in deliveries) if deliveries else False,
        "deliveries": deliveries,
    }


async def process_run_bamboo_report(
    db: Session,
    run_id: int,
    *,
    callback_url: str | None = None,
    bamboo_base_url: str | None = None,
    build_result_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate and dispatch Bamboo / CI status callbacks for a finished run and persist the report."""
    run = _load_run_with_relations(db, run_id)
    status_payload = format_bamboo_run_status(run, db)

    report = await dispatch_bamboo_status_report(
        status_payload,
        callback_url=callback_url,
        bamboo_base_url=bamboo_base_url,
        build_result_key=build_result_key,
    )
    if report.get("triggered"):
        cfg = dict(run.config_snapshot) if isinstance(run.config_snapshot, dict) else {}
        cfg["bamboo_report"] = report
        run.config_snapshot = cfg
        db.commit()
        db.refresh(run)

    return report


def build_bamboo_plan_script(
    job: Job,
    *,
    base_url: str = "http://localhost:8000",
    token_env_var: str = "BAMBOO_API_TOKEN",
) -> str:
    """Generate a ready-to-use shell script for a Bamboo Script task to trigger a job, poll status, and export JUnit XML."""
    clean_base = base_url.rstrip("/")
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f'TEST_CENTER_URL="${{TEST_CENTER_URL:-{clean_base}}}"\n'
        f'API_TOKEN="${{{token_env_var}:?Set {token_env_var} in Bamboo plan variables}}"\n'
        f'JOB_ID="{job.id}"\n\n'
        "# 1. Trigger job run in Test Center\n"
        'TRIGGER_RESP=$(curl -fsSL -X POST "${TEST_CENTER_URL}/api/bamboo/jobs/${JOB_ID}/trigger" \\\n'
        '  -H "Authorization: Bearer ${API_TOKEN}" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d "{\\"plan_key\\": \\"${bamboo_planKey:-}\\", \\"build_number\\": \\"${bamboo_buildNumber:-}\\", \\"build_result_key\\": \\"${bamboo_buildResultKey:-}\\", \\"branch\\": \\"${bamboo_planRepository_branchName:-}\\", \\"commit\\": \\"${bamboo_planRepository_revision:-}\\"}")\n\n'
        'RUN_ID=$(printf "%s" "${TRIGGER_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)[\'run_id\'])")\n'
        'echo "Triggered Test Center Run #${RUN_ID} for Job \'${JOB_ID}\'"\n\n'
        "# 2. Poll run status until finished\n"
        "while true; do\n"
        '  STATUS_RESP=$(curl -fsSL "${TEST_CENTER_URL}/api/bamboo/runs/${RUN_ID}" \\\n'
        '    -H "Authorization: Bearer ${API_TOKEN}")\n'
        '  FINISHED=$(printf "%s" "${STATUS_RESP}" | python3 -c "import sys,json; print(str(json.load(sys.stdin)[\'finished\']).lower())")\n'
        '  STATE=$(printf "%s" "${STATUS_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)[\'state\'])")\n'
        '  echo "Run #${RUN_ID} state: ${STATE}"\n'
        '  if [ "${FINISHED}" = "true" ]; then\n'
        "    break\n"
        "  fi\n"
        "  sleep 2\n"
        "done\n\n"
        "# 3. Download JUnit XML report for Bamboo's JUnit Parser task\n"
        "mkdir -p test-reports\n"
        'curl -fsSL "${TEST_CENTER_URL}/api/bamboo/runs/${RUN_ID}/junit.xml" \\\n'
        '  -H "Authorization: Bearer ${API_TOKEN}" \\\n'
        '  -o "test-reports/test-center-run-${RUN_ID}.xml"\n'
        'echo "Saved JUnit XML to test-reports/test-center-run-${RUN_ID}.xml"\n\n'
        "# 4. Exit with run's exit code (0 = pass, non-zero = fail/error)\n"
        'EXIT_CODE=$(printf "%s" "${STATUS_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\'exit_code\') or 0)")\n'
        'exit "${EXIT_CODE}"\n'
    )


async def _trigger_bamboo_job_impl(
    job: Job,
    req_data: BambooTriggerRequest,
    request: Request,
    response: Response,
    db: Session,
) -> dict[str, Any]:
    """Shared implementation for triggering a job via the Bamboo / CI API."""
    from backend.app.api.jobs import TriggerJobRunRequest, trigger_job_run

    bamboo_meta = _extract_bamboo_metadata(req_data, request)

    trigger_req = TriggerJobRunRequest(
        run_mode=req_data.run_mode,
        overrides=req_data.overrides,
        config=req_data.config,
        suites=req_data.suites,
        target_id=req_data.target_id,
        timeout=req_data.timeout,
        definition_ids=req_data.definition_ids,
        prevent_overlap=req_data.prevent_overlap,
    )
    created_detail = trigger_job_run(job_id=job.id, req=trigger_req, db=db)
    run_id = created_detail["id"]

    # Attach Bamboo / CI metadata to the created Run
    run = db.get(Run, run_id)
    if run is not None:
        cfg_snap = dict(run.config_snapshot) if isinstance(run.config_snapshot, dict) else {}
        cfg_snap["bamboo"] = bamboo_meta
        run.config_snapshot = cfg_snap

        git_info = dict(run.git_info) if isinstance(run.git_info, dict) else {}
        git_info.update(bamboo_meta)
        run.git_info = git_info

        db.commit()
        db.refresh(run)

    if req_data.wait:
        await _wait_for_run_terminal(run_id, timeout_seconds=req_data.wait_timeout)
        db.expire_all()

    run_loaded = _load_run_with_relations(db, run_id)
    payload = format_bamboo_run_status(run_loaded, db)
    if req_data.wait and payload["finished"]:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_201_CREATED
    return payload


@router.get("/info")
@ci_alias_router.get("/info", include_in_schema=False)
def get_bamboo_integration_info() -> dict[str, Any]:
    """Return non-sensitive Bamboo / CI integration configuration metadata for the UI."""
    settings = get_settings()
    token_env_var = getattr(settings, "BAMBOO_TOKEN_ENV_VAR", "BAMBOO_API_TOKEN")
    return {
        "enabled": True,
        "token_configured": is_bamboo_token_configured(),
        "token_env_var": token_env_var,
        "candidate_token_env_vars": list(_CANDIDATE_TOKEN_ENV_VARS),
        "bamboo_base_url": getattr(settings, "BAMBOO_BASE_URL", "") or None,
        "endpoints": {
            "trigger_job": "/api/bamboo/jobs/{job_id}/trigger",
            "trigger_generic": "/api/bamboo/trigger",
            "poll_run": "/api/bamboo/runs/{run_id}",
            "poll_run_status": "/api/bamboo/runs/{run_id}/status",
            "junit_xml": "/api/bamboo/runs/{run_id}/junit.xml",
            "public_junit_xml": "/api/runs/{run_id}/junit.xml",
            "report_status": "/api/bamboo/runs/{run_id}/report",
        },
    }


@router.post("/trigger", status_code=status.HTTP_201_CREATED)
@ci_alias_router.post("/trigger", status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def trigger_bamboo_run_from_body(
    request: Request,
    response: Response,
    req: BambooTriggerRequest | None = None,
    job_id: int | None = Query(default=None, description="Optional job_id query parameter"),
    job: str | None = Query(default=None, description="Optional job name or ID query parameter"),
    _token: str = Depends(require_bamboo_token),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Token-protected endpoint to trigger a job run by passing `job_id` or `job_name` in body or query."""
    req_data = req or BambooTriggerRequest()
    job_ref: int | str | None = (
        req_data.job_id
        if req_data.job_id is not None
        else (
            req_data.job_name
            if req_data.job_name is not None
            else (
                req_data.job
                if req_data.job is not None
                else (job_id if job_id is not None else job)
            )
        )
    )
    job_obj = _resolve_job_by_ref(db, job_ref)
    verify_bamboo_token_value(_token, job=job_obj)
    return await _trigger_bamboo_job_impl(job_obj, req_data, request, response, db)


@router.post("/jobs/{job_ref}/trigger", status_code=status.HTTP_201_CREATED)
@router.post("/jobs/{job_ref}/run", status_code=status.HTTP_201_CREATED)
@ci_alias_router.post("/jobs/{job_ref}/trigger", status_code=status.HTTP_201_CREATED, include_in_schema=False)
@ci_alias_router.post("/jobs/{job_ref}/run", status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def trigger_bamboo_job_run(
    job_ref: str,
    request: Request,
    response: Response,
    req: BambooTriggerRequest | None = None,
    wait: bool | None = Query(default=None, description="If true, wait for run completion before returning"),
    wait_timeout: float | None = Query(default=None, ge=0.1, le=600.0, description="Max seconds to wait when wait=true"),
    _token: str = Depends(require_bamboo_token),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Token-protected endpoint for a Bamboo plan step to trigger a job (by ID or name) and obtain its Run ID and poll/JUnit URLs."""
    job_obj = _resolve_job_by_ref(db, job_ref)
    verify_bamboo_token_value(_token, job=job_obj)
    req_data = req or BambooTriggerRequest()
    if wait is not None:
        req_data.wait = wait
    if wait_timeout is not None:
        req_data.wait_timeout = wait_timeout
    return await _trigger_bamboo_job_impl(job_obj, req_data, request, response, db)


@router.get("/runs/{run_id}")
@router.get("/runs/{run_id}/status")
@router.get("/jobs/{job_ref}/runs/{run_id}", include_in_schema=False)
@ci_alias_router.get("/runs/{run_id}", include_in_schema=False)
@ci_alias_router.get("/runs/{run_id}/status", include_in_schema=False)
@ci_alias_router.get("/jobs/{job_ref}/runs/{run_id}", include_in_schema=False)
async def poll_bamboo_run_status(
    run_id: int,
    job_ref: str | None = None,
    wait: bool = Query(default=False, description="If true, wait until run finishes or timeout expires"),
    timeout: float = Query(default=30.0, ge=0.1, le=600.0, description="Max seconds to wait when wait=true"),
    poll_interval: float = Query(default=0.15, ge=0.05, le=5.0, description="Poll interval in seconds when wait=true"),
    _token: str = Depends(require_bamboo_token),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Token-protected endpoint to poll a triggered run's execution status, pass/fail outcome, exit code, and summary."""
    run = _load_run_with_relations(db, run_id)
    if job_ref is not None:
        job_obj = _resolve_job_by_ref(db, job_ref)
        if run.job_id != job_obj.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run {run_id} does not belong to Job '{job_ref}'",
            )

    if wait and (run.status or "").lower() not in _TERMINAL_RUN_STATUSES:
        await _wait_for_run_terminal(
            run_id,
            timeout_seconds=timeout,
            poll_interval=poll_interval,
        )
        db.expire_all()
        run = _load_run_with_relations(db, run_id)

    return format_bamboo_run_status(run, db)


def build_junit_xml_http_response(
    xml_content: str,
    *,
    filename: str,
    download: bool = False,
) -> Response:
    """Create an HTTP Response serving a JUnit XML document."""
    disposition = "attachment" if download else "inline"
    return Response(
        content=xml_content,
        media_type="application/xml",
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/runs/{run_id}/junit.xml")
@router.get("/runs/{run_id}/junit")
@router.get("/jobs/{job_ref}/runs/{run_id}/junit.xml", include_in_schema=False)
@ci_alias_router.get("/runs/{run_id}/junit.xml", include_in_schema=False)
@ci_alias_router.get("/runs/{run_id}/junit", include_in_schema=False)
@ci_alias_router.get("/jobs/{job_ref}/runs/{run_id}/junit.xml", include_in_schema=False)
def get_bamboo_run_junit_xml(
    run_id: int,
    job_ref: str | None = None,
    suite: str | None = Query(default=None, description="Optional filter by suite name"),
    tester: str | None = Query(default=None, description="Optional filter by tester plugin name"),
    result_id: int | None = Query(default=None, description="Optional filter by Result ID"),
    download: bool = Query(default=False, description="If true, force attachment download"),
    _token: str = Depends(require_bamboo_token),
    db: Session = Depends(get_db),
) -> Response:
    """Token-protected JUnit XML export of a run's cases for Bamboo's JUnit Parser task."""
    run = _load_run_with_relations(db, run_id)
    if job_ref is not None:
        job_obj = _resolve_job_by_ref(db, job_ref)
        if run.job_id != job_obj.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run {run_id} does not belong to Job '{job_ref}'",
            )

    xml_str = build_run_junit_xml(
        run,
        suite_filter=suite,
        tester_filter=tester,
        result_id_filter=result_id,
    )
    return build_junit_xml_http_response(
        xml_str,
        filename=f"run-{run_id}-junit.xml",
        download=download,
    )


@router.post("/runs/{run_id}/report")
@ci_alias_router.post("/runs/{run_id}/report", include_in_schema=False)
async def report_bamboo_run_status(
    run_id: int,
    req: BambooReportRequest | None = None,
    _token: str = Depends(require_bamboo_token),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Token-protected endpoint to report a run's status back to Bamboo or a callback URL."""
    body = req or BambooReportRequest()
    return await process_run_bamboo_report(
        db,
        run_id,
        callback_url=body.callback_url,
        bamboo_base_url=body.bamboo_base_url,
        build_result_key=body.build_result_key,
    )
