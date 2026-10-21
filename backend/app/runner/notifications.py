"""Per-job notification evaluation and delivery (webhook and optional email).

Triggers on:
1. Failed run (`on_failure`): run or any result/case has status 'fail', 'failed', or 'error'.
2. Regression vs previous run (`on_regression`): reuses `compare_runs_and_results` from
   `backend.app.api.compare` to compare the completed run against the previous finished run
   of the same job.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.message import EmailMessage
import hashlib
import hmac
import json
import logging
import os
import smtplib
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.config import get_settings
from backend.app.db.base import utc_now
from backend.app.db.models.job import Job
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run

def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_iso_utc(dt: datetime | None) -> str | None:
    """Format a datetime as an ISO-8601 string with UTC timezone information."""
    if dt is None:
        return None
    utc_dt = _ensure_utc(dt)
    return utc_dt.isoformat() if utc_dt is not None else None


logger = logging.getLogger(__name__)

_SECRET_FIELD_NAMES = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "password",
    "token",
    "access_token",
    "auth_token",
    "bearer_token",
    "private_key",
    "_transport",
}

TERMINAL_RUN_STATUSES_FOR_BASELINE = ("completed", "pass", "fail", "failed", "error")
FAILURE_STATUSES = ("fail", "failed", "error")


def default_notification_settings() -> dict[str, Any]:
    """Return default per-job notification settings."""
    return {
        "enabled": False,
        "on_failure": True,
        "on_regression": True,
        "regression_threshold": 0.0,
        "metric_thresholds": {},
        "webhook_enabled": False,
        "webhook_url": None,
        "webhook_secret_env_var": None,
        "webhook_headers": {},
        "email_enabled": False,
        "email_to": [],
        "email_subject_prefix": None,
    }


def _parse_email_list(raw: Any) -> list[str]:
    """Parse email recipients from a list of strings or a comma/semicolon-separated string."""
    if raw is None:
        return []
    items: list[str] = []
    if isinstance(raw, str):
        for part in raw.replace(";", ",").split(","):
            addr = part.strip()
            if addr:
                items.append(addr)
    elif isinstance(raw, (list, tuple, set)):
        for entry in raw:
            if isinstance(entry, str):
                for part in entry.replace(";", ",").split(","):
                    addr = part.strip()
                    if addr:
                        items.append(addr)
            elif entry is not None:
                raise ValueError(f"Invalid email recipient entry: {entry!r}")
    else:
        raise ValueError("email_to must be a list of email strings or a comma-separated string")

    deduped: list[str] = []
    for addr in items:
        if "@" not in addr or any(ch.isspace() for ch in addr):
            raise ValueError(f"Invalid email address: {addr!r}")
        if addr not in deduped:
            deduped.append(addr)
    return deduped


def normalize_notification_settings(
    raw: dict[str, Any] | None = None,
    *,
    base: dict[str, Any] | None = None,
    webhook_url: str | None = None,
    notify_on_failure: bool | None = None,
    notify_on_regression: bool | None = None,
    regression_threshold: float | None = None,
    email_to: list[str] | str | None = None,
) -> dict[str, Any]:
    """Normalize and validate per-job notification settings from flexible input formats."""
    current = default_notification_settings()
    if isinstance(base, dict) and base:
        # Merge previously stored settings first
        for k in current:
            if k in base and base[k] is not None:
                current[k] = base[k]

    data: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

    # Support nested 'webhook' and 'email' dicts
    webhook_nested = data.get("webhook") if isinstance(data.get("webhook"), dict) else {}
    email_nested = data.get("email") if isinstance(data.get("email"), dict) else {}

    # 1. Resolve webhook_url
    raw_webhook_url = (
        webhook_url
        if webhook_url is not None
        else (
            data.get("webhook_url")
            if "webhook_url" in data
            else (
                data.get("url")
                if "url" in data
                else webhook_nested.get("url")
            )
        )
    )
    has_webhook_url_input = (
        webhook_url is not None
        or "webhook_url" in data
        or "url" in data
        or "url" in webhook_nested
    )
    if has_webhook_url_input:
        if raw_webhook_url is None or (isinstance(raw_webhook_url, str) and not raw_webhook_url.strip()):
            current["webhook_url"] = None
        elif isinstance(raw_webhook_url, str):
            cleaned_url = raw_webhook_url.strip()
            if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
                raise ValueError("webhook_url must start with http:// or https://")
            current["webhook_url"] = cleaned_url
        else:
            raise ValueError("webhook_url must be a valid HTTP/HTTPS URL string")

    # 2. Resolve webhook_secret_env_var
    if "webhook_secret_env_var" in data or "secret_env_var" in data or "secret_env_var" in webhook_nested:
        raw_sec_env = (
            data.get("webhook_secret_env_var")
            if "webhook_secret_env_var" in data
            else (data.get("secret_env_var") or webhook_nested.get("secret_env_var"))
        )
        if raw_sec_env is None or (isinstance(raw_sec_env, str) and not raw_sec_env.strip()):
            current["webhook_secret_env_var"] = None
        elif isinstance(raw_sec_env, str):
            current["webhook_secret_env_var"] = raw_sec_env.strip()
        else:
            raise ValueError("webhook_secret_env_var must be a string")

    # 3. Resolve webhook_headers
    raw_headers = (
        data.get("webhook_headers")
        if "webhook_headers" in data
        else (data.get("headers") if "headers" in data else webhook_nested.get("headers"))
    )
    if raw_headers is not None:
        if not isinstance(raw_headers, dict):
            raise ValueError("webhook_headers must be a dictionary of string headers")
        sanitized_headers: dict[str, str] = {}
        for hk, hv in raw_headers.items():
            if str(hk).lower() in _SECRET_FIELD_NAMES:
                continue
            if hv is not None:
                sanitized_headers[str(hk).strip()] = str(hv)
        current["webhook_headers"] = sanitized_headers

    # 4. Resolve email_to
    has_email_input = (
        email_to is not None
        or any(k in data for k in ("email_to", "emails", "recipients", "email_recipients", "to"))
        or any(k in email_nested for k in ("to", "recipients", "emails"))
    )
    if has_email_input:
        raw_emails = (
            email_to
            if email_to is not None
            else (
                data.get("email_to")
                if "email_to" in data
                else (
                    data.get("emails")
                    if "emails" in data
                    else (
                        data.get("recipients")
                        if "recipients" in data
                        else (
                            data.get("email_recipients")
                            if "email_recipients" in data
                            else (
                                data.get("to")
                                if "to" in data
                                else (
                                    email_nested.get("to")
                                    or email_nested.get("recipients")
                                    or email_nested.get("emails")
                                )
                            )
                        )
                    )
                )
            )
        )
        current["email_to"] = _parse_email_list(raw_emails)

    # 5. Resolve email_subject_prefix
    if (
        "email_subject_prefix" in data
        or "subject_prefix" in data
        or "subject_prefix" in email_nested
    ):
        raw_prefix = (
            data.get("email_subject_prefix")
            if "email_subject_prefix" in data
            else (data.get("subject_prefix") or email_nested.get("subject_prefix"))
        )
        current["email_subject_prefix"] = (
            str(raw_prefix).strip() if raw_prefix is not None and str(raw_prefix).strip() else None
        )

    # 6. Resolve trigger flags: on_failure and on_regression
    triggers_list = data.get("triggers") if "triggers" in data else data.get("events")
    if isinstance(triggers_list, list):
        norm_triggers = {str(t).strip().lower() for t in triggers_list}
        current["on_failure"] = bool(norm_triggers & {"failure", "fail", "failed", "error"})
        current["on_regression"] = "regression" in norm_triggers

    for key in ("on_failure", "notify_on_failure", "trigger_on_failure", "on_fail"):
        if key in data and data[key] is not None:
            current["on_failure"] = bool(data[key])
    if notify_on_failure is not None:
        current["on_failure"] = bool(notify_on_failure)

    for key in ("on_regression", "notify_on_regression", "trigger_on_regression"):
        if key in data and data[key] is not None:
            current["on_regression"] = bool(data[key])
    if notify_on_regression is not None:
        current["on_regression"] = bool(notify_on_regression)

    # 7. Resolve regression_threshold and metric_thresholds
    raw_threshold = (
        regression_threshold
        if regression_threshold is not None
        else (
            data.get("regression_threshold")
            if "regression_threshold" in data
            else (
                data.get("threshold_pct")
                if "threshold_pct" in data
                else data.get("threshold")
            )
        )
    )
    if raw_threshold is not None:
        try:
            thr_val = float(raw_threshold)
        except (TypeError, ValueError) as e:
            raise ValueError("regression_threshold must be a non-negative number") from e
        if thr_val < 0.0:
            raise ValueError("regression_threshold must be >= 0")
        current["regression_threshold"] = thr_val

    if "metric_thresholds" in data and data["metric_thresholds"] is not None:
        raw_mt = data["metric_thresholds"]
        if not isinstance(raw_mt, dict):
            raise ValueError("metric_thresholds must be a dictionary of {metric_name: float}")
        parsed_mt: dict[str, float] = {}
        for mk, mv in raw_mt.items():
            if mv is None:
                continue
            val_f = float(mv)
            if val_f < 0.0:
                raise ValueError(f"Threshold for metric '{mk}' must be >= 0")
            parsed_mt[str(mk)] = val_f
        current["metric_thresholds"] = parsed_mt

    # 8. Resolve channel enablement & master enabled flag
    if "webhook_enabled" in data and data["webhook_enabled"] is not None:
        current["webhook_enabled"] = bool(data["webhook_enabled"])
    elif "enabled" in webhook_nested and webhook_nested["enabled"] is not None:
        current["webhook_enabled"] = bool(webhook_nested["enabled"])
    elif has_webhook_url_input:
        current["webhook_enabled"] = bool(current["webhook_url"])

    if "email_enabled" in data and data["email_enabled"] is not None:
        current["email_enabled"] = bool(data["email_enabled"])
    elif "enabled" in email_nested and email_nested["enabled"] is not None:
        current["email_enabled"] = bool(email_nested["enabled"])
    elif has_email_input:
        current["email_enabled"] = bool(current["email_to"])

    if "enabled" in data and data["enabled"] is not None:
        current["enabled"] = bool(data["enabled"])
    elif has_webhook_url_input or has_email_input:
        current["enabled"] = bool(current["webhook_url"]) or bool(current["email_to"])

    return current


def find_previous_job_run(db: Session, job_id: int, current_run_id: int) -> Run | None:
    """Find the most recent finished Run for the same job prior to `current_run_id`."""
    stmt_prior = (
        select(Run)
        .options(
            selectinload(Run.results).selectinload(Result.cases),
            selectinload(Run.results).selectinload(Result.artifacts),
        )
        .where(
            Run.job_id == job_id,
            Run.id < current_run_id,
            Run.status.in_(TERMINAL_RUN_STATUSES_FOR_BASELINE),
        )
        .order_by(Run.id.desc())
    )
    prior_runs = db.execute(stmt_prior).scalars().all()
    for candidate in prior_runs:
        if candidate.results:
            return candidate

    # Fallback: check any other finished run of the job if IDs were not strictly ordered
    stmt_other = (
        select(Run)
        .options(
            selectinload(Run.results).selectinload(Result.cases),
            selectinload(Run.results).selectinload(Result.artifacts),
        )
        .where(
            Run.job_id == job_id,
            Run.id != current_run_id,
            Run.status.in_(TERMINAL_RUN_STATUSES_FOR_BASELINE),
        )
        .order_by(Run.finished_at.desc().nulls_last(), Run.id.desc())
    )
    for candidate in db.execute(stmt_other).scalars().all():
        if candidate.results:
            return candidate

    return None


def _pair_results_for_comparison(
    baseline_results: list[Result],
    current_results: list[Result],
) -> list[tuple[Result, Result]]:
    """Pair results from previous run and current run by test_definition_id or tester."""
    pairs: list[tuple[Result, Result]] = []
    used_base_ids: set[int] = set()

    # Pass 1: Match by exact test_definition_id + tester
    for curr in current_results:
        if curr.test_definition_id is None:
            continue
        matched = next(
            (
                b
                for b in baseline_results
                if b.id not in used_base_ids
                and b.test_definition_id == curr.test_definition_id
                and b.tester == curr.tester
            ),
            None,
        )
        if matched is not None:
            used_base_ids.add(matched.id)
            pairs.append((matched, curr))

    paired_curr_ids = {curr.id for _, curr in pairs}

    # Pass 2: Match remaining results by tester name in order
    for curr in current_results:
        if curr.id in paired_curr_ids:
            continue
        matched = next(
            (
                b
                for b in baseline_results
                if b.id not in used_base_ids and b.tester == curr.tester
            ),
            None,
        )
        if matched is not None:
            used_base_ids.add(matched.id)
            paired_curr_ids.add(curr.id)
            pairs.append((matched, curr))

    return pairs


def evaluate_run_for_notifications(
    db: Session,
    run: Run,
    job: Job | None = None,
    settings_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a completed Run for failure and regression vs previous run of the same Job."""
    if job is None and run.job_id is not None:
        job = db.get(Job, run.job_id)

    raw_settings: dict[str, Any] = {}
    if settings_override is not None:
        raw_settings = settings_override
    elif job is not None and isinstance(job.notifications, dict) and job.notifications:
        raw_settings = job.notifications
    elif isinstance(run.config_snapshot, dict) and isinstance(run.config_snapshot.get("notifications"), dict):
        raw_settings = run.config_snapshot["notifications"]

    settings = normalize_notification_settings(raw_settings)

    # 1. Evaluate Failure condition
    is_cancelled = run.status == "cancelled"
    failed_results: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []

    for res in run.results or []:
        if res.status in FAILURE_STATUSES:
            failed_results.append({
                "result_id": res.id,
                "test_definition_id": res.test_definition_id,
                "tester": res.tester,
                "status": res.status,
                "summary": res.summary,
            })
        for case in res.cases or []:
            if case.status in FAILURE_STATUSES:
                failed_cases.append({
                    "result_id": res.id,
                    "tester": res.tester,
                    "suite": case.suite,
                    "case_id": case.case_id,
                    "status": case.status,
                })

    has_failure = (not is_cancelled) and (
        run.status in FAILURE_STATUSES
        or len(failed_results) > 0
        or len(failed_cases) > 0
    )

    failure_summary_parts: list[str] = []
    if failed_results:
        for fr in failed_results:
            failure_summary_parts.append(
                f"{fr['tester']} ({fr['status']}): {fr.get('summary') or 'failed'}"
            )
    elif run.status in FAILURE_STATUSES:
        failure_summary_parts.append(f"Run status: {run.status}")
    if failed_cases:
        failure_summary_parts.append(f"{len(failed_cases)} case(s) failed")

    failure_info: dict[str, Any] = {
        "failed": has_failure,
        "run_status": run.status,
        "failed_results": failed_results,
        "failed_cases": failed_cases,
        "summary": "; ".join(failure_summary_parts) if failure_summary_parts else None,
    }

    # 2. Evaluate Regression vs previous run (reusing compare_runs_and_results)
    job_id = job.id if job is not None else run.job_id
    regression_threshold = float(settings.get("regression_threshold", 0.0))
    metric_thresholds = settings.get("metric_thresholds") or {}

    prev_run: Run | None = None
    if job_id is not None and not is_cancelled:
        prev_run = find_previous_job_run(db, job_id, run.id)

    has_regression = False
    compared = False
    total_metrics_regressed = 0
    total_cases_regressed = 0
    all_regressions: list[dict[str, Any]] = []
    step_comparisons: list[dict[str, Any]] = []

    if prev_run is not None and prev_run.results and run.results:
        from backend.app.api.compare import compare_runs_and_results

        pairs = _pair_results_for_comparison(list(prev_run.results), list(run.results))
        for prev_res, curr_res in pairs:
            try:
                cmp_data = compare_runs_and_results(
                    db=db,
                    result_ids=[prev_res.id, curr_res.id],
                    baseline_result_id=prev_res.id,
                    threshold=regression_threshold,
                    metric_thresholds=metric_thresholds,
                    allow_different_testers=True,
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to compare result #{prev_res.id} vs #{curr_res.id} for run #{run.id}: {exc}"
                )
                continue

            compared = True
            cmp_summary = cmp_data.get("summary") or {}
            step_has_reg = bool(cmp_summary.get("has_regression"))
            step_metrics_reg = int(cmp_summary.get("metrics_regressed", 0))
            step_cases_reg = int(cmp_summary.get("cases_regressed", 0))
            step_regs = list(cmp_summary.get("regressions") or [])

            # Also include per-case metric regressions in detailed list if not already present
            for case_row in cmp_data.get("cases") or []:
                if case_row.get("has_metric_regression"):
                    for m_row in case_row.get("metrics") or []:
                        if m_row.get("has_regression"):
                            for cell in m_row.get("comparisons") or []:
                                if cell.get("regression"):
                                    step_regs.append({
                                        "type": "case_metric",
                                        "suite": case_row.get("suite"),
                                        "case_id": case_row.get("case_id"),
                                        "name": m_row.get("name"),
                                        "unit": m_row.get("unit"),
                                        "better": m_row.get("better"),
                                        "baseline_run_id": prev_run.id,
                                        "baseline_result_id": prev_res.id,
                                        "baseline_value": m_row.get("baseline_value"),
                                        "run_id": cell.get("run_id"),
                                        "result_id": cell.get("result_id"),
                                        "value": cell.get("value"),
                                        "abs_delta": cell.get("abs_delta"),
                                        "pct_delta": cell.get("pct_delta"),
                                    })

            # Also check if top-level result status regressed from pass -> fail/error
            if prev_res.status == "pass" and curr_res.status in FAILURE_STATUSES:
                step_has_reg = True
                if not any(r.get("type") in ("case_status", "result_status") for r in step_regs):
                    step_regs.append({
                        "type": "result_status",
                        "tester": curr_res.tester,
                        "baseline_run_id": prev_run.id,
                        "baseline_result_id": prev_res.id,
                        "baseline_status": prev_res.status,
                        "run_id": run.id,
                        "result_id": curr_res.id,
                        "status": curr_res.status,
                    })

            if step_has_reg:
                has_regression = True
            total_metrics_regressed += step_metrics_reg
            total_cases_regressed += step_cases_reg
            all_regressions.extend(step_regs)
            step_comparisons.append({
                "tester": curr_res.tester,
                "test_definition_id": curr_res.test_definition_id,
                "baseline_run_id": prev_run.id,
                "baseline_result_id": prev_res.id,
                "current_run_id": run.id,
                "current_result_id": curr_res.id,
                "has_regression": step_has_reg,
                "summary": cmp_summary,
            })

    compare_url = (
        f"/compare?runs={prev_run.id},{run.id}&baseline={prev_run.id}"
        if prev_run is not None
        else None
    )

    regression_info: dict[str, Any] = {
        "has_regression": has_regression,
        "compared": compared,
        "baseline_run_id": prev_run.id if prev_run is not None else None,
        "current_run_id": run.id,
        "threshold": regression_threshold,
        "metrics_regressed": total_metrics_regressed,
        "cases_regressed": total_cases_regressed,
        "regressions": all_regressions,
        "compare_url": compare_url,
        "comparisons": step_comparisons,
    }

    reasons: list[str] = []
    if settings.get("on_failure", True) and has_failure:
        reasons.append("failure")
    if settings.get("on_regression", True) and has_regression:
        reasons.append("regression")

    has_active_channel = bool(
        (settings.get("webhook_enabled") and settings.get("webhook_url"))
        or (settings.get("email_enabled") and settings.get("email_to"))
    )
    should_notify = (
        bool(settings.get("enabled"))
        and not is_cancelled
        and len(reasons) > 0
        and has_active_channel
    )

    return {
        "should_notify": should_notify,
        "reasons": reasons,
        "failed": has_failure,
        "regressed": has_regression,
        "settings": settings,
        "failure": failure_info,
        "regression": regression_info,
        "baseline_run_id": prev_run.id if prev_run is not None else None,
    }


def build_notification_payload(
    run: Run | None,
    job: Job | None,
    evaluation: dict[str, Any],
    *,
    is_test: bool = False,
) -> dict[str, Any]:
    """Construct the structured notification payload for webhook and email channels."""
    now_iso = utc_now().isoformat()
    reasons = list(evaluation.get("reasons") or (["test"] if is_test else []))
    failure_info = evaluation.get("failure") or {}
    regression_info = evaluation.get("regression") or {}

    job_id = (
        job.id
        if job is not None
        else (
            run.job_id
            if run is not None
            else None
        )
    )
    job_name = (
        job.name
        if job is not None
        else (
            run.config_snapshot.get("job_name")
            if (run is not None and isinstance(run.config_snapshot, dict))
            else f"Job #{job_id}"
        )
    )

    run_id = run.id if run is not None else None
    baseline_run_id = regression_info.get("baseline_run_id")

    results_summary: list[dict[str, Any]] = []
    overall_result_status = "pass"
    if run is not None:
        from backend.app.api.compare import _sanitize_dict

        for r in run.results or []:
            if r.status == "error":
                overall_result_status = "error"
            elif r.status in ("fail", "failed") and overall_result_status != "error":
                overall_result_status = "fail"
            results_summary.append({
                "id": r.id,
                "result_id": r.id,
                "test_definition_id": r.test_definition_id,
                "tester": r.tester,
                "status": r.status,
                "summary": r.summary,
                "metrics": _sanitize_dict(r.metrics or {}),
            })
        if not run.results and run.status in FAILURE_STATUSES:
            overall_result_status = run.status

    reason_str = ", ".join(reasons) if reasons else "notification"
    if is_test:
        headline = f"[Test Center] Test notification for job '{job_name}'"
    else:
        headline = (
            f"[Test Center] Job '{job_name}' (Run #{run_id}) triggered alert: {reason_str}"
        )

    detail_lines: list[str] = [headline]
    if failure_info.get("failed") and failure_info.get("summary"):
        detail_lines.append(f"Failure: {failure_info['summary']}")
    if regression_info.get("has_regression"):
        regs = regression_info.get("regressions") or []
        reg_snippets: list[str] = []
        for reg in regs[:5]:
            rtype = reg.get("type")
            if rtype in ("metric", "case_metric"):
                pct = reg.get("pct_delta")
                pct_str = f" ({pct:+.2f}%)" if isinstance(pct, (int, float)) else ""
                reg_snippets.append(
                    f"{reg.get('name')}: {reg.get('baseline_value')} -> {reg.get('value')}{pct_str}"
                )
            elif rtype in ("case_status", "result_status"):
                label = reg.get("case_id") or reg.get("tester") or "status"
                reg_snippets.append(
                    f"{label}: {reg.get('baseline_status')} -> {reg.get('status')}"
                )
        reg_desc = "; ".join(reg_snippets) if reg_snippets else f"{len(regs)} regression(s)"
        detail_lines.append(
            f"Regression vs Run #{baseline_run_id}: {reg_desc}"
        )

    full_text = "\n".join(detail_lines)

    return {
        "event": "job_notification_test" if is_test else "job_notification",
        "is_test": is_test,
        "timestamp": now_iso,
        "reasons": reasons,
        "triggered_by": reasons,
        "failed": bool(failure_info.get("failed")),
        "has_failure": bool(failure_info.get("failed")),
        "regressed": bool(regression_info.get("has_regression")),
        "has_regression": bool(regression_info.get("has_regression")),
        "job_id": job_id,
        "job_name": job_name,
        "run_id": run_id,
        "run_status": run.status if run is not None else "completed",
        "result_status": overall_result_status,
        "baseline_run_id": baseline_run_id,
        "text": full_text,
        "message": headline,
        "summary": full_text,
        "regressions": regression_info.get("regressions") or [],
        "job": {
            "id": job_id,
            "name": job_name,
            "schedule": job.schedule if job is not None else "manual",
            "run_mode": (job.run_mode or "sequential") if job is not None else "sequential",
        },
        "run": (
            {
                "id": run.id,
                "job_id": job_id,
                "status": run.status,
                "result_status": overall_result_status,
                "queued_at": _format_iso_utc(run.queued_at),
                "started_at": _format_iso_utc(run.started_at),
                "finished_at": _format_iso_utc(run.finished_at),
                "url": f"/runs/{run.id}",
                "results": results_summary,
            }
            if run is not None
            else None
        ),
        "failure": failure_info,
        "regression": regression_info,
        "compare_url": regression_info.get("compare_url"),
    }


async def send_webhook_notification(
    webhook_url: str,
    payload: dict[str, Any],
    *,
    secret_env_var: str | None = None,
    custom_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Deliver a JSON notification payload to `webhook_url` via HTTP POST."""
    settings = get_settings()
    req_timeout = float(timeout if timeout is not None else settings.WEBHOOK_TIMEOUT_SECONDS)
    body_bytes = json.dumps(payload, default=str).encode("utf-8")

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": f"TestCenter-Webhook/{settings.VERSION}",
        "X-TestCenter-Event": str(payload.get("event", "job_notification")),
    }
    if payload.get("job_id") is not None:
        headers["X-TestCenter-Job-Id"] = str(payload["job_id"])
    if payload.get("run_id") is not None:
        headers["X-TestCenter-Run-Id"] = str(payload["run_id"])

    if custom_headers:
        for k, v in custom_headers.items():
            if str(k).lower() not in _SECRET_FIELD_NAMES and v is not None:
                headers[str(k)] = str(v)

    if secret_env_var:
        secret_val = os.environ.get(secret_env_var, "")
        if secret_val:
            sig = hmac.new(
                secret_val.encode("utf-8"),
                body_bytes,
                hashlib.sha256,
            ).hexdigest()
            headers["X-TestCenter-Signature"] = f"sha256={sig}"

    sent_at = utc_now().isoformat()
    try:
        async with httpx.AsyncClient(timeout=req_timeout) as client:
            response = await client.post(
                webhook_url,
                content=body_bytes,
                headers=headers,
            )
        is_ok = 200 <= response.status_code < 300
        return {
            "channel": "webhook",
            "target": webhook_url,
            "status": "sent" if is_ok else "failed",
            "ok": is_ok,
            "status_code": response.status_code,
            "error": None if is_ok else f"HTTP {response.status_code}: {response.text[:200]}",
            "sent_at": sent_at,
        }
    except Exception as exc:
        logger.warning(f"Webhook delivery to {webhook_url} failed: {exc}")
        return {
            "channel": "webhook",
            "target": webhook_url,
            "status": "failed",
            "ok": False,
            "status_code": None,
            "error": str(exc),
            "sent_at": sent_at,
        }


def send_email_notification_sync(
    email_to: list[str],
    payload: dict[str, Any],
    *,
    subject_prefix: str | None = None,
) -> dict[str, Any]:
    """Deliver an email notification using SMTP settings from application config."""
    settings = get_settings()
    sent_at = utc_now().isoformat()

    if not email_to:
        return {
            "channel": "email",
            "target": [],
            "status": "skipped",
            "ok": False,
            "error": "No email recipients configured",
            "sent_at": sent_at,
        }

    smtp_host = (os.getenv("SMTP_HOST") or settings.SMTP_HOST or "").strip()
    smtp_port = int(os.getenv("SMTP_PORT") or settings.SMTP_PORT or 587)
    smtp_user = (os.getenv("SMTP_USER") or settings.SMTP_USER or "").strip()
    smtp_from = (os.getenv("SMTP_FROM") or settings.SMTP_FROM or "test-center@localhost").strip()
    smtp_use_tls = (os.getenv("SMTP_USE_TLS") or str(settings.SMTP_USE_TLS)).lower() in (
        "1",
        "true",
        "yes",
    )
    smtp_use_ssl = (os.getenv("SMTP_USE_SSL") or str(settings.SMTP_USE_SSL)).lower() in (
        "1",
        "true",
        "yes",
    )
    pwd_env_var = settings.SMTP_PASSWORD_ENV_VAR or "SMTP_PASSWORD"
    smtp_password = os.environ.get(pwd_env_var, "") or os.environ.get("SMTP_PASSWORD", "")

    prefix = (subject_prefix or "[Test Center]").strip()
    job_name = payload.get("job_name") or f"Job #{payload.get('job_id')}"
    run_id = payload.get("run_id")
    reasons = payload.get("reasons") or []
    reason_badge = " & ".join(r.upper() for r in reasons) if reasons else "ALERT"

    if payload.get("is_test"):
        subject = f"{prefix} Test notification for job '{job_name}'"
    else:
        subject = f"{prefix} Job '{job_name}' (Run #{run_id}): {reason_badge}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = ", ".join(email_to)
    msg.set_content(str(payload.get("text") or payload.get("message") or ""))

    if not smtp_host:
        logger.warning("Email notification skipped: SMTP_HOST is not configured")
        return {
            "channel": "email",
            "target": list(email_to),
            "subject": subject,
            "status": "skipped",
            "ok": False,
            "error": "SMTP_HOST is not configured",
            "sent_at": sent_at,
        }

    try:
        smtp_cls = smtplib.SMTP_SSL if smtp_use_ssl else smtplib.SMTP
        with smtp_cls(smtp_host, smtp_port, timeout=10.0) as server:
            if smtp_use_tls and not smtp_use_ssl:
                server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)

        return {
            "channel": "email",
            "target": list(email_to),
            "subject": subject,
            "status": "sent",
            "ok": True,
            "error": None,
            "sent_at": sent_at,
        }
    except Exception as exc:
        logger.warning(f"Email notification to {email_to} failed: {exc}")
        return {
            "channel": "email",
            "target": list(email_to),
            "subject": subject,
            "status": "failed",
            "ok": False,
            "error": str(exc),
            "sent_at": sent_at,
        }


async def send_email_notification(
    email_to: list[str],
    payload: dict[str, Any],
    *,
    subject_prefix: str | None = None,
) -> dict[str, Any]:
    """Async wrapper for SMTP email notification delivery."""
    return await asyncio.to_thread(
        send_email_notification_sync,
        email_to,
        payload,
        subject_prefix=subject_prefix,
    )


async def dispatch_notifications(
    settings: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Dispatch notifications across enabled channels (webhook first, then optional email)."""
    deliveries: list[dict[str, Any]] = []

    # 1. Webhook first
    webhook_url = settings.get("webhook_url")
    if settings.get("webhook_enabled") and webhook_url:
        wh_res = await send_webhook_notification(
            webhook_url=webhook_url,
            payload=payload,
            secret_env_var=settings.get("webhook_secret_env_var"),
            custom_headers=settings.get("webhook_headers"),
        )
        deliveries.append(wh_res)

    # 2. Email optional
    email_to = settings.get("email_to") or []
    if settings.get("email_enabled") and email_to:
        em_res = await send_email_notification(
            email_to=email_to,
            payload=payload,
            subject_prefix=settings.get("email_subject_prefix"),
        )
        deliveries.append(em_res)

    return deliveries


async def process_run_notifications(
    db: Session,
    run_id: int,
    *,
    force: bool = False,
    settings_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a completed Run and dispatch notifications if triggered.

    Persists the evaluation and delivery summary in `run.config_snapshot['notification_report']`.
    """
    stmt = (
        select(Run)
        .options(
            selectinload(Run.job),
            selectinload(Run.results).selectinload(Result.cases),
            selectinload(Run.results).selectinload(Result.artifacts),
        )
        .where(Run.id == run_id)
    )
    run = db.execute(stmt).scalars().first()
    if run is None:
        return {"triggered": False, "reason": "run_not_found"}

    job = run.job
    if job is None and run.job_id is not None:
        job = db.get(Job, run.job_id)

    # If the run is not associated with any job and has no notification settings, skip
    has_snap_notifications = (
        isinstance(run.config_snapshot, dict)
        and isinstance(run.config_snapshot.get("notifications"), dict)
        and bool(run.config_snapshot.get("notifications"))
    )
    if job is None and not has_snap_notifications and settings_override is None:
        return {"triggered": False, "reason": "not_a_job_run"}

    evaluation = evaluate_run_for_notifications(
        db=db,
        run=run,
        job=job,
        settings_override=settings_override,
    )
    settings = evaluation["settings"]
    should_notify = evaluation["should_notify"] or force

    deliveries: list[dict[str, Any]] = []
    payload: dict[str, Any] | None = None

    if should_notify:
        payload = build_notification_payload(run=run, job=job, evaluation=evaluation)
        deliveries = await dispatch_notifications(settings=settings, payload=payload)

    report: dict[str, Any] = {
        "evaluated_at": utc_now().isoformat(),
        "triggered": bool(should_notify and len(deliveries) > 0),
        "should_notify": evaluation["should_notify"],
        "reasons": evaluation["reasons"],
        "failed": evaluation["failed"],
        "regressed": evaluation["regressed"],
        "baseline_run_id": evaluation["baseline_run_id"],
        "regression": {
            "has_regression": evaluation["regression"]["has_regression"],
            "compared": evaluation["regression"]["compared"],
            "baseline_run_id": evaluation["regression"]["baseline_run_id"],
            "threshold": evaluation["regression"]["threshold"],
            "metrics_regressed": evaluation["regression"]["metrics_regressed"],
            "cases_regressed": evaluation["regression"]["cases_regressed"],
            "regressions": evaluation["regression"]["regressions"],
            "compare_url": evaluation["regression"]["compare_url"],
        },
        "failure": evaluation["failure"],
        "deliveries": deliveries,
    }
    if payload is not None:
        report["payload"] = payload

    if isinstance(run.config_snapshot, dict):
        updated_snap = dict(run.config_snapshot)
        updated_snap["notification_report"] = report
        updated_snap["notifications_sent"] = deliveries
        run.config_snapshot = updated_snap
        db.commit()

    return report
