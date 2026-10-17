"""JUnit XML export generator for Test Center runs and results.

Produces standard JUnit XML (`<testsuites>` / `<testsuite>` / `<testcase>`) compatible with
Atlassian Bamboo's JUnit Parser task and other CI test result consumers.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.db.models.artifact import Artifact
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run

# Control characters illegal in XML 1.0 (excluding \t=0x09, \n=0x0A, \r=0x0D)
_ILLEGAL_XML_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ud800-\udfff\ufffe\uffff]"
)


def _sanitize_xml_text(value: Any) -> str:
    """Convert a value to string and strip characters illegal in XML 1.0."""
    if value is None:
        return ""
    text = str(value)
    return _ILLEGAL_XML_CHARS_RE.sub("", text)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_iso_timestamp(dt: datetime | None) -> str:
    """Format a datetime as an ISO-8601 UTC string suitable for JUnit XML timestamp attributes."""
    utc_dt = _ensure_utc(dt) or datetime.now(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_numeric_metric(raw_entry: Any) -> float | None:
    """Extract a numeric value from a metric entry (either raw number or {'value': ...} dict)."""
    if isinstance(raw_entry, (int, float)) and not isinstance(raw_entry, bool):
        return float(raw_entry)
    if isinstance(raw_entry, dict):
        val = raw_entry.get("value")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


def _extract_metric_display(raw_entry: Any) -> str:
    """Format a metric entry as a concise string with optional unit."""
    if isinstance(raw_entry, dict) and "value" in raw_entry:
        val = raw_entry.get("value")
        unit = raw_entry.get("unit")
        if unit:
            return f"{val} {unit}"
        return str(val)
    return str(raw_entry)


def _extract_case_duration_seconds(
    metrics: dict[str, Any] | None,
    detail: dict[str, Any] | None,
    default_sec: float = 0.0,
) -> float:
    """Extract execution time in seconds for a single test case from its metrics or detail."""
    ms_keys = (
        "latency_ms",
        "duration_ms",
        "wall_ms",
        "total_latency_ms",
        "final_latency_ms",
        "elapsed_ms",
        "time_ms",
        "ttft_ms",
    )
    sec_keys = (
        "duration_seconds",
        "duration_sec",
        "elapsed_seconds",
        "elapsed_sec",
        "time_seconds",
        "time_sec",
        "time",
        "duration",
    )

    for source in (metrics, detail):
        if not isinstance(source, dict):
            continue
        for k in ms_keys:
            if k in source:
                val = _extract_numeric_metric(source[k])
                if val is not None and val >= 0:
                    return round(val / 1000.0, 6)
        for k in sec_keys:
            if k in source:
                val = _extract_numeric_metric(source[k])
                if val is not None and val >= 0:
                    return round(val, 6)

    return max(0.0, default_sec)


def _extract_result_duration_seconds(
    result: Result,
    case_durations_sum: float,
    run_duration_sec: float | None,
) -> float:
    """Determine total duration in seconds for a Result."""
    if isinstance(result.metrics, dict):
        for k in ("duration_ms", "wall_ms", "total_latency_ms", "latency_mean_ms"):
            if k in result.metrics:
                val = _extract_numeric_metric(result.metrics[k])
                if val is not None and val >= 0:
                    return max(case_durations_sum, round(val / 1000.0, 6))
        for k in ("duration_seconds", "duration_sec", "elapsed_seconds", "wall_sec"):
            if k in result.metrics:
                val = _extract_numeric_metric(result.metrics[k])
                if val is not None and val >= 0:
                    return max(case_durations_sum, round(val, 6))

    if case_durations_sum > 0:
        return case_durations_sum
    if run_duration_sec is not None and run_duration_sec >= 0:
        return run_duration_sec
    return 0.0


def _extract_case_failure_info(
    case_id: str,
    status: str,
    metrics: dict[str, Any] | None,
    detail: dict[str, Any] | None,
) -> tuple[str, str, str]:
    """Extract (message, type_name, detail_body) for a failed, errored, or skipped test case."""
    norm_status = (status or "").strip().lower()
    default_type = (
        "AssertionError"
        if norm_status in ("fail", "failed")
        else ("Skipped" if norm_status in ("skip", "skipped", "cancelled") else "ExecutionError")
    )

    msg: str | None = None
    failure_type = default_type

    if isinstance(detail, dict):
        for key in ("error", "message", "reason", "failure_reason", "summary", "detail", "status_detail"):
            val = detail.get(key)
            if isinstance(val, str) and val.strip():
                msg = val.strip()
                break
            elif isinstance(val, dict) and isinstance(val.get("message"), str) and val["message"].strip():
                msg = val["message"].strip()
                break

        if isinstance(detail.get("error_type"), str) and detail["error_type"].strip():
            failure_type = detail["error_type"].strip()

        # Check if there are failed checks in detail["checks"]
        if not msg and isinstance(detail.get("checks"), list):
            failed_checks = [
                c
                for c in detail["checks"]
                if isinstance(c, dict) and c.get("passed") is False
            ]
            if failed_checks:
                parts = []
                for fc in failed_checks[:3]:
                    cname = fc.get("name") or fc.get("check") or "check"
                    cdetail = fc.get("detail") or fc.get("message") or "failed"
                    parts.append(f"{cname}: {cdetail}")
                msg = "; ".join(parts)

    if not msg:
        if norm_status in ("fail", "failed"):
            msg = f"Test case '{case_id}' failed"
        elif norm_status in ("skip", "skipped", "cancelled"):
            msg = f"Test case '{case_id}' was {norm_status}"
        else:
            msg = f"Test case '{case_id}' encountered an error"

    lines: list[str] = [f"Case: {case_id}", f"Status: {status}", f"Message: {msg}"]
    if isinstance(metrics, dict) and metrics:
        lines.append("Metrics:")
        for mk, mv in metrics.items():
            lines.append(f"  {mk}: {_extract_metric_display(mv)}")
    if isinstance(detail, dict) and detail:
        try:
            detail_json = json.dumps(detail, indent=2, sort_keys=True, default=str)
            lines.append(f"Detail:\n{detail_json}")
        except Exception:
            lines.append(f"Detail: {detail!r}")

    return _sanitize_xml_text(msg), _sanitize_xml_text(failure_type), _sanitize_xml_text("\n".join(lines))


def _format_case_system_out(
    metrics: dict[str, Any] | None,
    detail: dict[str, Any] | None,
) -> str:
    """Format case metrics and detail into a human-readable string for <system-out>."""
    parts: list[str] = []
    if isinstance(metrics, dict) and metrics:
        metric_lines = [f"{k}={_extract_metric_display(v)}" for k, v in metrics.items()]
        parts.append("Metrics: " + ", ".join(metric_lines))
    if isinstance(detail, dict) and detail:
        try:
            parts.append("Detail: " + json.dumps(detail, sort_keys=True, default=str))
        except Exception:
            parts.append(f"Detail: {detail!r}")
    return _sanitize_xml_text("\n".join(parts))


def _extract_run_hostname(run: Run | None) -> str:
    """Extract target name or base_url host from run.target_snapshot for the JUnit hostname attribute."""
    if run is None:
        return "test-center"
    tsnap = run.target_snapshot
    if isinstance(tsnap, dict):
        name = tsnap.get("name") or tsnap.get("base_url")
        if isinstance(name, str) and name.strip():
            return _sanitize_xml_text(name.strip())
    return "test-center"


def _append_result_testsuites(
    root: ET.Element,
    result: Result,
    *,
    run: Run | None = None,
    suite_filter: str | None = None,
    run_duration_sec: float | None = None,
    timestamp_str: str,
    hostname: str,
) -> tuple[int, int, int, int, float]:
    """Append <testsuite> elements for a Result onto root <testsuites> and return aggregate counts."""
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0

    tester_name = _sanitize_xml_text(result.tester or "unknown")
    sf_norm = suite_filter.strip().lower() if suite_filter and suite_filter.strip() else None

    # Group cases by suite preserving order
    suites_map: OrderedDict[str, list[Any]] = OrderedDict()
    for c in result.cases or []:
        suite_name = str(c.suite or "default")
        if sf_norm and suite_name.lower() != sf_norm:
            continue
        suites_map.setdefault(suite_name, []).append(c)

    # If the result has no cases (e.g. runner timeout/crash/cancel) and isn't filtered out by suite_filter
    if not suites_map and not (result.cases and sf_norm):
        res_status = (result.status or "pass").strip().lower()
        res_dur = _extract_result_duration_seconds(result, 0.0, run_duration_sec)
        suite_el = ET.SubElement(
            root,
            "testsuite",
            {
                "name": f"{tester_name}.execution",
                "package": tester_name,
                "tests": "1" if res_status in ("fail", "failed", "error", "cancelled", "pass") else "0",
                "failures": "1" if res_status in ("fail", "failed") else "0",
                "errors": "1" if res_status == "error" else "0",
                "skipped": "1" if res_status in ("cancelled", "skip", "skipped") else "0",
                "time": f"{res_dur:.3f}",
                "timestamp": timestamp_str,
                "hostname": hostname,
            },
        )
        _append_suite_properties(suite_el, result=result, run=run, suite_name="execution")

        if res_status in ("fail", "failed", "error", "cancelled", "pass"):
            tc_el = ET.SubElement(
                suite_el,
                "testcase",
                {
                    "name": "execution",
                    "classname": f"{tester_name}.execution",
                    "time": f"{res_dur:.3f}",
                },
            )
            summary_msg = _sanitize_xml_text(
                result.summary or f"Tester '{tester_name}' finished with status '{result.status}'"
            )
            if res_status in ("fail", "failed"):
                fail_el = ET.SubElement(
                    tc_el,
                    "failure",
                    {"message": summary_msg, "type": "AssertionError"},
                )
                fail_el.text = summary_msg
                total_failures += 1
            elif res_status == "error":
                err_el = ET.SubElement(
                    tc_el,
                    "error",
                    {"message": summary_msg, "type": "ExecutionError"},
                )
                err_el.text = summary_msg
                total_errors += 1
            elif res_status in ("cancelled", "skip", "skipped"):
                ET.SubElement(
                    tc_el,
                    "skipped",
                    {"message": summary_msg},
                )
                total_skipped += 1
            elif summary_msg:
                sys_out = ET.SubElement(tc_el, "system-out")
                sys_out.text = summary_msg

            total_tests += 1
            total_time += res_dur

        return total_tests, total_failures, total_errors, total_skipped, total_time

    for suite_name, cases in suites_map.items():
        clean_suite = _sanitize_xml_text(suite_name)
        full_suite_name = f"{tester_name}.{clean_suite}"

        suite_tests = len(cases)
        suite_failures = 0
        suite_errors = 0
        suite_skipped = 0
        suite_time = 0.0

        case_rows: list[tuple[Any, float, str]] = []
        for c in cases:
            c_status = (c.status or "pass").strip().lower()
            c_dur = _extract_case_duration_seconds(c.metrics, c.detail, default_sec=0.0)
            suite_time += c_dur
            if c_status in ("fail", "failed"):
                suite_failures += 1
            elif c_status == "error":
                suite_errors += 1
            elif c_status in ("skip", "skipped", "cancelled"):
                suite_skipped += 1
            case_rows.append((c, c_dur, c_status))

        # If individual cases didn't record latency_ms, fallback to result duration for suite_time
        if suite_time == 0.0 and len(suites_map) == 1:
            suite_time = _extract_result_duration_seconds(result, 0.0, run_duration_sec)

        suite_el = ET.SubElement(
            root,
            "testsuite",
            {
                "name": full_suite_name,
                "package": tester_name,
                "tests": str(suite_tests),
                "failures": str(suite_failures),
                "errors": str(suite_errors),
                "skipped": str(suite_skipped),
                "time": f"{suite_time:.3f}",
                "timestamp": timestamp_str,
                "hostname": hostname,
            },
        )
        _append_suite_properties(suite_el, result=result, run=run, suite_name=clean_suite)

        for c, c_dur, c_status in case_rows:
            cid = _sanitize_xml_text(c.case_id or f"case_{c.id}")
            tc_el = ET.SubElement(
                suite_el,
                "testcase",
                {
                    "name": cid,
                    "classname": full_suite_name,
                    "time": f"{c_dur:.3f}",
                },
            )

            if c_status in ("fail", "failed"):
                f_msg, f_type, f_body = _extract_case_failure_info(cid, c_status, c.metrics, c.detail)
                fail_el = ET.SubElement(
                    tc_el,
                    "failure",
                    {"message": f_msg, "type": f_type},
                )
                fail_el.text = f_body
            elif c_status == "error":
                e_msg, e_type, e_body = _extract_case_failure_info(cid, c_status, c.metrics, c.detail)
                err_el = ET.SubElement(
                    tc_el,
                    "error",
                    {"message": e_msg, "type": e_type},
                )
                err_el.text = e_body
            elif c_status in ("skip", "skipped", "cancelled"):
                s_msg, _, _ = _extract_case_failure_info(cid, c_status, c.metrics, c.detail)
                ET.SubElement(
                    tc_el,
                    "skipped",
                    {"message": s_msg},
                )

            sys_out_text = _format_case_system_out(c.metrics, c.detail)
            if sys_out_text:
                sys_out = ET.SubElement(tc_el, "system-out")
                sys_out.text = sys_out_text

        total_tests += suite_tests
        total_failures += suite_failures
        total_errors += suite_errors
        total_skipped += suite_skipped
        total_time += suite_time

    return total_tests, total_failures, total_errors, total_skipped, total_time


def _append_suite_properties(
    suite_el: ET.Element,
    *,
    result: Result | None,
    run: Run | None,
    suite_name: str,
) -> None:
    """Attach <properties> with run, job, Bamboo metadata, and aggregate metrics to a <testsuite>."""
    props_el = ET.SubElement(suite_el, "properties")

    def _add_prop(name: str, val: Any) -> None:
        if val is None:
            return
        ET.SubElement(
            props_el,
            "property",
            {
                "name": _sanitize_xml_text(name),
                "value": _sanitize_xml_text(val),
            },
        )

    if run is not None:
        _add_prop("run_id", run.id)
        _add_prop("run_status", run.status)
        if run.job_id is not None:
            _add_prop("job_id", run.job_id)
        if isinstance(run.config_snapshot, dict):
            if run.config_snapshot.get("job_name"):
                _add_prop("job_name", run.config_snapshot["job_name"])
            bamboo_meta = run.config_snapshot.get("bamboo")
            if isinstance(bamboo_meta, dict):
                for bk in ("plan_key", "build_number", "build_key", "build_result_key", "branch", "commit"):
                    if bamboo_meta.get(bk) is not None:
                        _add_prop(f"bamboo.{bk}", bamboo_meta[bk])

    if result is not None:
        _add_prop("result_id", result.id)
        _add_prop("tester", result.tester)
        _add_prop("suite", suite_name)
        _add_prop("result_status", result.status)
        if result.summary:
            _add_prop("summary", result.summary)
        if isinstance(result.metrics, dict):
            for mk, mv in result.metrics.items():
                _add_prop(f"metric.{mk}", _extract_metric_display(mv))


def build_run_junit_xml(
    run: Run,
    *,
    suite_filter: str | None = None,
    tester_filter: str | None = None,
    result_id_filter: int | None = None,
) -> str:
    """Generate a JUnit XML document for a Run and all of its Results / CaseResults."""
    primary_ts = run.started_at or run.queued_at or run.finished_at or getattr(run, "created_at", None)
    timestamp_str = _format_iso_timestamp(primary_ts)
    hostname = _extract_run_hostname(run)

    run_duration_sec: float | None = None
    if run.started_at and run.finished_at:
        s_utc = _ensure_utc(run.started_at)
        f_utc = _ensure_utc(run.finished_at)
        if s_utc and f_utc:
            run_duration_sec = max(0.0, (f_utc - s_utc).total_seconds())

    job_name: str | None = None
    if isinstance(run.config_snapshot, dict):
        job_name = run.config_snapshot.get("job_name")
    if not job_name and getattr(run, "job", None) is not None:
        job_name = run.job.name

    suite_title = f"Run #{run.id}"
    if job_name:
        suite_title = f"Run #{run.id} ({job_name})"
    elif isinstance(run.config_snapshot, dict) and run.config_snapshot.get("tester"):
        suite_title = f"Run #{run.id} ({run.config_snapshot['tester']})"

    root = ET.Element(
        "testsuites",
        {
            "id": str(run.id),
            "name": _sanitize_xml_text(suite_title),
            "tests": "0",
            "failures": "0",
            "errors": "0",
            "skipped": "0",
            "time": "0.000",
            "timestamp": timestamp_str,
        },
    )

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0

    tf_norm = tester_filter.strip().lower() if tester_filter and tester_filter.strip() else None
    results_list = list(run.results or [])

    if result_id_filter is not None:
        results_list = [r for r in results_list if r.id == result_id_filter]
    if tf_norm:
        results_list = [r for r in results_list if (r.tester or "").lower() == tf_norm]

    if results_list:
        for res in results_list:
            t_cnt, f_cnt, e_cnt, s_cnt, dur = _append_result_testsuites(
                root,
                res,
                run=run,
                suite_filter=suite_filter,
                run_duration_sec=run_duration_sec if len(results_list) == 1 else None,
                timestamp_str=timestamp_str,
                hostname=hostname,
            )
            total_tests += t_cnt
            total_failures += f_cnt
            total_errors += e_cnt
            total_skipped += s_cnt
            total_time += dur
    elif result_id_filter is None and not tf_norm and not suite_filter:
        # Run has no Result rows yet; if run is in a terminal failure/error/cancelled state, emit synthetic testcase
        run_status = (run.status or "queued").strip().lower()
        if run_status in ("fail", "failed", "error", "cancelled"):
            tester_name = "unknown"
            if isinstance(run.config_snapshot, dict):
                tester_name = _sanitize_xml_text(run.config_snapshot.get("tester", "unknown"))
            dur = run_duration_sec or 0.0
            suite_el = ET.SubElement(
                root,
                "testsuite",
                {
                    "name": f"{tester_name}.execution",
                    "package": tester_name,
                    "tests": "1",
                    "failures": "1" if run_status in ("fail", "failed") else "0",
                    "errors": "1" if run_status == "error" else "0",
                    "skipped": "1" if run_status == "cancelled" else "0",
                    "time": f"{dur:.3f}",
                    "timestamp": timestamp_str,
                    "hostname": hostname,
                },
            )
            _append_suite_properties(suite_el, result=None, run=run, suite_name="execution")
            tc_el = ET.SubElement(
                suite_el,
                "testcase",
                {
                    "name": "execution",
                    "classname": f"{tester_name}.execution",
                    "time": f"{dur:.3f}",
                },
            )
            msg = f"Run #{run.id} terminated with status '{run.status}'"
            if run_status in ("fail", "failed"):
                el = ET.SubElement(tc_el, "failure", {"message": msg, "type": "AssertionError"})
                el.text = msg
                total_failures = 1
            elif run_status == "error":
                el = ET.SubElement(tc_el, "error", {"message": msg, "type": "ExecutionError"})
                el.text = msg
                total_errors = 1
            else:
                ET.SubElement(tc_el, "skipped", {"message": msg})
                total_skipped = 1
            total_tests = 1
            total_time = dur

    eff_total_time = run_duration_sec if (run_duration_sec is not None and total_time == 0.0) else total_time
    root.set("tests", str(total_tests))
    root.set("failures", str(total_failures))
    root.set("errors", str(total_errors))
    root.set("skipped", str(total_skipped))
    root.set("time", f"{eff_total_time:.3f}")

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_body}\n'


def build_result_junit_xml(
    result: Result,
    *,
    run: Run | None = None,
    suite_filter: str | None = None,
) -> str:
    """Generate a JUnit XML document for a single Result."""
    eff_run = run or getattr(result, "run", None)
    primary_ts = (
        (eff_run.started_at or eff_run.queued_at or eff_run.finished_at)
        if eff_run is not None
        else getattr(result, "created_at", None)
    )
    timestamp_str = _format_iso_timestamp(primary_ts)
    hostname = _extract_run_hostname(eff_run)

    root = ET.Element(
        "testsuites",
        {
            "id": str(result.id),
            "name": _sanitize_xml_text(f"Result #{result.id} ({result.tester})"),
            "tests": "0",
            "failures": "0",
            "errors": "0",
            "skipped": "0",
            "time": "0.000",
            "timestamp": timestamp_str,
        },
    )

    t_cnt, f_cnt, e_cnt, s_cnt, dur = _append_result_testsuites(
        root,
        result,
        run=eff_run,
        suite_filter=suite_filter,
        run_duration_sec=None,
        timestamp_str=timestamp_str,
        hostname=hostname,
    )
    root.set("tests", str(t_cnt))
    root.set("failures", str(f_cnt))
    root.set("errors", str(e_cnt))
    root.set("skipped", str(s_cnt))
    root.set("time", f"{dur:.3f}")

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_body}\n'


def write_run_junit_artifact(
    db: Session,
    run_id: int,
    artifacts_dir: Path,
) -> Path | None:
    """Write junit.xml under data/artifacts/<run_id>/junit.xml and link it as an Artifact."""
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
    if not run:
        return None

    xml_content = build_run_junit_xml(run)
    run_dir = Path(artifacts_dir) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    junit_path = run_dir / "junit.xml"
    junit_path.write_text(xml_content, encoding="utf-8")

    if run.results:
        primary_result = run.results[0]
        already_linked = any(
            a.name == "junit.xml" or Path(a.path).name == "junit.xml"
            for res in run.results
            for a in (res.artifacts or [])
        )
        if not already_linked:
            db.add(
                Artifact(
                    result_id=primary_result.id,
                    name="junit.xml",
                    path=str(junit_path),
                    mime="application/xml",
                    size_bytes=junit_path.stat().st_size,
                )
            )
            db.commit()

    return junit_path
