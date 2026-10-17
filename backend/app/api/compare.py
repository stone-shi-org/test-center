"""API routes and core logic for comparing 2+ test runs or results side by side.

Supports:
- Selecting 2+ runs (`run_ids`) and/or results (`result_ids`) of the same tester
  (across different targets, models, or configurations).
- Choosing an explicit baseline (`baseline_run_id` / `baseline_result_id`) or defaulting
  to the first selected item.
- Aligned top-level metric table (baseline vs others, absolute and % delta,
  regression/improvement flag from `better=higher|lower` and a configurable threshold).
- Aligned per-case diff table (status changes, per-case metric deltas, handling cases
  and metrics missing on one side, and cross-model case alignment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.db.models.run import Run
from backend.app.db.models.result import Result
from backend.app.db.models.case_result import CaseResult
from backend.app.db.session import get_db

router = APIRouter(prefix="/compare", tags=["compare"])

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

_STATUS_RANK: dict[str, int] = {
    "pass": 3,
    "skip": 2,
    "fail": 1,
    "error": 0,
}


class CompareRequest(BaseModel):
    """Request payload for POST /api/compare."""

    run_ids: list[int] | None = Field(
        default=None,
        description="List of Run IDs to compare (2+ items when combined with result_ids)",
    )
    result_ids: list[int] | None = Field(
        default=None,
        description="List of Result IDs to compare (2+ items when combined with run_ids)",
    )
    baseline_run_id: int | None = Field(
        default=None,
        description="Optional Run ID to serve as the comparison baseline (defaults to first item)",
    )
    baseline_result_id: int | None = Field(
        default=None,
        description="Optional Result ID to serve as the comparison baseline (defaults to first item)",
    )
    baseline_id: int | None = Field(
        default=None,
        description="Optional alias for baseline_run_id or baseline_result_id",
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Percentage delta threshold (e.g. 5.0 for 5%) required to flag a regression or improvement",
    )
    threshold_pct: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional alias for threshold",
    )
    metric_thresholds: dict[str, float] | None = Field(
        default=None,
        description="Optional per-metric percentage threshold overrides {metric_name: threshold_pct}",
    )
    tester: str | None = Field(
        default=None,
        description="Optional tester name filter when a run contains multiple results",
    )
    suite: str | None = Field(
        default=None,
        description="Optional suite filter for per-case diff rows",
    )
    changed_only: bool = Field(
        default=False,
        description="When true, return only metrics and cases that changed or have missing entries",
    )
    regressed_only: bool = Field(
        default=False,
        description="When true, return only metrics and cases with regressions",
    )
    allow_different_testers: bool = Field(
        default=False,
        description="When true, allow comparing results from different testers",
    )


@dataclass
class ComparedItem:
    """Internal representation of a resolved (Run, Result) pair being compared."""

    run: Run
    result: Result

    @property
    def run_id(self) -> int:
        return self.run.id

    @property
    def result_id(self) -> int:
        return self.result.id

    @property
    def key(self) -> tuple[int, int]:
        return (self.run.id, self.result.id)


def _sanitize_dict(data: Any) -> Any:
    """Recursively remove any raw secret keys while keeping safe references like auth_env_var."""
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            if str(k).lower() in _SECRET_FIELD_NAMES:
                continue
            cleaned[k] = _sanitize_dict(v)
        return cleaned
    if isinstance(data, list):
        return [_sanitize_dict(item) for item in data]
    return data


def _parse_id_list_param(raw_values: list[str] | None, param_name: str) -> list[int]:
    """Parse query parameter values supporting both comma-separated and repeated params."""
    if not raw_values:
        return []
    parsed: list[int] = []
    for item in raw_values:
        if not item:
            continue
        for part in str(item).split(","):
            part_str = part.strip()
            if not part_str:
                continue
            try:
                parsed.append(int(part_str))
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid integer value {part_str!r} in '{param_name}'",
                ) from e
    return parsed


def _parse_metric_entry(raw_entry: Any) -> tuple[float | int | None, str | None, Literal["higher", "lower"] | None]:
    """Extract (numeric_value, unit, better) from a metric dictionary or scalar number."""
    if raw_entry is None:
        return None, None, None

    if isinstance(raw_entry, dict):
        val = raw_entry.get("value")
        num_val: float | int | None = (
            val if isinstance(val, (int, float)) and not isinstance(val, bool) else None
        )
        raw_unit = raw_entry.get("unit")
        unit = raw_unit if isinstance(raw_unit, str) else None
        raw_better = raw_entry.get("better")
        better: Literal["higher", "lower"] | None = (
            raw_better if raw_better in ("higher", "lower") else None
        )
        return num_val, unit, better

    if isinstance(raw_entry, (int, float)) and not isinstance(raw_entry, bool):
        return raw_entry, None, None

    return None, None, None


def compute_metric_delta(
    baseline_value: float | int | None,
    current_value: float | int | None,
    better: Literal["higher", "lower"] | str | None = None,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Compute absolute delta, percentage delta, and regression/improvement flags.

    Args:
        baseline_value: Numeric metric value in the baseline run, or None if missing.
        current_value: Numeric metric value in the compared run, or None if missing.
        better: Direction of improvement ('higher' or 'lower'), or None if neutral/unknown.
        threshold: Minimum percentage change (e.g. 5.0 for 5%) required to trigger
            regression or improvement flags.

    Returns:
        Dictionary with abs_delta, pct_delta, regression, improvement, and flag.
    """
    if baseline_value is None or current_value is None:
        return {
            "abs_delta": None,
            "pct_delta": None,
            "regression": False,
            "improvement": False,
            "flag": "missing",
        }

    base_f = float(baseline_value)
    curr_f = float(current_value)
    raw_diff = curr_f - base_f
    abs_delta = round(raw_diff, 6)
    if abs_delta == 0.0:
        abs_delta = 0.0

    if base_f == 0.0:
        pct_delta: float | None = 0.0 if curr_f == 0.0 else None
    else:
        pct_delta = round((raw_diff / abs(base_f)) * 100.0, 4)
        if pct_delta == 0.0:
            pct_delta = 0.0

    eff_threshold = max(0.0, float(threshold))
    if abs_delta == 0.0:
        exceeds_threshold = False
    elif eff_threshold <= 0.0:
        exceeds_threshold = True
    else:
        exceeds_threshold = True if pct_delta is None else (abs(pct_delta) >= eff_threshold)

    regression = False
    improvement = False
    flag = "unchanged"

    if exceeds_threshold:
        if better == "higher":
            if abs_delta > 0:
                improvement = True
                flag = "improvement"
            elif abs_delta < 0:
                regression = True
                flag = "regression"
        elif better == "lower":
            if abs_delta < 0:
                improvement = True
                flag = "improvement"
            elif abs_delta > 0:
                regression = True
                flag = "regression"
        else:
            flag = "changed"

    return {
        "abs_delta": abs_delta,
        "pct_delta": pct_delta,
        "regression": regression,
        "improvement": improvement,
        "flag": flag,
    }


def compute_case_status_diff(
    baseline_status: str | None,
    current_status: str | None,
) -> dict[str, Any]:
    """Compute per-case status change and regression/improvement flags vs baseline."""
    if baseline_status is None and current_status is None:
        return {
            "status_change": "missing",
            "status_changed": False,
            "status_regression": False,
            "status_improvement": False,
        }

    if baseline_status is None and current_status is not None:
        return {
            "status_change": "added",
            "status_changed": True,
            "status_regression": current_status in ("fail", "error"),
            "status_improvement": False,
        }

    if baseline_status is not None and current_status is None:
        return {
            "status_change": "missing",
            "status_changed": True,
            "status_regression": False,
            "status_improvement": False,
        }

    assert baseline_status is not None and current_status is not None
    if baseline_status == current_status:
        return {
            "status_change": "unchanged",
            "status_changed": False,
            "status_regression": False,
            "status_improvement": False,
        }

    base_rank = _STATUS_RANK.get(baseline_status, 1)
    curr_rank = _STATUS_RANK.get(current_status, 1)

    if curr_rank < base_rank:
        return {
            "status_change": "regression",
            "status_changed": True,
            "status_regression": True,
            "status_improvement": False,
        }
    if curr_rank > base_rank:
        return {
            "status_change": "improvement",
            "status_changed": True,
            "status_regression": False,
            "status_improvement": True,
        }

    return {
        "status_change": "changed",
        "status_changed": True,
        "status_regression": False,
        "status_improvement": False,
    }


def align_and_compare_metrics(
    items_metrics: list[tuple[int, int, dict[str, Any]]],
    baseline_key: tuple[int, int],
    threshold: float = 0.0,
    metric_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Align metrics across 2+ runs/results and compute deltas against the baseline.

    Args:
        items_metrics: List of (run_id, result_id, metrics_dict) in display order.
        baseline_key: (baseline_run_id, baseline_result_id) tuple.
        threshold: Default percentage threshold for regression/improvement flagging.
        metric_thresholds: Optional per-metric percentage threshold overrides.

    Returns:
        List of aligned metric row dictionaries.
    """
    baseline_entry = next(
        (m for r_id, res_id, m in items_metrics if (r_id, res_id) == baseline_key),
        {},
    )
    baseline_run_id, baseline_result_id = baseline_key

    # Collect union of all metric names preserving baseline-first deterministic order
    metric_names: list[str] = []
    seen_names: set[str] = set()
    for name in (baseline_entry or {}).keys():
        if isinstance(name, str) and name not in seen_names:
            metric_names.append(name)
            seen_names.add(name)
    for _, _, m_dict in items_metrics:
        for name in (m_dict or {}).keys():
            if isinstance(name, str) and name not in seen_names:
                metric_names.append(name)
                seen_names.add(name)

    aligned_rows: list[dict[str, Any]] = []
    for name in metric_names:
        # Resolve unit and better direction (prefer baseline, fallback to other runs)
        base_val, base_unit, base_better = _parse_metric_entry((baseline_entry or {}).get(name))
        unit = base_unit
        better = base_better

        for _, _, m_dict in items_metrics:
            _, u, b = _parse_metric_entry((m_dict or {}).get(name))
            if unit is None and u is not None:
                unit = u
            if better is None and b is not None:
                better = b

        eff_threshold = (
            float(metric_thresholds[name])
            if (metric_thresholds and name in metric_thresholds)
            else float(threshold)
        )

        values_list: list[dict[str, Any]] = []
        for run_id, result_id, m_dict in items_metrics:
            val, _, _ = _parse_metric_entry((m_dict or {}).get(name))
            is_base = (run_id, result_id) == baseline_key
            present = val is not None

            if is_base:
                if present:
                    abs_delta: float | None = 0.0
                    pct_delta: float | None = 0.0
                    regression = False
                    improvement = False
                    flag = "baseline"
                else:
                    abs_delta = None
                    pct_delta = None
                    regression = False
                    improvement = False
                    flag = "missing"
            else:
                delta_info = compute_metric_delta(
                    baseline_value=base_val,
                    current_value=val,
                    better=better,
                    threshold=eff_threshold,
                )
                abs_delta = delta_info["abs_delta"]
                pct_delta = delta_info["pct_delta"]
                regression = delta_info["regression"]
                improvement = delta_info["improvement"]
                flag = delta_info["flag"]

            values_list.append({
                "run_id": run_id,
                "result_id": result_id,
                "is_baseline": is_base,
                "value": val,
                "present": present,
                "delta": abs_delta,
                "abs_delta": abs_delta,
                "delta_abs": abs_delta,
                "pct_delta": pct_delta,
                "delta_pct": pct_delta,
                "percent_delta": pct_delta,
                "regression": regression,
                "is_regression": regression,
                "improvement": improvement,
                "is_improvement": improvement,
                "flag": flag,
                "status": flag,
            })

        non_base_cells = [c for c in values_list if not c["is_baseline"]]
        has_regression = any(c["regression"] for c in non_base_cells)
        has_improvement = any(c["improvement"] for c in non_base_cells)
        missing_in_some = any(not c["present"] for c in values_list)
        has_change = (
            any(c["flag"] in ("regression", "improvement", "changed") for c in non_base_cells)
            or missing_in_some
        )

        first_comp = non_base_cells[0] if non_base_cells else None
        primary_abs_delta = first_comp["abs_delta"] if first_comp else 0.0
        primary_pct_delta = first_comp["pct_delta"] if first_comp else 0.0
        primary_target_val = first_comp["value"] if first_comp else base_val
        if len(non_base_cells) == 1 and first_comp is not None:
            row_flag = first_comp["flag"]
        elif has_regression:
            row_flag = "regression"
        elif has_improvement:
            row_flag = "improvement"
        elif missing_in_some:
            row_flag = "missing"
        elif has_change:
            row_flag = "changed"
        else:
            row_flag = "unchanged"

        aligned_rows.append({
            "name": name,
            "metric": name,
            "unit": unit,
            "better": better,
            "threshold": eff_threshold,
            "baseline_run_id": baseline_run_id,
            "baseline_result_id": baseline_result_id,
            "baseline_value": base_val,
            "target_value": primary_target_val,
            "delta": primary_abs_delta,
            "abs_delta": primary_abs_delta,
            "delta_abs": primary_abs_delta,
            "pct_delta": primary_pct_delta,
            "delta_pct": primary_pct_delta,
            "percent_delta": primary_pct_delta,
            "regression": has_regression,
            "is_regression": has_regression,
            "has_regression": has_regression,
            "improvement": has_improvement,
            "is_improvement": has_improvement,
            "has_improvement": has_improvement,
            "changed": has_change,
            "has_change": has_change,
            "missing_in_some": missing_in_some,
            "flag": row_flag,
            "status": row_flag,
            "values": values_list,
            "comparisons": non_base_cells,
            "by_run_id": {str(c["run_id"]): c for c in values_list},
            "by_result_id": {str(c["result_id"]): c for c in values_list},
        })

    return aligned_rows


def _extract_model_agnostic_case_id(case: CaseResult) -> str | None:
    """Return a model-agnostic case_id if case_id embeds detail['model'] as '{suite}:{model}:{suffix}'."""
    detail = case.detail if isinstance(case.detail, dict) else {}
    model = detail.get("model")
    if not isinstance(model, str) or not model.strip():
        return None
    prefix = f"{case.suite}:{model}:"
    if case.case_id.startswith(prefix):
        suffix = case.case_id[len(prefix):]
        if suffix:
            return f"{case.suite}:{suffix}"
    return None


def align_and_compare_cases(
    items: list[ComparedItem],
    baseline_item: ComparedItem,
    threshold: float = 0.0,
    metric_thresholds: dict[str, float] | None = None,
    suite_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Align per-case results across 2+ runs/results and compute status and metric diffs."""
    # Determine whether model-parameterized case_ids (e.g. 'transcriptions_json:<model>:json')
    # should be normalized to align single-model runs across different models.
    exact_counts: dict[tuple[str, str], int] = {}
    agnostic_per_item_counts: dict[tuple[int, int], dict[tuple[str, str], int]] = {}
    agnostic_global_counts: dict[tuple[str, str], int] = {}

    for item in items:
        item_ag_counts: dict[tuple[str, str], int] = {}
        seen_exact_in_item: set[tuple[str, str]] = set()
        for c in item.result.cases:
            ex_key = (c.suite, c.case_id)
            if ex_key not in seen_exact_in_item:
                exact_counts[ex_key] = exact_counts.get(ex_key, 0) + 1
                seen_exact_in_item.add(ex_key)

            ag_id = _extract_model_agnostic_case_id(c)
            if ag_id is not None:
                ag_key = (c.suite, ag_id)
                item_ag_counts[ag_key] = item_ag_counts.get(ag_key, 0) + 1
        agnostic_per_item_counts[item.key] = item_ag_counts
        for ag_key, cnt in item_ag_counts.items():
            if cnt == 1:
                agnostic_global_counts[ag_key] = agnostic_global_counts.get(ag_key, 0) + 1

    def resolve_alignment_key(c: CaseResult) -> tuple[str, str]:
        ex_key = (c.suite, c.case_id)
        ag_id = _extract_model_agnostic_case_id(c)
        if ag_id is not None:
            ag_key = (c.suite, ag_id)
            # Normalize only if exact case_id is not shared across runs,
            # every run has at most 1 case for this agnostic key,
            # and at least 2 runs share the agnostic key.
            if (
                exact_counts.get(ex_key, 0) == 1
                and agnostic_global_counts.get(ag_key, 0) >= 2
                and all(
                    agnostic_per_item_counts[it.key].get(ag_key, 0) <= 1
                    for it in items
                )
            ):
                return ag_key
        return ex_key

    # Map each item's cases by alignment key and collect ordered union of keys (baseline first)
    item_case_maps: dict[tuple[int, int], dict[tuple[str, str], CaseResult]] = {}
    ordered_keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    for c in baseline_item.result.cases:
        k = resolve_alignment_key(c)
        if k not in seen_keys:
            ordered_keys.append(k)
            seen_keys.add(k)

    for item in items:
        cmap: dict[tuple[str, str], CaseResult] = {}
        for c in item.result.cases:
            k = resolve_alignment_key(c)
            cmap[k] = c
            if k not in seen_keys:
                ordered_keys.append(k)
                seen_keys.add(k)
        item_case_maps[item.key] = cmap

    baseline_cmap = item_case_maps[baseline_item.key]
    aligned_cases: list[dict[str, Any]] = []

    for suite, aligned_case_id in ordered_keys:
        if suite_filter and suite != suite_filter:
            continue

        key = (suite, aligned_case_id)
        base_case = baseline_cmap.get(key)
        baseline_status = base_case.status if base_case is not None else None

        # Align metrics for this case across all items
        case_items_metrics: list[tuple[int, int, dict[str, Any]]] = []
        for item in items:
            c_obj = item_case_maps[item.key].get(key)
            c_metrics = (c_obj.metrics or {}) if c_obj is not None else {}
            case_items_metrics.append((item.run_id, item.result_id, c_metrics))

        case_metric_rows = align_and_compare_metrics(
            items_metrics=case_items_metrics,
            baseline_key=baseline_item.key,
            threshold=threshold,
            metric_thresholds=metric_thresholds,
        )

        # Build per-item metric delta lookup: {(run_id, result_id): {metric_name: cell_with_meta}}
        per_item_metric_deltas: dict[tuple[int, int], dict[str, dict[str, Any]]] = {
            item.key: {} for item in items
        }
        for m_row in case_metric_rows:
            m_name = m_row["name"]
            for cell in m_row["values"]:
                cell_key = (cell["run_id"], cell["result_id"])
                per_item_metric_deltas[cell_key][m_name] = {
                    **cell,
                    "name": m_name,
                    "unit": m_row["unit"],
                    "better": m_row["better"],
                    "baseline_value": m_row["baseline_value"],
                }

        case_cells: list[dict[str, Any]] = []
        for item in items:
            c_obj = item_case_maps[item.key].get(key)
            is_base = item.key == baseline_item.key
            present = c_obj is not None
            curr_status = c_obj.status if c_obj is not None else None

            if is_base:
                if present:
                    status_info = {
                        "status_change": "baseline",
                        "status_changed": False,
                        "status_regression": False,
                        "status_improvement": False,
                    }
                else:
                    status_info = {
                        "status_change": "missing",
                        "status_changed": False,
                        "status_regression": False,
                        "status_improvement": False,
                    }
            else:
                status_info = compute_case_status_diff(baseline_status, curr_status)

            m_deltas = per_item_metric_deltas[item.key]
            cell_metric_regression = any(
                d["regression"] for d in m_deltas.values() if not is_base
            )
            cell_metric_improvement = any(
                d["improvement"] for d in m_deltas.values() if not is_base
            )

            case_cells.append({
                "run_id": item.run_id,
                "result_id": item.result_id,
                "case_result_id": c_obj.id if c_obj is not None else None,
                "original_case_id": c_obj.case_id if c_obj is not None else None,
                "is_baseline": is_base,
                "present": present,
                "status": curr_status,
                "status_change": status_info["status_change"],
                "status_changed": status_info["status_changed"],
                "status_regression": status_info["status_regression"],
                "status_improvement": status_info["status_improvement"],
                "metric_regression": cell_metric_regression,
                "metric_improvement": cell_metric_improvement,
                "regression": status_info["status_regression"] or cell_metric_regression,
                "improvement": status_info["status_improvement"] or cell_metric_improvement,
                "metrics": _sanitize_dict(c_obj.metrics) if c_obj is not None else {},
                "metric_deltas": m_deltas,
                "detail": _sanitize_dict(c_obj.detail) if c_obj is not None else None,
            })

        non_base_cells = [c for c in case_cells if not c["is_baseline"]]
        status_changed_any = any(c["status_changed"] for c in non_base_cells)
        has_status_regression = any(c["status_regression"] for c in non_base_cells)
        has_status_improvement = any(c["status_improvement"] for c in non_base_cells)
        has_metric_regression = any(m["has_regression"] for m in case_metric_rows)
        has_metric_improvement = any(m["has_improvement"] for m in case_metric_rows)
        has_metric_change = any(m["has_change"] for m in case_metric_rows)
        missing_in_some = any(not c["present"] for c in case_cells)

        has_regression = has_status_regression or has_metric_regression
        has_improvement = has_status_improvement or has_metric_improvement
        has_change = status_changed_any or has_metric_change or missing_in_some

        first_comp = non_base_cells[0] if non_base_cells else None
        primary_status_change = (
            first_comp["status_change"]
            if (len(non_base_cells) == 1 and first_comp is not None)
            else (
                "regression"
                if has_status_regression
                else "improvement"
                if has_status_improvement
                else "missing"
                if missing_in_some
                else "changed"
                if status_changed_any
                else "unchanged"
            )
        )

        aligned_cases.append({
            "case_id": aligned_case_id,
            "suite": suite,
            "baseline_run_id": baseline_item.run_id,
            "baseline_result_id": baseline_item.result_id,
            "baseline_status": baseline_status,
            "target_status": first_comp["status"] if first_comp else baseline_status,
            "status_change": primary_status_change,
            "status_changed": status_changed_any,
            "status_regression": has_status_regression,
            "status_improvement": has_status_improvement,
            "has_status_regression": has_status_regression,
            "has_status_improvement": has_status_improvement,
            "has_metric_regression": has_metric_regression,
            "has_metric_improvement": has_metric_improvement,
            "has_metric_change": has_metric_change,
            "regression": has_regression,
            "is_regression": has_regression,
            "has_regression": has_regression,
            "improvement": has_improvement,
            "is_improvement": has_improvement,
            "has_improvement": has_improvement,
            "changed": has_change,
            "has_change": has_change,
            "missing_in_some": missing_in_some,
            "metrics": case_metric_rows,
            "metric_changes": {m["name"]: m for m in case_metric_rows},
            "values": case_cells,
            "comparisons": non_base_cells,
            "by_run_id": {str(c["run_id"]): c for c in case_cells},
            "by_result_id": {str(c["result_id"]): c for c in case_cells},
        })

    return aligned_cases


def _extract_item_metadata(item: ComparedItem, is_baseline: bool) -> dict[str, Any]:
    """Extract target, model, config, and artifact metadata for a compared run/result."""
    run = item.run
    result = item.result

    raw_target = run.target_snapshot if isinstance(run.target_snapshot, dict) else {}
    target_snap = _sanitize_dict(raw_target)

    raw_config_snap = run.config_snapshot
    if isinstance(raw_config_snap, list) and raw_config_snap:
        first_snap = raw_config_snap[0] if isinstance(raw_config_snap[0], dict) else {}
        config_snap = _sanitize_dict(first_snap)
    elif isinstance(raw_config_snap, dict):
        config_snap = _sanitize_dict(raw_config_snap)
    else:
        config_snap = {}

    inner_config = (
        config_snap.get("config", {})
        if isinstance(config_snap.get("config"), dict)
        else {}
    )
    target_extra = (
        target_snap.get("extra", {})
        if isinstance(target_snap.get("extra"), dict)
        else {}
    )

    # Resolve model identifier across config, target.extra, target, or case details
    case_models: list[str] = []
    for c in result.cases:
        if isinstance(c.detail, dict):
            cm = c.detail.get("model")
            if isinstance(cm, str) and cm.strip() and cm not in case_models:
                case_models.append(cm.strip())

    model = (
        inner_config.get("model")
        or inner_config.get("transcription_model")
        or target_extra.get("model")
        or target_extra.get("transcription_model")
        or target_snap.get("model")
    )
    if not model and case_models:
        model = case_models[0] if len(case_models) == 1 else ", ".join(case_models)

    suites = config_snap.get("suites")
    if not isinstance(suites, list):
        suites = inner_config.get("suites") if isinstance(inner_config.get("suites"), list) else []

    artifacts_list = [
        {
            "id": a.id,
            "result_id": result.id,
            "run_id": run.id,
            "name": a.name,
            "path": a.path,
            "mime": a.mime,
            "size_bytes": a.size_bytes,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "download_url": f"/api/runs/{run.id}/artifacts/{a.id}/download",
        }
        for a in result.artifacts
    ]

    return {
        "run_id": run.id,
        "result_id": result.id,
        "job_id": run.job_id,
        "test_definition_id": result.test_definition_id or config_snap.get("test_definition_id"),
        "is_baseline": is_baseline,
        "tester": result.tester,
        "status": result.status,
        "result_status": result.status,
        "run_status": run.status,
        "summary": result.summary,
        "target_id": target_snap.get("id"),
        "target_name": target_snap.get("name"),
        "target_kind": target_snap.get("kind"),
        "base_url": target_snap.get("base_url") or inner_config.get("base_url"),
        "model": str(model) if model is not None else None,
        "models": case_models or ([str(model)] if model is not None else []),
        "suites": suites,
        "config": inner_config,
        "config_snapshot": config_snap,
        "target_snapshot": target_snap,
        "git_info": run.git_info,
        "queued_at": run.queued_at.isoformat() if run.queued_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "case_count": len(result.cases),
        "artifact_count": len(artifacts_list),
        "artifacts": artifacts_list,
    }


def _compute_config_and_target_diff(runs_meta: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute differences in target, model, suites, and config keys across compared runs."""
    target_names = [r.get("target_name") for r in runs_meta]
    base_urls = [r.get("base_url") for r in runs_meta]
    models = [r.get("model") for r in runs_meta]
    suites_list = [r.get("suites") for r in runs_meta]

    targets_differ = len({(r.get("target_id"), r.get("target_name"), r.get("base_url")) for r in runs_meta}) > 1
    models_differ = len({m for m in models}) > 1
    suites_differ = len({json_canonical(s) for s in suites_list}) > 1

    all_config_keys: list[str] = []
    seen_cfg_keys: set[str] = set()
    for r in runs_meta:
        cfg = r.get("config") if isinstance(r.get("config"), dict) else {}
        for k in cfg.keys():
            if k not in seen_cfg_keys:
                all_config_keys.append(k)
                seen_cfg_keys.add(k)

    config_keys_changed: list[str] = []
    config_changes: dict[str, Any] = {}
    for k in all_config_keys:
        vals_per_run = [
            {
                "run_id": r["run_id"],
                "result_id": r["result_id"],
                "value": (r.get("config") or {}).get(k),
            }
            for r in runs_meta
        ]
        distinct_vals = {json_canonical(v["value"]) for v in vals_per_run}
        if len(distinct_vals) > 1:
            config_keys_changed.append(k)
            config_changes[k] = vals_per_run

    return {
        "targets_differ": targets_differ,
        "models_differ": models_differ,
        "suites_differ": suites_differ,
        "config_keys_changed": config_keys_changed,
        "config_changes": config_changes,
        "targets": [
            {
                "run_id": r["run_id"],
                "result_id": r["result_id"],
                "target_id": r.get("target_id"),
                "target_name": r.get("target_name"),
                "base_url": r.get("base_url"),
                "model": r.get("model"),
            }
            for r in runs_meta
        ],
    }


def json_canonical(val: Any) -> str:
    """Serialize arbitrary JSON value deterministically for equality comparison."""
    import json
    try:
        return json.dumps(val, sort_keys=True, default=str)
    except Exception:
        return repr(val)


def compare_runs_and_results(
    db: Session,
    run_ids: list[int] | None = None,
    result_ids: list[int] | None = None,
    baseline_run_id: int | None = None,
    baseline_result_id: int | None = None,
    baseline_id: int | None = None,
    threshold: float = 0.0,
    metric_thresholds: dict[str, float] | None = None,
    tester: str | None = None,
    suite: str | None = None,
    changed_only: bool = False,
    regressed_only: bool = False,
    allow_different_testers: bool = False,
) -> dict[str, Any]:
    """Resolve runs/results from DB, align metrics and cases, and compute deltas."""
    req_run_ids = list(run_ids or [])
    req_result_ids = list(result_ids or [])

    if baseline_id is not None and baseline_run_id is None and baseline_result_id is None:
        if req_result_ids and not req_run_ids:
            baseline_result_id = baseline_id
        elif baseline_id in req_result_ids and baseline_id not in req_run_ids:
            baseline_result_id = baseline_id
        else:
            baseline_run_id = baseline_id

    if baseline_run_id is not None and baseline_run_id not in req_run_ids:
        req_run_ids = [baseline_run_id] + req_run_ids
    if baseline_result_id is not None and baseline_result_id not in req_result_ids:
        req_result_ids = [baseline_result_id] + req_result_ids

    if not req_run_ids and not req_result_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide at least 2 run_ids or result_ids to compare",
        )

    # Load all requested runs
    runs_by_id: dict[int, Run] = {}
    if req_run_ids:
        stmt_runs = (
            select(Run)
            .options(
                selectinload(Run.results).selectinload(Result.cases),
                selectinload(Run.results).selectinload(Result.artifacts),
            )
            .where(Run.id.in_(req_run_ids))
        )
        for r in db.execute(stmt_runs).scalars().all():
            runs_by_id[r.id] = r

        for rid in req_run_ids:
            if rid not in runs_by_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Run {rid} not found",
                )

    # Load all requested results
    results_by_id: dict[int, Result] = {}
    if req_result_ids:
        stmt_results = (
            select(Result)
            .options(
                selectinload(Result.run),
                selectinload(Result.cases),
                selectinload(Result.artifacts),
            )
            .where(Result.id.in_(req_result_ids))
        )
        for res in db.execute(stmt_results).scalars().all():
            results_by_id[res.id] = res

        for resid in req_result_ids:
            if resid not in results_by_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Result {resid} not found",
                )

    # Infer target tester if not explicitly provided and runs have multiple results
    inferred_tester = tester
    if not inferred_tester:
        if req_result_ids:
            first_res = results_by_id[req_result_ids[0]]
            inferred_tester = first_res.tester
        elif req_run_ids:
            first_run = runs_by_id[req_run_ids[0]]
            if len(first_run.results) == 1:
                inferred_tester = first_run.results[0].tester

    resolved_items: list[ComparedItem] = []
    seen_item_keys: set[tuple[int, int]] = set()

    for rid in req_run_ids:
        run_obj = runs_by_id[rid]
        if not run_obj.results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Run {rid} has no results to compare",
            )
        if tester:
            matching = [r for r in run_obj.results if r.tester == tester]
            if not matching:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Run {rid} has no result for tester '{tester}'",
                )
            chosen_res = matching[0]
        elif len(run_obj.results) == 1:
            chosen_res = run_obj.results[0]
        elif inferred_tester:
            matching = [r for r in run_obj.results if r.tester == inferred_tester]
            chosen_res = matching[0] if matching else run_obj.results[0]
        else:
            chosen_res = run_obj.results[0]

        item = ComparedItem(run=run_obj, result=chosen_res)
        if item.key not in seen_item_keys:
            resolved_items.append(item)
            seen_item_keys.add(item.key)

    for resid in req_result_ids:
        res_obj = results_by_id[resid]
        if tester and res_obj.tester != tester:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Result {resid} has tester '{res_obj.tester}', expected '{tester}'",
            )
        item = ComparedItem(run=res_obj.run, result=res_obj)
        if item.key not in seen_item_keys:
            resolved_items.append(item)
            seen_item_keys.add(item.key)

    if len(resolved_items) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 distinct runs or results are required for comparison",
        )

    # Validate same tester across all resolved items
    distinct_testers = {it.result.tester for it in resolved_items}
    if len(distinct_testers) > 1 and not allow_different_testers:
        sorted_testers = ", ".join(sorted(distinct_testers))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot compare runs from different testers ({sorted_testers}). All compared runs must use the same tester.",
        )

    # Determine baseline item
    baseline_item: ComparedItem | None = None
    if baseline_result_id is not None:
        baseline_item = next(
            (it for it in resolved_items if it.result_id == baseline_result_id),
            None,
        )
    elif baseline_run_id is not None:
        baseline_item = next(
            (it for it in resolved_items if it.run_id == baseline_run_id),
            None,
        )
    if baseline_item is None:
        baseline_item = resolved_items[0]

    # Align top-level metrics
    items_metrics = [
        (it.run_id, it.result_id, it.result.metrics or {})
        for it in resolved_items
    ]
    all_aligned_metrics = align_and_compare_metrics(
        items_metrics=items_metrics,
        baseline_key=baseline_item.key,
        threshold=threshold,
        metric_thresholds=metric_thresholds,
    )

    # Align per-case results and per-case metrics
    all_aligned_cases = align_and_compare_cases(
        items=resolved_items,
        baseline_item=baseline_item,
        threshold=threshold,
        metric_thresholds=metric_thresholds,
        suite_filter=suite,
    )

    # Build summary statistics across all aligned metrics and cases (before optional display filtering)
    metrics_regressed = sum(1 for m in all_aligned_metrics if m["has_regression"])
    metrics_improved = sum(1 for m in all_aligned_metrics if m["has_improvement"])
    metrics_missing = sum(1 for m in all_aligned_metrics if m["missing_in_some"])
    metrics_changed = sum(1 for m in all_aligned_metrics if m["has_change"])
    metrics_unchanged = len(all_aligned_metrics) - metrics_changed

    cases_regressed = sum(1 for c in all_aligned_cases if c["has_regression"])
    cases_status_regressed = sum(1 for c in all_aligned_cases if c["has_status_regression"])
    cases_metric_regressed = sum(1 for c in all_aligned_cases if c["has_metric_regression"])
    cases_improved = sum(1 for c in all_aligned_cases if c["has_improvement"])
    cases_status_improved = sum(1 for c in all_aligned_cases if c["has_status_improvement"])
    cases_status_changed = sum(1 for c in all_aligned_cases if c["status_changed"])
    cases_missing = sum(1 for c in all_aligned_cases if c["missing_in_some"])
    cases_changed = sum(1 for c in all_aligned_cases if c["has_change"])
    cases_unchanged = len(all_aligned_cases) - cases_changed

    regression_details: list[dict[str, Any]] = []
    for m in all_aligned_metrics:
        if m["has_regression"]:
            for cell in m["comparisons"]:
                if cell["regression"]:
                    regression_details.append({
                        "type": "metric",
                        "name": m["name"],
                        "unit": m["unit"],
                        "better": m["better"],
                        "baseline_run_id": baseline_item.run_id,
                        "baseline_result_id": baseline_item.result_id,
                        "baseline_value": m["baseline_value"],
                        "run_id": cell["run_id"],
                        "result_id": cell["result_id"],
                        "value": cell["value"],
                        "abs_delta": cell["abs_delta"],
                        "pct_delta": cell["pct_delta"],
                    })
    for c in all_aligned_cases:
        if c["has_status_regression"]:
            for cell in c["comparisons"]:
                if cell["status_regression"]:
                    regression_details.append({
                        "type": "case_status",
                        "suite": c["suite"],
                        "case_id": c["case_id"],
                        "baseline_run_id": baseline_item.run_id,
                        "baseline_result_id": baseline_item.result_id,
                        "baseline_status": c["baseline_status"],
                        "run_id": cell["run_id"],
                        "result_id": cell["result_id"],
                        "status": cell["status"],
                    })

    summary = {
        "has_regression": (metrics_regressed > 0) or (cases_regressed > 0),
        "has_improvement": (metrics_improved > 0) or (cases_improved > 0),
        "total_metrics": len(all_aligned_metrics),
        "metrics_changed": metrics_changed,
        "metrics_regressed": metrics_regressed,
        "metrics_improved": metrics_improved,
        "metrics_unchanged": metrics_unchanged,
        "metrics_missing": metrics_missing,
        "total_cases": len(all_aligned_cases),
        "cases_changed": cases_changed,
        "cases_status_changed": cases_status_changed,
        "cases_regressed": cases_regressed,
        "cases_status_regressed": cases_status_regressed,
        "cases_metric_regressed": cases_metric_regressed,
        "cases_improved": cases_improved,
        "cases_status_improved": cases_status_improved,
        "cases_unchanged": cases_unchanged,
        "cases_missing": cases_missing,
        "regressions": regression_details,
    }

    # Apply optional changed_only / regressed_only filters
    filtered_metrics = all_aligned_metrics
    filtered_cases = all_aligned_cases
    if regressed_only:
        filtered_metrics = [m for m in filtered_metrics if m["has_regression"]]
        filtered_cases = [c for c in filtered_cases if c["has_regression"]]
    elif changed_only:
        filtered_metrics = [m for m in filtered_metrics if m["has_change"]]
        filtered_cases = [c for c in filtered_cases if c["has_change"]]

    runs_meta = [
        _extract_item_metadata(it, is_baseline=(it.key == baseline_item.key))
        for it in resolved_items
    ]
    config_diff = _compute_config_and_target_diff(runs_meta)

    common_tester = (
        baseline_item.result.tester
        if len(distinct_testers) == 1
        else ",".join(sorted(distinct_testers))
    )

    return {
        "tester": common_tester,
        "threshold": float(threshold),
        "baseline_run_id": baseline_item.run_id,
        "baseline_result_id": baseline_item.result_id,
        "run_ids": [it.run_id for it in resolved_items],
        "result_ids": [it.result_id for it in resolved_items],
        "runs": runs_meta,
        "config_diff": config_diff,
        "summary": summary,
        "metrics": filtered_metrics,
        "metric_map": {m["name"]: m for m in filtered_metrics},
        "cases": filtered_cases,
        "case_map": {c["case_id"]: c for c in filtered_cases},
    }


@router.get("")
def get_compare(
    run_ids: list[str] | None = Query(
        None,
        description="Run IDs to compare (comma-separated or repeated, e.g. ?run_ids=1,2 or ?run_ids=1&run_ids=2)",
    ),
    runs: list[str] | None = Query(
        None,
        description="Alias for run_ids",
    ),
    result_ids: list[str] | None = Query(
        None,
        description="Result IDs to compare (comma-separated or repeated, e.g. ?result_ids=1,2 or ?result_ids=1&result_ids=2)",
    ),
    results: list[str] | None = Query(
        None,
        description="Alias for result_ids",
    ),
    baseline_run_id: int | None = Query(
        None,
        description="Optional Run ID to use as baseline (defaults to first item)",
    ),
    baseline_result_id: int | None = Query(
        None,
        description="Optional Result ID to use as baseline (defaults to first item)",
    ),
    baseline_id: int | None = Query(
        None,
        description="Optional alias for baseline_run_id or baseline_result_id",
    ),
    threshold: float = Query(
        0.0,
        ge=0.0,
        description="Percentage delta threshold (e.g. 5.0 for 5%) required to flag a regression or improvement",
    ),
    threshold_pct: float | None = Query(
        None,
        ge=0.0,
        description="Optional alias for threshold",
    ),
    tester: str | None = Query(
        None,
        description="Optional tester filter when a run contains multiple results",
    ),
    suite: str | None = Query(
        None,
        description="Optional suite filter for per-case diffs",
    ),
    changed_only: bool = Query(
        False,
        description="When true, return only metrics and cases that changed or have missing entries",
    ),
    regressed_only: bool = Query(
        False,
        description="When true, return only metrics and cases with regressions",
    ),
    allow_different_testers: bool = Query(
        False,
        description="When true, allow comparing results from different testers",
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Compare 2+ runs or results via GET query parameters."""
    parsed_run_ids = _parse_id_list_param(run_ids, "run_ids") + _parse_id_list_param(runs, "runs")
    parsed_result_ids = _parse_id_list_param(result_ids, "result_ids") + _parse_id_list_param(results, "results")
    eff_threshold = threshold_pct if threshold_pct is not None else threshold

    return compare_runs_and_results(
        db=db,
        run_ids=parsed_run_ids,
        result_ids=parsed_result_ids,
        baseline_run_id=baseline_run_id,
        baseline_result_id=baseline_result_id,
        baseline_id=baseline_id,
        threshold=eff_threshold,
        tester=tester,
        suite=suite,
        changed_only=changed_only,
        regressed_only=regressed_only,
        allow_different_testers=allow_different_testers,
    )


@router.post("")
def post_compare(
    req: CompareRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Compare 2+ runs or results via POST JSON body."""
    eff_threshold = req.threshold_pct if req.threshold_pct is not None else req.threshold

    return compare_runs_and_results(
        db=db,
        run_ids=req.run_ids,
        result_ids=req.result_ids,
        baseline_run_id=req.baseline_run_id,
        baseline_result_id=req.baseline_result_id,
        baseline_id=req.baseline_id,
        threshold=eff_threshold,
        metric_thresholds=req.metric_thresholds,
        tester=req.tester,
        suite=req.suite,
        changed_only=req.changed_only,
        regressed_only=req.regressed_only,
        allow_different_testers=req.allow_different_testers,
    )
