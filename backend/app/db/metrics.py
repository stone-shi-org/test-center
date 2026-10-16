from typing import Any, Literal

BetterType = Literal["higher", "lower"]
VALID_BETTER_VALUES = {"higher", "lower"}


def validate_metric(metric_data: Any) -> None:
    """Validate a single metric item dictionary.
    
    Expected format:
    {"value": <number>, "unit": <str | optional>, "better": "higher" | "lower" | optional}
    """
    if not isinstance(metric_data, dict):
        raise ValueError(f"Metric entry must be a dictionary, got {type(metric_data).__name__}")
    if "value" not in metric_data:
        raise ValueError("Metric entry must contain a 'value' field")
    if not isinstance(metric_data["value"], (int, float)):
        raise ValueError(f"Metric 'value' must be a number, got {type(metric_data['value']).__name__}")
    if "unit" in metric_data and metric_data["unit"] is not None and not isinstance(metric_data["unit"], str):
        raise ValueError(f"Metric 'unit' must be a string or None, got {type(metric_data['unit']).__name__}")
    if "better" in metric_data and metric_data["better"] is not None:
        if metric_data["better"] not in VALID_BETTER_VALUES:
            raise ValueError(
                f"Metric 'better' must be 'higher' or 'lower', got {metric_data['better']!r}"
            )


def format_metric(
    value: float | int,
    unit: str | None = None,
    better: BetterType | str | None = None,
) -> dict[str, Any]:
    """Helper to format a metric entry with value, unit, and better direction."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"Metric 'value' must be a number, got {type(value).__name__}")
    if better is not None and better not in VALID_BETTER_VALUES:
        raise ValueError(f"Metric 'better' must be 'higher' or 'lower', got {better!r}")
    
    entry: dict[str, Any] = {"value": value}
    if unit is not None:
        entry["unit"] = unit
    if better is not None:
        entry["better"] = better
    return entry


def validate_metrics_dict(metrics: Any) -> None:
    """Validate a dictionary of metrics {metric_name: metric_entry}."""
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics must be a dictionary, got {type(metrics).__name__}")
    for name, data in metrics.items():
        if not isinstance(name, str):
            raise ValueError(f"Metric name must be a string, got {type(name).__name__}")
        validate_metric(data)
