"""Cron expression parser, validator, human-readable describer, and next-run calculator.

Supports standard 5-field cron expressions:
    <minute> <hour> <day_of_month> <month> <day_of_week>

Field ranges:
    - minute:       0-59
    - hour:         0-23
    - day_of_month: 1-31
    - month:        1-12 (or JAN-DEC)
    - day_of_week:  0-7 (0 and 7 are Sunday, 1=Mon .. 6=Sat; or SUN-SAT)

Also supports standard macros:
    @yearly, @annually, @monthly, @weekly, @daily, @midnight, @hourly
and manual mode ('manual', '', 'none').
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

CRON_MACROS: dict[str, str] = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}

MONTH_NAMES: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

DOW_NAMES: dict[str, int] = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}

DOW_LABELS: dict[int, str] = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

MONTH_LABELS: dict[int, str] = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

MAX_DAYS_PER_MONTH: dict[int, int] = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

MANUAL_VALUES = {"", "manual", "none", "off", "disabled"}


@dataclass(frozen=True)
class ParsedCron:
    raw: str
    normalized: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_is_wildcard: bool
    dow_is_wildcard: bool


def is_manual_schedule(schedule: str | None) -> bool:
    """Return True if the schedule string represents manual-only execution."""
    if schedule is None:
        return True
    return schedule.strip().lower() in MANUAL_VALUES


def normalize_schedule(schedule: str | None) -> str:
    """Normalize a schedule string ('manual' or validated cron expression)."""
    if is_manual_schedule(schedule):
        return "manual"
    assert schedule is not None
    cleaned = " ".join(schedule.strip().split())
    parsed = parse_cron_expression(cleaned)
    return parsed.raw


def _parse_token_value(
    token: str,
    field_name: str,
    min_val: int,
    max_val: int,
    names: dict[str, int] | None = None,
    is_range_end_dow: bool = False,
) -> int:
    t = token.strip().lower()
    if not t:
        raise ValueError(f"Empty value in {field_name} field")

    if names and t in names:
        val = names[t]
        if is_range_end_dow and val == 0:
            val = 7
    else:
        try:
            val = int(t)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value '{token}' in {field_name} field (expected integer {min_val}-{max_val})"
            ) from exc

    if val < min_val or val > max_val:
        raise ValueError(
            f"Value {val} out of range in {field_name} field (allowed {min_val}-{max_val})"
        )
    return val


def _parse_cron_field(
    field_str: str,
    field_name: str,
    min_val: int,
    max_val: int,
    names: dict[str, int] | None = None,
    is_dow: bool = False,
) -> tuple[frozenset[int], bool]:
    """Parse a single cron field into a frozenset of matching integers and a wildcard flag."""
    raw = field_str.strip()
    if not raw:
        raise ValueError(f"Empty {field_name} field in cron expression")

    is_wildcard = raw == "*"
    values: set[int] = set()

    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Empty list item in {field_name} field: '{field_str}'")

        step = 1
        if "/" in part:
            slash_parts = part.split("/")
            if len(slash_parts) != 2 or not slash_parts[0] or not slash_parts[1]:
                raise ValueError(f"Invalid step syntax '{part}' in {field_name} field")
            base_part, step_str = slash_parts[0].strip(), slash_parts[1].strip()
            try:
                step = int(step_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid step value '{step_str}' in {field_name} field"
                ) from exc
            if step <= 0:
                raise ValueError(
                    f"Step value must be positive in {field_name} field, got {step}"
                )
        else:
            base_part = part

        if base_part == "*":
            start = min_val
            end = 6 if is_dow else max_val
        elif "-" in base_part:
            range_parts = base_part.split("-")
            if len(range_parts) != 2 or not range_parts[0] or not range_parts[1]:
                raise ValueError(f"Invalid range syntax '{base_part}' in {field_name} field")
            start = _parse_token_value(
                range_parts[0], field_name, min_val, max_val, names=names
            )
            end = _parse_token_value(
                range_parts[1],
                field_name,
                min_val,
                max_val,
                names=names,
                is_range_end_dow=(is_dow and start > 0),
            )
            if start > end:
                raise ValueError(
                    f"Invalid range '{base_part}' in {field_name} field: start ({start}) > end ({end})"
                )
        else:
            single_val = _parse_token_value(
                base_part, field_name, min_val, max_val, names=names
            )
            if "/" in part:
                start = single_val
                end = 6 if is_dow else max_val
            else:
                start = single_val
                end = single_val

        for v in range(start, end + 1, step):
            values.add(v % 7 if is_dow else v)

    if not values:
        raise ValueError(f"Field {field_name} matched no valid values: '{field_str}'")

    return frozenset(values), is_wildcard


def parse_cron_expression(expression: str) -> ParsedCron:
    """Parse and validate a cron expression string.

    Raises ValueError with a descriptive message if the expression is invalid.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Cron expression cannot be empty")

    raw = " ".join(expression.strip().split())
    lower = raw.lower()
    if lower in MANUAL_VALUES:
        raise ValueError("Schedule 'manual' is not a cron expression")

    if lower.startswith("@"):
        if lower not in CRON_MACROS:
            supported = ", ".join(sorted(CRON_MACROS.keys()))
            raise ValueError(f"Unknown cron macro '{raw}'. Supported macros: {supported}")
        expanded = CRON_MACROS[lower]
    else:
        expanded = raw

    parts = expanded.split()
    if len(parts) != 5:
        raise ValueError(
            f"Cron expression must have exactly 5 fields (minute hour day_of_month month day_of_week), got {len(parts)} in '{raw}'"
        )

    minutes, _ = _parse_cron_field(parts[0], "minute", 0, 59)
    hours, _ = _parse_cron_field(parts[1], "hour", 0, 23)
    dom, dom_wild = _parse_cron_field(parts[2], "day_of_month", 1, 31)
    months, _ = _parse_cron_field(parts[3], "month", 1, 12, names=MONTH_NAMES)
    dow, dow_wild = _parse_cron_field(
        parts[4], "day_of_week", 0, 7, names=DOW_NAMES, is_dow=True
    )

    # Check that day_of_month and month combination is possible when day_of_month is restricted
    # and day_of_week is wildcard
    if not dom_wild and dow_wild:
        max_possible_dom = max(MAX_DAYS_PER_MONTH[m] for m in months)
        if all(d > max_possible_dom for d in dom):
            raise ValueError(
                f"Impossible day_of_month {sorted(dom)} for selected month(s) {sorted(months)}"
            )

    return ParsedCron(
        raw=raw,
        normalized=expanded,
        minutes=minutes,
        hours=hours,
        days_of_month=dom,
        months=months,
        days_of_week=dow,
        dom_is_wildcard=dom_wild,
        dow_is_wildcard=dow_wild,
    )


def _day_matches(parsed: ParsedCron, year: int, month: int, day: int) -> bool:
    """Evaluate Vixie cron day matching rules for (year, month, day)."""
    try:
        dt_weekday = datetime(year, month, day).weekday()
    except ValueError:
        return False

    cron_dow = (dt_weekday + 1) % 7
    dom_match = day in parsed.days_of_month
    dow_match = cron_dow in parsed.days_of_week

    if parsed.dom_is_wildcard and parsed.dow_is_wildcard:
        return True
    if parsed.dom_is_wildcard:
        return dow_match
    if parsed.dow_is_wildcard:
        return dom_match
    # Standard cron: when both DOM and DOW are explicitly specified, match if either matches
    return dom_match or dow_match


def _ensure_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_next_cron_time(expression: str, after: datetime | None = None) -> datetime:
    """Compute the next UTC occurrence strictly after `after` for a cron expression."""
    parsed = parse_cron_expression(expression)
    base = _ensure_utc(after)

    # Cron has 1-minute resolution; next run is at least the next minute boundary
    candidate = base.replace(second=0, microsecond=0) + timedelta(minutes=1)
    max_year = candidate.year + 8

    sorted_months = sorted(parsed.months)
    sorted_hours = sorted(parsed.hours)
    sorted_minutes = sorted(parsed.minutes)

    while candidate.year <= max_year:
        # 1. Check month
        if candidate.month not in parsed.months:
            next_months = [m for m in sorted_months if m > candidate.month]
            if next_months:
                candidate = candidate.replace(
                    month=next_months[0], day=1, hour=0, minute=0
                )
            else:
                candidate = candidate.replace(
                    year=candidate.year + 1, month=sorted_months[0], day=1, hour=0, minute=0
                )
            continue

        # 2. Check day of month / day of week
        days_in_month = calendar.monthrange(candidate.year, candidate.month)[1]
        if candidate.day > days_in_month or not _day_matches(
            parsed, candidate.year, candidate.month, candidate.day
        ):
            next_day = candidate.day + 1
            while next_day <= days_in_month and not _day_matches(
                parsed, candidate.year, candidate.month, next_day
            ):
                next_day += 1

            if next_day <= days_in_month:
                candidate = candidate.replace(day=next_day, hour=0, minute=0)
            else:
                if candidate.month == 12:
                    candidate = candidate.replace(
                        year=candidate.year + 1, month=1, day=1, hour=0, minute=0
                    )
                else:
                    candidate = candidate.replace(
                        month=candidate.month + 1, day=1, hour=0, minute=0
                    )
            continue

        # 3. Check hour
        if candidate.hour not in parsed.hours:
            next_hours = [h for h in sorted_hours if h > candidate.hour]
            if next_hours:
                candidate = candidate.replace(hour=next_hours[0], minute=0)
            else:
                candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0)
            continue

        # 4. Check minute
        if candidate.minute not in parsed.minutes:
            next_minutes = [m for m in sorted_minutes if m > candidate.minute]
            if next_minutes:
                candidate = candidate.replace(minute=next_minutes[0])
            else:
                candidate = (candidate + timedelta(hours=1)).replace(minute=0)
            continue

        return candidate

    raise ValueError(f"Could not find next execution time for cron expression '{expression}'")


def get_upcoming_cron_times(
    expression: str,
    count: int = 5,
    after: datetime | None = None,
) -> list[datetime]:
    """Return the next `count` scheduled UTC datetimes for a cron expression."""
    if count <= 0 or is_manual_schedule(expression):
        return []
    results: list[datetime] = []
    cursor = _ensure_utc(after)
    for _ in range(count):
        nxt = get_next_cron_time(expression, after=cursor)
        results.append(nxt)
        cursor = nxt
    return results


def count_missed_cron_runs(
    expression: str,
    first_due: datetime,
    now: datetime,
    max_count: int = 1000,
) -> int:
    """Count how many scheduled occurrences elapsed in [first_due, now]."""
    first_utc = _ensure_utc(first_due)
    now_utc = _ensure_utc(now)
    if first_utc > now_utc or is_manual_schedule(expression):
        return 0

    count = 1
    cursor = first_utc
    while count < max_count:
        nxt = get_next_cron_time(expression, after=cursor)
        if nxt > now_utc:
            break
        count += 1
        cursor = nxt
    return count


def describe_cron_expression(schedule: str | None) -> str:
    """Return a concise human-readable description of a schedule or cron expression."""
    if is_manual_schedule(schedule):
        return "Manual trigger only"

    assert schedule is not None
    raw = " ".join(schedule.strip().split())
    lower = raw.lower()
    if lower in ("@hourly", "0 * * * *"):
        return "Every hour (at minute 0)"
    if lower in ("@daily", "@midnight", "0 0 * * *"):
        return "Daily at 00:00 UTC"
    if lower in ("@weekly", "0 0 * * 0"):
        return "Weekly on Sunday at 00:00 UTC"
    if lower in ("@monthly", "0 0 1 * *"):
        return "Monthly on day 1 at 00:00 UTC"
    if lower in ("@yearly", "@annually", "0 0 1 1 *"):
        return "Yearly on Jan 1 at 00:00 UTC"

    try:
        parsed = parse_cron_expression(raw)
    except ValueError:
        return raw

    parts = parsed.normalized.split()
    min_part, hr_part, dom_part, mon_part, dow_part = parts

    if min_part == "*" and hr_part == "*" and dom_part == "*" and mon_part == "*" and dow_part == "*":
        return "Every minute"

    if (
        min_part.startswith("*/")
        and hr_part == "*"
        and dom_part == "*"
        and mon_part == "*"
        and dow_part == "*"
    ):
        step = min_part.split("/")[1]
        return f"Every {step} minutes"

    if (
        min_part.isdigit()
        and hr_part.startswith("*/")
        and dom_part == "*"
        and mon_part == "*"
        and dow_part == "*"
    ):
        step = hr_part.split("/")[1]
        return f"Every {step} hours at :{int(min_part):02d}"

    if len(parsed.minutes) == 1 and len(parsed.hours) == 1:
        minute = next(iter(parsed.minutes))
        hour = next(iter(parsed.hours))
        time_str = f"{hour:02d}:{minute:02d} UTC"

        if dom_part == "*" and mon_part == "*" and dow_part == "*":
            return f"Daily at {time_str}"
        if dom_part == "*" and mon_part == "*" and len(parsed.days_of_week) == 1:
            dow_val = next(iter(parsed.days_of_week))
            return f"Weekly on {DOW_LABELS.get(dow_val, str(dow_val))} at {time_str}"
        if dom_part == "*" and mon_part == "*" and parsed.days_of_week == frozenset({1, 2, 3, 4, 5}):
            return f"Weekdays (Mon-Fri) at {time_str}"
        if len(parsed.days_of_month) == 1 and mon_part == "*" and dow_part == "*":
            dom_val = next(iter(parsed.days_of_month))
            return f"Monthly on day {dom_val} at {time_str}"
        if len(parsed.days_of_month) == 1 and len(parsed.months) == 1 and dow_part == "*":
            dom_val = next(iter(parsed.days_of_month))
            mon_val = next(iter(parsed.months))
            return f"Yearly on {MONTH_LABELS.get(mon_val, str(mon_val))} {dom_val} at {time_str}"

    return f"Cron ({parsed.raw})"


def validate_and_preview_schedule(
    schedule: str | None,
    upcoming_count: int = 5,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate a schedule string ('manual' or cron) and compute upcoming run times."""
    ref_now = _ensure_utc(now)
    if is_manual_schedule(schedule):
        return {
            "valid": True,
            "schedule": schedule or "manual",
            "normalized": "manual",
            "is_manual": True,
            "description": "Manual trigger only",
            "next_run_at": None,
            "upcoming_runs": [],
            "error": None,
        }

    assert schedule is not None
    try:
        parsed = parse_cron_expression(schedule)
        upcoming = get_upcoming_cron_times(parsed.raw, count=max(1, upcoming_count), after=ref_now)
        next_run = upcoming[0].isoformat() if upcoming else None
        return {
            "valid": True,
            "schedule": parsed.raw,
            "normalized": parsed.normalized,
            "is_manual": False,
            "description": describe_cron_expression(parsed.raw),
            "next_run_at": next_run,
            "upcoming_runs": [dt.isoformat() for dt in upcoming],
            "error": None,
        }
    except ValueError as exc:
        return {
            "valid": False,
            "schedule": schedule,
            "normalized": None,
            "is_manual": False,
            "description": None,
            "next_run_at": None,
            "upcoming_runs": [],
            "error": str(exc),
        }
