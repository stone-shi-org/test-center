"""Tests for cron expression validation, next-run calculation, enable/disable,
no overlapping runs, and missed-run policy after downtime.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from backend.app.db.models.job import Job
from backend.app.db.models.run import Run
from backend.app.runner.cron import (
    count_missed_cron_runs,
    describe_cron_expression,
    get_next_cron_time,
    get_upcoming_cron_times,
    parse_cron_expression,
    validate_and_preview_schedule,
)
from backend.app.runner.scheduler import CronScheduler, get_scheduler
from backend.app.runner.worker import get_worker


def _create_def(client, name="Sched Def"):
    res = client.post(
        "/api/test-definitions",
        json={
            "name": name,
            "tester": "dummy",
            "suites": ["smoke"],
            "config": {"duration_seconds": 0.05, "step_delay": 0.01},
        },
    )
    assert res.status_code == 201
    return res.json()


def test_cron_parser_and_validator():
    # Valid expressions
    p1 = parse_cron_expression("*/15 * * * *")
    assert p1.minutes == frozenset({0, 15, 30, 45})
    assert len(p1.hours) == 24

    p2 = parse_cron_expression("0,30 9-17 * JAN,JUN MON-FRI")
    assert p2.minutes == frozenset({0, 30})
    assert p2.hours == frozenset(range(9, 18))
    assert p2.months == frozenset({1, 6})
    assert p2.days_of_week == frozenset({1, 2, 3, 4, 5})

    # Sunday as 0 and 7 and range MON-SUN
    p3 = parse_cron_expression("0 0 * * 7")
    assert p3.days_of_week == frozenset({0})
    p4 = parse_cron_expression("0 0 * * MON-SUN")
    assert p4.days_of_week == frozenset({0, 1, 2, 3, 4, 5, 6})

    # Macros
    p_daily = parse_cron_expression("@daily")
    assert p_daily.normalized == "0 0 * * *"
    assert describe_cron_expression("@daily") == "Daily at 00:00 UTC"
    assert describe_cron_expression("*/5 * * * *") == "Every 5 minutes"
    assert describe_cron_expression("30 2 * * *") == "Daily at 02:30 UTC"
    assert describe_cron_expression("manual") == "Manual trigger only"

    # Invalid expressions
    invalid_cases = [
        "",
        "manual",
        "* * * *",  # only 4 fields
        "* * * * * *",  # 6 fields
        "60 * * * *",  # minute > 59
        "* 24 * * *",  # hour > 23
        "* * 0 * *",  # dom < 1
        "* * 32 * *",  # dom > 31
        "* * * 13 *",  # month > 12
        "* * * * 8",  # dow > 7
        "*/0 * * * *",  # zero step
        "10-5 * * * *",  # inverted range
        "0 0 31 2 *",  # Feb 31 is impossible
        "0 0 31 4 *",  # Apr 31 is impossible
        "@every_second",  # unknown macro
    ]
    for expr in invalid_cases:
        with pytest.raises(ValueError):
            parse_cron_expression(expr)


def test_next_cron_time_and_upcoming_calculations():
    base = datetime(2026, 10, 16, 14, 23, 45, tzinfo=timezone.utc)  # Friday

    # Every 15 minutes -> 14:30:00
    nxt = get_next_cron_time("*/15 * * * *", after=base)
    assert nxt == datetime(2026, 10, 16, 14, 30, 0, tzinfo=timezone.utc)

    # Daily at 02:00 -> next day 2026-10-17 02:00:00
    nxt_daily = get_next_cron_time("0 2 * * *", after=base)
    assert nxt_daily == datetime(2026, 10, 17, 2, 0, 0, tzinfo=timezone.utc)

    # Weekly on Monday (1) at 09:00 -> 2026-10-19 09:00:00
    nxt_mon = get_next_cron_time("0 9 * * MON", after=base)
    assert nxt_mon == datetime(2026, 10, 19, 9, 0, 0, tzinfo=timezone.utc)

    # Upcoming 4 runs for hourly
    upcoming = get_upcoming_cron_times("0 * * * *", count=4, after=base)
    assert upcoming == [
        datetime(2026, 10, 16, 15, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 16, 16, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 16, 17, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 16, 18, 0, 0, tzinfo=timezone.utc),
    ]

    # Leap day Feb 29 -> 2028-02-29 00:00:00
    nxt_leap = get_next_cron_time("0 0 29 2 *", after=base)
    assert nxt_leap == datetime(2028, 2, 29, 0, 0, 0, tzinfo=timezone.utc)


def test_api_cron_validation_and_job_enable_disable(app_and_client):
    client, _, _ = app_and_client
    d1 = _create_def(client, name="Cron Def 1")

    # 1. Validate cron endpoint (GET and POST)
    res_valid = client.post(
        "/api/jobs/validate-cron",
        json={"schedule": "*/10 * * * *", "count": 3},
    )
    assert res_valid.status_code == 200
    vdata = res_valid.json()
    assert vdata["valid"] is True
    assert vdata["is_manual"] is False
    assert vdata["description"] == "Every 10 minutes"
    assert vdata["next_run_at"] is not None
    assert len(vdata["upcoming_runs"]) == 3

    res_invalid = client.get("/api/jobs/validate-cron?schedule=99+*+*+*+*")
    assert res_invalid.status_code == 200
    inv_data = res_invalid.json()
    assert inv_data["valid"] is False
    assert "out of range" in inv_data["error"]

    # 2. Creating a job with an invalid cron expression returns 422
    res_bad_cron = client.post(
        "/api/jobs",
        json={
            "name": "Invalid Cron Job",
            "schedule": "invalid cron string",
            "definition_ids": [d1["id"]],
        },
    )
    assert res_bad_cron.status_code == 422

    # 3. Creating a job with an invalid missed_run_policy returns 422
    res_bad_policy = client.post(
        "/api/jobs",
        json={
            "name": "Invalid Policy Job",
            "schedule": "0 2 * * *",
            "missed_run_policy": "explode",
            "definition_ids": [d1["id"]],
        },
    )
    assert res_bad_policy.status_code == 422

    # 4. Create enabled cron job -> next_run_at and upcoming_runs are populated
    res_job = client.post(
        "/api/jobs",
        json={
            "name": "Hourly Regression",
            "schedule": "0 * * * *",
            "enabled": True,
            "missed_run_policy": "run_once",
            "definition_ids": [d1["id"]],
        },
    )
    assert res_job.status_code == 201
    job = res_job.json()
    job_id = job["id"]
    assert job["is_scheduled"] is True
    assert job["enabled"] is True
    assert job["missed_run_policy"] == "run_once"
    assert job["next_run_at"] is not None
    assert len(job["upcoming_runs"]) == 5
    assert job["upcoming_runs"][0] == job["next_run_at"]
    assert job["schedule_description"] == "Every hour (at minute 0)"

    # 5. Disable job via POST /api/jobs/{id}/disable -> next_run_at becomes None
    res_dis = client.post(f"/api/jobs/{job_id}/disable")
    assert res_dis.status_code == 200
    dis_job = res_dis.json()
    assert dis_job["enabled"] is False
    assert dis_job["next_run_at"] is None
    assert dis_job["upcoming_runs"] == []

    # 6. Enable job via POST /api/jobs/{id}/enable -> next_run_at is recomputed
    res_en = client.post(f"/api/jobs/{job_id}/enable")
    assert res_en.status_code == 200
    en_job = res_en.json()
    assert en_job["enabled"] is True
    assert en_job["next_run_at"] is not None
    assert len(en_job["upcoming_runs"]) == 5

    # 7. Toggle job via POST /api/jobs/{id}/toggle
    res_tog = client.post(f"/api/jobs/{job_id}/toggle")
    assert res_tog.status_code == 200
    assert res_tog.json()["enabled"] is False
    assert res_tog.json()["next_run_at"] is None

    res_tog2 = client.post(f"/api/jobs/{job_id}/toggle")
    assert res_tog2.status_code == 200
    assert res_tog2.json()["enabled"] is True
    assert res_tog2.json()["next_run_at"] is not None


def test_scheduler_feeds_queue_and_prevents_overlapping_runs(app_and_client):
    client, session_factory, _ = app_and_client
    d1 = _create_def(client, name="Overlap Def")

    res_job = client.post(
        "/api/jobs",
        json={
            "name": "Every 5m Job",
            "schedule": "*/5 * * * *",
            "enabled": True,
            "missed_run_policy": "skip",
            "definition_ids": [d1["id"]],
        },
    )
    assert res_job.status_code == 201
    job_id = res_job.json()["id"]

    # Set next_run_at to a known timestamp t1
    t1 = datetime(2026, 10, 16, 12, 5, 0, tzinfo=timezone.utc)
    with session_factory() as db:
        job_row = db.get(Job, job_id)
        job_row.next_run_at = t1
        db.commit()

    scheduler = get_scheduler()

    # 1. Tick before t1 (at 12:04:30) -> nothing triggered
    tick_early = scheduler.tick(now=datetime(2026, 10, 16, 12, 4, 30, tzinfo=timezone.utc))
    assert len(tick_early["triggered_runs"]) == 0

    # 2. Tick at t1 + 5s (12:05:05, within grace window) -> queues Run #1 and advances next_run_at to 12:10:00
    tick_due = scheduler.tick(now=datetime(2026, 10, 16, 12, 5, 5, tzinfo=timezone.utc))
    assert len(tick_due["triggered_runs"]) == 1
    run_id_1 = tick_due["triggered_runs"][0]["run_id"]
    assert tick_due["triggered_runs"][0]["next_run_at"] == datetime(
        2026, 10, 16, 12, 10, 0, tzinfo=timezone.utc
    ).isoformat()

    # Verify job API shows is_running=True and active_run_id=run_id_1
    job_state = client.get(f"/api/jobs/{job_id}").json()
    assert job_state["is_running"] is True
    assert job_state["active_run_id"] == run_id_1

    # 3. At 12:10:05, Run #1 is STILL queued (not yet finished).
    # Scheduler tick MUST NOT queue an overlapping second run!
    tick_overlap = scheduler.tick(now=datetime(2026, 10, 16, 12, 10, 5, tzinfo=timezone.utc))
    assert len(tick_overlap["triggered_runs"]) == 0
    assert len(tick_overlap["skipped_overlap"]) == 1
    assert tick_overlap["skipped_overlap"][0]["active_run_id"] == run_id_1
    assert tick_overlap["skipped_overlap"][0]["next_run_at"] == datetime(
        2026, 10, 16, 12, 15, 0, tzinfo=timezone.utc
    ).isoformat()

    # Also verify only 1 run exists for this job in DB
    runs_list = client.get(f"/api/jobs/{job_id}/runs").json()
    assert len(runs_list) == 1

    # 4. Execute Run #1 via RunWorker
    worker = get_worker()
    loop = asyncio.new_event_loop()
    executed = loop.run_until_complete(worker.poll_and_execute_once())
    loop.close()
    assert executed is True

    run_1_detail = client.get(f"/api/runs/{run_id_1}").json()
    assert run_1_detail["status"] == "completed"
    assert run_1_detail["config_snapshot"]["trigger"] == "cron"

    # 5. Now at 12:15:05, since Run #1 is completed, the next scheduled tick queues Run #2!
    tick_next = scheduler.tick(now=datetime(2026, 10, 16, 12, 15, 5, tzinfo=timezone.utc))
    assert len(tick_next["triggered_runs"]) == 1
    run_id_2 = tick_next["triggered_runs"][0]["run_id"]
    assert run_id_2 != run_id_1


def test_downtime_missed_run_policy_skip_vs_run_once(app_and_client):
    """Verify defined behavior after downtime for both 'skip' and 'run_once' (coalesce) policies."""
    client, session_factory, _ = app_and_client
    d1 = _create_def(client, name="Downtime Def")

    # Create two jobs with the same 5-minute schedule:
    # Job A has missed_run_policy='skip'
    # Job B has missed_run_policy='run_once'
    res_skip = client.post(
        "/api/jobs",
        json={
            "name": "Downtime Skip Job",
            "schedule": "*/5 * * * *",
            "enabled": True,
            "missed_run_policy": "skip",
            "definition_ids": [d1["id"]],
        },
    )
    assert res_skip.status_code == 201
    skip_job_id = res_skip.json()["id"]

    res_once = client.post(
        "/api/jobs",
        json={
            "name": "Downtime Coalesce Job",
            "schedule": "*/5 * * * *",
            "enabled": True,
            "missed_run_policy": "run_once",
            "definition_ids": [d1["id"]],
        },
    )
    assert res_once.status_code == 201
    once_job_id = res_once.json()["id"]

    # Both jobs were scheduled for 10:05:00 UTC before the server went down for 1 hour
    due_before_downtime = datetime(2026, 10, 16, 10, 5, 0, tzinfo=timezone.utc)
    with session_factory() as db:
        j_skip = db.get(Job, skip_job_id)
        j_once = db.get(Job, once_job_id)
        j_skip.next_run_at = due_before_downtime
        j_once.next_run_at = due_before_downtime
        db.commit()

    # Server comes back up at 11:02:00 UTC (12 missed slots: 10:05, 10:10, ..., 11:00)
    recovery_time = datetime(2026, 10, 16, 11, 2, 0, tzinfo=timezone.utc)
    expected_next_run = datetime(2026, 10, 16, 11, 5, 0, tzinfo=timezone.utc)

    scheduler = get_scheduler()
    recovery_summary = scheduler.recover_after_downtime(now=recovery_time)

    # 1. 'skip' job must NOT queue any run, and must advance next_run_at to 11:05:00
    assert len(recovery_summary["skipped_missed"]) == 1
    skipped_entry = recovery_summary["skipped_missed"][0]
    assert skipped_entry["job_id"] == skip_job_id
    assert skipped_entry["missed_count"] == 12
    assert skipped_entry["next_run_at"] == expected_next_run.isoformat()

    skip_runs = client.get(f"/api/jobs/{skip_job_id}/runs").json()
    assert len(skip_runs) == 0
    skip_job_after = client.get(f"/api/jobs/{skip_job_id}").json()
    assert skip_job_after["next_run_at"] == expected_next_run.isoformat()

    # 2. 'run_once' job must coalesce all 12 missed slots into EXACTLY 1 queued run
    # and advance next_run_at to 11:05:00
    assert len(recovery_summary["triggered_runs"]) == 1
    triggered_entry = recovery_summary["triggered_runs"][0]
    assert triggered_entry["job_id"] == once_job_id
    assert triggered_entry["coalesced_missed_count"] == 12
    assert triggered_entry["next_run_at"] == expected_next_run.isoformat()

    once_runs = client.get(f"/api/jobs/{once_job_id}/runs").json()
    assert len(once_runs) == 1
    assert once_runs[0]["config_snapshot"]["missed_run_coalesced"] is True
    assert once_runs[0]["config_snapshot"]["coalesced_missed_count"] == 12

    once_job_after = client.get(f"/api/jobs/{once_job_id}").json()
    assert once_job_after["next_run_at"] == expected_next_run.isoformat()
    assert once_job_after["last_scheduled_at"] == recovery_time.isoformat()

    # 3. Subsequent tick at 11:02:30 does not trigger any additional runs
    follow_up_tick = scheduler.tick(now=datetime(2026, 10, 16, 11, 2, 30, tzinfo=timezone.utc))
    assert len(follow_up_tick["triggered_runs"]) == 0
