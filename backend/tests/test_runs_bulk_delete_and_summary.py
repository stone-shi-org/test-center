"""Tests for POST /api/runs/bulk-delete and GET /api/runs?view=summary."""

from backend.app.db.base import utc_now
from backend.app.db.models.baseline import Baseline
from backend.app.db.models.case_result import CaseResult
from backend.app.db.models.result import Result
from backend.app.db.models.run import Run


def _make_run(session, status: str, *, with_cases: bool = False) -> int:
    run = Run(status=status, queued_at=utc_now(), config_snapshot={"tester": "dummy"}, target_snapshot={})
    session.add(run)
    session.flush()
    if with_cases:
        res = Result(run_id=run.id, tester="dummy", status="pass", metrics={"acc": 1.0}, summary="ok")
        session.add(res)
        session.flush()
        session.add_all([
            CaseResult(result_id=res.id, suite="s", case_id="c1", status="pass",
                       metrics={"latency": 1}, detail={"model": "m-1", "prompt": "x" * 1000}),
            CaseResult(result_id=res.id, suite="s", case_id="c2", status="fail",
                       metrics={"latency": 2}, detail={"prompt": "y" * 1000}),
        ])
    session.commit()
    return run.id


def test_bulk_delete_cancelled_skips_baselines_and_active(app_and_client):
    client, session_factory, artifacts_dir = app_and_client
    with session_factory() as s:
        c1 = _make_run(s, "cancelled")
        c2 = _make_run(s, "cancelled", with_cases=True)
        c_base = _make_run(s, "cancelled")
        done = _make_run(s, "completed")
        running = _make_run(s, "running")
        s.add(Baseline(name="keep-me", run_id=c_base, history=[]))
        s.commit()

    (artifacts_dir / str(c1)).mkdir(parents=True, exist_ok=True)
    (artifacts_dir / str(c1) / "f.txt").write_text("x")

    preview = client.post("/api/runs/bulk-delete", json={"status": "cancelled", "dry_run": True})
    assert preview.status_code == 200
    body = preview.json()
    assert body["dry_run"] is True
    assert sorted(body["deleted"]) == sorted([c1, c2])
    assert [sk["id"] for sk in body["skipped"]] == [c_base]
    assert client.get(f"/api/runs/{c1}").status_code == 200  # nothing deleted yet

    res = client.post("/api/runs/bulk-delete", json={"status": "cancelled"})
    assert res.status_code == 200
    assert res.json()["deleted_count"] == 2
    assert client.get(f"/api/runs/{c1}").status_code == 404
    assert client.get(f"/api/runs/{c2}").status_code == 404
    assert not (artifacts_dir / str(c1)).exists()
    for kept in (c_base, done, running):
        assert client.get(f"/api/runs/{kept}").status_code == 200

    # Explicit IDs: active runs are skipped, not deleted
    res = client.post("/api/runs/bulk-delete", json={"run_ids": [running, done]})
    assert res.json()["deleted"] == [done]
    assert res.json()["skipped"][0]["id"] == running


def test_bulk_delete_validation(app_and_client):
    client, _, _ = app_and_client
    assert client.post("/api/runs/bulk-delete", json={}).status_code == 400
    assert client.post("/api/runs/bulk-delete", json={"status": "running"}).status_code == 400


def test_list_runs_summary_view_omits_case_payloads(app_and_client):
    client, session_factory, _ = app_and_client
    with session_factory() as s:
        run_id = _make_run(s, "completed", with_cases=True)

    full = client.get("/api/runs").json()
    summary = client.get("/api/runs?view=summary").json()
    assert len(full) == len(summary) == 1

    full_cases = full[0]["results"][0]["cases"]
    sum_cases = summary[0]["results"][0]["cases"]
    assert full_cases[0]["detail"]["prompt"]
    assert [c["status"] for c in sum_cases] == [c["status"] for c in full_cases]
    by_case = {c["case_id"]: c for c in sum_cases}
    assert by_case["c1"]["detail"] == {"model": "m-1"}
    assert by_case["c2"]["detail"] == {}
    assert all(c["metrics"] == {} for c in sum_cases)
    # Result-level metrics are still present
    assert summary[0]["results"][0]["metrics"] == full[0]["results"][0]["metrics"]
    assert summary[0]["id"] == run_id

    # Model filter still works in summary mode (falls back to full case loading)
    assert len(client.get("/api/runs?view=summary&model=m-1").json()) == 1
    assert len(client.get("/api/runs?view=summary&model=nope").json()) == 0
