import json
import os
from pathlib import Path
import subprocess
import sys
from backend.app.config import BASE_DIR


def test_executor_subprocess_success(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts" / "1"
    artifacts_dir.mkdir(parents=True)

    payload = {
        "run_id": 1,
        "tester": "dummy",
        "config": {
            "duration_seconds": 0.1,
            "step_delay": 0.05,
            "fail_case": False,
            "create_artifacts": True,
        },
        "target": {"name": "local-test"},
        "suites": ["smoke"],
        "artifacts_dir": str(artifacts_dir),
    }

    payload_file = artifacts_dir / "payload.json"
    payload_file.write_text(json.dumps(payload))

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{BASE_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = [sys.executable, "-m", "backend.app.runner.executor", "--payload-file", str(payload_file)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)

    assert res.returncode == 0
    # stdout should have json lines
    lines = [json.loads(line) for line in res.stdout.strip().split("\n") if line.strip().startswith("{")]
    assert any(l.get("type") == "log" for l in lines)
    assert any(l.get("type") == "progress" for l in lines)
    assert any(l.get("type") == "completed" and l.get("status") == "pass" for l in lines)

    # Check result.json
    result_json = artifacts_dir / "result.json"
    assert result_json.exists()
    data = json.loads(result_json.read_text())
    assert data["status"] == "pass"
    assert "duration_ms" in data["metrics"]

    # Check run.log
    run_log = artifacts_dir / "run.log"
    assert run_log.exists()
    assert "Executor started for run 1" in run_log.read_text()


def test_executor_subprocess_error(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts" / "2"
    artifacts_dir.mkdir(parents=True)

    payload = {
        "run_id": 2,
        "tester": "dummy",
        "config": {
            "error": True,
        },
        "target": {},
        "artifacts_dir": str(artifacts_dir),
    }

    payload_file = artifacts_dir / "payload.json"
    payload_file.write_text(json.dumps(payload))

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{BASE_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = [sys.executable, "-m", "backend.app.runner.executor", "--payload-file", str(payload_file)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)

    assert res.returncode == 1
    result_json = artifacts_dir / "result.json"
    assert result_json.exists()
    data = json.loads(result_json.read_text())
    assert data["status"] == "error"
    assert "Simulated crash" in data["summary"]
