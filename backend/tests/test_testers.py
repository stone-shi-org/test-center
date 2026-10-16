import json
import pytest
from pathlib import Path

from backend.app.testers.base import TestContext, TesterResult, TesterCancelledError
from backend.app.testers.schema import validate_schema, apply_schema_defaults, SchemaValidationError
from backend.app.testers.registry import get_tester, list_testers, tester_registry
from backend.app.testers.dummy import DummyTester


def test_schema_defaults_and_validation():
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "default": 5, "minimum": 1},
            "enabled": {"type": "boolean", "default": True},
        },
    }

    # Apply defaults
    data = {"name": "test"}
    data_with_defaults = apply_schema_defaults(data, schema)
    assert data_with_defaults == {"name": "test", "count": 5, "enabled": True}

    # Validate valid data
    validate_schema(data_with_defaults, schema)

    # Missing required
    with pytest.raises(SchemaValidationError, match="missing required property 'name'"):
        validate_schema({}, schema)

    # Invalid type
    with pytest.raises(SchemaValidationError, match="expected type 'integer'"):
        validate_schema({"name": "test", "count": "not-an-int"}, schema)

    # Boundary violation
    with pytest.raises(SchemaValidationError, match="less than minimum"):
        validate_schema({"name": "test", "count": 0}, schema)


def test_tester_registry():
    testers = list_testers()
    assert any(t.name == "dummy" for t in testers)

    dummy = get_tester("dummy")
    assert isinstance(dummy, DummyTester)
    assert dummy.suites == ["smoke", "load", "edge_cases"]
    assert "properties" in dummy.config_schema


def test_dummy_tester_run_success(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=1, artifacts_dir=artifacts_dir, emit_stdout=False)

    dummy = DummyTester()
    config = {
        "duration_seconds": 0.1,
        "step_delay": 0.05,
        "fail_case": False,
        "create_artifacts": True,
    }
    target = {"name": "test-target", "kind": "local", "base_url": "http://localhost:8000"}

    result = dummy.run(config, target, ctx)

    assert result.status == "pass"
    assert "duration_ms" in result.metrics
    assert "throughput" in result.metrics
    assert len(result.cases) == 3
    assert all(c.status == "pass" for c in result.cases)

    # Verify artifacts
    assert len(result.artifacts) >= 2
    sample_file = artifacts_dir / "sample_output.txt"
    assert sample_file.exists()
    assert "Run ID: 1" in sample_file.read_text()

    metrics_file = artifacts_dir / "metrics_summary.json"
    assert metrics_file.exists()

    # Verify log file
    log_file = artifacts_dir / "run.log"
    assert log_file.exists()
    assert "Starting dummy tester" in log_file.read_text()


def test_dummy_tester_cancellation(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=2, artifacts_dir=artifacts_dir, emit_stdout=False)

    dummy = DummyTester()
    config = {
        "duration_seconds": 2.0,
        "step_delay": 0.1,
    }
    target = {"name": "test-target"}

    # Pre-cancel the context
    ctx.set_cancelled()
    assert ctx.is_cancelled is True

    with pytest.raises(TesterCancelledError):
        dummy.run(config, target, ctx)


def test_dummy_tester_failing_case(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    ctx = TestContext(run_id=3, artifacts_dir=artifacts_dir, emit_stdout=False)

    dummy = DummyTester()
    config = {
        "duration_seconds": 0.05,
        "step_delay": 0.02,
        "fail_case": True,
    }
    target = {"name": "test-target"}

    result = dummy.run(config, target, ctx)
    assert result.status == "fail"
    assert any(c.status == "fail" for c in result.cases)
