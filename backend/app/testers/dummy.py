"""Dummy tester plugin implementation for testing and verification."""

import json
from pathlib import Path
import time
from typing import Any

from backend.app.testers.base import (
    TesterPlugin,
    TestContext,
    TesterResult,
    TesterCancelledError,
)


class DummyTester(TesterPlugin):
    """Reference tester plugin that simulates test execution, streaming, and artifacts."""

    name = "dummy"
    description = "Dummy tester for testing executor lifecycle, cancellation, and artifact generation"
    suites = ["smoke", "load", "edge_cases"]

    config_schema = {
        "type": "object",
        "properties": {
          "duration_seconds": {
            "type": "number",
            "default": 1.0,
            "minimum": 0.0,
            "description": "Total duration in seconds to simulate work",
          },
          "step_delay": {
            "type": "number",
            "default": 0.2,
            "minimum": 0.0,
            "description": "Delay between steps in seconds",
          },
          "fail_case": {
            "type": "boolean",
            "default": False,
            "description": "Whether to include a failing test case",
          },
          "error": {
            "type": "boolean",
            "default": False,
            "description": "Whether to raise an unhandled exception to simulate crash",
          },
          "create_artifacts": {
            "type": "boolean",
            "default": True,
            "description": "Whether to generate sample artifact files",
          },
        },
    }

    def run(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
        ctx: TestContext,
    ) -> TesterResult:
        cfg = self.validate_and_apply_config(config)

        ctx.log(f"Starting dummy tester with target={target.get('name', 'default')}")

        if cfg.get("error"):
            ctx.log("Simulating fatal crash error in dummy tester!", level="error")
            raise RuntimeError("Simulated crash in dummy tester")

        duration = float(cfg.get("duration_seconds", 1.0))
        step_delay = float(cfg.get("step_delay", 0.2))
        fail_case = bool(cfg.get("fail_case", False))
        create_artifacts = bool(cfg.get("create_artifacts", True))

        # Determine suites to run
        selected_suites = cfg.get("suites") or target.get("suites") or self.suites
        suites_to_run = [s for s in self.suites if s in selected_suites]
        if not suites_to_run:
            suites_to_run = self.suites

        total_steps = max(len(suites_to_run), int(duration / max(step_delay, 0.05))) if duration > 0 else len(suites_to_run)
        step_time = duration / total_steps if total_steps > 0 else 0.0

        ctx.log(f"Executing suites: {suites_to_run} across {total_steps} steps")

        start_time = time.time()
        for step in range(1, total_steps + 1):
            ctx.check_cancelled()

            suite = suites_to_run[(step - 1) % len(suites_to_run)]
            ctx.log(f"Step {step}/{total_steps}: Executing suite '{suite}'")
            ctx.progress(step, total_steps, f"Running suite '{suite}' (step {step}/{total_steps})")

            # Sleep in small slices to be responsive to cancellation
            remaining = step_time
            while remaining > 0:
                ctx.check_cancelled()
                slice_time = min(remaining, 0.05)
                time.sleep(slice_time)
                remaining -= slice_time

        elapsed = time.time() - start_time
        ctx.log(f"Dummy execution completed in {elapsed:.3f}s")

        # Generate sample artifacts if requested
        if create_artifacts:
            sample_file = ctx.artifacts_dir / "sample_output.txt"
            sample_content = (
                f"Dummy Tester Output\n"
                f"Run ID: {ctx.run_id}\n"
                f"Target: {target.get('name', 'none')}\n"
                f"Elapsed: {elapsed:.3f}s\n"
            )
            sample_file.write_text(sample_content, encoding="utf-8")
            ctx.add_artifact("sample_output.txt", sample_file, mime="text/plain")
            ctx.log("Generated artifact 'sample_output.txt'")

            metrics_json_file = ctx.artifacts_dir / "metrics_summary.json"
            metrics_data = {"elapsed_seconds": elapsed, "suites_run": suites_to_run}
            metrics_json_file.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
            ctx.add_artifact("metrics_summary.json", metrics_json_file, mime="application/json")
            ctx.log("Generated artifact 'metrics_summary.json'")

        # Build result
        status = "fail" if fail_case else "pass"
        result = TesterResult(
            status=status,
            summary=f"Dummy tester completed ({len(suites_to_run)} suites, elapsed {elapsed:.2f}s)",
        )

        # Set standard metrics
        result.set_metric("duration_ms", round(elapsed * 1000, 2), unit="ms", better="lower")
        result.set_metric("throughput", 100.0, unit="req/s", better="higher")
        result.set_metric("success_rate", 50.0 if fail_case else 100.0, unit="%", better="higher")

        # Add cases
        for s in suites_to_run:
            case_status = "fail" if (fail_case and s == suites_to_run[-1]) else "pass"
            result.add_case(
                case_id=f"case_{s}_001",
                suite=s,
                status=case_status,
                metrics={"latency_ms": {"value": 25.0, "unit": "ms", "better": "lower"}},
                detail={"suite": s, "evaluated": True},
            )

        # Include registered artifacts from context
        for art in ctx.artifacts:
            result.artifacts.append(art)

        return result
