"""Subprocess executor entry point.

Runs a tester plugin in an isolated subprocess, streaming logs and events to stdout,
and writing results and artifacts to the run directory.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from typing import Any

from backend.app.testers.base import (
    TestContext,
    TesterResult,
    TesterCancelledError,
    TesterError,
    set_result_checkpoint_callback,
)
from backend.app.testers.registry import get_tester, TesterNotFoundError


def run_executor(payload: dict[str, Any]) -> int:
    """Execute tester defined in payload dictionary."""
    run_id = int(payload["run_id"])
    tester_name = payload["tester"]
    config = payload.get("config", {})
    target = payload.get("target", {})
    suites = payload.get("suites")
    if suites:
        config = dict(config)
        config["suites"] = suites

    artifacts_dir = Path(payload["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(payload.get("log_file", artifacts_dir / "run.log"))
    result_file = Path(payload.get("result_file", artifacts_dir / "result.json"))

    ctx = TestContext(
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        log_file=log_file,
        emit_stdout=True,
    )

    def handle_signal(sig, frame):
        ctx.set_cancelled()
        sig_name = signal.Signals(sig).name
        ctx.log(f"Received termination signal ({sig_name})", level="warning")
        raise TesterCancelledError(f"Terminated by signal {sig_name}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    ctx.log(f"Executor started for run {run_id}, tester '{tester_name}'")

    result: TesterResult
    exit_code: int = 0
    last_result_ref: list[TesterResult] = []

    def _write_checkpoint(res: TesterResult) -> None:
        last_result_ref[:] = [res]
        data = res.to_dict()
        existing_paths = {a["path"] for a in data.get("artifacts", [])}
        for art in ctx.artifacts:
            if art.path not in existing_paths:
                data.setdefault("artifacts", []).append(art.to_dict())
                existing_paths.add(art.path)
        tmp_file = result_file.with_suffix(".json.tmp")
        tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp_file, result_file)

    set_result_checkpoint_callback(_write_checkpoint)
    try:
        tester = get_tester(tester_name)
        # Check if run is coroutine or regular function
        run_fn = tester.run
        if inspect.iscoroutinefunction(run_fn):
            import asyncio
            result = asyncio.run(run_fn(config, target, ctx))
        else:
            result = run_fn(config, target, ctx)

        if result.status == "cancelled":
            exit_code = 2
        elif result.status == "error":
            exit_code = 1
        else:
            exit_code = 0

    except TesterCancelledError:
        ctx.log("Tester cancelled via TesterCancelledError", level="warning")
        prev = last_result_ref[0] if last_result_ref else None
        result = TesterResult(
            status="cancelled",
            summary="Execution cancelled by user or timeout",
            metrics=dict(prev.metrics) if prev else {},
            cases=list(prev.cases) if prev else [],
            artifacts=list(prev.artifacts) if prev else list(ctx.artifacts),
        )
        exit_code = 2

    except TesterNotFoundError as e:
        ctx.log(f"Tester plugin not found: {e}", level="error")
        result = TesterResult(
            status="error",
            summary=str(e),
            artifacts=list(ctx.artifacts),
        )
        exit_code = 1

    except Exception as e:
        ctx.log(f"Unhandled exception during execution: {traceback.format_exc()}", level="error")
        prev = last_result_ref[0] if last_result_ref else None
        result = TesterResult(
            status="error",
            summary=f"Execution error: {e}",
            metrics=dict(prev.metrics) if prev else {},
            cases=list(prev.cases) if prev else [],
            artifacts=list(prev.artifacts) if prev else list(ctx.artifacts),
        )
        exit_code = 1
    finally:
        set_result_checkpoint_callback(None)

    # Ensure all registered context artifacts are in result
    existing_paths = {a.path for a in result.artifacts}
    for art in ctx.artifacts:
        if art.path not in existing_paths:
            result.artifacts.append(art)
            existing_paths.add(art.path)

    # Persist result.json
    try:
        result_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        ctx.log(f"Saved result summary to {result_file.name}")
    except Exception as e:
        ctx.log(f"Failed to write result file: {e}", level="error")
        exit_code = 1

    # Emit final completion event on stdout
    completion_event = {
        "type": "completed",
        "run_id": run_id,
        "status": result.status,
        "exit_code": exit_code,
    }
    sys.stdout.write(json.dumps(completion_event) + "\n")
    sys.stdout.flush()

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Center Subprocess Runner")
    parser.add_argument("--payload-file", help="Path to JSON payload file", type=str)
    parser.add_argument("--payload", help="JSON payload string", type=str)

    args = parser.parse_args()

    payload_data: dict[str, Any]
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload_data = json.load(f)
    elif args.payload:
        payload_data = json.loads(args.payload)
    else:
        # Check stdin
        if not sys.stdin.isatty():
            payload_data = json.load(sys.stdin)
        else:
            parser.error("Must provide --payload-file, --payload, or pass JSON via stdin")

    sys.exit(run_executor(payload_data))


if __name__ == "__main__":
    main()
