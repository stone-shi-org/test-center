"""Piston execution client and Java test-case judge.

Ported from ``interview-question-bank/src/lib/judge-runner.ts`` with support for:
- Target/config ``runner_url`` resolution (matching the live-stt ``grpc_port`` pattern)
- Running all test cases without stopping on the first failure (``stop_on_first_failure=False``)
- Visible sample vs. hidden test case accounting
- Tool-mode helpers: ``compile_java_code``, ``run_java_samples``, ``run_java_custom``
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from backend.app.testers.llm_coding.harness import (
    MergedJavaSource,
    annotate_runtime_output,
    extract_java_code,
    merge_java_solution_with_harness,
)

try:
    import httpx
except ImportError:  # pragma: no cover - host fallback when running outside container
    httpx = None  # type: ignore[assignment]

DEFAULT_PISTON_URL = "http://piston:2000"
DEFAULT_JAVA_VERSION = "15.0.2"
PISTON_MAX_RUN_TIMEOUT_MS = 3000

# JEP 330 single-file source launch (`java Main.java`) reports compile errors on
# `run.stderr` and finishes with `error: compilation failed`.
JAVA_SOURCE_LAUNCH_COMPILE_ERROR_RE = re.compile(
    r"^error: compilation failed\s*$", re.MULTILINE
)

STATUS_ACCEPTED = "Accepted"
STATUS_WRONG_ANSWER = "Wrong Answer"
STATUS_COMPILE_ERROR = "Compile Error"
STATUS_RUNTIME_ERROR = "Runtime Error"
STATUS_TLE = "Time Limit Exceeded"
STATUS_PERFORMANCE_LIMIT = "Performance Limit Exceeded"
STATUS_NO_CODE = "no_code_extracted"


class RunnerConnectionError(RuntimeError):
    """Raised when the Piston sidecar cannot be reached or returns an HTTP error."""


@dataclass
class JudgeTestCase:
    input: str
    expected_output: str
    index: int = 0
    is_sample: bool = False


@dataclass
class TestCaseResult:
    case_index: int
    case_number: int
    is_sample: bool
    input: str
    expected: str
    actual: str
    stderr: str
    compile_output: str
    time_ms: float
    memory_kb: int
    status: str
    passed: bool
    performance_failed: bool = False
    status_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeRunResult:
    status: str
    passed: bool
    correct: int
    compilation_failed: bool
    compile_error_output: str
    test_results: list[TestCaseResult] = field(default_factory=list)
    total_cases: int = 0
    passed_cases: int = 0
    test_case_pass_rate: float = 0.0
    sample_total: int = 0
    sample_passed: int = 0
    hidden_total: int = 0
    hidden_passed: int = 0
    max_cpu_time_ms: float = 0.0
    run_time_ms: float = 0.0
    memory_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "correct": self.correct,
            "compilation_failed": self.compilation_failed,
            "compile_error_output": self.compile_error_output,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "test_case_pass_rate": self.test_case_pass_rate,
            "sample_total": self.sample_total,
            "sample_passed": self.sample_passed,
            "hidden_total": self.hidden_total,
            "hidden_passed": self.hidden_passed,
            "max_cpu_time_ms": self.max_cpu_time_ms,
            "run_time_ms": self.run_time_ms,
            "memory_bytes": self.memory_bytes,
            "test_results": [r.to_dict() for r in self.test_results],
        }


def resolve_runner_url(
    target: Any | None = None,
    config: dict[str, Any] | None = None,
    default: str | None = None,
) -> str:
    """Resolve the Piston sidecar URL with priority:
    1. Explicit ``config["runner_url"]`` override
    2. ``target.extra["runner_url"]`` (or ``target["extra"]["runner_url"]``)
    3. ``PISTON_URL`` / ``PISTON_API_URL`` environment variable
    4. ``default`` or ``http://piston:2000``
    """
    if config:
        cfg_url = config.get("runner_url")
        if isinstance(cfg_url, str) and cfg_url.strip():
            return cfg_url.strip().rstrip("/")

    target_extra: dict[str, Any] | None = None
    if target is not None:
        if isinstance(target, dict):
            extra_val = target.get("extra")
            if isinstance(extra_val, dict):
                target_extra = extra_val
            elif "runner_url" in target:
                target_extra = target
        else:
            extra_val = getattr(target, "extra", None)
            if isinstance(extra_val, dict):
                target_extra = extra_val

    if target_extra:
        extra_url = target_extra.get("runner_url")
        if isinstance(extra_url, str) and extra_url.strip():
            return extra_url.strip().rstrip("/")

    env_url = os.getenv("PISTON_URL") or os.getenv("PISTON_API_URL")
    if env_url and env_url.strip():
        return env_url.strip().rstrip("/")

    fallback = default or DEFAULT_PISTON_URL
    return fallback.strip().rstrip("/")


def clamp_piston_timeout_ms(timeout_ms: int | float | None) -> int:
    """Clamp per-case timeout to Piston's maximum (``min(test_timeout_value, 3000)``)."""
    if timeout_ms is None or timeout_ms <= 0:
        return PISTON_MAX_RUN_TIMEOUT_MS
    return min(max(int(timeout_ms), 50), PISTON_MAX_RUN_TIMEOUT_MS)


def map_piston_response_to_case_result(
    piston_data: dict[str, Any],
    merged: MergedJavaSource,
    tc: JudgeTestCase,
    duration_ms: float = 0.0,
    cpu_limit_ms: int | float | None = None,
    timeout_ms: int | float | None = None,
) -> tuple[bool, str, TestCaseResult]:
    """Map a raw Piston ``/api/v2/execute`` JSON response to ``(compilation_failed, compile_error, TestCaseResult)``."""
    compile_stage = piston_data.get("compile")
    if isinstance(compile_stage, dict) and compile_stage.get("code", 0) != 0:
        raw_err = (
            compile_stage.get("stderr")
            or compile_stage.get("output")
            or compile_stage.get("stdout")
            or "Compilation failed"
        )
        compile_err = annotate_runtime_output(str(raw_err), merged)
        return (
            True,
            compile_err,
            TestCaseResult(
                case_index=tc.index,
                case_number=tc.index + 1,
                is_sample=tc.is_sample,
                input=tc.input or "",
                expected=tc.expected_output or "",
                actual="",
                stderr="",
                compile_output=compile_err,
                time_ms=0.0,
                memory_kb=0,
                status=STATUS_COMPILE_ERROR,
                passed=False,
                performance_failed=False,
                status_detail=STATUS_COMPILE_ERROR,
            ),
        )

    run_stage = piston_data.get("run") or {}
    raw_stderr = str(run_stage.get("stderr") or "")
    stderr = annotate_runtime_output(raw_stderr, merged)
    stdout = str(run_stage.get("stdout") or "")
    run_code = run_stage.get("code", 0)
    signal = run_stage.get("signal")
    run_status = run_stage.get("status")

    if run_code != 0 and JAVA_SOURCE_LAUNCH_COMPILE_ERROR_RE.search(stderr):
        return (
            True,
            stderr,
            TestCaseResult(
                case_index=tc.index,
                case_number=tc.index + 1,
                is_sample=tc.is_sample,
                input=tc.input or "",
                expected=tc.expected_output or "",
                actual=stdout,
                stderr=stderr,
                compile_output=stderr,
                time_ms=0.0,
                memory_kb=0,
                status=STATUS_COMPILE_ERROR,
                passed=False,
                performance_failed=False,
                status_detail=STATUS_COMPILE_ERROR,
            ),
        )

    actual_clean = stdout.strip()
    expected_clean = (tc.expected_output or "").strip()
    is_passed = actual_clean == expected_clean
    status_description = STATUS_ACCEPTED
    status_detail = STATUS_ACCEPTED

    effective_timeout = clamp_piston_timeout_ms(timeout_ms)
    wall_time = run_stage.get("wall_time")
    timed_out = (
        signal in ("SIGTERM", "SIGKILL")
        or run_status == "TO"
        or "Execution Timed Out" in stderr
        or (
            isinstance(wall_time, (int, float))
            and wall_time >= effective_timeout
            and run_code != 0
        )
    )

    if timed_out:
        is_passed = False
        status_description = STATUS_TLE
        status_detail = STATUS_TLE
    elif run_code != 0:
        is_passed = False
        status_description = STATUS_RUNTIME_ERROR
        status_detail = STATUS_RUNTIME_ERROR
    elif not is_passed:
        status_description = STATUS_WRONG_ANSWER
        status_detail = STATUS_WRONG_ANSWER

    cpu_time = run_stage.get("cpu_time")
    if isinstance(cpu_time, (int, float)):
        actual_time_ms = float(cpu_time)
    elif isinstance(wall_time, (int, float)):
        actual_time_ms = float(wall_time)
    else:
        actual_time_ms = float(duration_ms)

    memory_bytes = int(run_stage.get("memory") or 0)
    memory_kb = round(memory_bytes / 1024)

    performance_failed = False
    if (
        is_passed
        and cpu_limit_ms is not None
        and cpu_limit_ms > 0
        and actual_time_ms > cpu_limit_ms
    ):
        is_passed = False
        performance_failed = True
        status_description = STATUS_PERFORMANCE_LIMIT
        status_detail = (
            f"{STATUS_PERFORMANCE_LIMIT} ({int(actual_time_ms)}ms > {int(cpu_limit_ms)}ms limit)"
        )

    return (
        False,
        "",
        TestCaseResult(
            case_index=tc.index,
            case_number=tc.index + 1,
            is_sample=tc.is_sample,
            input=tc.input or "",
            expected=tc.expected_output or "",
            actual=stdout,
            stderr=stderr,
            compile_output="",
            time_ms=actual_time_ms,
            memory_kb=memory_kb,
            status=status_description,
            passed=is_passed,
            performance_failed=performance_failed,
            status_detail=status_detail,
        ),
    )


class PistonClient:
    """HTTP client for Piston ``/api/v2`` endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_PISTON_URL,
        java_version: str = DEFAULT_JAVA_VERSION,
        request_timeout_s: float = 30.0,
        http_client: Any | None = None,
        execute_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_PISTON_URL).strip().rstrip("/")
        self.java_version = java_version or DEFAULT_JAVA_VERSION
        self.request_timeout_s = request_timeout_s
        self._http_client = http_client
        self._execute_hook = execute_hook

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if self._http_client is not None:
            try:
                resp = self._http_client.request(
                    method,
                    url,
                    json=payload,
                    timeout=self.request_timeout_s,
                )
            except Exception as exc:
                raise RunnerConnectionError(
                    f"Failed to connect to Piston at {url}: {exc}"
                ) from exc
            if resp.status_code >= 400:
                raise RunnerConnectionError(
                    f"Piston error ({resp.status_code}) at {url}: {resp.text}"
                )
            return resp.json()

        data_bytes = (
            json.dumps(payload).encode("utf-8") if payload is not None else None
        )
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(
            url, data=data_bytes, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RunnerConnectionError(
                f"Piston error ({exc.code}) at {url}: {err_body or exc.reason}"
            ) from exc
        except Exception as exc:
            raise RunnerConnectionError(
                f"Failed to connect to Piston at {url}: {exc}"
            ) from exc

    def get_runtimes(self) -> list[dict[str, Any]]:
        res = self._request_json("GET", "/api/v2/runtimes")
        return res if isinstance(res, list) else []

    def ensure_java_runtime(self) -> dict[str, Any]:
        """Ensure a Java runtime is installed on Piston, installing it if missing."""
        runtimes = self.get_runtimes()
        for rt in runtimes:
            if rt.get("language") == "java":
                return rt
        installed = self._request_json(
            "POST",
            "/api/v2/packages",
            {"language": "java", "version": self.java_version},
        )
        return installed if isinstance(installed, dict) else {"language": "java", "version": self.java_version}

    def execute_java(
        self,
        main_java_source: str,
        stdin: str = "",
        timeout_ms: int | float | None = PISTON_MAX_RUN_TIMEOUT_MS,
    ) -> dict[str, Any]:
        """Execute a single-file ``Main.java`` payload on Piston."""
        payload = {
            "language": "java",
            "version": self.java_version,
            "files": [{"name": "Main.java", "content": main_java_source}],
            "stdin": stdin or "",
            "run_timeout": clamp_piston_timeout_ms(timeout_ms),
        }
        if self._execute_hook is not None:
            return self._execute_hook(payload)
        res = self._request_json("POST", "/api/v2/execute", payload)
        if not isinstance(res, dict) or "run" not in res:
            msg = res.get("message") if isinstance(res, dict) else str(res)
            raise RunnerConnectionError(
                f"Invalid Piston response: {msg or 'missing run field'}"
            )
        return res


def judge_java_solution(
    client: PistonClient,
    code: str | None,
    harness: str,
    test_cases: list[dict[str, Any] | JudgeTestCase],
    sample_case_idx: list[int] | set[int] | None = None,
    timeout_ms: int | float | None = PISTON_MAX_RUN_TIMEOUT_MS,
    cpu_limit_ms: int | float | None = None,
    stop_on_first_failure: bool = False,
    extract_from_markdown: bool = False,
) -> JudgeRunResult:
    """Merge a Java candidate ``Solution`` with ``Harness.java`` and run test cases on Piston.

    By default ``stop_on_first_failure=False`` so every test case is executed and a
    partial ``test_case_pass_rate`` is computed.
    """
    sample_set = set(sample_case_idx or [])
    normalized_cases: list[JudgeTestCase] = []
    for i, tc in enumerate(test_cases):
        if isinstance(tc, JudgeTestCase):
            normalized_cases.append(
                JudgeTestCase(
                    input=tc.input,
                    expected_output=tc.expected_output,
                    index=i,
                    is_sample=(i in sample_set) or tc.is_sample,
                )
            )
        else:
            normalized_cases.append(
                JudgeTestCase(
                    input=str(tc.get("input") or ""),
                    expected_output=str(tc.get("expected_output") or tc.get("expected") or ""),
                    index=i,
                    is_sample=i in sample_set,
                )
            )

    total_cases = len(normalized_cases)
    sample_total = sum(1 for c in normalized_cases if c.is_sample)
    hidden_total = total_cases - sample_total

    candidate_code = extract_java_code(code) if extract_from_markdown else (code.strip() if code else None)
    if not candidate_code:
        return JudgeRunResult(
            status=STATUS_NO_CODE,
            passed=False,
            correct=0,
            compilation_failed=False,
            compile_error_output="",
            test_results=[],
            total_cases=total_cases,
            passed_cases=0,
            test_case_pass_rate=0.0,
            sample_total=sample_total,
            sample_passed=0,
            hidden_total=hidden_total,
            hidden_passed=0,
        )

    merged = merge_java_solution_with_harness(candidate_code, harness)
    test_results: list[TestCaseResult] = []
    compilation_failed = False
    compile_error_output = ""
    peak_memory_bytes = 0
    max_cpu_time_ms = 0.0
    run_start = time.perf_counter()

    for idx, tc in enumerate(normalized_cases):
        case_start = time.perf_counter()
        piston_data = client.execute_java(
            merged.code,
            stdin=tc.input,
            timeout_ms=timeout_ms,
        )
        duration_ms = (time.perf_counter() - case_start) * 1000.0

        run_mem = int((piston_data.get("run") or {}).get("memory") or 0)
        if run_mem > peak_memory_bytes:
            peak_memory_bytes = run_mem

        comp_failed, comp_err, case_res = map_piston_response_to_case_result(
            piston_data=piston_data,
            merged=merged,
            tc=tc,
            duration_ms=duration_ms,
            cpu_limit_ms=cpu_limit_ms,
            timeout_ms=timeout_ms,
        )
        if comp_failed:
            compilation_failed = True
            compile_error_output = comp_err
            test_results.append(case_res)
            # Remainder of cases cannot compile either; populate failed placeholders so
            # total_cases and sample/hidden counts remain consistent without redundant calls.
            for rem_tc in normalized_cases[idx + 1 :]:
                test_results.append(
                    TestCaseResult(
                        case_index=rem_tc.index,
                        case_number=rem_tc.index + 1,
                        is_sample=rem_tc.is_sample,
                        input=rem_tc.input,
                        expected=rem_tc.expected_output,
                        actual="",
                        stderr="",
                        compile_output=comp_err,
                        time_ms=0.0,
                        memory_kb=0,
                        status=STATUS_COMPILE_ERROR,
                        passed=False,
                        performance_failed=False,
                        status_detail=STATUS_COMPILE_ERROR,
                    )
                )
            break

        if case_res.time_ms > max_cpu_time_ms:
            max_cpu_time_ms = case_res.time_ms
        test_results.append(case_res)

        if not case_res.passed and stop_on_first_failure:
            break

    run_time_ms = (time.perf_counter() - run_start) * 1000.0
    passed_cases = sum(1 for r in test_results if r.passed)
    sample_passed = sum(1 for r in test_results if r.is_sample and r.passed)
    hidden_passed = sum(1 for r in test_results if (not r.is_sample) and r.passed)
    pass_rate = (passed_cases / total_cases) if total_cases > 0 else 0.0

    # Correctness is determined by hidden tests (or all tests if hidden_total == 0,
    # e.g. when running sample-only helper).
    if compilation_failed:
        overall_status = STATUS_COMPILE_ERROR
        correct = 0
        passed = False
    else:
        if hidden_total > 0:
            eval_results = [r for r in test_results if not r.is_sample]
        else:
            eval_results = test_results

        all_eval_passed = len(eval_results) > 0 and all(r.passed for r in eval_results)
        if all_eval_passed:
            correct = 1
            passed = True
            overall_status = STATUS_ACCEPTED
        else:
            correct = 0
            passed = False
            first_failed = next(
                (r for r in eval_results if not r.passed),
                next((r for r in test_results if not r.passed), None),
            )
            overall_status = first_failed.status if first_failed else STATUS_WRONG_ANSWER

    return JudgeRunResult(
        status=overall_status,
        passed=passed,
        correct=correct,
        compilation_failed=compilation_failed,
        compile_error_output=compile_error_output,
        test_results=test_results,
        total_cases=total_cases,
        passed_cases=passed_cases,
        test_case_pass_rate=pass_rate,
        sample_total=sample_total,
        sample_passed=sample_passed,
        hidden_total=hidden_total,
        hidden_passed=hidden_passed,
        max_cpu_time_ms=max_cpu_time_ms,
        run_time_ms=run_time_ms,
        memory_bytes=peak_memory_bytes,
    )


def compile_java_code(
    client: PistonClient,
    code: str,
    harness: str,
    timeout_ms: int | float | None = PISTON_MAX_RUN_TIMEOUT_MS,
) -> dict[str, Any]:
    """Tool helper for ``compile(code)``: returns compiler diagnostics without leaking tests."""
    extracted = extract_java_code(code) or (code.strip() if code else "")
    if not extracted:
        return {
            "compiled": False,
            "status": STATUS_NO_CODE,
            "diagnostics": "No Java code provided.",
        }
    merged = merge_java_solution_with_harness(extracted, harness)
    piston_data = client.execute_java(merged.code, stdin="", timeout_ms=timeout_ms)
    dummy_tc = JudgeTestCase(input="", expected_output="", index=0, is_sample=True)
    comp_failed, comp_err, _ = map_piston_response_to_case_result(
        piston_data=piston_data,
        merged=merged,
        tc=dummy_tc,
        timeout_ms=timeout_ms,
    )
    if comp_failed:
        return {
            "compiled": False,
            "status": STATUS_COMPILE_ERROR,
            "diagnostics": comp_err,
        }
    return {
        "compiled": True,
        "status": "Compiled",
        "diagnostics": "Compilation succeeded with no errors.",
    }


def run_java_custom(
    client: PistonClient,
    code: str,
    harness: str,
    stdin: str,
    timeout_ms: int | float | None = PISTON_MAX_RUN_TIMEOUT_MS,
) -> dict[str, Any]:
    """Tool helper for ``run_custom(code, stdin)``: runs candidate code on custom stdin."""
    extracted = extract_java_code(code) or (code.strip() if code else "")
    if not extracted:
        return {
            "status": STATUS_NO_CODE,
            "stdout": "",
            "stderr": "No Java code provided.",
            "time_ms": 0.0,
        }
    merged = merge_java_solution_with_harness(extracted, harness)
    start = time.perf_counter()
    piston_data = client.execute_java(merged.code, stdin=stdin or "", timeout_ms=timeout_ms)
    duration_ms = (time.perf_counter() - start) * 1000.0
    dummy_tc = JudgeTestCase(input=stdin or "", expected_output="", index=0, is_sample=True)
    comp_failed, comp_err, case_res = map_piston_response_to_case_result(
        piston_data=piston_data,
        merged=merged,
        tc=dummy_tc,
        duration_ms=duration_ms,
        timeout_ms=timeout_ms,
    )
    if comp_failed:
        return {
            "status": STATUS_COMPILE_ERROR,
            "stdout": "",
            "stderr": comp_err,
            "time_ms": 0.0,
        }
    status = case_res.status
    if status in (STATUS_ACCEPTED, STATUS_WRONG_ANSWER):
        status = "Executed"
    return {
        "status": status,
        "stdout": case_res.actual,
        "stderr": case_res.stderr,
        "time_ms": case_res.time_ms,
    }
