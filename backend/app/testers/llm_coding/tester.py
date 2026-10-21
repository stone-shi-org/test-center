"""LLM Coding Benchmark tester plugin (``llm_coding``).

Evaluates LLMs on verified Java coding problems from ``interview-question-bank``
(PrepLeet) using a dedicated Piston sidecar runner, with optional multi-attempt
feedback retries, OpenAI tool calling (``compile``, ``run_samples``, ``run_custom``,
``submit``), LLM code-quality judging, and composite scoring.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import time
from typing import Any

import httpx

from backend.app.db.metrics import format_metric
from backend.app.testers.base import (
    TesterCancelledError,
    TesterConfigError,
    TesterPlugin,
    TesterResult,
    TestContext,
)
from backend.app.testers.llm.client import ChatClientError, OpenAIChatClient
from backend.app.testers.llm_coding.dataset import (
    LoadedDataset,
    ProblemSpec,
    load_iqb_java_dataset,
)
from backend.app.testers.llm_coding.feedback import build_attempt_feedback
from backend.app.testers.llm_coding.harness import extract_java_code
from backend.app.testers.llm_coding.judge import (
    CodingJudgeResult,
    evaluate_with_coding_judge,
)
from backend.app.testers.llm_coding.piston import (
    JudgeRunResult,
    PistonClient,
    judge_java_solution,
    resolve_runner_url,
)
from backend.app.testers.llm_coding.prompts import (
    SYSTEM_PROMPT_STANDARD,
    SYSTEM_PROMPT_TOOLS,
    build_problem_prompt,
)
from backend.app.testers.llm_coding.report import format_llm_coding_report
from backend.app.testers.llm_coding.scoring import (
    DEFAULT_ATTEMPT_FACTORS,
    DEFAULT_WEIGHT_CORRECTNESS,
    DEFAULT_WEIGHT_JUDGE,
    aggregate_run_metrics,
    compute_case_score,
    parse_attempt_factors,
)
from backend.app.testers.llm_coding.tools import run_agent_tool_loop


class LLMCodingTester(TesterPlugin):
    """Tester plugin for Java coding benchmarks judged on Piston."""

    name = "llm_coding"
    description = (
        "LeetCode-style Java coding benchmark from interview-question-bank, "
        "judged on Piston with retries, agent tools, LLM judge, and composite score."
    )
    suites = ["all", "easy", "medium", "hard"]

    config_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            # Basic options
            "model": {
                "type": "string",
                "default": "auto",
                "description": "Model name to test (leave 'auto' to auto-detect from /models)",
            },
            "effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh"],
                "description": "Optional reasoning effort level (reasoning_effort) sent to the chat completions API",
            },
            "max_tokens": {
                "type": "integer",
                "default": 4096,
                "minimum": 64,
                "maximum": 65536,
                "description": "Maximum completion tokens per model generation turn",
            },
            "suites": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["all", "easy", "medium", "hard"],
                },
                "default": ["all"],
                "description": "Difficulty suites to run ('all' runs easy, medium, and hard)",
            },
            # Advanced: dataset & runner & filtering
            "dataset_dir": {
                "type": "string",
                "description": "Optional path override to the iqb-java dataset directory",
                "x-advanced": True,
            },
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Optional filter of specific problem IDs to evaluate",
                "x-advanced": True,
            },
            "difficulty": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                },
                "default": [],
                "description": "Optional difficulty filter (easy, medium, hard)",
                "x-advanced": True,
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Optional topic filter (matches problems with any listed topic)",
                "x-advanced": True,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "description": "Maximum number of problems to evaluate",
                "x-advanced": True,
            },
            "concurrency": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 16,
                "description": "Number of problems to evaluate concurrently",
                "x-advanced": True,
            },
            # NOTE: No 'default' on runner_url so apply_schema_defaults does not clobber target.extra['runner_url'].
            "runner_url": {
                "type": "string",
                "description": "Optional Piston sidecar URL override (defaults to target extra.runner_url or PISTON_URL)",
                "x-advanced": True,
            },
            # Advanced: Multi-attempt retry options (Section 4)
            "max_attempts": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 10,
                "description": "Maximum submission attempts per problem (typical retry value: 3)",
                "x-advanced": True,
            },
            "feedback_level": {
                "type": "string",
                "enum": ["verdict", "first_failure", "full"],
                "default": "first_failure",
                "description": "Sample-test feedback detail provided after a failed attempt",
                "x-advanced": True,
            },
            # Advanced: Agent tool calling options (Section 5)
            "tools_enabled": {
                "type": "boolean",
                "default": False,
                "description": "Enable OpenAI tool calling (compile, run_samples, run_custom, submit)",
                "x-advanced": True,
            },
            "max_tool_calls": {
                "type": "integer",
                "default": 12,
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum tool calls per problem when tools_enabled is true",
                "x-advanced": True,
            },
            "max_total_tokens": {
                "type": "integer",
                "default": 32768,
                "minimum": 256,
                "maximum": 524288,
                "description": "Maximum total tokens (prompt + completion) per problem in tool mode",
                "x-advanced": True,
            },
            # Advanced: LLM Judge options (Section 6)
            "judge_enabled": {
                "type": "boolean",
                "default": False,
                "description": "Enable LLM code-quality judge using dataset judge_prompt.md",
                "x-advanced": True,
            },
            "judge_target_id": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional Target ID for the LLM Judge (falls back to primary target)",
                "x-advanced": True,
            },
            "judge_model": {
                "type": "string",
                "description": "Optional model identifier override for the LLM Judge",
                "x-advanced": True,
            },
            "judge_failed_solutions": {
                "type": "boolean",
                "default": False,
                "description": "When true, also run the LLM judge on solutions that failed hidden tests",
                "x-advanced": True,
            },
            # Advanced: Scoring options (Section 7)
            "attempt_factors": {
                "type": "array",
                "items": {"type": "number"},
                "default": [1.0, 0.7, 0.5],
                "description": "Score multipliers for solving on attempt 1, 2, 3, ...",
                "x-advanced": True,
            },
            "weight_correctness": {
                "type": "number",
                "default": 0.8,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Weight of functional correctness when combining with LLM judge score",
                "x-advanced": True,
            },
            "weight_judge": {
                "type": "number",
                "default": 0.2,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Weight of normalized LLM judge score when judge is enabled",
                "x-advanced": True,
            },
            "per_case_timeout_ms": {
                "type": "integer",
                "default": 3000,
                "minimum": 500,
                "maximum": 10000,
                "description": "Per-case execution timeout in ms on Piston (clamped to max 3000ms by runner)",
                "x-advanced": True,
            },
            "cpu_limit_ms": {
                "type": "integer",
                "default": 3000,
                "minimum": 500,
                "description": "Per-case CPU time limit in ms before reporting Performance Limit Exceeded",
                "x-advanced": True,
            },
            "timeout_seconds": {
                "type": "number",
                "default": 180.0,
                "minimum": 5.0,
                "maximum": 1800.0,
                "description": "HTTP request timeout in seconds per LLM API call",
                "x-advanced": True,
            },
        },
    }

    def validate_and_apply_config(self, raw_config: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize optional None/empty fields and list fields before schema validation."""
        cleaned: dict[str, Any] = {}
        passthrough_private: dict[str, Any] = {}
        if isinstance(raw_config, dict):
            for k, v in raw_config.items():
                if k.startswith("_"):
                    passthrough_private[k] = v
                    continue
                if v is None:
                    continue
                if k in ("effort", "runner_url", "dataset_dir", "judge_model") and isinstance(v, str) and not v.strip():
                    continue
                if k in ("ids", "topics", "difficulty", "suites") and isinstance(v, str):
                    cleaned[k] = [item.strip() for item in v.split(",") if item.strip()]
                elif k == "attempt_factors" and isinstance(v, str):
                    cleaned[k] = parse_attempt_factors(v)
                else:
                    cleaned[k] = v

        validated = super().validate_and_apply_config(cleaned)
        validated.update(passthrough_private)
        return validated

    @staticmethod
    def _resolve_api_key(
        target_dict: dict[str, Any],
        config_dict: dict[str, Any],
        config_key_field: str = "api_key",
    ) -> str | None:
        auth_env_var = target_dict.get("auth_env_var")
        api_key: str | None = None
        if auth_env_var and isinstance(auth_env_var, str):
            api_key = os.environ.get(auth_env_var)
        if not api_key:
            api_key = config_dict.get(config_key_field) or os.environ.get("OPENAI_API_KEY")
        return api_key

    def _resolve_target_and_client(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
    ) -> tuple[Any, str]:
        if config.get("_chat_client") is not None:
            fake_client = config["_chat_client"]
            extra = target.get("extra") if isinstance(target.get("extra"), dict) else {}
            model = config.get("model") or extra.get("model") or target.get("model") or "fake-model"
            if str(model).strip().lower() in ("", "auto", "auto-detect"):
                model = "fake-model"
            return fake_client, str(model).strip()

        base_url = target.get("base_url") or config.get("base_url")
        if not base_url or not str(base_url).strip():
            raise TesterConfigError("Target definition is missing required 'base_url'")

        api_key = self._resolve_api_key(target, config, "api_key")
        timeout = float(config.get("timeout_seconds", 180.0))
        transport = config.get("_transport")

        client = OpenAIChatClient(
            base_url=str(base_url),
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )

        extra = target.get("extra") if isinstance(target.get("extra"), dict) else {}
        model = config.get("model") or extra.get("model") or target.get("model")
        if not model or str(model).strip().lower() in ("", "auto", "auto-detect"):
            detected = client.detect_model()
            model = detected or "gpt-4o-mini"

        return client, str(model).strip()

    def _resolve_judge(
        self,
        raw_config: dict[str, Any] | None,
        config: dict[str, Any],
        target: dict[str, Any],
        primary_client: Any,
        primary_model: str,
    ) -> tuple[Any | None, str | None, dict[str, Any]]:
        raw = raw_config if isinstance(raw_config, dict) else {}
        if raw.get("judge_enabled") is False:
            return None, None, {"enabled": False}

        has_judge_target_dict = isinstance(config.get("judge_target"), dict) or isinstance(
            target.get("judge_target"), dict
        )
        has_judge_target_id = config.get("judge_target_id") is not None
        has_judge_client = config.get("_judge_client") is not None
        judge_enabled = bool(
            config.get("judge_enabled")
            or has_judge_target_dict
            or has_judge_target_id
            or has_judge_client
        )

        if not judge_enabled:
            return None, None, {"enabled": False}

        if has_judge_client:
            j_client = config["_judge_client"]
            j_model = str(config.get("judge_model") or primary_model).strip()
            return (
                j_client,
                j_model,
                {
                    "enabled": True,
                    "target_id": config.get("judge_target_id"),
                    "target_name": "mock-judge-target",
                    "endpoint_url": getattr(j_client, "endpoint_url", "mock://judge"),
                    "model": j_model,
                },
            )

        judge_target: dict[str, Any] | None = None
        if isinstance(config.get("judge_target"), dict):
            judge_target = config["judge_target"]
        elif isinstance(target.get("judge_target"), dict):
            judge_target = target["judge_target"]
        elif has_judge_target_id:
            judge_target_id = int(config["judge_target_id"])
            try:
                from backend.app.db.models.target import Target as TargetModel
                from backend.app.db.session import SessionLocal

                with SessionLocal() as db:
                    t_obj = db.get(TargetModel, judge_target_id)
                    if t_obj is not None:
                        judge_target = t_obj.to_snapshot()
            except Exception:
                judge_target = None

            if judge_target is None:
                raise TesterConfigError(
                    f"Judge target ID {judge_target_id} was not found or could not be resolved"
                )
        else:
            # Fallback to primary target
            if config.get("_judge_transport") is None and config.get("_chat_client") is not None:
                j_model = str(config.get("judge_model") or primary_model).strip()
                return (
                    primary_client,
                    j_model,
                    {
                        "enabled": True,
                        "target_id": target.get("id"),
                        "target_name": target.get("name", "primary-target"),
                        "endpoint_url": getattr(primary_client, "endpoint_url", ""),
                        "model": j_model,
                    },
                )
            judge_target = dict(target)

        j_base_url = judge_target.get("base_url") or config.get("judge_base_url")
        if not j_base_url or not str(j_base_url).strip():
            raise TesterConfigError("Judge target is missing required 'base_url'")

        j_api_key = self._resolve_api_key(judge_target, config, "judge_api_key")
        timeout = float(config.get("timeout_seconds", 180.0))
        j_transport = config.get("_judge_transport") or config.get("_transport")

        judge_client = OpenAIChatClient(
            base_url=str(j_base_url),
            api_key=j_api_key,
            timeout=timeout,
            transport=j_transport,
        )

        j_extra = judge_target.get("extra") if isinstance(judge_target.get("extra"), dict) else {}
        judge_model = (
            config.get("judge_model")
            or j_extra.get("model")
            or judge_target.get("model")
        )
        if not judge_model or str(judge_model).strip().lower() in ("", "auto", "auto-detect"):
            judge_model = judge_client.detect_model() or primary_model or "gpt-4o-mini"

        judge_model = str(judge_model).strip()
        judge_info = {
            "enabled": True,
            "target_id": judge_target.get("id") or config.get("judge_target_id"),
            "target_name": judge_target.get("name", "judge-target"),
            "endpoint_url": judge_client.endpoint_url,
            "model": judge_model,
        }
        return judge_client, judge_model, judge_info

    def _resolve_piston_client(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
    ) -> tuple[PistonClient, str]:
        runner_url = resolve_runner_url(target=target, config=config)
        if config.get("_piston_client") is not None:
            p_client = config["_piston_client"]
            return p_client, getattr(p_client, "base_url", runner_url)

        http_client = None
        if config.get("_piston_transport") is not None:
            http_client = httpx.Client(transport=config["_piston_transport"], timeout=30.0)

        p_client = PistonClient(
            base_url=runner_url,
            http_client=http_client,
            execute_hook=config.get("_piston_execute_hook"),
        )
        return p_client, runner_url

    def _evaluate_single_problem(
        self,
        problem: ProblemSpec,
        dataset: LoadedDataset,
        chat_client: Any,
        model: str,
        piston_client: PistonClient,
        judge_client: Any | None,
        judge_model: str | None,
        judge_info: dict[str, Any],
        config: dict[str, Any],
        ctx: TestContext,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Evaluate a single Java coding problem (standard/retry or agent tool mode + optional judge)."""
        ctx.check_cancelled()

        max_tokens = int(config.get("max_tokens", 4096))
        effort = config.get("effort")
        max_attempts = max(1, int(config.get("max_attempts", 1)))
        feedback_level = str(config.get("feedback_level", "first_failure"))
        tools_enabled = bool(config.get("tools_enabled", False))
        max_tool_calls = int(config.get("max_tool_calls", 12))
        max_total_tokens = int(config.get("max_total_tokens", 32768))
        judge_failed_solutions = bool(config.get("judge_failed_solutions", False))
        attempt_factors = parse_attempt_factors(config.get("attempt_factors", DEFAULT_ATTEMPT_FACTORS))
        weight_correctness = float(config.get("weight_correctness", DEFAULT_WEIGHT_CORRECTNESS))
        weight_judge = float(config.get("weight_judge", DEFAULT_WEIGHT_JUDGE))

        configured_timeout = config.get("per_case_timeout_ms")
        eff_timeout_ms = int(configured_timeout) if configured_timeout is not None else max(int(problem.timeout_ms or 3000), 3000)

        configured_cpu_limit = config.get("cpu_limit_ms")
        if configured_cpu_limit is not None:
            eff_cpu_limit_ms = int(configured_cpu_limit)
        elif problem.cpu_limit_ms is not None:
            eff_cpu_limit_ms = max(int(problem.cpu_limit_ms), 3000)
        else:
            eff_cpu_limit_ms = None

        user_prompt = build_problem_prompt(problem, tools_enabled=tools_enabled)
        sys_prompt = SYSTEM_PROMPT_TOOLS if tools_enabled else SYSTEM_PROMPT_STANDARD
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        attempt_tps: list[float] = []
        tool_events: list[dict[str, Any]] = []
        tool_call_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        solved = False
        solved_attempt: int | None = None
        final_code = ""
        final_judge_run: JudgeRunResult | None = None

        if tools_enabled:
            tool_outcome = run_agent_tool_loop(
                chat_client=chat_client,
                piston_client=piston_client,
                problem=problem,
                model=model,
                initial_messages=messages,
                max_attempts=max_attempts,
                feedback_level=feedback_level,
                max_tool_calls=max_tool_calls,
                max_total_tokens=max_total_tokens,
                max_tokens_per_turn=max_tokens,
                effort=effort,
                timeout_ms=eff_timeout_ms,
                cpu_limit_ms=eff_cpu_limit_ms,
                should_cancel=lambda: ctx.is_cancelled,
            )
            ctx.check_cancelled()
            messages = tool_outcome.transcript
            attempts = tool_outcome.attempts
            tool_events = tool_outcome.tool_events
            tool_call_count = tool_outcome.tool_call_count
            prompt_tokens = tool_outcome.prompt_tokens
            completion_tokens = tool_outcome.completion_tokens
            solved = tool_outcome.solved
            solved_attempt = tool_outcome.solved_attempt
            final_code = tool_outcome.final_code
            final_judge_result = tool_outcome.final_judge_result
            final_judge_run = final_judge_result
        else:
            for attempt_num in range(1, max_attempts + 1):
                ctx.check_cancelled()
                stream_res = chat_client.stream_chat(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    effort=effort,
                )
                prompt_tokens += int(getattr(stream_res, "prompt_tokens", 0) or 0)
                completion_tokens += int(getattr(stream_res, "completion_tokens", 0) or 0)
                tps_val = getattr(stream_res, "tokens_per_sec", None)
                if tps_val is not None and float(tps_val) > 0:
                    cur_tps = float(tps_val)
                elif hasattr(stream_res, "decode_duration_ms") and getattr(stream_res, "decode_duration_ms") > 0:
                    dec_s = getattr(stream_res, "decode_duration_ms") / 1000.0
                    c_tok = int(getattr(stream_res, "completion_tokens", 0) or 0)
                    cur_tps = (c_tok / dec_s) if dec_s > 0 else 0.0
                else:
                    cur_tps = 0.0
                if cur_tps > 0:
                    attempt_tps.append(cur_tps)
                response_text = str(getattr(stream_res, "content", "") or "")
                messages.append({"role": "assistant", "content": response_text})

                extracted_code = extract_java_code(response_text)
                judge_run = judge_java_solution(
                    client=piston_client,
                    code=extracted_code,
                    harness=problem.harness_java,
                    test_cases=problem.tests,
                    sample_case_idx=problem.sample_case_idx,
                    timeout_ms=eff_timeout_ms,
                    cpu_limit_ms=eff_cpu_limit_ms,
                    stop_on_first_failure=False,
                    extract_from_markdown=False,
                )
                final_code = extracted_code or ""
                final_judge_run = judge_run

                attempt_entry: dict[str, Any] = {
                    "attempt": attempt_num,
                    "response": response_text,
                    "code": final_code,
                    "tokens_per_sec": round(cur_tps, 2),
                    "status": judge_run.status,
                    "passed": judge_run.passed,
                    "correct": judge_run.correct,
                    "passed_cases": judge_run.passed_cases,
                    "total_cases": judge_run.total_cases,
                    "test_case_pass_rate": judge_run.test_case_pass_rate,
                    "sample_passed": judge_run.sample_passed,
                    "sample_total": judge_run.sample_total,
                    "hidden_passed": judge_run.hidden_passed,
                    "hidden_total": judge_run.hidden_total,
                    "compile_error_output": judge_run.compile_error_output,
                    "run_time_ms": judge_run.run_time_ms,
                    "max_cpu_time_ms": judge_run.max_cpu_time_ms,
                }

                if judge_run.passed:
                    solved = True
                    solved_attempt = attempt_num
                    attempts.append(attempt_entry)
                    break

                if attempt_num < max_attempts:
                    fb_text = build_attempt_feedback(
                        judge_result=judge_run,
                        feedback_level=feedback_level,
                        attempt_number=attempt_num,
                        max_attempts=max_attempts,
                    )
                    attempt_entry["feedback"] = fb_text
                    messages.append({"role": "user", "content": fb_text})

                attempts.append(attempt_entry)

        wall_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        assert final_judge_run is not None

        # Optional LLM Judge evaluation (Section 6)
        judge_result: CodingJudgeResult | None = None
        should_run_judge = (
            judge_client is not None
            and judge_model is not None
            and (solved or judge_failed_solutions)
            and bool(final_code.strip())
        )
        if should_run_judge:
            ctx.check_cancelled()
            judge_result = evaluate_with_coding_judge(
                judge_client=judge_client,
                judge_model=str(judge_model),
                judge_target_name=str(judge_info.get("target_name", "judge-target")),
                judge_prompt_md=dataset.judge_prompt_md,
                problem=problem,
                candidate_code=final_code,
                judge_run=final_judge_run,
            )

        if tools_enabled:
            case_tps = tool_outcome.mean_tps
        else:
            case_tps = sum(attempt_tps) / len(attempt_tps) if attempt_tps else 0.0

        score_breakdown = compute_case_score(
            correct=1 if solved else 0,
            solved_attempt=solved_attempt,
            attempt_factors=attempt_factors,
            judge_enabled=bool(judge_client is not None),
            judge_overall=judge_result.overall if judge_result is not None else None,
            weight_correctness=weight_correctness,
            weight_judge=weight_judge,
        )

        total_tokens = prompt_tokens + completion_tokens
        tc_pass_rate_pct = round(final_judge_run.test_case_pass_rate * 100.0, 2)

        case_metrics: dict[str, Any] = {
            "score": format_metric(score_breakdown.case_score_100, unit="pts", better="higher"),
            "case_score": format_metric(score_breakdown.case_score_100, unit="pts", better="higher"),
            "correct": format_metric(score_breakdown.correct, unit="bool", better="higher"),
            "test_case_pass_rate": format_metric(tc_pass_rate_pct, unit="%", better="higher"),
            "attempts_used": format_metric(len(attempts), unit="attempts", better="lower"),
            "tool_call_count": format_metric(tool_call_count, unit="calls", better="lower"),
            "wall_time_ms": format_metric(wall_time_ms, unit="ms", better="lower"),
            "tokens_per_sec": format_metric(round(case_tps, 2), unit="tokens/s", better="higher"),
            "prompt_tokens": format_metric(prompt_tokens, unit="tokens", better="lower"),
            "completion_tokens": format_metric(completion_tokens, unit="tokens", better="lower"),
            "total_tokens": format_metric(total_tokens, unit="tokens", better="lower"),
        }
        if judge_result is not None:
            case_metrics["judge_overall"] = format_metric(
                judge_result.overall, unit="/5", better="higher"
            )
            case_metrics["judge_approach_correctness"] = format_metric(
                judge_result.approach_correctness, unit="/5", better="higher"
            )
            case_metrics["judge_complexity_vs_reference"] = format_metric(
                judge_result.complexity_vs_reference, unit="/5", better="higher"
            )
            case_metrics["judge_readability_idiom"] = format_metric(
                judge_result.readability_idiom, unit="/5", better="higher"
            )

        case_detail: dict[str, Any] = {
            "problem_id": problem.id,
            "title": problem.title,
            "difficulty": problem.difficulty,
            "topics": problem.topics,
            "source": problem.source,
            "leetcode_id": problem.leetcode_id,
            "dataset_version": dataset.manifest.version,
            "manifest_sha256": dataset.manifest.manifest_sha256,
            "problem_sha256": problem.sha256,
            "problem_metadata": problem.metadata_dict(),
            "verdict": final_judge_run.status,
            "passed": solved,
            "correct": score_breakdown.correct,
            "attempts_used": len(attempts),
            "solved_attempt": score_breakdown.solved_attempt,
            "attempt_factor": score_breakdown.attempt_factor,
            "case_score": score_breakdown.case_score,
            "case_score_100": score_breakdown.case_score_100,
            "passed_cases": final_judge_run.passed_cases,
            "total_cases": final_judge_run.total_cases,
            "test_case_pass_rate": tc_pass_rate_pct,
            "sample_passed": final_judge_run.sample_passed,
            "sample_total": final_judge_run.sample_total,
            "hidden_passed": final_judge_run.hidden_passed,
            "hidden_total": final_judge_run.hidden_total,
            "attempts": attempts,
            "transcript": messages,
            "judge": judge_result.to_dict() if judge_result is not None else None,
            "tools_enabled": tools_enabled,
            "tool_call_count": tool_call_count,
            "tool_events": tool_events,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_sec": round(case_tps, 2),
            "wall_time_ms": wall_time_ms,
        }

        agg_record: dict[str, Any] = {
            "problem_id": problem.id,
            "difficulty": problem.difficulty,
            "correct": score_breakdown.correct,
            "solved_attempt": score_breakdown.solved_attempt,
            "case_score": score_breakdown.case_score,
            "test_case_pass_rate": final_judge_run.test_case_pass_rate,
            "judge_overall": judge_result.overall if judge_result is not None else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_call_count": tool_call_count,
            "tokens_per_sec": round(case_tps, 2),
            "wall_time_ms": wall_time_ms,
        }

        return case_metrics, case_detail, agg_record

    def _update_result_metrics(
        self,
        result: TesterResult,
        completed_records: list[dict[str, Any]],
        judge_enabled: bool,
        dataset: LoadedDataset,
    ) -> None:
        agg = aggregate_run_metrics(completed_records, judge_enabled=judge_enabled)
        for m_name, m_obj in agg.items():
            result.set_metric(
                m_name,
                m_obj["value"],
                unit=m_obj.get("unit"),
                better=m_obj.get("better"),
            )
        result.set_metric("problems_evaluated", len(completed_records), unit="problems", better="higher")
        result.set_metric(
            "problems_solved",
            sum(1 for r in completed_records if r.get("correct") == 1),
            unit="problems",
            better="higher",
        )

    def run(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
        ctx: TestContext,
    ) -> TesterResult:
        raw_config = dict(config) if isinstance(config, dict) else {}
        config = self.validate_and_apply_config(config)
        ctx.check_cancelled()

        chat_client, model = self._resolve_target_and_client(config, target)
        judge_client, judge_model, judge_info = self._resolve_judge(
            raw_config, config, target, chat_client, model
        )
        piston_client, runner_url = self._resolve_piston_client(config, target)

        suites_cfg = config.get("suites") or ["all"]
        diff_cfg = list(config.get("difficulty") or [])
        if isinstance(suites_cfg, list):
            for s in suites_cfg:
                s_norm = str(s).strip().lower()
                if s_norm in ("easy", "medium", "hard") and s_norm not in diff_cfg:
                    diff_cfg.append(s_norm)

        try:
            dataset = load_iqb_java_dataset(
                dataset_dir=config.get("dataset_dir"),
                verify_hash=True,
                suite_id="all",
                ids=config.get("ids"),
                difficulty=diff_cfg if diff_cfg else None,
                topics=config.get("topics"),
                limit=config.get("limit"),
            )
        except Exception as exc:
            raise TesterConfigError(f"Failed to load iqb-java dataset: {exc}") from exc

        problems = dataset.problems
        if not problems:
            raise TesterConfigError("No coding problems matched the configured suite/filter criteria")

        concurrency = max(1, int(config.get("concurrency", 1)))
        max_attempts = max(1, int(config.get("max_attempts", 1)))
        tools_enabled = bool(config.get("tools_enabled", False))
        effort = config.get("effort")

        ctx.log(
            f"Starting llm_coding benchmark: model='{model}', problems={len(problems)}, "
            f"runner='{runner_url}', max_attempts={max_attempts}, tools={tools_enabled}, "
            f"judge={bool(judge_client)}, concurrency={concurrency}, "
            f"dataset_version={dataset.manifest.version} (sha256={dataset.manifest.manifest_sha256[:12]})"
        )

        result = TesterResult(status="pass")
        completed_records: list[dict[str, Any]] = []
        lock = threading.Lock()
        n_problems = len(problems)

        def _record_problem_completion(
            prob: ProblemSpec,
            c_metrics: dict[str, Any],
            c_detail: dict[str, Any],
            agg_rec: dict[str, Any],
        ) -> None:
            with lock:
                c_status = "pass" if c_detail.get("passed") else "fail"
                result.add_case(
                    case_id=prob.id,
                    suite=prob.difficulty,
                    status=c_status,
                    metrics=c_metrics,
                    detail=c_detail,
                )
                completed_records.append(agg_rec)
                self._update_result_metrics(
                    result=result,
                    completed_records=completed_records,
                    judge_enabled=bool(judge_client),
                    dataset=dataset,
                )
                done_count = len(completed_records)
                ctx.log(
                    f"[{done_count}/{n_problems}] {prob.id} ({prob.difficulty}) -> {c_detail['verdict']} "
                    f"(attempts={c_detail['attempts_used']}, tests={c_detail['passed_cases']}/{c_detail['total_cases']}, "
                    f"score={c_detail['case_score_100']:.1f})"
                )
                ctx.progress(
                    done_count,
                    n_problems,
                    f"Completed {done_count}/{n_problems}: {prob.id} ({c_detail['verdict']})",
                )

        if concurrency == 1:
            for prob in problems:
                ctx.check_cancelled()
                c_metrics, c_detail, agg_rec = self._evaluate_single_problem(
                    problem=prob,
                    dataset=dataset,
                    chat_client=chat_client,
                    model=model,
                    piston_client=piston_client,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    judge_info=judge_info,
                    config=config,
                    ctx=ctx,
                )
                _record_problem_completion(prob, c_metrics, c_detail, agg_rec)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                future_to_prob = {
                    pool.submit(
                        self._evaluate_single_problem,
                        prob,
                        dataset,
                        chat_client,
                        model,
                        piston_client,
                        judge_client,
                        judge_model,
                        judge_info,
                        config,
                        ctx,
                    ): prob
                    for prob in problems
                }
                for fut in as_completed(future_to_prob):
                    if ctx.is_cancelled:
                        for pending in future_to_prob:
                            pending.cancel()
                        ctx.check_cancelled()
                    prob = future_to_prob[fut]
                    c_metrics, c_detail, agg_rec = fut.result()
                    _record_problem_completion(prob, c_metrics, c_detail, agg_rec)

        # Sort result.cases to deterministic problem order
        order_map = {p.id: i for i, p in enumerate(problems)}
        result.cases.sort(key=lambda c: order_map.get(c.case_id, 999999))

        self._update_result_metrics(
            result=result,
            completed_records=completed_records,
            judge_enabled=bool(judge_client),
            dataset=dataset,
        )

        solved_count = sum(1 for r in completed_records if r.get("correct") == 1)
        result.status = "pass" if solved_count > 0 else "fail"
        score_val = result.metrics.get("score", {}).get("value", 0.0)
        s1_val = result.metrics.get("solve_at_1", {}).get("value", 0.0)
        sk_val = result.metrics.get("solve_at_k", {}).get("value", 0.0)
        tc_val = result.metrics.get("test_case_pass_rate", {}).get("value", 0.0)

        result.summary = (
            f"LLM Coding ({model}): score={score_val:.1f}/100, "
            f"solve@1={s1_val:.1f}%, solve@{max_attempts}={sk_val:.1f}% "
            f"({solved_count}/{n_problems} solved), test_cases={tc_val:.1f}%"
        )

        endpoint_url = getattr(chat_client, "endpoint_url", str(target.get("base_url") or ""))
        report_text = format_llm_coding_report(
            model=model,
            endpoint_url=endpoint_url,
            runner_url=runner_url,
            dataset_version=dataset.manifest.version,
            manifest_sha256=dataset.manifest.manifest_sha256,
            metrics=result.metrics,
            cases=[c.to_dict() for c in result.cases],
            judge_info=judge_info,
            effort=effort,
            tools_enabled=tools_enabled,
            max_attempts=max_attempts,
        )
        report_file = ctx.artifacts_dir / "llm_coding_report.txt"
        report_file.write_text(report_text, encoding="utf-8")
        ctx.add_artifact("llm_coding_report.txt", report_file, mime="text/plain")

        results_payload = {
            "tester": self.name,
            "model": model,
            "endpoint_url": endpoint_url,
            "runner_url": runner_url,
            "dataset": {
                "version": dataset.manifest.version,
                "exported_at": dataset.manifest.exported_at,
                "source": dataset.manifest.source,
                "manifest_sha256": dataset.manifest.manifest_sha256,
                "judge_prompt_sha256": dataset.manifest.judge_prompt_sha256,
                "problem_count": dataset.manifest.problem_count,
            },
            "config": {
                k: v for k, v in config.items() if not k.startswith("_")
            },
            "judge": judge_info,
            "status": result.status,
            "summary": result.summary,
            "metrics": result.metrics,
            "cases": [c.to_dict() for c in result.cases],
        }
        json_file = ctx.artifacts_dir / "llm_coding_results.json"
        json_file.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        ctx.add_artifact("llm_coding_results.json", json_file, mime="application/json")

        for art in ctx.artifacts:
            if art not in result.artifacts:
                result.artifacts.append(art)

        ctx.log(f"llm_coding benchmark finished: {result.summary}")
        return result
