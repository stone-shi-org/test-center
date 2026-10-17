"""OpenAI-compatible LLM chat benchmark and capability tester plugin."""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Any

from backend.app.testers.base import (
    TesterPlugin,
    TestContext,
    TesterResult,
    TesterConfigError,
    TesterError,
)
from backend.app.testers.llm.cases import (
    BUILTIN_CAPABILITY_CASES,
    CapabilityCase,
    build_judge_messages,
    get_builtin_cases,
    parse_judge_response,
)
from backend.app.testers.llm.client import (
    OpenAIChatClient,
    ChatClientError,
    ChatStreamResult,
)
from backend.app.testers.llm.metrics import compute_speed_summary
from backend.app.testers.llm.report import format_llm_report


_SUITE_ALIASES: dict[str, str] = {
    "coding": "coding",
    "code": "coding",
    "tool_call": "tool_call",
    "tool_call_json": "tool_call",
    "tool-call": "tool_call",
    "agentic": "tool_call",
    "constraints": "constraints",
    "constraint": "constraints",
    "constraint_following": "constraints",
    "writing": "constraints",
    "reasoning": "reasoning",
    "logic": "reasoning",
    "summarization": "summarization",
    "summary": "summarization",
}


class LLMTester(TesterPlugin):
    """Tester benchmarking OpenAI-compatible chat endpoints.

    Evaluates:
    - Speed Metrics (streaming): TTFT (mean/min/max/p50/p95/p99), decode tokens/s,
      prefill tokens/s, and end-to-end latency percentiles (p50/p95/p99).
    - Capability Cases:
      1. Coding (executed unit test checks in an isolated sandbox subprocess)
      2. Tool-call JSON (prerequisite selection, JSON validity, argument binding)
      3. Constraint following (structural & lexical constraints)
      4. Reasoning (multi-step logic riddle + terminal marker)
      5. Summarization (length, single-sentence, and concept coverage checks)
    - Optional LLM-as-a-Judge using a second target for 1-5 quality scoring and critique.
    """

    name = "llm"
    description = (
        "Benchmarks OpenAI-compatible chat endpoints for streaming speed (TTFT, tokens/s, "
        "latency percentiles), capability cases (coding, tool-call JSON, constraints, "
        "reasoning, summarization), and optional LLM-as-a-Judge via a second target."
    )
    suites = ["coding", "tool_call", "constraints", "reasoning", "summarization"]

    config_schema = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Chat model identifier (overrides target extra.model; auto-detected if omitted)",
            },
            "suites": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["coding", "tool_call", "constraints", "reasoning", "summarization"],
                },
                "default": ["coding", "tool_call", "constraints", "reasoning", "summarization"],
                "description": "Capability sub-suites to execute",
            },
            "max_tokens": {
                "type": "integer",
                "default": 2048,
                "minimum": 16,
                "maximum": 32768,
                "description": "Maximum completion tokens per request",
            },
            "timeout_seconds": {
                "type": "number",
                "default": 120.0,
                "minimum": 1.0,
                "description": "HTTP request timeout in seconds",
            },
            "repetitions": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 10,
                "description": "Number of streaming runs per case for latency percentile calculation",
            },
            "coding_timeout_seconds": {
                "type": "number",
                "default": 5.0,
                "minimum": 1.0,
                "maximum": 30.0,
                "description": "Subprocess timeout in seconds for sandboxed coding checks",
            },
            "judge_enabled": {
                "type": "boolean",
                "default": False,
                "description": "Enable optional LLM-as-a-Judge evaluation using a second target",
            },
            "judge_target_id": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional Target ID for the second target acting as LLM Judge",
            },
            "judge_model": {
                "type": "string",
                "description": "Optional model identifier override for the LLM Judge",
            },
            "judge_min_score": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 5,
                "description": "Minimum judge score (1-5) required to pass judge evaluation",
            },
            "judge_require_pass": {
                "type": "boolean",
                "default": True,
                "description": "When true, cases evaluated by the judge must also pass judge threshold",
            },
        },
    }

    def validate_and_apply_config(self, raw_config: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize optional None fields and suite aliases before schema validation."""
        cleaned: dict[str, Any] = {}
        if isinstance(raw_config, dict):
            for k, v in raw_config.items():
                if v is None:
                    continue
                if k == "suites" and isinstance(v, list):
                    normalized_suites = []
                    for item in v:
                        mapped = _SUITE_ALIASES.get(str(item).strip().lower(), item)
                        if mapped not in normalized_suites:
                            normalized_suites.append(mapped)
                    cleaned[k] = normalized_suites
                else:
                    cleaned[k] = v
        return super().validate_and_apply_config(cleaned)

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
    ) -> tuple[OpenAIChatClient, str]:
        """Resolve primary endpoint URL, credentials, model name, and client instance."""
        base_url = target.get("base_url") or config.get("base_url")
        if not base_url or not str(base_url).strip():
            raise TesterConfigError("Target definition is missing required 'base_url'")

        api_key = self._resolve_api_key(target, config, "api_key")
        timeout = float(config.get("timeout_seconds", 120.0))
        transport = config.get("_transport")

        client = OpenAIChatClient(
            base_url=str(base_url),
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )

        extra = target.get("extra") if isinstance(target.get("extra"), dict) else {}
        model = (
            config.get("model")
            or extra.get("model")
            or target.get("model")
        )
        if not model or str(model).strip().lower() in ("", "auto", "auto-detect"):
            detected = client.detect_model()
            model = detected or "gpt-4o-mini"

        return client, str(model).strip()

    def _resolve_judge(
        self,
        raw_config: dict[str, Any] | None,
        config: dict[str, Any],
        target: dict[str, Any],
    ) -> tuple[OpenAIChatClient | None, str | None, dict[str, Any]]:
        """Resolve optional LLM Judge client and model from second target or config."""
        raw = raw_config if isinstance(raw_config, dict) else {}

        # If caller explicitly passed judge_enabled=False, disable judge
        if raw.get("judge_enabled") is False:
            return None, None, {"enabled": False}

        has_judge_target_dict = isinstance(config.get("judge_target"), dict) or isinstance(
            target.get("judge_target"), dict
        )
        has_judge_target_id = config.get("judge_target_id") is not None
        judge_enabled = bool(
            config.get("judge_enabled") or has_judge_target_dict or has_judge_target_id
        )

        if not judge_enabled:
            return None, None, {"enabled": False}

        judge_target: dict[str, Any] | None = None
        if isinstance(config.get("judge_target"), dict):
            judge_target = config["judge_target"]
        elif isinstance(target.get("judge_target"), dict):
            judge_target = target["judge_target"]
        elif has_judge_target_id:
            judge_target_id = int(config["judge_target_id"])
            try:
                from backend.app.db.session import SessionLocal
                from backend.app.db.models.target import Target as TargetModel

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
            # Fallback to primary target if judge_enabled=True without a separate target
            judge_target = dict(target)

        j_base_url = judge_target.get("base_url") or config.get("judge_base_url")
        if not j_base_url or not str(j_base_url).strip():
            raise TesterConfigError("Judge target is missing required 'base_url'")

        j_api_key = self._resolve_api_key(judge_target, config, "judge_api_key")
        timeout = float(config.get("timeout_seconds", 120.0))
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
            judge_model = judge_client.detect_model() or "gpt-4o-mini"

        judge_model = str(judge_model).strip()
        judge_info = {
            "enabled": True,
            "target_id": judge_target.get("id") or config.get("judge_target_id"),
            "target_name": judge_target.get("name", "judge-target"),
            "endpoint_url": judge_client.endpoint_url,
            "model": judge_model,
            "min_score": int(config.get("judge_min_score", 3)),
            "require_pass": bool(config.get("judge_require_pass", True)),
        }
        return judge_client, judge_model, judge_info

    def run(
        self,
        config: dict[str, Any],
        target: dict[str, Any],
        ctx: TestContext,
    ) -> TesterResult:
        cfg = self.validate_and_apply_config(config)
        client, model = self._resolve_target_and_client(cfg, target)
        judge_client, judge_model, judge_info = self._resolve_judge(config, cfg, target)

        raw_suites = cfg.get("suites") or target.get("suites") or self.suites
        normalized_suites = [
            _SUITE_ALIASES.get(str(s).strip().lower(), str(s)) for s in raw_suites
        ]
        suites_to_run = [s for s in self.suites if s in normalized_suites]
        if not suites_to_run:
            suites_to_run = list(self.suites)

        cases_to_run: list[CapabilityCase] = get_builtin_cases(suites_to_run)
        if not cases_to_run:
            cases_to_run = list(BUILTIN_CAPABILITY_CASES)

        max_tokens = int(cfg.get("max_tokens", 2048))
        repetitions = int(cfg.get("repetitions", 1))
        coding_timeout = float(cfg.get("coding_timeout_seconds", 5.0))
        judge_min_score = int(cfg.get("judge_min_score", 3))
        judge_require_pass = bool(cfg.get("judge_require_pass", True))

        ctx.log(
            f"Starting LLM benchmark for target='{target.get('name', 'default')}', "
            f"model='{model}', suites={suites_to_run}, repetitions={repetitions}"
        )
        ctx.log(f"Endpoint: {client.endpoint_url} ({len(cases_to_run)} capability cases)")
        if judge_client and judge_model:
            ctx.log(
                f"LLM Judge enabled: model='{judge_model}', endpoint={judge_client.endpoint_url}, "
                f"min_score={judge_min_score}"
            )

        result = TesterResult(status="pass")
        all_ttfts_ms: list[float] = []
        all_latencies_ms: list[float] = []
        all_decode_tps: list[float] = []
        all_prefill_tps: list[float] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_decode_ms = 0.0
        judge_scores: list[int] = []
        judge_passes: list[bool] = []

        t_run_start = time.perf_counter()
        n_cases = len(cases_to_run)

        for idx, case in enumerate(cases_to_run):
            ctx.check_cancelled()
            ctx.log(f"Running case [{idx + 1}/{n_cases}] '{case.id}' ({case.name})...")

            messages = [
                {"role": "system", "content": case.system_prompt},
                {"role": "user", "content": case.prompt},
            ]

            case_ttfts: list[float] = []
            case_latencies: list[float] = []
            case_decode_tps: list[float] = []
            case_prefill_tps: list[float] = []
            last_stream: ChatStreamResult | None = None

            for rep in range(repetitions):
                ctx.check_cancelled()
                try:
                    stream_res = client.stream_chat(
                        messages=messages,
                        model=model,
                        max_tokens=max_tokens,
                    )
                except ChatClientError as e:
                    ctx.log(f"Error calling LLM endpoint on case '{case.id}': {e}", level="error")
                    raise TesterError(f"LLM API error: {e}") from e

                last_stream = stream_res
                case_ttfts.append(stream_res.ttft_ms)
                case_latencies.append(stream_res.latency_ms)
                case_decode_tps.append(stream_res.tokens_per_sec)
                case_prefill_tps.append(stream_res.prefill_tps)

                all_ttfts_ms.append(stream_res.ttft_ms)
                all_latencies_ms.append(stream_res.latency_ms)
                all_decode_tps.append(stream_res.tokens_per_sec)
                all_prefill_tps.append(stream_res.prefill_tps)

                total_prompt_tokens += stream_res.prompt_tokens
                total_completion_tokens += stream_res.completion_tokens
                total_decode_ms += (
                    stream_res.decode_duration_ms
                    if stream_res.decode_duration_ms > 0
                    else stream_res.latency_ms
                )

            assert last_stream is not None
            response_text = last_stream.content

            # Evaluate capability locally
            outcome = case.evaluator(response_text, timeout_seconds=coding_timeout)
            local_passed = outcome.passed

            # Optional LLM-as-a-Judge evaluation on second target
            judge_detail: dict[str, Any] = {"enabled": bool(judge_client)}
            case_judge_score: int | None = None
            case_judge_passed: bool | None = None

            if judge_client and judge_model:
                ctx.check_cancelled()
                judge_msgs = build_judge_messages(
                    case=case,
                    response_text=response_text,
                    local_passed=local_passed,
                    local_message=outcome.message,
                    min_score=judge_min_score,
                )
                try:
                    judge_raw, j_usage, j_dt = judge_client.chat(
                        messages=judge_msgs,
                        model=judge_model,
                        max_tokens=512,
                        response_format={"type": "json_object"},
                    )
                    parsed_j = parse_judge_response(judge_raw, min_score=judge_min_score)
                    case_judge_score = parsed_j["score"]
                    case_judge_passed = parsed_j["passed"]
                    judge_detail = {
                        "enabled": True,
                        "model": judge_model,
                        "endpoint_url": judge_client.endpoint_url,
                        "score": case_judge_score,
                        "passed": case_judge_passed,
                        "justification": parsed_j["justification"],
                        "latency_ms": j_dt,
                        "usage": j_usage,
                        "error": parsed_j["error"],
                    }
                    if case_judge_score is not None:
                        judge_scores.append(case_judge_score)
                    if case_judge_passed is not None:
                        judge_passes.append(case_judge_passed)
                    ctx.log(
                        f"  Judge rating for '{case.id}': score={case_judge_score}/5, "
                        f"passed={case_judge_passed}"
                    )
                except ChatClientError as e:
                    ctx.log(f"  Judge call failed for '{case.id}': {e}", level="warning")
                    judge_detail = {
                        "enabled": True,
                        "model": judge_model,
                        "endpoint_url": judge_client.endpoint_url,
                        "score": None,
                        "passed": None,
                        "justification": None,
                        "error": str(e),
                    }

            if judge_client and judge_require_pass and case_judge_passed is not None:
                overall_case_passed = local_passed and case_judge_passed
            else:
                overall_case_passed = local_passed

            case_status = "pass" if overall_case_passed else "fail"
            mean_case_ttft = round(float(statistics.mean(case_ttfts)), 2)
            mean_case_lat = round(float(statistics.mean(case_latencies)), 2)
            mean_case_tps = round(float(statistics.mean(case_decode_tps)), 2)
            mean_case_prefill = round(float(statistics.mean(case_prefill_tps)), 2)

            case_metrics: dict[str, Any] = {
                "ttft_ms": {"value": mean_case_ttft, "unit": "ms", "better": "lower"},
                "latency_ms": {"value": mean_case_lat, "unit": "ms", "better": "lower"},
                "tokens_per_sec": {"value": mean_case_tps, "unit": "tokens/s", "better": "higher"},
                "tps": {"value": mean_case_tps, "unit": "tokens/s", "better": "higher"},
                "prefill_tps": {"value": mean_case_prefill, "unit": "tokens/s", "better": "higher"},
                "prompt_tokens": {"value": last_stream.prompt_tokens, "unit": "tokens"},
                "completion_tokens": {"value": last_stream.completion_tokens, "unit": "tokens"},
                "total_tokens": {"value": last_stream.total_tokens, "unit": "tokens"},
            }
            if case_judge_score is not None:
                case_metrics["judge_score"] = {
                    "value": case_judge_score,
                    "unit": "score",
                    "better": "higher",
                }

            case_detail: dict[str, Any] = {
                "case_name": case.name,
                "category": case.category,
                "description": case.description,
                "prompt": case.prompt,
                "response": response_text,
                "passed": overall_case_passed,
                "local_passed": local_passed,
                "eval_message": outcome.message,
                "checks": outcome.checks,
                "judge": judge_detail,
                **outcome.extra,
            }

            result.add_case(
                case_id=case.id,
                suite=case.suite,
                status=case_status,
                metrics=case_metrics,
                detail=case_detail,
            )

            ctx.log(
                f"  Case '{case.id}' -> {case_status.upper()} | "
                f"TTFT={mean_case_ttft:.2f}ms, TPS={mean_case_tps:.2f}, "
                f"Latency={mean_case_lat:.2f}ms | {outcome.message}"
            )
            ctx.progress(idx + 1, n_cases, f"Completed case {idx + 1}/{n_cases}: {case.id}")

        total_wall_ms = round((time.perf_counter() - t_run_start) * 1000.0, 2)
        speed_stats = compute_speed_summary(
            ttfts_ms=all_ttfts_ms,
            latencies_ms=all_latencies_ms,
            decode_tps_list=all_decode_tps,
            prefill_tps_list=all_prefill_tps,
        )

        # Populate TTFT metrics
        result.set_metric("ttft_ms", speed_stats["ttft_mean_ms"], unit="ms", better="lower")
        result.set_metric("ttft_mean_ms", speed_stats["ttft_mean_ms"], unit="ms", better="lower")
        result.set_metric("ttft_min_ms", speed_stats["ttft_min_ms"], unit="ms", better="lower")
        result.set_metric("ttft_max_ms", speed_stats["ttft_max_ms"], unit="ms", better="lower")
        result.set_metric("ttft_p50_ms", speed_stats["ttft_p50_ms"], unit="ms", better="lower")
        result.set_metric("ttft_p95_ms", speed_stats["ttft_p95_ms"], unit="ms", better="lower")
        result.set_metric("ttft_p99_ms", speed_stats["ttft_p99_ms"], unit="ms", better="lower")

        # Populate Latency percentiles & summary metrics
        result.set_metric("latency_mean_ms", speed_stats["latency_mean_ms"], unit="ms", better="lower")
        result.set_metric("latency_min_ms", speed_stats["latency_min_ms"], unit="ms", better="lower")
        result.set_metric("latency_max_ms", speed_stats["latency_max_ms"], unit="ms", better="lower")
        result.set_metric("latency_p50_ms", speed_stats["latency_p50_ms"], unit="ms", better="lower")
        result.set_metric("latency_p95_ms", speed_stats["latency_p95_ms"], unit="ms", better="lower")
        result.set_metric("latency_p99_ms", speed_stats["latency_p99_ms"], unit="ms", better="lower")
        result.set_metric("p50_ms", speed_stats["latency_p50_ms"], unit="ms", better="lower")
        result.set_metric("p95_ms", speed_stats["latency_p95_ms"], unit="ms", better="lower")
        result.set_metric("p99_ms", speed_stats["latency_p99_ms"], unit="ms", better="lower")
        result.set_metric("total_wall_ms", total_wall_ms, unit="ms", better="lower")

        # Populate Tokens/sec throughput metrics
        aggregate_tps = (
            round(total_completion_tokens / (total_decode_ms / 1000.0), 2)
            if total_decode_ms > 0
            else 0.0
        )
        result.set_metric("tps", speed_stats["decode_tps_mean"], unit="tokens/s", better="higher")
        result.set_metric("tokens_per_sec", speed_stats["decode_tps_mean"], unit="tokens/s", better="higher")
        result.set_metric("decode_tps_mean", speed_stats["decode_tps_mean"], unit="tokens/s", better="higher")
        result.set_metric("decode_tps_p50", speed_stats["decode_tps_p50"], unit="tokens/s", better="higher")
        result.set_metric("decode_tps_p95", speed_stats["decode_tps_p95"], unit="tokens/s", better="higher")
        result.set_metric("prefill_tps_mean", speed_stats["prefill_tps_mean"], unit="tokens/s", better="higher")
        result.set_metric("aggregate_tps", aggregate_tps, unit="tokens/s", better="higher")

        # Token usage & capability pass rate metrics
        total_tokens = total_prompt_tokens + total_completion_tokens
        result.set_metric("prompt_tokens", total_prompt_tokens, unit="tokens", better="lower")
        result.set_metric("completion_tokens", total_completion_tokens, unit="tokens", better="higher")
        result.set_metric("total_tokens", total_tokens, unit="tokens", better="lower")

        passed_cases = sum(1 for c in result.cases if c.status == "pass")
        total_cases = len(result.cases)
        pass_rate = round(passed_cases / total_cases * 100.0, 2) if total_cases > 0 else 0.0
        result.set_metric("cases_passed", passed_cases, unit="cases", better="higher")
        result.set_metric("cases_total", total_cases, unit="cases", better="higher")
        result.set_metric("pass_rate", pass_rate, unit="%", better="higher")

        # Optional LLM Judge aggregate metrics
        if judge_client:
            mean_judge_score = (
                round(float(statistics.mean(judge_scores)), 2) if judge_scores else 0.0
            )
            judge_pass_rate = (
                round(sum(1 for p in judge_passes if p) / len(judge_passes) * 100.0, 2)
                if judge_passes
                else 0.0
            )
            result.set_metric("judge_score_mean", mean_judge_score, unit="score", better="higher")
            result.set_metric("judge_pass_rate", judge_pass_rate, unit="%", better="higher")
            result.set_metric("judge_cases_evaluated", len(judge_scores), unit="cases", better="higher")

        # Overall status
        result.status = "fail" if passed_cases < total_cases else "pass"

        judge_summary_part = ""
        if judge_client and "judge_score_mean" in result.metrics:
            judge_summary_part = (
                f", judge={result.metrics['judge_score_mean']['value']}/5.0"
            )

        result.summary = (
            f"LLM benchmark ({model}): {passed_cases}/{total_cases} cases passed ({pass_rate}%), "
            f"TTFT={speed_stats['ttft_mean_ms']}ms, "
            f"TPS={speed_stats['decode_tps_mean']} tok/s, "
            f"p95={speed_stats['latency_p95_ms']}ms"
            f"{judge_summary_part}"
        )

        # Write artifacts
        report_text = format_llm_report(
            model=model,
            base_url=client.endpoint_url,
            metrics=result.metrics,
            cases=[c.to_dict() for c in result.cases],
            judge_info=judge_info,
        )
        report_file = ctx.artifacts_dir / "llm_report.txt"
        report_file.write_text(report_text, encoding="utf-8")
        ctx.add_artifact("llm_report.txt", report_file, mime="text/plain")

        detailed_json = {
            "tester": self.name,
            "model": model,
            "endpoint_url": client.endpoint_url,
            "judge": judge_info,
            "status": result.status,
            "summary": result.summary,
            "metrics": result.metrics,
            "suites_executed": suites_to_run,
            "cases": [c.to_dict() for c in result.cases],
        }
        json_file = ctx.artifacts_dir / "llm_results.json"
        json_file.write_text(json.dumps(detailed_json, indent=2), encoding="utf-8")
        ctx.add_artifact("llm_results.json", json_file, mime="application/json")

        for art in ctx.artifacts:
            if art not in result.artifacts:
                result.artifacts.append(art)

        ctx.log(f"LLM test finished with status: {result.status.upper()}")
        return result
