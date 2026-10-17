"""OpenAI-compatible LLM chat benchmark and capability tester package."""

from backend.app.testers.llm.cases import (
    BUILTIN_CAPABILITY_CASES,
    CapabilityCase,
    EvaluationOutcome,
    build_judge_messages,
    coerce_judge_score,
    evaluate_coding,
    evaluate_constraint_following,
    evaluate_reasoning,
    evaluate_summarization,
    evaluate_tool_call_json,
    extract_json_object,
    extract_python_code,
    get_builtin_cases,
    parse_judge_response,
)
from backend.app.testers.llm.client import (
    ChatClientError,
    ChatStreamResult,
    OpenAIChatClient,
    strip_think_blocks,
)
from backend.app.testers.llm.metrics import (
    compute_percentiles,
    compute_speed_summary,
    estimate_tokens,
)
from backend.app.testers.llm.report import format_llm_report
from backend.app.testers.llm.tester import LLMTester

__all__ = [
    "LLMTester",
    "OpenAIChatClient",
    "ChatClientError",
    "ChatStreamResult",
    "strip_think_blocks",
    "CapabilityCase",
    "EvaluationOutcome",
    "BUILTIN_CAPABILITY_CASES",
    "get_builtin_cases",
    "evaluate_coding",
    "evaluate_tool_call_json",
    "evaluate_constraint_following",
    "evaluate_reasoning",
    "evaluate_summarization",
    "extract_json_object",
    "extract_python_code",
    "coerce_judge_score",
    "build_judge_messages",
    "parse_judge_response",
    "estimate_tokens",
    "compute_percentiles",
    "compute_speed_summary",
    "format_llm_report",
]
