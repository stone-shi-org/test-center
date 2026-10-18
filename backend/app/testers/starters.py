"""Built-in starter templates per tester plugin.

Starter templates provide ready-to-use configurations for each tester without
being tied to a specific target (`target_id` is omitted/null). Users can load
starters directly on the Run Trigger page or copy them into `test_definitions`
from the Templates page.
"""

from copy import deepcopy
from typing import Any

STARTER_TEMPLATES: list[dict[str, Any]] = [
    # live-stt
    {
        "tester": "live-stt",
        "name": "HTTP smoke",
        "description": "Quick HTTP health and JSON transcription smoke check.",
        "suites": ["admin", "transcriptions_json"],
        "config": {},
    },
    {
        "tester": "live-stt",
        "name": "HTTP full contract",
        "description": "Default 7 HTTP contract suites covering JSON, SSE, diarization, TTS, admin, error handling, and concurrency.",
        "suites": [
            "transcriptions_json",
            "transcriptions_sse",
            "diarization",
            "tts",
            "admin",
            "error_handling",
            "concurrency",
        ],
        "config": {},
    },
    {
        "tester": "live-stt",
        "name": "gRPC streaming",
        "description": "gRPC streaming contract verification suite.",
        "suites": ["grpc_streaming"],
        "config": {},
    },
    {
        "tester": "live-stt",
        "name": "Quick accuracy (WER)",
        "description": "Fast batch WER evaluation on up to 3 meetings (max 60s each).",
        "suites": ["accuracy_wer"],
        "config": {
            "max_meetings": 3,
            "max_audio_duration_sec": 60,
            "wer_threshold": 0.3,
        },
    },
    {
        "tester": "live-stt",
        "name": "Streaming accuracy (gRPC)",
        "description": "Streaming accuracy evaluation over gRPC on up to 3 meetings (max 60s each).",
        "suites": ["accuracy_streaming"],
        "config": {
            "streaming_protocol": "grpc",
            "max_meetings": 3,
            "max_audio_duration_sec": 60,
        },
    },
    {
        "tester": "live-stt",
        "name": "gRPC long call",
        "description": "Long-duration gRPC streaming stability and memory growth test.",
        "suites": ["grpc_long_call"],
        "config": {
            "include_long_call": True,
        },
    },
    # llm
    {
        "tester": "llm",
        "name": "Quick check",
        "description": "Fast deterministic check covering coding and output constraint suites.",
        "suites": ["coding", "constraints"],
        "config": {},
    },
    {
        "tester": "llm",
        "name": "Full capability",
        "description": "All 5 LLM capability suites: coding, tool calling, constraints, reasoning, and summarization.",
        "suites": [
            "coding",
            "tool_call",
            "constraints",
            "reasoning",
            "summarization",
        ],
        "config": {},
    },
    {
        "tester": "llm",
        "name": "Reasoning, high effort",
        "description": "Reasoning benchmark suite with high reasoning effort and 8192 max tokens.",
        "suites": ["reasoning"],
        "config": {
            "effort": "high",
            "max_tokens": 8192,
        },
    },
    # embedding
    {
        "tester": "embedding",
        "name": "Quick latency",
        "description": "Quick single-input embedding latency and dimension verification.",
        "suites": ["latency"],
        "config": {},
    },
    {
        "tester": "embedding",
        "name": "Retrieval quality",
        "description": "Semantic retrieval ranking and deterministic consistency suites.",
        "suites": ["retrieval", "consistency"],
        "config": {},
    },
    {
        "tester": "embedding",
        "name": "Full",
        "description": "All embedding suites: latency, batch throughput, consistency, and retrieval.",
        "suites": ["latency", "throughput", "consistency", "retrieval"],
        "config": {},
    },
    # dummy
    {
        "tester": "dummy",
        "name": "Smoke",
        "description": "Fast built-in dummy smoke suite for verifying runner pipeline.",
        "suites": ["smoke"],
        "config": {},
    },
]


def get_starter_templates(tester: str | None = None) -> list[dict[str, Any]]:
    """Return a deep copy of starter templates, optionally filtered by tester."""
    items = [
        deepcopy(item)
        for item in STARTER_TEMPLATES
        if tester is None or item["tester"] == tester
    ]
    return items
