"""Tester plugin contracts, registry, and implementations."""

from backend.app.testers.base import (
    TesterPlugin,
    TestContext,
    TesterResult,
    CaseResultData,
    ArtifactData,
    TesterError,
    TesterCancelledError,
    TesterConfigError,
)
from backend.app.testers.schema import validate_schema, apply_schema_defaults, SchemaValidationError
from backend.app.testers.registry import (
    tester_registry,
    get_tester,
    list_testers,
    register_tester,
    TesterNotFoundError,
)
from backend.app.testers.dummy import DummyTester
from backend.app.testers.embedding import EmbeddingTester
from backend.app.testers.llm import LLMTester
from backend.app.testers.llm_coding import LLMCodingTester
from backend.app.testers.live_stt import LiveSTTTester
from backend.app.testers.starters import STARTER_TEMPLATES, get_starter_templates

__all__ = [
    "TesterPlugin",
    "TestContext",
    "TesterResult",
    "CaseResultData",
    "ArtifactData",
    "TesterError",
    "TesterCancelledError",
    "TesterConfigError",
    "SchemaValidationError",
    "validate_schema",
    "apply_schema_defaults",
    "tester_registry",
    "get_tester",
    "list_testers",
    "register_tester",
    "TesterNotFoundError",
    "DummyTester",
    "EmbeddingTester",
    "LLMTester",
    "LLMCodingTester",
    "LiveSTTTester",
    "STARTER_TEMPLATES",
    "get_starter_templates",
]
