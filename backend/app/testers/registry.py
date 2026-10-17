"""Registry for discovering and managing tester plugins."""

from typing import Any
from backend.app.testers.base import TesterPlugin, TesterError
from backend.app.testers.dummy import DummyTester
from backend.app.testers.embedding import EmbeddingTester
from backend.app.testers.llm import LLMTester
from backend.app.testers.live_stt import LiveSTTTester


class TesterNotFoundError(TesterError):
    """Raised when a requested tester plugin is not found in the registry."""
    pass


class TesterRegistry:
    """Registry maintaining available tester plugin instances."""

    def __init__(self) -> None:
        self._testers: dict[str, TesterPlugin] = {}
        self._aliases: dict[str, str] = {}

    def register(self, tester: TesterPlugin, aliases: list[str] | None = None) -> None:
        """Register a tester plugin and optional lookup aliases."""
        if not tester.name:
            raise ValueError("Tester plugin must define a non-empty 'name'")
        self._testers[tester.name] = tester
        for alias in (aliases or []):
            if alias and alias != tester.name:
                self._aliases[alias] = tester.name

    def get(self, name: str) -> TesterPlugin:
        """Retrieve a tester plugin by name or alias."""
        if name in self._testers:
            return self._testers[name]
        canonical = self._aliases.get(name)
        if canonical and canonical in self._testers:
            return self._testers[canonical]
        raise TesterNotFoundError(f"Tester plugin '{name}' not found. Available: {list(self._testers.keys())}")

    def list_all(self) -> list[TesterPlugin]:
        """List all registered tester plugins."""
        return list(self._testers.values())

    def get_info_list(self) -> list[dict[str, Any]]:
        """Return metadata dictionaries for all registered plugins."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "suites": t.suites,
                "config_schema": t.config_schema,
            }
            for t in self._testers.values()
        ]


# Global registry singleton
tester_registry = TesterRegistry()

# Register built-in plugins
tester_registry.register(DummyTester())
tester_registry.register(EmbeddingTester())
tester_registry.register(LLMTester())
tester_registry.register(
    LiveSTTTester(),
    aliases=[
        "live_stt",
        "live-stt-http",
        "live_stt_http",
        "live-stt-admin",
        "live_stt_admin",
        "live-stt-contract",
        "live_stt_contract",
    ],
)


def get_tester(name: str) -> TesterPlugin:
    """Helper to get a registered tester plugin."""
    return tester_registry.get(name)


def list_testers() -> list[TesterPlugin]:
    """Helper to list all registered tester plugins."""
    return tester_registry.list_all()


def register_tester(tester: TesterPlugin, aliases: list[str] | None = None) -> None:
    """Helper to register a tester plugin."""
    tester_registry.register(tester, aliases=aliases)
