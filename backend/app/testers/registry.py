"""Registry for discovering and managing tester plugins."""

from typing import Any
from backend.app.testers.base import TesterPlugin, TesterError
from backend.app.testers.dummy import DummyTester
from backend.app.testers.embedding import EmbeddingTester
from backend.app.testers.llm import LLMTester


class TesterNotFoundError(TesterError):
    """Raised when a requested tester plugin is not found in the registry."""
    pass


class TesterRegistry:
    """Registry maintaining available tester plugin instances."""

    def __init__(self) -> None:
        self._testers: dict[str, TesterPlugin] = {}

    def register(self, tester: TesterPlugin) -> None:
        """Register a tester plugin."""
        if not tester.name:
            raise ValueError("Tester plugin must define a non-empty 'name'")
        self._testers[tester.name] = tester

    def get(self, name: str) -> TesterPlugin:
        """Retrieve a tester plugin by name."""
        if name not in self._testers:
            raise TesterNotFoundError(f"Tester plugin '{name}' not found. Available: {list(self._testers.keys())}")
        return self._testers[name]

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


def get_tester(name: str) -> TesterPlugin:
    """Helper to get a registered tester plugin."""
    return tester_registry.get(name)


def list_testers() -> list[TesterPlugin]:
    """Helper to list all registered tester plugins."""
    return tester_registry.list_all()


def register_tester(tester: TesterPlugin) -> None:
    """Helper to register a tester plugin."""
    tester_registry.register(tester)
