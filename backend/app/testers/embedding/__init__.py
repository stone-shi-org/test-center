"""Embedding tester plugin package."""

from backend.app.testers.embedding.tester import EmbeddingTester
from backend.app.testers.embedding.client import OpenAIEmbeddingClient, EmbeddingClientError
from backend.app.testers.embedding.corpus import (
    BUILTIN_DOCUMENTS,
    BUILTIN_QUERIES,
    get_default_corpus,
    DocumentItem,
    QueryItem,
)

__all__ = [
    "EmbeddingTester",
    "OpenAIEmbeddingClient",
    "EmbeddingClientError",
    "BUILTIN_DOCUMENTS",
    "BUILTIN_QUERIES",
    "get_default_corpus",
    "DocumentItem",
    "QueryItem",
]
