"""OpenAI-compatible embedding HTTP client."""

from __future__ import annotations

import time
from typing import Any
import httpx
import numpy as np


class EmbeddingClientError(Exception):
    """Exception raised for errors during embedding API communication."""
    pass


class OpenAIEmbeddingClient:
    """Client for calling OpenAI-compatible embedding endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.raw_base_url = base_url.strip()
        self.api_key = api_key.strip() if api_key else None
        self.timeout = timeout
        self.transport = transport
        self.default_headers = default_headers or {}
        self.endpoint_url = self._normalize_endpoint_url(self.raw_base_url)

    @staticmethod
    def _normalize_endpoint_url(base_url: str) -> str:
        """Ensure endpoint URL correctly targets /embeddings."""
        url = base_url.rstrip("/")
        if url.endswith("/embeddings"):
            return url
        return f"{url}/embeddings"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.default_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None = None,
    ) -> tuple[np.ndarray, dict[str, int], float]:
        """Embed a list of text strings.

        Args:
            texts: List of strings to embed.
            model: Embedding model name.
            dimensions: Optional dimensionality parameter.

        Returns:
            Tuple of:
            - vectors: 2D numpy array [N, embedding_dim] of float32
            - usage: Dict with 'prompt_tokens' and 'total_tokens'
            - elapsed_ms: Request latency in milliseconds
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32), {"prompt_tokens": 0, "total_tokens": 0}, 0.0

        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        if dimensions is not None and dimensions > 0:
            payload["dimensions"] = dimensions

        headers = self._build_headers()

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                resp = client.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers,
                )
        except httpx.RequestError as e:
            raise EmbeddingClientError(f"HTTP request to {self.endpoint_url} failed: {e}") from e

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise EmbeddingClientError(
                f"Embedding request to {self.endpoint_url} failed with status {resp.status_code}: {resp.text}"
            )

        try:
            body = resp.json()
        except Exception as e:
            raise EmbeddingClientError(f"Failed to parse JSON response from {self.endpoint_url}: {e}") from e

        data_items = body.get("data")
        if not isinstance(data_items, list) or len(data_items) == 0:
            raise EmbeddingClientError(f"Invalid embedding response: missing or empty 'data' field in {body}")

        # Sort items by index to guarantee strict ordering matching input texts
        try:
            sorted_items = sorted(data_items, key=lambda item: item.get("index", 0))
            vectors = np.array([item["embedding"] for item in sorted_items], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as e:
            raise EmbeddingClientError(f"Malformed embedding item in response data: {e}") from e

        if vectors.shape[0] != len(texts):
            raise EmbeddingClientError(
                f"Received {vectors.shape[0]} embeddings, expected {len(texts)} for inputs"
            )

        usage_raw = body.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }

        return vectors, usage, elapsed_ms
