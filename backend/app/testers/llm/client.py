"""OpenAI-compatible chat completion HTTP client (streaming and non-streaming)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any
import httpx

from backend.app.testers.llm.metrics import estimate_tokens


class ChatClientError(Exception):
    """Exception raised for errors during OpenAI-compatible chat API communication."""
    pass


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks if present, falling back to inner text if empty."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned
    # If everything was inside <think> tags, strip only the tags themselves
    inner = re.sub(r"</?think>", "", text).strip()
    return inner


@dataclass
class ChatStreamResult:
    """Telemetry and output from a single streaming chat completion request."""
    content: str
    raw_content: str
    reasoning: str
    ttft_ms: float
    latency_ms: float
    decode_duration_ms: float
    tokens_per_sec: float
    prefill_tps: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    chunk_count: int


class OpenAIChatClient:
    """Client for calling OpenAI-compatible /chat/completions and /models endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.raw_base_url = base_url.strip()
        self.api_key = api_key.strip() if api_key else None
        self.timeout = timeout
        self.transport = transport
        self.default_headers = default_headers or {}
        self.endpoint_url = self._normalize_endpoint_url(self.raw_base_url)
        self.models_url = self._normalize_models_url(self.raw_base_url)

    @staticmethod
    def _normalize_endpoint_url(base_url: str) -> str:
        """Ensure endpoint URL targets /chat/completions."""
        url = base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return f"{url}/chat/completions"

    @staticmethod
    def _normalize_models_url(base_url: str) -> str:
        """Ensure models URL targets /models."""
        url = base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]
        if url.endswith("/models"):
            return url
        return f"{url}/models"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.default_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def detect_model(self) -> str | None:
        """Query /models to auto-detect the first loaded model ID, or return None on failure."""
        headers = self._build_headers()
        try:
            with httpx.Client(timeout=min(self.timeout, 10.0), transport=self.transport) as client:
                resp = client.get(self.models_url, headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                data = body.get("data")
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict) and first.get("id"):
                        return str(first["id"])
        except Exception:
            pass
        return None

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        reasoning_effort: str | None = None,
        effort: str | None = None,
    ) -> ChatStreamResult:
        """Execute a streaming chat completion request and collect speed telemetry."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stop:
            payload["stop"] = stop
        eff = reasoning_effort if reasoning_effort is not None else effort
        if eff is not None and str(eff).strip():
            payload["reasoning_effort"] = str(eff).strip()

        headers = self._build_headers()
        prompt_text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))

        t0 = time.perf_counter()
        first_token_time: float | None = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_raw: dict[str, Any] | None = None
        chunk_count = 0
        saw_sse = False
        fallback_lines: list[str] = []

        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                with client.stream(
                    "POST",
                    self.endpoint_url,
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        err_bytes = resp.read()
                        err_text = err_bytes.decode("utf-8", errors="replace")
                        raise ChatClientError(
                            f"Chat request to {self.endpoint_url} failed with status {resp.status_code}: {err_text}"
                        )

                    for raw_line in resp.iter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            saw_sse = True
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if not isinstance(chunk, dict):
                                continue

                            if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
                                usage_raw = chunk["usage"]

                            choices = chunk.get("choices")
                            if isinstance(choices, list) and len(choices) > 0:
                                choice = choices[0] if isinstance(choices[0], dict) else {}
                                delta = choice.get("delta") or choice.get("message") or {}
                                if not isinstance(delta, dict):
                                    delta = {}
                                r_piece = delta.get("reasoning") or delta.get("reasoning_content") or ""
                                c_piece = delta.get("content") or ""

                                if r_piece or c_piece:
                                    now = time.perf_counter()
                                    if first_token_time is None:
                                        first_token_time = now
                                    chunk_count += 1

                                if r_piece:
                                    reasoning_parts.append(str(r_piece))
                                if c_piece:
                                    content_parts.append(str(c_piece))
                        else:
                            fallback_lines.append(line)
        except httpx.RequestError as e:
            raise ChatClientError(f"HTTP request to {self.endpoint_url} failed: {e}") from e

        t_end = time.perf_counter()

        # Handle fallback if server returned non-streaming JSON despite stream=True
        if not saw_sse and fallback_lines:
            raw_body = "\n".join(fallback_lines)
            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError as e:
                raise ChatClientError(
                    f"Failed to parse chat response from {self.endpoint_url}: {e}"
                ) from e

            if isinstance(body, dict):
                if isinstance(body.get("usage"), dict):
                    usage_raw = body["usage"]
                choices = body.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    c_text = msg.get("content") or ""
                    r_text = msg.get("reasoning") or msg.get("reasoning_content") or ""
                    if c_text:
                        content_parts.append(str(c_text))
                    if r_text:
                        reasoning_parts.append(str(r_text))
                    if c_text or r_text:
                        first_token_time = t_end
                        chunk_count = 1

        raw_content = "".join(content_parts)
        raw_reasoning = "".join(reasoning_parts)

        if raw_content:
            cleaned_content = strip_think_blocks(raw_content)
        elif raw_reasoning:
            cleaned_content = strip_think_blocks(raw_reasoning)
        else:
            cleaned_content = ""

        combined_generated = raw_reasoning + raw_content
        est_prompt = estimate_tokens(prompt_text)
        est_completion = estimate_tokens(combined_generated)

        prompt_tokens = est_prompt
        completion_tokens = est_completion

        if isinstance(usage_raw, dict):
            api_prompt = int(usage_raw.get("prompt_tokens", 0) or 0)
            api_completion = int(usage_raw.get("completion_tokens", 0) or 0)
            if api_prompt > 0:
                prompt_tokens = api_prompt
            if api_completion > 0:
                completion_tokens = api_completion

        total_tokens = prompt_tokens + completion_tokens

        latency_ms = max(0.01, (t_end - t0) * 1000.0)
        if first_token_time is not None:
            ttft_ms = max(0.01, min((first_token_time - t0) * 1000.0, latency_ms))
            decode_duration_ms = max(0.0, (t_end - first_token_time) * 1000.0)
        else:
            ttft_ms = latency_ms
            decode_duration_ms = 0.0

        if decode_duration_ms > 0.05:
            tokens_per_sec = completion_tokens / (decode_duration_ms / 1000.0)
        else:
            tokens_per_sec = completion_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

        prefill_tps = prompt_tokens / (ttft_ms / 1000.0) if ttft_ms > 0 else 0.0

        return ChatStreamResult(
            content=cleaned_content,
            raw_content=raw_content,
            reasoning=raw_reasoning,
            ttft_ms=round(ttft_ms, 2),
            latency_ms=round(latency_ms, 2),
            decode_duration_ms=round(decode_duration_ms, 2),
            tokens_per_sec=round(tokens_per_sec, 2),
            prefill_tps=round(prefill_tps, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            chunk_count=chunk_count,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        effort: str | None = None,
    ) -> tuple[str, dict[str, int], float]:
        """Execute a non-streaming chat completion request (used e.g. for LLM Judge calls).

        Returns:
            Tuple of (content_text, usage_dict, elapsed_ms)
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format
        eff = reasoning_effort if reasoning_effort is not None else effort
        if eff is not None and str(eff).strip():
            payload["reasoning_effort"] = str(eff).strip()

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
            raise ChatClientError(f"HTTP request to {self.endpoint_url} failed: {e}") from e

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise ChatClientError(
                f"Chat request to {self.endpoint_url} failed with status {resp.status_code}: {resp.text}"
            )

        try:
            body = resp.json()
        except Exception as e:
            raise ChatClientError(f"Failed to parse JSON response from {self.endpoint_url}: {e}") from e

        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise ChatClientError(f"Invalid chat response: missing or empty 'choices' field in {body}")

        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message") or first_choice.get("delta") or {}
        content = str(message.get("content") or "").strip()
        content = strip_think_blocks(content)

        usage_raw = body.get("usage") or {}
        prompt_tokens = int(usage_raw.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage_raw.get("completion_tokens", 0) or 0)
        total_tokens = int(usage_raw.get("total_tokens", 0) or (prompt_tokens + completion_tokens))

        return content, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }, round(elapsed_ms, 2)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
        tool_choice: str | dict[str, Any] = "auto",
        reasoning_effort: str | None = None,
        effort: str | None = None,
    ) -> dict[str, Any]:
        """Execute a chat completion turn with OpenAI tool definitions enabled.

        Returns a dict with:
            - content: str
            - tool_calls: list[dict[str, Any]]
            - message: dict[str, Any] (assistant message suitable for appending to transcript)
            - usage: dict[str, int]
            - latency_ms: float
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "stream": False,
        }
        eff = reasoning_effort if reasoning_effort is not None else effort
        if eff is not None and str(eff).strip():
            payload["reasoning_effort"] = str(eff).strip()

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
            raise ChatClientError(f"HTTP request to {self.endpoint_url} failed: {e}") from e

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise ChatClientError(
                f"Chat tool request to {self.endpoint_url} failed with status {resp.status_code}: {resp.text}"
            )

        try:
            body = resp.json()
        except Exception as e:
            raise ChatClientError(f"Failed to parse JSON response from {self.endpoint_url}: {e}") from e

        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise ChatClientError(f"Invalid chat response: missing or empty 'choices' field in {body}")

        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message") or first_choice.get("delta") or {}
        raw_content = str(message.get("content") or "").strip()
        content = strip_think_blocks(raw_content)
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[dict[str, Any]] = []
        if isinstance(raw_tool_calls, list):
            for idx, tc in enumerate(raw_tool_calls):
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    tool_calls.append(
                        {
                            "id": str(tc.get("id") or f"call_{idx + 1}"),
                            "type": "function",
                            "function": {
                                "name": str(fn.get("name") or ""),
                                "arguments": fn.get("arguments")
                                if isinstance(fn.get("arguments"), str)
                                else json.dumps(fn.get("arguments") or {}),
                            },
                        }
                    )

        usage_raw = body.get("usage") or {}
        prompt_tokens = int(usage_raw.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage_raw.get("completion_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_text = " ".join(str(m.get("content") or "") for m in messages if isinstance(m, dict))
            prompt_tokens = estimate_tokens(prompt_text)
        if completion_tokens <= 0:
            comp_text = content + " " + json.dumps(tool_calls)
            completion_tokens = estimate_tokens(comp_text)
        total_tokens = int(usage_raw.get("total_tokens", 0) or (prompt_tokens + completion_tokens))

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        return {
            "content": content,
            "tool_calls": tool_calls,
            "message": assistant_msg,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "latency_ms": round(elapsed_ms, 2),
        }
