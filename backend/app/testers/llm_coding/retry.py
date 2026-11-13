"""Retry transient gateway failures around the chat client used by ``llm_coding``.

A short upstream outage (e.g. Omniroute returning 502 ``fetch failed (ECONNREFUSED)`` and then
429 ``all targets were skipped`` while the model server restarts) used to zero every problem that
hit it. ``RetryingChatClient`` retries those with exponential backoff. It does not retry request
timeouts or long gateway timeouts: repeating a 10-minute generation would just burn another 10
minutes, and the result is reported as Truncated Response / Error as before.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from backend.app.testers.llm.client import ChatClientError

# Status codes / messages that mean "the gateway or upstream is briefly unavailable".
TRANSIENT_ERROR_RE = re.compile(
    r"status (?:429|500|502|503|504)\b|\[(?:429|500|502|503|504)\]|ECONNREFUSED|ECONNRESET|"
    r"fetch failed|temporarily unavailable|all targets were skipped|connection refused|"
    r"ConnectError|RemoteProtocolError|Server disconnected",
    re.IGNORECASE,
)
# Never retried, even if the message also matches above (e.g. a 504 for a combo timeout).
NON_RETRYABLE_RE = re.compile(r"timed? ?out|timeout", re.IGNORECASE)

DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_DELAY_S = 30.0
MAX_DELAY_S = 300.0


def is_transient_error(message: str) -> bool:
    return bool(TRANSIENT_ERROR_RE.search(message or "")) and not NON_RETRYABLE_RE.search(message or "")


class UpstreamStreamError(ChatClientError):
    """The gateway sent an error event inside the stream and no content (not a truncated answer)."""


class RetryingChatClient:
    """Wraps a chat client; retries transient failures of ``stream_chat`` / ``chat`` / ``chat_with_tools``.

    A stream that carries only an error event (no content) is turned into ``UpstreamStreamError``
    so it is reported as an Error, not a Truncated Response. All other attributes (``endpoint_url``,
    ``detect_model``, ...) are delegated to the wrapped client.
    """

    def __init__(
        self,
        inner: Any,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay_s: float = DEFAULT_BASE_DELAY_S,
        should_cancel: Callable[[], bool] | None = None,
        log: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._max_retries = max(0, int(max_retries))
        self._base_delay_s = max(0.0, float(base_delay_s))
        self._should_cancel = should_cancel
        self._log = log
        self._sleep = sleep
        self.retries = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            if self._should_cancel and self._should_cancel():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._sleep(min(1.0, remaining))

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(self._inner, method)
        attempt = 0
        while True:
            try:
                res = fn(*args, **kwargs)
                if method == "stream_chat":
                    stream_error = getattr(res, "stream_error", None)
                    if stream_error and not str(getattr(res, "content", "") or ""):
                        raise UpstreamStreamError(f"Upstream error in response stream: {stream_error}")
                return res
            except ChatClientError as exc:
                msg = str(exc)
                if attempt >= self._max_retries or not is_transient_error(msg):
                    raise
                if self._should_cancel and self._should_cancel():
                    raise
                delay = min(MAX_DELAY_S, self._base_delay_s * (2**attempt))
                attempt += 1
                self.retries += 1
                if self._log:
                    self._log(
                        f"Transient gateway error ({msg[:160]}); retry {attempt}/{self._max_retries} in {delay:.0f}s"
                    )
                self._wait(delay)

    def stream_chat(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("stream_chat", *args, **kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("chat", *args, **kwargs)

    def chat_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("chat_with_tools", *args, **kwargs)
