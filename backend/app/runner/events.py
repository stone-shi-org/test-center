"""In-memory event broadcaster for live SSE and WebSocket streaming."""

import asyncio
from typing import Any, AsyncIterator


class RunEventBroadcaster:
    """Manages real-time log and progress event distribution for runs."""

    def __init__(self) -> None:
        self._subscribers: dict[int, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def register(self, run_id: int) -> asyncio.Queue:
        """Register a subscriber queue for a given run."""
        async with self._lock:
            if run_id not in self._subscribers:
                self._subscribers[run_id] = set()
            q: asyncio.Queue = asyncio.Queue(maxsize=1000)
            self._subscribers[run_id].add(q)
            return q

    async def unregister(self, run_id: int, queue: asyncio.Queue) -> None:
        """Unregister a subscriber queue."""
        async with self._lock:
            if run_id in self._subscribers:
                self._subscribers[run_id].discard(queue)
                if not self._subscribers[run_id]:
                    del self._subscribers[run_id]

    def publish(self, run_id: int, event: dict[str, Any]) -> None:
        """Publish an event to all active listeners for run_id (thread-safe)."""
        queues = self._subscribers.get(run_id, set())
        for q in list(queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def subscribe(self, run_id: int) -> AsyncIterator[dict[str, Any]]:
        """Async iterator yielding events for run_id until stream completes."""
        queue = await self.register(run_id)
        try:
            while True:
                event = await queue.get()
                yield event
                event_type = event.get("type")
                if event_type in ("end", "completed"):
                    break
        finally:
            await self.unregister(run_id, queue)


# Global broadcaster singleton
event_broadcaster = RunEventBroadcaster()


def get_broadcaster() -> RunEventBroadcaster:
    return event_broadcaster
