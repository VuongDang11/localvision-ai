import threading
from collections import deque


class EventStore:
    """Keep the most recent events in memory. Week 2 replaces this with SQLite."""

    def __init__(self, max_events: int = 200) -> None:
        # deque with maxlen drops the oldest event automatically when full.
        self._events = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._next_event_id = 1

    def add(self, event: dict) -> dict:
        with self._lock:
            stored_event = {"event_id": self._next_event_id, **event}
            self._next_event_id += 1
            self._events.append(stored_event)
            return stored_event

    def list_recent(self, limit: int = 50) -> list[dict]:
        """Return up to `limit` events, newest first."""
        with self._lock:
            return list(reversed(self._events))[:limit]