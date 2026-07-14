"""Internal event bus for the smart_room runtime.

Lightweight pub/sub — components publish events, the automation engine
and other subscribers receive them. No external dependency.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventBus:
    """Simple synchronous event bus."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, data: Dict[str, Any] | None = None) -> None:
        data = data or {}
        for handler in self._subscribers.get(event_type, []):
            try:
                handler({"type": event_type, **data})
            except Exception as e:
                logger.error("Event handler %s failed for %s: %s", handler, event_type, e)

    def publish_all(self, event: Dict[str, Any]) -> None:
        """Publish a pre-built event dict to all subscribers of its type."""
        event_type = event.get("type", "")
        for handler in self._subscribers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error("Event handler failed for %s: %s", event_type, e)
