"""World-context provider for the smart_room plugin.

Provides ambient room context to Marvi sessions — injected at session start
and available to the voice instant lane's prompt suffix. The runtime NEVER
writes memory; context is read-only ambient awareness.

Per v0.3 §B:
1. On-demand: smart_room_state tool returns full snapshot
2. Ambient session context: one compact line via build_context_line()
3. Subconscious surface: meaningful transitions via stage-1 fetcher
4. Memory: runtime NEVER writes memory — subconscious proposes
5. Activity feed: transitions append to activity.jsonl with source: "world"
"""

from __future__ import annotations

import logging
from typing import Optional

from plugins.smart_room.bridge import read_state_snapshot, build_context_line

logger = logging.getLogger(__name__)


def get_context_line() -> Optional[str]:
    """Return a compact room context line for session priming.

    Called by the context provider registered in __init__.py.
    Returns None when no state is available (plugin not configured yet).
    """
    try:
        return build_context_line()
    except Exception as e:
        logger.debug("smart_room context provider failed: %s", e)
        return None
