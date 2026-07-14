"""Bridge: tool handlers → runtime RPC (non-blocking, bounded wait).

Communicates with the runtime child process via a local JSON-RPC protocol
over stdio (request_id + typed acks). Tool handlers must never block the
gateway loop — bounded wait via asyncio/threadpool, timeout returns
structured DEVICE_TIMEOUT failure.

Design mirrors google_meet's file-based communication but uses a JSON-RPC
socket for lower latency (the smart_room runtime is always-on, not per-session).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional
from pathlib import Path

from hermes_constants import get_hermes_home


# Socket path for JSON-RPC communication with the runtime child process.
# On Windows, we use a named pipe; on Linux/macOS, a Unix domain socket.
# For simplicity and cross-platform support, we use a localhost TCP socket.

_DEFAULT_PORT = 17842  # smart_room runtime RPC port
_TIMEOUT_SECONDS = 5.0  # bounded wait for runtime responses


def _state_file() -> Path:
    """Runtime state snapshot file path."""
    return Path(get_hermes_home()) / "smart_room" / "state.json"


def _rpc_port() -> int:
    """Get the RPC port from env or default."""
    return int(os.environ.get("SMART_ROOM_RPC_PORT", _DEFAULT_PORT))


def call_runtime(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a JSON-RPC request to the runtime and return the result.

    Raises RuntimeError if the runtime is not running or times out.
    This function is blocking — tool handlers call it via run_in_threadpool
    or asyncio.to_thread in the gateway to avoid blocking the event loop.
    """
    import socket

    request = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex[:12],
        "method": method,
        "params": params,
    }

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(_TIMEOUT_SECONDS)
        sock.connect(("127.0.0.1", _rpc_port()))
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        raise RuntimeError(
            f"smart_room runtime is not running on port {_rpc_port()}: {e}"
        ) from e

    try:
        data = json.dumps(request).encode("utf-8")
        sock.sendall(data + b"\n")

        # Read response (newline-delimited JSON)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk

        if not buf:
            raise RuntimeError("runtime closed connection without response")

        response = json.loads(buf.decode("utf-8").strip())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result", {})
    except (json.JSONDecodeError, socket.timeout) as e:
        raise RuntimeError(f"runtime communication error: {e}") from e
    finally:
        sock.close()


def read_state_snapshot() -> Optional[Dict[str, Any]]:
    """Read the last known state snapshot from disk (fallback when runtime is down).

    This is the file-based fallback path — tool handlers can use this to
    return stale-but-useful state when the runtime process is restarting.
    """
    p = _state_file()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_context_line() -> Optional[str]:
    """Build a compact one-line room context for session priming.

    Called by the context provider registered in __init__.py.
    Returns None if no state is available (plugin not configured yet).
    """
    state = read_state_snapshot()
    if not state:
        return None

    parts = []
    # Presence
    presence = state.get("presence", {})
    if presence.get("detected"):
        conf = presence.get("confidence", 0)
        reason = presence.get("reason", "ble")
        parts.append(f"Shereef present ({reason}, conf {conf:.2f})")
    else:
        parts.append("Shereef not in room")

    # Location
    loc = state.get("location", {})
    if loc.get("home"):
        parts.append("phone: home")
    elif loc.get("zone"):
        parts.append(f"phone: {loc['zone']}")

    # Light
    light = state.get("light", {})
    if light.get("on"):
        scene = light.get("scene", "custom")
        brightness = light.get("brightness", 0)
        parts.append(f"light {brightness}% ({scene})")
    else:
        parts.append("light off")

    # Mode
    modes = state.get("modes", {})
    active_mode = next(
        (m for m, v in modes.items() if v and m not in ("manual_override", "work_return")),
        None
    )
    if active_mode:
        parts.append(f"mode: {active_mode}")

    if not parts:
        return None

    return "Room: " + ", ".join(parts) + "."
