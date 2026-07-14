"""Subprocess lifecycle manager for the smart_room runtime.

The runtime runs as ONE child process spawned and supervised by the plugin
when the gateway starts. Auto-restart with backoff. A runtime crash never
touches the gateway.

Mirrors google_meet/process_manager.py but with persistent (always-on)
semantics rather than per-session.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home


def _root() -> Path:
    return Path(get_hermes_home()) / "smart_room"


def _active_file() -> Path:
    return _root() / ".runtime.json"


def _read_active() -> Optional[Dict[str, Any]]:
    p = _active_file()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_active(data: Dict[str, Any]) -> None:
    p = _active_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def _clear_active() -> None:
    try:
        _active_file().unlink()
    except FileNotFoundError:
        pass


def _pid_alive(pid: int) -> bool:
    from gateway.status import _pid_exists
    return _pid_exists(pid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Spawn the smart_room runtime subprocess.

    If already running, do nothing. The runtime is always-on — it starts
    when the gateway starts and restarts on crash with backoff.
    """
    existing = _read_active()
    if existing and _pid_alive(int(existing.get("pid", 0))):
        return {"ok": True, "already_running": True, **existing}

    _root().mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if config:
        env["SMART_ROOM_CONFIG"] = json.dumps(config)

    log_path = _root() / "runtime.log"
    log_fh = open(log_path, "ab", buffering=0)

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "plugins.smart_room.runtime.app"],
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()

    record = {
        "pid": proc.pid,
        "started_at": time.time(),
        "log_path": str(log_path),
        "restart_count": 0,
    }
    _write_active(record)
    return {"ok": True, **record}


def stop(*, reason: str = "requested") -> Dict[str, Any]:
    """Stop the runtime cleanly."""
    active = _read_active()
    if not active:
        return {"ok": False, "reason": "runtime not running"}

    pid = int(active.get("pid", 0))
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.5)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    _clear_active()
    return {"ok": True, "reason": reason}


def status() -> Dict[str, Any]:
    """Check runtime liveness."""
    active = _read_active()
    if not active:
        return {"ok": False, "reason": "runtime not running"}
    pid = int(active.get("pid", 0))
    alive = _pid_alive(pid) if pid else False
    return {
        "ok": True,
        "alive": alive,
        "pid": pid,
        "started_at": active.get("started_at"),
        "restart_count": active.get("restart_count", 0),
        "log_path": active.get("log_path"),
    }


def restart() -> Dict[str, Any]:
    """Restart the runtime (stop + start)."""
    stop(reason="restart requested")
    time.sleep(1)
    return start()


def health_check() -> Dict[str, Any]:
    """Full health check — runtime + devices + MQTT."""
    s = status()
    if not s.get("alive"):
        return {"ok": False, "runtime": "down", "detail": s}

    # Try to reach the runtime via RPC
    try:
        from plugins.smart_room.bridge import call_runtime
        result = call_runtime("get_health", {})
        return {"ok": True, "runtime": "up", **result}
    except RuntimeError as e:
        return {"ok": False, "runtime": "unreachable", "error": str(e), **s}
