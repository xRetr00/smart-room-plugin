"""Gateway-owned lifecycle and crash supervision for the room runtime."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

_lock = threading.RLock()
_supervisor_stop = threading.Event()
_supervisor_thread: Optional[threading.Thread] = None
_supervisor_config: Dict[str, Any] = {}
_supervisor_home: Optional[Path] = None
_process: Optional[subprocess.Popen] = None
_atexit_registered = False


@contextmanager
def _start_lock():
    """Serialize runtime starts across Desktop, gateway, and test processes."""
    _root().mkdir(parents=True, exist_ok=True)
    with open(_root() / ".start.lock", "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _root() -> Path:
    return (_supervisor_home or Path(get_hermes_home())) / "smart_room"


def _active_file() -> Path:
    return _root() / ".runtime.json"


def _rpc_token_file() -> Path:
    return _root() / ".rpc-token"


def _write_rpc_token(token: str) -> None:
    path = _rpc_token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _read_active() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_active_file().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _write_active(data: Dict[str, Any]) -> None:
    path = _active_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _clear_active() -> None:
    try:
        _active_file().unlink()
    except FileNotFoundError:
        pass


def _clear_rpc_token() -> None:
    try:
        _rpc_token_file().unlink()
    except FileNotFoundError:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    from gateway.status import _pid_exists

    return _pid_exists(pid)


def _rpc_port(config: Dict[str, Any]) -> int:
    return int((config.get("runtime") or {}).get("rpc_port", 17842))


def _call_runtime(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call the runtime under the supervisor's captured profile context."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.smart_room.bridge import call_runtime

    token = set_hermes_home_override(_root().parent)
    try:
        return call_runtime(method, params)
    finally:
        reset_hermes_home_override(token)


def _managed_runtime_alive(active: Optional[Dict[str, Any]] = None) -> bool:
    """Confirm the recorded PID is our authenticated runtime, not a reused PID."""
    active = active or _read_active()
    if not active:
        return False
    pid = int(active.get("pid", 0))
    if not _pid_alive(pid):
        return False
    if (
        _process is not None
        and _process.pid == pid
        and _process.poll() is None
        and time.time() - float(active.get("started_at", 0)) < 5
    ):
        return True
    try:
        return bool(_call_runtime("ping", {}).get("success"))
    except RuntimeError:
        return False


def start(config: Optional[Dict[str, Any]] = None, *, restart_count: int = 0) -> Dict[str, Any]:
    with _start_lock():
        return _start(config, restart_count=restart_count)


def _start(config: Optional[Dict[str, Any]] = None, *, restart_count: int = 0) -> Dict[str, Any]:
    """Start one runtime process, or return the live process already recorded."""
    global _process
    with _lock:
        existing = _read_active()
        if existing and _managed_runtime_alive(existing):
            return {"ok": True, "already_running": True, **existing}
        if existing:
            existing_pid = int(existing.get("pid", 0))
            if (
                _process is not None
                and _process.pid == existing_pid
                and _process.poll() is None
            ):
                try:
                    os.kill(existing_pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                for _ in range(20):
                    if not _pid_alive(existing_pid):
                        break
                    time.sleep(0.1)
            _clear_active()

        cfg = dict(config or _supervisor_config)
        _root().mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # Desktop secret writes update the profile .env without mutating a
        # long-lived gateway process. Resolve this plugin's credentials at
        # spawn time so Apply/restart always gives the child the latest keys.
        from dotenv import dotenv_values

        # Read the captured profile directly. Gateway multiplexing deliberately
        # isolates process-global secrets, so a generic load_env() call may see
        # a different active scope than this supervised child belongs to.
        stored_env = dotenv_values(_root().parent / ".env", encoding="utf-8")
        for name in (
            "SMART_ROOM_MQTT_USERNAME",
            "SMART_ROOM_MQTT_PASSWORD",
            "SMART_ROOM_TUYA_BULB_KEY",
            "SMART_ROOM_TUYA_HE20_KEY",
        ):
            if value := stored_env.get(name):
                env[name] = value
        env["SMART_ROOM_RPC_PORT"] = str(_rpc_port(cfg))
        token = secrets.token_urlsafe(32)
        _write_rpc_token(token)
        env["SMART_ROOM_RPC_TOKEN"] = token
        # Preserve the exact profile selected by the gateway in the child.
        env["MARVI_HOME"] = str(_root().parent)
        log_path = _root() / "runtime.log"
        with open(log_path, "ab", buffering=0) as log_fh:
            _process = subprocess.Popen(
                [sys.executable, "-m", "plugins.smart_room.runtime.app"],
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                close_fds=True,
            )
        record = {
            "pid": _process.pid,
            "started_at": time.time(),
            "log_path": str(log_path),
            "restart_count": restart_count,
            "rpc_port": _rpc_port(cfg),
        }
        _write_active(record)
        return {"ok": True, **record}


def stop(*, reason: str = "requested") -> Dict[str, Any]:
    """Stop the managed runtime with a bounded graceful wait."""
    global _process
    with _lock:
        active = _read_active()
        if not active:
            _process = None
            return {"ok": False, "reason": "runtime not running"}
        pid = int(active.get("pid", 0))
        owned_locally = _process is not None and _process.pid == pid
        authenticated = False
        try:
            authenticated = bool(_call_runtime("shutdown", {}).get("success"))
        except RuntimeError:
            pass
        # Never signal a merely recorded PID unless it is this manager's
        # child or answered the authenticated runtime RPC. PIDs are reusable.
        if _pid_alive(pid) and (owned_locally or authenticated):
            for _ in range(50):
                if not _pid_alive(pid):
                    break
                time.sleep(0.1)
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        _clear_active()
        _clear_rpc_token()
        _process = None
        return {
            "ok": owned_locally or authenticated or not _pid_alive(pid),
            "reason": reason if owned_locally or authenticated else "stale runtime record cleared",
        }


def status() -> Dict[str, Any]:
    active = _read_active()
    if not active:
        return {"ok": False, "alive": False, "reason": "runtime not running"}
    pid = int(active.get("pid", 0))
    return {"ok": True, "alive": _managed_runtime_alive(active), **active}


def restart() -> Dict[str, Any]:
    previous = _read_active() or {}
    stop(reason="restart requested")
    return start(restart_count=int(previous.get("restart_count", 0)) + 1)


def _supervise_loop() -> None:
    failures = 0
    while not _supervisor_stop.is_set():
        current = status()
        if current.get("alive"):
            if time.time() - float(current.get("started_at", 0)) > 60:
                failures = 0
            _supervisor_stop.wait(2)
            continue
        failures += 1
        if _supervisor_stop.wait(min(30, 2 ** min(failures - 1, 5))):
            break
        try:
            start(_supervisor_config, restart_count=failures)
        except Exception:
            continue


def _supervise() -> None:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(_root().parent)
    try:
        _supervise_loop()
    finally:
        reset_hermes_home_override(token)


def start_supervisor(config: Dict[str, Any]) -> Dict[str, Any]:
    """Start the runtime and one gateway-lifetime crash monitor."""
    global _atexit_registered, _supervisor_config, _supervisor_home, _supervisor_thread
    with _lock:
        _supervisor_config = dict(config)
        _supervisor_home = Path(get_hermes_home())
        result = start(_supervisor_config)
        if _supervisor_thread and _supervisor_thread.is_alive():
            return result
        _supervisor_stop.clear()
        _supervisor_thread = threading.Thread(
            target=_supervise, name="smart_room_supervisor", daemon=True
        )
        _supervisor_thread.start()
        if not _atexit_registered:
            atexit.register(stop_supervisor)
            _atexit_registered = True
        return result


def stop_supervisor() -> None:
    global _supervisor_config, _supervisor_home, _supervisor_thread
    _supervisor_stop.set()
    thread = _supervisor_thread
    if thread and thread is not threading.current_thread():
        thread.join(timeout=4)
    _supervisor_thread = None
    stop(reason="gateway stopped")
    _supervisor_config = {}
    _supervisor_home = None


def health_check() -> Dict[str, Any]:
    current = status()
    if not current.get("alive"):
        return {"ok": False, "runtime": "down", "detail": current}
    try:
        return {"ok": True, "runtime": "up", **_call_runtime("get_health", {})}
    except RuntimeError as exc:
        return {"ok": False, "runtime": "unreachable", "error": str(exc), **current}
