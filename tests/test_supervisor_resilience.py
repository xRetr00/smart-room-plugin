"""The crash monitor must not be the thing that crashes.

Recorded from a real failure: the supervisor thread died twice in one day with
an uncaught exception out of `bridge.call_runtime`, and the Gateway hosting it
went on reporting the room as healthy. Nothing restarted the runtime after
that, because the only thing that would have was the thread that had just died.

    CRITICAL uncaught exception in thread smart_room_supervisor
      process_manager.py:354 _supervise_loop -> status()
      process_manager.py:166 _managed_runtime_alive
      bridge.py:97  chunk = sock.recv(4096)

The hole was narrow: `call_runtime` caught `OSError` around `connect` and only
`socket.timeout` around `recv`, so a reset mid-call -- the runtime exiting
while its answer was being read -- escaped as a bare `OSError` past three
handlers written to expect `RuntimeError`.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from plugins.smart_room import bridge, process_manager


class _Reset:
    """A socket that connects, accepts the request, then drops the connection.

    Which is what the runtime looks like from here when it exits mid-call.
    """

    def __init__(self, failure: BaseException) -> None:
        self.failure = failure

    def settimeout(self, _seconds: float) -> None: ...
    def connect(self, _address: tuple[str, int]) -> None: ...
    def sendall(self, _data: bytes) -> None: ...
    def close(self) -> None: ...

    def recv(self, _size: int) -> bytes:
        raise self.failure


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionResetError(10054, "An existing connection was forcibly closed"),
        socket.timeout("timed out"),
        OSError(22, "Invalid argument"),
        BrokenPipeError(32, "Broken pipe"),
    ],
    ids=["reset", "timeout", "oserror", "broken-pipe"],
)
def test_a_dropped_connection_becomes_a_runtime_error(failure, monkeypatch) -> None:
    """Every caller in this package is written to expect `RuntimeError`."""
    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: _Reset(failure))
    monkeypatch.setattr(bridge, "_rpc_token", lambda: "t", raising=False)
    monkeypatch.setattr(bridge, "_rpc_port", lambda: 17842, raising=False)

    with pytest.raises(RuntimeError):
        bridge.call_runtime("ping", {})


def test_the_liveness_check_answers_false_rather_than_raising(monkeypatch) -> None:
    monkeypatch.setattr(
        process_manager, "_read_active", lambda: {"pid": 1, "started_at": 0}
    )
    monkeypatch.setattr(process_manager, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        process_manager,
        "_call_runtime",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError(10054, "reset")),
    )
    # "Cannot confirm" is the same answer as "not alive", and the caller wants
    # a bool. Raising out of a liveness check is how a monitor becomes a crash.
    assert process_manager._managed_runtime_alive() is False


def test_the_supervisor_survives_a_status_that_raises(monkeypatch) -> None:
    """The loop treats an unreadable state as "not alive" and keeps going.

    Without this the thread ends on the first raise and the room is left with
    nothing watching it -- silently, because the Gateway stays up.
    """
    calls: list[int] = []
    restarts: list[int] = []

    def status() -> dict:
        calls.append(1)
        if len(calls) == 1:
            raise OSError(10054, "reset")
        process_manager._supervisor_stop.set()
        return {"alive": True, "started_at": 0}

    monkeypatch.setattr(process_manager, "status", status)
    monkeypatch.setattr(
        process_manager, "start", lambda *_a, **_k: restarts.append(1) or {}
    )
    process_manager._supervisor_stop.clear()
    try:
        finished = threading.Event()

        def run() -> None:
            process_manager._supervise_loop()
            finished.set()

        threading.Thread(target=run, daemon=True).start()
        assert finished.wait(10), "the supervisor died on a raising status()"
    finally:
        process_manager._supervisor_stop.set()

    assert len(calls) >= 2, "it stopped after the first failure"


def test_a_restart_that_raises_does_not_end_the_loop(monkeypatch) -> None:
    seen: list[int] = []

    def status() -> dict:
        seen.append(1)
        if len(seen) > 3:
            process_manager._supervisor_stop.set()
        return {"alive": False, "started_at": 0}

    monkeypatch.setattr(process_manager, "status", status)
    monkeypatch.setattr(
        process_manager,
        "start",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no")),
    )
    process_manager._supervisor_stop.clear()
    try:
        finished = threading.Event()

        def run() -> None:
            process_manager._supervise_loop()
            finished.set()

        threading.Thread(target=run, daemon=True).start()
        assert finished.wait(20), "a failing restart killed the supervisor"
    finally:
        process_manager._supervisor_stop.set()


def test_a_runtime_error_response_still_reaches_the_caller(monkeypatch) -> None:
    """Widening the handler must not swallow the runtime's own errors."""

    class Answering(_Reset):
        def recv(self, _size: int) -> bytes:
            return json.dumps({"error": "device not found"}).encode() + b"\n"

    monkeypatch.setattr(
        socket, "socket", lambda *_a, **_k: Answering(RuntimeError("unused"))
    )
    monkeypatch.setattr(bridge, "_rpc_token", lambda: "t", raising=False)
    monkeypatch.setattr(bridge, "_rpc_port", lambda: 17842, raising=False)

    with pytest.raises(RuntimeError, match="device not found"):
        bridge.call_runtime("ping", {})
