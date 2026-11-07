"""How the session commands reach a session: a session server's
(:mod:`arachne.server`), over its protocol, or, with ``--once``, one held in
the command's own process, answered by the same dispatcher a server
answers with. Either way an answer is the server's result, a JSON object,
and an error answer is raised as :class:`Failed`, carrying the protocol's
error object.

Three functions find the host's servers: :func:`connect_or_start`, the
server a session is asked of, and :func:`sessions` and :func:`stop`, the
servers running. They are the one place the session commands learn where
servers are.
"""

from __future__ import annotations

__all__ = [
    "Client",
    "Failed",
    "Gone",
    "InProcess",
    "Remote",
    "connect_or_start",
    "sessions",
    "stop",
]

import contextlib
import json
import os
import select
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

from ..server import protocol
from ..server.protocol import RpcError

#: How long a server may take to say it is ready, in seconds.
START_S = 120.0

#: How long past a request's deadline its answer is waited for, in seconds:
#: by then the server has answered the deadline's error, or has ended.
ANSWER_MARGIN_S = 5 * protocol.ENDING_S

#: How long a running server is given to answer a ``ping`` or a
#: ``shutdown`` asked by :func:`sessions` and :func:`stop`, in seconds.
ASK_S = 5.0


class Failed(Exception):
    """An error answer: ``error`` is the protocol's error object, ``{code,
    message, data?}``.
    """

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error.get("message"))
        self.error = error

    @property
    def code(self) -> int:
        code: int = self.error["code"]
        return code


class Gone(Exception):
    """The server ended, or stopped answering, before it answered."""


class Client(Protocol):
    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """The result of ``method`` asked with ``params``; raises
        :class:`Failed` for an error answer.
        """
        ...


class InProcess:
    """A session held in this process, answered by the server's own
    dispatcher. ``deadline_s`` is read and checked as a server reads it,
    but no watchdog keeps it: nothing ends this process for a request.
    Whatever the scene prints goes to standard error, as a server's does.
    """

    def __init__(self) -> None:
        from ..server.dispatch import Dispatcher

        self._dispatcher = Dispatcher()

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from ..server.dispatch import deadline, internal_error

        try:
            _, given = deadline(params)
            with contextlib.redirect_stdout(sys.stderr):
                found = self._dispatcher.handle(method, given)
        except RpcError as error:
            raise Failed(protocol.error(None, error)["error"]) from None
        try:
            # The answer as a server would send it.
            result: dict[str, Any] = json.loads(json.dumps(found, allow_nan=False))
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise Failed(protocol.error(None, internal_error(error))["error"]) from None
        return result


class Remote:
    """A connection to a session server: one request at a time, each line
    a message, read from the file descriptor ``fd`` and sent by ``send``.
    """

    def __init__(self, fd: int, send: Callable[[bytes], object]) -> None:
        self._fd = fd
        self._send = send
        self._buffer = b""
        self._ids = 0

    def ready(self, wait_s: float = START_S) -> dict[str, Any]:
        """The server's ready line's params."""
        message = self._message(wait_s)
        if message.get("method") != "ready":
            raise Gone(f"The server's first line is no ready line: {message!r}.")
        params: dict[str, Any] = message.get("params", {})
        major = str(params.get("protocol", "")).split(".")[0]
        if major != protocol.PROTOCOL_VERSION.split(".")[0]:
            raise Gone(
                f"The server speaks protocol {params.get('protocol')!r},"
                f" this client {protocol.PROTOCOL_VERSION!r}."
            )
        return params

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ids += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._ids,
            "method": method,
            "params": params,
        }
        try:
            self._send(protocol.encode(request))
        except OSError as error:
            raise Gone(f"The server took no request: {error}.") from None
        given = params.get("deadline_s")
        wait_s = (protocol.DEADLINE_S if given is None else given) + ANSWER_MARGIN_S
        answer = self._message(wait_s)
        if answer.get("id") != self._ids:
            raise Gone(f"The server answered another request: {answer!r}.")
        if "error" in answer:
            raise Failed(answer["error"])
        result: dict[str, Any] = answer["result"]
        return result

    def _message(self, wait_s: float) -> dict[str, Any]:
        until = time.monotonic() + wait_s
        while b"\n" not in self._buffer:
            left = until - time.monotonic()
            if left <= 0:
                raise Gone(f"The server did not answer in {wait_s:g} s.")
            try:
                ready, _, _ = select.select([self._fd], [], [], left)
                chunk = os.read(self._fd, 65536) if ready else None
            except OSError as error:
                raise Gone(f"The server's connection failed: {error}.") from None
            if chunk is None:
                continue
            if not chunk:
                raise Gone("The server ended without answering.")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            message: dict[str, Any] = json.loads(line)
        except ValueError:
            raise Gone(f"The server wrote a line that is no JSON: {line!r}.") from None
        return message


@contextlib.contextmanager
def connect_or_start(file: str, scene: str, config: dict[str, Any]) -> Iterator[Remote]:
    """A connection to a server for the session of ``scene`` in ``file``
    under ``config``. The caller opens the session on it.

    Here, each connection starts a server of its own on its standard
    streams, and ends it when the connection ends: asked to shut down, then
    killed if it has not ended within the protocol's ending time.
    """
    process = subprocess.Popen(
        [sys.executable, "-m", "arachne.server", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    stdin = process.stdin

    def send(data: bytes) -> None:
        stdin.write(data)
        stdin.flush()

    try:
        remote = Remote(process.stdout.fileno(), send)
        remote.ready()
        yield remote
        with contextlib.suppress(Failed, Gone):
            remote.call("shutdown", {"deadline_s": ASK_S})
    finally:
        with contextlib.suppress(OSError):
            process.stdin.close()
        try:
            process.wait(protocol.ENDING_S * 2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()


def sessions() -> list[dict[str, Any]]:
    """The servers running on this host, each ``{key, pid, socket,
    generation}``: ``key`` is what :func:`stop` takes, ``generation`` the
    generation of the session it holds (null with none open, or when it
    was busy and did not answer).

    Here, they are the servers listening on a socket in
    ``ARACHNE_SESSION_DIR`` (none when it is unset), each keyed by its pid.
    """
    directory = os.environ.get("ARACHNE_SESSION_DIR")
    if not directory or not os.path.isdir(directory):
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.sock")):
        if not path.stem.isdigit():
            continue
        try:
            with _connected(path) as remote:
                try:
                    generation = remote.call("ping", {"deadline_s": ASK_S})[
                        "generation"
                    ]
                except (Failed, Gone):
                    generation = None
        except OSError:
            # A socket nothing listens on: a server that ended uncleanly.
            continue
        pid = int(path.stem)
        found.append(
            {"key": pid, "pid": pid, "socket": str(path), "generation": generation}
        )
    return found


def stop(key: str) -> bool:
    """Ask the server ``key`` names (as :func:`sessions` lists it) to shut
    down: whether it said it would.
    """
    for entry in sessions():
        if str(entry["key"]) == str(key):
            try:
                with _connected(Path(entry["socket"])) as remote:
                    remote.call("shutdown", {"deadline_s": ASK_S})
            except (OSError, Failed, Gone):
                return False
            return True
    return False


@contextlib.contextmanager
def _connected(path: Path) -> Iterator[Remote]:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(ASK_S)
        connection.connect(os.fspath(path))
        connection.settimeout(None)
        yield Remote(connection.fileno(), connection.sendall)
    finally:
        connection.close()
