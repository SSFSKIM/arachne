"""The session server's process: its transports, its one request at a time,
its deadlines and its idle end.

:func:`serve` answers requests read off the process's standard input
(``stdio``) or off the connections to a Unix socket (``socket``), one at a
time, in the order read, whichever connection sent them. Whatever else
writes to standard output (the scene's own prints, the engine's logging)
is sent to standard error: the protocol's stream holds protocol lines only.
"""

from __future__ import annotations

__all__ = ["IDLE_S", "main", "serve"]

import argparse
import contextlib
import math
import os
import selectors
import shutil
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

from . import protocol
from .dispatch import Dispatcher, deadline
from .protocol import RpcError
from .watchdog import Watchdog

#: How long a server waits for a request before it ends itself, in seconds,
#: unless ``ARACHNE_SESSION_IDLE_S`` says otherwise.
IDLE_S = 600.0


class _Channel:
    """One stream requests are read from and answered on: lines read by
    ``read()`` (``b""`` at its end), answers written by ``write(data)``.
    """

    def __init__(
        self,
        fd: int,
        read: Callable[[], bytes],
        write: Callable[[bytes], None],
        close: Callable[[], None],
    ) -> None:
        self.fd = fd
        self._read = read
        self._write = write
        self._close = close
        self._buffer = b""
        self._writing = threading.Lock()
        self.open = True

    def lines(self) -> list[bytes] | None:
        """The whole lines read now; ``None`` at the stream's end."""
        try:
            chunk = self._read()
        except OSError:
            chunk = b""
        if not chunk:
            return None
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        return [line for line in lines if line.strip()]

    def send(self, data: bytes) -> None:
        with self._writing:
            if not self.open:
                return
            try:
                self._write(data)
            except OSError:
                # The other end is gone: nothing more is sent there.
                self.open = False

    def close(self) -> None:
        self.open = False
        self._close()


def _write_all(fd: int) -> Callable[[bytes], None]:
    def write(data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]

    return write


class _Server:
    def __init__(
        self, idle_s: float, deadline_s: float, cleanup: Callable[[], None]
    ) -> None:
        self.idle_s = idle_s
        self.deadline_s = deadline_s
        self.dispatcher = Dispatcher()
        self.watchdog = Watchdog()
        self._cleanup = cleanup

    def run(self, selector: selectors.BaseSelector, ends_with: _Channel | None) -> int:
        """Answer what the selector's channels send until a request ends the
        server, ``ends_with`` ends, or nothing came for ``idle_s``.
        """
        last = time.monotonic()
        while True:
            left = self.idle_s - (time.monotonic() - last)
            if left <= 0:
                print(f"idle for {self.idle_s:g} s: the server ends", file=sys.stderr)
                return 0
            for key, _ in selector.select(left):
                if isinstance(key.fileobj, socket.socket) and key.data is None:
                    selector.register(*_accept(key.fileobj))
                    continue
                channel: _Channel = key.data
                lines = channel.lines()
                if lines is None:
                    selector.unregister(channel.fd)
                    channel.close()
                    if channel is ends_with:
                        return 0
                    continue
                for line in lines:
                    self.answer(channel, line)
                    last = time.monotonic()
                    if self.dispatcher.ending:
                        return 0

    def answer(self, channel: _Channel, line: bytes) -> None:
        """Answer ``line`` within a deadline, writing the answer included:
        the server's until the request's own is read, then the request's.
        """
        limit = self.deadline_s
        self.watchdog.arm(limit, lambda: self._overran(channel, None, limit))
        try:
            self._answer(channel, line)
        finally:
            self.watchdog.disarm()

    def _answer(self, channel: _Channel, line: bytes) -> None:
        try:
            request = protocol.parse(line)
        except RpcError as found:
            channel.send(protocol.encode(protocol.error(found.request_id, found)))
            return
        try:
            given, params = deadline(request.params)
        except RpcError as found:
            if request.has_id:
                channel.send(protocol.encode(protocol.error(request.id, found)))
            return
        limit = self.deadline_s if given is None else given
        self.watchdog.arm(limit, lambda: self._overran(channel, request, limit))
        try:
            reply = protocol.result(
                request.id, self.dispatcher.handle(request.method, params)
            )
        except RpcError as found:
            reply = protocol.error(request.id, found)
        if request.has_id:
            channel.send(_encoded(reply))

    def _overran(
        self, channel: _Channel, request: protocol.Request | None, limit: float
    ) -> None:
        """End the server: ``request`` (or a line not yet read as one) ran
        past its deadline, ``limit`` s. Its error is sent for at most
        :data:`~.protocol.DEADLINE_WRITE_S`: a client that stopped reading,
        or a write of the request's own answer that stalled, can't hold the
        server.
        """
        if request is not None and request.has_id:
            found = RpcError(
                protocol.DEADLINE_EXCEEDED,
                f"{request.method} ran past its deadline of {limit:g} s:"
                " the server ends.",
                {"method": request.method, "deadline_s": limit},
            )
            data = protocol.encode(protocol.error(request.id, found))
            writer = threading.Thread(target=channel.send, args=(data,), daemon=True)
            writer.start()
            writer.join(protocol.DEADLINE_WRITE_S)
        self._cleanup()
        os._exit(protocol.EXIT_DEADLINE)


def _encoded(reply: dict[str, Any]) -> bytes:
    try:
        return protocol.encode(reply)
    except (TypeError, ValueError) as found:
        failed = RpcError(protocol.INTERNAL_ERROR, f"The answer is no JSON: {found}")
        return protocol.encode(protocol.error(reply.get("id"), failed))


def _accept(listener: socket.socket) -> tuple[int, int, _Channel]:
    connection, _ = listener.accept()
    connection.setblocking(True)
    channel = _Channel(
        connection.fileno(),
        lambda: connection.recv(65536),
        connection.sendall,
        connection.close,
    )
    return connection.fileno(), selectors.EVENT_READ, channel


def _socket_dir() -> tuple[str, bool]:
    """Where the socket goes, and whether the directory is the server's own:
    ``ARACHNE_SESSION_DIR`` when set, else a private temporary directory.
    """
    named = os.environ.get("ARACHNE_SESSION_DIR")
    if named:
        os.makedirs(named, mode=0o700, exist_ok=True)
        return named, False
    return tempfile.mkdtemp(prefix="arachne-"), True


def serve(
    transport: str,
    *,
    idle_s: float = IDLE_S,
    deadline_s: float = protocol.DEADLINE_S,
) -> int:
    """Serve requests on ``transport`` (``"stdio"`` or ``"socket"``) until a
    ``shutdown`` request, the end of standard input (``stdio``), or
    ``idle_s`` seconds without one: the exit status. A request that gives
    no ``deadline_s`` has ``deadline_s``.
    """
    # The protocol's stream is the original standard output; whatever else
    # writes there now writes to standard error.
    sys.stdout.flush()
    out = os.dup(1)
    os.dup2(2, 1)
    selector = selectors.DefaultSelector()
    if transport == "stdio":
        channel = _Channel(0, lambda: os.read(0, 65536), _write_all(out), lambda: None)
        selector.register(0, selectors.EVENT_READ, channel)
        server = _Server(idle_s, deadline_s, lambda: None)
        channel.send(protocol.encode(protocol.ready("stdio", os.getpid())))
        return server.run(selector, ends_with=channel)
    directory, own = _socket_dir()
    path = os.path.join(directory, f"{os.getpid()}.sock")

    def cleanup() -> None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        if own:
            shutil.rmtree(directory, ignore_errors=True)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(path)
        listener.listen()
        selector.register(listener, selectors.EVENT_READ, None)
        server = _Server(idle_s, deadline_s, cleanup)
        _write_all(out)(protocol.encode(protocol.ready("socket", os.getpid(), path)))
        os.close(out)
        return server.run(selector, ends_with=None)
    finally:
        listener.close()
        cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arachne.server",
        description="Serve one scene's session over JSON-RPC 2.0, a message a line.",
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--stdio",
        dest="transport",
        action="store_const",
        const="stdio",
        help="on standard input and output",
    )
    where.add_argument(
        "--socket",
        dest="transport",
        action="store_const",
        const="socket",
        help="on a Unix socket in ARACHNE_SESSION_DIR, else a private directory",
    )
    parser.add_argument(
        "--deadline-s",
        type=_seconds,
        default=None,
        help=(
            "how long a request that gives no deadline_s may run, in seconds:"
            f" else ARACHNE_SESSION_DEADLINE_S, else {protocol.DEADLINE_S:g}"
        ),
    )
    args = parser.parse_args(argv)
    try:
        idle_s = _seconds(os.environ.get("ARACHNE_SESSION_IDLE_S") or str(IDLE_S))
        deadline_s = args.deadline_s or _seconds(
            os.environ.get("ARACHNE_SESSION_DEADLINE_S") or str(protocol.DEADLINE_S)
        )
    except argparse.ArgumentTypeError as found:
        parser.error(str(found))
    return serve(args.transport, idle_s=idle_s, deadline_s=deadline_s)


def _seconds(text: str) -> float:
    """``text`` as seconds, a finite number above 0."""
    try:
        found = float(text)
    except ValueError:
        found = math.nan
    if not (math.isfinite(found) and found > 0):
        raise argparse.ArgumentTypeError(f"{text!r} is no number of seconds above 0")
    return found
