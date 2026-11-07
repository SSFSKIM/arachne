"""The session server's process: its transports, its one request at a time,
its deadlines and its end.

:func:`serve` answers requests read off the process's standard input
(``stdio``) or off the connections to a Unix socket (``socket``), one at a
time, in the order read, whichever connection sent them. Whatever else
writes to standard output (the scene's own prints, the engine's logging)
is sent to standard error: the protocol's stream holds protocol lines only.

Every way out of the process ends in :func:`os._exit`, each step before it
(a last error sent, the socket removed, a word on standard error, the
streams flushed) given at most :data:`~.protocol.ENDING_S` on a thread of
its own: nothing a client, a full pipe or the file system does can keep
the process alive past its end.
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
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from . import protocol
from .dispatch import Dispatcher, deadline, internal_error
from .protocol import RpcError
from .registry import Registration
from .store import Store
from .watchdog import Ticket, Watchdog

#: How long a server waits for a request before it ends itself, in seconds,
#: unless ``ARACHNE_SESSION_IDLE_S`` says otherwise.
IDLE_S = 600.0


def _bounded(work: Callable[[], object], seconds: float = protocol.ENDING_S) -> None:
    """Run ``work`` on a thread of its own for at most ``seconds``; whatever
    it raises is dropped, and a ``work`` still running is left behind.
    """

    def run() -> None:
        with contextlib.suppress(BaseException):
            work()

    worker = threading.Thread(target=run, name="ending", daemon=True)
    worker.start()
    worker.join(seconds)


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(BaseException):
            stream.flush()


def _end(status: int) -> NoReturn:
    """End the process with ``status``, its streams flushed if they can be
    within :data:`~.protocol.ENDING_S`.
    """
    try:
        _bounded(_flush)
    finally:
        os._exit(status)


def _say(message: str) -> None:
    """A word on standard error, if it can be written within
    :data:`~.protocol.ENDING_S`.
    """
    _bounded(lambda: print(message, file=sys.stderr, flush=True))


@dataclass(frozen=True)
class _Line:
    """A line read whole, and when (:func:`time.monotonic`). ``data`` is
    ``None`` for a line longer than :data:`~.protocol.MAX_MESSAGE_BYTES`,
    dropped as it was read.
    """

    data: bytes | None
    received: float


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
        # Whether the line being read is past the longest, and dropped.
        self._dropping = False
        self._writing = threading.Lock()
        self.open = True

    def lines(self) -> list[_Line] | None:
        """The lines read whole now; ``None`` at the stream's end. A line
        past the longest message is never kept: it is dropped as it comes,
        and given as one ``_Line`` with no data once its end is read.
        """
        try:
            chunk = self._read()
        except OSError:
            chunk = b""
        if not chunk:
            return None
        received = time.monotonic()
        *whole, rest = (self._buffer + chunk).split(b"\n")
        lines: list[_Line] = []
        for line in whole:
            if self._dropping or len(line) > protocol.MAX_MESSAGE_BYTES:
                self._dropping = False
                lines.append(_Line(None, received))
            elif line.strip():
                lines.append(_Line(line, received))
        if self._dropping or len(rest) > protocol.MAX_MESSAGE_BYTES:
            self._dropping, rest = True, b""
        self._buffer = rest
        return lines

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
        with contextlib.suppress(OSError):
            self._close()


def _write_all(fd: int) -> Callable[[bytes], None]:
    def write(data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]

    return write


class _Server:
    def __init__(
        self,
        idle_s: float,
        deadline_s: float,
        cleanup: Callable[[], None],
        *,
        store: Store | None = None,
        registration: Registration | None = None,
    ) -> None:
        self.idle_s = idle_s
        self.deadline_s = deadline_s
        self.dispatcher = Dispatcher(store)
        self.watchdog = Watchdog()
        self._cleanup = cleanup
        self.registration = registration

    def _registered(self, *, busy: bool) -> None:
        """Keep the server's entry in the registry, if it has one, and, once
        a request is answered, the host's memory budget. Neither can fail
        the server.
        """
        registration = self.registration
        if registration is None:
            return
        dispatcher = self.dispatcher
        with contextlib.suppress(Exception):
            registration.write(dispatcher.held, dispatcher.generation, busy=busy)
        if not busy and not dispatcher.ending:
            with contextlib.suppress(Exception):
                registration.enforce()

    def run(self, selector: selectors.BaseSelector, ends_with: _Channel | None) -> int:
        """Answer what the selector's channels send until a request ends the
        server, ``ends_with`` ends, or nothing came for ``idle_s``.
        """
        last = time.monotonic()
        while True:
            left = self.idle_s - (time.monotonic() - last)
            if left <= 0:
                _say(f"idle for {self.idle_s:g} s: the server ends")
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

    def answer(self, channel: _Channel, line: _Line) -> None:
        """Answer ``line`` by its deadline, counted from when it was read
        whole, writing the answer included: the server's until the request
        gives its own. A request answered and its deadline's error are one
        or the other, never both (:class:`~.watchdog.Ticket`).
        """
        read: list[protocol.Request] = []
        limit = [self.deadline_s]
        self._registered(busy=True)
        ticket = self.watchdog.arm(
            line.received + self.deadline_s,
            lambda respond: self._expired(channel, read, limit[0], respond),
        )
        try:
            reply = self._reply(line, ticket, read, limit)
            if reply is not None and self.watchdog.answering(ticket):
                channel.send(reply)
        finally:
            self.watchdog.end(ticket)
        self._registered(busy=False)

    def _reply(
        self,
        line: _Line,
        ticket: Ticket,
        read: list[protocol.Request],
        limit: list[float],
    ) -> bytes | None:
        """``line``'s answer as it is sent, or ``None`` for none (a
        notification's); ``read`` gets the request once it is read, and
        ``limit`` its deadline.
        """
        if line.data is None:
            too_long = RpcError(
                protocol.INVALID_REQUEST,
                f"A message is at most {protocol.MAX_MESSAGE_BYTES} bytes:"
                " this line was dropped.",
            )
            return protocol.encode(protocol.error(None, too_long))
        try:
            request = protocol.parse(line.data)
        except RpcError as found:
            if not found.answered:
                return None
            return protocol.encode(protocol.error(found.request_id, found))
        read.append(request)
        try:
            given, params = deadline(request.params)
        except RpcError as found:
            if not request.has_id:
                return None
            return protocol.encode(protocol.error(request.id, found))
        if given is not None:
            limit[0] = given
        # The request read: its deadline, its own or the server's, from when
        # its line was read whole; already past, it ends here, unrun.
        self.watchdog.move(ticket, line.received + limit[0])
        try:
            value = protocol.result(
                request.id, self.dispatcher.handle(request.method, params)
            )
        except RpcError as found:
            value = protocol.error(request.id, found)
        return _encoded(value) if request.has_id else None

    def _expired(
        self,
        channel: _Channel,
        read: list[protocol.Request],
        limit: float,
        respond: bool,
    ) -> NoReturn:
        """End the server: the request ``read`` holds (or a line not yet read
        as one) ran past its deadline, ``limit`` s. Its error is sent when
        the answer is still the deadline's (``respond``), for at most
        :data:`~.protocol.ENDING_S`; what a render running wrote is removed
        within as long, and the socket and the server's entry too; then the
        process exits, whatever each did.
        """
        try:
            if respond and read and read[0].has_id:
                request = read[0]

                def send() -> None:
                    found = RpcError(
                        protocol.DEADLINE_EXCEEDED,
                        f"{request.method} ran past its deadline of {limit:g} s:"
                        " the server ends.",
                        {"method": request.method, "deadline_s": limit},
                    )
                    channel.send(protocol.encode(protocol.error(request.id, found)))

                _bounded(send)
            _bounded(self.dispatcher.abandon)
            _bounded(self._cleanup)
        finally:
            os._exit(protocol.EXIT_DEADLINE)


def _encoded(reply: dict[str, Any]) -> bytes:
    """``reply`` as it is sent; an answer that is no JSON is the server's
    internal error, with the request's id, or null when that is no JSON.
    """
    try:
        return protocol.encode(reply)
    except (TypeError, ValueError, OverflowError, RecursionError) as found:
        failed = internal_error(found)
    for request_id in (reply.get("id"), None):
        with contextlib.suppress(TypeError, ValueError, OverflowError, RecursionError):
            return protocol.encode(protocol.error(request_id, failed))
    plain = RpcError(protocol.INTERNAL_ERROR, "The answer is no JSON.")
    return protocol.encode(protocol.error(None, plain))


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
    memory_mb: float | None = None,
) -> int:
    """Serve requests on ``transport`` (``"stdio"`` or ``"socket"``) until a
    ``shutdown`` request, the end of standard input (``stdio``), or
    ``idle_s`` seconds without one: the exit status. A request that gives
    no ``deadline_s`` has ``deadline_s``. Its answers are kept in the store
    (:class:`~.store.Store`, ``ARACHNE_STORE``). On a socket in
    ``ARACHNE_SESSION_DIR``, the server keeps its entry in the registry
    there, and the host's memory budget of ``memory_mb`` megabytes, if any
    (:mod:`.registry`).
    """
    # The protocol's stream is the original standard output; whatever else
    # writes there now writes to standard error.
    sys.stdout.flush()
    out = os.dup(1)
    os.dup2(2, 1)
    selector = selectors.DefaultSelector()
    store = Store()
    if transport == "stdio":
        channel = _Channel(0, lambda: os.read(0, 65536), _write_all(out), lambda: None)
        selector.register(0, selectors.EVENT_READ, channel)
        server = _Server(idle_s, deadline_s, lambda: None, store=store)
        channel.send(protocol.encode(protocol.ready("stdio", os.getpid())))
        return server.run(selector, ends_with=channel)
    directory, own = _socket_dir()
    path = os.path.join(directory, f"{os.getpid()}.sock")
    registration = (
        None if own else Registration(Path(directory), os.getpid(), path, memory_mb)
    )

    def cleanup() -> None:
        if registration is not None:
            registration.remove()
        with contextlib.suppress(OSError):
            os.unlink(path)
        if own:
            shutil.rmtree(directory, ignore_errors=True)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(path)
        listener.listen()
        selector.register(listener, selectors.EVENT_READ, None)
        server = _Server(
            idle_s, deadline_s, cleanup, store=store, registration=registration
        )
        if registration is not None:
            registration.write(None, None, busy=False)
        _write_all(out)(protocol.encode(protocol.ready("socket", os.getpid(), path)))
        os.close(out)
        return server.run(selector, ends_with=None)
    finally:
        listener.close()
        _bounded(cleanup)


def main(argv: list[str] | None = None) -> NoReturn:
    """Serve as the command line says, then end the process
    (:func:`os._exit`, its streams flushed within
    :data:`~.protocol.ENDING_S`): ``0`` on a server's own end, ``1`` when the
    server failed, ``2`` for a command line it refuses.
    """
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
        budget = os.environ.get("ARACHNE_SESSION_MEMORY_MB")
        memory_mb = _megabytes(budget) if budget else None
    except argparse.ArgumentTypeError as found:
        parser.error(str(found))
    try:
        status = serve(
            args.transport, idle_s=idle_s, deadline_s=deadline_s, memory_mb=memory_mb
        )
    except BaseException:
        _bounded(traceback.print_exc)
        status = 1
    _end(status)


def _seconds(text: str) -> float:
    """``text`` as seconds: a finite number above 0, at most
    :data:`~.protocol.MAX_DEADLINE_S`.
    """
    try:
        found = float(text)
    except ValueError:
        found = math.nan
    if not (math.isfinite(found) and 0 < found <= protocol.MAX_DEADLINE_S):
        raise argparse.ArgumentTypeError(
            f"{text!r} is no number of seconds above 0 and at most"
            f" {protocol.MAX_DEADLINE_S:g}"
        )
    return found


def _megabytes(text: str) -> float:
    """``text`` as megabytes: a finite number above 0."""
    try:
        found = float(text)
    except ValueError:
        found = math.nan
    if not (math.isfinite(found) and found > 0):
        raise argparse.ArgumentTypeError(
            f"ARACHNE_SESSION_MEMORY_MB {text!r} is no number of megabytes above 0"
        )
    return found
