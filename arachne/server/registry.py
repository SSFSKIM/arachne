"""The host's session servers: a registry in a directory, the clients that
find or start a server there, and the memory budget the servers keep.

The directory is ``ARACHNE_SESSION_DIR``, else a per-user one
(:func:`session_dir`), mode 0700. Each server serving on a socket there
keeps an entry beside its socket, ``<pid>.json``: the session it holds
(its file, by real path, scene and config, the working directory the server
runs in, and their key), its generation,
the memory it has held at most (``rss_mb``), when it last answered, and
whether it is answering now. A server writes its entry as it starts
serving, keeps it as it answers, and removes it, with its socket, as it
ends. An entry whose process is gone, or whose socket no longer listens (a
server killed, its pid since reused), is swept by whoever reads the
registry next, with its socket; so is a socket left with no entry.

:func:`connect_or_start` finds the server holding a session, or starts one
(detached, so that no caller's end ends it), and opens the session there:
asked again for the session it holds, a server answers it
(:mod:`arachne.server`'s ``open``). Two callers asking for one session at
once start one server. A session is of the working directory it was asked
from, where its server runs: a relative path in its config, and the
configuration file of the directory, are read from there, so callers in two
directories never share a server.

Under ``ARACHNE_SESSION_MEMORY_MB``, every server, once it has answered,
ends the least recently used of the other servers of its directory that are
not answering, while together they hold more than that many megabytes: it
asks each to shut down, and kills it if it has not ended within a second or
two. Neither this nor a server's idle end waits for a caller. Servers of
two directories never see each other.
"""

from __future__ import annotations

__all__ = [
    "Connection",
    "Entry",
    "Registration",
    "RemoteError",
    "ServerGone",
    "connect_or_start",
    "session_dir",
    "session_key",
    "sessions",
    "stop",
    "stop_all",
]

import contextlib
import fcntl
import hashlib
import json
import os
import resource
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import protocol

if TYPE_CHECKING:
    from ..typing import StrPath

#: The longest path a Unix socket's address holds, its terminating NUL
#: included.
SOCKET_PATH_MAX = 104 if sys.platform == "darwin" else 108
#: The longest socket name a server takes in the directory: the largest pid.
_LONGEST_NAME = "/4194304.sock"

#: How long a client waits for a server it started to be ready, in seconds.
START_S = 60.0
#: How long a server asked to shut down has, then how long a killed one
#: has, to end, in seconds.
SHUTDOWN_S = 2.0
KILLED_S = 2.0

# A server's process detached from whoever starts it: a fork whose parent
# exits at once, so the server's parent is the system's, its own session.
_DETACH = (
    "import os, sys\n"
    "if os.fork() == 0:\n"
    "    os.setsid()\n"
    "    os.execv(sys.argv[1], sys.argv[1:])\n"
    "os._exit(0)\n"
)


class RemoteError(Exception):
    """An error answer: its JSON-RPC ``code``, ``message`` and ``data``."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"{message} ({code})")
        self.code = code
        self.message = message
        self.data = data


class ServerGone(ConnectionError):
    """The server ended, or closed the connection, without answering."""


def session_key(file: str, scene: str, config: Mapping[str, Any], cwd: str) -> str:
    """The registry's key of the session of ``scene`` of ``file`` (a real
    path) under ``config``, asked from the working directory ``cwd`` (a real
    path).
    """
    text = json.dumps([file, scene, dict(config), cwd], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cwd() -> str:
    return os.path.realpath(os.getcwd())


def _default_dir() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "arachne"
    return Path(tempfile.gettempdir()) / f"arachne-{os.getuid()}"


def session_dir(directory: StrPath | None = None) -> Path:
    """The registry's directory: ``directory``, else ``ARACHNE_SESSION_DIR``,
    else ``$XDG_RUNTIME_DIR/arachne``, else ``arachne-<uid>`` in the
    temporary directory. Made, mode 0700, when missing; refused when another
    user owns it, or when a server's socket path there would not fit a Unix
    socket's address.
    """
    found = Path(
        os.path.abspath(
            directory or os.environ.get("ARACHNE_SESSION_DIR") or _default_dir()
        )
    )
    longest = len(os.fsencode(found)) + len(_LONGEST_NAME) + 1
    if longest > SOCKET_PATH_MAX:
        raise ValueError(
            f"{found} is too long a directory for a session's socket: its socket"
            f" path would take {longest} bytes, and a Unix socket's address holds"
            f" {SOCKET_PATH_MAX}. Name a shorter ARACHNE_SESSION_DIR."
        )
    found.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = found.stat()
    if status.st_uid != os.getuid():
        raise PermissionError(f"{found} belongs to another user.")
    if status.st_mode & 0o077:
        found.chmod(0o700)
    return found


@dataclass(frozen=True)
class Entry:
    """A server's entry in the registry. ``key``, ``file`` (a real path),
    ``scene``, ``config`` and ``generation`` are ``None`` while it holds no
    session. ``cwd``: the working directory the server runs in, a real
    path. ``rss_mb``: the most memory it has held, in megabytes;
    ``last_used``: when it last began or ended answering
    (:func:`time.time`); ``busy``: whether it is answering now.
    """

    pid: int
    socket: str
    key: str | None
    file: str | None
    scene: str | None
    config: dict[str, Any] | None
    cwd: str
    generation: int | None
    rss_mb: float
    last_used: float
    busy: bool


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _listening(path: str) -> bool:
    """Whether a server listens on the socket ``path``."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        probe.connect(path)
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _forget(directory: Path, pid: int) -> None:
    """Remove what a server left in ``directory``: its entry, socket, log."""
    for suffix in (".json", ".sock", ".log"):
        with contextlib.suppress(OSError):
            (directory / f"{pid}{suffix}").unlink()


def _pid_of(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def _read(path: Path) -> Entry | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Entry(**data)
    except (OSError, ValueError, TypeError):
        return None


def _swept(directory: Path) -> list[Entry]:
    """The entries of ``directory`` whose servers live, by pid; what the
    others left is removed.
    """
    entries: list[Entry] = []
    for path in sorted(directory.glob("*.json")):
        pid = _pid_of(path)
        if pid is None:
            continue
        entry = _read(path)
        if entry is None or not _alive(pid) or not _listening(entry.socket):
            _forget(directory, pid)
            continue
        entries.append(entry)
    for path in [*directory.glob("*.sock"), *directory.glob("*.log")]:
        pid = _pid_of(path)
        if pid is None or (directory / f"{pid}.json").exists() or _alive(pid):
            continue
        with contextlib.suppress(OSError):
            path.unlink()
    return sorted(entries, key=lambda entry: entry.pid)


def sessions(directory: StrPath | None = None) -> list[Entry]:
    """The servers of the registry, by pid, what dead ones left swept first."""
    return _swept(session_dir(directory))


@contextlib.contextmanager
def _locked(path: Path, wait: float | None = None) -> Iterator[bool]:
    """Hold the lock ``path`` (made when missing): whether it was had within
    ``wait`` seconds (forever with ``None``).
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if wait is None:
            fcntl.flock(handle, fcntl.LOCK_EX)
            had = True
        else:
            until = time.monotonic() + wait
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    had = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= until:
                        had = False
                        break
                    time.sleep(0.05)
        yield had
    finally:
        os.close(handle)


class Connection:
    """One connection to the server at the socket ``socket``, process
    ``pid``, holding the session ``key``; ``opened`` is what its ``open``
    answered.
    """

    def __init__(self, socket_path: str, pid: int, key: str) -> None:
        self.socket = socket_path
        self.pid = pid
        self.key = key
        self.opened: dict[str, Any] = {}
        self._buffer = b""
        self._ids = 0
        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._connection.settimeout(SHUTDOWN_S)
        try:
            self._connection.connect(socket_path)
        except BaseException:
            self._connection.close()
            raise
        self._connection.settimeout(None)

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        """The result of ``method`` asked with ``params`` (``deadline_s``
        among them). Raises :class:`RemoteError` for an error answer, and
        :class:`ServerGone` when the server ended without one.
        """
        self._ids += 1
        request = {"jsonrpc": "2.0", "id": self._ids, "method": method}
        if params:
            request["params"] = params
        deadline = params.get("deadline_s")
        # The server answers by its deadline, or ends within its way out.
        wait = None if deadline is None else deadline + 4 * protocol.ENDING_S
        try:
            self._connection.sendall(protocol.encode(request))
            line = self._line(wait)
        except (BrokenPipeError, ConnectionResetError) as found:
            raise ServerGone(f"The server {self.pid} ended: {found}.") from None
        response = json.loads(line)
        if "error" in response:
            error = response["error"]
            if error["code"] == protocol.DEADLINE_EXCEEDED:
                # The server ends on its way out: once this returns, it has.
                _ended(self.pid, 4 * protocol.ENDING_S)
            raise RemoteError(error["code"], error["message"], error.get("data"))
        result: dict[str, Any] = response["result"]
        return result

    def _line(self, wait: float | None) -> bytes:
        until = None if wait is None else time.monotonic() + wait
        while b"\n" not in self._buffer:
            if until is not None:
                left = until - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"The server {self.pid} gave no answer.")
                self._connection.settimeout(left)
            try:
                chunk = self._connection.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                raise ServerGone(f"The server {self.pid} ended without answering.")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _start(directory: Path, cwd: str) -> tuple[int, str]:
    """Start a server on a socket in ``directory``, detached, in the working
    directory ``cwd``: its pid and socket, once it is ready.
    """
    log = tempfile.NamedTemporaryFile(  # noqa: SIM115 (renamed, never closed here)
        dir=directory, prefix=".start-", suffix=".log", delete=False
    )
    try:
        launcher = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _DETACH,
                sys.executable,
                "-m",
                "arachne.server",
                "--socket",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
            cwd=cwd,
            env={**os.environ, "ARACHNE_SESSION_DIR": os.fspath(directory)},
        )
        launcher.wait()
        assert launcher.stdout is not None
        line = _ready_line(launcher.stdout.fileno())
        launcher.stdout.close()
    finally:
        log.close()
    try:
        ready = json.loads(line)["params"]
        pid, path = int(ready["pid"]), str(ready["socket"])
    except (ValueError, KeyError, TypeError):
        tail = Path(log.name).read_text(errors="replace")[-2000:]
        with contextlib.suppress(OSError):
            os.unlink(log.name)
        raise RuntimeError(f"A session server did not start:\n{tail}") from None
    os.replace(log.name, directory / f"{pid}.log")
    return pid, path


def _ready_line(fd: int) -> bytes:
    """The first line on ``fd``, waited for at most :data:`START_S`; ``b""``
    when it ends first.
    """
    buffer = b""
    until = time.monotonic() + START_S
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while b"\n" not in buffer:
            left = until - time.monotonic()
            if left <= 0 or not selector.select(left):
                return b""
            chunk = os.read(fd, 65536)
            if not chunk:
                return b""
            buffer += chunk
    return buffer.split(b"\n", 1)[0]


def _end(directory: Path, pid: int, path: str) -> None:
    """End the server ``pid``: ask it to shut down, then kill it if it has
    not ended within :data:`SHUTDOWN_S`; what it left is removed.
    """
    with (
        contextlib.suppress(OSError, ValueError, RemoteError, TimeoutError),
        Connection(path, pid, "") as connection,
    ):
        connection.call("shutdown", deadline_s=SHUTDOWN_S / 2)
    if not _ended(pid, SHUTDOWN_S):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
        _ended(pid, KILLED_S)
    _forget(directory, pid)


def _ended(pid: int, wait: float) -> bool:
    until = time.monotonic() + wait
    while _alive(pid):
        if time.monotonic() >= until:
            return False
        time.sleep(0.02)
    return True


def connect_or_start(
    file: StrPath,
    scene: str,
    config: Mapping[str, Any] | None = None,
    *,
    directory: StrPath | None = None,
    deadline_s: float | None = None,
) -> Connection:
    """A connection to the server holding the session of the scene class
    ``scene`` of ``file`` under ``config``, started and opened when no
    server of the registry (``directory``, :func:`session_dir`) holds it:
    the session opened there, as ``open`` opens it, by the request's
    ``deadline_s``. Asked again, it finds the same server, which answers
    the session it holds, caught up with its files. A server it started
    whose ``open`` failed is ended, and the error raised
    (:class:`RemoteError`). The session is of the working directory it is
    asked from, where a server it starts runs.
    """
    found_dir = session_dir(directory)
    named = os.path.abspath(file)
    config = dict(config or {})
    cwd = _cwd()
    key = session_key(os.path.realpath(file), scene, config, cwd)
    params: dict[str, Any] = {"file": named, "scene": scene, "config": config}
    if deadline_s is not None:
        params["deadline_s"] = deadline_s
    with _locked(found_dir / "locks" / f"{key}.lock"):
        for entry in _swept(found_dir):
            if entry.key != key:
                continue
            try:
                connection = Connection(entry.socket, entry.pid, key)
            except OSError:
                continue
            try:
                connection.opened = connection.call("open", **params)
            except ServerGone:
                connection.close()
                continue
            except BaseException as found:
                connection.close()
                if (
                    isinstance(found, RemoteError)
                    and found.code == protocol.OTHER_SESSION
                ):
                    continue
                raise
            return connection
        pid, path = _start(found_dir, cwd)
        try:
            connection = Connection(path, pid, key)
        except BaseException:
            _end(found_dir, pid, path)
            raise
        try:
            connection.opened = connection.call("open", **params)
        except BaseException:
            connection.close()
            _end(found_dir, pid, path)
            raise
        return connection


def stop(target: str | int, directory: StrPath | None = None) -> list[Entry]:
    """End the servers whose key, or pid, is ``target``: the entries ended."""
    found_dir = session_dir(directory)
    ended = [
        entry
        for entry in _swept(found_dir)
        if entry.key == str(target) or str(entry.pid) == str(target)
    ]
    for entry in ended:
        _end(found_dir, entry.pid, entry.socket)
    return ended


def stop_all(directory: StrPath | None = None) -> list[Entry]:
    """End every server of the registry: the entries ended."""
    found_dir = session_dir(directory)
    ended = _swept(found_dir)
    for entry in ended:
        _end(found_dir, entry.pid, entry.socket)
    return ended


# The server's side


def _rss_mb() -> float:
    """The most memory this process has held, in megabytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Bytes on macOS, kilobytes elsewhere.
    return peak / (1 << 20) if sys.platform == "darwin" else peak / (1 << 10)


class Registration:
    """A server's entry in the registry ``directory``: the server ``pid``,
    serving on ``socket`` from the working directory it started in, keeping
    the host's ``budget_mb`` (none when ``None``).
    """

    def __init__(
        self, directory: Path, pid: int, socket_path: str, budget_mb: float | None
    ) -> None:
        self.directory = directory
        self.pid = pid
        self.socket = socket_path
        self.budget_mb = budget_mb
        # Where it started: a scene's code changing directory moves no key.
        self.cwd = _cwd()
        self._path = directory / f"{pid}.json"
        #: The session's key and generation last written.
        self._written: tuple[str | None, int | None] | None = None

    def write(
        self,
        held: tuple[str, str, dict[str, Any]] | None,
        generation: int | None,
        *,
        busy: bool,
        if_changed: bool = False,
    ) -> None:
        """Write the entry: ``held``, the session's file, scene and config,
        at ``generation``; ``busy``, whether a request is being answered.
        With ``if_changed``, only when the session's key or generation
        changed since the entry was last written.
        """
        file, scene, config = held if held is not None else (None, None, None)
        key = None if held is None else session_key(*held, self.cwd)
        if if_changed and self._written == (key, generation):
            return
        entry = Entry(
            pid=self.pid,
            socket=self.socket,
            key=key,
            file=file,
            scene=scene,
            config=config,
            cwd=self.cwd,
            generation=generation,
            rss_mb=_rss_mb(),
            last_used=time.time(),
            busy=busy,
        )
        handle, temporary = tempfile.mkstemp(prefix=".entry-", dir=self.directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump(asdict(entry), out)
            os.replace(temporary, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        self._written = (key, generation)

    def remove(self) -> None:
        """Remove the entry, and the server's log."""
        for path in (self._path, self.directory / f"{self.pid}.log"):
            with contextlib.suppress(OSError):
                path.unlink()

    def enforce(self) -> None:
        """End the least recently used of the directory's other servers
        that are not answering, while they all hold more than the budget.
        """
        if self.budget_mb is None:
            return
        with _locked(self.directory / "registry.lock", wait=SHUTDOWN_S) as had:
            if not had:
                return
            entries = _swept(self.directory)
            held = sum(entry.rss_mb for entry in entries)
            others = sorted(
                (e for e in entries if e.pid != self.pid and not e.busy),
                key=lambda entry: entry.last_used,
            )
            for entry in others:
                if held <= self.budget_mb:
                    break
                _end(self.directory, entry.pid, entry.socket)
                held -= entry.rss_mb
