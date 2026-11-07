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
directories never share a server. Nor do callers on two engines (two
interpreters, two versions or builds of the engine or its native core): a
session's key holds the engine it is asked on. A second caller for a
session waits its turn behind the first for as long as its deadline.

A server is ended only once it is reserved for its end: ``<pid>.lock``
beside its entry, which the server itself holds while it answers a request,
and whoever ends it takes while it does not. A reserved server answers
nothing more: the next request it reads, it drops unanswered, and ends,
its caller seeing :class:`ServerGone` (which :func:`connect_or_start` takes
as a server to start afresh). So a server answering a request is never
ended under it, and a server ended is never killed in the middle of one.

Under ``ARACHNE_SESSION_MEMORY_MB``, every server, once it has answered,
ends the least recently used of the other servers of its directory it can
reserve, while together they hold more than that many megabytes: it asks
each to shut down, and kills it if it has not ended within a second or two.
Still over the budget once no other server is left to end, it ends itself,
its answer sent. Neither this nor a server's idle end waits for a caller.
Servers of two directories never see each other.

What a client here raises is a :class:`ClientError`.
"""

from __future__ import annotations

__all__ = [
    "BadRequest",
    "ClientError",
    "ClientTimeout",
    "Connection",
    "Entry",
    "Registration",
    "RemoteError",
    "ServerGone",
    "StartFailed",
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
import math
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

from .._config.utils import config_file_paths
from . import protocol, store

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


class ClientError(Exception):
    """What a client of the registry raises: one of the kinds below, or,
    for a server that answered a line that is not the protocol's, this.
    """


class RemoteError(ClientError):
    """An error answer: its JSON-RPC ``code``, ``message`` and ``data``."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"{message} ({code})")
        self.code = code
        self.message = message
        self.data = data


class ServerGone(ClientError, ConnectionError):
    """The server ended, or closed the connection, without answering."""


class ClientTimeout(ClientError, TimeoutError):
    """No answer, or no turn at the session's server, within the deadline."""


class StartFailed(ClientError, RuntimeError):
    """A server started for a session did not come to serve."""


class BadRequest(ClientError, ValueError):
    """What was asked can't be sent: a parameter the wire can't carry (a
    value JSON doesn't hold, a number that isn't finite), or a registry
    directory no server can serve in.
    """


def session_key(
    file: str,
    scene: str,
    config: Mapping[str, Any],
    cwd: str,
    *,
    settings: str | None = None,
) -> str:
    """The registry's key of the session of ``scene`` of ``file`` (a real
    path) under ``config``, asked from the working directory ``cwd`` (a real
    path), on this engine (:func:`_engine_identity`), under the
    configuration files ``settings`` digests (:func:`_settings_files`, as
    they read now unless given).
    """
    if settings is None:
        settings = _settings_files(cwd)
    text = json.dumps(
        [file, scene, dict(config), cwd, _engine_identity(), settings], sort_keys=True
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _settings_files(cwd: str) -> str:
    """A digest of the configuration files in force from ``cwd``
    (:func:`~arachne._config.utils.config_file_paths`, the folder's read
    from there), by content: what a server read as it started, so that an
    edit to one is another session.
    """
    digest = hashlib.sha256()
    for path in config_file_paths():
        found = path if path.is_absolute() else Path(cwd) / path
        try:
            content = found.read_bytes()
        except OSError:
            content = b"\0absent"
        digest.update(os.fsencode(found) + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


def _engine_identity() -> str:
    """The engine a session is asked on: the interpreter, by real path and
    version, and the engine's fingerprint, its native core included
    (:func:`.store.engine_fingerprint`).
    """
    executable = os.path.realpath(sys.executable)
    return f"{executable}\n{sys.version}\n{store.engine_fingerprint()}"


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
        raise BadRequest(
            f"{found} is too long a directory for a session's socket: its socket"
            f" path would take {longest} bytes, and a Unix socket's address holds"
            f" {SOCKET_PATH_MAX}. Name a shorter ARACHNE_SESSION_DIR."
        )
    found.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = found.stat()
    if status.st_uid != os.getuid():
        raise BadRequest(f"{found} belongs to another user.")
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
    """Remove what a server left in ``directory``: its entry, socket, log
    and lock.
    """
    for suffix in (".json", ".sock", ".log", ".lock"):
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
    left = [
        path
        for suffix in (".sock", ".log", ".lock")
        for path in directory.glob(f"*{suffix}")
    ]
    for path in left:
        pid = _pid_of(path)
        if pid is None or (directory / f"{pid}.json").exists() or _alive(pid):
            continue
        with contextlib.suppress(OSError):
            path.unlink()
    return sorted(entries, key=lambda entry: entry.pid)


def sessions(directory: StrPath | None = None) -> list[Entry]:
    """The servers of the registry, by pid, what dead ones left swept first."""
    return _swept(session_dir(directory))


def _flock(handle: int, wait: float) -> bool:
    """Take the lock of the open file ``handle``: whether it was had within
    ``wait`` seconds.
    """
    until = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= until:
                return False
            time.sleep(0.05)


@contextlib.contextmanager
def _locked(path: Path, wait: float) -> Iterator[bool]:
    """Hold the lock ``path`` (made when missing): whether it was had within
    ``wait`` seconds.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        yield _flock(handle, wait)
    finally:
        os.close(handle)


@contextlib.contextmanager
def _reserved(directory: Path, pid: int, wait: float = 0.0) -> Iterator[bool]:
    """Reserve the server ``pid`` of ``directory`` for its end, as long as
    this holds: whether it was reserved within ``wait`` seconds. A server
    answering a request holds its own reservation, and a server with no
    lock (gone, or never come to serve) is never reserved.
    """
    try:
        handle = os.open(directory / f"{pid}.lock", os.O_RDWR)
    except OSError:
        yield False
        return
    try:
        yield _flock(handle, wait)
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
        except OSError as found:
            self._connection.close()
            raise ServerGone(
                f"The server {pid} takes no connection: {found}."
            ) from None
        except BaseException:
            self._connection.close()
            raise
        self._connection.settimeout(None)

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        """The result of ``method`` asked with ``params`` (``deadline_s``
        among them). Raises :class:`RemoteError` for an error answer,
        :class:`ServerGone` when the server ended without one,
        :class:`ClientTimeout` when none came by the deadline and
        :class:`BadRequest`, sending nothing, for parameters the wire can't
        carry.
        """
        self._ids += 1
        request = {"jsonrpc": "2.0", "id": self._ids, "method": method}
        if params:
            request["params"] = params
        try:
            line = protocol.encode(request)
        except (TypeError, ValueError) as found:
            raise BadRequest(f"{method}'s parameters can't be sent: {found}.") from None
        deadline = params.get("deadline_s")
        # The server answers by its deadline, or ends within its way out; a
        # deadline it refuses is answered at once.
        wait = None
        if isinstance(deadline, (int, float)) and not isinstance(deadline, bool):
            wait = max(deadline, 0) + 4 * protocol.ENDING_S
        try:
            self._connection.sendall(line)
            answer = self._line(wait)
        except ClientError:
            raise
        except OSError as found:
            raise ServerGone(f"The server {self.pid} ended: {found}.") from None
        try:
            response = json.loads(answer)
            if "error" not in response:
                result: dict[str, Any] = response["result"]
                return result
            error = response["error"]
            code, message = error["code"], error["message"]
        except (ValueError, TypeError, KeyError):
            raise ClientError(
                f"The server {self.pid} answered a line that is not the protocol's:"
                f" {answer[:200]!r}."
            ) from None
        if code == protocol.DEADLINE_EXCEEDED:
            # The server ends on its way out: once this returns, it has.
            _ended(self.pid, 4 * protocol.ENDING_S)
        raise RemoteError(code, message, error.get("data"))

    def _line(self, wait: float | None) -> bytes:
        until = None if wait is None else time.monotonic() + wait
        while b"\n" not in self._buffer:
            if until is not None:
                left = until - time.monotonic()
                if left <= 0:
                    raise ClientTimeout(f"The server {self.pid} gave no answer.")
                self._connection.settimeout(left)
            try:
                chunk = self._connection.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                raise ServerGone(
                    f"The server {self.pid} ended without answering: a server ends"
                    " idle, stopped, or for the host's memory budget, and"
                    " connect_or_start opens its session again."
                )
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
        raise StartFailed(f"A session server did not start:\n{tail}") from None
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


def _end(directory: Path, pid: int, path: str, wait: float) -> bool:
    """End the server ``pid`` once it is reserved for its end, within
    ``wait`` seconds: ask it to, then kill it if it has not ended within
    :data:`SHUTDOWN_S`; what it left is removed. Whether it was reserved:
    a server not reserved (answering a request all that time) is left as
    it is.
    """
    with _reserved(directory, pid, wait) as reserved:
        if not reserved:
            return False
        # Reserved, it ends at the next line it reads, answering nothing.
        _ask_to_end(path)
        if not _ended(pid, SHUTDOWN_S):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
            _ended(pid, KILLED_S)
    _forget(directory, pid)
    return True


def _ask_to_end(path: str) -> None:
    """Send the server at ``path`` a ``shutdown``, waiting for no answer."""
    with contextlib.suppress(OSError), socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(SHUTDOWN_S)
        connection.connect(path)
        connection.sendall(protocol.encode({"jsonrpc": "2.0", "method": "shutdown"}))


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
    asked from, where a server it starts runs. A caller asking while
    another asks for the same session waits its turn for as long as the
    deadline (the protocol's, :data:`~.protocol.DEADLINE_S`, unless
    ``deadline_s``), then raises :class:`ClientTimeout`: the deadline bounds
    the caller's whole wait, its turn, a start and the ``open`` together,
    the ``open`` sent with what is left of it.
    """
    if deadline_s is not None and not _positive(deadline_s):
        raise BadRequest(
            f"deadline_s is a number of seconds above 0, not {deadline_s!r}."
        )
    wait = protocol.DEADLINE_S if deadline_s is None else deadline_s
    until = time.monotonic() + wait
    found_dir = session_dir(directory)
    named = os.path.abspath(file)
    config = dict(config or {})
    params: dict[str, Any] = {"file": named, "scene": scene, "config": config}
    # What the wire can't carry is refused here, before a turn or a start.
    try:
        protocol.encode({"jsonrpc": "2.0", "id": 0, "method": "open", "params": params})
    except (TypeError, ValueError) as found:
        raise BadRequest(f"open's parameters can't be sent: {found}.") from None
    cwd = _cwd()
    key = session_key(os.path.realpath(file), scene, config, cwd)

    def opened(connection: Connection) -> dict[str, Any]:
        # The open has what is left of the deadline, nothing when none is.
        left = until - time.monotonic()
        if left <= 0:
            raise ClientTimeout(
                f"The deadline, {wait:g} s, passed before the session could be opened."
            )
        return connection.call("open", **params, deadline_s=left)

    with _locked(found_dir / "locks" / f"{key}.lock", wait) as had:
        if not had:
            raise ClientTimeout(
                f"Another caller held the turn at this session for {wait:g} s,"
                " its deadline."
            )
        for entry in _swept(found_dir):
            if entry.key != key:
                continue
            try:
                connection = Connection(entry.socket, entry.pid, key)
            except OSError:
                continue
            try:
                connection.opened = opened(connection)
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
        except BaseException as found:
            _end(found_dir, pid, path, SHUTDOWN_S)
            if isinstance(found, ServerGone):
                raise StartFailed(f"A session server started, then {found}") from None
            raise
        try:
            connection.opened = opened(connection)
        except BaseException:
            connection.close()
            _end(found_dir, pid, path, SHUTDOWN_S)
            raise
        return connection


def _positive(seconds: object) -> bool:
    return (
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and math.isfinite(seconds)
        and seconds > 0
    )


def stop(target: str | int, directory: StrPath | None = None) -> list[Entry]:
    """End the servers whose key, or pid, is ``target``: the entries ended.
    A server answering a request past :data:`SHUTDOWN_S` is never killed in
    the middle of it: asked to end, it ends once it has answered.
    """
    found_dir = session_dir(directory)
    ended = [
        entry
        for entry in _swept(found_dir)
        if entry.key == str(target) or str(entry.pid) == str(target)
    ]
    for entry in ended:
        _stop(found_dir, entry)
    return ended


def stop_all(directory: StrPath | None = None) -> list[Entry]:
    """End every server of the registry: the entries ended, as :func:`stop`
    ends them.
    """
    found_dir = session_dir(directory)
    ended = _swept(found_dir)
    for entry in ended:
        _stop(found_dir, entry)
    return ended


def _stop(directory: Path, entry: Entry) -> None:
    if not _end(directory, entry.pid, entry.socket, SHUTDOWN_S):
        _ask_to_end(entry.socket)


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
        # The configuration files as the engine read them when it loaded: an
        # edit since is another session, which this server does not hold.
        self._settings = _settings_files(self.cwd)
        self._path = directory / f"{pid}.json"
        #: The session's key and generation last written.
        self._written: tuple[str | None, int | None] | None = None
        # The server's reservation for its end: its own while it answers.
        self._lock_path = directory / f"{pid}.lock"
        self._lock = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)

    def admit(self) -> bool:
        """Take the server's own reservation for a request it read: false
        when it is reserved for its end, and must answer nothing more.
        Held until :meth:`release`.
        """
        return _flock(self._lock, 0.0)

    def release(self) -> None:
        """Give the reservation back: the request is answered."""
        fcntl.flock(self._lock, fcntl.LOCK_UN)

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
        key = (
            None
            if held is None
            else session_key(*held, self.cwd, settings=self._settings)
        )
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
        """Remove the entry, the server's log and its lock."""
        for path in (self._path, self.directory / f"{self.pid}.log", self._lock_path):
            with contextlib.suppress(OSError):
                path.unlink()

    def enforce(self) -> bool:
        """End the least recently used of the directory's other servers it
        can reserve at once (none answering a request), while they all hold
        more than the budget: whether they still do, with no other server
        left to end, and this one must end itself.
        """
        if self.budget_mb is None:
            return False
        with _locked(self.directory / "registry.lock", wait=SHUTDOWN_S) as had:
            if not had:
                return False
            entries = _swept(self.directory)
            held = sum(entry.rss_mb for entry in entries)
            others = sorted(
                (e for e in entries if e.pid != self.pid and not e.busy),
                key=lambda entry: entry.last_used,
            )
            for entry in others:
                if held <= self.budget_mb:
                    break
                if _end(self.directory, entry.pid, entry.socket, 0.0):
                    held -= entry.rss_mb
            return held > self.budget_mb
