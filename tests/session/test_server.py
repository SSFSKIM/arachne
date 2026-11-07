"""The session server: a session served in a process of its own, over JSON-RPC
2.0, one message per line, on its standard streams or on a Unix socket.

Every answer is a form of :mod:`arachne.api_json` with the generation it is
of; a finding is an answer, and a scene's error is an error carrying its
explanation; an edit is executed again before the next request is answered;
a request past its deadline ends the process, and so does an idle one.

Each test starts a server of its own and ends it. How long a test waits for
an answer is ``ARACHNE_TEST_SERVER_WAIT_S`` (60 s by default).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

import arachne
from arachne import api_json
from arachne.server import protocol
from arachne.server.watchdog import Watchdog

#: How long a test waits for the server to be ready, or to answer.
WAIT_S = float(os.environ.get("ARACHNE_TEST_SERVER_WAIT_S", "60"))

SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}
ROOT = Path(arachne.__file__).resolve().parent.parent
TRANSPORTS = ("stdio", "socket")

SHOWN = """\
    from arachne import RIGHT, Scene, Square, Text


    class Shown(Scene):
        def construct(self):
            print("the scene's own output")
            square = Square(side_length=1)
            label = Text("hi", font_size=24).next_to(square, RIGHT)
            self.add(square, label)
            self.play(square.animate.shift(RIGHT * 0.5), run_time=1)
            self.wait(0.5)
"""

EDGE = """\
    from arachne import RIGHT, Scene, Text
    from arachne.lint.bridge import sweep


    class Edge(Scene):
        def construct(self):
            word = Text("edge", font_size=36).to_edge(RIGHT, buff=-0.4)
            self.add(word)
            self.wait(0.5)
            sweep()
            self.wait(0.5)
"""

RAISES = """\
    from arachne import Scene


    class Raises(Scene):
        def construct(self):
            self.wait(1)
            raise ValueError("no such value")  # @raise
"""

SLEEPS = """\
    import time

    from arachne import Scene


    class Sleeps(Scene):
        def construct(self):
            time.sleep(60)
"""

NAPS = """\
    import time

    from arachne import Scene


    class Naps(Scene):
        def construct(self):
            time.sleep(1.5)
            self.wait(0.5)
"""

EXITS = """\
    import sys

    from arachne import Scene


    class Exits(Scene):
        def construct(self):
            self.wait(0.5)
            sys.exit(4)  # @exit
"""

MANY = """\
    from arachne import Dot, Scene, VGroup


    class Many(Scene):
        def construct(self):
            self.add(VGroup(*(Dot() for _ in range(400))))
            self.wait(0.5)
"""

HOOK_SCENES = ROOT / "tests" / "timeline" / "hook_scenes.py"

FORWARD_RAISES = """\
    from pathlib import Path

    from arachne import RIGHT, Dot, MappingCamera, Scene


    class ForwardRaises(Scene):
        \"\"\"A record that draws nothing, so each frame is a forward run's,
        and a run that raises once a file beside it says so.\"\"\"

        def __init__(self, **kwargs):
            super().__init__(camera_class=MappingCamera, **kwargs)
            self.renderer.camera.mapping_func = lambda p: p + 2 * RIGHT

        def construct(self):
            self.add(Dot())
            self.wait(0.5)
            if Path(__file__).with_name("raise-now").exists():
                raise RuntimeError("raised in a forward run")  # @raise
            self.wait(0.5)
"""

HOSTILE = """\
    from arachne import Scene


    class Hostile(Exception):
        def __str__(self):
            raise SystemExit("no message")

        def add_note(self, note):
            raise RuntimeError("no notes")


    class RaisesHostile(Scene):
        def construct(self):
            self.wait(0.5)
            raise Hostile()  # @raise
"""

ENGINE_RAISES = """\
    from arachne import Scene


    class PlaysNothing(Scene):
        def construct(self):
            self.play()  # @play
"""

FLOODS = """\
    import os
    import threading
    import time

    from arachne import Scene


    def _flood():
        time.sleep(2)
        while True:
            os.write(2, b"x" * 65536)


    class Floods(Scene):
        \"\"\"Output kept in the standard output's buffer until the process
        ends, and, from 2 s on, a standard error that never empties.\"\"\"

        def construct(self):
            print("kept until the end")
            threading.Thread(target=_flood, daemon=True).start()
            self.wait(0.5)
"""


class _Lines:
    """Lines read from a file descriptor, waited for at most ``WAIT_S``."""

    def __init__(self, fd: int, read: Any) -> None:
        self._fd = fd
        self._read = read
        self._buffer = b""

    def next(self, wait: float = WAIT_S) -> bytes | None:
        """The next line, or ``None`` at end of stream."""
        until = time.monotonic() + wait
        while b"\n" not in self._buffer:
            left = until - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no line in {wait} s")
            ready, _, _ = select.select([self._fd], [], [], left)
            if not ready:
                continue
            chunk = self._read()
            if not chunk:
                return None
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line


class Server:
    """A server process under test, and one connection to it."""

    def __init__(
        self,
        transport: str,
        cwd: Path,
        env: dict[str, str],
        args: tuple[str, ...],
        *,
        unread_stderr: bool = False,
    ) -> None:
        self.transport = transport
        self._socket: socket.socket | None = None
        # Closed by end(), which every test's fixture calls. An unread pipe
        # fills, as a parent that never reads it lets it.
        self._stderr: Any = (
            None if unread_stderr else tempfile.TemporaryFile()  # noqa: SIM115
        )
        self.process = subprocess.Popen(
            [sys.executable, "-m", "arachne.server", f"--{transport}", *args],
            cwd=cwd,
            env={**os.environ, "PYTHONPATH": str(ROOT), **env},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if unread_stderr else self._stderr,
        )
        try:
            self._start()
        except BaseException:
            # Never a process left behind by a start that failed.
            self.end()
            raise

    def _start(self) -> None:
        assert self.process.stdout is not None
        out = self.process.stdout.fileno()
        self._out = _Lines(out, lambda: os.read(out, 65536))
        line = self._out.next()
        assert line is not None, self.stderr()
        self.ready = json.loads(line)
        self._ids = 0
        if self.transport == "socket":
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(self.ready["params"]["socket"])
            connection = self._socket
            self._in = _Lines(connection.fileno(), lambda: connection.recv(65536))
        else:
            self._in = self._out

    def send(self, message: Any) -> None:
        """Send ``message``: a JSON value, or text or bytes sent as they are,
        each as a line.
        """
        if isinstance(message, bytes):
            data = message + b"\n"
        else:
            text = message if isinstance(message, str) else json.dumps(message)
            data = text.encode() + b"\n"
        if self._socket is not None:
            self._socket.sendall(data)
        else:
            assert self.process.stdin is not None
            self.process.stdin.write(data)
            self.process.stdin.flush()

    def receive(self, wait: float = WAIT_S) -> dict[str, Any] | None:
        line = self._in.next(wait)
        return None if line is None else json.loads(line)

    def ask(self, method: str, **params: Any) -> int:
        """Send a request for ``method`` with ``params``: its id."""
        self._ids += 1
        self.send(
            {"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params}
        )
        return self._ids

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        """The response to ``method`` asked with ``params``."""
        self.ask(method, **params)
        response = self.receive()
        assert response is not None, self.stderr()
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == self._ids
        return response

    def result(self, method: str, **params: Any) -> dict[str, Any]:
        """The result of ``method``, which must not be an error."""
        response = self.call(method, **params)
        assert "error" not in response, (response["error"], self.stderr())
        result: dict[str, Any] = response["result"]
        return result

    def error(self, method: str, **params: Any) -> dict[str, Any]:
        response = self.call(method, **params)
        assert "result" not in response, response
        error: dict[str, Any] = response["error"]
        return error

    def exited(self, wait: float = WAIT_S) -> int:
        return self.process.wait(wait)

    def stderr(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace")

    def end(self) -> None:
        """Kill the process if it still runs, reap it and close its streams;
        once is enough.
        """
        if self._socket is not None:
            self._socket.close()
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)
        self.process.wait(WAIT_S)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        if self._stderr is not None:
            self._stderr.close()


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    """A directory for sockets, short enough for a socket's path."""
    found = Path(tempfile.mkdtemp(prefix="as-", dir="/tmp"))
    yield found
    found.chmod(0o700)
    shutil.rmtree(found, ignore_errors=True)


@pytest.fixture
def serve(tmp_path: Path, socket_dir: Path) -> Iterator[Any]:
    """Start servers, each ended after the test."""
    started: list[Server] = []

    def start(
        transport: str = "stdio",
        *args: str,
        unread_stderr: bool = False,
        **env: str,
    ) -> Server:
        env = {"ARACHNE_SESSION_DIR": str(socket_dir), **env}
        env = {name: value for name, value in env.items() if value != ""}
        found = Server(transport, tmp_path, env, args, unread_stderr=unread_stderr)
        started.append(found)
        return found

    yield start
    for server in started:
        server.end()


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text))
    return path


def _open(server: Server, path: Path, scene: str, **params: Any) -> dict[str, Any]:
    media = str(path.parent / "media")
    return server.result(
        "open",
        file=str(path),
        scene=scene,
        config={**SMALL, "media_dir": media},
        **params,
    )


def _valid(form: str, out: dict[str, Any]) -> dict[str, Any]:
    validator = Draft202012Validator(api_json.schema(form))
    errors = sorted(validator.iter_errors(out), key=lambda e: list(e.path))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
    return out


def _line(path: Path, tag: str) -> int:
    lines = path.read_text().splitlines()
    return next(i for i, line in enumerate(lines, 1) if f"# @{tag}" in line)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_every_method_answers_in_its_form(serve, tmp_path, socket_dir, transport):
    server = serve(transport)
    ready = server.ready
    assert ready["method"] == "ready"
    assert ready["params"]["protocol"] == protocol.PROTOCOL_VERSION
    assert ready["params"]["pid"] == server.process.pid
    assert ready["params"]["transport"] == transport
    if transport == "socket":
        assert Path(ready["params"]["socket"]).parent == socket_dir
    else:
        assert "socket" not in ready["params"]

    assert server.result("ping") == {"generation": None}

    path = _write(tmp_path, "shown.py", SHOWN)
    opened = _open(server, path, "Shown")
    assert opened["generation"] == 1
    record = _valid("record", opened["record"])
    assert record["scene_class"] == "Shown"
    assert record["generation"] == 1

    inspected = server.result("inspect")
    assert inspected["generation"] == 1
    assert _valid("record", inspected["record"]) == record

    at = server.result("state_at", t=0.5)
    assert at["generation"] == 1
    state = _valid("state", at["state"])
    assert state["t_exact"] == "1/2"
    assert server.result("state_at", t="1/2")["state"] == state
    texts = [node for node in state["nodes"] if node["label_or_text"] == "hi"]
    assert texts

    linted = server.result("lint", key_frames_dir=str(tmp_path / "lint-keys"))
    assert linted["generation"] == 1
    _valid("lint_result", linted["lint_result"])
    assert linted["lint_result"]["report"]["pass"] is True
    assert _valid("key_frames", linted["key_frames"])["frames"]

    asked = server.result(
        "query", t=0.5, what=["bbox", "overlaps", "offscreen"], ids=[texts[0]["id"]]
    )
    assert asked["generation"] == 1
    _valid("query", asked["query"])

    png = tmp_path / "frames" / "half.png"
    drawn = server.result("frame", t=0.5, path=str(png))
    assert drawn["generation"] == 1
    frame = _valid("frame", drawn["frame"])
    assert frame["path"] == str(png)
    assert frame["t_exact"] == "1/2"
    with Image.open(png) as image:
        assert image.size == (SMALL["pixel_width"], SMALL["pixel_height"])

    keys = server.result("key_frames", out_dir=str(tmp_path / "keys"))
    assert keys["generation"] == 1
    assert _valid("key_frames", keys["key_frames"])["frames"]

    assert server.result("reload") == {"generation": 1, "reloaded": False}
    assert server.result("reload", force=True) == {"generation": 2, "reloaded": True}
    assert server.result("inspect")["record"]["generation"] == 2

    assert server.result("close") == {"generation": None}
    assert server.error("inspect")["code"] == protocol.NO_SESSION

    assert server.result("shutdown") == {"generation": None}
    assert server.exited() == 0
    if transport == "socket":
        assert not Path(ready["params"]["socket"]).exists()
    # The scene's own output never reaches the protocol's stream.
    assert "the scene's own output" in server.stderr()


def test_a_finding_never_ends_the_process(serve, tmp_path):
    server = serve()
    _open(server, _write(tmp_path, "edge.py", EDGE), "Edge")
    report = server.result("lint")["lint_result"]["report"]
    assert report["pass"] is False
    assert report["violations"]
    assert server.result("ping") == {"generation": 1}
    assert server.result("inspect")["generation"] == 1
    assert server.process.poll() is None


def test_an_edit_between_two_lints_executes_again(serve, tmp_path):
    server = serve()
    path = _write(tmp_path, "shown.py", SHOWN)
    _open(server, path, "Shown")
    first = server.result("lint")
    assert first["generation"] == 1
    assert server.result("inspect")["record"]["duration"] == 1.5
    # The same size, within the same second: only the digests tell.
    stat_before = path.stat()
    path.write_text(path.read_text().replace("run_time=1)", "run_time=2)"))
    os.utime(path, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    assert path.stat().st_size == stat_before.st_size
    second = server.result("lint")
    assert second["generation"] == 2
    record = server.result("inspect")["record"]
    assert (record["duration"], record["generation"]) == (2.5, 2)
    assert record["divergences"] == []
    assert record["stale"] is False


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_a_request_past_its_deadline_ends_the_server(serve, tmp_path, transport):
    server = serve(transport)
    path = _write(tmp_path, "sleeps.py", SLEEPS)
    server.result("ping", deadline_s=5)
    deadline = 1.0
    sent = time.monotonic()
    error = server.error(
        "open",
        file=str(path),
        scene="Sleeps",
        config={**SMALL, "media_dir": str(tmp_path / "media")},
        deadline_s=deadline,
    )
    assert error["code"] == protocol.DEADLINE_EXCEEDED
    assert error["data"] == {"method": "open", "deadline_s": deadline, "ends": True}
    assert server.exited(wait=deadline + 1) == protocol.EXIT_DEADLINE
    assert time.monotonic() - sent <= deadline + 1
    if transport == "socket":
        assert not Path(server.ready["params"]["socket"]).exists()


def test_errors_of_the_protocol_and_refusals_keep_their_codes(serve, tmp_path):
    server = serve()
    assert server.error("no_such_method")["code"] == protocol.METHOD_NOT_FOUND
    server.send("{not json")
    response = server.receive()
    assert response is not None
    assert response["id"] is None
    assert response["error"]["code"] == protocol.PARSE_ERROR
    server.send({"jsonrpc": "2.0", "id": 7, "method": 3})
    response = server.receive()
    assert response is not None
    assert (response["id"], response["error"]["code"]) == (7, protocol.INVALID_REQUEST)
    server.send([{"jsonrpc": "2.0", "id": 8, "method": "ping"}])
    response = server.receive()
    assert response is not None
    assert response["error"]["code"] == protocol.INVALID_REQUEST
    # A notification is not answered: the next answer is the next request's.
    server.send({"jsonrpc": "2.0", "method": "ping"})
    assert server.error("inspect")["code"] == protocol.NO_SESSION
    assert server.error("ping", deadline_s=-1)["code"] == protocol.INVALID_PARAMS

    path = _write(tmp_path, "shown.py", SHOWN)
    _open(server, path, "Shown")
    assert server.error("state_at")["code"] == protocol.INVALID_PARAMS
    assert server.error("state_at", t=0.5, when=1)["code"] == protocol.INVALID_PARAMS
    assert server.error("state_at", t="half")["code"] == protocol.INVALID_PARAMS
    other = server.error("open", file=str(path), scene="Other")
    assert other["code"] == protocol.OTHER_SESSION
    # The same scene again is the session the server holds.
    assert _open(server, path, "Shown")["generation"] == 1
    png = str(tmp_path / "f.png")
    for method, params in [
        ("frame", {"t": 0.5, "path": png, "quality": "x"}),
        ("key_frames", {"out_dir": str(tmp_path / "k"), "quality": "x"}),
        ("lint", {"exempt": [""]}),
        ("lint", {"at": "sometimes"}),
        ("lint", {"rules": [], "sample_fps": 8}),
        ("query", {"t": 0.5, "what": "bbox"}),
        # Neither a ZeroDivisionError nor an unhashable rule is the server's.
        ("state_at", {"t": "1/0"}),
        ("state_at", {"t": "1e999"}),
        ("lint", {"rules": [{}]}),
        ("query", {"t": 0.5, "what": [{}]}),
        ("lint", {"t_range": [1, 0.5]}),
        ("lint", {"exempt": [{}]}),
    ]:
        error = server.error(method, **params)
        assert error["code"] == protocol.INVALID_PARAMS, (method, params, error)
    assert server.result("ping") == {"generation": 1}


def test_a_raising_scene_answers_its_explained_error(serve, tmp_path):
    server = serve()
    path = _write(tmp_path, "raises.py", RAISES)
    response = server.call(
        "open",
        file=str(path),
        scene="Raises",
        config={**SMALL, "media_dir": str(tmp_path / "media")},
    )
    error = response["error"]
    assert error["code"] == protocol.SCENE_ERROR
    found = _valid("scene_error", error["data"])
    assert (found["cls"], found["message"]) == ("ValueError", "no such value")
    assert found["line"] == _line(path, "raise")
    assert (found["t"], found["t_exact"]) == (1.0, "1/1")
    assert 'raise ValueError("no such value")' in found["report"]
    assert "ValueError: no such value" in found["report"]
    assert error["message"] == "ValueError: no such value"
    assert (found["raised_by"], found["raised_in"]) == ("scene", None)
    # Nothing opened, and the server goes on.
    assert server.result("ping") == {"generation": None}

    # Raised inside the engine, through the scene's code.
    path = _write(tmp_path, "nothing.py", ENGINE_RAISES)
    found = server.error("open", file=str(path), scene="PlaysNothing", config=SMALL)[
        "data"
    ]
    assert (found["cls"], found["line"]) == ("ValueError", _line(path, "play"))
    assert found["raised_by"] == "engine"
    assert found["raised_in"]["path"].startswith("arachne/")


def test_an_idle_server_ends_itself(serve):
    server = serve("socket", ARACHNE_SESSION_DIR="", ARACHNE_SESSION_IDLE_S="1")
    found = Path(server.ready["params"]["socket"])
    # No directory named: a private one of its own.
    assert stat.S_IMODE(found.parent.stat().st_mode) == 0o700
    assert server.exited() == 0
    assert not found.parent.exists()


def test_a_watchdog_ends_only_what_overran_its_deadline():
    fired: list[tuple[str, bool]] = []
    done = threading.Event()

    def expired(name: str) -> Any:
        def end(respond: bool) -> None:
            fired.append((name, respond))
            done.set()

        return end

    watchdog = Watchdog()
    ended = watchdog.arm(time.monotonic() + 0.05, expired("ended"))
    watchdog.end(ended)
    # Far past the longest wait a thread can make: waited out in parts, and
    # the watchdog keeps the next deadline.
    far = watchdog.arm(time.monotonic() + 1e20, expired("far"))
    time.sleep(0.1)
    watchdog.end(far)
    watchdog.arm(time.monotonic() + 0.05, expired("next"))
    assert done.wait(WAIT_S)
    time.sleep(0.2)
    assert fired == [("next", True)]


def test_a_request_ends_answered_or_expired_never_both():
    watchdog = Watchdog()
    calls: list[bool] = []
    # Answered first: the expiry that comes after sends nothing.
    ticket = watchdog.arm(time.monotonic() + WAIT_S, calls.append)
    assert watchdog.answering(ticket)
    watchdog.expire(ticket)
    assert calls == [False]

    # Expired first: the answer is the expiry's, and the request's is not.
    inside, release = threading.Event(), threading.Event()

    def expired(respond: bool) -> None:
        calls.append(respond)
        inside.set()
        release.wait(WAIT_S)

    ticket = watchdog.arm(time.monotonic() + WAIT_S, expired)
    expiring = threading.Thread(target=watchdog.expire, args=(ticket,))
    expiring.start()
    assert inside.wait(WAIT_S)
    assert not watchdog.answering(ticket)
    release.set()
    expiring.join(WAIT_S)
    assert calls == [False, True]

    # Ended: nothing expires it.
    ticket = watchdog.arm(time.monotonic() + WAIT_S, calls.append)
    assert watchdog.answering(ticket)
    watchdog.end(ticket)
    watchdog.expire(ticket)
    assert calls == [False, True]


def test_a_request_is_kept_by_its_own_deadline_from_the_start():
    from types import SimpleNamespace

    from arachne.server.serve import _Server

    # The server's own deadline is the shorter: a request giving its own is
    # never kept by the server's, not even while its line is read.
    server = _Server(600, 1.0, lambda: None)
    armed: list[float] = []
    arm = server.watchdog.arm

    def recorded(at: float, expired: Any) -> Any:
        armed.append(at)
        return arm(at, expired)

    server.watchdog.arm = recorded  # type: ignore[method-assign]
    received = time.monotonic()
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {"deadline_s": 30},
    }
    sent: list[bytes] = []
    line = SimpleNamespace(data=protocol.encode(request), received=received)
    server.answer(SimpleNamespace(send=sent.append), line)  # type: ignore[arg-type]
    assert armed == [received + 30]
    assert json.loads(sent[0])["result"] == {"generation": None}


def _answered_aside(server: Any, data: bytes | None, received: float) -> list[bytes]:
    """What ``server`` sends for the line ``data``, read at ``received``,
    answered on a thread of its own: one that may wait for an end that a
    test stands in for.
    """
    from types import SimpleNamespace

    sent: list[bytes] = []
    channel = SimpleNamespace(send=sent.append)
    line = SimpleNamespace(data=data, received=received)
    answering = threading.Thread(
        target=server.answer, args=(channel, line), daemon=True
    )
    answering.start()
    answering.join(WAIT_S)
    assert not answering.is_alive(), "the line was held for the server's end"
    return sent


def test_a_line_that_runs_nothing_is_answered_however_long_it_waited():
    from arachne.server.serve import _Server

    server = _Server(600, 1.0, lambda: None)
    ended: list[bool] = []
    server._expired = lambda channel, read, limit, respond: ended.append(respond)  # type: ignore[method-assign]
    # Every line waited past the server's deadline of 1 s behind another.
    waited = time.monotonic() - 10
    lines = [
        None,  # past the longest message
        b"{not json",
        protocol.encode(
            {
                "jsonrpc": "2.0",
                "id": "p",
                "method": "ping",
                "params": {"deadline_s": "soon"},
            }
        ),
        protocol.encode({"jsonrpc": "2.0", "method": "shutdown"}),
        protocol.encode({"jsonrpc": "2.0", "id": 7, "method": "ping"}),
    ]
    answers = [
        json.loads(data)
        for line in lines
        for data in _answered_aside(server, line, waited)
    ]
    assert ended == []
    assert [(answer["id"], answer["error"]["code"]) for answer in answers] == [
        (None, protocol.INVALID_REQUEST),
        (None, protocol.PARSE_ERROR),
        ("p", protocol.INVALID_PARAMS),
        (7, protocol.DEADLINE_EXCEEDED),
    ]
    # Past its deadline before its turn: answered so, and never run, nor the
    # notification before it.
    assert answers[-1]["error"]["data"] == {
        "method": "ping",
        "deadline_s": 1.0,
        "ends": False,
    }
    assert not server.dispatcher.ending


def test_an_unrun_request_with_a_tiny_deadline_is_answered_that_it_did_not_run():
    from arachne.server.serve import _Server

    server = _Server(600, 1.0, lambda: None)
    ended: list[bool] = []
    server._expired = lambda channel, read, limit, respond: ended.append(respond)  # type: ignore[method-assign]
    # Under a millisecond: gone before its answer could be written in it.
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "ping",
        "params": {"deadline_s": 1e-9},
    }
    (sent,) = _answered_aside(server, protocol.encode(request), time.monotonic() - 1)
    assert ended == []
    error = json.loads(sent)["error"]
    assert error["code"] == protocol.DEADLINE_EXCEEDED
    assert error["data"]["ends"] is False


def test_a_request_expired_as_it_ran_holds_the_server_for_its_end():
    from types import SimpleNamespace

    from arachne.server.serve import _Server

    server = _Server(600, 60.0, lambda: None)
    expired = threading.Event()
    # The server's end, stood in for: it would exit the process.
    server._expired = lambda channel, read, limit, respond: expired.set()  # type: ignore[method-assign]

    def handle(method: str, params: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.3)
        return {"generation": None}

    server.dispatcher.handle = handle  # type: ignore[method-assign]
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ping",
        "params": {"deadline_s": 0.1},
    }
    line = SimpleNamespace(data=protocol.encode(request), received=time.monotonic())
    sent: list[bytes] = []
    channel = SimpleNamespace(send=sent.append)
    answering = threading.Thread(
        target=server.answer, args=(channel, line), daemon=True
    )
    answering.start()
    assert expired.wait(WAIT_S)
    # Its answer is the deadline's, and the thread that answers requests takes
    # no next one before the process ends.
    answering.join(1.0)
    assert answering.is_alive()
    assert sent == []


def test_a_listener_out_of_descriptors_waits_rather_than_spins(monkeypatch):
    import errno
    import queue
    import selectors

    from arachne.server import serve

    directory = tempfile.mkdtemp(prefix="al-", dir="/tmp")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(os.path.join(directory, "s"))
    listener.listen()
    # A connection waiting to be accepted, which the process has no descriptor
    # left for.
    waiting = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    waiting.connect(os.path.join(directory, "s"))
    accepts: list[float] = []

    def refused(found: socket.socket) -> Any:
        accepts.append(time.monotonic())
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(serve, "_accept", refused)
    mine, theirs = socket.socketpair()
    channel = serve._Channel(
        mine.fileno(), lambda: mine.recv(65536), mine.sendall, mine.close
    )
    selector = selectors.DefaultSelector()
    selector.register(listener, selectors.EVENT_READ, None)
    selector.register(mine.fileno(), selectors.EVENT_READ, channel)
    arrivals: queue.SimpleQueue[tuple[Any, Any]] = queue.SimpleQueue()
    reading = threading.Thread(
        target=serve._reading, args=(selector, arrivals), daemon=True
    )
    reading.start()
    try:
        time.sleep(0.3)
        assert 1 <= len(accepts) <= 2
        # A channel's end frees a descriptor: the listener is tried again at
        # once, its wait cut short.
        tried = len(accepts)
        ended = time.monotonic()
        theirs.close()
        assert arrivals.get(timeout=WAIT_S) == (channel, serve._ENDED)
        assert _wait_for(lambda: len(accepts) > tried, 0.5)
        assert accepts[tried] - ended < 0.5
    finally:
        with contextlib.suppress(KeyError, ValueError):
            selector.unregister(listener)
        selector.close()
        for each in (waiting, listener, mine, theirs):
            each.close()
        shutil.rmtree(directory, ignore_errors=True)


def _wait_for(condition: Any, wait: float) -> bool:
    until = time.monotonic() + wait
    while not condition():
        if time.monotonic() >= until:
            return False
        time.sleep(0.01)
    return True


def test_what_a_request_publishes_never_holds_its_deadline_back():
    watchdog = Watchdog()
    fired: list[bool] = []
    expired, publishing, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def end(respond: bool) -> None:
        fired.append(respond)
        expired.set()

    ticket = watchdog.arm(time.monotonic() + 0.2, end)

    def publish() -> None:
        publishing.set()
        release.wait(WAIT_S)

    claiming = threading.Thread(target=watchdog.claim, args=(ticket, publish))
    claiming.start()
    try:
        assert publishing.wait(WAIT_S)
        # A publication that overruns the deadline: the watchdog ends the
        # request all the same, and without the deadline's error, the
        # request having claimed its answer.
        assert expired.wait(1.0)
        assert fired == [False]
    finally:
        release.set()
        claiming.join(WAIT_S)
        watchdog.end(ticket)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_a_scene_that_exits_never_ends_the_server(serve, tmp_path, transport):
    server = serve(transport)
    path = _write(tmp_path, "exits.py", EXITS)
    response = server.call(
        "open",
        file=str(path),
        scene="Exits",
        config={**SMALL, "media_dir": str(tmp_path / "media")},
    )
    error = response["error"]
    assert error["code"] == protocol.SCENE_ERROR
    found = _valid("scene_error", error["data"])
    assert (found["cls"], found["line"]) == ("SystemExit", _line(path, "exit"))
    assert found["t_exact"] == "1/2"
    assert server.result("ping") == {"generation": None}

    # Reloaded into an exit: answered, and the server goes on.
    shown = _write(tmp_path, "shown.py", SHOWN)
    _open(server, shown, "Shown")
    shown.write_text(
        shown.read_text().replace("self.wait(0.5)", "raise SystemExit(5)  # @exit")
    )
    error = server.error("lint")
    assert error["code"] == protocol.SCENE_ERROR
    assert error["data"]["cls"] == "SystemExit"
    assert error["data"]["line"] == _line(shown, "exit")
    assert server.result("ping") == {"generation": 1}
    assert server.result("close") == {"generation": None}
    again = _write(tmp_path, "again.py", SHOWN)
    assert _open(server, again, "Shown")["generation"] == 1
    assert server.process.poll() is None


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_a_client_that_stops_reading_cannot_hold_the_server(serve, tmp_path, transport):
    server = serve(transport)
    _open(server, _write(tmp_path, "many.py", MANY), "Many")
    deadline = 2.0
    # Answers far past what the stream holds, none of them read.
    for _ in range(100):
        server.ask("inspect", deadline_s=deadline)
    sent = time.monotonic()
    assert server.exited(wait=deadline + 3) == protocol.EXIT_DEADLINE
    assert time.monotonic() - sent <= deadline + 2


@pytest.mark.parametrize("how", ["env", "flag"])
def test_a_request_without_a_deadline_has_the_server_s(serve, tmp_path, how):
    if how == "env":
        server = serve("stdio", ARACHNE_SESSION_DEADLINE_S="1")
    else:
        server = serve("stdio", "--deadline-s", "1")
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    # A deadline the request gives is its own, longer or shorter.
    naps = _write(tmp_path, "naps.py", NAPS)
    opened = server.result(
        "open", file=str(naps), scene="Naps", config=config, deadline_s=30
    )
    assert opened["generation"] == 1
    assert server.result("close") == {"generation": None}
    sleeps = _write(tmp_path, "sleeps.py", SLEEPS)
    error = server.error("open", file=str(sleeps), scene="Sleeps", config=config)
    assert error["code"] == protocol.DEADLINE_EXCEEDED
    # It ran past its deadline: the server ends.
    assert error["data"] == {"method": "open", "deadline_s": 1.0, "ends": True}
    assert server.exited(wait=2) == protocol.EXIT_DEADLINE


def test_lines_queued_past_their_deadline_are_answered_and_the_server_goes_on(
    serve, tmp_path
):
    server = serve("stdio", "--deadline-s", "1")
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    naps = _write(tmp_path, "naps.py", NAPS)
    # The open naps past the server's deadline of every line queued behind it.
    opening = server.ask(
        "open", file=str(naps), scene="Naps", config=config, deadline_s=30
    )
    server.send(b"{not json")
    server.send(
        {
            "jsonrpc": "2.0",
            "id": "p",
            "method": "ping",
            "params": {"deadline_s": "soon"},
        }
    )
    server.send({"jsonrpc": "2.0", "method": "shutdown"})
    expired = server.ask("ping")
    then = server.ask("ping", deadline_s=30)
    answers = [server.receive() for _ in range(5)]
    assert answers[0]["id"] == opening
    assert answers[0]["result"]["generation"] == 1
    # Lines that run nothing are answered as ever, never ended for their wait.
    assert (answers[1]["id"], answers[1]["error"]["code"]) == (
        None,
        protocol.PARSE_ERROR,
    )
    assert (answers[2]["id"], answers[2]["error"]["code"]) == (
        "p",
        protocol.INVALID_PARAMS,
    )
    # A request past its deadline before its turn is answered so, unrun, and
    # the server goes on; the notification behind it never ran either.
    assert answers[3]["id"] == expired
    assert answers[3]["error"]["code"] == protocol.DEADLINE_EXCEEDED
    assert answers[3]["error"]["data"] == {
        "method": "ping",
        "deadline_s": 1.0,
        "ends": False,
    }
    assert answers[4] == {"jsonrpc": "2.0", "id": then, "result": {"generation": 1}}
    assert server.process.poll() is None


def test_a_refusal_keeps_its_type_and_a_failure_is_internal(serve, tmp_path):
    server = serve()
    refused = server.error(
        "open", file=str(HOOK_SCENES), scene="UpdateMobjectsTwice", config=SMALL
    )
    assert refused["code"] == protocol.REFUSED
    assert refused["data"]["type"] == "SceneHookRefused"
    assert refused["data"]["reason"] == "hook.update_mobjects.twice"
    assert refused["data"]["message"]

    _open(server, _write(tmp_path, "shown.py", SHOWN), "Shown")
    taken = tmp_path / "taken"
    taken.write_text("a file, not a directory")
    failed = server.error("frame", t=0.5, path=str(taken / "frame.png"))
    assert failed["code"] == protocol.INTERNAL_ERROR
    data = failed["data"]
    assert data["type"] in {"FileExistsError", "NotADirectoryError"}
    assert data["message"]
    assert data["traceback"].startswith("Traceback")
    assert data["traceback"].rstrip().endswith(f"{data['type']}: {data['message']}")
    assert data["traceback"].count('  File "') <= protocol.TRACEBACK_FRAMES
    assert server.result("ping") == {"generation": 1}


def test_an_answer_that_is_no_json_is_an_internal_error_with_no_bad_id():
    from arachne.server.serve import _encoded

    sent = json.loads(_encoded({"jsonrpc": "2.0", "id": object(), "result": {}}))
    assert sent["id"] is None
    error = sent["error"]
    assert error["code"] == protocol.INTERNAL_ERROR
    assert error["data"]["type"] == "TypeError"
    assert error["data"]["message"]
    assert error["data"]["traceback"].startswith("Traceback")
    sent = json.loads(_encoded({"jsonrpc": "2.0", "id": 4, "result": {"t": math.nan}}))
    assert (sent["id"], sent["error"]["code"]) == (4, protocol.INTERNAL_ERROR)
    assert sent["error"]["data"]["type"] == "ValueError"


def test_a_line_too_deep_to_read_is_no_json(monkeypatch):
    def too_deep(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError("maximum recursion depth exceeded while decoding")

    monkeypatch.setattr(protocol.json, "loads", too_deep)
    with pytest.raises(protocol.RpcError) as info:
        protocol.parse(b'{"jsonrpc": "2.0", "id": 1, "method": "ping"}')
    assert info.value.code == protocol.PARSE_ERROR


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_malformed_lines_are_answered_and_the_server_goes_on(serve, transport):
    server = serve(transport)
    depth = 100_000
    parse, invalid = protocol.PARSE_ERROR, protocol.INVALID_REQUEST
    for line, codes in [
        # Too deep for the decoder on some interpreters (3.12, 3.13), a list
        # that is no request on others (3.14): an error either way.
        ("[" * depth + "]" * depth, {parse, invalid}),
        ('{"jsonrpc": "2.0", "id": 1, "method": "ping", "x": NaN}', {parse}),
        ('{"jsonrpc": "2.0", "id": Infinity, "method": "ping"}', {parse}),
        ('{"jsonrpc": "2.0", "id": 1e999, "method": "ping"}', {invalid}),
        (b'{"jsonrpc": "2.0", "id": 1, "method": "\xff"}', {parse}),
    ]:
        server.send(line)
        response = server.receive()
        assert response is not None, server.stderr()
        assert response["id"] is None, line[:40]
        assert response["error"]["code"] in codes, line[:40]
        assert server.result("ping") == {"generation": None}
    huge = "1" + "0" * 400
    for deadline in (huge, "1e20", "-0.0"):
        server.send(
            f'{{"jsonrpc": "2.0", "id": "d", "method": "ping",'
            f' "params": {{"deadline_s": {deadline}}}}}'
        )
        response = server.receive()
        assert response is not None, server.stderr()
        assert response["id"] == "d"
        assert response["error"]["code"] == protocol.INVALID_PARAMS, deadline
        assert server.result("ping") == {"generation": None}
    # A notification is never answered, whatever is wrong with it.
    for notification in [
        {"jsonrpc": "2.0", "method": "ping", "params": [1]},
        {"jsonrpc": "2.0", "method": "ping", "params": {"deadline_s": "soon"}},
        {"jsonrpc": "2.0", "method": "state_at", "params": {}},
        {"jsonrpc": "2.0", "method": "nothing"},
        # A bad or missing jsonrpc member: still a notification, unanswered
        # and unrun.
        {"method": "ping"},
        {"jsonrpc": "1.0", "method": "shutdown"},
    ]:
        server.send(notification)
    assert server.result("ping") == {"generation": None}
    assert server.process.poll() is None
    # A param given as null is not given, whether the method knows it or not.
    assert server.result("ping", when=None) == {"generation": None}


def test_a_line_past_the_largest_message_is_refused_and_dropped(serve):
    server = serve()
    server.send(
        b'{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"x": "'
        + b"x" * protocol.MAX_MESSAGE_BYTES
        + b'"}}'
    )
    response = server.receive()
    assert response is not None, server.stderr()
    assert (response["id"], response["error"]["code"]) == (
        None,
        protocol.INVALID_REQUEST,
    )
    assert str(protocol.MAX_MESSAGE_BYTES) in response["error"]["message"]
    assert server.result("ping") == {"generation": None}


@pytest.mark.parametrize("whose", ["its own", "the server's"])
def test_a_queued_request_s_clock_starts_when_it_arrives(serve, tmp_path, whose):
    if whose == "its own":
        server, params = serve(), {"deadline_s": 1}
    else:
        server, params = serve("stdio", "--deadline-s", "1"), {}
    naps = _write(tmp_path, "naps.py", NAPS)
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    opening = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "open",
        "params": {
            "file": str(naps),
            "scene": "Naps",
            "config": config,
            "deadline_s": 30,
        },
    }
    queued = {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": params}
    # One write: the ping waits out the open, past its deadline.
    server.send(json.dumps(opening) + "\n" + json.dumps(queued))
    first = server.receive()
    assert first is not None
    assert (first["id"], "result" in first) == (1, True)
    second = server.receive()
    assert second is not None, server.stderr()
    assert second["id"] == 2
    assert second["error"]["code"] == protocol.DEADLINE_EXCEEDED
    # Unrun, and the server goes on.
    assert second["error"]["data"] == {
        "method": "ping",
        "deadline_s": 1.0,
        "ends": False,
    }
    assert server.result("ping", deadline_s=30) == {"generation": 1}


def test_a_deadline_exit_outlives_a_cleanup_that_fails(serve, tmp_path, socket_dir):
    server = serve("socket")
    socket_dir.chmod(0o500)
    path = _write(tmp_path, "sleeps.py", SLEEPS)
    server.ask(
        "open",
        file=str(path),
        scene="Sleeps",
        config={**SMALL, "media_dir": str(tmp_path / "media")},
        deadline_s=1,
    )
    sent = time.monotonic()
    assert server.exited(wait=WAIT_S) == protocol.EXIT_DEADLINE
    assert time.monotonic() - sent <= 1 + protocol.ENDING_S + 2


@pytest.mark.parametrize("ending", ["idle", "eof"])
def test_an_ending_server_never_waits_on_its_standard_error(serve, tmp_path, ending):
    server = serve("stdio", unread_stderr=True, ARACHNE_SESSION_IDLE_S="4")
    path = _write(tmp_path, "floods.py", FLOODS)
    opened = server.result(
        "open",
        file=str(path),
        scene="Floods",
        config={**SMALL, "media_dir": str(tmp_path / "media"), "verbosity": "ERROR"},
    )
    assert opened["generation"] == 1
    # Past the flood's start: standard error is full from here on.
    time.sleep(3)
    if ending == "eof":
        assert server.process.stdin is not None
        server.process.stdin.close()
    began = time.monotonic()
    assert server.exited(wait=WAIT_S) == 0
    assert time.monotonic() - began <= 1 + 2 * protocol.ENDING_S + 2


def test_a_forward_run_s_error_carries_its_clock(serve, tmp_path):
    server = serve()
    path = _write(tmp_path, "forward.py", FORWARD_RAISES)
    _open(server, path, "ForwardRaises")
    (tmp_path / "raise-now").write_text("")
    error = server.error("frame", t=0.75, path=str(tmp_path / "f.png"))
    assert error["code"] == protocol.SCENE_ERROR
    found = _valid("scene_error", error["data"])
    assert (found["cls"], found["line"]) == ("RuntimeError", _line(path, "raise"))
    assert found["t_exact"] == "1/2"
    assert found["raised_by"] == "scene"
    # It was the forward run's: the session is what it was.
    record = server.result("inspect")["record"]
    assert (record["generation"], record["stale"]) == (1, False)


def test_a_hostile_exception_is_still_explained(serve, tmp_path):
    server = serve()
    path = _write(tmp_path, "hostile.py", HOSTILE)
    error = server.error(
        "open",
        file=str(path),
        scene="RaisesHostile",
        config={**SMALL, "media_dir": str(tmp_path / "media")},
    )
    assert error["code"] == protocol.SCENE_ERROR
    found = _valid("scene_error", error["data"])
    assert (found["cls"], found["line"]) == ("Hostile", _line(path, "raise"))
    assert found["t_exact"] == "1/2"
    assert found["message"]
    assert "Hostile" in found["report"]
    assert server.result("ping") == {"generation": None}


def test_a_deadline_past_the_longest_is_refused_at_start(serve):
    with pytest.raises(AssertionError, match="deadline"):
        serve("stdio", "--deadline-s", "1e20")
