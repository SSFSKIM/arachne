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

import json
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

    def __init__(self, transport: str, cwd: Path, env: dict[str, str]) -> None:
        self.transport = transport
        # Closed by end(), which every test's fixture calls.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115
        self.process = subprocess.Popen(
            [sys.executable, "-m", "arachne.server", f"--{transport}"],
            cwd=cwd,
            env={**os.environ, "PYTHONPATH": str(ROOT), **env},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
        )
        assert self.process.stdout is not None
        out = self.process.stdout.fileno()
        self._out = _Lines(out, lambda: os.read(out, 65536))
        line = self._out.next()
        assert line is not None, self.stderr()
        self.ready = json.loads(line)
        self._ids = 0
        self._socket: socket.socket | None = None
        if transport == "socket":
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(self.ready["params"]["socket"])
            connection = self._socket
            self._in = _Lines(connection.fileno(), lambda: connection.recv(65536))
        else:
            self._in = self._out

    def send(self, message: Any) -> None:
        """Send ``message``: a JSON value, or text sent as it is."""
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

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        """The response to ``method`` asked with ``params``."""
        self._ids += 1
        self.send(
            {"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params}
        )
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
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace")

    def end(self) -> None:
        if self._socket is not None:
            self._socket.close()
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)
        self.process.wait(WAIT_S)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self._stderr.close()


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    """A directory for sockets, short enough for a socket's path."""
    found = Path(tempfile.mkdtemp(prefix="as-", dir="/tmp"))
    yield found
    shutil.rmtree(found, ignore_errors=True)


@pytest.fixture
def serve(tmp_path: Path, socket_dir: Path) -> Iterator[Any]:
    """Start servers, each ended after the test."""
    started: list[Server] = []

    def start(transport: str = "stdio", **env: str) -> Server:
        env = {"ARACHNE_SESSION_DIR": str(socket_dir), **env}
        env = {name: value for name, value in env.items() if value != ""}
        found = Server(transport, tmp_path, env)
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
    assert error["data"] == {"method": "open", "deadline_s": deadline}
    assert server.exited(wait=deadline + 1) == protocol.EXIT_DEADLINE
    assert time.monotonic() - sent <= deadline + 1
    if transport == "socket":
        assert not Path(server.ready["params"]["socket"]).exists()


def test_errors_of_the_protocol_and_refusals_keep_their_codes(serve, tmp_path):
    server = serve()
    assert server.error("render")["code"] == protocol.METHOD_NOT_FOUND
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
    refused = server.error("frame", t=0.5, path=str(tmp_path / "f.png"), quality="x")
    assert refused["code"] == protocol.REFUSED
    assert refused["data"]["type"] == "ValueError"
    assert "Unknown quality" in refused["data"]["message"]
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
    # Nothing opened, and the server goes on.
    assert server.result("ping") == {"generation": None}


def test_an_idle_server_ends_itself(serve):
    server = serve("socket", ARACHNE_SESSION_DIR="", ARACHNE_SESSION_IDLE_S="1")
    found = Path(server.ready["params"]["socket"])
    # No directory named: a private one of its own.
    assert stat.S_IMODE(found.parent.stat().st_mode) == 0o700
    assert server.exited() == 0
    assert not found.parent.exists()


def test_a_watchdog_ends_only_what_overran_its_deadline():
    fired: list[str] = []
    done = threading.Event()

    def overran(name: str) -> Any:
        def end() -> None:
            fired.append(name)
            done.set()

        return end

    watchdog = Watchdog()
    watchdog.arm(0.05, overran("disarmed"))
    watchdog.disarm()
    watchdog.arm(5, overran("rearmed"))
    watchdog.disarm()
    watchdog.arm(0.05, overran("overran"))
    assert done.wait(WAIT_S)
    time.sleep(0.2)
    assert fired == ["overran"]
