"""The session commands: ``arachne inspect``, ``frame``, ``lint`` and
``session``, thin clients of a session server, or, with ``--once``, of a
session in their own process.

``--json`` prints the answer as the server gives it; without it, the
answer is formatted for reading. A finding is an answer: ``lint`` exits 0
with ``pass`` false, and 1 only with ``--strict``. A scene's error exits 2
with its trimmed report; a failure of the engine or the session exits 3.
``arachne SCENE_FILE`` still renders.

A session asked of a server stays resident in the host's registry
(``ARACHNE_SESSION_DIR``), found again by the next command. Each test has a
registry of its own, and every server in it is ended before the test
returns.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from jsonschema import Draft202012Validator
from PIL import Image

from arachne import __version__, api_json
from arachne.__main__ import main
from arachne.cli import client
from arachne.server import protocol, registry

#: How long a test waits for a server, in seconds.
WAIT_S = float(os.environ.get("ARACHNE_TEST_SERVER_WAIT_S", "60"))

SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}
MODES = ("server", "once")

SHOWN = """\
    from arachne import RIGHT, Scene, Square


    class Shown(Scene):
        def construct(self):
            print("the scene's own output")
            square = Square(side_length=1)
            self.add(square)
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

STILL = """\
    from arachne import Scene, Square


    class Still(Scene):
        def construct(self):
            self.add(Square())
            self.wait(0.5)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture(autouse=True)
def session_dir(monkeypatch) -> Iterator[Path]:
    """A registry of the test's own, under /tmp (a socket's whole path must
    fit a Unix socket's address); every server in it ends with the test.
    """
    found = Path(tempfile.mkdtemp(prefix="ac-", dir="/tmp"))
    monkeypatch.setenv("ARACHNE_SESSION_DIR", str(found))
    try:
        yield found
    finally:
        pids = {entry.pid for entry in registry.stop_all(found)}
        pids |= {int(p.stem) for p in found.glob("*.json") if p.stem.isdigit()}
        for pid in pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        shutil.rmtree(found, ignore_errors=True)
        assert not [pid for pid in pids if _alive(pid)], "a server outlived its test"


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text))
    return path


def _config(tmp_path: Path) -> str:
    return json.dumps({**SMALL, "media_dir": str(tmp_path / "media")})


def _invoke(*args: str, env: dict[str, str] | None = None) -> Result:
    return CliRunner().invoke(main, list(args), prog_name="arachne", env=env)


def _session_args(mode: str, tmp_path: Path) -> list[str]:
    once = ["--once"] if mode == "once" else []
    return ["--config", _config(tmp_path), *once]


def _valid(form: str, out: dict[str, Any]) -> dict[str, Any]:
    validator = Draft202012Validator(api_json.schema(form))
    errors = sorted(validator.iter_errors(out), key=lambda e: list(e.path))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
    return out


def _line(path: Path, tag: str) -> int:
    lines = path.read_text().splitlines()
    return next(i for i, line in enumerate(lines, 1) if f"# @{tag}" in line)


def _without(value: Any, name: str) -> Any:
    """``value`` with every key ``name`` dropped, at any depth."""
    if isinstance(value, dict):
        return {k: _without(v, name) for k, v in value.items() if k != name}
    if isinstance(value, list):
        return [_without(each, name) for each in value]
    return value


# inspect


def test_inspect_json_is_the_record_form(tmp_path):
    path = _write(tmp_path, "shown.py", SHOWN)
    found = _invoke(
        "inspect", str(path), "Shown", "--json", *_session_args("server", tmp_path)
    )
    assert found.exit_code == 0, found.output
    # The standard output is the answer and nothing else: no splash, and
    # not the scene's own prints.
    answer = json.loads(found.stdout)
    assert answer["generation"] == 1
    record = _valid("record", answer["record"])
    assert record["scene_class"] == "Shown"
    assert record["duration"] == 1.5


def test_inspect_formats_the_record_for_reading(tmp_path):
    path = _write(tmp_path, "shown.py", SHOWN)
    found = _invoke("inspect", str(path), "Shown", *_session_args("once", tmp_path))
    assert found.exit_code == 0, found.output
    text = found.stdout
    assert "Shown" in text
    assert "1.5 s" in text
    assert "96x54" in text
    # One line a play, each with its kind and the scene line that ran it.
    lines = path.read_text().splitlines()
    play_line = next(i for i, line in enumerate(lines, 1) if "self.play(" in line)
    wait_line = next(i for i, line in enumerate(lines, 1) if "self.wait(" in line)
    assert f"line {play_line}" in text
    assert f"line {wait_line}" in text
    assert "the scene's own output" not in text


# lint


def test_a_finding_is_an_answer_unless_strict(tmp_path):
    path = _write(tmp_path, "edge.py", EDGE)
    args = ["lint", str(path), "Edge", "--json", *_session_args("server", tmp_path)]
    found = _invoke(*args)
    assert found.exit_code == 0, found.output
    lint_result = _valid("lint_result", json.loads(found.stdout)["lint_result"])
    assert lint_result["report"]["pass"] is False
    assert lint_result["report"]["violations"]

    strict = _invoke(*args, "--strict")
    assert strict.exit_code == 1, strict.output
    # The same answer, but for how long the lint took.
    assert _without(json.loads(strict.stdout)["lint_result"], "duration_s") == (
        _without(lint_result, "duration_s")
    )


def test_lint_formats_each_finding_for_reading(tmp_path):
    path = _write(tmp_path, "edge.py", EDGE)
    args = ["lint", str(path), "Edge", *_session_args("once", tmp_path)]
    answer = json.loads(_invoke(*args, "--json").stdout)
    found = _invoke(*args)
    assert found.exit_code == 0, found.output
    header = found.stdout.splitlines()[0]
    assert header.startswith("FAIL")
    # duration_s is how long the lint took, not a span of the scene.
    assert "completed in" in header
    assert " over " not in header
    for violation in answer["lint_result"]["report"]["violations"]:
        assert violation["code"] in found.stdout
        assert violation["message"] in found.stdout
        if violation["fix"]:
            assert violation["fix"] in found.stdout


def test_a_passing_lint_says_so(tmp_path):
    path = _write(tmp_path, "shown.py", SHOWN)
    found = _invoke(
        "lint", str(path), "Shown", "--strict", *_session_args("once", tmp_path)
    )
    assert found.exit_code == 0, found.output
    assert "PASS" in found.stdout


def test_lint_passes_its_options_on(tmp_path):
    path = _write(tmp_path, "edge.py", EDGE)
    args = ["lint", str(path), "Edge", "--json", *_session_args("once", tmp_path)]
    everything = json.loads(_invoke(*args).stdout)["lint_result"]["report"]
    assert everything["violations"]
    # A range before any finding holds keeps none of them.
    early = _invoke(*args, "--t-range", "0,0.1", "--fps", "8")
    assert early.exit_code == 0, early.output
    assert json.loads(early.stdout)["lint_result"]["report"]["violations"] == []
    # Every object a finding names, exempted, leaves nothing.
    ids = sorted({obj["id"] for v in everything["violations"] for obj in v["objects"]})
    exempted = _invoke(*args, "--exempt", json.dumps(ids))
    assert exempted.exit_code == 0, exempted.output
    assert json.loads(exempted.stdout)["lint_result"]["report"]["pass"] is True


def test_a_caller_s_mistake_is_a_usage_error(tmp_path):
    path = _write(tmp_path, "shown.py", SHOWN)
    args = ["lint", str(path), "Shown", *_session_args("once", tmp_path)]
    backwards = _invoke(*args, "--t-range", "1,0")
    assert backwards.exit_code == 2
    assert "t_range" in backwards.stderr
    not_json = _invoke(*args, "--exempt", "[nope")
    assert not_json.exit_code == 2
    assert "--exempt" in not_json.stderr


@pytest.mark.parametrize("mode", MODES)
def test_a_caller_s_mistake_under_json_prints_the_error_too(tmp_path, mode):
    path = _write(tmp_path, "shown.py", SHOWN)
    found = _invoke(
        "lint",
        str(path),
        "Shown",
        "--json",
        "--t-range",
        "1,0",
        *_session_args(mode, tmp_path),
    )
    assert found.exit_code == 2, found.output
    error = json.loads(found.stdout)
    assert error["code"] == protocol.INVALID_PARAMS
    assert "t_range" in error["message"]
    assert "t_range" in found.stderr


#: Command lines no session can be asked: a number that is not finite,
#: as text or as JSON, each refused before any session is reached.
NOT_FINITE = {
    "t-range nan": ["lint", "--t-range", "nan,1"],
    "t-range overflow": ["lint", "--t-range", "0,1e999"],
    "fps nan": ["lint", "--fps", "nan"],
    "fps overflow": ["lint", "--fps", "1e999"],
    "deadline nan": ["lint", "--deadline-s", "nan"],
    "config NaN": ["lint", "--config", '{"frame_rate": NaN}'],
    "config overflow": ["lint", "--config", '{"frame_rate": 1e999}'],
    "config Infinity": ["inspect", "--config", '{"frame_rate": -Infinity}'],
    "exempt NaN": ["lint", "--exempt", "[NaN]"],
    "exempt overflow": ["lint", "--exempt", "[[1e999, 2]]"],
    "t nan": ["frame", "--t", "nan"],
    "t overflow": ["frame", "--t", "1e999"],
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(NOT_FINITE))
def test_a_number_that_is_not_finite_is_a_usage_error(tmp_path, mode, case):
    path = _write(tmp_path, "shown.py", SHOWN)
    command, *options = NOT_FINITE[case]
    once = ["--once"] if mode == "once" else []
    small = ["--config", _config(tmp_path)]
    found = _invoke(command, str(path), "Shown", *small, *options, *once, "--json")
    assert found.exit_code == 2, found.output
    assert found.exception is None or isinstance(found.exception, SystemExit)
    assert "Error" in found.stderr
    assert "Traceback" not in found.output


@pytest.mark.parametrize("mode", MODES)
def test_a_request_that_is_no_json_is_the_caller_s_mistake(tmp_path, mode):
    path = _write(tmp_path, "shown.py", SHOWN)
    with client.reached(
        path,
        "Shown",
        {**SMALL, "media_dir": str(tmp_path / "media")},
        once=mode == "once",
        deadline_s=60,
    ) as (asked, opened):
        assert opened["record"]["scene_class"] == "Shown"
        for value in (float("nan"), float("inf")):
            with pytest.raises(client.Failed) as failed:
                asked.call("state_at", {"t": value, "deadline_s": 60})
            assert failed.value.code == protocol.INVALID_PARAMS
            assert "JSON" in failed.value.error["message"]
        # The session took nothing from it.
        assert asked.call("ping", {"deadline_s": 60})["generation"] == 1


def test_once_closes_its_session_by_the_deadline_asked(tmp_path, monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []
    call = client.InProcess.call

    def recorded(self, method, params):
        calls.append((method, dict(params)))
        return call(self, method, params)

    monkeypatch.setattr(client.InProcess, "call", recorded)
    path = _write(tmp_path, "shown.py", SHOWN)
    found = _invoke(
        "inspect",
        str(path),
        "Shown",
        "--deadline-s",
        "42",
        *_session_args("once", tmp_path),
    )
    assert found.exit_code == 0, found.output
    assert [method for method, _ in calls] == ["open", "close"]
    assert all(params["deadline_s"] == 42 for _, params in calls)


@pytest.mark.parametrize("mode", MODES)
def test_once_and_the_server_answer_alike(tmp_path, mode):
    path = _write(tmp_path, "edge.py", EDGE)
    other = "once" if mode == "server" else "server"
    for command in ("inspect", "lint"):
        mine, theirs = (
            _invoke(command, str(path), "Edge", "--json", *_session_args(m, tmp_path))
            for m in (mode, other)
        )
        assert mine.exit_code == theirs.exit_code == 0, (mine.output, theirs.output)
        assert _without(json.loads(mine.stdout), "duration_s") == _without(
            json.loads(theirs.stdout), "duration_s"
        )


# a scene's error, and the session's


@pytest.mark.parametrize("mode", MODES)
def test_a_raising_scene_exits_2_with_its_trimmed_report(tmp_path, mode):
    path = _write(tmp_path, "raises.py", RAISES)
    found = _invoke(
        "lint", str(path), "Raises", "--json", *_session_args(mode, tmp_path)
    )
    assert found.exit_code == 2, found.output
    error = json.loads(found.stdout)
    assert error["code"] == protocol.SCENE_ERROR
    scene_error = _valid("scene_error", error["data"])
    assert scene_error["line"] == _line(path, "raise")
    assert scene_error["t"] == 1
    # The trimmed report, on the standard error, with or without --json.
    assert scene_error["report"] in found.stderr
    assert "ValueError: no such value" in found.stderr
    assert f"raises.py:{_line(path, 'raise')}" in found.stderr

    plain = _invoke("lint", str(path), "Raises", *_session_args(mode, tmp_path))
    assert plain.exit_code == 2
    assert plain.stdout == ""
    assert scene_error["report"] in plain.stderr


def test_a_request_past_its_deadline_exits_3(tmp_path):
    path = _write(tmp_path, "sleeps.py", SLEEPS)
    found = _invoke(
        "inspect",
        str(path),
        "Sleeps",
        "--deadline-s",
        "1",
        *_session_args("server", tmp_path),
    )
    assert found.exit_code == 3, found.output
    assert "deadline" in found.stderr


# frame


@pytest.mark.parametrize("mode", MODES)
def test_frame_writes_the_png_asked_for(tmp_path, mode):
    path = _write(tmp_path, "shown.py", SHOWN)
    png = tmp_path / "frames" / "half.png"
    found = _invoke(
        "frame",
        str(path),
        "Shown",
        "--t",
        "1/2",
        "-o",
        str(png),
        "--json",
        *_session_args(mode, tmp_path),
    )
    assert found.exit_code == 0, found.output
    frame = _valid("frame", json.loads(found.stdout)["frame"])
    assert frame["t_exact"] == "1/2"
    assert frame["path"] == str(png)
    with Image.open(png) as image:
        assert image.size == (SMALL["pixel_width"], SMALL["pixel_height"])


def test_frame_draws_at_the_quality_asked_for(tmp_path, monkeypatch):
    path = _write(tmp_path, "shown.py", SHOWN)
    monkeypatch.chdir(tmp_path)
    found = _invoke(
        "frame",
        str(path),
        "Shown",
        "--t",
        "1.5",
        "-q",
        "l",
        *_session_args("once", tmp_path),
    )
    assert found.exit_code == 0, found.output
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    assert str(pngs[0]) in found.stdout
    with Image.open(pngs[0]) as image:
        assert image.size == (854, 480)


# session


@contextlib.contextmanager
def _started(
    *args: str, env: dict[str, str]
) -> Iterator[Callable[..., subprocess.Popen[bytes]]]:
    """Start processes, each ``args`` plus what the caller appends; every
    one still running at the end is killed and reaped.
    """
    started: list[subprocess.Popen[bytes]] = []

    def start(*more: str) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            [sys.executable, *args, *more],
            env={**os.environ, **env},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        started.append(process)
        return process

    try:
        yield start
    finally:
        for process in started:
            if process.poll() is None:
                process.send_signal(signal.SIGKILL)
            process.wait(WAIT_S)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()


def _ready(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    assert process.stdout is not None
    line = process.stdout.readline()
    assert line, "the server wrote no ready line"
    ready: dict[str, Any] = json.loads(line)
    assert ready["method"] == "ready"
    return ready


def test_session_serve_speaks_the_protocol_on_its_standard_streams():
    with _started("-m", "arachne", "session", "serve", "--stdio", env={}) as start:
        process = start()
        # The first line is the ready line: no splash before it.
        assert _ready(process)["params"]["transport"] == "stdio"
        assert process.stdin is not None
        assert process.stdout is not None
        for request_id, method in enumerate(("ping", "shutdown"), 1):
            request = {"jsonrpc": "2.0", "id": request_id, "method": method}
            process.stdin.write(json.dumps(request).encode() + b"\n")
            process.stdin.flush()
            answer = json.loads(process.stdout.readline())
            assert answer == {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"generation": None},
            }
        assert process.wait(WAIT_S) == 0


@pytest.mark.parametrize(
    "transports", [[], ["--stdio", "--socket"], ["--socket", "--stdio"]]
)
def test_session_serve_takes_exactly_one_transport(transports):
    # In a process of its own: a serve that took one would serve.
    found = subprocess.run(
        [sys.executable, "-m", "arachne", "session", "serve", *transports],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        env={**os.environ, "ARACHNE_SESSION_IDLE_S": "2"},
    )
    assert found.returncode == 2, found.stderr
    assert "exactly one of --stdio and --socket" in found.stderr


def _listed() -> list[dict[str, Any]]:
    found = _invoke("session", "list", "--json")
    assert found.exit_code == 0, found.output
    listed: list[dict[str, Any]] = json.loads(found.stdout)
    return listed


def test_a_session_stays_resident_and_is_found_again(tmp_path):
    path = _write(tmp_path, "edge.py", EDGE)
    args = [str(path), "Edge", "--json", *_session_args("server", tmp_path)]
    first = _invoke("inspect", *args)
    assert first.exit_code == 0, first.output
    (entry,) = _listed()
    assert entry["scene"] == "Edge"
    assert entry["file"] == os.path.realpath(path)
    assert entry["generation"] == 1
    assert _alive(entry["pid"])
    # The next command, a lint, is answered by the same server.
    linted = _invoke("lint", *args)
    assert linted.exit_code == 0, linted.output
    assert json.loads(linted.stdout)["generation"] == 1
    assert [found["pid"] for found in _listed()] == [entry["pid"]]


def test_session_list_stop_and_stop_all(tmp_path):
    path = _write(
        tmp_path,
        "scenes.py",
        SHOWN + "\n\n" + textwrap.indent("class Again(Shown):\n    pass\n", "    "),
    )
    for scene in ("Shown", "Again"):
        opened = _invoke(
            "inspect", str(path), scene, *_session_args("server", tmp_path)
        )
        assert opened.exit_code == 0, opened.output
    sessions = _listed()
    assert sorted(entry["scene"] for entry in sessions) == ["Again", "Shown"]
    human = _invoke("session", "list")
    for entry in sessions:
        assert entry["key"] in human.stdout
        assert str(entry["pid"]) in human.stdout

    # By key; then by pid.
    first, second = sorted(sessions, key=lambda entry: entry["scene"])
    stopped = _invoke("session", "stop", first["key"])
    assert stopped.exit_code == 0, stopped.output
    assert str(first["pid"]) in stopped.stdout
    assert not _alive(first["pid"])
    assert [entry["pid"] for entry in _listed()] == [second["pid"]]
    assert _invoke("session", "stop", first["key"]).exit_code == 1

    all_stopped = _invoke("session", "stop-all")
    assert all_stopped.exit_code == 0, all_stopped.output
    assert str(second["pid"]) in all_stopped.stdout
    assert not _alive(second["pid"])
    assert _listed() == []
    assert "no sessions" in _invoke("session", "list").stdout


def test_a_session_stopped_by_its_pid(tmp_path):
    path = _write(tmp_path, "shown.py", SHOWN)
    opened = _invoke("inspect", str(path), "Shown", *_session_args("server", tmp_path))
    assert opened.exit_code == 0, opened.output
    (entry,) = _listed()
    assert _invoke("session", "stop", str(entry["pid"])).exit_code == 0
    assert not _alive(entry["pid"])


# the splash


def _splash() -> str:
    return f"Arachne v{__version__}"


def test_the_splash_still_comes_before_an_eager_exit():
    found = _invoke("--show-splash", "--help")
    assert found.exit_code == 0, found.output
    assert found.stdout.startswith(_splash())
    assert "Usage: arachne" in found.stdout
    # Asked after --help, it comes too late, as it always has.
    late = _invoke("--help", "--show-splash")
    assert not late.stdout.startswith(_splash())
    assert "Usage: arachne" in late.stdout


@pytest.mark.parametrize("command", ["inspect", "frame", "lint", "session"])
def test_no_splash_comes_before_the_session_commands(command):
    found = _invoke("--show-splash", command, "--help")
    assert found.exit_code == 0, found.output
    assert _splash() not in found.stdout
    assert f"Usage: arachne {command}" in found.stdout


def test_the_splash_still_comes_before_the_other_commands():
    found = _invoke("cfg", "--help")
    assert found.stdout.startswith(_splash())


# rendering


def test_arachne_scene_file_still_renders(tmp_path):
    path = _write(tmp_path, "still.py", STILL)
    media = tmp_path / "media"
    command = [
        sys.executable,
        "-m",
        "arachne",
        str(path),
        "Still",
        "--resolution",
        "32,18",
        "--frame_rate",
        "4",
        "--media_dir",
        str(media),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    assert "Arachne v" in done.stdout
    assert list(media.rglob("Still.mp4"))
