"""The host's sessions: a registry of servers, one session each, under
``ARACHNE_SESSION_DIR``; found again rather than started twice; the least
recently used ended when they hold more memory than the host allows; the
entries of servers that died swept; and a session ended answers again from
the store of answers, without executing its scene.
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import time
from pathlib import Path

import pytest

from arachne.server import protocol, registry

from .conftest import WAIT_S, alive, gone, write

SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}

STILL = """\
    from arachne import Scene, Square


    class Still(Scene):
        def construct(self):
            self.add(Square(side_length=1))
            self.wait(0.5)


    class Other(Still):
        pass


    class Third(Still):
        pass
"""

COUNTED = """\
    from pathlib import Path

    from arachne import Scene, Square


    class Counted(Scene):
        \"\"\"A scene that counts its executions in a file beside it.\"\"\"

        def construct(self):
            with open(Path(__file__).with_name("runs.log"), "a") as runs:
                runs.write("run\\n")
            self.add(Square(side_length=1))
            self.wait(0.5)
"""


def _runs(path: Path) -> int:
    log = path.with_name("runs.log")
    return len(log.read_text().splitlines()) if log.exists() else 0


def _wait_until(condition, wait: float = WAIT_S) -> bool:
    until = time.monotonic() + wait
    while not condition():
        if time.monotonic() > until:
            return False
        time.sleep(0.05)
    return True


def test_a_session_is_found_again_never_started_twice(host, tmp_path):
    path = write(tmp_path, "still.py", STILL)
    first = host.connect(path, "Still", SMALL)
    assert first.opened["generation"] == 1
    assert first.opened["record"]["scene_class"] == "Still"
    # The entry names the session before its open is answered, so a caller
    # asking again at once finds it.
    (entry,) = registry.sessions(host.directory)
    assert (entry.pid, entry.key) == (first.pid, first.key)
    again = host.connect(path, "Still", SMALL)
    assert again.pid == first.pid
    assert again.opened["generation"] == 1
    other = host.connect(path, "Other", SMALL)
    assert other.pid != first.pid

    mode = stat.S_IMODE(host.directory.stat().st_mode)
    assert mode == 0o700
    found = {entry.pid: entry for entry in registry.sessions(host.directory)}
    assert set(found) == {first.pid, other.pid}
    entry = found[first.pid]
    assert (entry.file, entry.scene, entry.config) == (
        os.path.realpath(path),
        "Still",
        SMALL,
    )
    assert entry.key == first.key != found[other.pid].key
    assert entry.generation == 1
    assert entry.rss_mb > 0
    assert not entry.busy
    assert Path(entry.socket) == host.directory / f"{first.pid}.sock"


def test_three_sessions_under_a_budget_that_fits_two(host, tmp_path, monkeypatch):
    path = write(tmp_path, "still.py", STILL)
    oldest = host.connect(path, "Still", SMALL)
    kept = host.connect(path, "Other", SMALL)
    sizes = {entry.pid: entry.rss_mb for entry in registry.sessions(host.directory)}
    two = sizes[oldest.pid] + sizes[kept.pid]
    budget = two + min(sizes.values()) / 2
    monkeypatch.setenv("ARACHNE_SESSION_MEMORY_MB", f"{budget:.1f}")

    newest = host.connect(path, "Third", SMALL)

    # The newest server ends the least recently used, by itself, once it
    # has answered: the caller asks for nothing.
    assert gone(oldest.pid)

    def left() -> set[int]:
        return {entry.pid for entry in registry.sessions(host.directory)}

    assert _wait_until(lambda: left() == {kept.pid, newest.pid})
    assert not (host.directory / f"{oldest.pid}.json").exists()
    assert not (host.directory / f"{oldest.pid}.sock").exists()
    assert alive(kept.pid)
    assert kept.call("ping") == {"generation": 1}


def test_a_dead_servers_entry_and_socket_are_swept(host, tmp_path):
    path = write(tmp_path, "still.py", STILL)
    killed = host.connect(path, "Still", SMALL)
    os.kill(killed.pid, signal.SIGKILL)
    assert gone(killed.pid)
    # A kill leaves what the server would have removed.
    assert (host.directory / f"{killed.pid}.json").exists()
    assert (host.directory / f"{killed.pid}.sock").exists()

    assert registry.sessions(host.directory) == []
    assert not (host.directory / f"{killed.pid}.json").exists()
    assert not (host.directory / f"{killed.pid}.sock").exists()

    started = host.connect(path, "Still", SMALL)
    assert started.pid != killed.pid
    assert started.opened["generation"] == 1


def test_a_socket_left_with_no_entry_is_swept(host, tmp_path):
    path = write(tmp_path, "still.py", STILL)
    killed = host.connect(path, "Still", SMALL)
    os.kill(killed.pid, signal.SIGKILL)
    assert gone(killed.pid)
    (host.directory / f"{killed.pid}.json").unlink()
    registry.sessions(host.directory)
    assert not (host.directory / f"{killed.pid}.sock").exists()


def test_session_directories_never_see_each_other(host, tmp_path):
    path = write(tmp_path, "still.py", STILL)
    mine = host.connect(path, "Still", SMALL)
    elsewhere = type(host)(Path(host.directory.parent / (host.directory.name + "b")))
    elsewhere.directory.mkdir(mode=0o700)
    try:
        assert registry.sessions(elsewhere.directory) == []
        theirs = elsewhere.connect(path, "Still", SMALL)
        assert theirs.pid != mine.pid
        assert [e.pid for e in registry.sessions(host.directory)] == [mine.pid]
        assert [e.pid for e in registry.sessions(elsewhere.directory)] == [theirs.pid]
    finally:
        elsewhere.end()
        shutil.rmtree(elsewhere.directory, ignore_errors=True)


def test_a_session_is_of_the_working_directory_it_was_asked_from(
    host, tmp_path, monkeypatch
):
    # Relative paths in a config, and a folder's configuration file, are
    # read from the server's working directory: the caller's.
    path = write(tmp_path, "still.py", STILL)
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    monkeypatch.chdir(here)
    first = host.connect(path, "Still", SMALL)
    assert host.connect(path, "Still", SMALL).pid == first.pid
    monkeypatch.chdir(there)
    elsewhere = host.connect(path, "Still", SMALL)
    assert elsewhere.pid != first.pid
    found = {entry.pid: entry.cwd for entry in registry.sessions(host.directory)}
    assert found == {
        first.pid: str(here.resolve()),
        elsewhere.pid: str(there.resolve()),
    }


def test_a_directory_too_long_for_a_socket_is_refused_before_a_start(tmp_path):
    path = write(tmp_path, "still.py", STILL)
    deep = Path("/tmp") / ("d" * 120)
    with pytest.raises(ValueError, match="socket"):
        registry.connect_or_start(path, "Still", SMALL, directory=deep)
    assert not deep.exists()


def test_stop_ends_one_session_and_stop_all_the_rest(host, tmp_path):
    path = write(tmp_path, "still.py", STILL)
    one = host.connect(path, "Still", SMALL)
    two = host.connect(path, "Other", SMALL)
    three = host.connect(path, "Third", SMALL)

    stopped = registry.stop(one.key, host.directory)
    assert [entry.pid for entry in stopped] == [one.pid]
    assert gone(one.pid)
    stopped = registry.stop(two.pid, host.directory)
    assert [entry.pid for entry in stopped] == [two.pid]
    assert gone(two.pid)
    assert registry.stop("no-such-session", host.directory) == []

    assert [entry.pid for entry in registry.stop_all(host.directory)] == [three.pid]
    assert gone(three.pid)
    assert registry.sessions(host.directory) == []
    assert not list(host.directory.glob("*.sock"))


def test_a_server_ended_answers_again_from_the_store_without_executing(host, tmp_path):
    path = write(tmp_path, "counted.py", COUNTED)
    first = host.connect(path, "Counted", SMALL)
    assert _runs(path) == 1
    linted = first.call("lint")
    registry.stop(first.key, host.directory)
    assert gone(first.pid)

    again = host.connect(path, "Counted", SMALL)
    assert again.pid != first.pid
    assert again.opened == first.opened
    assert again.call("lint") == linted
    assert again.call("inspect")["record"] == first.opened["record"]
    # Neither the open nor the lint executed the scene.
    assert _runs(path) == 1

    # An edit is executed, as the next generation.
    path.write_text(path.read_text().replace("side_length=1", "side_length=2"))
    edited = again.call("lint")
    assert _runs(path) == 2
    assert edited["generation"] == 2
    assert again.call("inspect")["record"]["generation"] == 2


def test_an_error_answer_reaches_the_caller_and_a_failed_start_is_ended(host, tmp_path):
    path = write(tmp_path, "still.py", STILL)
    with pytest.raises(registry.RemoteError) as raised:
        host.connect(path, "NoSuchScene", SMALL)
    assert raised.value.code == protocol.SCENE_ERROR
    # The server started for it holds nothing, and is not left behind.
    assert registry.sessions(host.directory) == []
    opened = host.connect(path, "Still", SMALL)
    with pytest.raises(registry.RemoteError) as raised:
        opened.call("state_at")
    assert raised.value.code == protocol.INVALID_PARAMS
