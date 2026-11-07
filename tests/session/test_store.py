"""The store of answers: what a session answers from it is what executing the
scene would answer now. It is kept under the execution as it was asked for
(the file, the scene, the configuration, the working directory) and under
the engine that answered; a session opened from it executes once a file its
execution read, at any time, holds anything else.
"""

from __future__ import annotations

import importlib.util
import os
import textwrap
from pathlib import Path

import pytest

from arachne.server import store
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import RpcError
from arachne.server.store import Store

SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}

#: A square as large as the working directory's ``size.txt`` says.
READS_HERE = """\
    from pathlib import Path

    from arachne import Scene, Square


    class Input(Scene):
        def construct(self):
            self.add(Square(side_length=float(Path("size.txt").read_text())))
            self.wait(0.5)
"""

#: A square as large as ``size.txt`` beside it said when the module loaded.
READS_AT_LOAD = """\
    from pathlib import Path

    from arachne import Scene, Square

    SIZE = float(Path(__file__).with_name("size.txt").read_text())


    class Input(Scene):
        def construct(self):
            self.add(Square(side_length=SIZE))
            self.wait(0.5)
"""


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text))
    return path


def _answered(params: dict, method: str, asked: dict, *, kept: bool) -> dict:
    """``method``'s answer, ``asked``, of the session ``params`` opens: from a
    server with the store (``kept``) or without it.
    """
    server = Dispatcher(Store()) if kept else Dispatcher()
    try:
        server.handle("open", params)
        return server.handle(method, asked)
    finally:
        server.handle("close", {})


def test_a_session_asked_from_another_directory_is_another_execution(
    tmp_path, monkeypatch
):
    path = _write(tmp_path, "scene.py", READS_HERE)
    here, there = tmp_path / "here", tmp_path / "there"
    for directory, size in ((here, "1"), (there, "2")):
        directory.mkdir()
        (directory / "size.txt").write_text(size)
    params = {"file": str(path), "scene": "Input", "config": SMALL}
    monkeypatch.chdir(here)
    _answered(params, "state_at", {"t": 0}, kept=True)

    monkeypatch.chdir(there)
    kept = _answered(params, "state_at", {"t": 0}, kept=True)
    fresh = _answered(params, "state_at", {"t": 0}, kept=False)
    assert kept == fresh


def test_a_file_read_as_the_module_loaded_is_digested_for_freshness(tmp_path):
    path = _write(tmp_path, "scene.py", READS_AT_LOAD)
    size = tmp_path / "size.txt"
    size.write_text("1")
    params = {"file": str(path), "scene": "Input", "config": SMALL}
    _answered(params, "state_at", {"t": 0}, kept=True)

    size.write_text("2")
    kept = _answered(params, "state_at", {"t": 0}, kept=True)
    fresh = _answered(params, "state_at", {"t": 0}, kept=False)
    assert kept == fresh


def _opened_and_closed(params: dict) -> None:
    """Open the session from the store and close it: its manifest kept anew."""
    server = Dispatcher(Store())
    server.handle("open", params)
    server.handle("close", {})


def test_an_answer_kept_before_a_load_time_file_changed_is_never_given_after(
    tmp_path,
):
    path = _write(tmp_path, "scene.py", READS_AT_LOAD)
    size = tmp_path / "size.txt"
    size.write_text("1")
    params = {"file": str(path), "scene": "Input", "config": SMALL}
    _answered(params, "state_at", {"t": 0}, kept=True)

    size.write_text("2")
    # An open executes the edit and keeps its manifest; its execution's
    # identity is the first one's, since load-time reads are not in it.
    _opened_and_closed(params)
    kept = _answered(params, "state_at", {"t": 0}, kept=True)
    fresh = _answered(params, "state_at", {"t": 0}, kept=False)
    assert kept == fresh


def test_a_render_is_never_checked_against_a_lint_of_other_load_time_data(tmp_path):
    path = _write(tmp_path, "scene.py", READS_AT_LOAD)
    size = tmp_path / "size.txt"
    size.write_text("1")
    params = {"file": str(path), "scene": "Input", "config": SMALL}
    _answered(params, "lint", {}, kept=True)

    size.write_text("2")
    _opened_and_closed(params)
    out = tmp_path / "render.mp4"
    rendered = _answered(params, "render", {"out": str(out)}, kept=True)["render"]
    # Checked against the summary of its own load-time data, kept apart from
    # the first's: one summary for each.
    assert rendered["self_check"]["status"] == "checked"
    assert "divergence" not in rendered
    assert len(list(Path(os.environ["ARACHNE_STORE"]).rglob("summaries/*.json"))) == 2


def test_a_session_held_executes_again_when_a_file_read_at_load_changes(tmp_path):
    path = _write(tmp_path, "scene.py", READS_AT_LOAD)
    size = tmp_path / "size.txt"
    size.write_text("1")
    server = Dispatcher(Store())
    try:
        server.handle("open", {"file": str(path), "scene": "Input", "config": SMALL})
        before = server.handle("state_at", {"t": 0})
        size.write_text("2")
        after = server.handle("state_at", {"t": 0})
    finally:
        server.handle("close", {})
    assert (before["generation"], after["generation"]) == (1, 2)
    assert before["state"] != after["state"]


def _rendered(params: dict, out: Path) -> tuple[bool, dict]:
    """A render of the session ``params`` opens, on a server with the store:
    whether the server opened it from the store, and the render's answer,
    its path aside.
    """
    server = Dispatcher(Store())
    try:
        server.handle("open", params)
        from_store = server.session is None
        rendered = server.handle("render", {"out": str(out)})
    finally:
        server.handle("close", {})
    del rendered["render"]["path"]
    return from_store, rendered


def test_a_render_is_checked_alike_whether_its_session_executed_or_was_stored(
    tmp_path,
):
    path = _write(tmp_path, "scene.py", READS_AT_LOAD)
    (tmp_path / "size.txt").write_text("1")
    params = {"file": str(path), "scene": "Input", "config": SMALL}
    # No lint: the first server executes the session, the second opens it
    # from the store; the render's check is of the files, not of the history.
    executed = _rendered(params, tmp_path / "executed.mp4")
    stored = _rendered(params, tmp_path / "stored.mp4")
    assert (executed[0], stored[0]) == (False, True)
    assert executed[1]["render"]["self_check"] == {
        "status": "checked",
        "reason": None,
        "reference": "own",
    }
    assert stored[1] == executed[1]


class _Killed(BaseException):
    """The process ended where it is raised, as a deadline's exit ends it."""


def test_a_session_ended_before_its_summary_was_kept_leaves_no_manifest(
    tmp_path, monkeypatch
):
    path = _write(tmp_path, "scene.py", READS_AT_LOAD)
    (tmp_path / "size.txt").write_text("1")
    params = {"file": str(path), "scene": "Input", "config": SMALL}

    def killed(*args, **kwargs):
        raise _Killed

    with monkeypatch.context() as patched:
        patched.setattr(Store, "keep_summary", killed)
        with pytest.raises(RpcError, match="_Killed"):
            Dispatcher(Store()).handle("open", params)
    # Nothing a session opened from the store would trust without the summary.
    assert Store().manifest(str(path.resolve()), "Input", SMALL) is None
    executed, rendered = _rendered(params, tmp_path / "render.mp4")
    assert not executed
    assert rendered["render"]["self_check"]["status"] == "checked"


def test_the_store_is_kept_under_the_native_core_too(monkeypatch):
    core = importlib.util.find_spec("arachne_core")
    if core is None:
        assert store._core_identity() == "absent"
    else:
        assert store._core_identity() != "absent"
    found = store.engine_fingerprint()
    store.engine_fingerprint.cache_clear()
    try:
        with monkeypatch.context() as patched:
            patched.setattr(store, "_core_identity", lambda: "another build")
            assert store.engine_fingerprint() != found
    finally:
        store.engine_fingerprint.cache_clear()
    assert store.engine_fingerprint() == found
