"""A session's render or clip drawn by workers (``workers`` above 1): the
server forks nothing; it runs the render in a process it starts afresh,
which forks the workers, and reads its answer and its self-check as it
reads its own render's.

The scene ``WHERE`` writes, as it renders, the process it ran in, its
parent and every file its process holds open, so that the test sees where
the render ran and that it holds none of the registry's locks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import av
import numpy as np
import pytest

from arachne.server import protocol, registry

from .conftest import gone, write
from .test_render import CHECKED, SMALL, _beside, _valid

WHERE = """\
    import json
    import os
    from pathlib import Path

    from arachne import RIGHT, UP, Dot, Scene, config


    def held():
        found = []
        for fd in range(1024):
            try:
                st = os.fstat(fd)
            except OSError:
                continue
            found.append([st.st_dev, st.st_ino])
        return found


    class Where(Scene):
        def construct(self):
            dot = Dot()
            self.add(dot)
            self.play(dot.animate.shift(RIGHT), run_time=0.5)
            if not config.dry_run:
                Path(__file__).with_name("where.json").write_text(json.dumps({
                    "pid": os.getpid(), "parent": os.getppid(), "held": held(),
                }))
            self.wait(0.4)
            self.play(dot.animate.shift(UP), run_time=0.5)
"""

FAILING = """\
    from arachne import RIGHT, Dot, Scene, config


    class Failing(Scene):
        def construct(self):
            dot = Dot()
            self.play(dot.animate.shift(RIGHT), run_time=0.5)
            if not config.dry_run:
                raise ValueError("no render of this scene")
            self.wait(0.5)
"""


def _pictures(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def _locks(directory: Path, pid: int) -> set[tuple[int, int]]:
    """The registry's lock files of the server ``pid``, by device and inode."""
    found = set()
    for name in (f"{pid}.alive", f"{pid}.lock"):
        path = directory / name
        if path.exists():
            st = path.stat()
            found.add((st.st_dev, st.st_ino))
    return found


#: The session's configuration, its renders written losslessly (PNG frames
#: in MP4), so that two renders' frames compare exactly.
LOSSLESS = {**SMALL, "video_codec": "png", "pixel_format": "rgb24"}


def test_a_render_with_workers_runs_in_a_process_of_its_own(host, tmp_path):
    path = write(tmp_path, "where.py", WHERE)
    session = host.connect(path, "Where", LOSSLESS)
    where = tmp_path / "where.json"
    one = tmp_path / "one.mp4"
    found = session.call("render", out=str(one), deadline_s=120)
    assert _valid("render", found["render"])["self_check"] == CHECKED
    # One worker: rendered in the server's own process.
    assert json.loads(where.read_text())["pid"] == session.pid

    out = tmp_path / "two.mp4"
    found = session.call("render", out=str(out), workers=2, deadline_s=120)
    rendered = _valid("render", found["render"])
    assert rendered["path"] == str(out)
    assert rendered["self_check"] == CHECKED
    assert "divergence" not in rendered
    assert [p["index"] for p in rendered["timeline"]] == [0, 1, 2]
    ran = json.loads(where.read_text())
    assert ran["pid"] != session.pid
    assert ran["parent"] == session.pid
    # It held none of the files the registry locks.
    locks = _locks(host.directory, session.pid)
    assert locks
    assert not locks & {tuple(each) for each in ran["held"]}
    assert gone(ran["pid"], wait=10)
    # The same frames as the server's own render, and nothing left beside.
    one_frames, two_frames = _pictures(one), _pictures(out)
    assert len(one_frames) == len(two_frames) == 21
    for index, (a, b) in enumerate(zip(one_frames, two_frames, strict=True)):
        assert np.array_equal(a, b), (index, int((a != b).any(axis=-1).sum()))
    assert _beside(out) == []
    # The server is as it was: registered, and answering.
    assert [s.pid for s in registry.sessions(host.directory)] == [session.pid]
    assert session.call("ping") == {"generation": 1}


def test_a_clip_with_workers_takes_the_plays_meeting_its_span(host, tmp_path):
    path = write(tmp_path, "where.py", WHERE)
    session = host.connect(path, "Where", SMALL)
    out = tmp_path / "clip.mp4"
    found = session.call("clip", t0=0.6, t1=1.0, out=str(out), workers=2)
    clip = _valid("render", found["clip"])
    shown = [p["index"] for p in clip["timeline"] if not p["skipped"]]
    assert shown == [1, 2]
    assert clip["self_check"] == CHECKED
    # The 21 frames less play 0's 8.
    assert len(_pictures(out)) == 13
    assert json.loads((tmp_path / "where.json").read_text())["pid"] != session.pid


def test_what_a_scene_raises_rendering_is_answered_alike(host, tmp_path):
    """A scene that raises as it renders is answered as the same error
    whether its render ran in the server or in a process of its own.
    """
    path = write(tmp_path, "failing.py", FAILING)
    session = host.connect(path, "Failing", SMALL)
    out = tmp_path / "failing.mp4"
    answers = []
    for workers in (1, 2):
        with pytest.raises(registry.RemoteError) as raised:
            session.call("render", out=str(out), workers=workers, deadline_s=120)
        assert raised.value.code == protocol.SCENE_ERROR
        answers.append((str(raised.value), raised.value.data))
    assert answers[0] == answers[1]
    error = answers[1][1]
    assert error["cls"] == "ValueError"
    assert error["message"] == "no render of this scene"
    assert error["line"] == 9
    assert error["raised_by"] == "scene"
    assert not out.exists()
    assert _beside(out) == []
    assert session.call("ping") == {"generation": 1}


SLOW_WHERE = """\
    import os
    import time
    from pathlib import Path

    from arachne import Dot, Scene, config


    class SlowWhere(Scene):
        \"\"\"A render that says where it runs, then stalls.\"\"\"

        def construct(self):
            self.add(Dot())
            self.wait(0.5)
            if not config.dry_run:
                Path(__file__).with_name("pid").write_text(str(os.getpid()))
                time.sleep(60)
            self.wait(0.5)
"""


def test_a_render_with_workers_past_its_deadline_ends_its_process(host, tmp_path):
    path = write(tmp_path, "slow.py", SLOW_WHERE)
    session = host.connect(path, "SlowWhere", SMALL)
    out = tmp_path / "slow.mp4"
    with pytest.raises(registry.RemoteError) as raised:
        session.call("render", out=str(out), workers=2, deadline_s=4.0)
    assert raised.value.code == protocol.DEADLINE_EXCEEDED
    assert raised.value.data["ends"] is True
    assert gone(session.pid, wait=5)
    render = int((tmp_path / "pid").read_text())
    assert render != session.pid
    # No process of the render outlives the server.
    assert gone(render, wait=5)
    assert not out.exists()
    assert _beside(out) == []
    assert registry.sessions(host.directory) == []


def test_what_a_render_s_workers_refuse(host, tmp_path):
    path = write(tmp_path, "where.py", WHERE)
    session = host.connect(path, "Where", SMALL)
    out = str(tmp_path / "x.mp4")
    for workers in (0, -1, 1.5, "two", True):
        with pytest.raises(registry.RemoteError) as raised:
            session.call("render", out=out, workers=workers)
        assert raised.value.code == protocol.INVALID_PARAMS, workers
        with pytest.raises(registry.RemoteError) as raised:
            session.call("clip", t0=0, t1=1, out=out, workers=workers)
        assert raised.value.code == protocol.INVALID_PARAMS, workers
    assert not Path(out).exists()
    assert os.path.exists(path)
