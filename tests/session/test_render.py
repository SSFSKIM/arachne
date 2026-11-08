"""``clip`` and ``render``: a fresh execution of the session's scene through
the render path, written to the file the caller names only once it is whole,
and checked against the summary the store keeps of the session's execution.

Each render here is small: a second or so of a few pixels.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import av
import click
import pytest
from jsonschema import Draft202012Validator

from arachne import api_json
from arachne.cli.asking import EXIT_STRICT
from arachne.cli.session.render import _done, formatted
from arachne.server import protocol, registry
from arachne.server.store import summary_key
from arachne.session import Session

from .conftest import gone, write

#: The session's configuration: a render at it is 96 by 54 at 15 frames a
#: second, as is the store's summary of its lint.
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}

#: What a render checked against a summary like its own says of its check:
#: against its session's own execution, or against the execution a lint in
#: the session judged.
CHECKED = {"status": "checked", "reason": None, "reference": "own"}
LINTED = {"status": "checked", "reason": None, "reference": "lint"}

CLIPPED = """\
    from arachne import RIGHT, Dot, Scene


    class Clipped(Scene):
        def construct(self):
            dot = Dot()
            self.add(dot)
            self.wait(1)
            self.play(dot.animate.shift(RIGHT), run_time=1)
            self.wait(1)
"""

SCATTERED = """\
    import numpy as np

    from arachne import Dot, Scene


    class Scattered(Scene):
        \"\"\"A dot wherever the scene's seed puts it.\"\"\"

        def construct(self):
            self.add(Dot().move_to([*np.random.uniform(-2, 2, 2), 0]))
            self.wait(1)
"""

SLOW = """\
    import time

    from arachne import Dot, Scene, config


    class Slow(Scene):
        \"\"\"A render that stalls once its first second is written; the
        session's own execution, which writes nothing, never does.\"\"\"

        def construct(self):
            self.add(Dot())
            self.wait(0.5)
            if not config.dry_run:
                time.sleep(60)
            self.wait(0.5)
"""


def _valid(form: str, out: dict) -> dict:
    validator = Draft202012Validator(api_json.schema(form))
    errors = sorted(validator.iter_errors(out), key=lambda e: list(e.path))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
    return out


def _frames(path: Path) -> tuple[int, int, int]:
    """The video's frame count, width and height."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        return frames, stream.codec_context.width, stream.codec_context.height


def _beside(out: Path) -> list[Path]:
    """What a render left in ``out``'s directory, ``out`` aside."""
    return sorted(p for p in out.parent.iterdir() if p.name.startswith(f".{out.name}"))


def test_a_clip_takes_the_plays_meeting_its_span(host, tmp_path):
    path = write(tmp_path, "clipped.py", CLIPPED)
    # Executed as the low quality renders, so the clip is checked against it.
    low = {"pixel_width": 854, "pixel_height": 480, "frame_rate": 15}
    session = host.connect(path, "Clipped", low)
    out = tmp_path / "out" / "clip.mp4"
    found = session.call(
        "clip", t0=1.2, t1=1.8, quality="low_quality", out=str(out), deadline_s=120
    )
    assert found["generation"] == 1
    clip = _valid("render", found["clip"])
    assert clip["path"] == str(out)
    # The play of 1-2 s meets the span; the waits either side do not.
    assert _frames(out) == (15, 854, 480)
    shown = [play for play in clip["timeline"] if not play["skipped"]]
    assert [(p["index"], p["start"], p["end"]) for p in shown] == [(1, 1.0, 2.0)]
    assert clip["self_check"] == CHECKED
    assert "divergence" not in clip
    assert _beside(out) == []


def test_a_span_touching_a_play_at_one_instant_takes_it_too(host, tmp_path):
    path = write(tmp_path, "clipped.py", CLIPPED)
    session = host.connect(path, "Clipped", SMALL)
    out = tmp_path / "clip.mp4"
    found = session.call("clip", t0=1, t1="2/1", out=str(out), deadline_s=120)
    shown = [p["index"] for p in found["clip"]["timeline"] if not p["skipped"]]
    assert shown == [0, 1, 2]
    assert _frames(out) == (45, 96, 54)
    assert found["clip"]["self_check"] == CHECKED


def test_a_render_with_nothing_like_it_to_check_against_says_so(host, tmp_path):
    path = write(tmp_path, "clipped.py", CLIPPED)
    # 96 by 54 is 16:9; the low quality's 854 by 480 is not.
    session = host.connect(path, "Clipped", SMALL)
    out = tmp_path / "clip.mp4"
    found = session.call(
        "clip", t0=1.2, t1=1.8, quality="low_quality", out=str(out), deadline_s=120
    )
    clip = _valid("render", found["clip"])
    assert clip["self_check"]["status"] == "unchecked"
    assert "frame rate and pixel aspect" in clip["self_check"]["reason"]
    assert clip["self_check"]["reference"] is None
    assert "divergence" not in clip


def test_a_render_checks_itself_against_its_session_s_execution(host, tmp_path):
    path = write(tmp_path, "scattered.py", SCATTERED)
    session = host.connect(path, "Scattered", SMALL)
    session.call("lint")
    out = tmp_path / "render.mp4"
    honest = session.call("render", out=str(out), deadline_s=120)
    assert "divergence" not in _valid("render", honest["render"])
    assert honest["render"]["self_check"] == LINTED
    assert _frames(out) == (15, 96, 54)

    # Plant, as the session's kept summary, an execution under another seed.
    planted = Session.open(path, "Scattered", config={**SMALL, "seed": 7})
    try:
        other = planted.summary().to_json()
    finally:
        planted.close()
    (kept,) = Path(os.environ["ARACHNE_STORE"]).rglob("summaries/*.json")
    kept.write_text(other)

    checked = session.call("render", out=str(out), deadline_s=120)
    divergence = _valid("render", checked["render"])["divergence"]
    assert divergence["code"] == "replay.diverged"
    assert divergence["kind"] == "execution"
    assert divergence["play"] == 0
    assert checked["render"]["self_check"] == LINTED
    # A finding is an answer: the render was written all the same.
    assert _frames(out) == (15, 96, 54)


def test_a_render_checked_against_another_interpreter_s_summary_is_unchecked(
    host, tmp_path, capsys
):
    """A kept summary another interpreter made can't be compared with the
    render's execution: the render is unchecked, saying why with both
    interpreters named, in its reply, in the command line's reading of it,
    and in a strict command's exit; never checked, and never diverged.
    """
    path = write(tmp_path, "scattered.py", SCATTERED)
    session = host.connect(path, "Scattered", SMALL)
    session.call("lint")
    (kept,) = Path(os.environ["ARACHNE_STORE"]).rglob("summaries/*.json")
    foreign = json.loads(kept.read_text())
    foreign["interpreter"] = "other-99"
    kept.write_text(json.dumps(foreign))

    out = tmp_path / "render.mp4"
    found = session.call("render", out=str(out), deadline_s=120)
    rendered = _valid("render", found["render"])
    check = rendered["self_check"]
    assert check["status"] == "unchecked"
    assert check["reference"] is None
    assert sys.implementation.cache_tag in check["reason"]
    assert "other-99" in check["reason"]
    assert "divergence" not in rendered
    assert _frames(out) == (15, 96, 54)

    assert formatted(rendered).splitlines()[-1] == (
        f"self_check: unchecked ({check['reason']})"
    )
    _done(found, "render", as_json=False, strict=False)
    with pytest.raises(click.exceptions.Exit) as stopped:
        _done(found, "render", as_json=True, strict=True)
    assert stopped.value.exit_code == EXIT_STRICT
    capsys.readouterr()


def test_a_render_says_which_execution_it_was_checked_against(host, tmp_path):
    path = write(tmp_path, "clipped.py", CLIPPED)
    session = host.connect(path, "Clipped", SMALL)
    out = str(tmp_path / "render.mp4")
    # No lint: checked against the session's own execution.
    assert session.call("render", out=out)["render"]["self_check"] == CHECKED
    # A lint judged this very execution, at the render's frame rate and
    # pixel aspect.
    session.call("lint")
    assert session.call("render", out=out)["render"]["self_check"] == LINTED
    # The source changed and reloaded: the lint judged an earlier generation.
    path.write_text(path.read_text().replace("run_time=1", "run_time=0.5"))
    assert session.call("reload")["reloaded"]
    rendered = session.call("render", out=out)
    assert rendered["generation"] == 2
    assert rendered["render"]["self_check"] == CHECKED


def test_a_render_past_its_deadline_leaves_nothing_at_out_and_ends_the_server(
    host, tmp_path
):
    path = write(tmp_path, "slow.py", SLOW)
    session = host.connect(path, "Slow", SMALL)
    out = tmp_path / "slow.mp4"
    deadline = 2.0
    sent = time.monotonic()
    with pytest.raises(registry.RemoteError) as raised:
        session.call("render", out=str(out), deadline_s=deadline)
    assert raised.value.code == protocol.DEADLINE_EXCEEDED
    # It expired as it ran, and says the server ends.
    assert raised.value.data["ends"] is True
    assert gone(session.pid, wait=deadline + 1)
    assert time.monotonic() - sent <= deadline + 1.5
    assert not out.exists()
    assert _beside(out) == []
    assert registry.sessions(host.directory) == []


#: A server answering one render whose deadline passes once it has been
#: checked and before it is written: the check is held until the deadline's
#: error has been sent, as a render slow at that point would be.
AT_PUBLICATION = """\
    import json
    import sys
    import threading
    import time
    from types import SimpleNamespace

    from arachne.server import protocol, render
    from arachne.server.serve import _Server

    scene, out, config = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    server = _Server(600, 3, lambda: None)
    server.dispatcher.handle("open", {"file": scene, "scene": "Still", "config": config})
    expired = threading.Event()
    checked = render._check


    def held(*args):
        found = checked(*args)
        if not expired.wait(10):
            raise RuntimeError("the deadline never came")
        return found


    render._check = held


    class Channel:
        def send(self, data):
            if json.loads(data).get("error", {}).get("code") == protocol.DEADLINE_EXCEEDED:
                expired.set()
                # The render goes on while the server ends.
                threading.Event().wait(0.8)


    request = {"jsonrpc": "2.0", "id": 1, "method": "render",
               "params": {"out": out, "deadline_s": 3}}
    line = SimpleNamespace(data=protocol.encode(request), received=time.monotonic())
    server.answer(Channel(), line)
    threading.Event().wait(10)
"""

STILL = """\
    from arachne import Dot, Scene


    class Still(Scene):
        def construct(self):
            self.add(Dot())
            self.wait(0.5)
"""


def test_a_render_past_its_deadline_as_it_is_written_leaves_out_as_it_was(tmp_path):
    path = write(tmp_path, "still.py", STILL)
    out = tmp_path / "still.mp4"
    out.write_bytes(b"the render before")
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(AT_PUBLICATION),
            str(path),
            str(out),
            json.dumps(SMALL),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == protocol.EXIT_DEADLINE, done.stderr
    assert out.read_bytes() == b"the render before"
    assert _beside(out) == []


def test_what_clip_and_render_refuse(host, tmp_path):
    path = write(tmp_path, "clipped.py", CLIPPED)
    session = host.connect(path, "Clipped", SMALL)
    out = str(tmp_path / "x.mp4")
    refused = [
        ("render", {"out": str(tmp_path)}),
        ("clip", {"t0": 1, "t1": 2, "out": str(tmp_path)}),
        ("clip", {"t0": 2, "t1": 1, "out": out}),
        ("clip", {"t0": 3.5, "t1": 4, "out": out}),
        ("clip", {"t0": 1, "out": out}),
        ("render", {"plays": [0, 2], "out": out}),
        ("render", {"plays": [2, 3], "out": out}),
        ("render", {"plays": [], "out": out}),
        ("render", {"plays": [1], "quality": "no_quality", "out": out}),
    ]
    for method, params in refused:
        with pytest.raises(registry.RemoteError) as raised:
            session.call(method, **params)
        assert raised.value.code == protocol.INVALID_PARAMS, (method, params)
    with pytest.raises(registry.RemoteError, match="contiguous ascending run"):
        session.call("render", plays=[1, 0], out=out)
    assert not Path(out).exists()
    assert session.call("ping") == {"generation": 1}


def test_a_render_is_checked_only_against_an_execution_like_its_own():
    identity = {"source_digest": "a", "input_digests": [], "seed": 0}

    def key(**config):
        return summary_key("/scene.py", "Scene", identity, config)

    sixteen_nine = key(pixel_width=1920, pixel_height=1080, frame_rate=30)
    # Another size of the same aspect: the same execution.
    assert key(pixel_width=1280, pixel_height=720, frame_rate=30) == sixteen_nine
    # Another aspect: the camera fits another frame to it.
    assert key(pixel_width=854, pixel_height=480, frame_rate=30) != sixteen_nine
    # Another frame rate: a play is evaluated at other instants.
    assert key(pixel_width=1920, pixel_height=1080, frame_rate=60) != sixteen_nine
    other = {**identity, "seed": 1}
    assert summary_key("/scene.py", "Scene", other, {}) != key()
