"""``clip`` and ``render``: a fresh execution of the session's scene through
the render path, written to the file the caller names only once it is whole,
and checked against the execution summary of the session's latest lint.

Each render here is small: a second or so of a few pixels.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import av
import pytest
from jsonschema import Draft202012Validator

from arachne import api_json
from arachne.server import protocol, registry
from arachne.server.store import summary_key
from arachne.session import Session

from .conftest import gone, write

#: The session's configuration: a render at it is 96 by 54 at 15 frames a
#: second, as is the store's summary of its lint.
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}

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


def test_a_render_checks_itself_against_the_latest_lint(host, tmp_path):
    path = write(tmp_path, "scattered.py", SCATTERED)
    session = host.connect(path, "Scattered", SMALL)
    session.call("lint")
    out = tmp_path / "render.mp4"
    honest = session.call("render", out=str(out), deadline_s=120)
    assert "divergence" not in _valid("render", honest["render"])
    assert _frames(out) == (15, 96, 54)

    # Plant, as the lint's summary, an execution under another seed.
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
    # A finding is an answer: the render was written all the same.
    assert _frames(out) == (15, 96, 54)


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
    assert gone(session.pid, wait=deadline + 1)
    assert time.monotonic() - sent <= deadline + 1.5
    assert not out.exists()
    assert _beside(out) == []
    assert registry.sessions(host.directory) == []


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
