"""Media objects drawn on a device, the launch mode's: a clip's transparent
and half-transparent pixels show what is behind them; the routing fixture's
clip is drawn by the compositor to the still's pixels; a frame that decodes
otherwise draws nothing; and old scenes keep their bytes around a scene
that drew a clip. Collected only where the GPU build is installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.assets import load_asset
from arachne.mobject.media import frames
from arachne.mobject.media.frames import FrameCache
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError

from .conftest import SCENES, SMALL, line_of

ROOT = Path(__file__).resolve().parents[2]
#: An opaque backdrop the clip stands over.
BACKDROP = (32, 96, 160)
INSTANTS = (Fraction(0), Fraction(1, 4), Fraction(1, 2))


def _pixels(session, t: Fraction) -> np.ndarray:
    return np.asarray(session.frame(t)).astype(int)


def _close(a: np.ndarray, b: np.ndarray, within: int = 1) -> bool:
    return a.shape == b.shape and int(np.abs(a - b).max()) <= within


def test_a_clip_s_transparent_pixels_show_what_is_behind_them(session_of, gpu_door):
    backdrop = "#{:02x}{:02x}{:02x}".format(*BACKDROP)
    drawn = _pixels(session_of("ClipOverBackdrop", background_color=backdrop), 0)
    assert gpu_door.draws == 1
    clip = load_asset("fixture:numbered-clip")
    straight = frames.checked_frame(clip, 0).astype(float)
    alpha = straight[..., 3:] / 255
    painted = straight[..., :3] * alpha + np.array(BACKDROP) * (1 - alpha)
    # Transparent, half-transparent and opaque pixels are all there.
    assert {0, 128, 255} <= set(np.unique(straight[..., 3]).astype(int))
    assert _close(drawn[..., :3], np.round(painted))
    assert (drawn[..., 3] == 255).all()


def test_the_routing_fixture_s_clip_is_drawn_by_the_compositor_as_the_still(
    session_of, gpu_door
):
    still = session_of("FixtureImage")
    clip = session_of("FixtureClip")
    for t in INSTANTS:
        assert _close(_pixels(clip, t), _pixels(still, t))
    assert gpu_door.draws == len(INSTANTS)
    (opened,) = gpu_door.opened
    named = {"name": "gpu", "pin": dict(opened.pin)}
    assert clip.renderer == named
    assert [answer.renderer for answer in clip.answers(list(INSTANTS))] == [named] * 3


def test_clips_and_a_sprite_moved_and_faded_draw_on_the_device(session_of, gpu_door):
    for name in ("ClipMoved", "SpriteAlone", "TwoOfOneAsset"):
        session = session_of(name)
        assert session.route == "gpu"
        pictures = [_pixels(session, t) for t in (Fraction(1, 5), Fraction(1))]
        assert pictures[0].shape == (54, 96, 4)
    assert gpu_door.draws == 6


def test_a_clip_frame_that_decodes_otherwise_draws_nothing(
    tmp_path, monkeypatch, gpu_door
):
    monkeypatch.setattr(frames, "cache", FrameCache())
    decoded = frames.decode

    def changed(data: bytes) -> np.ndarray:
        pixels = decoded(data)
        pixels[2, 2, 1] ^= 8
        return pixels

    monkeypatch.setattr(frames, "decode", changed)
    server = Dispatcher()
    picture = tmp_path / "frame.png"
    try:
        server.handle(
            "open", {"file": str(SCENES), "scene": "FixtureClip", "config": SMALL}
        )
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.25, "path": str(picture)})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    finding = refused.value.data["finding"]
    assert finding["code"] == "media.inexact"
    assert finding["detail"]["reason"] == "digest"
    assert finding["line"] == line_of("fixture-clip").line
    assert gpu_door.draws == 0
    assert not picture.exists()


#: Opens each old scene, then a scene that drew clips on the launch mode's
#: device, then each old scene again, in one fresh interpreter.
PROGRAM = """\
import hashlib, json, sys

from tests.helpers.gpu_door import install

door = install()
from arachne.session import Session

directory, media, scenes = sys.argv[1:4]
config = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4, "media_dir": media}


def opened(file, scene):
    session = Session.open(file, scene, config=config)
    try:
        return {
            "route": session.route,
            "routes": [play.route for play in session.record.plays],
            "frames": [
                hashlib.sha256(session.frame(t).tobytes()).hexdigest()
                for t in (0, 0.25, 0.5, 0.7)
            ],
        }
    finally:
        session.close()


old = f"{directory}/old.py"
fresh = {scene: opened(old, scene) for scene in ("OldSvg", "OldImage")}
clips = opened(scenes, "ClipMoved")
drawn = door.draws
after = {scene: opened(old, scene) for scene in ("OldSvg", "OldImage")}
print(json.dumps({"fresh": fresh, "clips": clips, "drawn": drawn, "after": after}))
"""


def test_old_scenes_draw_the_same_bytes_after_a_scene_drew_clips(tmp_path):
    from tests.assets.test_old_constructors import _scenes

    directory = _scenes(tmp_path / "scenes")
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PROGRAM), str(directory)]
        + [str(tmp_path / "media"), str(SCENES)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ},
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["clips"]["route"] == "gpu"
    assert found["drawn"] == 4
    assert found["fresh"] == found["after"]
    assert all(scene["route"] == "cpu" for scene in found["fresh"].values())
