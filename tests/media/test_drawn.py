"""Media objects drawn on a device, the launch mode's: a clip's transparent
and half-transparent pixels show what is behind them; the routing fixture's
clip is drawn by the compositor to the still's pixels; a frame that decodes
otherwise draws nothing; and old scenes keep their bytes around a scene
that drew a clip. Collected only where the GPU build is installed.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from arachne.assets import load_asset
from arachne.mobject.media import frames
from arachne.mobject.media.frames import FrameCache
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError
from tests.helpers.gpu_door import run_child

from .conftest import SCENES, SMALL, line_of

#: An opaque backdrop the clip stands over.
BACKDROP = (32, 96, 160)
INSTANTS = (Fraction(0), Fraction(1, 4), Fraction(1, 2))


def _pixels(session, t: Fraction) -> np.ndarray:
    return np.asarray(session.frame(t)).astype(int)


def _close(a: np.ndarray, b: np.ndarray, within: int = 1) -> bool:
    return a.shape == b.shape and int(np.abs(a - b).max()) <= within


def test_a_clip_s_transparent_pixels_show_what_is_behind_them(session_of, door):
    backdrop = "#{:02x}{:02x}{:02x}".format(*BACKDROP)
    drawn = _pixels(session_of("ClipOverBackdrop", background_color=backdrop), 0)
    assert door.draws == 1
    clip = load_asset("fixture:numbered-clip")
    straight = frames.checked_frame(clip, 0).astype(float)
    alpha = straight[..., 3:] / 255
    painted = straight[..., :3] * alpha + np.array(BACKDROP) * (1 - alpha)
    # Transparent, half-transparent and opaque pixels are all there.
    assert {0, 128, 255} <= set(np.unique(straight[..., 3]).astype(int))
    assert _close(drawn[..., :3], np.round(painted))
    assert (drawn[..., 3] == 255).all()


def test_the_routing_fixture_s_clip_is_drawn_by_the_compositor_as_the_still(
    session_of, door
):
    still = session_of("FixtureImage")
    clip = session_of("FixtureClip")
    for t in INSTANTS:
        assert _close(_pixels(clip, t), _pixels(still, t))
    assert door.draws == len(INSTANTS)
    (opened,) = door.opened
    named = {"name": "gpu", "pin": dict(opened.pin)}
    assert clip.renderer == named
    assert [answer.renderer for answer in clip.answers(list(INSTANTS))] == [named] * 3


def test_clips_and_a_sprite_moved_and_faded_draw_on_the_device(session_of, door):
    for name in ("ClipMoved", "SpriteAlone", "TwoOfOneAsset"):
        session = session_of(name)
        assert session.route == "gpu"
        pictures = [_pixels(session, t) for t in (Fraction(1, 5), Fraction(1))]
        assert pictures[0].shape == (54, 96, 4)
    assert door.draws == 6


def test_a_clip_frame_that_decodes_otherwise_draws_nothing(tmp_path, monkeypatch, door):
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
    assert door.draws == 0
    assert not picture.exists()


def test_a_warm_clip_whose_frame_decodes_otherwise_once_evicted_draws_nothing(
    tmp_path, monkeypatch, door
):
    """A clip already drawn on the device, its frame then let go by the cache
    and decoded again otherwise, is refused at the next frame with the line
    that made it, and nothing more is drawn.
    """
    kept = FrameCache()
    monkeypatch.setattr(frames, "cache", kept)
    server = Dispatcher()
    drawn, refused_at = tmp_path / "drawn.png", tmp_path / "refused.png"
    try:
        server.handle(
            "open", {"file": str(SCENES), "scene": "FixtureClip", "config": SMALL}
        )
        server.handle("frame", {"t": 0.25, "path": str(drawn)})
        assert drawn.exists()
        assert door.draws == 1
        assert len(kept) == 1
        kept.clear()
        decoded = frames.decode

        def changed(data: bytes) -> np.ndarray:
            pixels = decoded(data)
            pixels[3, 3, 0] ^= 16
            return pixels

        monkeypatch.setattr(frames, "decode", changed)
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.5, "path": str(refused_at)})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    finding = refused.value.data["finding"]
    assert finding["code"] == "media.inexact"
    assert finding["detail"]["reason"] == "digest"
    assert finding["line"] == line_of("fixture-clip").line
    assert door.draws == 1
    assert not refused_at.exists()


#: Opens each old scene, then a scene that drew clips on the launch mode's
#: device, then each old scene again, in one fresh interpreter.
PROGRAM = """\
import hashlib, json, sys

from arachne.compose import device
from arachne.session import Session
from tests.helpers.gpu_door import LaunchDoor

door = device.door = LaunchDoor()

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
    printed = run_child(PROGRAM, str(directory), str(tmp_path / "media"), str(SCENES))
    found = json.loads(printed.strip().splitlines()[-1])
    assert found["clips"]["route"] == "gpu"
    assert found["drawn"] == 4
    assert found["fresh"] == found["after"]
    assert all(scene["route"] == "cpu" for scene in found["fresh"].values())
