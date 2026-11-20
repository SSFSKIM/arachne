"""Media objects at rest: built at the size their manifests declare, on their
first frame; placed, scaled, ordered and faded as images are, without ever
changing the frame they show; never morphed; and held by the record as an
index and a digest, never as pixels.
"""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import arachne
from arachne import (
    FadeIn,
    Group,
    ImageMobject,
    Scene,
    Sprite,
    Square,
    Still,
    Transform,
    VideoMobject,
)
from arachne.routing import held
from arachne.timeline.execution import play_keys
from arachne.timeline.record import NodeState, content_fields

from .conftest import copied
from .media_scenes import StillMoved

ROOT = Path(arachne.__file__).resolve().parents[1]


def test_each_stands_at_its_manifest_size_on_its_first_frame():
    clip = VideoMobject("fixture:numbered-clip")
    sprite = Sprite("fixture:stepper")
    still = Still("fixture:swatch")
    assert (clip.width, clip.height) == pytest.approx((3.2, 1.8))
    assert (sprite.width, sprite.height) == pytest.approx((1.2, 1.2))
    assert (still.width, still.height) == pytest.approx((2, 2))
    assert np.allclose(clip.get_center(), 0)
    for media in (clip, sprite, still):
        assert media.frame == 0
        assert media.opacity == 1
    assert (clip.frame_count, clip.frame_rate, clip.duration) == (48, 24, 2)
    assert (sprite.frame_count, sprite.frame_rate) == (8, 12)
    assert sprite.duration == Fraction(2, 3)


def test_a_clip_and_a_sprite_declare_their_rows_and_a_still_none():
    assert held(VideoMobject("fixture:numbered-clip")) == {"media.video"}
    assert held(Sprite("fixture:stepper")) == {"media.sprite"}
    assert held(Still("fixture:swatch")) == frozenset()
    # A copy holds what its original held.
    assert held(VideoMobject("fixture:numbered-clip").copy()) == {"media.video"}


def test_an_asset_of_another_kind_names_the_class_that_shows_it():
    with pytest.raises(ValueError, match="show it with VideoMobject"):
        Still("fixture:numbered-clip")
    with pytest.raises(ValueError, match="show it with Still"):
        Sprite("fixture:swatch")
    with pytest.raises(ValueError, match="show it with Rig"):
        VideoMobject("fixture:dial")


def test_a_fade_writes_the_opacity_and_leaves_the_frame():
    clip = VideoMobject("fixture:numbered-clip")
    full = np.array(clip.get_pixel_array())
    clip.set_opacity(0.5)
    half = clip.get_pixel_array()
    assert clip.frame == 0
    assert np.array_equal(half[..., :3], full[..., :3])
    assert np.array_equal(half[..., 3], (full[..., 3] * 0.5).astype(np.uint8))
    clip.fade(1)
    assert clip.opacity == 0
    assert not clip.get_pixel_array()[..., 3].any()
    with pytest.raises(AttributeError, match="asset's frames"):
        clip.pixel_array = full


def _states(scene_class: type[Scene], name: str) -> list[NodeState]:
    """Every state the record keeps of the media object ``name`` in a run
    of ``scene_class``.
    """
    from arachne.session import Session

    from .conftest import SCENES, SMALL

    session = Session.open(SCENES, scene_class.__name__, config=dict(SMALL))
    try:
        found = [
            state
            for node in session.record._nodes.values()
            if type(node.mob).__name__ == name
            for state in node.states
        ]
        assert found
        return found
    finally:
        session.close()


def test_the_record_holds_no_pixels_and_every_version_shows_the_first_frame():
    states = _states(StillMoved, "Still")
    for state in states:
        assert not any(
            isinstance(value, np.ndarray) and value.ndim == 3 for value in state.values
        ), state.keys
        assert state.get("_frame") == 0
    opacities = {state.get("_opacity") for state in states}
    assert {0.0, 0.5, 1.0} <= opacities


CAPTURED = """\
from arachne import *


class Copied(Scene):
    def construct(self):
        clip = VideoMobject({asset!r})
        self.play(FadeIn(clip))
        self.play(clip.animate.shift(RIGHT).set_opacity(0.5))
"""


def test_one_asset_s_bytes_in_two_directories_key_one_capture(tmp_path, session_of):
    """The capture key holds an asset's digest, not where it was read from:
    the same bytes in two directories key every play's state alike, and an
    object that holds another digest keys apart.
    """
    keys = []
    for place in ("one", "two"):
        directory = copied(tmp_path / place, "numbered-clip")
        scene = tmp_path / f"{place}.py"
        scene.write_text(CAPTURED.format(asset=str(directory)))
        session = session_of("Copied", scene, media_dir=str(tmp_path / "media"))
        keys.append(play_keys(session.record))
    assert len(keys[0]) == 2
    assert keys[0] == keys[1]
    one = VideoMobject(copied(tmp_path / "three", "numbered-clip"))
    other = VideoMobject(copied(tmp_path / "four", "numbered-clip"))
    fields = [list(content_fields(NodeState.of(m))) for m in (one, other)]
    assert one.asset.directory != other.asset.directory
    assert fields[0] == fields[1]
    other._digest = "sha256:" + "0" * 64
    assert list(content_fields(NodeState.of(other))) != fields[0]


def test_fade_in_a_shift_and_a_scale_move_the_object_and_keep_its_frame(session_of):
    session = session_of("ClipMoved")
    ids = [n.id for n in session.state_at(0).nodes if n.type == "VideoMobject"]
    (clip,) = ids
    instants = (Fraction(1, 10), Fraction(1, 2), Fraction(3, 4), Fraction(3, 2))
    centres = [session.state_at(t).node(clip).box for t in instants]
    assert len(set(centres)) == len(centres)
    for node in session.record._nodes.values():
        if type(node.mob) is VideoMobject:
            assert {state.get("_frame") for state in node.states} == {0}


def test_two_assets_do_not_morph_and_the_error_names_the_fade():
    clip, sprite = VideoMobject("fixture:numbered-clip"), Sprite("fixture:stepper")
    for start, end in (
        (clip, sprite),
        (clip, Square()),
        (clip, Still("fixture:swatch")),
    ):
        with pytest.raises(ValueError, match=r"do not morph.*FadeOut, FadeIn"):
            Transform(start.copy(), end).begin()
    # A media object is transformed into a copy of itself, its frame kept.
    moved = clip.copy().shift([1, 0, 0]).set_opacity(0.25)
    play = Transform(clip, moved)
    play.begin()
    play.interpolate(0.5)
    assert clip.opacity == pytest.approx(0.625)
    assert clip.frame == 0
    FadeIn(Group(clip.copy(), sprite.copy())).begin()


def test_no_object_morphs_into_a_media_object_and_nothing_is_touched_first():
    clip = VideoMobject("fixture:numbered-clip")
    image = ImageMobject(np.array(clip.get_pixel_array()))
    for start, end in (
        (Square(), clip),
        (image, clip),
        (Group(Square()), Group(clip.copy())),
        (clip.copy(), Sprite("fixture:stepper")),
    ):
        start.add_updater(lambda mobject, dt: None)
        points = np.array(start.points)
        family = start.get_family()
        with pytest.raises(ValueError, match=r"do not morph.*FadeOut, FadeIn"):
            Transform(start, end).begin()
        # Refused before Animation.begin suspended its updaters, and before
        # align_data changed either side.
        assert not start.updating_suspended
        assert np.array_equal(start.points, points)
        assert start.get_family() == family
        assert len(end.get_family()) == len(end.submobjects) + 1


def test_importing_arachne_decodes_no_frame_and_opens_no_asset():
    code = """
import json, sys
opened = []
sys.addaudithook(lambda event, args: event == "open" and opened.append(str(args[0])))
import arachne
from arachne.mobject.media import frames
print(json.dumps({
    "opened": [p for p in opened if "fixtures" in p or p.endswith(".png")],
    "cached": len(frames.cache),
}))
"""
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == {"opened": [], "cached": 0}
