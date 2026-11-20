"""A rig's anchors are points of its parts' material: an outline anchor is a
landmark of its part's outline as posed and placed, an interior anchor a
point carried with its part, at its keyed positions on a reshaped part;
each is read from the rig's state alone. An anchor the manifest places on
a slot's outline or off its part's outline refuses the rig, and one it does
not have is a finding.
"""

from __future__ import annotations

import copy
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import LEFT
from arachne.assets import Anchor, Rig
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.draw import Drawing

from . import draw_fixture_pack, rig_scenes
from .helpers import admitted, copied, edited, refused

SCENES = Path(rig_scenes.__file__)


def _scene_of(asset, point) -> np.ndarray:
    size = asset.manifest.size
    (ox, oy), s = size.origin, size.scene_units_per_unit
    return np.array([(point[0] - ox) * s, (oy - point[1]) * s, 0.0])


def test_an_outline_anchor_turns_with_its_part_and_moves_with_its_rig():
    box = Rig("fixture:lidded-box")
    knob = box.anchor("knob")
    assert isinstance(knob, Anchor)
    assert knob.resolved
    np.testing.assert_allclose(
        knob.point(), _scene_of(box.asset, (200, 108)), atol=1e-9
    )
    box.pose(lid_open=90).shift(LEFT)
    # The knob is 100 to the hinge's right and 32 above it; turned a quarter
    # counterclockwise it is 100 above the hinge and 32 to its left.
    hinge = _scene_of(box.asset, (100, 140))
    s = box.asset.manifest.size.scene_units_per_unit
    expected = hinge + np.array([-32 * s, 100 * s, 0]) + LEFT
    np.testing.assert_allclose(knob.point(), expected, atol=1e-9)


def test_an_interior_anchor_on_a_slot_rides_the_slot():
    figure = Rig("fixture:figure")
    tip = figure.anchor("hand_tip")
    np.testing.assert_allclose(
        tip.point(), _scene_of(figure.asset, (195, 302)), atol=1e-12
    )
    figure.pose(reach=1)
    # Slid by its key in its own place, then turned with the arm.
    shoulder = _scene_of(figure.asset, (195, 168))
    slid = _scene_of(figure.asset, (195 + 12, 302 - 4)) - shoulder
    angle = math.radians(70)
    turned = np.array(
        [
            slid[0] * math.cos(angle) - slid[1] * math.sin(angle),
            slid[0] * math.sin(angle) + slid[1] * math.cos(angle),
            0,
        ]
    )
    np.testing.assert_allclose(tip.point(), shoulder + turned, atol=1e-9)


def test_an_anchor_is_a_value_that_copies_as_itself():
    knob = Rig("fixture:lidded-box").anchor("knob")
    assert copy.copy(knob) is knob
    assert copy.deepcopy(knob) is knob


@pytest.fixture
def swelling():
    admission.DEMOTED.clear()
    session = Session.open(
        SCENES,
        "Swelling",
        config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 8},
    )
    yield session
    session.close()
    admission.DEMOTED.clear()


def _radii(swell: float) -> tuple[float, float]:
    """The swelling layer's radii at ``swell``, from the generator's keys."""
    keys = draw_fixture_pack.SWELL
    for (a, rx0, ry0), (b, rx1, ry1) in zip(keys, keys[1:], strict=False):
        if a <= swell <= b:
            w = (swell - a) / (b - a)
            return rx0 + w * (rx1 - rx0), ry0 + w * (ry1 - ry0)
    raise AssertionError(swell)


def test_anchors_of_a_reshaped_part_follow_it_as_the_record_draws_it(swelling):
    record = swelling.record
    section = next(m for m in swelling.scene.mobjects if isinstance(m, Rig))
    tip, granule = section.anchor("tip"), section.anchor("granule")
    # The swelling layer's tip is the first point of its outline.
    outline = section.part("inner_layer").submobjects[0]
    instants = [Fraction(k, 8) for k in (0, 2, 4, 6, 8)] + [Fraction(11, 8)]
    for t in instants:
        play, capture = record.capture_at(t)
        landmark = np.asarray(record.node_state(outline, capture).get("_points"))[0]
        with Drawing(record) as drawing:
            drawing.show(capture, capture.roots)
            at_tip, at_granule = tip.point(), granule.point()
            swell = section.value("swell")
        np.testing.assert_allclose(at_tip, landmark, atol=1e-9, err_msg=f"at {t}")
        if play is not None and play.index == 0:
            canvas = draw_fixture_pack.granule(*_radii(swell))
            expected = _scene_of(section.asset, canvas) + LEFT
            # The pack holds the generator's numbers to three decimals.
            scale = section.asset.manifest.size.scene_units_per_unit
            np.testing.assert_allclose(at_granule, expected, atol=5e-4 * scale)
    # It swelled all the way, and the tip went with it.
    assert swell == 1


def test_an_outline_anchor_on_a_slot_refuses_the_rig(tmp_path):
    directory = edited(
        copied(tmp_path, "figure"),
        lambda d: d["anchors"][0].update(attach="outline"),
    )
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == "anchors[0].attach"
    assert found.detail["name"] == "hand_tip"
    assert "slot" in found.detail["reason"]


def test_an_outline_anchor_off_its_parts_outline_refuses_the_rig(tmp_path):
    directory = edited(
        copied(tmp_path, "lidded-box"),
        lambda d: d["anchors"][0].update(at=[200, 109]),
    )
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == "anchors[0].at"
    # Within half a canvas unit, it is on the outline.
    near = admitted(
        edited(
            copied(tmp_path, "lidded-box", "near"),
            lambda d: d["anchors"][0].update(at=[200, 108.4]),
        )
    )
    point = Rig(near).anchor("knob").point()
    np.testing.assert_allclose(
        point, Rig("fixture:lidded-box").anchor("knob").point(), atol=1e-12
    )


def test_an_anchor_the_manifest_does_not_have_is_a_finding_outside_a_scene():
    found = refused(
        "annotation.anchor_unknown", lambda: Rig("fixture:figure").anchor("chset")
    )
    assert found.detail == {"asset": "figure", "name": "chset", "nearest": ["chest"]}


@pytest.mark.parametrize("still", [False, True])
def test_a_render_refuses_a_scene_that_asks_an_unknown_anchor(tmp_path, still):
    """The anchor is asked after the scene's last play: a movie is not
    finished, and a last frame is not written.
    """
    from arachne import tempconfig
    from arachne.lint.raised import RaisedFinding

    options = {
        "media_dir": str(tmp_path),
        "pixel_width": 96,
        "pixel_height": 54,
        "frame_rate": 4,
        "disable_caching": True,
    }
    if still:
        options["save_last_frame"] = True
    with tempconfig(options), pytest.raises(RaisedFinding) as raised:
        rig_scenes.StillWithUnknownAnchor().render()
    assert raised.value.code == "annotation.anchor_unknown"
    written = [p.name for p in tmp_path.rglob("*") if p.suffix in (".png", ".mp4")]
    assert "StillWithUnknownAnchor.mp4" not in written
    assert not any(name.endswith(".png") for name in written)


def test_a_live_drawn_render_writes_no_frame_after_an_unknown_anchor(tmp_path):
    """A camera the record cannot draw draws each frame as its play runs:
    the anchor asked before the play refuses the play before a frame of
    the sequence is written.
    """
    from arachne import tempconfig
    from arachne.lint.raised import RaisedFinding

    options = {
        "media_dir": str(tmp_path),
        "pixel_width": 96,
        "pixel_height": 54,
        "frame_rate": 4,
        "disable_caching": True,
        "format": "png-sequence",
    }
    with tempconfig(options), pytest.raises(RaisedFinding) as raised:
        rig_scenes.LiveWithUnknownAnchor().render()
    assert raised.value.code == "annotation.anchor_unknown"
    assert not list(tmp_path.rglob("*.png"))
