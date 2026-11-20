"""``apart`` and the sweep read a picture alike: a picture no pixel of
which shows is nothing to stand clear of, and a picture read as a path in
a group with an unfilled leaf is its edge all round, never a curve
through its corners. Alike for a still, a clip, a sprite and a rig's
raster member.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from tests.lint import clearance_picture_scenes

SCENES = Path(clearance_picture_scenes.__file__)


@pytest.fixture(scope="module")
def found():
    from arachne.session import Session

    session = Session.open(
        SCENES,
        "ApartFromPicturesAsDrawn",
        config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 15},
    )
    yield session.scene.found
    session.close()


def _codes(rows) -> list[str]:
    return [row["code"] for row in rows]


@pytest.mark.parametrize("kind", ["still", "clip", "sprite", "member"])
def test_a_picture_no_pixel_of_which_shows_is_nothing_to_stand_clear_of(found, kind):
    # Shown, the dot inside it is not clear of it.
    assert _codes(found[kind]["shows"]) == ["contact.held"]
    assert found[kind]["faded"] == []
    assert found[kind]["faded_target"] == []


@pytest.mark.parametrize("kind", ["still", "clip", "sprite", "member"])
def test_a_picture_read_as_a_path_is_its_edge_all_round(found, kind):
    assert _codes(found[kind]["edge"]) == ["contact.held"]
    # Its middle, where a curve through its corners passes, is clear.
    assert found[kind]["middle"] == []


def _open(scene: str):
    from arachne.session import Session

    return Session.open(
        SCENES, scene, config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}
    )


def test_a_picture_whose_pixels_are_refused_is_present_and_raises_nothing():
    session = _open("ApartFromARefusedStill")
    try:
        found = session.scene.found
    finally:
        session.close()
    assert _codes(found["subject"]) == ["contact.held"]
    assert _codes(found["target"]) == ["contact.held"]


def test_a_picture_in_a_group_rings_nothing_still_or_raster_member():
    from arachne.lint.clearance import TOUCH

    session = _open("ApartFromAWordAcrossAPicture")
    try:
        found = session.scene.found
    finally:
        session.close()
    for name in ("still", "member"):
        assert TOUCH < found[name]["gap"] < clearance_picture_scenes.WORD_PAD, name
    assert found["still"]["found"]
    assert _codes(found["member"]["found"]) == _codes(found["still"]["found"])


def test_a_fill_only_declared_a_picture_is_swept_as_an_image_at_its_box():
    import numpy as np

    from arachne.lint import lint
    from arachne.lint.parts import PartsCache
    from arachne.lint.rules import sweep
    from arachne.timeline.routes import _resolve

    session = _open("SweptPictureFill")
    try:
        lint(session, rules=(sweep,))
        record, t = session.record, Fraction(1, 10)
        resolved = _resolve(record, t)
        parts = PartsCache(memo=False).build(
            record,
            t,
            resolved.play,
            resolved.capture,
            answer="recorded",
            target=(96, 54),
        )
        marker = next(
            m for m in session.scene.mobjects if getattr(m, "fill_is_picture", False)
        )
        points = np.asarray(marker.points)[:, :2]
    finally:
        session.close()
    (image,) = [part for part in parts.parts if part.kind == "image"]
    (leaf,) = image.leaves
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    # Its corners are its points' box's, in CE's image order.
    assert np.allclose(leaf.points, [[x0, y1], [x1, y1], [x0, y0], [x1, y0]], atol=1e-9)
