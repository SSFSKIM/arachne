"""``apart`` and the sweep read a picture alike: a picture no pixel of
which shows is nothing to stand clear of, and a picture read as a path in
a group with an unfilled leaf is its edge all round, never a curve
through its corners. Alike for a still, a clip, a sprite and a rig's
raster member.
"""

from __future__ import annotations

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
