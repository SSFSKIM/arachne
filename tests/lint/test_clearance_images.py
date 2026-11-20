"""``apart`` asked about pictures (a still, a clip, a sprite, CE's image):
each is read as the sweep's parts read an image, a filled leaf over its
corners, and is never refused.
"""

from __future__ import annotations

from pathlib import Path

from tests.lint import clearance_image_scenes

SCENES = Path(clearance_image_scenes.__file__)


def _answers(found):
    return {
        ask: [(f["code"], f["detail"]) for f in rows] for ask, rows in found.items()
    }


def test_apart_answers_for_every_picture_and_alike(session_of):
    found = session_of("ApartFromPictures", SCENES, frame_rate=15).scene.found
    assert set(found) == {"still", "clip", "sprite", "image"}
    answers = {kind: _answers(asked) for kind, asked in found.items()}
    reference = answers["image"]
    # A picture on a square's edge is not clear of it; one away from it is.
    assert [code for code, _ in reference["touching"]] == ["contact.held"]
    assert [code for code, _ in reference["touching_target"]] == ["contact.held"]
    assert reference["away"] == reference["away_target"] == []
    for kind in ("still", "clip", "sprite"):
        assert answers[kind] == reference, kind


def test_apart_reads_a_declared_picture_fill_whole(session_of):
    """A leaf that says its fill is a picture is held to its box, whatever
    its fill opacity: a dot in an outline's hollow is clear of the outline,
    but not of a picture that fills it.
    """
    found = session_of("ApartFromAPictureFill", SCENES, frame_rate=15).scene.found
    assert found["plain"] == []
    assert [f["code"] for f in found["declared"]] == ["contact.held"]
