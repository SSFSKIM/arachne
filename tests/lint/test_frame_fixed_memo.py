"""A leaf fixed in a ThreeDCamera's frame is drawn unprojected, so where it is
placed follows from its own version and the flat view alone: the leaf memo
keeps it while the camera turns, and a lint with the memo still reads every
part as one without it.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.camera.projection import ThreeDView, record_view
from arachne.lint import RULES, lint
from arachne.lint.leaf_memo import LeafMemo
from arachne.timeline.routes import _resolve

from . import memo_scenes
from .test_leaf_memo import _counted, _is_probe
from .test_leaf_memo_answers import _lint_and_parts, _view_changes

SCENES = Path(memo_scenes.__file__)
FPS = 15


def test_a_leaf_fixed_in_the_frame_is_placed_once_while_the_camera_turns(
    session_of, monkeypatch
):
    session = session_of("TurningFixedProbe3D", SCENES, frame_rate=FPS)
    counts = _counted(monkeypatch)
    lint(session, rules=RULES)
    assert counts["reads"] > 1
    assert counts["placed"] == 1


def _places(session, mob_of, changes):
    """Which of ``changes`` (named views) place ``mob_of``'s leaf again, read
    after the view as read.
    """
    record = session.record
    capture = _resolve(record, Fraction(3, FPS)).capture
    camera = record._scene.renderer.camera
    (mob,) = [m for m in capture.draw_list() if mob_of(m)]
    memo = LeafMemo()
    made: list[str] = []

    def place(name, view):
        viewing = memo.viewing(record, capture, camera, view)
        assert viewing is not None
        node = viewing.node(mob, ())
        assert node is not None
        viewing.lays(node, lambda: None)
        viewing.placed(node, lambda: made.append(name))

    view = record_view(record)
    place("as read", view)
    for name, changed in changes(view):
        place("as read", view)
        place(name, changed)
    return made


def _three_d(name: str) -> bool:
    return name.startswith("three_d.")


def test_a_fixed_leaf_is_placed_again_for_any_flat_field_but_no_projection_field(
    session_of,
):
    session = session_of("TurningFixedProbe3D", SCENES, frame_rate=FPS)
    names = [name for name, _ in _view_changes(record_view(session.record))]
    assert {n.removeprefix("three_d.") for n in names if _three_d(n)} == {
        "phi",
        "theta",
        "gamma",
        "focal_distance",
        "zoom",
        "exponential_projection",
    }
    assert len(ThreeDView.__dataclass_fields__) == 6
    fixed = _places(session, _is_probe, _view_changes)
    assert fixed == ["as read", *(n for n in names if not _three_d(n))]
    unfixed = _places(session, lambda m: type(m).__name__ == "Dot", _view_changes)
    assert unfixed == ["as read", *names]


@pytest.mark.parametrize(
    "scene",
    [
        "TurningFixedProbe3D",
        "FixedProbeChangesWhileTurning",
        "FixedProbeUnderFrameCentreAndZoom",
        "ProbeFixedMidTurn",
        "ProbeUnfixedMidTurn",
        "FixedGroupWhileTurning",
        "FixedOrientationWhileTurning",
    ],
)
def test_a_lint_with_the_memo_reads_every_part_as_one_without_it_as_the_camera_turns(
    session_of, monkeypatch, scene
):
    fresh = _lint_and_parts(session_of, monkeypatch, scene, memo=False)
    kept = _lint_and_parts(session_of, monkeypatch, scene, memo=True)
    assert fresh[1]
    assert kept == fresh
