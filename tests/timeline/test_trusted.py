"""Route 1, the trusted algebra: a trusted play evaluated ahead, at any
instant from its last evaluated one, equals it evaluated frame by frame; and
what the algebra refuses goes to ticks.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Blink, FocusOn, Scene, tempconfig
from arachne.animation.animation import prepare_animation
from arachne.timeline.trusted import GROUPS, LEAVES, iter_animations

from . import route_scenes
from .conftest import SMALL, render

ROUTE_SCENES = Path(route_scenes.__file__)


@pytest.mark.parametrize("name", sorted(route_scenes.TRUSTED_PLAYS))
def test_a_trusted_play_evaluated_ahead_equals_it_evaluated_frame_by_frame(
    session_of, name
):
    """A render at 60 fps evaluates the play at every frame from the one
    before; the session evaluated it at its start and its events only, and
    answers every other instant analytically from those.
    """
    scene_class = getattr(route_scenes, f"Trusted{name}")
    _, frames = render(scene_class, 60)
    session = session_of(scene_class.__name__, ROUTE_SCENES, grid=None)
    play = session.record.plays[0]
    assert play.route == "trusted", play.admission.reasons
    analytic = 0
    for k, frame in enumerate(frames):
        t = Fraction(k, 60)
        if t < play.end and not play.held and play.base_at(t).instant != t:
            analytic += 1
            assert session.state_at(t).answer == "analytic"
        assert np.array_equal(frame, np.asarray(session.frame(t))), (name, k)
    # A held wait is one sample answering its whole span.
    assert analytic > 30 or play.held


def test_every_entry_of_the_algebra_has_its_own_play():
    """Two entries are tested elsewhere: FocusOn's target dot follows the
    focus point by an updater, so its play ticks (tested with the refusals);
    Blink's parts are UpdateFromFunc, so its play is a state function
    (``Readouts`` in the admission tests).
    """
    with tempconfig(SMALL):
        scene = Scene()
        played: set[type] = set()
        for build in route_scenes.TRUSTED_PLAYS.values():
            played |= {
                type(a) for a in iter_animations([prepare_animation(build(scene))])
            }
    assert set(LEAVES) | set(GROUPS) <= played | {FocusOn, Blink}


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("OverlappingTransforms", "two parts of AnimationGroup write the same mobject"),
        ("Wiggle", "not a function of alpha"),
        ("MoveAlongPath", "reads its path live"),
        ("ChangeSpeed", "keeps its own clock"),
        ("FocusOn", "copies carry updaters"),
        ("UserRate", "rate function"),
        ("UserSubclass", "_Shifting is not in the trusted algebra"),
        ("BackwardGroup", "runs its parts' time back"),
    ],
)
def test_what_the_algebra_refuses_is_ticked(session_of, name, reason):
    session = session_of(f"Refused{name}", ROUTE_SCENES)
    play = session.record.plays[0]
    assert play.route == "ticked"
    assert any(reason in r for r in play.admission.reasons), play.admission.reasons


@pytest.mark.parametrize("fps", [4, 15, 30, 60])
def test_overlapping_transforms_end_the_same_whatever_the_frame_rate(fps):
    """CE ends the shorter part at the first frame after 0.8 s and lets the
    longer one overwrite it from then on, so its end depends on the grid; on
    the tick it is one run.
    """
    scene, _ = render(route_scenes.RefusedOverlappingTransforms, fps)
    reference, _ = render(route_scenes.RefusedOverlappingTransforms, 60)
    assert scene.record.plays[0].route == "ticked"
    dot, expected = scene.mobjects[-1], reference.mobjects[-1]
    assert np.array_equal(dot.points, expected.points)


def test_a_path_function_binding_a_script_array_sees_it_as_the_play_did(session_of):
    """``Rotate`` binds its centre array in its path function; an evaluation
    after the play reads the contents the play had, whatever the array holds
    by then.
    """
    session = session_of("TrustedRotate", ROUTE_SCENES)
    play = session.record.plays[0]
    t = Fraction(1, 7)
    before = session.state_at(t)
    for animation in play._animations:
        centres = animation.path_func.__closure__
        for cell in centres or ():
            if isinstance(cell.cell_contents, np.ndarray):
                cell.cell_contents += 5
    assert session.state_at(t) == before
