"""Admission is by proof: what the proof can't cover ticks, with a named
reason, and comes out the same whichever instants are evaluated.

Each scene here once passed for a state function or a trusted play while
its frames depended on the frame rate, or ran scene code during an
evaluation.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import DOWN, RED, UP, Dot, ValueTracker
from arachne.timeline.purity import Refused, prove_setter_body

from . import soundness_scenes
from .conftest import render

SOUNDNESS_SCENES = Path(soundness_scenes.__file__)


def _ticked_with(session_of, name: str, reason: str):
    session = session_of(name, SOUNDNESS_SCENES)
    play = session.record.plays[-1]
    assert play.route == "ticked", play.admission.reasons
    assert any(reason in r for r in play.admission.reasons), play.admission.reasons
    return session


def _same_at_shared_instants(scene_class, rates=(4, 30)) -> None:
    frames = {fps: render(scene_class, fps)[1] for fps in rates}
    for slow, fast in itertools.combinations(rates, 2):
        shared = [k for k in range(len(frames[slow])) if k * fast % slow == 0]
        for k in shared:
            assert np.array_equal(frames[slow][k], frames[fast][k * fast // slow]), (
                slow,
                fast,
                k,
            )


# 1. Changing an array the callback did not make.


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("ArrayAugmented", "changes in place a name or value it did not make"),
        ("ArrayOutKeyword", "out= or where="),
        ("ArrayOutPositional", "is given an output array"),
    ],
)
def test_a_callback_writing_into_an_array_it_read_ticks(session_of, name, reason):
    _ticked_with(session_of, name, reason)
    _same_at_shared_instants(getattr(soundness_scenes, name))


# 2. A setter given less than its whole argument.


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("PartialFill", "a conditional decides what is set"),
        ("NoneFill", "is given None"),
        ("NarrowedColour", "arguments its property test does not cover"),
    ],
)
def test_a_setter_that_may_leave_its_output_unassigned_ticks(session_of, name, reason):
    _ticked_with(session_of, name, reason)


def test_a_partial_fill_is_the_same_whatever_the_frame_rate():
    _same_at_shared_instants(soundness_scenes.PartialFill, (4, 15, 30))


def test_a_conditional_helper_value_reaching_a_setter_as_none_is_refused():
    d = Dot()

    def pick(a):
        return RED if a > 0.5 else None

    with pytest.raises(Refused, match="given None"):
        prove_setter_body(lambda m, a: m.set_fill(pick(a)), d, None, takes_alpha=True)


# 3. A callback part's timing.


def test_a_callback_part_with_a_scene_rate_function_ticks(session_of):
    _ticked_with(session_of, "RandomRate", "rate function")


def test_a_random_rate_function_is_the_same_whatever_the_frame_rate():
    _same_at_shared_instants(soundness_scenes.RandomRate)


# 4. A callback part's writes in a composition.


def test_a_callback_part_writing_what_another_part_writes_ticks(session_of):
    _ticked_with(session_of, "CallbackOverlap", "write the same mobject")
    _same_at_shared_instants(soundness_scenes.CallbackOverlap)


# 5. What a read calls.


def test_a_graph_read_calling_an_unproven_function_ticks(session_of):
    _ticked_with(session_of, "CountingGraph", "underlying_function")
    _same_at_shared_instants(soundness_scenes.CountingGraph)


# 6. The mobjects' own methods.


def test_a_mobject_overriding_an_engine_method_takes_a_transform_off_route_1(
    session_of,
):
    session = _ticked_with(
        session_of, "CountedTransform", "overrides the engine's interpolate"
    )
    counter = session.module.CountingDot.calls
    before = len(counter)
    session.state_at(Fraction(1, 7))
    session.frame(Fraction(1, 7))
    assert len(counter) == before


# 7. An array a path binds, as the play saw it.


def test_a_rotation_about_a_read_only_view_is_answered_as_the_play_saw_it(session_of):
    """The script changes the view's base after the play; an answer between
    two evaluated instants still rotates about the centre the play had.
    """
    session = session_of("ViewCentre", SOUNDNESS_SCENES)
    assert session.record.plays[0].route == "trusted"
    analytic = session.state_at(Fraction(1, 7))
    assert analytic.answer == "analytic"
    scene, _ = render(soundness_scenes.ViewCentre, 7)
    recorded = scene.record.state_at(Fraction(1, 7))
    assert recorded.answer == "recorded"
    assert analytic.nodes == recorded.nodes


# 8. Bindings, one per holder.


def test_two_cells_of_one_name_are_two_bindings():
    first, second = ValueTracker(1), ValueTracker(2)

    def reader(t):
        return lambda: t.get_value()

    f, g = reader(first), reader(second)
    proof = prove_setter_body(lambda m: m.set_x(f() + g()), Dot(), None)
    cells = [b for b in proof.bindings if b.where == "cell" and b.name == "t"]
    assert len(cells) == 2
    assert {id(b.value) for b in cells} == {id(first), id(second)}
    assert cells[0].holder is not cells[1].holder


def test_default_arguments_and_nested_contents_are_bound():
    tracker = ValueTracker(1)
    offsets = [[0.0, 1.0], np.array([2.0, 3.0])]

    def place(m, k=tracker):
        m.set_x(k.get_value() + offsets[0][1] + offsets[1][0])

    proof = prove_setter_body(place, Dot(), None)
    (default,) = [b for b in proof.bindings if b.where == "default"]
    assert default.name == "k"
    assert default.value is tracker
    assert default.holder is place
    (nested,) = [b for b in proof.bindings if b.name == "offsets"]
    inner_list, inner_array = nested.contents
    assert inner_list == (0.0, 1.0)
    assert np.array_equal(inner_array, [2.0, 3.0])
    assert not inner_array.flags.writeable
    offsets[1][0] = 9
    assert inner_array[0] == 2.0


# 9. The cameras of a MultiCamera's displays.


def test_a_display_camera_is_drawn_with_its_own_values_at_each_instant(session_of):
    """The zoomed camera's background is red in the first play and blue in
    the second; frames drawn out of order still show each.
    """
    session = session_of("ZoomBackground", SOUNDNESS_SCENES)
    later = np.asarray(session.frame(Fraction(3, 4)))
    earlier = np.asarray(session.frame(Fraction(1, 4)))
    _, frames = render(soundness_scenes.ZoomBackground, 8)
    assert np.array_equal(earlier, frames[2])
    assert np.array_equal(later, frames[6])
    assert not np.array_equal(earlier, later)


def test_setters_are_proven_on_their_argument_domain_only():
    d, other = Dot(), Dot()
    prove_setter_body(lambda m: m.next_to(other, UP, buff=0.1), d, None)
    with pytest.raises(Refused, match="does not cover"):
        prove_setter_body(lambda m: m.next_to(other, UP, aligned_edge=DOWN), d, None)
    with pytest.raises(Refused, match="does not cover"):
        prove_setter_body(
            lambda m: m.move_to(UP, coor_mask=np.array([1, 0, 0])), d, None
        )
