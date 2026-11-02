"""Admission is by proof: what the proof can't cover ticks, with a named
reason, and comes out the same whichever instants are evaluated.

Each scene here once passed for a state function or a trusted play while
its frames depended on the frame rate, or ran scene code during an
evaluation.
"""

from __future__ import annotations

import itertools
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import (
    BLUE,
    DOWN,
    RED,
    UP,
    Dot,
    Graph,
    ImageMobjectFromCamera,
    ParametricFunction,
    SVGMobject,
    ThreeDAxes,
    TracedPath,
    ValueTracker,
    Variable,
    VGroup,
    tempconfig,
)
from arachne.camera.camera import Camera
from arachne.timeline import setters
from arachne.timeline.dispatch import draw_methods, draw_refusal, engine_patches
from arachne.timeline.purity import Refused, prove_builder, prove_setter_body
from arachne.timeline.routes import RecordRefused

from . import soundness_scenes
from .conftest import SMALL, render

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
        ("ArrayOutKeyword", "numpy.add is given out=, which its entry does not admit"),
        ("ArrayOutPositional", "numpy.add is given 3 arguments by position, past"),
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


# Fix wave 2: admission by allowlist.


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("PutAlongAxis", "numpy.put_along_axis, which no library entry admits"),
        ("RandomColour", "random_color, which no library entry admits"),
    ],
)
def test_a_library_call_outside_the_table_ticks(session_of, name, reason):
    _ticked_with(session_of, name, reason)
    _same_at_shared_instants(getattr(soundness_scenes, name))


def test_a_value_picked_from_items_that_may_be_none_ticks(session_of):
    _ticked_with(session_of, "PickedNone", "is given None")
    _same_at_shared_instants(soundness_scenes.PickedNone, (4, 15, 30))


def test_a_value_picked_from_an_array_of_objects_ticks(session_of):
    _ticked_with(session_of, "ObjectArrayFill", "dtype='O'")
    _same_at_shared_instants(soundness_scenes.ObjectArrayFill, (4, 15, 30))


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            lambda m, a: m.set_fill(np.array([None, RED])[int(a > 0.5)]),
            "numpy.array is given None",
        ),
        (
            lambda m, a: m.set_fill(np.array([None, RED], dtype=object)[0]),
            "numpy.array is given dtype=object",
        ),
        (
            lambda m, a: m.set_fill(np.asarray([RED, None], "O")[0]),
            "numpy.asarray is given dtype='O'",
        ),
        (lambda m, a: m.set_fill(np.full(2, None)[0]), "numpy.full is given None"),
        (
            lambda m, a: m.set_fill(np.where(a > 0.5, None, RED)[()]),
            "numpy.where is given None",
        ),
        (
            lambda m, a: m.set_fill(RED, np.max([None, a])),
            "numpy.max is given None",
        ),
        (
            lambda m, a: m.set_fill(RED, np.zeros(2).astype("O")[0]),
            "astype is given dtype='O'",
        ),
    ],
    ids=["inferred", "object", "asarray", "full", "where", "max", "astype"],
)
def test_an_array_holding_none_or_objects_is_refused(body, reason):
    with pytest.raises(Refused, match=re.escape(reason)):
        prove_setter_body(body, Dot(), None, takes_alpha=True)


def test_an_array_of_numbers_written_out_is_admitted():
    prove_setter_body(
        lambda m, a: m.set_fill(RED, np.array([0.5, a], dtype="float")[1]),
        Dot(),
        None,
        takes_alpha=True,
    )


def test_a_stored_method_reads_its_receiver(session_of):
    _ticked_with(session_of, "BoundReceiver", "writes after it")
    _same_at_shared_instants(soundness_scenes.BoundReceiver)


def test_an_instance_interpolate_takes_a_transform_off_route_1(session_of):
    session = _ticked_with(session_of, "InstanceInterpolate", "on the instance")
    counter = session.module.InstanceInterpolate.calls
    before = len(counter)
    session.state_at(Fraction(1, 7))
    session.frame(Fraction(1, 7))
    assert len(counter) == before


def test_an_instance_setter_reached_through_a_group_ticks(session_of):
    _ticked_with(session_of, "InstanceSetFill", "on the instance")
    _same_at_shared_instants(soundness_scenes.InstanceSetFill)


# What else the prover admitted by default.


def test_an_animation_method_replaced_on_the_instance_takes_it_off_route_1(
    session_of,
):
    session = _ticked_with(
        session_of, "InstanceAnimation", "interpolate_mobject is replaced"
    )
    counter = session.module.InstanceAnimation.calls
    before = len(counter)
    session.state_at(Fraction(1, 7))
    session.frame(Fraction(1, 7))
    assert len(counter) == before


def test_a_mobject_whose_update_is_replaced_ticks(session_of):
    _ticked_with(session_of, "InstanceUpdate", "update is the scene's code")
    _same_at_shared_instants(soundness_scenes.InstanceUpdate)


def test_an_engine_method_replaced_by_the_scene_ticks_every_play(session_of):
    _ticked_with(session_of, "PatchedEngine", "replaces the engine's Dot.interpolate")
    assert engine_patches() == ()


def test_a_builder_calling_a_random_method_of_what_it_built_ticks(session_of):
    _ticked_with(session_of, "ShuffledBuilder", "it calls VGroup.shuffle")
    _same_at_shared_instants(soundness_scenes.ShuffledBuilder)


class _Arithmetic(float):
    """A number whose arithmetic is a scene's code."""

    def __add__(self, other):
        return float(self) + other + 1


class _Array(np.ndarray):
    """An array whose arithmetic is a scene's code."""


class _Callable:
    def __call__(self, t):
        return np.array([t, t, 0.0])


def test_values_are_registered_by_their_exact_type():
    d = Dot()
    number = _Arithmetic(1.0)
    with pytest.raises(Refused, match="_Arithmetic"):
        prove_setter_body(lambda m: m.set_x(number + 1), d, None)
    array = np.zeros(3).view(_Array)
    with pytest.raises(Refused, match="not an array of numbers"):
        prove_setter_body(lambda m: m.move_to(array + 1), d, None)


def test_a_ufunc_made_from_a_python_function_is_no_library_entry():
    d = Dot()
    made = np.frompyfunc(lambda x: x, 1, 1)
    with pytest.raises(Refused, match="no library entry admits"):
        prove_setter_body(lambda m, a: m.set_x(made(a)), d, None, takes_alpha=True)


def test_a_callable_handed_to_a_constructor_must_be_proven():
    shape = _Callable()
    with pytest.raises(Refused, match="no library entry admits"):
        prove_builder(lambda: ParametricFunction(shape, t_range=[0, 1]), None)


def test_a_function_mapped_over_an_array_it_did_not_make_may_not_change_it():
    rows = np.zeros((2, 3))

    def bump(row):
        row += 0.1
        return row

    with pytest.raises(Refused, match="did not make"):
        prove_setter_body(lambda m: m.move_to(list(map(bump, rows))[0]), Dot(), None)


_ROWS = [np.zeros(3)]


def _through_dict(m):
    q = {"a": _ROWS[0]}["a"]
    q += 0.02
    m.move_to(q)


def _through_dict_comprehension(m):
    q = {k: _ROWS[0] for k in "a"}["a"]  # noqa: C420
    q += 0.02
    m.move_to(q)


def _through_dict_call(m):
    q = dict([("a", _ROWS[0])])["a"]  # noqa: C406
    q += 0.02
    m.move_to(q)


def _through_concatenation(m):
    q = (_ROWS + [])[0]
    q += 0.02
    m.move_to(q)


@pytest.mark.parametrize(
    "drift",
    [
        _through_dict,
        _through_dict_comprehension,
        _through_dict_call,
        _through_concatenation,
    ],
)
def test_a_container_holding_an_array_it_read_still_holds_that_array(drift):
    with pytest.raises(Refused, match="did not make"):
        prove_setter_body(drift, Dot(), None)


def test_mobjects_listed_on_a_mobject_are_read():
    holder, label = Dot(), Dot()
    holder.labels = [label]
    proof = prove_setter_body(lambda m: m.set_x(holder.labels[0].get_x()), Dot(), None)
    assert any(read is label for read in proof.reads)


def test_methods_of_a_built_mobject_and_constructors_are_admitted_by_name_only():
    assert setters.fresh_method(VGroup, "shift") == "mobject"
    for name in ("shuffle", "save_image", "show", "add_updater", "apply_function"):
        assert setters.fresh_method(VGroup, name) is None, name
    assert setters.constructor_kind(Dot) == "reads"
    for cls in (Variable, ImageMobjectFromCamera, SVGMobject, Graph, TracedPath):
        assert setters.constructor_kind(cls) is None, cls


def test_setters_are_proven_on_their_argument_domain_only():
    d, other = Dot(), Dot()
    prove_setter_body(lambda m: m.next_to(other, UP, buff=0.1), d, None)
    with pytest.raises(Refused, match="does not cover"):
        prove_setter_body(lambda m: m.next_to(other, UP, aligned_edge=DOWN), d, None)
    with pytest.raises(Refused, match="does not cover"):
        prove_setter_body(
            lambda m: m.move_to(UP, coor_mask=np.array([1, 0, 0])), d, None
        )


# Drawing runs no scene code.


def _drawn_serially(fps: int) -> list[np.ndarray]:
    """``OverriddenFill``'s frames as CE draws them: the camera capturing the
    dot at each frame of each wait, with the colour of that wait.
    """
    cls = soundness_scenes.CountingFill
    frames: list[np.ndarray] = []
    with tempconfig(
        {**SMALL, "frame_rate": fps, "dry_run": True, "disable_caching": True}
    ):
        camera = soundness_scenes.OverriddenFill().renderer.camera
        dot = cls(radius=0.3)
        for colour in (RED, BLUE):
            cls.colour = colour
            camera.reset()
            camera.capture_mobjects([dot])
            frames.extend([camera.pixel_array.copy()] * (fps // 2))
    return frames


def test_the_draw_methods_are_read_from_the_camera_code():
    names = draw_methods(Camera)
    for name in (
        "get_fill_rgbas",
        "get_stroke_rgbas",
        "get_stroke_width",
        "get_gradient_start_and_end_points",
        "pixel_array",
    ):
        assert name in names, name
    assert "get_anchors" not in names


def test_a_mobject_overriding_a_draw_method_is_drawn_as_the_scene_runs(session_of):
    session = session_of("OverriddenFill", SOUNDNESS_SCENES)
    record = session.record
    assert record.refusal.startswith("draw.user_override: CountingFill.get_fill_rgbas:")
    calls = session.module.CountingFill.calls
    before = len(calls)
    with pytest.raises(RecordRefused, match="draw.user_override"):
        session.frame(Fraction(1, 4))
    with pytest.raises(RecordRefused):
        session.state_at(Fraction(1, 4))
    assert len(calls) == before
    _, frames = render(soundness_scenes.OverriddenFill, 6)
    expected = _drawn_serially(6)
    assert len(frames) == len(expected)
    for k, (frame, reference) in enumerate(zip(frames, expected, strict=True)):
        assert np.array_equal(frame, reference), k
    assert not np.array_equal(frames[0], frames[-1])


def test_a_mobject_overriding_a_draw_method_that_joins_mid_play_is_drawn_live(
    session_of,
):
    """Its play's frames were to be drawn from the record: the scene is
    executed again, drawing that play as it runs, as CE draws it.
    """
    session = session_of("FillJoinsMidPlay", SOUNDNESS_SCENES)
    assert session.record.refusal.startswith("draw.user_override: CountingFill.")
    scene, frames = render(soundness_scenes.FillJoinsMidPlay, 6)
    assert scene.record.execution.number == 2
    assert list(scene.record.execution.known) == [0]
    cls = soundness_scenes.CountingFill
    with tempconfig(
        {**SMALL, "frame_rate": 6, "dry_run": True, "disable_caching": True}
    ):
        camera = soundness_scenes.FillJoinsMidPlay().renderer.camera
        camera.reset()
        blank = camera.pixel_array.copy()
        cls.colour = RED
        camera.capture_mobjects([cls(radius=0.3)])
        dot = camera.pixel_array.copy()
    expected = [dot if k / 6 > 0.4 else blank for k in range(6)]
    assert len(frames) == len(expected)
    for k, (frame, reference) in enumerate(zip(frames, expected, strict=True)):
        assert np.array_equal(frame, reference), k


def test_what_the_engine_puts_on_an_instance_is_the_engine_s_draw():
    """ThreeDAxes gives its pieces gradient points of their own."""
    assert draw_refusal(Camera(), [ThreeDAxes()]) is None
