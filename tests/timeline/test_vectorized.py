"""Trusted plays evaluated over their kept instants at once
(``vectorize_trusted``): every kept capture's content key, unrounded, and
every frame's bytes are what the scalar path gives, frame by frame.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Circle, ShowPassingFlash, Square, Transform, VGroup, linear
from arachne.mobject.mobject import Mobject
from arachne.mobject.placement import Placement
from arachne.mobject.types.vectorized_mobject import VMobject
from arachne.mobject.values import Paint
from arachne.timeline import vectorized
from arachne.timeline.draw import Drawing
from arachne.timeline.record import IDENTITY_FIELDS, RegisteredState
from arachne.timeline.routes import _resolve
from arachne.utils.bezier import integer_interpolate, interpolate, partial_bezier_points

from . import route_scenes, vectorized_scenes
from .conftest import render
from .test_routes import _assert_shared_frames_agree
from .test_updater_kind import SCENE_CODE_SCENES

VECTORIZED_SCENES = Path(vectorized_scenes.__file__)


def keys(scene) -> list[tuple[Fraction, bytes]]:
    """Every drawn capture of the scene's record, and its final state, with
    the content key of what it holds, unrounded.
    """
    record = scene.timeline.ensure_record()
    assert record.final is not None
    return [
        (capture.instant, RegisteredState(record, capture).content_key())
        for capture in [*record._captures, record.final]
    ]


def same_value(a, b) -> bool:
    """Whether two values a recorded state holds are the same, across two
    executions: arrays by type, dtype and bytes, numbers by type and repr,
    a paint field by field, a placement by its key, containers item by item,
    functions by their code, any other object by its type.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, Paint):
        return same_value(a.fields(), b.fields())
    if isinstance(a, Placement):
        return a.key == b.key
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if not a.dtype.hasobject:
            return a.tobytes() == b.tobytes()
        if a.dtype.names:
            return all(same_value(a[name], b[name]) for name in a.dtype.names)
        return all(map(same_value, a.ravel().tolist(), b.ravel().tolist()))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(map(same_value, a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(map(same_value, a.values(), b.values()))
    if a is None or isinstance(a, (bool, int, float, complex, str, np.generic)):
        return repr(a) == repr(b)
    code = getattr(a, "__code__", None)
    return code is None or code is getattr(b, "__code__", None)


def animation_states(scene) -> list[tuple[int, list[tuple[str, dict]]]]:
    """What the record keeps of each trusted play's animations, by capture."""
    return [
        (seq, [(type(animation).__name__, state) for animation, state in states])
        for play in scene.timeline.ensure_record().plays
        if play.route == "trusted"
        for seq, states in sorted(play._animation_states.items())
    ]


def node_values(record, capture) -> list[tuple[str, list[tuple[str, object]]]]:
    """Every node a capture reaches, with each value its state holds (none
    naming an object of this process).
    """
    return [
        (
            type(mob).__name__,
            [
                (key, value)
                for key, value in zip(state.keys, state.values, strict=True)
                if key not in IDENTITY_FIELDS
            ],
        )
        for mob, state in RegisteredState(record, capture).nodes()
    ]


def recorded_values(scene) -> list:
    """What every drawn capture of the scene's record, and its final state,
    holds, value by value.
    """
    record = scene.timeline.ensure_record()
    return [node_values(record, c) for c in [*record._captures, record.final]]


def evaluated_values(session, t: Fraction) -> list:
    """What every node the answer at ``t`` reaches holds once the record has
    brought it there: for an instant between captures, the play evaluated
    from its last capture.
    """
    record = session.record
    resolved = _resolve(record, t)
    capture = resolved.capture
    with Drawing(record) as drawing:
        drawing.show(capture, capture.draw_list())
        if resolved.prepare is not None:
            resolved.prepare(drawing)
        seen: set[int] = set()
        found = []
        for root in capture.draw_list():
            for mob in root.get_family():
                if id(mob) in seen:
                    continue
                seen.add(id(mob))
                found.append(
                    (
                        type(mob).__name__,
                        [
                            (key, value)
                            for key, value in vars(mob).items()
                            if key not in IDENTITY_FIELDS
                        ],
                    )
                )
    return found


def vectorized_plays(scene) -> list[bool]:
    return [play.vectorized for play in scene.timeline.ensure_record().plays]


def assert_same_as_scalar(scene_class, fps: int = 15, **settings):
    """Render ``scene_class`` with the levers on and with them off, and
    require equal content keys at every capture, every recorded value the
    same in type and bytes, the same animation states kept at each, and
    equal bytes at every frame. Returns the levers-on scene.
    """
    on_scene, on = render(scene_class, fps, **settings)
    off_scene, off = render(scene_class, fps, speed_levers=False)
    assert not any(vectorized_plays(off_scene))
    assert keys(on_scene) == keys(off_scene)
    on_values, off_values = recorded_values(on_scene), recorded_values(off_scene)
    for index, (a, b) in enumerate(zip(on_values, off_values, strict=True)):
        assert same_value(a, b), f"capture {index}"
    assert same_value(animation_states(on_scene), animation_states(off_scene))
    assert len(on) == len(off)
    for index, (a, b) in enumerate(zip(on, off, strict=True)):
        assert a.tobytes() == b.tobytes(), f"frame {index}"
    return on_scene


#: The plays of the trusted algebra (one per entry) that are vectorized:
#: blends on a straight path, reveals and parts that write nothing, composed
#: by groups that are not a Succession.
VECTORIZED = {
    "Add",
    "Animation",
    "AnimationGroup",
    "ApplyComplexFunction",
    "ApplyFunction",
    "ApplyMatrix",
    "ApplyMethod",
    "ApplyPointwiseFunction",
    "Broadcast",
    "Create",
    "FadeIn",
    "FadeOut",
    "FadeToColor",
    "Flash",
    "GrowArrow",
    "GrowFromCenter",
    "GrowFromEdge",
    "GrowFromPoint",
    "Indicate",
    "LaggedStart",
    "LaggedStartMap",
    "MethodAnimation",
    "MoveToTarget",
    "ReplacementTransform",
    "Restore",
    "ScaleInPlace",
    "ShowPassingFlash",
    "ShowPassingFlashWithThinningStrokeWidth",
    "ShrinkToCenter",
    "Transform",
    "TransformFromCopy",
    "TransformMatchingShapes",
    "Uncreate",
}


@pytest.mark.parametrize("name", sorted(route_scenes.TRUSTED_PLAYS))
def test_each_entry_of_the_algebra_is_what_the_scalar_path_gives(name):
    scene_class = getattr(route_scenes, f"Trusted{name}")
    scene = assert_same_as_scalar(scene_class)
    play = scene.timeline.ensure_record().plays[0]
    assert play.route == "trusted"
    assert play.vectorized is (name in VECTORIZED), play.vectorized


@pytest.mark.parametrize(
    "name", ["Grouped", "Lagged", "Nested", "Rated", "Revealed", "Copied"]
)
def test_compositions_and_rate_functions_are_what_the_scalar_path_gives(name):
    scene = assert_same_as_scalar(getattr(vectorized_scenes, name))
    plays = scene.timeline.ensure_record().plays
    assert [play.route for play in plays] == ["trusted"] * len(plays)
    assert all(vectorized_plays(scene)), vectorized_plays(scene)


def test_a_succession_keeps_the_scalar_path():
    """A part begun while the play runs may run the scene's code as it
    begins, which can change what another part interpolates: a play with a
    Succession, its events exact or found by bisection, is evaluated frame
    by frame.
    """
    scene = assert_same_as_scalar(vectorized_scenes.Successions)
    record = scene.timeline.ensure_record()
    assert [play.route for play in record.plays] == ["trusted"] * 2
    assert all(play.events for play in record.plays)
    assert not any(vectorized_plays(scene))


def test_a_leaf_outside_the_vectorized_kinds_keeps_the_whole_play_scalar():
    scene = assert_same_as_scalar(vectorized_scenes.Mixed)
    (play,) = scene.timeline.ensure_record().plays
    assert play.route == "trusted"
    assert not play.vectorized
    assert vectorized.refusal(play._animations) is not None


@pytest.mark.parametrize("name", ["SinglePrecisionColors", "SinglePrecisionWidths"])
def test_a_style_held_in_another_precision_keeps_the_whole_play_scalar(name):
    """A style in single precision is mixed in numpy's promoted arithmetic,
    which the vectorized path does not reproduce: the play stays scalar.
    """
    scene = assert_same_as_scalar(getattr(vectorized_scenes, name))
    (play,) = scene.timeline.ensure_record().plays
    assert play.route == "trusted"
    assert not play.vectorized
    assert "style" in vectorized.refusal(play._animations)


#: Scenes whose play computes with a number of the scene's own type, one
#: operand of the vectorized evaluator each.
COUNTED = [
    "CountedProportion",
    "CountedSquish",
    "CountedWidth",
    "CountedLag",
    "CountedFlash",
    "CountedRunTime",
    "CountedInert",
]


@pytest.mark.parametrize("name", COUNTED)
def test_a_number_of_the_scenes_own_type_keeps_the_whole_play_scalar(name):
    """A number whose class defines its own arithmetic runs the scene's code
    each time it is computed with: the play stays scalar, so that code runs
    when, and as often as, the scalar path runs it.
    """
    scene = assert_same_as_scalar(getattr(vectorized_scenes, name))
    play = scene.timeline.ensure_record().plays[0]
    assert play.route == "trusted"
    assert not play.vectorized
    assert "Counted" in vectorized.refusal(play._animations)


def test_a_play_with_one_kept_instant_is_not_batched():
    scene = assert_same_as_scalar(vectorized_scenes.Brief)
    assert vectorized_plays(scene) == [False, True]


@pytest.mark.parametrize("name", ["Transform", "Create", "LaggedStart"])
def test_the_lever_alone_turns_it_off(name):
    scene_class = getattr(route_scenes, f"Trusted{name}")
    scene, _ = render(scene_class, 15, vectorize_trusted=False)
    assert vectorized_plays(scene) == [False, False]


@pytest.mark.parametrize("name", ["UserRate", "UserSubclass", "Wiggle"])
def test_a_play_the_trusted_table_refuses_is_never_vectorized(name):
    scene_class = getattr(route_scenes, f"Refused{name}")
    scene = assert_same_as_scalar(scene_class)
    (play,) = scene.timeline.ensure_record().plays
    assert play.route != "trusted"
    assert not play.vectorized


@pytest.mark.parametrize("scene_class", SCENE_CODE_SCENES, ids=lambda c: c.__name__)
def test_a_trusted_play_that_runs_the_scenes_code_keeps_the_scalar_path(scene_class):
    """The scene's frame hooks, a watch read at every kept capture, and a
    part that begins while the play runs can each change what the play's
    animations interpolate after the play began.
    """
    scene = assert_same_as_scalar(scene_class)
    routes = [play.route for play in scene.timeline.ensure_record().plays]
    assert "trusted" in routes
    assert not any(vectorized_plays(scene))


@pytest.mark.parametrize("name", ["Grouped", "Revealed"])
def test_samples_agree_at_the_instants_frame_rates_share(name):
    scene_class = getattr(vectorized_scenes, name)
    for fps in (4, 15, 30, 60):
        scene, _ = render(scene_class, fps)
        assert all(vectorized_plays(scene)), fps
    _assert_shared_frames_agree(scene_class)


@contextmanager
def leaf_calls() -> Iterator[list[int]]:
    """Count calls of the mobject methods a blend and a reveal run on each
    leaf, made from the engine's code.
    """
    codes = {
        Mobject.interpolate.__code__,
        VMobject.pointwise_become_partial.__code__,
    }
    counted = [0]
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            counted[0] += 1

    sys.setprofile(profile)
    try:
        yield counted
    finally:
        sys.setprofile(previous)


def test_a_vectorized_play_calls_no_leaf_method_while_it_runs():
    """``Grouped`` has five leaves, each interpolated as its animation
    begins and as it finishes, and at none of the fifteen frames between;
    with the levers off, at every frame its part runs as well.
    """
    with leaf_calls() as on:
        scene, frames = render(vectorized_scenes.Grouped, 15)
    assert vectorized_plays(scene) == [True]
    assert len(frames) == 15
    assert on[0] == 5 * 2
    with leaf_calls() as off:
        render(vectorized_scenes.Grouped, 15, speed_levers=False)
    assert off[0] > 4 * on[0]


@pytest.mark.parametrize("name", ["Grouped", "Nested", "Revealed"])
def test_an_instant_between_captures_is_the_scalar_evaluation_there(session_of, name):
    on = session_of(name, VECTORIZED_SCENES)
    off = session_of(name, VECTORIZED_SCENES, speed_levers=False)
    assert all(play.vectorized for play in on.record.plays)
    assert not any(play.vectorized for play in off.record.plays)
    play = on.record.plays[0]
    grid = on.record.grid
    for k in range(int(play.duration * grid)):
        t = play.start + Fraction(2 * k + 1, 2 * grid)
        state_on, state_off = on.state_at(t), off.state_at(t)
        assert state_on.answer == "analytic"
        assert state_on.nodes == state_off.nodes, t
        assert same_value(evaluated_values(on, t), evaluated_values(off, t)), t
        assert np.asarray(on.frame(t)).tobytes() == np.asarray(off.frame(t)).tobytes()


def test_extra_instants_a_session_asks_for_are_batched(session_of):
    instants = [Fraction(1, 7), Fraction(2, 7), Fraction(3, 7)]
    on = session_of("Grouped", VECTORIZED_SCENES, grid=None, instants=instants)
    off = session_of(
        "Grouped", VECTORIZED_SCENES, grid=None, instants=instants, speed_levers=False
    )
    (play,) = on.record.plays
    assert play.vectorized
    assert play._instants == [Fraction(0), *instants]
    assert play.sample_keys(None) == off.record.plays[0].sample_keys(None)


# The arithmetic, against CE's scalar functions.

ALPHAS = np.array([0.0, -0.0, 1e-300, 0.1, 1 / 3, 0.5, 0.7, 1 - 1e-16, 1.0, 1.25, -0.3])


def test_a_batched_blend_is_ces_interpolate_at_each_alpha():
    rng = np.random.default_rng(3)
    start = rng.normal(size=(8, 3)) * 7
    start[0] = [0.0, -0.0, 1e300]
    end = rng.normal(size=(8, 3)) * 1e-3
    batched = vectorized.blend(start, end, ALPHAS)
    for alpha, row in zip(ALPHAS, batched, strict=True):
        assert row.tobytes() == interpolate(start, end, float(alpha)).tobytes(), alpha


def test_integer_interpolation_is_ces_at_each_alpha():
    alphas = np.array([*ALPHAS, 0.25, 0.75, 2 / 3, 0.999, 1 / 9])
    for curves in (1, 3, 7, 12):
        indices, residues = vectorized.integer_interpolation(curves, alphas)
        for alpha, index, residue in zip(alphas, indices, residues, strict=True):
            expected = integer_interpolate(0, curves, float(alpha))
            assert (int(index), float(residue)) == expected, (curves, alpha)


def test_batched_partial_curves_are_ces_partial_bezier_points():
    """Curves read at every offset of a longer path, and windows that touch
    the ends (``a == 1``, ``b == 0``) and the integers CE passes.
    """
    rng = np.random.default_rng(5)
    path = rng.normal(size=(4 * 9, 3)) * 3
    windows = [
        (0, 0.5),
        (0.0, 1),
        (0.25, 1),
        (1, 1),
        (1.0, 0.3),
        (0.3, 0),
        (0, 0),
        *((float(a), float(b)) for a, b in np.sort(rng.random((40, 2)), axis=1)),
    ]
    indices = rng.integers(0, 9, size=len(windows))
    curves = path.reshape(9, 4, 3)[indices]
    lows = np.array([a for a, _ in windows], dtype=float)
    highs = np.array([b for _, b in windows], dtype=float)
    batched = vectorized.partial_curves(curves, lows, highs)
    for (a, b), index, row in zip(windows, indices, batched, strict=True):
        piece = path[4 * index : 4 * (index + 1)]
        assert row.tobytes() == partial_bezier_points(piece, a, b).tobytes(), (a, b)


@pytest.mark.parametrize("width", [0.0, 0.1, 0.6, 1.5])
def test_a_flash_window_is_ces_at_each_alpha(width):
    flash = ShowPassingFlash(Circle(), time_width=width)
    alphas = np.array([*ALPHAS, 0.05, 0.4, 0.95, 1 / (1 + width)])
    lower, upper = vectorized._flash_bounds(flash, alphas)
    for alpha, low, high in zip(alphas, lower, upper, strict=True):
        a, b = flash._get_bounds(float(alpha))
        assert (repr(float(low)), repr(float(high))) == (
            repr(float(a)),
            repr(float(b)),
        ), alpha


def test_at_alpha_one_a_blend_takes_the_end_style_itself():
    """As CE's ``interpolate_color`` does: the end's own style values,
    shared, not a mix equal to them.
    """

    def ended(levers: bool) -> tuple[VMobject, VMobject]:
        transform = Transform(Square().set_fill(opacity=0.5), Circle())
        transform.begin()
        if levers:
            instants = [Fraction(0), Fraction(1, 2), Fraction(1)]
            vector = vectorized.plan([transform], instants, Fraction(0))
            assert vector is not None
            assert vector.install(transform, Fraction(1))
        else:
            transform.interpolate(1.0)
        (leaf,) = transform.mobject.family_members_with_points()
        (end,) = transform.target_copy.family_members_with_points()
        return leaf, end

    for levers in (True, False):
        leaf, end = ended(levers)
        for name in ("fill_rgbas", "stroke_rgbas", "sheen_direction"):
            assert getattr(leaf, name) is getattr(end, name), (levers, name)


def _blended(levers: bool, lag: bool, at: Fraction) -> list[Paint]:
    """A Transform's leaves' paints at ``at``: from the plan, or interpolated."""
    start, end = Square(), Circle()
    for mob, width in ((start, 1.2), (end, 3.7)):
        mob.stroke_width = mob.background_stroke_width = mob.sheen_factor = width
    if lag:
        start, end = VGroup(start, start.copy()), VGroup(end, end.copy())
        for mob in (*start, *end):
            mob.stroke_width = mob.background_stroke_width = mob.sheen_factor = 2
    transform = Transform(start, end, rate_func=linear, lag_ratio=0.5 if lag else 0)
    transform.begin()
    if levers:
        instants = [Fraction(0), Fraction(1, 4)]
        vector = vectorized.plan([transform], instants, Fraction(0))
        assert vector is not None
        assert vector.install(transform, at)
    else:
        transform.interpolate(float(at))
    return [leaf._paint for leaf in transform.mobject.family_members_with_points()]


@pytest.mark.parametrize(
    ("lag", "at"),
    [(False, Fraction(1, 4)), (True, Fraction(0)), (True, Fraction(1, 4))],
)
def test_a_blends_style_numbers_have_the_scalar_paths_types(lag, at):
    """A float mixed stays a float, and an int mixed at a clamped zero stays
    an int, as CE's expression leaves them.
    """
    assert same_value(_blended(True, lag, at), _blended(False, lag, at))
