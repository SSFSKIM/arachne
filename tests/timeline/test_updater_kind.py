"""Two answers read once instead of at every call: whether an updater takes
``dt`` (while the speed levers are on), and the families a trusted play's
animations zip (under ``vectorize_trusted``). Each must be the answer a read
at that call would give.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from arachne import (
    RIGHT,
    AnimationGroup,
    ApplyFunction,
    ApplyPointwiseFunction,
    Axes,
    Circle,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    LaggedStart,
    Scene,
    Square,
    Succession,
    Transform,
    VGroup,
    Wait,
    linear,
    tempconfig,
)
from arachne.animation import animation as animation_module
from arachne.animation.animation import keeping_families, zipped_families
from arachne.animation.transform import Transform as TransformClass
from arachne.lint.bridge import watch
from arachne.mobject.mobject import Mobject
from arachne.mobject.updater_kind import takes_dt

from .conftest import render
from .test_video_output import CASES


def read(updater: Any) -> bool:
    """The kind, read as Manim CE reads it at every call."""
    return "dt" in inspect.signature(updater).parameters


@pytest.fixture(params=[True, False], ids=["levers_on", "levers_off"])
def levers(request):
    with tempconfig({"speed_levers": request.param}):
        yield request.param


class Reads:
    """Every object ``inspect.signature`` was asked to read."""

    def __init__(self) -> None:
        self.objects: list[Any] = []

    def __getitem__(self, obj: Any) -> int:
        return sum(1 for seen in self.objects if seen is obj)


@pytest.fixture
def reads(monkeypatch) -> Reads:
    """How often ``inspect.signature`` reads each object."""
    found = Reads()
    real = inspect.signature

    def counted(obj, *args, **kwargs):
        found.objects.append(obj)
        return real(obj, *args, **kwargs)

    monkeypatch.setattr(inspect, "signature", counted)
    return found


# The updater kind


def plain(mob):
    pass


def timed(mob, dt):
    pass


def keyword_dt(mob, *, dt=0.0):
    pass


def anything(*args):
    pass


class Holder:
    def plain(self, mob):
        pass

    def timed(self, mob, dt):
        pass

    def receiver_named_dt(dt, mob):  # noqa: N805
        pass


class Unhashable:
    """A receiver that cannot key a dictionary: its bound methods still can."""

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other):
        return True

    def timed(self, mob, dt):
        pass


class Calls:
    def __call__(self, mob, dt):
        pass


KINDS = [
    pytest.param(plain, id="function"),
    pytest.param(timed, id="function_dt"),
    pytest.param(keyword_dt, id="keyword_only_dt"),
    pytest.param(anything, id="var_positional"),
    pytest.param(lambda m: None, id="lambda"),
    pytest.param(lambda m, dt: None, id="lambda_dt"),
    pytest.param(Holder().plain, id="method"),
    pytest.param(Holder().timed, id="method_dt"),
    pytest.param(Holder().receiver_named_dt, id="method_receiver_named_dt"),
    pytest.param(Unhashable().timed, id="method_of_unhashable"),
    pytest.param(Calls(), id="callable_object"),
    pytest.param(functools.partial(timed, None), id="partial_binding_mob"),
    pytest.param(functools.partial(plain), id="partial"),
    pytest.param(functools.partial(timed, dt=1.0), id="partial_dt_keyword"),
    pytest.param(functools.wraps(timed)(lambda *a: None), id="wrapped"),
]


@pytest.mark.parametrize("updater", KINDS)
def test_the_kind_is_the_one_its_signature_gives(levers, updater):
    expected = read(updater)
    assert takes_dt(updater) is expected
    assert takes_dt(updater) is expected


def test_a_function_or_method_is_read_once_while_it_is_unchanged(reads):
    def fresh(mob):
        pass

    def fresh_dt(mob, dt):
        pass

    updaters = [fresh, fresh_dt, Holder().timed, lambda m, dt: None]
    for _ in range(5):
        for updater in updaters:
            takes_dt(updater)
    # A bound method is a new object at each attribute read; the one an
    # updater list holds is the same one at every call.
    assert [reads[u] for u in updaters] == [1, 1, 1, 1]


def test_with_the_levers_off_every_call_reads_the_signature(reads):
    with tempconfig({"speed_levers": False}):
        for _ in range(3):
            takes_dt(timed)
    assert reads[timed] == 3


def test_a_function_whose_signature_is_set_or_replaced_reads_the_new_one(reads):
    def updater(mob):
        pass

    assert takes_dt(updater) is False
    updater.__signature__ = inspect.signature(timed)  # type: ignore[attr-defined]
    assert takes_dt(updater) is read(updater) is True
    updater.__signature__ = inspect.signature(plain)  # type: ignore[attr-defined]
    assert takes_dt(updater) is read(updater) is False
    updater.__signature__ = None  # type: ignore[attr-defined]
    assert takes_dt(updater) is read(updater) is False
    del updater.__signature__  # type: ignore[attr-defined]
    updater.__code__ = timed.__code__
    assert takes_dt(updater) is read(updater) is True
    updater.__code__ = plain.__code__
    assert takes_dt(updater) is read(updater) is False


def test_an_updater_whose_signature_is_set_after_it_was_added_is_called_as_ce_calls_it():
    received = []

    def updater(mob, *rest):
        received.append(rest)

    mob = Mobject()
    mob.add_updater(updater)
    mob.update(0.25)
    updater.__signature__ = inspect.signature(timed)  # type: ignore[attr-defined]
    mob.update(0.5)
    assert mob.get_time_based_updaters() == [updater]
    assert mob.has_time_based_updater()
    del updater.__signature__  # type: ignore[attr-defined]
    mob.update(0.75)
    assert received == [(), (0.5,), ()]
    assert not mob.has_time_based_updater()


def test_a_method_follows_its_function(reads):
    class Owner:
        def update(self, mob):
            pass

    method = Owner().update
    assert takes_dt(method) is False
    assert takes_dt(method) is False
    assert reads[method] == 1
    method.__func__.__code__ = Holder.timed.__code__
    assert takes_dt(method) is read(method) is True
    method.__func__.__signature__ = inspect.signature(Holder.plain)  # type: ignore[attr-defined]
    assert takes_dt(method) is read(method) is False


def test_a_callable_object_is_read_at_every_call(reads):
    class Updater:
        def __call__(self, mob):
            pass

    updater = Updater()
    assert takes_dt(updater) is False
    Updater.__call__ = Calls.__call__  # type: ignore[method-assign]
    assert takes_dt(updater) is read(updater) is True
    updater.__signature__ = inspect.signature(plain)  # type: ignore[attr-defined]
    assert takes_dt(updater) is read(updater) is False
    # Three reads by takes_dt, and two by read().
    assert reads[updater] == 5


def test_a_partial_is_read_at_every_call(reads):
    updater = functools.partial(plain)
    assert takes_dt(updater) is False
    updater.__signature__ = inspect.signature(timed)  # type: ignore[attr-defined]
    assert takes_dt(updater) is read(updater) is True
    assert reads[updater] == 3


def test_a_wrapped_function_follows_what_it_wraps(reads):
    updater = functools.wraps(timed)(lambda *a: None)
    assert takes_dt(updater) is True
    updater.__wrapped__ = plain  # type: ignore[attr-defined]
    assert takes_dt(updater) is read(updater) is False
    assert reads[updater] == 3


def test_a_signature_that_cannot_be_read_raises_at_every_call(reads):
    def updater(mob):
        pass

    updater.__signature__ = 5  # type: ignore[attr-defined]
    for _ in range(2):
        with pytest.raises(TypeError):
            takes_dt(updater)
    assert reads[updater] == 2
    del updater.__signature__  # type: ignore[attr-defined]
    assert takes_dt(updater) is False
    with pytest.raises(TypeError):
        takes_dt(5)


class Integrates(Scene):
    """A one-second play at 60 fps beside two updaters: one takes dt."""

    def construct(self):
        def drift(mob, dt):
            mob.shift(RIGHT * dt)

        def follow(mob):
            mob.set_y(0)

        self.read_updaters = (drift, follow)
        dot = Dot()
        dot.add_updater(drift)
        dot.add_updater(follow)
        self.add(dot)
        self.play(Create(Square()), run_time=1)


def test_across_a_sixty_frame_play_each_updater_is_read_once(reads):
    scene, frames = render(Integrates, 60)
    assert len(frames) == 60
    assert [reads[u] for u in scene.read_updaters] == [1, 1]


def test_across_a_sixty_frame_play_with_only_vectorize_trusted_off_each_is_read_once(
    reads,
):
    # The kind answers to the master switch alone.
    scene, frames = render(Integrates, 60, vectorize_trusted=False)
    assert len(frames) == 60
    assert [reads[u] for u in scene.read_updaters] == [1, 1]


def test_across_a_sixty_frame_play_with_the_levers_off_each_call_reads(reads):
    scene, frames = render(Integrates, 60, speed_levers=False)
    assert all(reads[u] >= 60 for u in scene.read_updaters)


# The trusted families zip


@contextmanager
def zips_taken() -> Iterator[list[int]]:
    """Count the families zips an interpolation takes: calls of a
    ``get_all_families_zipped`` made by :func:`zipped_families`.
    """
    taken = [0]
    zips = {
        animation_module.Animation.get_all_families_zipped.__code__,
        TransformClass.get_all_families_zipped.__code__,
    }
    caller = zipped_families.__code__
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if (
            event == "call"
            and frame.f_code in zips
            and frame.f_back is not None
            and frame.f_back.f_code is caller
        ):
            taken[0] += 1

    sys.setprofile(profile)
    try:
        yield taken
    finally:
        sys.setprofile(previous)


class Morphs(Scene):
    """One trusted play evaluated frame by frame (its path is an arc, which
    the vectorized path leaves scalar), sixty frames at 60 fps.
    """

    def construct(self):
        self.play(Transform(Square(), Circle(), path_arc=np.pi / 2), run_time=1)


def test_a_trusted_play_zips_its_families_once_while_it_runs():
    with zips_taken() as taken:
        scene, frames = render(Morphs, 60)
    plays = scene.timeline.ensure_record().plays
    assert [(play.route, play.vectorized) for play in plays] == [("trusted", False)]
    assert len(frames) == 60
    # Its begin and its finish zip as they always have; the sixty frames
    # between them, once.
    assert taken[0] == 3
    with zips_taken() as taken:
        render(Morphs, 60, speed_levers=False)
    assert taken[0] == 62
    with zips_taken() as taken:
        render(Morphs, 60, vectorize_trusted=False)
    assert taken[0] == 62


def test_a_trusted_play_that_runs_the_scenes_code_zips_at_every_frame():
    for scene_class in SCENE_CODE_SCENES:
        with zips_taken() as on:
            scene, _ = render(scene_class, 15)
        with zips_taken() as off:
            render(scene_class, 15, speed_levers=False)
        routes = [play.route for play in scene.timeline.ensure_record().plays]
        assert "trusted" in routes, (scene_class.__name__, routes)
        assert on[0] == off[0], scene_class.__name__


def same(left: list[tuple], right: list[tuple]) -> bool:
    return len(left) == len(right) and all(
        len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))
        for a, b in zip(left, right, strict=True)
    )


def test_a_kept_zip_is_the_zip_and_is_taken_again_when_the_animation_begins_again():
    transform = Transform(Square(), Circle())
    transform.begin()
    with keeping_families([transform]):
        first = zipped_families(transform)
        assert zipped_families(transform) is first
        assert same(first, list(transform.get_all_families_zipped()))
        transform.begin()
        again = zipped_families(transform)
        assert again is not first
        assert same(again, list(transform.get_all_families_zipped()))
    outside = zipped_families(transform)
    assert outside is not again
    assert zipped_families(transform) is not outside


def test_an_animation_the_play_does_not_hold_zips_at_every_call():
    held, other = Transform(Square(), Circle()), Create(Square())
    held.begin()
    other.begin()
    with keeping_families([held]):
        assert zipped_families(other) is not zipped_families(other)


class Staggered(Scene):
    """Trusted plays whose parts begin while the play runs."""

    def construct(self):
        square = Square()
        self.play(Succession(Create(square), Transform(square, Circle())), run_time=1)
        dots = [Dot(RIGHT * i) for i in range(3)]
        self.play(LaggedStart(*(FadeIn(d) for d in dots), lag_ratio=0.5), run_time=1)


class CounterMoves(Scene):
    """A trusted play whose frame hook changes its mobject's members: a
    number shifted while the scene's ``update_to_time`` counts into it.
    """

    def construct(self):
        self.count = DecimalNumber(1, num_decimal_places=0)
        self.add(self.count)
        self.play(self.count.animate.shift(2 * RIGHT), run_time=0.5, rate_func=linear)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "count") and self.count.number < 9:
            self.count.set_value(self.count.number + 1)


class GroupLoses(Scene):
    """A trusted play whose frame hook takes a member out of the group it
    creates.
    """

    def construct(self):
        self.group = VGroup(Square().shift(2 * RIGHT), Square().shift(-2 * RIGHT))
        self.play(Create(self.group), run_time=1, rate_func=linear)

    def update_to_time(self, t):
        super().update_to_time(t)
        group = getattr(self, "group", None)
        if group is not None and t > 0.3 and len(group) == 2:
            group.remove(group[0])


class WatchLoses(Scene):
    """A trusted play beside a watched function that takes a member out of
    the group the play creates.
    """

    def construct(self):
        axes = Axes()
        group = VGroup(Square().shift(2 * RIGHT), Square().shift(-2 * RIGHT))
        reads = [0]

        def point():
            reads[0] += 1
            if reads[0] == 6 and len(group) == 2:
                group.remove(group[0])
            return axes.c2p(0, 0)

        watch("axes.outside", point, reference=axes)
        self.play(Create(group), run_time=1, rate_func=linear)


class AddLoses(Scene):
    """A trusted play whose part begins mid-play and calls the scene's own
    ``add``, which takes a member out of a group another part creates.
    """

    def construct(self):
        self.group = VGroup(Square().shift(2 * RIGHT), Square().shift(-2 * RIGHT))
        self.play(
            AnimationGroup(Create(self.group), Succession(Wait(0.4), FadeIn(Dot()))),
            run_time=1,
            rate_func=linear,
        )

    def add(self, *mobjects):
        group = getattr(self, "group", None)
        if (
            group is not None
            and len(group) == 2
            and self.animations
            and any(type(m) is Dot for m in mobjects)
        ):
            group.remove(group[0])
        return super().add(*mobjects)


class FunctionLoses(Scene):
    """A trusted play whose part begins mid-play and calls the scene's
    function as it begins, which takes a member out of a group another part
    creates.
    """

    def construct(self):
        group = VGroup(Square().shift(2 * RIGHT), Square().shift(-2 * RIGHT))

        def moved(mob):
            if len(group) == 2:
                group.remove(group[0])
            return mob.shift(RIGHT)

        self.play(
            AnimationGroup(
                Create(group), Succession(Wait(0.4), ApplyFunction(moved, Dot()))
            ),
            run_time=1,
            rate_func=linear,
        )


class PointwiseLoses(Scene):
    """As :class:`FunctionLoses`, the function applied to each point."""

    def construct(self):
        group = VGroup(Square().shift(2 * RIGHT), Square().shift(-2 * RIGHT))

        def moved(point):
            if len(group) == 2:
                group.remove(group[0])
            return point + RIGHT

        self.play(
            AnimationGroup(
                Create(group),
                Succession(Wait(0.4), ApplyPointwiseFunction(moved, Dot())),
            ),
            run_time=1,
            rate_func=linear,
        )


#: Scenes whose trusted plays run the scene's code while they run.
SCENE_CODE_SCENES = (
    CounterMoves,
    GroupLoses,
    WatchLoses,
    AddLoses,
    FunctionLoses,
    PointwiseLoses,
)

FRAME_SCENES: list[Any] = [
    *(pytest.param(case.values[0], id=case.id) for case in CASES),
    pytest.param(Staggered, id="parts_begin_while_running"),
    pytest.param(Integrates, id="updaters"),
    pytest.param(CounterMoves, id="frame_hook_sets_a_number"),
    pytest.param(GroupLoses, id="frame_hook_removes_a_member"),
    pytest.param(WatchLoses, id="watch_removes_a_member"),
    pytest.param(AddLoses, id="scene_add_removes_a_member"),
    pytest.param(FunctionLoses, id="function_at_begin_removes_a_member"),
    pytest.param(PointwiseLoses, id="pointwise_function_removes_a_member"),
]


@pytest.mark.parametrize("scene_class", FRAME_SCENES)
def test_frames_are_the_same_with_the_lever_on_and_off(
    scene_class: Callable[[], Scene],
):
    on_scene, on = render(scene_class, 15, vectorize_trusted=True)
    _, off = render(scene_class, 15, speed_levers=False)
    routes = [play.route for play in on_scene.timeline.ensure_record().plays]
    if scene_class is Staggered:
        assert routes == ["trusted", "trusted"]
    if scene_class in SCENE_CODE_SCENES:
        assert "trusted" in routes, routes
    assert len(on) == len(off)
    for index, (a, b) in enumerate(zip(on, off, strict=True)):
        assert np.array_equal(a, b), f"frame {index}"
