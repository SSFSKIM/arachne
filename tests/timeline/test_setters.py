"""Route 2's setter table, by property: each admitted (receiver type, method)
pair, applied after any sequence of earlier calls from a start state, gives
the state one call gives from that start state. Pairs whose result depends on
the geometry they are given are not admitted.
"""

from __future__ import annotations

import inspect
import random
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from arachne import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    DecimalNumber,
    Dot,
    Line,
    Mobject,
    Square,
    ValueTracker,
    VMobject,
)
from arachne.timeline.setters import SETTERS, Setter, _resolve, setter_for

#: Earlier call sequences tried per pair, and their longest length.
TRIALS = 25
LONGEST = 5

COLORS = [RED, BLUE, GREEN, YELLOW]
DIRECTIONS = [UP, DOWN, LEFT, RIGHT, UP + RIGHT]


def _point(draw: random.Random) -> np.ndarray:
    return np.array([draw.uniform(-4, 4), draw.uniform(-3, 3), draw.uniform(-1, 1)])


def _corners(draw: random.Random) -> list[np.ndarray]:
    return [_point(draw) for _ in range(draw.randint(2, 6))]


#: Per admitted pair: the start state, and a draw of one call's arguments.
#: Arguments that are mobjects are built inside the call, as a callback builds
#: them.
CASES: dict[
    tuple[str, str], tuple[Callable[[], Any], Callable[[random.Random], tuple]]
] = {
    ("arachne.mobject.mobject.Mobject", "move_to"): (
        lambda: Square().shift(LEFT),
        lambda d: (d.choice([_point, lambda e: Dot(_point(e))])(d),),
    ),
    ("arachne.mobject.mobject.Mobject", "set_x"): (
        lambda: Square().shift(UP),
        lambda d: (d.uniform(-4, 4),),
    ),
    ("arachne.mobject.mobject.Mobject", "set_y"): (
        lambda: Square().shift(RIGHT),
        lambda d: (d.uniform(-3, 3),),
    ),
    ("arachne.mobject.mobject.Mobject", "set_z"): (
        lambda: Square(),
        lambda d: (d.uniform(-1, 1),),
    ),
    ("arachne.mobject.mobject.Mobject", "next_to"): (
        lambda: Square(side_length=0.5),
        lambda d: (Dot(_point(d)), d.choice(DIRECTIONS), d.uniform(0, 1)),
    ),
    ("arachne.mobject.mobject.Mobject", "set_color"): (
        lambda: Mobject(),
        lambda d: (d.choice(COLORS),),
    ),
    ("arachne.mobject.mobject.Mobject", "become"): (
        lambda: Square(),
        lambda d: (
            d.choice([Circle, Square])().move_to(_point(d)).set_color(d.choice(COLORS)),
        ),
    ),
    ("arachne.mobject.types.vectorized_mobject.VMobject", "set_color"): (
        lambda: Square(),
        lambda d: (d.choice(COLORS),),
    ),
    ("arachne.mobject.types.vectorized_mobject.VMobject", "set_fill"): (
        lambda: Square(),
        lambda d: (d.choice(COLORS), d.uniform(0, 1)),
    ),
    ("arachne.mobject.types.vectorized_mobject.VMobject", "set_stroke"): (
        lambda: Square(),
        lambda d: (d.choice(COLORS), d.uniform(0, 8), d.uniform(0, 1)),
    ),
    ("arachne.mobject.types.vectorized_mobject.VMobject", "set_opacity"): (
        lambda: Square().set_fill(RED, 0.5),
        lambda d: (d.uniform(0, 1),),
    ),
    ("arachne.mobject.types.vectorized_mobject.VMobject", "set_points_as_corners"): (
        lambda: VMobject(),
        lambda d: (_corners(d),),
    ),
    ("arachne.mobject.value_tracker.ValueTracker", "set_value"): (
        lambda: ValueTracker(1.5),
        lambda d: (d.uniform(-100, 100),),
    ),
    ("arachne.mobject.text.numbers.DecimalNumber", "set_value"): (
        lambda: DecimalNumber(3.14, color=RED).shift(LEFT),
        lambda d: (
            d.choice([0, 1, -2.5, 10.25, 123.456, -0.5, d.uniform(-1000, 1000)]),
        ),
    ),
}


def _state(mob: Mobject) -> list[tuple[str, Any]]:
    """Everything a frame or a query reads of ``mob``'s family."""
    state: list[tuple[str, Any]] = []
    for member in mob.get_family():
        state.append(("type", type(member).__name__))
        state.append(("points", np.array(member.points, dtype=float)))
        state.append(("color", str(member.color)))
        if isinstance(member, VMobject):
            state.append(("fill", np.array(member.get_fill_rgbas(), dtype=float)))
            state.append(("stroke", np.array(member.get_stroke_rgbas(), dtype=float)))
            state.append(("width", float(member.get_stroke_width())))
        if isinstance(member, ValueTracker):
            state.append(("value", float(member.get_value())))
    return state


def _same(first: list[tuple[str, Any]], second: list[tuple[str, Any]]) -> bool:
    """The same state, to the rounding of a translation (``move_to`` after
    ``move_to`` shifts twice where one call shifts once).
    """
    if [key for key, _ in first] != [key for key, _ in second]:
        return False
    for (_, a), (_, b) in zip(first, second, strict=True):
        if isinstance(a, np.ndarray):
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-9, atol=1e-9):
                return False
        elif isinstance(a, float):
            if not np.isclose(a, b, rtol=1e-9, atol=1e-9):
                return False
        elif a != b:
            return False
    return True


def history_free(
    start: Callable[[], Any],
    call: Callable[[Any, tuple], Any],
    arguments: Callable[[random.Random], tuple],
    seed: int,
) -> bool:
    """Whether ``call`` with each drawn final arguments, after every drawn
    sequence of earlier calls, leaves what it leaves applied once to a fresh
    start state.
    """
    draw = random.Random(seed)
    for _ in range(TRIALS):
        earlier = [arguments(draw) for _ in range(draw.randint(1, LONGEST))]
        final = arguments(draw)
        once = start()
        call(once, final)
        after = start()
        for args in earlier:
            call(after, args)
        call(after, final)
        if not _same(_state(once), _state(after)):
            return False
    return True


def _rebuild(args: tuple) -> tuple:
    """Fresh copies of the mobject arguments, as a callback builds them anew
    in every call.
    """
    return tuple(a.copy() if isinstance(a, Mobject) else a for a in args)


def test_every_admitted_pair_has_a_property_case():
    assert {(s.owner, s.name) for s in SETTERS} == set(CASES)


@pytest.mark.parametrize(
    "setter", SETTERS, ids=lambda s: f"{s.owner.rsplit('.', 1)[1]}.{s.name}"
)
def test_a_setter_domain_is_the_leading_parameters_of_the_method(setter: Setter):
    function = _resolve(setter.owner).__dict__[setter.name]
    parameters = list(inspect.signature(function).parameters)[1:]
    assert parameters[: len(setter.params)] == list(setter.params)


def _shapes(setter: Setter) -> list[tuple[int, bool]]:
    """Every way a call may pass the domain: its first ``k`` parameters,
    the ones after the first by name or not.
    """
    return [
        (count, by_name)
        for count in range(1, len(setter.params) + 1)
        for by_name in ((False, True) if count > 1 else (False,))
    ]


@pytest.mark.parametrize(
    "setter", SETTERS, ids=lambda s: f"{s.owner.rsplit('.', 1)[1]}.{s.name}"
)
def test_an_admitted_setter_gives_one_state_whatever_came_before(setter: Setter):
    """On its whole domain: each number of leading arguments a call site may
    pass, positionally or by name (a call site passes the same ones each
    time).
    """
    start, arguments = CASES[(setter.owner, setter.name)]
    receiver = start()
    assert setter_for(receiver, setter.name) == setter
    for count, by_name in _shapes(setter):

        def call(mob: Any, args: tuple, count=count, by_name=by_name) -> None:
            args = _rebuild(args)[:count]
            if by_name:
                named = dict(zip(setter.params[1:count], args[1:], strict=True))
                getattr(mob, setter.name)(args[0], **named)
            else:
                getattr(mob, setter.name)(*args)

        assert history_free(start, call, arguments, seed=hash(setter.name) % 1000), (
            count,
            by_name,
        )


def _polyline() -> VMobject:
    return VMobject().set_points_as_corners([LEFT, 0.5 * UP, RIGHT])


def _line_through(d: random.Random) -> tuple:
    """A line's end drawn so that it sometimes meets its start."""
    return (LEFT, d.choice([LEFT, RIGHT, 2 * RIGHT, UP]))


@pytest.mark.parametrize(
    ("start", "name", "arguments"),
    [
        # A path reaching zero length collapses for good: CE scales it to
        # zero and no later call restores it (``Line`` rebuilds itself first).
        (_polyline, "put_start_and_end_on", _line_through),
        # Relative changes accumulate.
        (lambda: Square(), "shift", lambda d: (_point(d),)),
        (lambda: Square(), "scale", lambda d: (d.uniform(0.5, 2),)),
        (lambda: Square(), "rotate", lambda d: (d.uniform(0, 3),)),
        (lambda: ValueTracker(0), "increment_value", lambda d: (d.uniform(-1, 1),)),
    ],
    ids=["put_start_and_end_on", "shift", "scale", "rotate", "increment_value"],
)
def test_a_method_whose_result_depends_on_what_came_before_is_not_admitted(
    start, name, arguments
):
    assert setter_for(start(), name) is None

    def call(mob: Any, args: tuple) -> None:
        getattr(mob, name)(*args)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "put_start_and_end_on has been called")
        assert not history_free(start, call, arguments, seed=7)
    assert setter_for(Line(), "put_start_and_end_on") is None


def test_a_setter_is_admitted_by_its_receiver_type_not_its_name():
    class Custom(Square):
        def move_to(self, point, *args, **kwargs):
            return self.shift(point)

    assert setter_for(Square(), "move_to") is not None
    assert setter_for(Custom(), "move_to") is None
    assert setter_for(Custom(), "set_x") is None  # a user class is not the engine's
