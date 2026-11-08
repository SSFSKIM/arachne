"""Trusted plays evaluated at all their kept instants at once
(``vectorize_trusted``).

A trusted play (route 1) is a known function of the instant
(:mod:`.trusted`). When every leaf of it is a straight-path blend or a
partial reveal (the two terms :meth:`.PlayRecord.leaf_terms` describes), or
writes nothing, and its leaves are composed by CE's group timing, the route
computes every leaf's value at every instant it will keep before it
evaluates the first: the alphas each animation and part receives, by CE's
own rules (a group's time and a blend's sub-alphas by the animation's own
calls, one instant at a time, so that each number of the record is CE's in
type as well as value; a reveal's sub-alphas, which the record does not
keep, through the rate function's array twin, :func:`.rates.array_twin`),
and the points and style a leaf takes by CE's own formulas. Every expression is CE's, in CE's operation order, one
element at a time (numpy fuses no multiply and add; the reveal's portion
matrices are applied by numpy's matrix product over a stack of curves,
which multiplies each curve as CE's call on it does), so each value is bit
for bit what CE's scalar call computes. At each instant the play is then
evaluated as before, through the scene's hooks, except that each of its
animations takes the values computed for that instant instead of being
interpolated: no animation's ``interpolate`` and no leaf's ``interpolate``
or ``pointwise_become_partial`` runs. A group is left holding the state its
``interpolate`` would have left at that instant, and what each value
replaces, shares or keeps is what CE's call replaces, shares or keeps.

The record keeps its animations' states at every kept instant as before
(:meth:`VectorPlan.snapshot`): a leaf's, which nothing changes while the
play runs, is copied once and shared. An instant between kept ones is
evaluated from those states by the animations themselves, as without the
lever, and the record's readers of them (``leaf_terms``) read them as
before.

A play stays scalar, whole, when any of its animations is outside those
kinds (:func:`refusal`): arc and circle paths, ``Rotate``, ``Write`` and
the other leaves that interpolate their own way, a mobject whose class
interpolates its own way (an image, a point cloud), and a ``Succession``,
whose parts begin while the play runs: a part begun then may run the
scene's code as it begins, so no value is computed before that. So does a
play that computes with a number of any other type than Python's and
numpy's double-precision and integer scalars (:data:`_EXACT`), or with a
paint array that is not float64: its run time, a rate function's or a rate
factory's numbers, a lag ratio, a flash's time width, a group's time, a
style number or color. A number of the scene's own class may run the
scene's code each time it is computed with, and numpy promotes a lower
precision in ways these arrays do not follow. So does an integer past
2**53, which a double does not hold exactly, or a reveal's integer lag
whose products with its leaves' indices pass int64; a group whose timings
or masks are not CE's arrays of its own parts (a scene may assign them);
and an animation or mobject holding its own attribute under a method the
vectorized path stands in for. So does a
play the route evaluates at one instant only, and any trusted play whose
evaluation may run the scene's code (``routes._runs_scene_code``).
"""

from __future__ import annotations

__all__ = [
    "VectorPlan",
    "blend",
    "install",
    "installing",
    "integer_interpolation",
    "partial_curves",
    "plan",
    "refusal",
]

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from operator import is_
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from arachne.animation.animation import Animation

#: The plan whose values the play being evaluated takes, while one is.
_ACTIVE: VectorPlan | None = None

#: The types of number the vectorized path computes with: Python's and
#: numpy's double-precision and integer scalars, whose arithmetic is their
#: own. Any other number (a subclass, which may define its own arithmetic, or
#: a lower precision, which numpy promotes in ways the arrays here do not
#: follow) keeps the whole play scalar.
_EXACT = frozenset({bool, int, float, np.float64, np.int64})

#: The largest integer the vectorized path computes with: a double holds
#: every integer up to it exactly, so numpy's conversions of it, and its
#: comparisons with doubles, are Python's.
_LARGEST_INTEGER = 2**53

#: The bound an integer the arrays compute in int64 stays under.
_INT64_BOUND = 2**63


def _exact(*values: Any) -> bool:
    for value in values:
        if type(value) not in _EXACT:
            return False
        integer = type(value) is not float and type(value) is not np.float64
        if integer and abs(int(value)) > _LARGEST_INTEGER:
            return False
    return True


def _number(value: Any) -> str:
    """A number the plan does not compute with, as a refusal names it."""
    if type(value) in _EXACT:
        return f"an integer past 2**53, {value}"
    return f"a {type(value).__name__}"


#: The methods the vectorized path stands in for, by the object whose class
#: it checks them on: an instance's own attribute of one of these names would
#: be what the scalar path calls.
_ANIMATION_METHODS = (
    "interpolate",
    "interpolate_mobject",
    "interpolate_submobject",
    "get_sub_alpha",
    "get_all_families_zipped",
    "_get_bounds",
)
_MOBJECT_METHODS = (
    "interpolate",
    "interpolate_color",
    "pointwise_become_partial",
    "set_points",
    "get_num_curves",
)


def _own_method(obj: Any, names: Sequence[str]) -> str | None:
    """The first of ``names`` that ``obj`` holds as its own attribute."""
    own = getattr(obj, "__dict__", {})
    return next((name for name in names if name in own), None)


# The arithmetic: CE's expressions over arrays of alphas.


def blend(start: Any, end: Any, alphas: np.ndarray) -> np.ndarray:
    """CE's ``interpolate(start, end, alpha)``, ``(1 - alpha) * start + alpha
    * end``, at each of ``alphas``: one row per alpha.
    """
    column = alphas.reshape(-1, *([1] * max(np.ndim(start), np.ndim(end))))
    return (1 - column) * np.asarray(start) + column * np.asarray(end)


def integer_interpolation(
    end: int, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CE's ``integer_interpolate(0, end, alpha)`` at each of ``alphas``: the
    indices and the residues.
    """
    index = np.zeros(len(alphas), dtype=np.int64)
    residue = np.zeros(len(alphas))
    top = alphas >= 1
    index[top] = int(end - 1)
    residue[top] = 1.0
    middle = ~top & ~(alphas <= 0)
    alpha = alphas[middle]
    index[middle] = ((1 - alpha) * 0 + alpha * end).astype(np.int64)
    residue[middle] = ((end - 0) * alpha) % 1
    return index, residue


def partial_curves(curves: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CE's ``partial_bezier_points(curve, a, b)`` for each cubic curve of
    ``curves`` (shape ``(n, 4, 3)``) with its own ``a`` and ``b``: the
    portion matrix built entry by entry as CE builds it, and applied to each
    curve by the same matrix product.
    """
    out = np.empty(curves.shape)
    at_end = a == 1
    at_start = ~at_end & (b == 0)
    out[at_end] = curves[at_end][:, -1:]
    out[at_start] = curves[at_start][:, :1]
    rest = ~(at_end | at_start)
    a, b = a[rest], b[rest]
    ma, mb = 1 - a, 1 - b
    a2, b2, ma2, mb2 = a * a, b * b, ma * ma, mb * mb
    a3, b3, ma3, mb3 = a2 * a, b2 * b, ma2 * ma, mb2 * mb
    entries = [
        *(ma3, 3 * ma2 * a, 3 * ma * a2, a3),
        *(ma2 * mb, 2 * ma * a * mb + ma2 * b, a2 * mb + 2 * ma * a * b, a2 * b),
        *(ma * mb2, a * mb2 + 2 * ma * mb * b, 2 * a * mb * b + ma * b2, a * b2),
        *(mb3, 3 * mb2 * b, 3 * mb * b2, b3),
    ]
    matrices = np.stack(entries, axis=-1).reshape(-1, 4, 4)
    out[rest] = matrices @ curves[rest]
    return out


def _create_bounds(animation: Animation, alphas: np.ndarray) -> tuple[Any, Any]:
    """``Create._get_bounds``: ``(0, alpha)``."""
    return np.zeros(len(alphas)), alphas


def _flash_bounds(animation: Animation, alphas: np.ndarray) -> tuple[Any, Any]:
    """``ShowPassingFlash._get_bounds``, ``min`` and ``max`` as Python's
    builtins choose between equal values.
    """
    width = animation.time_width  # type: ignore[attr-defined]
    upper = (1 - alphas) * 0 + alphas * (1 + width)
    lower = upper - width
    upper = np.where(upper > 1, 1.0, upper)
    lower = np.where(lower < 0, 0.0, lower)
    return lower, upper


# The plan: which animations a play is made of, as the vectorized path reads
# them.


class _Group:
    """A group whose interpolation is CE's ``AnimationGroup.interpolate``."""

    __slots__ = ("animation", "parts")

    def __init__(self, animation: Animation, parts: list[Any]) -> None:
        self.animation = animation
        self.parts = parts


class _Leaf:
    """A leaf animation: ``kind`` is ``"blend"``, ``"reveal"`` or ``None``
    (it writes nothing), ``families`` what it interpolates, ``flip`` whether
    it interpolates at ``1 - alpha`` (``TransformFromCopy``).
    """

    __slots__ = ("animation", "kind", "families", "flip", "twin", "bounds")

    def __init__(
        self,
        animation: Animation,
        kind: str | None,
        families: Sequence[tuple[Any, ...]] = (),
        flip: bool = False,
        twin: Any = None,
        bounds: Callable[..., tuple[Any, Any]] | None = None,
    ) -> None:
        self.animation = animation
        self.kind = kind
        self.families = families
        self.flip = flip
        self.twin = twin
        self.bounds = bounds


def refusal(animations: Sequence[Animation]) -> str | None:
    """Why a trusted play of ``animations`` stays scalar, or ``None`` when it
    is evaluated vectorized.
    """
    for animation in animations:
        node = _top(animation)
        if isinstance(node, str):
            return node
    return None


def _twin(function: Any) -> Any:
    from .rates import array_twin

    try:
        return array_twin(function)
    except (KeyError, TypeError):
        return None


def _rate_refusal(name: str, function: Any) -> str | None:
    """Why ``name``'s rate function is not one the plan may call ahead of
    the play: one with no array twin, or a factory's closure over a number
    that is not exact (:data:`_EXACT`), at any depth.
    """
    from .rates import _factory_parts

    if _twin(function) is None:
        return f"{name}'s rate function has no array twin"
    while (parts := _factory_parts(function)) is not None:
        kind, cells = parts
        numbers = ("a", "b") if kind == "squish" else ("proportion",)
        for number in numbers:
            if not _exact(cells[number]):
                held = _number(cells[number])
                return f"{name}'s rate function computes with {held}"
        function = cells["func"]
    return None


def _top(animation: Animation) -> _Group | _Leaf | str:
    """The plan of a play's animation: its node, its run time exact."""
    if not _exact(animation.run_time):
        held = _number(animation.run_time)
        return f"{type(animation).__name__}'s run time is {held}"
    return _node(animation)


def _leaf_refusal(name: str, animation: Animation) -> str | None:
    """Why the numbers ``get_sub_alpha`` computes a leaf's sub-alphas with
    are not ones the plan may compute with ahead of the play.
    """
    reason = _rate_refusal(name, animation.rate_func)
    if reason is not None:
        return reason
    if not _exact(animation.lag_ratio):
        return f"{name}'s lag ratio is {_number(animation.lag_ratio)}"
    if type(animation.reverse_rate_function) is not bool:
        kind_name = type(animation.reverse_rate_function).__name__
        return f"{name}'s reverse_rate_function is a {kind_name}"
    return None


def _node(animation: Animation) -> _Group | _Leaf | str:
    from arachne.animation.animation import Add, Animation, Wait
    from arachne.animation.composition import AnimationGroup, Succession
    from arachne.animation.creation import Create, ShowPartial
    from arachne.animation.indication import ShowPassingFlash
    from arachne.animation.transform import Transform, TransformFromCopy
    from arachne.utils.bezier import interpolate

    cls = type(animation)
    name = cls.__name__
    own = _own_method(animation, _ANIMATION_METHODS)
    if own is not None:
        return f"{name} has its own {own}"
    if isinstance(animation, Succession):
        return f"{name} begins its parts while the play runs"
    if isinstance(animation, AnimationGroup):
        if cls.interpolate is not AnimationGroup.interpolate:
            return f"{name} interpolates its parts its own way"
        reason = _rate_refusal(name, animation.rate_func)
        if reason is None:
            reason = _timing_refusal(name, animation)
        if reason is not None:
            return reason
        parts: list[Any] = []
        for part in animation.animations:
            node = _node(part)
            if isinstance(node, str):
                return node
            parts.append(node)
        return _Group(animation, parts)
    if cls.interpolate in (Wait.interpolate, Add.interpolate):
        return _Leaf(animation, None)
    flips = {Animation.interpolate: False, TransformFromCopy.interpolate: True}
    flip = flips.get(cls.interpolate)
    if (
        flip is None
        or cls.interpolate_mobject is not Animation.interpolate_mobject
        or cls.get_sub_alpha is not Animation.get_sub_alpha
    ):
        return f"{name} interpolates its own way"
    # A leaf that writes nothing still computes its sub-alphas on the scalar
    # path, so its numbers are held to the same types.
    reason = _leaf_refusal(name, animation)
    if reason is not None:
        return reason
    if cls.interpolate_submobject is Animation.interpolate_submobject:
        return _Leaf(animation, None)
    twin = _twin(animation.rate_func)
    if (
        isinstance(animation, Transform)
        and cls.interpolate_submobject is Transform.interpolate_submobject
        and cls.get_all_families_zipped is Transform.get_all_families_zipped
    ):
        if animation.path_func is not interpolate:
            return f"{name}'s path is not straight"
        families = list(animation.get_all_families_zipped())
        for family in families:
            reason = _blendable(*family)
            if reason is not None:
                return f"{name} blends {reason}"
        return _Leaf(animation, "blend", families, flip, twin)
    windows: dict[Any, Callable[..., tuple[Any, Any]]] = {
        Create._get_bounds: _create_bounds,
        ShowPassingFlash._get_bounds: _flash_bounds,
    }
    bounds = windows.get(getattr(cls, "_get_bounds", None))
    if (
        isinstance(animation, ShowPartial)
        and cls.interpolate_submobject is ShowPartial.interpolate_submobject
        and cls.get_all_families_zipped is Animation.get_all_families_zipped
        and bounds is not None
    ):
        width = getattr(animation, "time_width", None)
        if bounds is _flash_bounds and not _exact(width):
            return f"{name}'s time width is {_number(width)}"
        families = list(animation.get_all_families_zipped())
        # Each leaf's lower bound is its index times the lag, in int64.
        lag = animation.lag_ratio
        if (
            not isinstance(lag, float)
            and abs(int(lag)) * max(len(families) - 1, 0) >= _INT64_BOUND
        ):
            return f"{name}'s lag ratio times its leaves passes int64"
        for family in families:
            reason = _revealable(*family)
            if reason is not None:
                return f"{name} reveals {reason}"
        return _Leaf(animation, "reveal", families, flip, twin, bounds)
    return f"{name} is neither a straight-path blend nor a reveal"


def _timing_refusal(name: str, group: Any) -> str | None:
    """Why the timings and masks ``group`` holds are not CE's arrays of its
    own parts, which the plan computes its part alphas from.
    """
    count = len(group.animations)
    for held in (group.max_end_time, group.anim_group_time):
        if not _exact(held):
            return f"{name}'s time is {_number(held)}"
    timings = group.anims_with_timings
    fields = getattr(getattr(timings, "dtype", None), "fields", None) or {}
    if (
        type(timings) is not np.ndarray
        or timings.shape != (count,)
        or set(fields) != {"anim", "start", "end"}
        or fields["start"][0] != np.float64
        or fields["end"][0] != np.float64
        or any(
            a is not b for a, b in zip(timings["anim"], group.animations, strict=True)
        )
    ):
        return f"{name}'s timings are not CE's float64 timings of its parts"
    for mask in ("anims_begun", "anims_finished"):
        held = getattr(group, mask)
        if (
            type(held) is not np.ndarray
            or held.dtype != np.bool_
            or held.shape != (count,)
        ):
            return f"{name}'s {mask} is not a bool array of its parts"
    return None


def _engine_points(mob: Any) -> bool:
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.types.vectorized_mobject import VMobject

    if not isinstance(mob, VMobject) or type(mob).points is not Mobject.points:
        return False
    points = mob.__dict__.get("_points")
    return (
        isinstance(points, np.ndarray)
        and points.dtype == np.float64
        and points.ndim == 2
        and points.shape[1] == 3
    )


def _blendable(leaf: Any, start: Any, end: Any) -> str | None:
    """Why ``leaf.interpolate(start, end, alpha)`` is not CE's straight-path
    blend of points and paint, if it is not.
    """
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.types.vectorized_mobject import VMobject, _interpolable
    from arachne.mobject.values import Paint

    if not all(_engine_points(mob) for mob in (leaf, start, end)):
        return f"a {type(leaf).__name__} whose points are not a VMobject's"
    for mob in (leaf, start, end):
        own = _own_method(mob, _MOBJECT_METHODS)
        if own is not None:
            return f"a {type(mob).__name__} with its own {own}"
    cls = type(leaf)
    if (
        cls.interpolate is not Mobject.interpolate
        or cls.interpolate_color is not VMobject.interpolate_color
    ):
        return f"a {cls.__name__}, which interpolates its own way"
    if start.points.shape != end.points.shape:
        return "paths of different lengths"
    for mob in (start, end):
        if not all(getattr(type(mob), f) is getattr(VMobject, f) for f in Paint.FIELDS):
            return f"a {type(mob).__name__}, which reads its style its own way"
    if not isinstance(leaf.__dict__.get("_paint"), Paint) or not _interpolable(
        start._paint, end._paint
    ):
        return "styles CE's interpolation does not mix"
    for paint in (start._paint, end._paint):
        for name, kind in Paint.FIELD_KINDS.items():
            value = paint.__dict__[name]
            if kind is None:
                if not _exact(value):
                    return f"a style number that is {_number(value)}"
            elif np.ndim(value) == 0:
                return "styles CE's interpolation does not mix"
            elif value.dtype != np.float64:
                return f"styles held in {value.dtype}"
    return None


def _revealable(leaf: Any, start: Any, *rest: Any) -> str | None:
    """Why ``leaf.pointwise_become_partial(start, a, b)`` is not CE's reveal
    window of cubic curves, if it is not.
    """
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.types.vectorized_mobject import VMobject

    if rest or not (_engine_points(leaf) and _engine_points(start)):
        return f"a {type(leaf).__name__} whose points are not a VMobject's"
    for mob in (leaf, start):
        own = _own_method(mob, _MOBJECT_METHODS)
        if own is not None:
            return f"a {type(mob).__name__} with its own {own}"
    cls = type(leaf)
    if (
        cls.pointwise_become_partial is not VMobject.pointwise_become_partial
        or cls.set_points is not Mobject.set_points
        or cls.n_points_per_curve is not VMobject.n_points_per_curve
        or type(start).get_num_curves is not VMobject.get_num_curves
    ):
        return f"a {cls.__name__}, which takes its part its own way"
    counts = (leaf.n_points_per_cubic_curve, start.n_points_per_cubic_curve, leaf.dim)
    if any(type(count) is not int for count in counts) or counts != (4, 4, 3):
        return "curves that are not cubic"
    return None


def plan(
    animations: Sequence[Animation], instants: Sequence[Fraction], start: Fraction
) -> VectorPlan | None:
    """The values a trusted play of ``animations`` that began at ``start``
    takes at each of ``instants``, its kept instants in order; ``None`` when
    it stays scalar (:func:`refusal`).
    """
    nodes = []
    for animation in animations:
        node = _top(animation)
        if isinstance(node, str):
            return None
        nodes.append(node)
    return VectorPlan(animations, nodes, instants, start)


_GROUP, _BLEND, _POINTS, _RAW_POINTS = range(4)


class VectorPlan:
    """What each animation of a play writes at each of its kept instants,
    computed when the play's run begins, and the snapshots of its
    animations the record keeps at them (:func:`.routes.snapshot_animations`).
    """

    def __init__(
        self,
        animations: Sequence[Animation],
        nodes: Sequence[_Group | _Leaf],
        instants: Sequence[Fraction],
        start: Fraction,
    ) -> None:
        from .hooks import _alpha
        from .trusted import iter_animations

        self._index = {instant: k for k, instant in enumerate(instants)}
        #: Per top-level animation (by id), its writes at each instant.
        self._steps: dict[int, list[list[tuple[Any, ...]]]] = {}
        count = len(instants)
        positions = np.arange(count)
        for animation, node in zip(animations, nodes, strict=True):
            steps: list[list[tuple[Any, ...]]] = [[] for _ in range(count)]
            alphas = [
                _alpha(instant - start, animation.run_time) for instant in instants
            ]
            _evaluate(node, alphas, positions, steps)
            self._steps[id(animation)] = steps
        #: The last snapshot of each animation, with what its ``__dict__``
        #: held when it was taken.
        self._snapshots: list[tuple[Animation, tuple[Any, ...] | None, Any]] = [
            (animation, None, None) for animation in iter_animations(animations)
        ]

    def install(self, animation: Animation, instant: Fraction) -> bool:
        """Give ``animation``'s leaves (and its groups) what interpolating it
        at ``instant`` gives them; false for an animation of another play.
        """
        steps = self._steps.get(id(animation))
        if steps is None:
            return False
        k = self._index.get(instant)
        if k is None:
            raise RuntimeError(
                f"The play was evaluated at {instant}, an instant its plan does"
                " not hold."
            )
        for step in steps[k]:
            code, target = step[0], step[1]
            if code is _BLEND:
                target._points = step[2]
                target.__dict__.pop("_placement", None)
                target._paint = target._paint.replace(**step[3])
            elif code is _POINTS:
                target._points = step[2]
                target.__dict__.pop("_placement", None)
            elif code is _RAW_POINTS:
                target._points = step[2]
            else:
                target.anim_group_time = step[2]
                target.anims_begun = step[3]
                target.anims_finished = step[4]
        return True

    def snapshot(self) -> tuple[tuple[Animation, dict[str, Any]], ...]:
        """:func:`.routes.snapshot_animations` of the play's animations: an
        animation whose ``__dict__`` holds the same objects as at the last
        snapshot shares that snapshot's state. A leaf's never changes while
        its play runs here, since nothing interpolates it; a group's does at
        every instant it is evaluated at.
        """
        from .routes import animation_state

        taken = []
        for index, (animation, held, state) in enumerate(self._snapshots):
            own = vars(animation)
            if held is None or not _holds(held, own):
                state = animation_state(animation)
                self._snapshots[index] = (animation, (*own, *own.values()), state)
            taken.append((animation, state))
        return tuple(taken)


def _holds(held: tuple[Any, ...], own: dict[str, Any]) -> bool:
    """Whether ``own`` holds exactly the keys and values ``held`` lists."""
    if len(held) != 2 * len(own):
        return False
    return all(map(is_, held, (*own, *own.values())))


def _evaluate(
    node: _Group | _Leaf,
    alphas: Sequence[Any],
    positions: np.ndarray,
    steps: list[list[tuple[Any, ...]]],
) -> None:
    """Append to ``steps[position]`` what interpolating ``node``'s animation
    at each of ``alphas`` writes, the alpha it receives at the kept instant
    ``position`` (both in order), each the very number CE's call receives:
    its type as well as its value.
    """
    if isinstance(node, _Group):
        _evaluate_group(node, alphas, positions, steps)
        return
    if node.kind is None or not node.families:
        return
    animation = node.animation
    if node.flip:
        alphas = [1 - alpha for alpha in alphas]
    count = len(node.families)
    if node.kind == "blend":
        # A blend's style numbers take the type of the sub-alpha that mixes
        # them (a rate function returns a float, a numpy float, or the int
        # its unit interval clamps to), so each is CE's own, at each instant.
        sub_alpha = animation.get_sub_alpha
        for index, family in enumerate(node.families):
            subs = [sub_alpha(alpha, index, count) for alpha in alphas]
            _blends(family, subs, positions, steps)
        return
    lag = animation.lag_ratio
    full_length = (count - 1) * lag + 1
    values = np.array(alphas, dtype=float) * full_length
    raw = values[:, None] - np.arange(count) * lag
    if animation.reverse_rate_function:
        raw = 1 - raw
    twins = np.asarray(node.twin(raw.ravel()), dtype=float).reshape(raw.shape)
    for index, family in enumerate(node.families):
        lower, upper = node.bounds(animation, twins[:, index])  # type: ignore[misc]
        _reveals(family, lower, upper, positions, steps)


def _evaluate_group(
    node: _Group,
    alphas: Sequence[Any],
    positions: np.ndarray,
    steps: list[list[tuple[Any, ...]]],
) -> None:
    """CE's ``AnimationGroup.interpolate`` at each alpha in turn: the group
    time (by its own rate function, once per instant: the time a group is
    left holding is a number of the record), which parts it interpolates, at
    what alpha each, and the state it is left holding, from the state it
    holds now.
    """
    group = node.animation
    rate, length = group.rate_func, group.max_end_time  # type: ignore[attr-defined]
    held = [rate(alpha) * length for alpha in alphas]
    times = np.array(held, dtype=float)
    timings = group.anims_with_timings  # type: ignore[attr-defined]
    starts, ends = timings["start"], timings["end"]
    before = np.empty(len(times))
    before[0] = group.anim_group_time  # type: ignore[attr-defined]
    before[1:] = times[:-1]
    back = times < before
    begun = times[:, None] >= starts
    finished = times[:, None] > ends
    begun_before = np.vstack([group.anims_begun[None], begun[:-1]])  # type: ignore[attr-defined]
    finished_before = np.vstack([group.anims_finished[None], finished[:-1]])  # type: ignore[attr-defined]
    updated = (begun_before | begun) & (~finished_before | ~finished)
    run_times = ends - starts
    zero = run_times == 0
    run_times[zero] = 1
    subs = (times[:, None] - starts) / run_times
    subs[~back[:, None] & ((subs > 1) | zero)] = 1
    subs[back[:, None] & ((subs < 0) | zero)] = 0
    for j, part in enumerate(node.parts):
        mask = updated[:, j]
        if mask.any():
            # CE hands each part an element of its numpy array of sub-alphas.
            _evaluate(part, list(subs[mask, j]), positions[mask], steps)
    for m, position in enumerate(positions):
        steps[position].append(
            (_GROUP, group, held[m], begun[m].copy(), finished[m].copy())
        )


def _blends(
    family: tuple[Any, ...],
    alphas: Sequence[Any],
    positions: np.ndarray,
    steps: list[list[tuple[Any, ...]]],
) -> None:
    """``leaf.interpolate(start, end, sub)`` on CE's straight path at each
    sub-alpha of ``alphas``: the points and style arrays for all at once,
    then each style number by CE's expression at its own sub-alpha
    (``end``'s own value at alpha 1).
    """
    from arachne.mobject.values import FrozenPoints, Paint, adopt

    leaf, start, end = family
    subs = np.array(alphas, dtype=float)
    points = adopt(blend(start.points, end.points, subs), FrozenPoints)
    first, second = start._paint.__dict__, end._paint.__dict__
    columns: dict[str, Any] = {}
    for name, kind in Paint.FIELD_KINDS.items():
        if kind is not None:
            columns[name] = adopt(blend(first[name], second[name], subs), kind)
    for m, position in enumerate(positions):
        alpha = alphas[m]
        fields = {}
        for name in Paint.FIELDS:
            if alpha == 1.0:
                fields[name] = second[name]
            elif name in columns:
                fields[name] = columns[name][m]
            else:
                fields[name] = (1 - alpha) * first[name] + alpha * second[name]
        steps[position].append((_BLEND, leaf, points[m], fields))


def _reveals(
    family: tuple[Any, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    positions: np.ndarray,
    steps: list[list[tuple[Any, ...]]],
) -> None:
    """``leaf.pointwise_become_partial(start, lower, upper)`` at each window,
    as CE's ``VMobject`` takes it: the whole path shared, nothing for a path
    of no curves, one curve's part, or a first and last part around whole
    curves.
    """
    from arachne.mobject.values import FrozenPoints, adopt

    leaf, start = family
    path = start.points
    whole = (lower <= 0) & (upper >= 1)
    count = start.get_num_curves()
    if count == 0:
        for position in positions[whole]:
            steps[position].append((_POINTS, leaf, path))
        return
    curves = np.asarray(path)[: 4 * count].reshape(count, 4, 3)
    low, low_residue = integer_interpolation(count, lower)
    high, high_residue = integer_interpolation(count, upper)
    one = ~whole & (low == high)
    several = ~whole & ~one
    single = adopt(
        partial_curves(curves[low[one]], low_residue[one], high_residue[one]),
        FrozenPoints,
    )
    firsts = partial_curves(
        curves[low[several]], low_residue[several], np.ones(several.sum())
    )
    lasts = partial_curves(
        curves[high[several]], np.zeros(several.sum()), high_residue[several]
    )
    taken = {"one": 0, "several": 0}
    for m, position in enumerate(positions):
        if whole[m]:
            steps[position].append((_POINTS, leaf, path))
        elif one[m]:
            steps[position].append((_POINTS, leaf, single[taken["one"]]))
            taken["one"] += 1
        else:
            n = taken["several"]
            taken["several"] += 1
            first, last = int(low[m]), int(high[m])
            points = np.empty((4 * (last - first + 1), 3))
            points[:4] = firsts[n]
            points[4:-4] = path[4 * (first + 1) : 4 * last]
            points[-4:] = lasts[n]
            steps[position].append((_RAW_POINTS, leaf, adopt(points, FrozenPoints)))


@contextmanager
def installing(vector: VectorPlan) -> Iterator[None]:
    """Have the play's evaluation take ``vector``'s values inside the block."""
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, vector
    try:
        yield
    finally:
        _ACTIVE = previous


def install(animation: Animation, instant: Fraction) -> bool:
    """Give ``animation`` its values at ``instant`` from the plan of the play
    being evaluated, if it has one (:meth:`VectorPlan.install`).
    """
    vector = _ACTIVE
    return vector is not None and vector.install(animation, instant)
