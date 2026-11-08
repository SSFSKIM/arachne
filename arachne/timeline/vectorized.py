"""Trusted plays evaluated at all their kept instants at once
(``vectorize_trusted``).

A trusted play (route 1) is a known function of the instant
(:mod:`.trusted`). When every leaf of it is a straight-path blend or a
partial reveal (the two terms :meth:`.PlayRecord.leaf_terms` describes), or
writes nothing, and its leaves are composed by CE's group timing, the route
computes every leaf's value at every instant it will keep before it
evaluates the first: each animation's alphas through its rate function's
array twin (:func:`.rates.array_twin`), the groups' part alphas and the
leaves' sub-alphas by CE's own rules, and the points and style a leaf takes
by CE's own formulas. Every expression is CE's, in CE's operation order, one
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

    __slots__ = ("animation", "parts", "twin")

    def __init__(self, animation: Animation, parts: list[Any], twin: Any) -> None:
        self.animation = animation
        self.parts = parts
        self.twin = twin


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
        node = _node(animation)
        if isinstance(node, str):
            return node
    return None


def _twin(function: Any) -> Any:
    from .rates import array_twin

    try:
        return array_twin(function)
    except (KeyError, TypeError):
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
    if isinstance(animation, Succession):
        return f"{name} begins its parts while the play runs"
    if isinstance(animation, AnimationGroup):
        if cls.interpolate is not AnimationGroup.interpolate:
            return f"{name} interpolates its parts its own way"
        twin = _twin(animation.rate_func)
        if twin is None:
            return f"{name}'s rate function has no array twin"
        parts: list[Any] = []
        for part in animation.animations:
            node = _node(part)
            if isinstance(node, str):
                return node
            parts.append(node)
        return _Group(animation, parts, twin)
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
    if cls.interpolate_submobject is Animation.interpolate_submobject:
        return _Leaf(animation, None)
    twin = _twin(animation.rate_func)
    if twin is None:
        return f"{name}'s rate function has no array twin"
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
        families = list(animation.get_all_families_zipped())
        for family in families:
            reason = _revealable(*family)
            if reason is not None:
                return f"{name} reveals {reason}"
        return _Leaf(animation, "reveal", families, flip, twin, bounds)
    return f"{name} is neither a straight-path blend nor a reveal"


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
    for name, kind in Paint.FIELD_KINDS.items():
        if kind is not None and 0 in (
            np.ndim(start._paint.__dict__[name]),
            np.ndim(end._paint.__dict__[name]),
        ):
            return "styles CE's interpolation does not mix"
    return None


def _revealable(leaf: Any, start: Any, *rest: Any) -> str | None:
    """Why ``leaf.pointwise_become_partial(start, a, b)`` is not CE's reveal
    window of cubic curves, if it is not.
    """
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.types.vectorized_mobject import VMobject

    if rest or not (_engine_points(leaf) and _engine_points(start)):
        return f"a {type(leaf).__name__} whose points are not a VMobject's"
    cls = type(leaf)
    if (
        cls.pointwise_become_partial is not VMobject.pointwise_become_partial
        or cls.set_points is not Mobject.set_points
        or cls.n_points_per_curve is not VMobject.n_points_per_curve
        or type(start).get_num_curves is not VMobject.get_num_curves
    ):
        return f"a {cls.__name__}, which takes its part its own way"
    if (
        leaf.n_points_per_cubic_curve != 4
        or start.n_points_per_cubic_curve != 4
        or leaf.dim != 3
    ):
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
        node = _node(animation)
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
            alphas = np.array(
                [_alpha(instant - start, animation.run_time) for instant in instants]
            )
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
    alphas: np.ndarray,
    positions: np.ndarray,
    steps: list[list[tuple[Any, ...]]],
) -> None:
    """Append to ``steps[position]`` what interpolating ``node``'s animation
    at each alpha of ``alphas`` writes, the alpha it receives at the kept
    instant ``position`` (both in order).
    """
    if isinstance(node, _Group):
        _evaluate_group(node, alphas, positions, steps)
        return
    if node.kind is None or not node.families:
        return
    animation = node.animation
    if node.flip:
        alphas = 1 - alphas
    count = len(node.families)
    lag = animation.lag_ratio
    full_length = (count - 1) * lag + 1
    raw = (alphas * full_length)[:, None] - np.arange(count) * lag
    if animation.reverse_rate_function:
        raw = 1 - raw
    subs = np.asarray(node.twin(raw.ravel()), dtype=float).reshape(raw.shape)
    for index, family in enumerate(node.families):
        if node.kind == "blend":
            _blends(family, subs[:, index], positions, steps)
        else:
            lower, upper = node.bounds(animation, subs[:, index])  # type: ignore[misc]
            _reveals(family, lower, upper, positions, steps)


def _evaluate_group(
    node: _Group,
    alphas: np.ndarray,
    positions: np.ndarray,
    steps: list[list[tuple[Any, ...]]],
) -> None:
    """CE's ``AnimationGroup.interpolate`` at each alpha in turn: the group
    time, which parts it interpolates, at what alpha each, and the state it
    is left holding, from the state it holds now.
    """
    group = node.animation
    times = node.twin(alphas) * group.max_end_time  # type: ignore[attr-defined]
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
            _evaluate(part, subs[mask, j], positions[mask], steps)
    for m, position in enumerate(positions):
        steps[position].append(
            (_GROUP, group, times[m], begun[m].copy(), finished[m].copy())
        )


def _blends(
    family: tuple[Any, ...],
    subs: np.ndarray,
    positions: np.ndarray,
    steps: list[list[tuple[Any, ...]]],
) -> None:
    """``leaf.interpolate(start, end, sub)`` on CE's straight path at each
    sub-alpha: the points, then each style field as ``interpolate_color``
    mixes it (``end``'s own value at alpha 1).
    """
    from arachne.mobject.values import FrozenPoints, Paint, adopt

    leaf, start, end = family
    points = adopt(blend(start.points, end.points, subs), FrozenPoints)
    first, second = start._paint.__dict__, end._paint.__dict__
    columns: dict[str, Any] = {}
    for name, kind in Paint.FIELD_KINDS.items():
        if kind is not None:
            columns[name] = adopt(blend(first[name], second[name], subs), kind)
    for m, position in enumerate(positions):
        alpha = subs[m]
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
