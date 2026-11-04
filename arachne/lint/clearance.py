"""How far one mobject stands clear of others, each measured whole, as the
bridge's :func:`~arachne.lint.bridge.apart` asks it.

The sweep (:mod:`.rules`) takes the frame apart into the units it draws.
A pairwise check takes its arguments as the scene hands them over instead:
each argument is one side, whatever it holds, read from the live scene in
scene units, drawn or not.

**A side.** A side is a *region* when it is a text (a text mobject, or a
group of nothing but texts) or when every leaf it holds is filled; it is
then held to its box (a text, against anything but a text, to each glyph's
box and the glyph's outline within :data:`INK_NEAR` of it; a filled shape
of several leaves to each leaf's box). Anything else is a *path*, held to
points along its strokes. An axes or number line handed over as an
argument stands for its furniture, each piece a side: the line and the
ticks (strokes content rides, held clear of a text subject only), the tip
and each number.

**What a pair is held to.** ``min_gap`` when given; else two texts a line
of the shorter, never less than :data:`TEXT_FLOOR`; a closed ring around
some but not all of a text's glyphs only touching (:data:`TOUCH`); else
``pad``. A pair closer than that is a finding: ``contact.held`` when no
text is in it, otherwise ``gap.text`` when the two are apart and the
clearance asked is at least :data:`TEXT_FLOOR`, and ``overlap.text`` when
it is less or the two touch. Nothing else excuses a pair, a mobject held
against itself included.
"""

from __future__ import annotations

__all__ = ["INK_NEAR", "TEXT_FLOOR", "TOUCH", "Side", "clear_of", "sides"]

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np

from arachne.timeline.origins import origin_of

from .findings import Finding, FindingObject
from .parts import Box
from .rules._geometry import (
    box_gap,
    closest,
    least_box_gap,
    outline_edges,
    signed_to_box,
    signed_to_outline,
)

#: The least clearance two texts are held to, in scene units.
TEXT_FLOOR = 0.3
#: How near a glyph's box a stroke point comes before its outline decides.
INK_NEAR = 0.3
#: Clearance under this is touching.
TOUCH = 0.005
#: Scene units of line per unit of stroke width.
STROKE_UNIT = 0.01
#: Scene units between samples along a stroke; samples per curve at least
#: and at most.
SPACING = 0.02
_SAMPLES = (8, 512)
#: A stroke whose ends lie this near each other is closed.
_CLOSED = 1e-4


class _Ink(NamedTuple):
    """A glyph's points as :func:`.rules._geometry.outline_edges` reads them."""

    points: np.ndarray
    curve_size: int


@dataclass
class Side:
    """One side of a pair: what it is, its box (``None`` for a path with no
    points), and what it is measured by.
    """

    mob: Any
    label: str
    region: bool
    text: bool = False
    box: Box | None = None
    height: float = 0.0
    glyph_boxes: list[Box] = field(default_factory=list)
    glyph_ink: list[_Ink] = field(default_factory=list)
    glyph_reach: list[float] = field(default_factory=list)
    leaf_boxes: list[Box] | None = None
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    ends: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    #: An axis's line or ticks: what content rides by construction.
    structural: bool = False

    def boxes(self) -> list[Box]:
        """What a region is held to against anything but a text."""
        if self.text:
            return self.glyph_boxes or [self._box()]
        return self.leaf_boxes or [self._box()]

    def _box(self) -> Box:
        assert self.box is not None
        return self.box


def clear_of(
    subject: Any,
    against: Any,
    *,
    min_gap: float | None,
    pad: float,
    names: dict[int, str],
) -> list[Finding]:
    """The findings of ``subject`` not clear of each side of ``against``
    (:func:`arguments`), in their order, timed at zero: the caller gives
    them their instant.
    """
    (mob, label), *items = arguments(subject, against)
    first = _side(mob, label)
    found: list[Finding] = []
    for item in items:
        for other in sides(*item, against_text=first.text):
            finding = _pair(first, other, min_gap, pad, names)
            if finding is not None:
                found.append(finding)
    return found


def arguments(subject: Any, against: Any) -> list[tuple[Any, str | None]]:
    """The subject, then each argument of ``against`` (a mobject, a
    ``(label, mobject)`` pair, or a list of them), as mobject and label;
    raises :class:`TypeError` for anything else.
    """
    from arachne.mobject.mobject import Mobject

    # One argument, or a list of them; a tuple is one (a labelled mobject).
    items = [against] if isinstance(against, (Mobject, tuple)) else list(against)
    out = [_labelled(item) for item in (subject, *items)]
    for mob, _ in out:
        if not isinstance(mob, Mobject):
            raise TypeError(f"apart checks mobjects, not {type(mob).__name__}")
    return out


def _labelled(item: Any) -> tuple[Any, str | None]:
    """``item``'s mobject and the label it was handed with, if any."""
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[1], item[0]
    return item, None


def sides(
    mob: Any, label: str | None = None, *, against_text: bool = True
) -> list[Side]:
    """The sides one argument stands for: itself, or an axes' or number
    line's furniture (its line and ticks only against a text).
    """
    from arachne.mobject.graphing.coordinate_systems import Axes
    from arachne.mobject.graphing.number_line import NumberLine

    if not isinstance(mob, (Axes, NumberLine)):
        return [_side(mob, label)]
    furniture = _furniture(mob)
    if not against_text:
        furniture = [side for side in furniture if not side.structural]
    return furniture


def _furniture(axes: Any) -> list[Side]:
    from arachne.mobject.graphing.coordinate_systems import Axes

    lines = (
        [("x", axes.x_axis), ("y", axes.y_axis)]
        if isinstance(axes, Axes)
        else [("", axes)]
    )
    out: list[Side] = []
    for tag, axis in lines:
        stem = f"{tag}-axis" if tag else "axis"
        out.append(_stroke(axis, f"{stem} line", _samples([axis])))
        tip = getattr(axis, "tip", None)
        if tip is not None and tip.family_members_with_points():
            out.append(_side(tip, f"{stem} tip"))
        ticks = getattr(axis, "ticks", None)
        if ticks is not None and len(ticks):
            out.append(
                _stroke(
                    ticks, f"{stem} ticks", _samples(ticks.family_members_with_points())
                )
            )
        numbers = getattr(axis, "numbers", None)
        if numbers is not None and len(numbers):
            out.extend(_side(number, None) for number in numbers)
    return out


def _stroke(mob: Any, label: str, points: np.ndarray) -> Side:
    """A structural stroke: points only, no box but theirs."""
    side = Side(mob, label, region=False, points=points, structural=True)
    if len(points):
        side.box = (*points.min(axis=0), *points.max(axis=0))
        side.ends = points[[0, -1]]
    return side


# A side


def _side(mob: Any, label: str | None) -> Side:
    leaves = [leaf for leaf in mob.family_members_with_points() if len(leaf.points)]
    low = mob.get_corner(np.array([-1.0, -1.0, 0.0]))[:2]
    high = mob.get_corner(np.array([1.0, 1.0, 0.0]))[:2]
    box: Box = (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
    called = label or _called(mob)
    text = _is_text(mob)
    if text or (leaves and all(_filled(leaf) for leaf in leaves)):
        own = [_leaf_box(leaf) for leaf in leaves]
        side = Side(
            mob, called, region=True, text=text, box=box, height=float(mob.height)
        )
        if text:
            side.glyph_boxes = own
            side.glyph_ink = [
                _Ink(np.asarray(leaf.points, dtype=float)[:, :2], _per_curve(leaf))
                for leaf in leaves
            ]
            side.glyph_reach = [_reach(leaf) for leaf in leaves]
        elif len(own) > 1:
            side.leaf_boxes = own
        return side
    points = _samples(leaves)
    side = Side(mob, called, region=False, points=points)
    if len(points):
        side.box = box
    side.ends = (
        np.vstack(
            [np.asarray(leaf.points, dtype=float)[[0, -1], :2] for leaf in leaves]
        )
        if leaves
        else np.empty((0, 2))
    )
    return side


def _is_text(mob: Any) -> bool:
    """A text, or a group of nothing but texts."""
    from arachne.mobject.text.numbers import DecimalNumber
    from arachne.mobject.text.tex_mobject import MathTex, Tex
    from arachne.mobject.text.text_mobject import MarkupText, Text
    from arachne.mobject.types.vectorized_mobject import VGroup

    if isinstance(mob, (Text, MarkupText, Tex, MathTex, DecimalNumber)):
        return True
    if isinstance(mob, VGroup) and mob.submobjects:
        return all(_is_text(sub) for sub in mob.submobjects)
    return False


def _filled(leaf: Any) -> bool:
    opacity = getattr(leaf, "get_fill_opacity", None)
    return opacity is not None and float(opacity()) > 0


def _per_curve(leaf: Any) -> int:
    return int(getattr(leaf, "n_points_per_curve", 4))


def _leaf_box(leaf: Any) -> Box:
    points = np.asarray(leaf.points, dtype=float)[:, :2]
    low, high = points.min(axis=0), points.max(axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _reach(leaf: Any) -> float:
    """How far past its points a leaf's own stroke lays ink."""
    from arachne.mobject.types.vectorized_mobject import VMobject

    if not isinstance(leaf, VMobject):
        return 0.0
    shown = float(leaf.get_stroke_opacity()) > 0
    return 0.5 * STROKE_UNIT * float(leaf.get_stroke_width()) if shown else 0.0


def _samples(leaves: Sequence[Any]) -> np.ndarray:
    """Points along the leaves' strokes about :data:`SPACING` apart (by
    each leaf's longest curve), or the leaves' points when they are no
    cubic curves.
    """
    out: list[np.ndarray] = []
    for leaf in leaves:
        points = np.asarray(leaf.points, dtype=float)
        if not len(points):
            continue
        per = _per_curve(leaf)
        whole = len(points) - len(points) % per
        if per != 4 or whole < 4:
            out.append(points[:, :2])
            continue
        curves = points[:whole].reshape(-1, 4, points.shape[1])
        polygon = np.linalg.norm(np.diff(curves, axis=1), axis=2).sum(axis=1)
        low, high = _SAMPLES
        n = int(np.clip(np.ceil(polygon.max() / SPACING), low, high))
        s = np.linspace(0.0, 1.0, n)[:, None]
        weights = np.hstack(
            [(1 - s) ** 3, 3 * s * (1 - s) ** 2, 3 * s**2 * (1 - s), s**3]
        )
        out.append(
            np.einsum("sk,ckd->csd", weights, curves).reshape(-1, points.shape[1])[
                :, :2
            ]
        )
    return np.vstack(out) if out else np.empty((0, 2))


def _called(mob: Any) -> str:
    said = getattr(mob, "text", None) or getattr(mob, "tex_string", None)
    kind = type(mob).__name__
    if isinstance(said, str) and said.strip():
        return f"{kind} {' '.join(said.split())!r}"
    return kind


# A pair


def _pair(
    a: Side, b: Side, min_gap: float | None, pad: float, names: dict[int, str]
) -> Finding | None:
    required = _required(a, b, min_gap, pad)
    measured = _measure(a, b)
    if measured >= required:
        return None
    if not (a.text or b.text):
        code = "contact.held"
    elif measured > 0 and required >= TEXT_FLOOR - 1e-9:
        code = "gap.text"
    else:
        code = "overlap.text"
    detail = (
        {"overlap": round(-measured, 4)}
        if code == "overlap.text"
        else {"gap": round(measured, 4), "required": round(required, 4)}
    )
    return Finding(
        code,
        Fraction(0),
        Fraction(0),
        (_object(a, names), _object(b, names)),
        message=(
            f"{a.label} is not clear of {b.label}: {measured:.3f} units apart,"
            f" against {required:.3f}."
        ),
        detail=detail,
    )


def _required(a: Side, b: Side, min_gap: float | None, pad: float) -> float:
    if min_gap is not None:
        return float(min_gap)
    if a.text and b.text:
        return max(TEXT_FLOOR, min(a.height, b.height))
    if _rings_some_of(a, b) or _rings_some_of(b, a):
        return TOUCH
    return float(pad)


def _rings_some_of(ring: Side, text: Side) -> bool:
    """A closed stroke around some but not all of a text's glyphs."""
    if ring.region or not text.text or not text.glyph_boxes or ring.box is None:
        return False
    ends = ring.ends.reshape(-1, 2, 2)
    if not len(ends) or not np.all(
        np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1) < _CLOSED
    ):
        return False
    x0, y0, x1, y1 = ring.box
    inside = sum(
        x0 < g[0] and y0 < g[1] and g[2] < x1 and g[3] < y1 for g in text.glyph_boxes
    )
    return 0 < inside < len(text.glyph_boxes)


def _measure(a: Side, b: Side) -> float:
    """How far apart ``a`` and ``b`` are; at or below zero, touching."""
    if a.region and b.region:
        if a.text and b.text:
            return box_gap(a._box(), b._box())
        return least_box_gap(a.boxes(), b.boxes())
    for region, path in ((a, b), (b, a)):
        if region.region:
            if not len(path.points):
                return math.inf
            if region.text:
                return _from_glyphs(path.points, region)
            return min(
                float(signed_to_box(path.points, g).min()) for g in region.boxes()
            )
    return closest(a.points, b.points)


def _from_glyphs(points: np.ndarray, text: Side) -> float:
    """How far a stroke's points are from a text's glyphs: from each
    glyph's box less its own stroke, and within :data:`INK_NEAR` of it from
    its outline, nearest box first.
    """
    reach = text.glyph_reach
    to_box = [
        signed_to_box(points, box) - r
        for box, r in zip(text.glyph_boxes, reach, strict=True)
    ]
    best = math.inf
    for index in sorted(range(len(to_box)), key=lambda i: float(to_box[i].min())):
        here = to_box[index]
        if float(here.min()) >= best:
            break
        edges = outline_edges(text.glyph_ink[index])  # type: ignore[arg-type]
        near = here < INK_NEAR
        if edges is not None and near.any():
            here = here.copy()
            here[near] = signed_to_outline(points[near], edges) - reach[index]
        best = min(best, float(here.min()))
    return best


def _object(side: Side, names: dict[int, str]) -> FindingObject:
    origin = origin_of(side.mob)
    said = getattr(side.mob, "text", None)
    name = names.get(id(side.mob))
    return FindingObject(
        id=origin.id,
        source_line=origin.scene_line,
        kind=type(side.mob).__name__,
        label_or_text=said if isinstance(said, str) else (name or side.label),
        bbox=side.box,
        screen_bbox=None,
        name=name,
    )
