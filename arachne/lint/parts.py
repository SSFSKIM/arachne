"""What the lint's rules read at an instant: the scene's drawn parts.

A capture's mobjects are read as a frame draws them, in its paint order:
the play's static mobjects, then its moving ones, or the whole scene when
nothing moves, each family flattened into the members that have points.
Each drawn member is a :class:`Leaf`: its points where the camera puts
them, in screen units of the home frame (the configuration's frame,
centred, y up, so equal to scene units under a still camera), its box,
its box with half its stroke as drawn around it, its box in scene units,
its fill and stroke, and its place in the paint order. A leaf that draws
nothing (fully transparent, as alignment padding is) is left out.

A :class:`Part` is what a rule measures: a text (a ``Text``, ``MathTex``,
number and the like, whole) as the ink of its glyph leaves, and every other
leaf on its own. Each part has its stable id, creating line and name, the
groups holding it, and whether it is axis furniture (an axis line, its tip,
ticks and numbers) or part of a grid (a number plane or vector field). Two
copies of one text with the same ink in one place are one text, counting
its ``copies`` (a transform's copy left on its original). Reading texts as
whole and equal ink as one follows ManimGX's diagnostics, credited in
``NOTICE.md``.
"""

from __future__ import annotations

__all__ = ["Leaf", "Part", "Parts", "build_parts"]

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.timeline.draw import Drawing, camera_nodes

from ._screen import (
    Box,
    View,
    draw_kind,
    pixel_points,
    record_view,
    scene_line,
    view_of,
)
from .findings import FindingObject, Holder

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, PlayRecord, Record
    from arachne.timeline.source import SourceLine

#: Points closer than this (scene units) are one: where a subpath closes.
_SAME_POINT = 1e-6
#: Decimals ink is compared to, in screen units, for two copies of a text.
_INK_DECIMALS = 3


@dataclass(frozen=True, eq=False)
class Leaf:
    """One drawn member of a family, as the camera draws it.

    ``points`` are its points on the screen, ``(n, 2)`` in home-frame
    units: a path's Bézier control points (``curve_size`` per curve, as the
    drawing strokes them), a point cloud's points, an image's corners.
    ``stroke_width`` is its widest visible stroke (its background stroke
    included) as drawn, in home-frame units, 0 when it has none.
    """

    id: str
    type: str
    draw: str
    points: np.ndarray
    screen_box: Box
    ink_box: Box
    scene_box: Box
    fill_opacity: float
    stroke_opacity: float
    stroke_width: float
    closed: bool
    order: int
    curve_size: int = 4

    def outline(self, samples: int = 17) -> np.ndarray | None:
        """A path's curves as polylines, ``(curves, samples, 2)``; ``None``
        for a leaf that is no path or has no whole curve.
        """
        n = self.curve_size
        if self.draw != "path" or len(self.points) < n:
            return None
        curves = self.points[: len(self.points) // n * n].reshape(-1, n, 2)
        t = np.linspace(0.0, 1.0, samples)
        degree = n - 1
        from math import comb

        basis = np.stack(
            [comb(degree, k) * (1 - t) ** (degree - k) * t**k for k in range(n)],
            axis=1,
        )
        return np.einsum("tk,ckd->ctd", basis, curves)


@dataclass(frozen=True, eq=False)
class Part:
    """A thing a rule measures: a whole text (``kind == "text"``) or one
    drawn leaf (``"path"``, ``"points"`` or ``"image"``).

    ``order`` is where its first leaf is drawn, ``top`` where its last is.
    ``root`` is the top-level mobject holding it (itself when it is one),
    ``within`` the groups holding it, outermost first.
    """

    id: str
    type: str
    kind: str
    name: str | None
    source_line: SourceLine | None
    scene_line: int | None
    text: str | None
    leaves: tuple[Leaf, ...]
    screen_box: Box
    ink_box: Box
    scene_box: Box
    order: int
    top: int
    root: str
    within: tuple[Holder, ...]
    furniture: bool
    grid: bool
    copies: int = 1

    def as_object(self, parts: Parts) -> FindingObject:
        """The part as a finding names it, among ``parts``: its box in scene
        units and in pixels of the target resolution.
        """
        return FindingObject(
            id=self.id,
            source_line=self.scene_line,
            kind=self.type,
            label_or_text=self.text if self.text is not None else self.name,
            bbox=self.scene_box,
            screen_bbox=target_pixels(parts, self.screen_box),
            name=self.name,
            within=self.within,
        )


@dataclass(frozen=True, eq=False)
class Parts:
    """The parts drawn at ``t``, in paint order: ``play`` holds ``t``
    (``None`` after the last play), and ``answer`` says what answered it
    (``"recorded"``, the record; ``"forward"``, a forward run). ``frame``
    is the picture's box in home-frame units; ``target`` the resolution
    pixel boxes are given at, ``(width, height)``: the session's.
    """

    t: Fraction
    play: int | None
    answer: str
    view: View
    parts: tuple[Part, ...]
    target: tuple[int, int]

    @property
    def frame(self) -> Box:
        w, h = self.view.home_width / 2, self.view.home_height / 2
        return (-w, -h, w, h)

    @cached_property
    def texts(self) -> tuple[Part, ...]:
        return tuple(part for part in self.parts if part.kind == "text")

    def get(self, part_id: str) -> Part:
        """The part with stable id ``part_id``."""
        for part in self.parts:
            if part.id == part_id:
                return part
        raise KeyError(part_id)


def target_pixels(parts: Parts, box: Box) -> Box:
    """A box in home-frame units, in pixels of ``parts``' target resolution,
    from the top-left corner with y down.
    """
    view = parts.view
    width, height = parts.target
    sx, sy = width / view.home_width, height / view.home_height
    w, h = view.home_width / 2, view.home_height / 2
    x0, y0, x1, y1 = box
    return ((x0 + w) * sx, (h - y1) * sy, (x1 + w) * sx, (h - y0) * sy)


def build_parts(
    record: Record,
    t: Fraction,
    play: PlayRecord | None,
    capture: Capture,
    *,
    answer: str,
    target: tuple[int, int],
    prepare: Callable[[Drawing], None] | None = None,
    live: bool = False,
) -> Parts:
    """The parts ``capture`` draws, the frame at ``t`` of ``play`` (``None``
    for the final capture): read from the record, every mobject given its
    captured state (and ``prepare`` run once it is), or with ``live`` from
    the scene as it is now, a forward run holding that capture. Pixel boxes
    are at ``target``, ``(width, height)``.
    """
    if live:
        # The run is drawing as it goes: its camera holds the map its frames
        # are drawn with, and its record is not finished.
        view = view_of(record._scene.renderer.camera)
        return _read(record, t, play, capture, answer, view, target)
    static, moving = _lists(play, capture)
    with Drawing(record) as drawing:
        drawing.show(capture, [*capture.draw_list(), *static, *moving])
        if prepare is not None:
            prepare(drawing)
        return _read(record, t, play, capture, answer, record_view(record), target)


def _lists(
    play: PlayRecord | None, capture: Capture
) -> tuple[list[Mobject], list[Mobject]]:
    """What a frame of ``capture`` draws, in order: the play's static
    mobjects under its moving ones, or the whole scene when nothing moves.
    """
    if play is not None and capture.moving:
        return list(play.static), list(capture.moving)
    return [], capture.draw_list()


def _read(
    record: Record,
    t: Fraction,
    play: PlayRecord | None,
    capture: Capture,
    answer: str,
    view: View,
    target: tuple[int, int],
) -> Parts:
    camera = record._scene.renderer.camera
    names = play.names if play is not None else record.final_names
    hidden = {id(mob) for mob in camera_nodes(camera)}
    static, moving = _lists(play, capture)
    order: dict[int, int] = {}
    drawn: list[Mobject] = []
    for position, mob in enumerate(
        [
            *camera.get_mobjects_to_display(static),
            *camera.get_mobjects_to_display(moving),
        ]
    ):
        if id(mob) not in order:
            drawn.append(mob)
        # A mobject drawn twice is seen where it was drawn last.
        order[id(mob)] = position
    chains = _chains([*capture.draw_list(), *static, *moving], hidden)
    reader = _Reader(record, camera, view, names, chains)
    texts: dict[int, list[Leaf]] = {}
    text_of: dict[int, Mobject] = {}
    parts: list[Part] = []
    for mob in drawn:
        reached = chains.get(id(mob))
        if reached is None or draw_kind(mob) is None:
            continue
        chain = reached[1]
        leaf = reader.leaf(mob, order[id(mob)])
        if leaf is None:
            continue
        text = next((m for m in (*chain, mob) if _is_text(m)), None)
        if text is None:
            parts.append(reader.part(mob, chain, (leaf,), kind=leaf.draw))
        else:
            texts.setdefault(id(text), []).append(leaf)
            text_of[id(text)] = text
    seen_ink: dict[bytes, int] = {}
    for key, leaves in texts.items():
        text = text_of[key]
        part = reader.part(text, chains[key][1], tuple(leaves), kind="text")
        ink = _ink_key(leaves)
        first = seen_ink.get(ink)
        if first is None:
            seen_ink[ink] = len(parts)
            parts.append(part)
        else:
            # One text, drawn as often as it has copies, the last on top.
            kept = parts[first]
            parts[first] = replace(
                kept, copies=kept.copies + 1, top=max(kept.top, part.top)
            )
    parts.sort(key=lambda part: part.order)
    index = play.index if play is not None else None
    return Parts(t, index, answer, view, tuple(parts), target)


def _chains(
    roots: list[Mobject], hidden: set[int]
) -> dict[int, tuple[Mobject, tuple[Mobject, ...]]]:
    """Each mobject reachable from ``roots`` (but the camera's own), by id,
    with the mobjects holding it, outermost first: where it is first reached.
    """
    chains: dict[int, tuple[Mobject, tuple[Mobject, ...]]] = {}
    stack: list[tuple[Mobject, tuple[Mobject, ...]]] = [
        (mob, ()) for mob in reversed(roots)
    ]
    while stack:
        mob, chain = stack.pop()
        if id(mob) in chains or id(mob) in hidden:
            continue
        chains[id(mob)] = (mob, chain)
        inner = (*chain, mob)
        stack.extend((child, inner) for child in reversed(mob.submobjects))
    return chains


def _text_types() -> tuple[type, ...]:
    from arachne.mobject.text.numbers import DecimalNumber
    from arachne.mobject.text.tex_mobject import SingleStringMathTex
    from arachne.mobject.text.text_mobject import MarkupText, Paragraph, Text
    from arachne.mobject.text.typst_mobject import Typst

    return (Text, MarkupText, Paragraph, SingleStringMathTex, DecimalNumber, Typst)


def _is_text(mob: Mobject) -> bool:
    return isinstance(mob, _text_types())


def _grid_types() -> tuple[type, ...]:
    from arachne.mobject.graphing.coordinate_systems import NumberPlane, PolarPlane
    from arachne.mobject.vector_field import VectorField

    return (NumberPlane, PolarPlane, VectorField)


def _furniture(line: Mobject) -> Iterator[Mobject]:
    """An axis's furniture: its line, and the families of its tip, ticks
    and numbers.
    """
    yield line
    for attribute in ("tip", "ticks", "numbers"):
        held = line.__dict__.get(attribute)
        if held is not None and hasattr(held, "get_family"):
            yield from held.get_family()


def _ink_key(leaves: list[Leaf]) -> bytes:
    points = np.vstack([leaf.points for leaf in leaves])
    return bytes(np.round(points, _INK_DECIMALS).tobytes())


class _Reader:
    """Reads leaves and parts off mobjects that hold the state shown."""

    def __init__(
        self,
        record: Record,
        camera: Any,
        view: View,
        names: dict[int, str],
        chains: dict[int, tuple[Mobject, tuple[Mobject, ...]]],
    ) -> None:
        from arachne.mobject.graphing.number_line import NumberLine

        self.record = record
        self.camera = camera
        self.view = view
        self.names = names
        #: The ids of axis furniture, and of what a grid holds or is.
        self.furniture: set[int] = set()
        self.grids: set[int] = set()
        grids = _grid_types()
        for key, (mob, chain) in chains.items():
            if any(isinstance(m, grids) for m in (*chain, mob)):
                self.grids.add(key)
            if isinstance(mob, NumberLine):
                self.furniture.update(id(m) for m in _furniture(mob))

    def leaf(self, mob: Mobject, order: int) -> Leaf | None:
        """``mob`` as drawn, or ``None`` when it draws nothing."""
        kind = draw_kind(mob)
        assert kind is not None
        points = np.asarray(mob.points, dtype=float)
        if not len(points):
            return None
        fill, stroke, width = _paint(mob, kind)
        if kind == "path" and fill <= 0 and width <= 0:
            return None
        if kind == "points" and stroke <= 0:
            return None
        screen = _screen(self.view, pixel_points(self.camera, self.view, mob, points))
        box = _box(screen)
        drawn = (
            width
            * self.view.cairo_line_width_multiple
            * self.view.home_width
            / self.view.frame_width
        )
        half = drawn / 2
        return Leaf(
            id=self.record.origin(mob).id,
            type=type(mob).__name__,
            draw=kind,
            points=screen,
            screen_box=box,
            ink_box=(box[0] - half, box[1] - half, box[2] + half, box[3] + half),
            scene_box=_box(points[:, :2]),
            fill_opacity=fill,
            stroke_opacity=stroke,
            stroke_width=drawn,
            closed=kind == "path" and _closed(mob, points),
            order=order,
            curve_size=int(getattr(mob, "n_points_per_cubic_curve", 4)),
        )

    def part(
        self,
        mob: Mobject,
        chain: tuple[Mobject, ...],
        leaves: tuple[Leaf, ...],
        *,
        kind: str,
    ) -> Part:
        from arachne.timeline.state import _text

        origin = self.record.origin(mob)
        holders = tuple(self._holder(m) for m in chain)
        is_text = kind == "text"
        return Part(
            id=origin.id,
            type=type(mob).__name__,
            kind=kind,
            name=self.names.get(id(mob)),
            source_line=origin.source_line,
            scene_line=scene_line(origin),
            text=(_text(mob) or "") if is_text else None,
            leaves=leaves,
            screen_box=_union(leaf.screen_box for leaf in leaves),
            ink_box=_union(leaf.ink_box for leaf in leaves),
            scene_box=_union(leaf.scene_box for leaf in leaves),
            order=min(leaf.order for leaf in leaves),
            top=max(leaf.order for leaf in leaves),
            root=holders[0].id if holders else origin.id,
            within=holders,
            furniture=id(mob) in self.furniture,
            grid=id(mob) in self.grids,
        )

    def _holder(self, mob: Mobject) -> Holder:
        origin = self.record.origin(mob)
        return Holder(
            origin.id, self.names.get(id(mob)), scene_line(origin), type(mob).__name__
        )


def _paint(mob: Mobject, kind: str) -> tuple[float, float, float]:
    """``mob``'s largest fill opacity, its largest visible stroke opacity and
    its widest visible stroke (background stroke included): what it lays.
    """
    if kind == "points":
        rgbas = np.asarray(getattr(mob, "rgbas", np.zeros((0, 4))))
        alpha = float(rgbas[:, 3].max()) if rgbas.size else 0.0
        return 0.0, alpha, 0.0
    if kind == "image":
        return 1.0, 0.0, 0.0
    fill = float(np.max(mob.get_fill_opacities(), initial=0.0))
    stroke = width = 0.0
    for background in (False, True):
        size = float(np.max(mob.get_stroke_width(background), initial=0.0))
        opacity = float(np.max(mob.get_stroke_opacities(background), initial=0.0))
        if size > 0 and opacity > 0:
            stroke, width = max(stroke, opacity), max(width, size)
    return fill, stroke, width


def _screen(view: View, pixels: np.ndarray) -> np.ndarray:
    """Pixels of ``view`` in home-frame units, centred, y up."""
    x = (pixels[:, 0] - view.pixel_width / 2) * view.home_width / view.pixel_width
    y = (view.pixel_height / 2 - pixels[:, 1]) * view.home_height / view.pixel_height
    return np.column_stack([x, y])


def _box(points: np.ndarray) -> Box:
    low, high = points.min(axis=0), points.max(axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _union(boxes: Any) -> Box:
    boxes = list(boxes)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _closed(mob: Mobject, points: np.ndarray) -> bool:
    """Whether every subpath of a path ends where it starts."""
    n = int(getattr(mob, "n_points_per_cubic_curve", 4))
    if len(points) < n:
        return False
    curves = points[: len(points) // n * n].reshape(-1, n, points.shape[1])
    start = 0
    for index in range(len(curves)):
        last = index == len(curves) - 1
        if last or not np.allclose(
            curves[index, -1], curves[index + 1, 0], atol=_SAME_POINT
        ):
            if not np.allclose(curves[start, 0], curves[index, -1], atol=_SAME_POINT):
                return False
            start = index + 1
    return True
