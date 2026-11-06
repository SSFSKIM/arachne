"""What the lint's rules read at an instant: the scene's drawn parts.

A capture's mobjects are read as a frame draws them, in its paint order:
the play's static mobjects, then its moving ones, or the whole scene when
nothing moves, each family flattened into the members that have points.
The scene's camera's own mobjects (its frame, trackers and light) are how
it views and are not read; a zoomed display and another camera's frame the
scene shows are. Each drawn member is a :class:`Leaf`: its points where the
camera puts them, in screen units of the home frame (the configuration's
frame, centred, y up, so equal to scene units under a still camera), its
box, the box of what its draw paints (its ink, the shared projection's), its
box in scene units, its fill and stroke, and its place in the paint order. A
leaf whose draw paints nothing (fully transparent, as alignment padding is)
is left out.

A :class:`Part` is what a rule measures: a text (a ``Text``, ``MathTex``,
number and the like, whole) as the ink of its glyph leaves, and every other
leaf on its own, a text's background rectangle among them. Each part has its
stable id, creating line and name, the groups holding it, and whether it is
axis furniture (an axis line, its tip, ticks and numbers) or part of a grid
(a number plane or vector field, or a lattice of lines: :func:`_lattice`).
Two copies of one text laying the same ink
in one place are one text, counting its ``copies`` (a transform's copy left
on its original); a copy drawn otherwise (a wider stroke) is a text of its
own. Reading texts as whole and equal ink as one follows ManimGX's
diagnostics, credited in ``NOTICE.md``.
"""

from __future__ import annotations

__all__ = ["Leaf", "Part", "Parts", "PartsCache", "Unit", "build_parts"]

from collections.abc import Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.camera.projection import (
    Box,
    View,
    draw_kind,
    member_box,
    pixel_points,
    point_thickness,
    record_view,
    view_of,
)
from arachne.timeline.draw import Drawing

from .findings import FindingObject, Holder

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, PlayRecord, Record
    from arachne.timeline.source import SourceLine

#: Points closer than this (scene units) are one: where a subpath closes.
_SAME_POINT = 1e-6
#: Decimals ink is compared to, in screen units, for two copies of a text.
_INK_DECIMALS = 3
#: Lines a lattice needs each way.
LATTICE_LINES = 3
#: How near (a share of its box's longer side) a lattice line's end lies to
#: the edge of the box, and a point of it to its straight course.
_LATTICE_NEAR = 1e-3
#: Radians within which two lattice lines run one way.
_LATTICE_ANGLE = 1e-3


@dataclass(frozen=True)
class Unit:
    """The thing a clearance sweep measures a leaf as part of, whole.

    Walking down from the scene's top level, a vector group that draws
    nothing of its own opens into its members, and the first mobject that
    does not open is the unit: a shape with its tip, a dashed line with its
    dashes, a group that is no vector group, a container of the scene's own,
    a text with its own plate. Axes (and a lone number line) are their
    furniture instead: each axis's line, its tip, its ticks together, and
    each of its numbers, ``furniture`` naming which (``"line"``, ``"tip"``,
    ``"ticks"``, ``"number"``; ``None`` for content). ``text`` says whether
    the unit is a text (a text type, or a vector group of nothing but
    texts). ``within`` are the groups holding it, outermost first.
    ``family`` is what the unit is made from, each mobject in it once: the
    unit's own family, or for furniture its axes' (or lone number line's)
    whole family. An exemption naming any of these, or a group holding the
    unit, names the unit. ``tip`` says whether it carries a tip of its own
    (an arrow), which makes an open stroke a thing on screen. ``grid`` says
    whether it is part of a grid, as :attr:`Part.grid`.
    """

    id: str
    type: str
    name: str | None
    source_line: SourceLine | None
    scene_line: int | None
    text: bool
    furniture: str | None
    within: tuple[Holder, ...]
    family: tuple[Holder, ...] = ()
    tip: bool = False
    grid: bool = False


@dataclass(frozen=True, eq=False)
class Leaf:
    """One drawn member of a family, as the camera draws it.

    ``points`` are its points on the screen, ``(n, 2)`` in home-frame
    units: a path's Bézier control points (``curve_size`` per curve, as the
    drawing strokes them), a point cloud's points, an image's corners.
    ``ink_box`` is the box of what its draw paints, ``screen_box`` of its
    points. ``stroke_width`` is its widest visible stroke (its background
    stroke included) as drawn, in home-frame units, 0 when it has none; a
    point cloud's is the side of the square it draws each point as, though
    the frame cuts its ink. ``line_width`` is its foreground stroke alone,
    and ``halo_width`` and ``halo_opacity`` its background stroke, widths as
    drawn, each 0 when that stroke is not seen. ``first_fill_opacity`` is
    the opacity of its fill's first row, which says whether it is filled.
    ``unit`` is what a sweep measures it as part of (:class:`Unit`),
    ``None`` for a leaf no sweep measures: a point cloud, an image, or what
    is mounted on axes beside their furniture. ``plate`` says whether it is
    in its unit's background rectangle: under a text, not one of its glyphs.
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
    line_width: float = 0.0
    halo_width: float = 0.0
    halo_opacity: float = 0.0
    unit: Unit | None = None
    first_fill_opacity: float = 0.0
    plate: bool = False

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
    ``fixed`` holds the stable ids of what a ThreeDCamera keeps fixed in
    the frame there (empty under any other camera).
    """

    t: Fraction
    play: int | None
    answer: str
    view: View
    parts: tuple[Part, ...]
    target: tuple[int, int]
    fixed: frozenset[str] = frozenset()

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
    live: bool = False,
) -> Parts:
    """The parts ``capture`` draws, the frame at ``t`` of ``play`` (``None``
    for the final capture): read from the record, every mobject given its
    captured state, or with ``live`` from the scene as it is now, a forward
    run holding that capture, through the camera's inputs as the capture
    recorded them (its callbacks' values, not the callbacks). Pixel boxes
    are at ``target``, ``(width, height)``.
    """
    if live:
        # The run is drawing as it goes: its camera holds the map its frames
        # are drawn with, and its record is not finished. Read through the
        # values the capture recorded, so that reading runs none of the
        # scene's code and leaves the run as it would be unread.
        timeline = record._scene._arachne_timeline
        with Drawing(record) as drawing:
            drawing.show_camera(capture)
            view = view_of(
                record._scene.renderer.camera,
                home=timeline.home,
                configured_pixels=timeline.configured_pixels,
            )
            return _read(record, t, play, capture, answer, view, target)
    static, moving = _lists(play, capture)
    with Drawing(record) as drawing:
        drawing.show(capture, [*capture.draw_list(), *static, *moving])
        return _read(record, t, play, capture, answer, record_view(record), target)


class PartsCache:
    """The parts one lint built, each by everything :func:`build_parts`
    reads: the execution, the instant, its play, the capture shown, the
    answer, the target, and whether a live scene was read. A reader asking
    again for what another built is handed that; any other state is built.

    A live read is of a forward run's scene as the run holds the capture at
    ``t``: the run reaches each instant once, and every reader of it is
    shown the scene there before the run moves on.
    """

    def __init__(self) -> None:
        self._built: dict[
            tuple[Any, ...], tuple[Record, PlayRecord | None, Capture, Parts]
        ] = {}

    def build(
        self,
        record: Record,
        t: Fraction,
        play: PlayRecord | None,
        capture: Capture,
        *,
        answer: str,
        target: tuple[int, int],
        live: bool = False,
    ) -> Parts:
        """:func:`build_parts` of these, built once for the lint."""
        key = (id(record), t, id(play), id(capture), answer, target, live)
        found = self._built.get(key)
        if found is None:
            built = build_parts(
                record, t, play, capture, answer=answer, target=target, live=live
            )
            # What the key names by id is held, so no id in it is reused.
            found = (record, play, capture, built)
            self._built[key] = found
        return found[-1]


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
    hidden = _viewing_nodes(camera)
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
        leaf = reader.leaf(mob, order[id(mob)], chain)
        if leaf is None:
            continue
        text = next((m for m in (*chain, mob) if _is_text(m)), None)
        if text is None or _is_backdrop(mob, chain):
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
    extras = capture.camera
    fixed = frozenset(
        record.origin(m).id for m in (extras.fixed_in_frame if extras else ())
    )
    return Parts(t, index, answer, view, tuple(parts), target, fixed)


def _viewing_nodes(camera: Any) -> set[int]:
    """The ids of the camera's own mobjects: its frame, trackers and light,
    which it views with. The displays a camera keeps of other cameras'
    pictures are not among them: they are drawn as the scene's.
    """
    from arachne.mobject.mobject import Mobject

    return {id(value) for value in vars(camera).values() if isinstance(value, Mobject)}


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


def _is_backdrop(mob: Mobject, chain: tuple[Mobject, ...]) -> bool:
    """Whether ``mob`` is, or is in, a background rectangle: under a text,
    not its glyphs.
    """
    from arachne.mobject.geometry.shape_matchers import BackgroundRectangle

    return any(isinstance(m, BackgroundRectangle) for m in (*chain, mob))


def _grid_types() -> tuple[type, ...]:
    from arachne.mobject.graphing.coordinate_systems import NumberPlane, PolarPlane
    from arachne.mobject.vector_field import VectorField

    return (NumberPlane, PolarPlane, VectorField)


def _lattice(group: Mobject) -> bool:
    """Whether ``group``, a mobject at the scene's top level, is a lattice of
    lines: everything it draws a stroke that is open, straight and unfilled,
    in exactly two directions, at least :data:`LATTICE_LINES` each way, and
    each from one edge to the opposite edge of the box of the group's own
    lines (the group's bounding box, not the frame's). Lines one way only
    are not a grid: force lines or rulings, where a touch is real.
    """
    from arachne.mobject.types.vectorized_mobject import VMobject

    strokes = []
    for member in group.get_family():
        points = np.asarray(member.points, dtype=float)
        if not len(points):
            continue
        if not isinstance(member, VMobject) or np.any(member.get_fill_opacities() > 0):
            return False
        strokes.append(points[:, :2])
    if len(strokes) < 2 * LATTICE_LINES:
        return False
    every = np.vstack(strokes)
    low, high = every.min(axis=0), every.max(axis=0)
    near = _LATTICE_NEAR * float(max(high - low))
    if near <= 0:
        return False
    directions: list[tuple[float, int]] = []
    for points in strokes:
        start, end = points[0], points[-1]
        course = end - start
        length = float(np.hypot(*course))
        if length <= near:
            return False
        rel = points - start
        off = np.abs(course[0] * rel[:, 1] - course[1] * rel[:, 0]) / length
        if float(off.max()) > near or not _edge_to_edge(start, end, low, high, near):
            return False
        angle = float(np.arctan2(course[1], course[0])) % np.pi
        for index, (seen, count) in enumerate(directions):
            if min(abs(angle - seen), np.pi - abs(angle - seen)) <= _LATTICE_ANGLE:
                directions[index] = (seen, count + 1)
                break
        else:
            directions.append((angle, 1))
    return len(directions) == 2 and all(n >= LATTICE_LINES for _, n in directions)


def _edge_to_edge(
    start: np.ndarray, end: np.ndarray, low: np.ndarray, high: np.ndarray, near: float
) -> bool:
    """Whether a line's ends lie on opposite edges of the box ``low``,
    ``high`` (left and right, or bottom and top), within ``near``.
    """
    for axis in (0, 1):
        ends = sorted((float(start[axis]), float(end[axis])))
        if abs(ends[0] - low[axis]) <= near and abs(ends[1] - high[axis]) <= near:
            return True
    return False


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
    """What a text's leaves lay: their points, and each one's ink and
    stroke as drawn.
    """
    points = np.vstack([leaf.points for leaf in leaves])
    drawn = np.array([(*leaf.ink_box, leaf.stroke_width) for leaf in leaves])
    return b"".join(
        np.round(values, _INK_DECIMALS).tobytes() for values in (points, drawn)
    )


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
        self._units: dict[int, Unit] = {}
        #: Each source's family, and each unit's background rectangle (ids).
        self._families: dict[int, tuple[Holder, ...]] = {}
        self._plates: dict[str, set[int]] = {}
        #: The ids of axis furniture, and of what a grid holds or is.
        self.furniture: set[int] = set()
        self.grids: set[int] = set()
        grids = _grid_types()
        lattices = {
            key for key, (mob, chain) in chains.items() if not chain and _lattice(mob)
        }
        for key, (mob, chain) in chains.items():
            path = (*chain, mob)
            if any(isinstance(m, grids) for m in path) or id(path[0]) in lattices:
                self.grids.add(key)
            if isinstance(mob, NumberLine):
                self.furniture.update(id(m) for m in _furniture(mob))

    def leaf(
        self, mob: Mobject, order: int, chain: tuple[Mobject, ...] = ()
    ) -> Leaf | None:
        """``mob`` as drawn, held by ``chain`` (outermost first), or
        ``None`` when it draws nothing.
        """
        kind = draw_kind(mob)
        assert kind is not None
        points = np.asarray(mob.points, dtype=float)
        if not len(points):
            return None
        fill, stroke, width = _paint(mob, kind)
        if kind == "points" and stroke <= 0:
            return None
        ink = member_box(self.camera, self.view, mob, ink=True, home=True)
        if ink is None:
            return None
        screen = _screen(self.view, pixel_points(self.camera, self.view, mob, points))
        box = _box(screen)
        # Stroke widths as drawn: Cairo's units per width, on the home frame.
        scale = (
            self.view.cairo_line_width_multiple
            * self.view.home_width
            / self.view.frame_width
        )
        line, halo, halo_opacity = _strokes(mob, kind)
        drawn = point_thickness(self.view, mob) if kind == "points" else width * scale
        unit = self.unit(mob, chain) if kind == "path" else None
        return Leaf(
            id=self.record.origin(mob).id,
            type=type(mob).__name__,
            draw=kind,
            points=screen,
            screen_box=box,
            ink_box=ink,
            scene_box=_box(points[:, :2]),
            fill_opacity=fill,
            stroke_opacity=stroke,
            stroke_width=drawn,
            closed=kind == "path" and _closed(mob, points),
            order=order,
            curve_size=int(getattr(mob, "n_points_per_cubic_curve", 4)),
            line_width=line * scale,
            halo_width=halo * scale,
            halo_opacity=halo_opacity,
            unit=unit,
            first_fill_opacity=_first_fill(mob) if kind == "path" else fill,
            plate=unit is not None and id(mob) in self._plates.get(unit.id, ()),
        )

    def unit(self, mob: Mobject, chain: tuple[Mobject, ...]) -> Unit | None:
        """What a sweep measures ``mob`` (held by ``chain``) as part of:
        :class:`Unit`. ``None`` when no sweep measures it.
        """
        from arachne.mobject.graphing.coordinate_systems import Axes
        from arachne.mobject.graphing.number_line import NumberLine
        from arachne.mobject.types.vectorized_mobject import VGroup
        from arachne.mobject.value_tracker import ValueTracker

        path = (*chain, mob)
        for depth, held in enumerate(path):
            if isinstance(held, ValueTracker):
                return None
            if isinstance(held, (Axes, NumberLine)):
                return self._furniture_unit(held, path)
            if isinstance(held, VGroup) and held.submobjects and not len(held.points):
                continue
            return self._unit_of(held, path[:depth], furniture=None, source=held)
        return None

    def _furniture_unit(self, axes: Mobject, path: tuple[Mobject, ...]) -> Unit | None:
        """The furniture of ``axes`` the leaf at the end of ``path`` is in:
        an axis's line, its tip, its ticks or one of its numbers; ``None``
        for anything else mounted on it.
        """
        from arachne.mobject.graphing.coordinate_systems import Axes

        on = {id(m): index for index, m in enumerate(path)}
        lines = [axes.x_axis, axes.y_axis] if isinstance(axes, Axes) else [axes]
        for line in lines:
            if id(line) not in on:
                continue
            if path[-1] is line:
                return self._unit_of(
                    line, path[: on[id(line)]], furniture="line", source=axes
                )
            pieces = [
                ("tip", getattr(line, "tip", None)),
                ("ticks", getattr(line, "ticks", None)),
                *(("number", n) for n in getattr(line, "numbers", None) or ()),
            ]
            for role, piece in pieces:
                if piece is not None and id(piece) in on:
                    return self._unit_of(
                        piece, path[: on[id(piece)]], furniture=role, source=axes
                    )
        return None

    def _unit_of(
        self,
        mob: Mobject,
        above: tuple[Mobject, ...],
        *,
        furniture: str | None,
        source: Mobject,
    ) -> Unit:
        """The unit ``mob`` (held by ``above``), made from ``source``: itself,
        or the axes its furniture belongs to.
        """
        found = self._units.get(id(mob))
        if found is None:
            origin = self.record.origin(mob)
            found = Unit(
                id=origin.id,
                type=type(mob).__name__,
                name=self.names.get(id(mob)),
                source_line=origin.source_line,
                scene_line=origin.scene_line,
                text=_text_like(mob),
                furniture=furniture,
                within=tuple(self._holder(m) for m in above),
                family=self._family(source),
                tip=_has_tip(mob),
                grid=id(mob) in self.grids,
            )
            self._units[id(mob)] = found
            plate = getattr(mob, "background_rectangle", None)
            if furniture is None and plate is not None and hasattr(plate, "get_family"):
                self._plates[found.id] = {id(m) for m in plate.get_family()}
        return found

    def _family(self, source: Mobject) -> tuple[Holder, ...]:
        found = self._families.get(id(source))
        if found is None:
            seen: dict[int, Holder] = {}
            for member in source.get_family():
                if id(member) not in seen:
                    seen[id(member)] = self._holder(member)
            found = tuple(seen.values())
            self._families[id(source)] = found
        return found

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
            scene_line=origin.scene_line,
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
            origin.id, self.names.get(id(mob)), origin.scene_line, type(mob).__name__
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


def _strokes(mob: Mobject, kind: str) -> tuple[float, float, float]:
    """``mob``'s foreground stroke width, and its background stroke's width
    and opacity, each 0 where that stroke is not seen; widths as set, before
    the camera scales them.
    """
    if kind != "path":
        return 0.0, 0.0, 0.0
    seen = []
    for background in (False, True):
        size = float(np.max(mob.get_stroke_width(background), initial=0.0))
        opacity = float(np.max(mob.get_stroke_opacities(background), initial=0.0))
        seen.append((size, opacity) if size > 0 and opacity > 0 else (0.0, 0.0))
    (line, _), (halo, halo_opacity) = seen
    return line, halo, halo_opacity


def _first_fill(mob: Mobject) -> float:
    """The opacity of the first row of ``mob``'s fill (0 when it has none),
    which a vector mobject reports as its fill opacity.
    """
    opacities = np.asarray(mob.get_fill_opacities(), dtype=float).ravel()
    return float(opacities[0]) if opacities.size else 0.0


def _has_tip(mob: Mobject) -> bool:
    """Whether ``mob`` carries a tip of its own, as an arrow does."""
    has_tip = getattr(mob, "has_tip", None)
    return bool(callable(has_tip) and has_tip())


def _text_like(mob: Mobject) -> bool:
    """Whether a sweep holds ``mob`` as a text: a text type, or a vector
    group of nothing but texts.
    """
    from arachne.mobject.text.numbers import DecimalNumber
    from arachne.mobject.text.tex_mobject import MathTex
    from arachne.mobject.text.text_mobject import MarkupText, Text
    from arachne.mobject.text.typst_mobject import Typst
    from arachne.mobject.types.vectorized_mobject import VGroup

    if isinstance(mob, (Text, MarkupText, MathTex, DecimalNumber, Typst)):
        return True
    return (
        isinstance(mob, VGroup)
        and bool(mob.submobjects)
        and all(_text_like(sub) for sub in mob.submobjects)
    )


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
    # A subpath ends at each curve the next does not start where it ends,
    # and at the last.
    joined = np.isclose(curves[:-1, -1], curves[1:, 0], atol=_SAME_POINT).all(axis=1)
    ends = np.append(np.flatnonzero(~joined), len(curves) - 1)
    starts = np.insert(ends[:-1] + 1, 0, 0)
    return bool(np.isclose(curves[starts, 0], curves[ends, -1], atol=_SAME_POINT).all())
