"""The asset reader's SVG: a part file read from memory into vector shapes.

An asset's part files are drawn exactly or not at all. :func:`read_svg`
parses a file's bytes, writes nothing anywhere, and refuses
(:class:`~.svg_check.Unreadable`, with the reason) a file that is not SVG or
that holds anything the shape conversion would not draw exactly as the file
says (:mod:`.svg_check`, which fails closed).

Coordinates are the file's user units, y down: the asset's canvas. The root's
``viewBox`` is kept to compare canvases, never applied, so a file's
``width`` and ``height`` scale nothing. Shapes are converted by the
conversion :class:`~arachne.mobject.svg.svg_mobject.SVGMobject` uses, each
an ordinary :class:`~arachne.mobject.types.vectorized_mobject.VMobject` with
its file's flat paint and stroke join, its stroke as wide as the file's
transform makes it. A stroke the engine would draw otherwise than the file
means is refused: one under a transform that is not a similarity, an open
subpath ending where it starts, a subpath starting where the one before it
ends, and a miter join the engine's own miter limit cuts otherwise than the
file's (:mod:`.joins`).

Two such strokes still pass, drawn otherwise than SVG draws them:

- ends that nearly meet. Whether a subpath's ends, or one subpath's end and
  the next one's start, meet is judged in canvas units by the camera's own
  test, but the camera tests again in scene units, after the asset's scale,
  with the same absolute tolerance. An asset drawn small can therefore have
  ends the reader holds apart and the camera joins: ``M 0 0 L 100 0
  L 0 0.00001`` passes, and is closed once the scale makes the gap smaller
  than the camera's tolerance;
- drawing on after ``Z`` with no move: in ``M 0 0 L 10 0 L 10 10 Z L 5 5`` SVG
  starts a new subpath at ``0 0`` and caps it there, while the engine runs the
  closed outline on into ``L 5 5`` and joins the two.
"""

from __future__ import annotations

__all__ = ["SvgFile", "Unreadable", "read_svg", "structure"]

import io
import re
from collections.abc import Iterator, Sequence
from itertools import pairwise
from typing import NamedTuple
from xml.etree import ElementTree as ET

import numpy as np
import svgelements as se

from ..constants import LineJointType
from ..mobject.svg.svg_mobject import SVGMobject, VMobjectFromSVGPath
from ..mobject.types.vectorized_mobject import VMobject
from .joins import ENGINE_LIMIT, disagreeing_corner, engine_closes
from .svg_check import VIEWPORT, Unreadable, check, prune

#: The miter limit a file means when it states none.
SVG_LIMIT = 4.0
#: How far from a similarity a transform may be, relative to its scale.
_SIMILAR = 1e-9
#: The joins other than miter the engine strokes as SVG does.
_JOINS = {"round": LineJointType.ROUND, "bevel": LineJointType.BEVEL}


class SvgFile:
    """A part file, parsed: its ``view_box`` (``None`` without one) and its
    shapes, by the ``id`` of the element holding them.
    """

    def __init__(self, view_box: tuple[float, ...] | None, root: se.SVG) -> None:
        self.view_box = view_box
        self._root = root

    def shapes(self, element: str | None = None) -> list[VMobject] | None:
        """The shapes the file draws, in document order, in canvas
        coordinates: all of them, or those of the element whose ``id`` is
        ``element`` (a group, or one shape); ``None`` when the file has no
        such element. A shape with no points is left out. Raises
        :class:`Unreadable` for a miter join the engine would draw otherwise
        than the file means.
        """
        if element is None:
            found: se.SVGElement = self._root
        else:
            target = _find(self._root, element)
            if target is None:
                return None
            found = target
        return [mob for shape in _drawn(found) if (mob := _convert(shape)) is not None]


def read_svg(data: bytes) -> SvgFile:
    """The part file whose bytes are ``data``; raises :class:`Unreadable`."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise Unreadable(f"it is not well-formed XML ({error})") from None
    prune(root)
    check(root)
    view_box = _view_box(root.get("viewBox"))
    for name in VIEWPORT:
        root.attrib.pop(name, None)
    try:
        parsed = se.SVG.parse(io.BytesIO(ET.tostring(root)), reify=True)
    except Exception as error:
        raise Unreadable(f"its shapes cannot be parsed ({error})") from None
    return SvgFile(view_box, parsed)


def structure(shapes: Sequence[VMobject]) -> list[list[tuple[int, bool]]]:
    """Each shape's subpaths, in order, as their point counts and whether
    the engine strokes each closed: what a stored outline shares with its
    rest outline.
    """
    return [
        [
            (len(subpath), engine_closes(shape, subpath))
            for subpath in shape.get_subpaths()
        ]
        for shape in shapes
    ]


def _view_box(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        numbers = tuple(float(item) for item in re.split(r"[\s,]+", value.strip()))
    except ValueError:
        numbers = ()
    if len(numbers) != 4:
        raise Unreadable(f"its viewBox {value!r} is not four numbers")
    return numbers


def _find(node: se.SVGElement, element: str) -> se.SVGElement | None:
    """The first element whose ``id`` is ``element``, in document order,
    outside what a ``use`` copies.
    """
    for child in _children(node):
        if child.values.get("id") == element:
            return child
        if isinstance(child, se.Group):
            found = _find(child, element)
            if found is not None:
                return found
    return None


def _children(node: se.SVGElement) -> list[se.SVGElement]:
    return list(node) if isinstance(node, (se.Group, se.Use)) else []


def _drawn(node: se.SVGElement) -> Iterator[se.SVGElement]:
    """The shapes in ``node`` and below, in document order."""
    if isinstance(node, (se.Group, se.Use)):
        for child in node:
            yield from _drawn(child)
    elif isinstance(node, se.Shape):
        yield node


def _convert(shape: se.SVGElement) -> VMobject | None:
    """``shape`` as a vector object, by :class:`SVGMobject`'s conversion;
    ``None`` for one with no points.
    """
    mob: VMobject
    if isinstance(shape, se.Path):
        mob = VMobjectFromSVGPath(shape)
    elif isinstance(shape, se.SimpleLine):
        mob = SVGMobject.line_to_mobject(shape)
    elif isinstance(shape, se.Rect):
        mob = SVGMobject.rect_to_mobject(shape)
    elif isinstance(shape, (se.Circle, se.Ellipse)):
        mob = SVGMobject.ellipse_to_mobject(shape)
    elif isinstance(shape, se.Polygon):
        mob = SVGMobject.polygon_to_mobject(shape)
    elif isinstance(shape, se.Polyline):
        mob = VMobject().set_points_as_corners([[x, y, 0.0] for x, y in shape])
    else:
        return None
    if not mob.has_points():
        return None
    SVGMobject.apply_style_to_mobject(mob, shape)
    if shape.apply:
        SVGMobject.handle_transform(mob, shape.transform)
    if _stroked(mob):
        _stroke(mob, shape)
    return mob


def _stroked(mob: VMobject) -> bool:
    return mob.get_stroke_width() > 0 and mob.get_stroke_opacity() > 0


def _stroke(mob: VMobject, shape: se.Shape) -> None:
    """Draw ``mob``'s stroke as its file means, or refuse it: SVG strokes a
    shape in its own coordinates, then transforms it; the engine strokes the
    transformed outline at one width, closing every subpath that ends where
    it starts and running on into the next subpath that starts where one
    ends. The two agree under a similarity, at the width it scales the
    stroke to, on a stroke whose open subpaths end apart from their start
    and whose subpaths each start apart from where the one before ends,
    and at joins :func:`_join` accepts.
    """
    transform = str(shape.values.get("transform", "")).strip()
    if not _similarity(se.Matrix(transform)):
        raise Unreadable(
            f"a stroke under the transform {transform!r}, which scales unevenly or"
            " skews; SVG strokes before the transform, the engine at one width"
            " after it: apply the transform to the path, or stroke after it"
        )
    mob.set_stroke(width=shape.implicit_stroke_width)
    runs = _runs(shape)
    for run in runs:
        if not run.closed and mob.consider_points_equals_2d(run.start, run.end):
            x, y = run.end
            raise Unreadable(
                f"a stroked subpath ends at ({x:g}, {y:g}), where it starts,"
                " without Z; SVG caps both ends and the engine joins them:"
                " close it with Z, or end it apart from its start"
            )
    for before, after in pairwise(runs):
        if after.moved_to and mob.consider_points_equals_2d(before.end, after.start):
            x, y = after.start
            raise Unreadable(
                f"a stroked subpath starts at ({x:g}, {y:g}), where the one before"
                " it ends; SVG caps both there and the engine strokes them as one,"
                " joined: keep the segments as separate elements, or join their"
                " nodes"
            )
    _join(mob, shape)


def _similarity(matrix: se.Matrix) -> bool:
    """Whether ``matrix`` keeps shapes: a uniform scale, rotation,
    reflection and translation, its two columns at right angles and of one
    length.
    """
    a, b, c, d = (float(v) for v in (matrix.a, matrix.b, matrix.c, matrix.d))
    scale = a * a + b * b + c * c + d * d
    slack = _SIMILAR * scale
    return abs(a * c + b * d) <= slack and abs(a * a + b * b - c * c - d * d) <= slack


class _Run(NamedTuple):
    """A subpath SVG draws, in canvas coordinates: where it starts and ends,
    whether a ``Z`` closes it, and whether a move starts it.
    """

    start: np.ndarray
    end: np.ndarray
    closed: bool
    moved_to: bool


def _runs(shape: se.Shape) -> list[_Run]:
    """The subpaths SVG draws in ``shape`` that go anywhere from their
    start, in order; none for a shape that is closed whole (a rect, circle,
    ellipse or polygon) or a line.
    """
    found: list[tuple[se.Point, se.Point, bool, bool]] = []
    if isinstance(shape, se.Path):
        start: se.Point | None = None
        end: se.Point | None = None
        moved = False  # whether the subpath goes anywhere yet
        moved_to = False
        for segment in shape:
            if isinstance(segment, se.Move):  # ends an open subpath
                if moved:
                    found.append((start, end, False, moved_to))
                start = end = segment.end
                moved, moved_to = False, True
            elif isinstance(segment, se.Close):  # the next starts where it did
                if moved:
                    found.append((start, start, True, moved_to))
                end = start
                moved, moved_to = False, False
            else:
                start = segment.start if start is None else start
                end = segment.end
                moved = moved or segment.length() > 0
        if moved:
            found.append((start, end, False, moved_to))
    elif isinstance(shape, se.Polyline) and len(shape) > 1:
        found.append((shape[0], shape[-1], False, True))
    matrix = se.Matrix(shape.transform) if shape.apply else se.Matrix()
    return [
        _Run(
            np.array(_xy(matrix.point_in_matrix_space(first))),
            np.array(_xy(matrix.point_in_matrix_space(last))),
            closed,
            moved_to,
        )
        for first, last, closed, moved_to in found
    ]


def _xy(point: se.Point) -> tuple[float, float]:
    return (float(point.x), float(point.y))


def _join(mob: VMobject, shape: se.SVGElement) -> None:
    """Give ``mob`` its file's ``stroke-linejoin``; refuse a miter join the
    engine would draw otherwise than the file's ``stroke-miterlimit`` means
    (:mod:`.joins`).
    """
    join = str(shape.values.get("stroke-linejoin", "miter")).strip()
    if join != "miter":
        mob.joint_type = _JOINS[join]
        return
    limit = float(str(shape.values.get("stroke-miterlimit", SVG_LIMIT)).strip())
    found = disagreeing_corner(mob, limit)
    if found is not None:
        angle, (x, y) = found
        raise Unreadable(
            f"a miter join of {angle:.1f} degrees at ({x:g}, {y:g}) is cut to a"
            f" bevel under one of its stroke-miterlimit {limit:g} and the engine's"
            f' {ENGINE_LIMIT:g}, and not the other: set stroke-miterlimit="10" or'
            " use a round or bevel join"
        )
