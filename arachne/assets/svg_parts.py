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
its file's flat paint.
"""

from __future__ import annotations

__all__ = ["SvgFile", "Unreadable", "read_svg", "structure"]

import io
import re
from collections.abc import Iterator, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import svgelements as se

from ..mobject.svg.svg_mobject import SVGMobject, VMobjectFromSVGPath
from ..mobject.types.vectorized_mobject import VMobject
from .svg_check import VIEWPORT, Unreadable, check


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
        such element. A shape with no points is left out.
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
    each is closed: what a stored outline shares with its rest outline.
    """
    return [
        [
            (len(subpath), bool(np.allclose(subpath[0], subpath[-1], atol=1e-9)))
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
    return mob
