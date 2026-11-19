"""The asset reader's SVG: a part file read from memory into vector shapes.

An asset's part files are drawn exactly or not at all. :func:`read_svg`
parses a file's bytes, writes nothing anywhere, and refuses
(:class:`Unreadable`, with the reason) a file that is not SVG, or that holds
anything the shape conversion would drop or change: text, an embedded image,
a gradient or pattern paint, a clip path, a mask, a filter, a marker, a
style sheet, a nested viewport, or a reference to anything but a fragment of
the same file. Elements of other namespaces, and the contents of ``title``,
``desc`` and ``metadata``, are not drawn by anything and are passed over.

Coordinates are the file's user units, y down: the asset's canvas. The root's
``viewBox`` is kept to compare canvases, never applied, so a file's
``width`` and ``height`` scale nothing. Shapes are converted by the
conversion :class:`~arachne.mobject.svg.svg_mobject.SVGMobject` uses, each
an ordinary :class:`~arachne.mobject.types.vectorized_mobject.VMobject` with
its file's flat paint.
"""

from __future__ import annotations

__all__ = ["SvgFile", "Unreadable", "read_svg"]

import io
import re
from collections.abc import Iterator
from xml.etree import ElementTree as ET

import svgelements as se

from ..mobject.svg.svg_mobject import SVGMobject, VMobjectFromSVGPath
from ..mobject.types.vectorized_mobject import VMobject

_SVG = "http://www.w3.org/2000/svg"
_XLINK = "http://www.w3.org/1999/xlink"

#: The elements whose shapes are drawn.
_SHAPES = frozenset(
    {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
)
#: The elements that only hold or place others.
_STRUCTURE = frozenset({"g", "defs", "use"})
#: The elements nothing draws, whose contents are passed over.
_INERT = frozenset({"title", "desc", "metadata"})
#: Presentation properties whose any value but ``none`` the conversion drops.
_DROPPED = (
    "clip-path",
    "mask",
    "filter",
    "marker",
    "marker-start",
    "marker-mid",
    "marker-end",
)
#: The root's attributes that place its viewport, which the canvas ignores.
_VIEWPORT = ("width", "height", "viewBox", "preserveAspectRatio", "x", "y")


class Unreadable(Exception):
    """A part file the reader cannot draw exactly; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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
    if _local(root.tag) != "svg" or _namespace(root.tag) not in ("", _SVG):
        raise Unreadable("its root element is not <svg>")
    _check(root, is_root=True)
    view_box = _view_box(root.get("viewBox"))
    for name in _VIEWPORT:
        root.attrib.pop(name, None)
    try:
        parsed = se.SVG.parse(io.BytesIO(ET.tostring(root)), reify=True)
    except Exception as error:
        raise Unreadable(f"its shapes cannot be parsed ({error})") from None
    return SvgFile(view_box, parsed)


def _local(tag: str) -> str:
    return tag.rpartition("}")[2]


def _namespace(tag: str) -> str:
    return tag[1:].partition("}")[0] if tag.startswith("{") else ""


def _check(element: ET.Element, *, is_root: bool = False) -> None:
    """Refuse what in ``element`` and below the conversion would not draw as
    the file says.
    """
    if not isinstance(element.tag, str) or _namespace(element.tag) not in ("", _SVG):
        return  # a comment, or another namespace's: nothing draws it
    name = _local(element.tag)
    if name in _INERT:
        return
    if not (is_root or name in _SHAPES or name in _STRUCTURE):
        raise Unreadable(f"it holds a <{name}> element, which the reader does not draw")
    for attribute, value in element.attrib.items():
        _check_attribute(name, attribute, value)
    for child in element:
        _check(child)


def _check_attribute(element: str, attribute: str, value: str) -> None:
    namespace, local = _namespace(attribute), _local(attribute)
    if local == "href" and namespace in ("", _XLINK):
        if not value.strip().startswith("#"):
            raise Unreadable(
                f"its <{element}> refers to {value!r}, which is not a part of the"
                " same file"
            )
        return
    if namespace:
        return
    properties = [(local, value)]
    if local == "style":
        properties = [
            (key.strip(), found.strip())
            for key, _, found in (item.partition(":") for item in value.split(";"))
        ]
    for key, found in properties:
        if key in ("fill", "stroke") and "url(" in found:
            raise Unreadable(
                f"its <{element}> is painted with {found!r}, a gradient or pattern;"
                " the reader draws flat colours only"
            )
        if key in _DROPPED and found not in ("", "none"):
            raise Unreadable(
                f"its <{element}> has {key} {found!r}, which the reader does not draw"
            )


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
