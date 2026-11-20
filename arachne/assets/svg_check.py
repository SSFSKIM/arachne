"""What a part file may hold: everything the shape conversion draws exactly
as the file says, and nothing else.

:func:`check` reads a part file's element tree before it is converted and
refuses (:class:`Unreadable`, with the reason) anything outside that set. It
fails closed: an element, an attribute or a style property it does not list
is refused, not passed over. What it lists:

- the elements ``svg`` (the root only), ``g``, ``defs``, ``use``, ``path``,
  ``rect``, ``circle``, ``ellipse``, ``line``, ``polyline`` and ``polygon``;
  ``title``, ``desc`` and ``metadata``, whose contents nothing draws, and
  elements and attributes of other namespaces, which nothing draws either
  (:func:`prune` removes those elements before the check and the
  conversion read the tree);
- each element's own geometry, as plain numbers in user units (``px`` at
  most): no percentage and no other unit, since the canvas is the file's user
  units and its viewport is never applied; a ``d`` or ``points`` that parses
  whole; a ``transform`` of the six SVG functions;
- flat paint: ``fill`` and ``stroke`` of a colour, ``none`` or
  ``currentColor``; their opacities; ``stroke-width``; ``color``;
- ``stroke-linejoin`` ``miter``, ``round`` or ``bevel``, and a
  ``stroke-miterlimit`` of at least 1 (whether the engine draws a miter join
  as its limit means is the reader's to decide, corner by corner);
- properties the conversion drops, only at the one value it draws:
  ``stroke-linecap`` ``butt``, ``stroke-dasharray`` ``none``,
  ``stroke-dashoffset`` 0, ``fill-rule`` ``nonzero``, ``opacity`` 1,
  ``display`` ``inline``, ``visibility`` ``visible``, ``paint-order``
  ``normal``, ``vector-effect`` ``none``, and ``none`` for ``clip-path``,
  ``mask``, ``filter`` and the markers;
- ``id``, ``class``, ``data-*`` and rendering hints, which draw nothing;
- ``href`` only to an element of the same file that exists.
"""

from __future__ import annotations

__all__ = ["Unreadable", "check", "prune"]

import math
import re
from collections.abc import Callable
from xml.etree import ElementTree as ET

import svgelements as se

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"


class Unreadable(Exception):
    """A part file the reader cannot draw exactly; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBERS = re.compile(_NUMBER)
_PLAIN = re.compile(rf"\s*({_NUMBER})\s*")
_LENGTH = re.compile(rf"\s*({_NUMBER})(?:px)?\s*")
_LIST = re.compile(rf"[\s,]*(?:{_NUMBER}[\s,]*)*")
_PATH = re.compile(rf"(?:[\s,]|[MmZzLlHhVvCcSsQqTtAa]|{_NUMBER})*")
_FUNCTION = re.compile(r"\s*([A-Za-z]+)\s*\(([^()]*)\)[\s,]*")
#: How many numbers each transform function takes.
_TRANSFORMS = {
    "matrix": (6,),
    "translate": (1, 2),
    "scale": (1, 2),
    "rotate": (1, 3),
    "skewX": (1,),
    "skewY": (1,),
}
_HEX = re.compile(r"#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})")
_BLACK = se.Color.rgb_to_int(0, 0, 0)

#: The shapes, and the geometry attributes of each element.
SHAPES = frozenset({"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"})
_GEOMETRY: dict[str, tuple[str, ...]] = {
    "g": (),
    "defs": (),
    "use": ("x", "y"),
    "path": ("d",),
    "rect": ("x", "y", "width", "height", "rx", "ry"),
    "circle": ("cx", "cy", "r"),
    "ellipse": ("cx", "cy", "rx", "ry"),
    "line": ("x1", "y1", "x2", "y2"),
    "polyline": ("points",),
    "polygon": ("points",),
}
#: Lengths that may not be negative.
_EXTENTS = frozenset({"width", "height", "r", "rx", "ry"})
#: The root's attributes that place its viewport, which the canvas ignores.
VIEWPORT = ("width", "height", "viewBox", "preserveAspectRatio", "x", "y")
_ROOT = (*VIEWPORT, "version", "baseProfile", "enable-background")
#: The elements nothing draws, whose contents are passed over.
_INERT = frozenset({"title", "desc", "metadata"})
#: Attributes and properties that draw nothing.
_INERT_ATTRIBUTES = frozenset(
    {
        "id",
        "class",
        "shape-rendering",
        "color-rendering",
        "color-interpolation",
        "enable-background",
    }
)
#: Properties the conversion drops, each with the one value it draws.
_FIXED: dict[str, str | float] = {
    "stroke-linecap": "butt",
    "stroke-dasharray": "none",
    "stroke-dashoffset": 0.0,
    "fill-rule": "nonzero",
    "opacity": 1.0,
    "display": "inline",
    "visibility": "visible",
    "paint-order": "normal",
    "vector-effect": "none",
    "clip-path": "none",
    "mask": "none",
    "filter": "none",
    "marker": "none",
    "marker-start": "none",
    "marker-mid": "none",
    "marker-end": "none",
}


def prune(root: ET.Element) -> None:
    """Remove from ``root``'s tree, in place, every subtree nothing draws:
    ``title``, ``desc``, ``metadata`` and other namespaces' elements, with
    all they hold. The check and the conversion then read one tree, so no
    shape the check passes over is drawn.
    """
    for parent in list(root.iter()):
        for child in list(parent):
            if _inert(child):
                parent.remove(child)


def check(root: ET.Element) -> None:
    """Refuse the part file whose tree, :func:`pruned <prune>`, is ``root``
    unless the conversion draws all of it exactly; raises
    :class:`Unreadable`.
    """
    if _local(root.tag) != "svg" or _namespace(root.tag) not in ("", SVG):
        raise Unreadable("its root element is not <svg>")
    ids = {found for element in root.iter() if (found := element.get("id")) is not None}
    for element in root.iter():
        name = _local(element.tag)
        if element is not root and name not in _GEOMETRY:
            raise Unreadable(
                f"it holds a <{name}> element, which the reader does not draw"
            )
        geometry = () if element is root else _GEOMETRY[name]
        for attribute, value in element.attrib.items():
            if element is root and attribute in _ROOT:
                continue  # the viewport, which the canvas ignores
            _attribute(name, attribute, value, geometry, ids)


def _local(tag: str) -> str:
    return tag.rpartition("}")[2]


def _namespace(tag: str) -> str:
    return tag[1:].partition("}")[0] if tag.startswith("{") else ""


def _inert(element: ET.Element) -> bool:
    """Whether nothing in ``element`` is drawn: a comment, another
    namespace's element, or one of :data:`_INERT`.
    """
    return (
        not isinstance(element.tag, str)
        or _namespace(element.tag) not in ("", SVG)
        or _local(element.tag) in _INERT
    )


def _attribute(
    element: str, attribute: str, value: str, geometry: tuple[str, ...], ids: set[str]
) -> None:
    namespace, name = _namespace(attribute), _local(attribute)
    if name == "href" and namespace in ("", XLINK):
        _reference(element, value, ids)
        return
    if namespace or name in _INERT_ATTRIBUTES or name.startswith("data-"):
        return
    if name in geometry:
        _geometry(element, name, value)
    elif name == "transform":
        _transform(element, value)
    elif name == "style":
        for item in value.split(";"):
            if item.strip():
                key, _, found = item.partition(":")
                _property(element, key.strip(), found.strip())
    else:
        _property(element, name, value.strip())


def _refuse(element: str, what: str) -> Unreadable:
    return Unreadable(f"its <{element}> has {what}")


def _reference(element: str, value: str, ids: set[str]) -> None:
    target = value.strip()
    if not target.startswith("#"):
        raise Unreadable(
            f"its <{element}> refers to {value!r}, which is not a part of the same file"
        )
    if target[1:] not in ids:
        raise _refuse(
            element, f"a reference to {value!r}, which the file does not hold"
        )


def _number(element: str, name: str, value: str, pattern: re.Pattern[str]) -> float:
    matched = pattern.fullmatch(value)
    if matched is None:
        why = (
            "a percentage, which the canvas cannot resolve"
            if "%" in value
            else "not a plain number in user units"
        )
        raise _refuse(element, f"{name} {value!r}, {why}")
    number = float(matched.group(1))
    if not math.isfinite(number):
        raise _refuse(element, f"{name} {value!r}, which is not a finite number")
    return number


def _finite(element: str, name: str, value: str) -> list[float]:
    numbers = [float(n) for n in _NUMBERS.findall(value)]
    if not all(math.isfinite(n) for n in numbers):
        raise _refuse(element, f"a {name} with a number that is not finite")
    return numbers


def _geometry(element: str, name: str, value: str) -> None:
    if name == "d":
        if _PATH.fullmatch(value) is None:
            raise _refuse(element, f"path data {value!r} that is not SVG path data")
        _finite(element, name, value)
        try:
            se.Path(value)
        except Exception:
            raise _refuse(element, f"path data {value!r} that does not parse") from None
    elif name == "points":
        if _LIST.fullmatch(value) is None or len(_finite(element, name, value)) % 2:
            raise _refuse(element, f"points {value!r} that are not pairs of numbers")
    elif _number(element, name, value, _LENGTH) < 0 and name in _EXTENTS:
        raise _refuse(element, f"{name} {value!r}, which is negative")


def _transform(element: str, value: str) -> None:
    rest = value
    while rest.strip():
        matched = _FUNCTION.match(rest)
        if matched is None:
            raise _refuse(element, f"a transform {value!r} that does not parse")
        function, arguments = matched.groups()
        if _LIST.fullmatch(arguments) is None or len(
            _finite(element, "transform", arguments)
        ) not in _TRANSFORMS.get(function, ()):
            raise _refuse(element, f"a transform {value!r} that does not parse")
        rest = rest[matched.end() :]


def _colour(value: str) -> bool:
    if value in ("none", "currentColor") or _HEX.fullmatch(value):
        return True
    if any(
        pattern.fullmatch(value)
        for pattern in (
            se.REGEX_COLOR_RGB,
            se.REGEX_COLOR_RGB_PERCENT,
            se.REGEX_COLOR_HSL,
        )
    ):
        return True
    named = value.replace(" ", "").lower()
    return named == "black" or se.Color.parse_color_lookup(named) != _BLACK


def _opacity(value: str) -> bool:
    matched = _PLAIN.fullmatch(value)
    return matched is not None and 0 <= float(matched.group(1)) <= 1


_PROPERTIES: dict[str, Callable[[str], bool]] = {
    "fill": _colour,
    "stroke": _colour,
    "color": lambda value: value != "currentColor" and _colour(value),
    "fill-opacity": _opacity,
    "stroke-opacity": _opacity,
    "stroke-linejoin": lambda value: value in ("miter", "round", "bevel"),
    "stroke-miterlimit": lambda value: (
        (matched := _PLAIN.fullmatch(value)) is not None
        and float(matched.group(1)) >= 1
    ),
}


def _property(element: str, name: str, value: str) -> None:
    if name in ("fill", "stroke") and "url(" in value:
        raise _refuse(
            element,
            f"{name} {value!r}, a gradient or pattern; the reader draws flat colours only",
        )
    accepted = _PROPERTIES.get(name)
    if accepted is not None:
        if not accepted(value):
            raise _refuse(element, f"{name} {value!r}, which the reader does not draw")
        return
    if name == "stroke-width":
        if _number(element, name, value, _LENGTH) < 0:
            raise _refuse(element, f"{name} {value!r}, which is negative")
        return
    if name not in _FIXED:
        raise _refuse(element, f"{name} {value!r}, which the reader does not draw")
    drawn = _FIXED[name]
    if isinstance(drawn, float):
        matched = _PLAIN.fullmatch(value)
        same = matched is not None and float(matched.group(1)) == drawn
    else:
        same = value == drawn
    if not same:
        raise _refuse(
            element,
            f"{name} {value!r}, which the conversion drops: it draws only {drawn!r}",
        )
