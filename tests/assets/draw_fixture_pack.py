"""Draw the engine's fixture assets: every part file from the engine's own
vector objects, then every manifest.

The pack is six plain assets, each exercising part of the manifest:

- ``lidded-box``: a body and a lid on a hinge, both groups of one file; a
  ``rotate``, an ``outline`` anchor.
- ``lidded-crate``: the same part names in another drawing, for morphing.
- ``dial``: a face, a needle and a lamp slot, each a file of its own; a
  ``rotate`` whose default is mid-range.
- ``gate-valve``: a body, a stem that slides and a state slot; a
  ``translate``.
- ``figure``: torso, head, upper and lower arm, a hand slot and an eyes slot;
  a parent chain three deep; a ``warp`` of the head with three keys; one
  parameter that rotates the upper arm and translates the hand; four anchors.
- ``layered-section``: four nested layers, one swelling under a ``shape``
  driver with three keys; an ``outline`` anchor at its tip and an
  ``interior`` anchor with keyed positions.

Then each asset is admitted by the engine's admission tool, and its
manifest records the checks' results with its digest, every one passing.

Three more assets each carry one defect admission must find, and nothing
else it finds: ``open-seam`` (a pane that slides in its frame and opens a
seam inside the figure: ``hole``), ``loose-part`` (a handle pulled clear of
its body: ``detached_part``) and ``stray-anchor`` (a lid's anchor named on
the body: ``anchor_off_part``). Their manifests record the checks as the tool
finds them, failing.

Numbers are rounded to three decimals and files are written as the
repository's hooks leave them (no trailing space, one final newline), so that
drawing the pack again gives the committed files.

Run as ``python tests/assets/draw_fixture_pack.py <directory> [<defects>]``;
the committed pack is ``arachne/assets/fixtures``, and the defects are
``tests/assets/defects``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from arachne import Circle, Polygon, Rectangle, RoundedRectangle, VMobject
from arachne.assets.admission import AdmissionReport, admit

#: Every fixture asset's name, in the order drawn.
NAMES = (
    "lidded-box",
    "lidded-crate",
    "dial",
    "gate-valve",
    "figure",
    "layered-section",
)

#: Every planted defect's name, and the one check it fails.
DEFECTS = {
    "open-seam": "hole",
    "loose-part": "detached_part",
    "stray-anchor": "anchor_off_part",
}

#: Canvas units to scene units, for every fixture.
SCALE = 0.01

#: The line width of a fixture's outlines.
LINE = 3

PROVENANCE = {
    "source_url": None,
    "author": "the engine's fixture generator",
    "license": "none",
    "credit_line": None,
    "retrieved_at": None,
}


# Shapes, in canvas coordinates (y down)


def rect(x: float, y: float, w: float, h: float, corner: float = 0) -> VMobject:
    """The rectangle from ``(x, y)``, ``w`` wide and ``h`` tall."""
    if corner:
        mob: VMobject = RoundedRectangle(width=w, height=h, corner_radius=corner)
    else:
        mob = Rectangle(width=w, height=h)
    return mob.move_to([x + w / 2, y + h / 2, 0])


def ellipse(cx: float, cy: float, rx: float, ry: float) -> VMobject:
    """The ellipse about ``(cx, cy)`` with radii ``rx`` and ``ry``."""
    mob = Circle(radius=rx)
    mob.stretch_to_fit_height(2 * ry)
    return mob.move_to([cx, cy, 0])


def polygon(*corners: tuple[float, float]) -> VMobject:
    """The polygon through ``corners``."""
    return Polygon(*[[x, y, 0] for x, y in corners])


def turned_rect(cx: float, cy: float, w: float, h: float, degrees: float) -> VMobject:
    """A ``w`` by ``h`` rectangle about ``(cx, cy)``, turned by ``degrees``."""
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return polygon(
        *[(cx + x * cos - y * sin, cy + x * sin + y * cos) for x, y in corners]
    )


class Shape:
    """A shape of a part file: its outline and its paint."""

    def __init__(
        self, mob: VMobject, fill: str, stroke: str | None = None, width: float = LINE
    ) -> None:
        self.points = np.round(np.asarray(mob.points)[:, :2], 3) + 0.0
        self.fill = fill
        self.stroke = stroke
        self.width = width

    def path(self) -> str:
        """The shape's path data: a move, then a cubic per curve, closed
        where a subpath ends where it began.
        """
        commands: list[str] = []
        start = end = None
        for i in range(0, len(self.points), 4):
            p0, p1, p2, p3 = self.points[i : i + 4]
            if start is None or not np.array_equal(p0, end):
                if start is not None:
                    commands.append(closing(start, end))
                commands.append(f"M {num(p0[0])} {num(p0[1])}")
                start = p0
            commands.append(
                "C " + " ".join(f"{num(x)} {num(y)}" for x, y in (p1, p2, p3))
            )
            end = p3
        if start is not None:
            commands.append(closing(start, end))
        return " ".join(c for c in commands if c)

    def element(self) -> str:
        paint = f'fill="{self.fill}"'
        if self.stroke is not None:
            paint += f' stroke="{self.stroke}" stroke-width="{num(self.width)}"'
        return f'<path d="{self.path()}" {paint}/>'


def closing(start: np.ndarray, end: np.ndarray) -> str:
    """``Z`` for a subpath that ends where it began, else nothing."""
    return "Z" if np.array_equal(start, end) else ""


def num(value: float) -> str:
    """``value`` rounded to three decimals, written shortest."""
    text = f"{round(float(value), 3) + 0.0:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def bounds(shapes: list[Shape]) -> tuple[float, float, float, float]:
    """The box of every point of ``shapes``: ``x0, y0, x1, y1``."""
    points = np.concatenate([shape.points for shape in shapes])
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


# Files


def svg_file(view_box: tuple[int, int], groups: dict[str | None, list[Shape]]) -> str:
    """A part file on a canvas ``view_box`` wide and tall: each group of
    shapes under its ``id``, or at the top for ``None``.
    """
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg"'
        f' viewBox="0 0 {view_box[0]} {view_box[1]}">'
    ]
    for name, shapes in groups.items():
        if name is None:
            lines += [f"  {shape.element()}" for shape in shapes]
        else:
            lines.append(f'  <g id="{name}">')
            lines += [f"    {shape.element()}" for shape in shapes]
            lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def json_text(value: Any, indent: int = 0, used: int = 0) -> str:
    """``value`` as JSON, ``indent`` deep with ``used`` columns of its line
    taken before it: on one line where it fits in 88 columns, else one item
    per line.
    """
    flat = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, (dict, list)) or not value or used + len(flat) <= 88:
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, list):
        items = [pad + json_text(item, indent + 2, indent + 2) for item in value]
    else:
        items = [
            f"{pad}{json.dumps(key)}: "
            + json_text(item, indent + 2, indent + 4 + len(json.dumps(key)))
            for key, item in value.items()
        ]
    brackets = "[]" if isinstance(value, list) else "{}"
    return brackets[0] + "\n" + ",\n".join(items) + "\n" + " " * indent + brackets[1]


def clean(value: Any) -> Any:
    """``value`` with its numbers rounded to three decimals, whole ones as
    integers.
    """
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        rounded = round(float(value), 3) + 0.0
        return int(rounded) if rounded.is_integer() else rounded
    return value


class Pack:
    """Writes one asset's files into its directory."""

    def __init__(self, root: Path, name: str, canvas: tuple[int, int]) -> None:
        self.directory = root / name
        self.name = name
        self.canvas = canvas
        self.shown: list[Shape] = []

    def file(
        self, path: str, groups: dict[str | None, list[Shape]], shown: bool = True
    ) -> str:
        """Write the part file ``path``; its shapes count in the asset's
        natural size when ``shown`` (drawn at the default pose).
        """
        target = self.directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(svg_file(self.canvas, groups))
        if shown:
            self.shown += [shape for shapes in groups.values() for shape in shapes]
        return path

    def manifest(self, origin: tuple[float, float], **members: Any) -> None:
        """Write the manifest, its size measured from the shapes shown."""
        x0, y0, x1, y1 = bounds(self.shown)
        document = {
            "manifest_version": "1.0",
            "name": self.name,
            "version": 1,
            "kind": "parts",
            "requires": [],
            "size": {
                "scene_units_per_unit": SCALE,
                "origin": list(origin),
                "width": (x1 - x0) * SCALE,
                "height": (y1 - y0) * SCALE,
            },
            **members,
            "provenance": PROVENANCE,
        }
        text = json_text(clean(document))
        (self.directory / "manifest.json").write_text(text + "\n")


def part(
    name: str,
    file: str,
    parent: str | None,
    pivot: tuple[float, float],
    order: int,
    element: str | None = None,
    lattice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "file": file}
    if element is not None:
        entry["element"] = element
    entry |= {"parent": parent, "pivot": list(pivot), "order": order}
    if lattice is not None:
        entry["lattice"] = lattice
    return entry


def slot(
    name: str,
    parent: str,
    pivot: tuple[float, float],
    order: int,
    default: str,
    drawings: dict[str, str],
) -> dict[str, Any]:
    return {
        "name": name,
        "parent": parent,
        "pivot": list(pivot),
        "order": order,
        "default": default,
        "drawings": [{"name": n, "file": f} for n, f in drawings.items()],
    }


def keys(*pairs: tuple[float, Any], member: str = "value") -> list[dict[str, Any]]:
    return [{"at": at, member: value} for at, value in pairs]


# The assets


def lidded(root: Path, name: str, crate: bool) -> None:
    """A body and a lid hinged at the body's top left corner."""
    pack = Pack(root, name, (400, 300))
    if crate:
        body = [
            Shape(rect(90, 130, 220, 130), "#9c6b3f", "#3e2a1a"),
            Shape(rect(90, 170, 220, 6), "#7a5230"),
            Shape(rect(90, 215, 220, 6), "#7a5230"),
        ]
        lid = [
            Shape(rect(86, 110, 228, 20), "#8a5a32", "#3e2a1a"),
            Shape(rect(185, 100, 30, 10), "#3e2a1a"),
        ]
        hinge, knob, origin = (90, 130), (200, 100), (200, 180)
    else:
        body = [
            Shape(rect(100, 140, 200, 120), "#c8a165", "#4a3426"),
            Shape(rect(100, 180, 200, 14), "#a8824a"),
        ]
        lid = [
            Shape(rect(96, 118, 208, 22, corner=4), "#b48a50", "#4a3426"),
            Shape(rect(190, 108, 20, 10), "#4a3426"),
        ]
        hinge, knob, origin = (100, 140), (200, 108), (200, 184)
    file = pack.file(f"{name}.svg", {"body": body, "lid": lid})
    pack.manifest(
        origin,
        parts=[
            part("body", file, None, (200, 260), 0, element="body"),
            part("lid", file, "body", hinge, 1, element="lid"),
        ],
        slots=[],
        parameters=[
            {
                "name": "lid_open",
                "min": 0,
                "max": 110,
                "default": 0,
                "unit": "deg",
                "drives": [
                    {
                        "part": "lid",
                        "kind": "rotate",
                        "unit": "deg",
                        "keys": keys((0, 0), (110, 110)),
                    }
                ],
            }
        ],
        anchors=[
            {"name": "knob", "part": "lid", "attach": "outline", "at": list(knob)}
        ],
    )


def dial(root: Path) -> None:
    """A face with ticks, a needle turning about its hub, and a lamp."""
    pack = Pack(root, "dial", (300, 300))
    ticks = [
        Shape(
            turned_rect(
                150 + 88 * math.sin(math.radians(a)),
                160 - 88 * math.cos(math.radians(a)),
                4,
                14,
                a,
            ),
            "#333333",
        )
        for a in range(-120, 121, 30)
    ]
    face = pack.file(
        "pieces/face.svg",
        {
            None: [
                Shape(ellipse(150, 160, 100, 100), "#f4f1e8", "#333333", 4),
                *ticks,
                Shape(ellipse(150, 160, 8, 8), "#333333"),
            ]
        },
    )
    needle = pack.file(
        "pieces/needle.svg",
        {
            None: [
                Shape(polygon((150, 78), (156, 160), (150, 172), (144, 160)), "#c0392b")
            ]
        },
    )
    off = pack.file(
        "slots/lamp-off.svg",
        {None: [Shape(ellipse(150, 215, 10, 10), "#6b6b6b", "#333333", 2)]},
    )
    on = pack.file(
        "slots/lamp-on.svg",
        {None: [Shape(ellipse(150, 215, 10, 10), "#f5c518", "#333333", 2)]},
        shown=False,
    )
    pack.manifest(
        (150, 160),
        parts=[
            part("face", face, None, (150, 160), 0),
            part("needle", needle, "face", (150, 160), 2),
        ],
        slots=[slot("lamp", "face", (150, 215), 1, "off", {"off": off, "on": on})],
        parameters=[
            {
                "name": "reading",
                "min": 0,
                "max": 100,
                "default": 50,
                "unit": "1",
                "drives": [
                    {
                        "part": "needle",
                        "kind": "rotate",
                        "unit": "deg",
                        "keys": keys((0, 120), (100, -120)),
                    }
                ],
            }
        ],
        anchors=[
            {
                "name": "needle_tip",
                "part": "needle",
                "attach": "outline",
                "at": [150, 78],
            }
        ],
    )


def gate_valve(root: Path) -> None:
    """A pipe with a housing, a stem and gate that slide up, and a flag."""
    pack = Pack(root, "gate-valve", (400, 300))
    file = pack.file(
        "gate-valve.svg",
        {
            "body": [
                Shape(rect(40, 170, 320, 50), "#8fa3b8", "#2c3e50"),
                Shape(rect(160, 110, 80, 120, corner=8), "#7f93a8", "#2c3e50"),
                Shape(rect(185, 90, 30, 20), "#7f93a8", "#2c3e50"),
            ],
            "stem": [
                Shape(rect(196, 40, 8, 135), "#5d6d7e"),
                Shape(rect(172, 168, 56, 54), "#34495e", "#2c3e50"),
                Shape(ellipse(200, 40, 36, 8), "#c0392b", "#2c3e50"),
            ],
        },
    )
    closed = pack.file(
        "slots/state-closed.svg", {None: [Shape(rect(300, 80, 40, 30), "#c0392b")]}
    )
    opened = pack.file(
        "slots/state-open.svg",
        {None: [Shape(rect(300, 80, 40, 30), "#27ae60")]},
        shown=False,
    )
    pack.manifest(
        (200, 150),
        parts=[
            part("body", file, None, (200, 195), 0, element="body"),
            part("stem", file, "body", (200, 168), 1, element="stem"),
        ],
        slots=[
            slot(
                "state",
                "body",
                (300, 110),
                2,
                "closed",
                {"closed": closed, "open": opened},
            )
        ],
        parameters=[
            {
                "name": "opening",
                "min": 0,
                "max": 1,
                "default": 0,
                "unit": "1",
                "drives": [
                    {
                        "part": "stem",
                        "kind": "translate",
                        "keys": keys((0, [0, 0]), (1, [0, -60])),
                    }
                ],
            }
        ],
        anchors=[
            {"name": "wheel", "part": "stem", "attach": "outline", "at": [236, 40]},
            {"name": "flow", "part": "body", "attach": "interior", "at": [80, 195]},
        ],
    )


def figure(root: Path) -> None:
    """A standing figure whose arm reaches out and whose head turns."""
    pack = Pack(root, "figure", (300, 400))
    skin, cloth, dark = "#f2c9a0", "#4f7cac", "#2f4b6e"
    file = pack.file(
        "figure.svg",
        {
            "torso": [
                Shape(rect(118, 290, 26, 90, corner=6), dark),
                Shape(rect(156, 290, 26, 90, corner=6), dark),
                Shape(rect(110, 150, 80, 150, corner=20), cloth, dark, 2),
            ],
            "upper_arm": [Shape(rect(186, 160, 18, 72, corner=9), cloth, dark, 2)],
            "lower_arm": [Shape(rect(187, 226, 16, 62, corner=8), skin)],
        },
    )
    head = pack.file(
        "pieces/head.svg",
        {None: [Shape(ellipse(150, 110, 32, 38), skin, "#8a6a50", 2)]},
    )
    eyes_open = pack.file(
        "slots/eyes-open.svg",
        {
            None: [
                Shape(ellipse(138, 105, 4, 5), "#222222"),
                Shape(ellipse(162, 105, 4, 5), "#222222"),
            ]
        },
    )
    eyes_closed = pack.file(
        "slots/eyes-closed.svg",
        {
            None: [
                Shape(rect(133, 104, 10, 2), "#222222"),
                Shape(rect(157, 104, 10, 2), "#222222"),
            ]
        },
        shown=False,
    )
    hand_open = pack.file(
        "slots/hand-open.svg", {None: [Shape(ellipse(195, 296, 10, 10), skin)]}
    )
    hand_point = pack.file(
        "slots/hand-point.svg",
        {
            None: [
                Shape(ellipse(195, 294, 9, 9), skin),
                Shape(rect(193, 298, 4, 18, corner=2), skin),
            ]
        },
        shown=False,
    )
    turn = [4, 8, 4] * 3
    lattice = {"cols": 2, "rows": 2, "bounds": [114, 68, 186, 152]}
    # The eyes, a slot under the warped head, slide as far as the warp
    # carries the points they sit at: a warp moves its own part only.
    eyes = [(138, 105), (162, 105)]
    follow = np.mean([warp_at(lattice, turn, eye) for eye in eyes])
    pack.manifest(
        (150, 225),
        parts=[
            part("torso", file, None, (150, 300), 0, element="torso"),
            part(
                "head",
                head,
                "torso",
                (150, 148),
                1,
                lattice=lattice,
            ),
            part("upper_arm", file, "torso", (195, 168), 3, element="upper_arm"),
            part("lower_arm", file, "upper_arm", (195, 230), 4, element="lower_arm"),
        ],
        slots=[
            slot(
                "eyes",
                "head",
                (150, 105),
                2,
                "open",
                {"open": eyes_open, "closed": eyes_closed},
            ),
            slot(
                "hand",
                "lower_arm",
                (195, 290),
                5,
                "open",
                {"open": hand_open, "point": hand_point},
            ),
        ],
        parameters=[
            {
                "name": "reach",
                "min": 0,
                "max": 1,
                "default": 0,
                "unit": "1",
                "drives": [
                    {
                        "part": "upper_arm",
                        "kind": "rotate",
                        "unit": "deg",
                        "keys": keys((0, 0), (1, 70)),
                    },
                    {
                        "part": "hand",
                        "kind": "translate",
                        "keys": keys((0, [0, 0]), (1, [12, -4])),
                    },
                ],
            },
            {
                "name": "head_turn",
                "min": -25,
                "max": 25,
                "default": 0,
                "unit": "deg",
                "drives": [
                    {
                        "part": "head",
                        "kind": "warp",
                        "keys": keys(
                            (-25, [[-dx, 0] for dx in turn]),
                            (0, [[0, 0]] * 9),
                            (25, [[dx, 0] for dx in turn]),
                            member="offsets",
                        ),
                    },
                    {
                        "part": "eyes",
                        "kind": "translate",
                        "keys": keys(
                            (-25, [-follow, 0]), (0, [0, 0]), (25, [follow, 0])
                        ),
                    },
                ],
            },
        ],
        anchors=[
            {
                "name": "hand_tip",
                "part": "hand",
                "attach": "interior",
                "at": [195, 302],
            },
            {"name": "head_top", "part": "head", "attach": "outline", "at": [150, 72]},
            {"name": "chest", "part": "torso", "attach": "interior", "at": [150, 190]},
            {
                "name": "elbow",
                "part": "lower_arm",
                "attach": "interior",
                "at": [195, 232],
            },
        ],
    )


def warp_at(
    lattice: dict[str, Any], offsets: list[float], point: tuple[float, float]
) -> float:
    """The horizontal offset a lattice whose points move by ``offsets``
    (one per point, row by row) gives ``point``, blended bilinearly.
    """
    x0, y0, x1, y1 = lattice["bounds"]
    cols, rows = lattice["cols"], lattice["rows"]
    u = (point[0] - x0) / (x1 - x0) * cols
    v = (point[1] - y0) / (y1 - y0) * rows
    i, j = min(int(u), cols - 1), min(int(v), rows - 1)
    fu, fv = u - i, v - j
    grid = np.reshape(offsets, (rows + 1, cols + 1))
    return float(
        (1 - fu) * (1 - fv) * grid[j, i]
        + fu * (1 - fv) * grid[j, i + 1]
        + (1 - fu) * fv * grid[j + 1, i]
        + fu * fv * grid[j + 1, i + 1]
    )


#: The swelling layer's radii at each key of ``swell``.
SWELL = ((0, 90, 55), (0.5, 105, 60), (1, 120, 66))
#: The section's centre, the swelling layer's rest radii and the granule's
#: rest place in it.
CENTRE, REST, GRANULE = (200, 150), (90, 55), (250, 140)


def granule(rx: float, ry: float) -> list[float]:
    """The granule's place inside the swelling layer of radii ``rx``, ``ry``:
    where it is at rest, carried as the layer stretches.
    """
    (cx, cy), (x, y) = CENTRE, GRANULE
    return [cx + (x - cx) * rx / REST[0], cy + (y - cy) * ry / REST[1]]


def layered_section(root: Path) -> None:
    """Four nested layers; the third swells through stored outlines."""
    pack = Pack(root, "layered-section", (400, 300))
    cx, cy = CENTRE
    layers = [
        ("wall", 160, 110, "#e8d5b7", "#8d6e63"),
        ("outer_layer", 135, 90, "#f6e3c4", "#8d6e63"),
        ("inner_layer", 90, 55, "#d9a3a3", "#8e4b4b"),
        ("core", 40, 28, "#8e5b8e", "#4a2a4a"),
    ]
    files = {
        name: pack.file(
            f"pieces/{name}.svg", {None: [Shape(ellipse(cx, cy, rx, ry), fill, line)]}
        )
        for name, rx, ry, fill, line in layers
    }
    fill, line = layers[2][3:]
    outlines = [
        (at, files["inner_layer"])
        if (rx, ry) == REST
        else (
            at,
            pack.file(
                f"shapes/inner_layer-{i}.svg",
                {None: [Shape(ellipse(cx, cy, rx, ry), fill, line)]},
                shown=False,
            ),
        )
        for i, (at, rx, ry) in enumerate(SWELL)
    ]
    parents = [None, "wall", "outer_layer", "inner_layer"]
    pack.manifest(
        CENTRE,
        parts=[
            part(name, files[name], parent, CENTRE, i)
            for i, ((name, *_), parent) in enumerate(zip(layers, parents, strict=True))
        ],
        slots=[],
        parameters=[
            {
                "name": "swell",
                "min": 0,
                "max": 1,
                "default": 0,
                "unit": "1",
                "drives": [
                    {
                        "part": "inner_layer",
                        "kind": "shape",
                        "keys": keys(*outlines, member="file"),
                    }
                ],
            }
        ],
        anchors=[
            {
                "name": "tip",
                "part": "inner_layer",
                "attach": "outline",
                "at": [cx + REST[0], cy],
            },
            {
                "name": "granule",
                "part": "inner_layer",
                "attach": "interior",
                "at": list(GRANULE),
                "keys": [{"at": at, "point": granule(rx, ry)} for at, rx, ry in SWELL],
            },
        ],
    )


# The planted defects


def one_parameter(
    name: str,
    low: float,
    high: float,
    part: str,
    kind: str,
    out: list[Any],
    **more: Any,
) -> dict[str, Any]:
    """A parameter from ``low`` (its default) to ``high`` that drives
    ``part`` by ``kind`` from the output ``out[0]`` to ``out[1]``.
    """
    driver = {"part": part, "kind": kind, **more}
    driver["keys"] = keys((low, out[0]), (high, out[1]))
    return {
        "name": name,
        "min": low,
        "max": high,
        "default": low,
        "unit": "1",
        "drives": [driver],
    }


def open_seam(root: Path) -> None:
    """A frame whose pane slides and opens a seam inside it."""
    pack = Pack(root, "open-seam", (300, 300))
    frame = pack.file(
        "pieces/frame.svg",
        {
            None: [
                Shape(rect(50, 50, 200, 20), "#6d4c41", "#3e2723"),
                Shape(rect(50, 230, 200, 20), "#6d4c41", "#3e2723"),
                Shape(rect(50, 70, 20, 160), "#6d4c41", "#3e2723"),
                Shape(rect(230, 70, 20, 160), "#6d4c41", "#3e2723"),
            ]
        },
    )
    pane = pack.file(
        "pieces/pane.svg", {None: [Shape(rect(66, 66, 168, 168), "#b3e5fc")]}
    )
    pack.manifest(
        (150, 150),
        parts=[
            part("frame", frame, None, (150, 150), 1),
            part("pane", pane, "frame", (150, 150), 0),
        ],
        slots=[],
        parameters=[
            one_parameter("slide", 0, 1, "pane", "translate", [[0, 0], [10, 0]])
        ],
        anchors=[
            {"name": "latch", "part": "frame", "attach": "interior", "at": [240, 150]}
        ],
    )


def loose_part(root: Path) -> None:
    """A body whose handle is pulled clear of it."""
    pack = Pack(root, "loose-part", (300, 300))
    file = pack.file(
        "loose-part.svg",
        {
            "body": [Shape(rect(60, 120, 180, 120), "#90a4ae", "#37474f")],
            "handle": [Shape(rect(120, 90, 60, 36, corner=8), "#ffb300", "#37474f")],
        },
    )
    pack.manifest(
        (150, 165),
        parts=[
            part("body", file, None, (150, 180), 0, element="body"),
            part("handle", file, "body", (150, 108), 1, element="handle"),
        ],
        slots=[],
        parameters=[
            one_parameter("pull", 0, 1, "handle", "translate", [[0, 0], [0, -60]])
        ],
        anchors=[
            {"name": "grip", "part": "handle", "attach": "interior", "at": [150, 100]},
            {"name": "base", "part": "body", "attach": "interior", "at": [150, 200]},
        ],
    )


def stray_anchor(root: Path) -> None:
    """A box whose lid's anchor is named on the body."""
    pack = Pack(root, "stray-anchor", (400, 300))
    file = pack.file(
        "stray-anchor.svg",
        {
            "body": [Shape(rect(80, 140, 240, 120), "#a1887f", "#4e342e")],
            "lid": [
                Shape(rect(76, 118, 248, 22), "#8d6e63", "#4e342e"),
                Shape(rect(190, 108, 20, 10), "#4e342e"),
            ],
        },
    )
    pack.manifest(
        (200, 184),
        parts=[
            part("body", file, None, (200, 260), 0, element="body"),
            part("lid", file, "body", (80, 140), 1, element="lid"),
        ],
        slots=[],
        parameters=[
            one_parameter("lid_open", 0, 90, "lid", "rotate", [0, 90], unit="deg")
        ],
        anchors=[
            {"name": "knob", "part": "body", "attach": "interior", "at": [200, 112]}
        ],
    )


# Admission


def record_checks(directory: Path) -> AdmissionReport:
    """Admit the asset in ``directory`` and write what admission found into
    its manifest, before its provenance.
    """
    report = admit(directory)
    path = directory / "manifest.json"
    document = json.loads(path.read_text())
    provenance = document.pop("provenance")
    document |= {"checks": report.checks(), "provenance": provenance}
    path.write_text(json_text(document) + "\n")
    return report


def draw(root: Path) -> None:
    """Draw every fixture asset into ``root``, one directory each, and
    admit it.
    """
    lidded(root, "lidded-box", crate=False)
    lidded(root, "lidded-crate", crate=True)
    dial(root)
    gate_valve(root)
    figure(root)
    layered_section(root)
    for name in NAMES:
        report = record_checks(root / name)
        if not report.passed:
            raise AssertionError("\n".join(report.lines()))


def draw_defects(root: Path) -> None:
    """Draw every planted defect into ``root``, one directory each, and
    record what admission finds of it.
    """
    open_seam(root)
    loose_part(root)
    stray_anchor(root)
    for name in DEFECTS:
        record_checks(root / name)


if __name__ == "__main__":
    draw(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        draw_defects(Path(sys.argv[2]))
