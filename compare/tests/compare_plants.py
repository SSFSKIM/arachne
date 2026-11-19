"""Scenes built in code for the comparator's tests: a tiny rasterizer and the planted defects.

Two "renderers" differ only in where they sample a pixel for coverage: the first takes an
8 x 8 grid, the second four of those points, spread as a rotated grid. Colour (a texture
included) is computed once per pixel, at its centre, as a GPU shades once per pixel under
multisampling, so the two agree inside every shape and differ only where an edge passes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction

import numpy as np
from arachne_compare import ArrayFrames, ChangedSet, Footprint, HandedMaps, ObjectMap

SIZE = (270, 480)
GRID = tuple(((i + 0.5) / 8, (j + 0.5) / 8) for j in range(8) for i in range(8))
ROOKS = tuple(
    ((i + 0.5) / 8, (j + 0.5) / 8) for i, j in ((1, 5), (5, 7), (3, 1), (7, 3))
)
PATTERNS = {"fine": GRID, "sparse": ROOKS}
BACKGROUND = (24, 28, 36)

Point = tuple[float, float]
Texture = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Shape:
    """One drawable: convex polygons in one colour, or filled with a texture."""

    name: str
    polygons: tuple[tuple[Point, ...], ...]
    colour: tuple[int, int, int] = (200, 200, 200)
    texture: Texture | None = field(default=None, compare=False)

    def moved(self, dx: float, dy: float = 0.0) -> Shape:
        return replace(
            self,
            polygons=tuple(
                tuple((x + dx, y + dy) for x, y in p) for p in self.polygons
            ),
        )


def rect(
    name: str, x0: float, y0: float, x1: float, y1: float, colour=(230, 230, 230)
) -> Shape:
    return Shape(name, (((x0, y0), (x1, y0), (x1, y1), (x0, y1)),), colour)


def _coverage(polygon: Sequence[Point], box, pattern) -> np.ndarray:
    x0, y0, x1, y1 = box
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    area = sum(
        polygon[k][0] * polygon[(k + 1) % len(polygon)][1]
        - polygon[(k + 1) % len(polygon)][0] * polygon[k][1]
        for k in range(len(polygon))
    )
    sign = 1.0 if area > 0 else -1.0
    covered = np.zeros(xs.shape)
    for dx, dy in pattern:
        px, py = xs + dx, ys + dy
        inside = np.ones(xs.shape, dtype=bool)
        for k in range(len(polygon)):
            (ax, ay), (bx, by) = polygon[k], polygon[(k + 1) % len(polygon)]
            inside &= sign * ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) >= 0
        covered += inside
    return covered / len(pattern)


def _box(polygons, size) -> tuple[int, int, int, int]:
    xs = [x for p in polygons for x, _ in p]
    ys = [y for p in polygons for _, y in p]
    h, w = size
    return (
        max(int(np.floor(min(xs))), 0),
        max(int(np.floor(min(ys))), 0),
        min(int(np.ceil(max(xs))), w),
        min(int(np.ceil(max(ys))), h),
    )


def render(
    shapes: Sequence[Shape], pattern: str = "fine", size=SIZE
) -> tuple[np.ndarray, ObjectMap]:
    """Paint the shapes in order over the background.

    Each one's footprint is where it shows: where taking it out would change the frame.
    """
    h, w = size
    canvas = np.empty((h, w, 3), np.float64)
    canvas[:] = BACKGROUND
    covers = []
    for shape in shapes:
        box = _box(shape.polygons, size)
        x0, y0, x1, y1 = box
        cover = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)))
        for polygon in shape.polygons:
            cover = np.maximum(cover, _coverage(polygon, box, PATTERNS[pattern]))
        if shape.texture is not None:
            ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            colour = shape.texture(xs + 0.5, ys + 0.5)
        else:
            colour = np.broadcast_to(
                np.array(shape.colour, np.float64), cover.shape + (3,)
            )
        region = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = (
            region * (1 - cover[..., None]) + colour * cover[..., None]
        )
        covers.append((box, cover))
    objects = {}
    showing = np.ones((h, w))
    for shape, (box, cover) in reversed(list(zip(shapes, covers, strict=True))):
        x0, y0, x1, y1 = box
        objects[shape.name] = Footprint(box, (cover > 0) & (showing[y0:y1, x0:x1] > 0))
        showing[y0:y1, x0:x1] *= 1 - cover
    frame = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
    return frame, ObjectMap(size, objects, complete=True)


def _unit(*parts: object) -> float:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def stripes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A texture of sharp steps: diagonal bands six pixels wide, dark and bright."""
    band = (np.floor((x + y) / 6) % 2).astype(bool)
    out = np.empty(x.shape + (3,))
    out[band] = (230, 210, 60)
    out[~band] = (40, 60, 150)
    return out


SURFACE = Shape(
    "surface",
    (((300.0, 150.0), (420.0, 165.0), (405.0, 245.0), (290.0, 230.0)),),
    texture=stripes,
)


def busy(count: int = 56, seed: str = "busy") -> list[Shape]:
    """At least 50 filled convex polygons at every angle, and one textured surface."""
    shapes = []
    for k in range(count):
        cx = 20 + _unit(seed, k, "x") * 440
        cy = 20 + _unit(seed, k, "y") * 230
        radius = 9 + _unit(seed, k, "r") * 16
        sides = 3 + int(_unit(seed, k, "n") * 3)
        turn = _unit(seed, k, "a") * 2 * np.pi
        polygon = tuple(
            (
                cx + radius * np.cos(turn + 2 * np.pi * s / sides),
                cy + radius * np.sin(turn + 2 * np.pi * s / sides),
            )
            for s in range(sides)
        )
        colour = tuple(int(40 + _unit(seed, k, c) * 200) for c in "rgb")
        shapes.append(Shape(f"shape-{k:02d}", (polygon,), colour))  # type: ignore[arg-type]
    return [*shapes, SURFACE]


LINE = rect("line", 60, 12, 80, 13)  # 20 pixels: under the cluster threshold
DOT = rect("dot", 100, 12, 102, 14)
GLYPH = Shape(  # a thin "T": one-pixel strokes, 14 pixels
    "glyph",
    (
        ((130, 10), (137, 10), (137, 11), (130, 11)),
        ((133, 11), (134, 11), (134, 18), (133, 18)),
    ),
    (240, 240, 240),
)
SMALL = (LINE, DOT, GLYPH)


def single(frame: np.ndarray) -> ArrayFrames:
    return ArrayFrames([frame], Fraction(30))


def nothing_changed(a: int, b: int) -> ChangedSet:
    return ChangedSet(frozenset())


def handed(
    reference: Sequence[ObjectMap], candidate: Sequence[ObjectMap]
) -> HandedMaps:
    return HandedMaps(dict(enumerate(reference)), dict(enumerate(candidate)))


def renderers(mode: str) -> tuple[str, str]:
    """The reference's and the candidate's renderer in each mode."""
    return ("fine", "fine") if mode == "same" else ("fine", "sparse")
