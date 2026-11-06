"""The things a sweep measures at an instant: each :class:`~arachne.lint.parts.Unit`
with its drawn leaves, read as a region (a text, or a shape filled in every
leaf) held to boxes, or a path held to the points along its strokes. A leaf
whose fill covers less than one reference pixel's area
(:data:`~arachne.lint.codes.REFERENCE_RESOLUTION`) draws no fill: a straight
shaft its opacity fills is a stroke.
"""

from __future__ import annotations

__all__ = ["Measured", "fill_area", "measured_units"]

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

from ..codes import REFERENCE_RESOLUTION
from ..findings import FindingObject
from ..parts import Box, Leaf, Parts, Unit, target_pixels
from ._geometry import anchor_box, meets, stroke_samples, union

#: A filled shape no wider and no taller than this marks a point.
MARKER_SIZE = 0.36
#: A stroke whose ends lie this close together is a ring.
RING_GAP = 1e-4


@dataclass(eq=False)
class Measured:
    """A unit at an instant: its leaves in paint order, the top-level
    mobject holding it (``root``), what it says when it is a text, and its
    object as a finding names it.
    """

    unit: Unit
    leaves: tuple[Leaf, ...]
    root: str
    said: str | None
    object: FindingObject
    #: Where the last copy of a text drawn more than once lays its glyphs:
    #: the frame shows that copy (0 when none is drawn after the first).
    copy_top: int = 0
    #: One reference pixel's area, in home-frame units squared: a fill that
    #: covers less draws none.
    pixel_area: float = 0.0

    @property
    def id(self) -> str:
        return self.unit.id

    @property
    def text(self) -> bool:
        return self.unit.text

    @property
    def furniture(self) -> bool:
        return self.unit.furniture is not None

    @property
    def grid(self) -> bool:
        """Part of a grid: a plane, a vector field, a lattice of lines."""
        return self.unit.grid

    @property
    def structural(self) -> bool:
        """An axis's line or ticks: what content rides by construction."""
        return self.unit.furniture in ("line", "ticks")

    @cached_property
    def region(self) -> bool:
        """Held to boxes: a text, or a shape filled in every leaf (by its
        fill's first row, over at least one reference pixel's area).
        """
        return self.text or all(self._filled(leaf) for leaf in self.leaves)

    def _filled(self, leaf: Leaf) -> bool:
        if leaf.first_fill_opacity <= 0:
            return False
        return leaf.draw != "path" or fill_area(leaf) >= self.pixel_area

    @cached_property
    def outer(self) -> Box:
        """The box of its anchors."""
        box = anchor_box(self.leaves)
        assert box is not None
        return box

    @cached_property
    def glyphs(self) -> tuple[Leaf, ...]:
        """A text's leaves less its own background rectangle."""
        return tuple(leaf for leaf in self.leaves if not leaf.plate)

    @cached_property
    def glyph_box(self) -> Box:
        """The box of a text's glyphs, without its plate."""
        return anchor_box(self.glyphs) or self.outer

    @cached_property
    def boxes(self) -> list[Box]:
        """The boxes a region is held to against anything but a text: each
        leaf's (a text's glyphs, a filled shape's pieces), or its own when
        it has one leaf.
        """
        if self.text or len(self.leaves) > 1:
            return [leaf.screen_box for leaf in self.leaves]
        return [self.outer]

    @cached_property
    def marker(self) -> bool:
        """A filled shape small enough to mark a point, not axis furniture."""
        if self.text or not self.region or self.furniture:
            return False
        x0, y0, x1, y1 = union([leaf.screen_box for leaf in self.leaves])
        return x1 - x0 <= MARKER_SIZE and y1 - y0 <= MARKER_SIZE

    @cached_property
    def points(self) -> np.ndarray:
        """Points along its strokes, ``(n, 2)``."""
        return np.vstack([stroke_samples(leaf) for leaf in self.leaves])

    @cached_property
    def ends(self) -> np.ndarray:
        """Each leaf's first and last point."""
        return np.vstack([leaf.points[[0, -1]] for leaf in self.leaves])

    @cached_property
    def reach(self) -> float:
        """How far its widest stroke lays ink past its points."""
        return max(leaf.line_width for leaf in self.leaves) / 2

    @cached_property
    def ring(self) -> bool:
        """Every leaf ends where it starts."""
        return all(
            float(np.linalg.norm(leaf.points[0] - leaf.points[-1])) < RING_GAP
            for leaf in self.leaves
        )

    @cached_property
    def depth(self) -> tuple[float, float] | None:
        """The least and greatest depth of its leaves toward a ThreeDCamera
        (the greater the nearer), when the camera projects every one of
        them; ``None`` otherwise (:attr:`~arachne.lint.parts.Leaf.depth`).
        """
        spans = [leaf.depth for leaf in self.leaves if leaf.depth is not None]
        if len(spans) < len(self.leaves):
            return None
        return min(low for low, _ in spans), max(high for _, high in spans)

    @cached_property
    def top(self) -> int:
        """Where its last ink is drawn."""
        return max(self.copy_top, *(leaf.order for leaf in self.leaves))

    @cached_property
    def glyph_top(self) -> int:
        """Where a text's last glyph is drawn, its plate aside."""
        return max(self.copy_top, *(leaf.order for leaf in self.glyphs or self.leaves))

    def on(self, frame: Box) -> bool:
        return meets(self.outer, frame)


_cache: WeakKeyDictionary[Parts, list[Measured]] = WeakKeyDictionary()


def measured_units(parts: Parts) -> list[Measured]:
    """The units drawn at ``parts``' instant, in the order their first
    leaves are drawn. A unit whose leaves all sit at one point lays no ink
    a viewer sees, and is left out.
    """
    found = _cache.get(parts)
    if found is None:
        found = _gather(parts)
        _cache[parts] = found
    return found


def _gather(parts: Parts) -> list[Measured]:
    grouped: dict[str, dict[str, Any]] = {}
    for part in parts.parts:
        for leaf in part.leaves:
            if leaf.unit is None:
                continue
            entry = grouped.setdefault(
                leaf.unit.id,
                {"unit": leaf.unit, "leaves": [], "root": part.root, "copy_top": 0},
            )
            entry["leaves"].append(leaf)
            if part.copies > 1:
                # A text drawn again in place: its part's top is its last copy's.
                entry["copy_top"] = max(entry["copy_top"], part.top)
    said = {part.id: part.text for part in parts.parts if part.kind == "text"}
    view = parts.view
    # One reference pixel: the home frame's shorter side over the reference
    # picture's, never the session's render size.
    pixel = min(view.home_width, view.home_height) / min(REFERENCE_RESOLUTION)
    units = []
    for entry in grouped.values():
        leaves = tuple(sorted(entry["leaves"], key=lambda leaf: leaf.order))
        if not _extended(leaves):
            continue
        unit = entry["unit"]
        units.append(
            Measured(
                unit,
                leaves,
                entry["root"],
                said.get(unit.id) if unit.text else None,
                _object(parts, unit, leaves, said.get(unit.id)),
                entry["copy_top"],
                pixel * pixel,
            )
        )
    units.sort(key=lambda u: u.leaves[0].order)
    return units


def fill_area(leaf: Leaf) -> float:
    """The area a path's fill covers, in home-frame units squared: each of
    its subpaths closed and measured by the shoelace formula along its
    curves; 0 for a straight one.
    """
    curves = leaf.outline()
    if curves is None:
        return 0.0
    total = 0.0
    start = 0
    for index in range(1, len(curves) + 1):
        if (
            index == len(curves)
            or float(np.linalg.norm(curves[index - 1][-1] - curves[index][0])) > 1e-9
        ):
            ring = curves[start:index].reshape(-1, 2)
            x, y = ring[:, 0], ring[:, 1]
            total += (
                abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2
            )
            start = index
    return total


def _extended(leaves: Iterable[Leaf]) -> bool:
    return any(
        max(
            leaf.screen_box[2] - leaf.screen_box[0],
            leaf.screen_box[3] - leaf.screen_box[1],
        )
        > 0
        for leaf in leaves
    )


def _object(
    parts: Parts, unit: Unit, leaves: tuple[Leaf, ...], said: str | None
) -> FindingObject:
    return FindingObject(
        id=unit.id,
        source_line=unit.scene_line,
        kind=unit.type,
        label_or_text=said if unit.text and said is not None else unit.name,
        bbox=union([leaf.scene_box for leaf in leaves]),
        screen_bbox=target_pixels(parts, union([leaf.screen_box for leaf in leaves])),
        name=unit.name,
        within=unit.within,
    )
