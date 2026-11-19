"""Region marks: outlines that hold something, containers, stacks and
grids.
"""

from __future__ import annotations

__all__ = ["Capsule", "CellGrid", "DoubleBoundary", "Drum", "LayerStack", "Vessel"]

from collections.abc import Sequence
from typing import Any

import numpy as np

from arachne.constants import DL, ORIGIN, TAU
from arachne.mobject.geometry.polygram import Rectangle, Square
from arachne.mobject.text.text_mobject import Text
from arachne.mobject.types.vectorized_mobject import VGroup, VMobject
from arachne.utils.color import BLUE, WHITE, ManimColor, ParsableManimColor

from ._paths import arc, ellipse_arc, line, rounded_rect


def _fit_text(text: Text, width: float, height: float) -> Text:
    """``text`` scaled down, never up, to fit ``width`` by ``height``."""
    scale = min(1.0, width / text.width, height / text.height)
    if scale < 1.0:
        text.scale(scale)
    return text


class Capsule(VMobject):
    """A stadium: a ``width`` by ``height`` outline whose two short ends are
    half circles. A rod-shaped cell, a flow chart's terminal, a state.
    """

    def __init__(
        self,
        width: float = 2.0,
        height: float = 0.8,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("a capsule has a positive width and height")
        super().__init__(color=color, **kwargs)
        self.set_points(rounded_rect(ORIGIN, width, height, min(width, height) / 2))


class DoubleBoundary(VMobject):
    """A closed double line round a region: a ``width`` by ``height``
    outline with corners of ``corner_radius`` (a capsule's by default) and
    a second one ``gap`` inside it, every point of it the same distance
    in. A membrane, a system's wall, a trust boundary.
    """

    def __init__(
        self,
        width: float = 4.0,
        height: float = 2.5,
        corner_radius: float | None = None,
        gap: float = 0.12,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if not 0 < 2 * gap < min(width, height):
            raise ValueError("a double boundary's gap is positive and fits inside it")
        super().__init__(color=color, **kwargs)
        radius = min(width, height) / 2 if corner_radius is None else corner_radius
        self.gap = float(gap)
        outer = rounded_rect(ORIGIN, width, height, radius)
        inner = rounded_rect(
            ORIGIN, width - 2 * gap, height - 2 * gap, max(0.0, radius - gap)
        )
        self.set_points(np.concatenate([outer, inner]))


class Vessel(VGroup):
    """An open-topped container ``width`` by ``height`` with fluid standing
    ``level`` scene units deep in it: a solution, a column of fluid, a
    buffer filling. ``walls`` is the container, ``fluid`` the fluid drawn
    behind it; :meth:`set_level` changes the depth.
    """

    def __init__(
        self,
        width: float = 1.6,
        height: float = 2.0,
        level: float = 1.0,
        corner_radius: float = 0.2,
        color: ParsableManimColor = WHITE,
        fluid_color: ParsableManimColor = BLUE,
        fluid_opacity: float = 0.6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vessel_width = float(width)
        self.vessel_height = float(height)
        self.corner_radius = max(0.0, min(corner_radius, width / 2, height / 2))
        r = self.corner_radius
        w, h = width / 2, height / 2
        walls = VMobject(color=color)
        corner_y = -h + r
        walls.set_points(
            np.concatenate(
                [
                    line(np.array([-w, h, 0.0]), np.array([-w, corner_y, 0.0])),
                    arc(np.array([-w + r, corner_y, 0.0]), r, TAU / 2, TAU / 4),
                    line(np.array([-w + r, -h, 0.0]), np.array([w - r, -h, 0.0])),
                    arc(np.array([w - r, corner_y, 0.0]), r, 3 * TAU / 4, TAU / 4),
                    line(np.array([w, corner_y, 0.0]), np.array([w, h, 0.0])),
                ]
            )
        )
        self.fluid = VMobject(
            color=ManimColor(fluid_color), fill_opacity=fluid_opacity, stroke_width=0
        )
        self.walls = walls
        self.add(self.fluid, self.walls)
        self.level = 0.0
        self.set_level(level)

    def set_level(self, level: float) -> Vessel:
        """Fill the vessel to ``level`` scene units above its inside bottom
        (between 0 and its height), where it stands now.
        """
        if not 0 <= level <= self.vessel_height:
            raise ValueError(
                f"a vessel {self.vessel_height} tall holds fluid 0 to"
                f" {self.vessel_height} deep, not {level}"
            )
        self.level = float(level)
        # Where the walls stand now: their box, whatever moved them.
        left, bottom = self.walls.get_corner(DL)[:2]
        right = self.walls.get_right()[0]
        scale = (right - left) / self.vessel_width
        if self.level == 0:
            self.fluid.clear_points()
            return self
        depth = self.level * scale
        r = min(self.corner_radius * scale, depth)
        top = bottom + depth
        corner_y = bottom + r
        self.fluid.set_points(
            np.concatenate(
                [
                    line(np.array([left, top, 0.0]), np.array([left, corner_y, 0.0])),
                    arc(np.array([left + r, corner_y, 0.0]), r, TAU / 2, TAU / 4)
                    if r > 0
                    else np.empty((0, 3)),
                    line(
                        np.array([left + r, bottom, 0.0]),
                        np.array([right - r, bottom, 0.0]),
                    ),
                    arc(np.array([right - r, corner_y, 0.0]), r, 3 * TAU / 4, TAU / 4)
                    if r > 0
                    else np.empty((0, 3)),
                    line(np.array([right, corner_y, 0.0]), np.array([right, top, 0.0])),
                    line(np.array([right, top, 0.0]), np.array([left, top, 0.0])),
                ]
            )
        )
        return self


class Drum(VGroup):
    """A flat-drawn cylinder ``width`` across and ``height`` tall, its ends
    ellipses ``depth`` deep: a gas cylinder, a data store, a barrel.
    ``body`` is its side and front, ``top`` its top face.
    """

    def __init__(
        self,
        width: float = 1.2,
        height: float = 1.6,
        depth: float = 0.3,
        color: ParsableManimColor = WHITE,
        fill_opacity: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        w, h, ry = width / 2, height / 2, depth / 2
        top_centre = np.array([0.0, h - ry, 0.0])
        bottom_centre = np.array([0.0, -h + ry, 0.0])
        self.body = VMobject(color=color, fill_opacity=fill_opacity)
        self.body.set_points(
            np.concatenate(
                [
                    line(top_centre + [-w, 0, 0], bottom_centre + [-w, 0, 0]),
                    ellipse_arc(bottom_centre, w, ry, TAU / 2, TAU / 2),
                    line(bottom_centre + [w, 0, 0], top_centre + [w, 0, 0]),
                    ellipse_arc(top_centre, w, ry, 0.0, -TAU / 2),
                ]
            )
        )
        self.top = VMobject(color=color, fill_opacity=fill_opacity)
        self.top.set_points(ellipse_arc(top_centre, w, ry, 0.0, TAU))
        self.add(self.body, self.top)


class LayerStack(VGroup):
    """Labelled slabs one on another, the first of ``labels`` on top: the
    layers of an atmosphere or a film, of a tissue, of a model.
    ``thickness`` is one slab's, or one per slab; ``colors`` one per slab.
    """

    def __init__(
        self,
        labels: Sequence[str] = ("top", "middle", "bottom"),
        width: float = 3.0,
        thickness: float | Sequence[float] = 0.5,
        colors: Sequence[ParsableManimColor] | None = None,
        font_size: float = 24,
        fill_opacity: float = 0.4,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if not labels:
            raise ValueError("a layer stack has at least one layer")
        super().__init__(**kwargs)
        count = len(labels)
        heights = (
            [float(thickness)] * count
            if isinstance(thickness, (int, float))
            else [float(t) for t in thickness]
        )
        if len(heights) != count:
            raise ValueError("a layer stack has one thickness, or one per layer")
        palette = list(colors) if colors is not None else [color] * count
        if len(palette) != count:
            raise ValueError("a layer stack has one colour per layer")
        self.slabs = VGroup()
        self.labels = VGroup()
        top = sum(heights) / 2
        for name, tall, shade in zip(labels, heights, palette, strict=True):
            slab = Rectangle(
                width=width,
                height=tall,
                color=color,
                fill_color=ManimColor(shade),
                fill_opacity=fill_opacity,
            )
            slab.move_to(np.array([0.0, top - tall / 2, 0.0]))
            text = _fit_text(
                Text(name, font_size=font_size, color=color), width * 0.9, tall * 0.7
            )
            text.move_to(slab.get_center())
            self.slabs.add(slab)
            self.labels.add(text)
            top -= tall
        self.add(self.slabs, self.labels)

    def layer(self, index: int) -> Rectangle:
        """The slab ``index`` from the top."""
        slab = self.slabs[index]
        assert isinstance(slab, Rectangle)
        return slab


class CellGrid(VGroup):
    """``rows`` by ``cols`` square cells ``cell_size`` across, addressed by
    row (from the top) and column (from the left): memory or an array, a
    crystal's lattice seen flat, a sheet of cells. ``values``, row by row,
    writes a text in each cell it names (``None`` leaves one empty).
    """

    def __init__(
        self,
        rows: int = 3,
        cols: int = 4,
        cell_size: float = 0.6,
        values: Sequence[Sequence[str | None]] | None = None,
        font_size: float = 24,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("a cell grid has at least one row and one column")
        super().__init__(**kwargs)
        self.rows, self.cols = int(rows), int(cols)
        self.cells = VGroup()
        self.entries = VGroup()
        entry_at: dict[tuple[int, int], int] = {}
        left = -cols * cell_size / 2 + cell_size / 2
        top = rows * cell_size / 2 - cell_size / 2
        for row in range(rows):
            for col in range(cols):
                square = Square(side_length=cell_size, color=color)
                square.move_to(
                    np.array([left + col * cell_size, top - row * cell_size, 0])
                )
                self.cells.add(square)
        if values is not None:
            if len(values) > rows or any(len(r) > cols for r in values):
                raise ValueError("a cell grid's values fit its rows and columns")
            for row, entries in enumerate(values):
                for col, value in enumerate(entries):
                    if value is None:
                        continue
                    text = _fit_text(
                        Text(str(value), font_size=font_size, color=color),
                        cell_size * 0.8,
                        cell_size * 0.6,
                    )
                    text.move_to(self.cell(row, col).get_center())
                    entry_at[(row, col)] = len(self.entries)
                    self.entries.add(text)
        #: Where each written cell's text is among ``entries``.
        self._entry_at = entry_at
        self.add(self.cells, self.entries)

    def cell(self, row: int, col: int) -> Square:
        """The cell in ``row`` (from the top) and ``col`` (from the left)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(
                f"no cell ({row}, {col}) in a {self.rows}x{self.cols} grid"
            )
        square = self.cells[row * self.cols + col]
        assert isinstance(square, Square)
        return square

    def entry(self, row: int, col: int) -> Text | None:
        """The text written in the cell at ``row``, ``col``, if any."""
        self.cell(row, col)
        index = self._entry_at.get((row, col))
        if index is None:
            return None
        text = self.entries[index]
        assert isinstance(text, Text)
        return text
