"""Diagram pieces with ports: a block and a node."""

from __future__ import annotations

__all__ = ["Block", "Node"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.constants import DOWN, LEFT, ORIGIN, RIGHT, UP
from arachne.mobject.geometry.arc import Circle
from arachne.mobject.text.text_mobject import Text
from arachne.mobject.types.vectorized_mobject import VMobject
from arachne.typing import Vector3DLike
from arachne.utils.color import WHITE, ParsableManimColor

from ._paths import as_point, line, rounded_rect
from ._style import apply_style, opacities_of, scaled, set_opacities, take_style
from .ports import Ported
from .shapes import _fit_text

if TYPE_CHECKING:
    pass

#: The ports a block or a node has unless told others.
SIDES = {"left": LEFT, "right": RIGHT, "top": UP, "bottom": DOWN}

#: Where a block's inside diagram may reach, as a share of its body.
INSIDE_SHARE = 0.85


def _side_point(
    width: float, height: float, side: np.ndarray, share: float
) -> tuple[np.ndarray, np.ndarray]:
    """A point ``share`` of the way along one side of a ``width`` by
    ``height`` box about the origin (left to right, bottom to top), and the
    side's outward direction.
    """
    side = as_point(side)
    if abs(side[0]) >= abs(side[1]):
        outward = np.array([np.sign(side[0]), 0.0, 0.0])
        point = np.array([outward[0] * width / 2, -height / 2 + share * height, 0.0])
    else:
        outward = np.array([0.0, np.sign(side[1]), 0.0])
        point = np.array([-width / 2 + share * width, outward[1] * height / 2, 0.0])
    return point, outward


class Block(Ported):
    """A rounded box ``width`` by ``height`` with a header strip holding
    ``title``, and ports: a component of a system, a stage of an apparatus,
    a step of a process.

    ``ports`` maps a name to a side (``LEFT``, ``RIGHT``, ``UP``, ``DOWN``)
    or to a side and a share of the way along it (from its left or bottom
    end); by default the four sides' middles, named ``left``, ``right``,
    ``top`` and ``bottom``. ``inside`` is a diagram of what the block holds,
    fitted into its body and hidden until
    :class:`~arachne.mobject.diagram.ZoomInto` opens it. Style keywords go
    to the body; the header strip, the title and the inside keep theirs.
    """

    def __init__(
        self,
        title: str | None = None,
        width: float = 2.4,
        height: float = 1.4,
        header: float = 0.45,
        corner_radius: float = 0.12,
        ports: Mapping[str, Vector3DLike | tuple[Vector3DLike, float]] | None = None,
        inside: VMobject | None = None,
        font_size: float = 28,
        color: ParsableManimColor = WHITE,
        fill_opacity: float = 0.15,
        **kwargs: Any,
    ) -> None:
        style = take_style(kwargs)
        super().__init__(**kwargs)
        self.body = VMobject(color=color, fill_opacity=fill_opacity)
        self.body.set_points(rounded_rect(ORIGIN, width, height, corner_radius))
        apply_style([self.body], style)
        self.add(self.body)
        self.title: Text | None = None
        self.header = 0.0
        if title is not None:
            # At least a corner's radius, so the strip holds both top corners.
            self.header = max(min(header, height / 2), min(corner_radius, height / 2))
            strip = VMobject(color=color, fill_opacity=min(1.0, 2.5 * fill_opacity))
            strip.set_stroke(width=0)
            top = height / 2
            floor = top - self.header
            outline = rounded_rect(ORIGIN, width, height, corner_radius)
            # The body's outline above the header's floor, closed along it.
            curves = outline.reshape(-1, 4, 3)
            kept = [c for c in curves if min(c[:, 1]) >= floor - 1e-9]
            strip.set_points(
                np.concatenate(
                    [
                        line(np.array([width / 2, floor, 0.0]), kept[0][0]),
                        *kept,
                        line(kept[-1][-1], np.array([-width / 2, floor, 0.0])),
                        line(
                            np.array([-width / 2, floor, 0.0]),
                            np.array([width / 2, floor, 0.0]),
                        ),
                    ]
                )
            )
            self.strip = strip
            self.title = _fit_text(
                Text(title, font_size=font_size, color=color),
                width * 0.85,
                self.header * 0.65,
            )
            self.title.move_to(np.array([0.0, top - self.header / 2, 0.0]))
            self.add(strip, self.title)
        chosen = SIDES if ports is None else ports
        placed = {}
        for name, where in chosen.items():
            if isinstance(where, tuple) and len(where) == 2 and np.ndim(where[1]) == 0:
                side, share = where
            else:
                side, share = where, 0.5
            placed[name] = _side_point(width, height, as_point(side), float(share))
        self._set_ports(placed)
        self.inside: VMobject | None = None
        if inside is not None:
            self._hold(inside, width, height)

    def _hold(self, inside: VMobject, width: float, height: float) -> None:
        room_w = width * INSIDE_SHARE
        room_h = (height - self.header) * INSIDE_SHARE
        scale = min(room_w / inside.width, room_h / inside.height)
        inside.scale(scale)
        inside.move_to(np.array([0.0, -self.header / 2, 0.0]))
        # How it shows once opened: each member's opacities, in family order.
        self.inside_shown = tuple(opacities_of(m) for m in inside.get_family())
        for member, shown in zip(inside.get_family(), self.inside_shown, strict=True):
            set_opacities(member, scaled(shown, 0.0))
        self.inside = inside
        self.add(inside)


class Node(Ported):
    """A disc of ``radius`` with ``label`` in it and ports: an atom in a
    structure, a vertex or a host, a junction or a meter (by its label).

    ``ports`` maps a name to the direction from the centre it sits in; by
    default ``left``, ``right``, ``top`` and ``bottom``. Style keywords go
    to the disc; the label keeps its own.
    """

    def __init__(
        self,
        label: str | None = None,
        radius: float = 0.35,
        ports: Mapping[str, Vector3DLike] | None = None,
        font_size: float = 28,
        color: ParsableManimColor = WHITE,
        fill_opacity: float = 0.15,
        **kwargs: Any,
    ) -> None:
        style = take_style(kwargs)
        super().__init__(**kwargs)
        self.disc = Circle(radius=radius, color=color, fill_opacity=fill_opacity)
        apply_style([self.disc], style)
        self.add(self.disc)
        self.label: Text | None = None
        if label is not None:
            self.label = _fit_text(
                Text(label, font_size=font_size, color=color),
                radius * 1.3,
                radius * 1.1,
            )
            self.add(self.label)
        chosen = SIDES if ports is None else ports
        placed = {}
        for name, toward in chosen.items():
            unit = as_point(toward)
            unit = unit / np.linalg.norm(unit)
            placed[name] = (unit * radius, unit)
        self._set_ports(placed)
