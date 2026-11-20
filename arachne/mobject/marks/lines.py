"""Line marks between two points: several lines at once, a zigzag, a coil,
a wave, and a measuring line.
"""

from __future__ import annotations

__all__ = ["Coil", "DimensionLine", "MultiLine", "WavyLine", "Zigzag"]

import math
from typing import Any

import numpy as np

from arachne.constants import LEFT, RIGHT, TAU
from arachne.mobject.text.text_mobject import Text
from arachne.mobject.types.vectorized_mobject import VMobject
from arachne.typing import Point3DLike
from arachne.utils.color import WHITE, ParsableManimColor

from ._paths import as_point, direction, head, hermite, line, normal, polyline


def _leads(
    a: np.ndarray, b: np.ndarray, lead: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """The two ends of a mark's body between ``a`` and ``b``, ``lead`` in
    from each (at most a quarter of the length), and the body's length.
    """
    length = float(np.linalg.norm(b - a))
    lead = max(0.0, min(lead, length / 4))
    u = direction(a, b)
    return a + u * lead, b - u * lead, length - 2 * lead


class MultiLine(VMobject):
    """``count`` parallel lines (one to three) from ``start`` to ``end``,
    ``spacing`` apart: a bond's order, parallel conductors, redundant links.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        count: int = 2,
        spacing: float = 0.1,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if count not in (1, 2, 3):
            raise ValueError(f"a multi-line has one to three lines, not {count}")
        super().__init__(color=color, **kwargs)
        self.count = count
        a, b = as_point(start), as_point(end)
        n = normal(direction(a, b))
        offsets = [(i - (count - 1) / 2) * spacing for i in range(count)]
        self.set_points(np.concatenate([line(a + n * o, b + n * o) for o in offsets]))


class Zigzag(VMobject):
    """A zigzag of ``turns`` peaks between straight leads ``lead`` long,
    from ``start`` to ``end``, ``amplitude`` to either side: a resistor, a
    spring seen flat, a strained bond.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        turns: int = 6,
        amplitude: float = 0.15,
        lead: float = 0.2,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if turns < 1:
            raise ValueError("a zigzag has at least one turn")
        super().__init__(color=color, **kwargs)
        self.turns = int(turns)
        a, b = as_point(start), as_point(end)
        u, n = direction(a, b), normal(direction(a, b))
        body_a, body_b, body = _leads(a, b, lead)
        corners = [a, body_a]
        halves = 2 * self.turns
        for index in range(halves):
            side = 1 if index % 2 == 0 else -1
            corners.append(
                body_a + u * body * (index + 0.5) / halves + n * side * amplitude
            )
        corners += [body_b, b]
        self.set_points(polyline(corners))


class Coil(VMobject):
    """A helix of ``turns`` loops drawn flat between straight leads, from
    ``start`` to ``end``, ``radius`` to either side: a spring, an inductor,
    a helical molecule. It is the side view of a helix seen a little off
    its axis, so each turn shows as a loop.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        turns: int = 6,
        radius: float = 0.2,
        lead: float = 0.2,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if turns < 1:
            raise ValueError("a coil has at least one turn")
        super().__init__(color=color, **kwargs)
        self.turns = int(turns)
        a, b = as_point(start), as_point(end)
        u, n = direction(a, b), normal(direction(a, b))
        body_a, body_b, body = _leads(a, b, lead)
        # How far a turn leans along the axis: past pitch/2pi, it loops.
        lean = min(0.6 * radius, body / 4)
        # The helix leaves the left lead rising from the axis at its first
        # loop's back and ends half a turn short of whole, coming down to the
        # axis at its last loop's front: the right lead leaves the last loop
        # as the left lead enters the first, mirrored, and the drawn helix is
        # its own mirror image. Its ``turns`` tops span the body.
        sweep = TAU * (self.turns - 0.5)
        pitch = (body - 2 * lean) / (self.turns - 0.5)

        def at(phi: float) -> np.ndarray:
            along = pitch * phi / TAU + lean * (1 - math.cos(phi))
            return body_a + u * along + n * radius * math.sin(phi)

        def slope(phi: float) -> np.ndarray:
            along = pitch / TAU + lean * math.sin(phi)
            return u * along + n * radius * math.cos(phi)

        body_points = hermite(at, slope, 0.0, sweep, 8 * self.turns - 4)
        self.set_points(np.concatenate([line(a, body_a), body_points, line(body_b, b)]))


class WavyLine(VMobject):
    """A sinusoid of ``wavelengths`` whole waves from ``start`` to ``end``,
    ``amplitude`` to either side: a photon, a pressure wave, an analogue
    signal.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        wavelengths: float = 4,
        amplitude: float = 0.15,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if wavelengths <= 0:
            raise ValueError("a wavy line has a positive number of wavelengths")
        super().__init__(color=color, **kwargs)
        self.wavelengths = float(wavelengths)
        a, b = as_point(start), as_point(end)
        u, n = direction(a, b), normal(direction(a, b))
        length = float(np.linalg.norm(b - a))
        k = TAU * self.wavelengths

        def at(t: float) -> np.ndarray:
            return a + u * length * t + n * amplitude * math.sin(k * t)

        def slope(t: float) -> np.ndarray:
            return u * length + n * amplitude * k * math.cos(k * t)

        pieces = max(4, math.ceil(8 * self.wavelengths))
        self.set_points(hermite(at, slope, 0.0, 1.0, pieces))


class DimensionLine(VMobject):
    """A double-ended measuring line from ``start`` to ``end``: a tick
    across each end, a head at each end pointing at its tick, and, with
    ``label``, the label upright in a gap at its middle. ``offset`` moves the
    whole line to the side (left of the course when positive), as a
    dimension sits beside what it measures.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        label: str | None = None,
        offset: float = 0.0,
        tick: float = 0.25,
        tip_length: float = 0.15,
        font_size: float = 24,
        buff: float = 0.1,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, **kwargs)
        a, b = as_point(start), as_point(end)
        u, n = direction(a, b), normal(direction(a, b))
        a, b = a + n * offset, b + n * offset
        length = float(np.linalg.norm(b - a))
        reach = min(tip_length, length / 4)
        self.label: Text | None = None
        runs = [(a + u * reach, b - u * reach)]
        if label is not None:
            text = Text(label, font_size=font_size, color=color)
            text.move_to((a + b) / 2)
            # The label's extent along the line, and room either side.
            half = (abs(u[0]) * text.width + abs(u[1]) * text.height) / 2 + buff
            half = min(half, length / 2 - reach)
            middle = (a + b) / 2
            runs = [
                (a + u * reach, middle - u * half),
                (middle + u * half, b - u * reach),
            ]
            self.label = text
        pieces = [line(p, q) for p, q in runs if float(np.linalg.norm(q - p)) > 1e-9]
        pieces += [
            line(a + n * tick / 2, a - n * tick / 2),
            line(b + n * tick / 2, b - n * tick / 2),
        ]
        self.set_points(np.concatenate(pieces))
        for tip, pointing in ((a, -u), (b, u)):
            cap = VMobject(color=color, fill_opacity=1.0, stroke_width=0)
            cap.set_points(head(tip, pointing, reach, reach * 0.9))
            self.add(cap)
        if self.label is not None:
            self.add(self.label)
