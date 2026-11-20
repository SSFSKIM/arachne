"""Circuit and logic marks: plates, a switch, a one-way mark and the logic
gates.
"""

from __future__ import annotations

__all__ = ["LogicGate", "OneWayMark", "PlatePair", "Switch"]

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from arachne.constants import LEFT, RIGHT, TAU
from arachne.mobject.geometry.arc import Circle
from arachne.mobject.types.vectorized_mobject import VGroup, VMobject
from arachne.typing import Point3DLike
from arachne.utils.color import WHITE, ParsableManimColor

from ._paths import arc, as_point, direction, hermite, line, normal, polyline
from ._style import apply_style, take_style
from .ports import Ported


class PlatePair(VMobject):
    """Two parallel plates ``gap`` apart across the middle of the course
    from ``start`` to ``end``, with leads to each end: a capacitor, an
    electrochemical cell (one plate long, one short), a membrane as a
    capacitor. ``lengths`` are the plates', the one nearer ``start`` first.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        gap: float = 0.25,
        lengths: Sequence[float] = (0.8, 0.8),
        leads: bool = True,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if len(lengths) != 2 or min(lengths) <= 0:
            raise ValueError("a plate pair has two positive plate lengths")
        super().__init__(color=color, **kwargs)
        self.lengths = (float(lengths[0]), float(lengths[1]))
        self.gap = float(gap)
        a, b = as_point(start), as_point(end)
        u, n = direction(a, b), normal(direction(a, b))
        middle = (a + b) / 2
        near, far = middle - u * gap / 2, middle + u * gap / 2
        strokes = [
            line(near + n * self.lengths[0] / 2, near - n * self.lengths[0] / 2),
            line(far + n * self.lengths[1] / 2, far - n * self.lengths[1] / 2),
        ]
        if leads:
            strokes += [line(a, near), line(far, b)]
        self.set_points(np.concatenate(strokes))


class Switch(VGroup):
    """A contact from ``start`` to ``end``, drawn open (its blade lifted by
    ``open_angle``) or closed: a circuit switch, a relay or a bit, a gated
    channel. :meth:`set_closed` changes it in place. Style keywords go to
    the contacts and the blade.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        closed: bool = False,
        open_angle: float = TAU / 12,
        contact_radius: float = 0.06,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        style = take_style(kwargs)
        super().__init__(**kwargs)
        self._start, self._end = as_point(start), as_point(end)
        self.open_angle = float(open_angle)
        self.contacts = VGroup(
            Circle(radius=contact_radius, color=color).move_to(self._start),
            Circle(radius=contact_radius, color=color).move_to(self._end),
        )
        self.blade = VMobject(color=color)
        apply_style([*self.contacts, self.blade], style)
        self.add(self.contacts, self.blade)
        self.closed = bool(closed)
        self.set_closed(self.closed)

    def set_closed(self, closed: bool) -> Switch:
        """Draw the switch closed or open, where its contacts are now: the
        blade pivots on the first contact and runs from its ring's edge to
        the second ring's edge, or as far lifted, so it draws nothing inside
        a ring. Each edge is where the blade's line meets the ring as drawn,
        so a switch stretched unevenly or turned still meets its rings.
        """
        self.closed = bool(closed)
        first, second = (c.get_center() for c in self.contacts)
        u = direction(first, second)
        length = float(np.linalg.norm(second - first))
        angle = 0.0 if self.closed else self.open_angle
        along = u * math.cos(angle) + normal(u) * math.sin(angle)
        near = _rim_reach(self.contacts[0], first, along)
        far = _rim_reach(self.contacts[1], second, -u)
        self.blade.set_points(
            line(first + along * near, first + along * (length - far))
        )
        return self


def _rim_reach(ring: VMobject, centre: np.ndarray, way: np.ndarray) -> float:
    """How far from ``centre`` along the unit direction ``way`` the outline
    of ``ring`` lies, read off its drawn curves (so a ring stretched unevenly
    or turned is met where it is drawn); 0 when the ray meets none of it.
    """
    curves = np.asarray(ring.points, dtype=float).reshape(-1, 4, 3)
    # Fine enough that a chord lies within 1e-6 of a unit ring's curve.
    t = np.linspace(0.0, 1.0, 257)[:, None, None]
    outline = (
        (1 - t) ** 3 * curves[:, 0]
        + 3 * (1 - t) ** 2 * t * curves[:, 1]
        + 3 * (1 - t) * t**2 * curves[:, 2]
        + t**3 * curves[:, 3]
    )
    starts = outline[:-1].reshape(-1, 3)[:, :2] - centre[:2]
    ends = outline[1:].reshape(-1, 3)[:, :2] - centre[:2]
    edge = ends - starts
    # Solve centre + s * way = start + r * edge, for s >= 0 and 0 <= r <= 1.
    cross = way[0] * edge[:, 1] - way[1] * edge[:, 0]
    usable = np.abs(cross) > 1e-15
    safe = np.where(usable, cross, 1.0)
    s = (starts[:, 0] * edge[:, 1] - starts[:, 1] * edge[:, 0]) / safe
    r = (starts[:, 0] * way[1] - starts[:, 1] * way[0]) / safe
    hit = usable & (s >= 0) & (r >= -1e-12) & (r <= 1 + 1e-12)
    return float(s[hit].max()) if hit.any() else 0.0


class OneWayMark(VMobject):
    """A filled triangle against a bar, pointing from ``start`` to ``end``,
    with leads to each: a diode, a check valve, a one-way link.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        size: float = 0.35,
        leads: bool = True,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, **kwargs)
        a, b = as_point(start), as_point(end)
        u, n = direction(a, b), normal(direction(a, b))
        middle = (a + b) / 2
        back, apex = middle - u * size / 2, middle + u * size / 2
        strokes = [line(apex + n * size / 2, apex - n * size / 2)]
        if leads:
            strokes += [line(a, back), line(apex, b)]
        self.set_points(np.concatenate(strokes))
        self.triangle = VMobject(color=color, fill_opacity=1.0, stroke_width=0)
        self.triangle.set_points(
            polyline([apex, back + n * size / 2, back - n * size / 2], closed=True)
        )
        self.add(self.triangle)


#: The gates :class:`LogicGate` draws, and whether each inverts its output.
GATES = {
    "and": False,
    "nand": True,
    "or": False,
    "nor": True,
    "xor": False,
    "xnor": True,
    "buffer": False,
    "not": True,
}


class LogicGate(Ported):
    """One of the standard gate outlines (``kind``, a key of :data:`GATES`)
    ``size`` tall, with ``inputs`` input ports ``in0``, ``in1``, ... (top to
    bottom) on short leads at its back and an output port ``out`` on a lead
    at its front: digital logic, Boolean algebra, threshold or regulatory
    logic. A buffer and a not take one input; the others two or more.
    Style keywords go to the outline, the leads and the bubble.
    """

    def __init__(
        self,
        kind: str = "and",
        inputs: int | None = None,
        size: float = 1.0,
        lead: float = 0.25,
        color: ParsableManimColor = WHITE,
        fill_opacity: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if kind not in GATES:
            raise ValueError(f"a logic gate is one of {', '.join(GATES)}; not {kind!r}")
        single = kind in ("buffer", "not")
        count = (1 if single else 2) if inputs is None else int(inputs)
        if single and count != 1:
            raise ValueError(f"a {kind} gate has one input")
        if not single and count < 2:
            raise ValueError(f"a {kind} gate has two or more inputs")
        style = take_style(kwargs)
        super().__init__(**kwargs)
        self.kind = kind
        self.inputs = count
        h = size / 2
        curved = kind in ("or", "nor", "xor", "xnor")
        bulge = 0.22 * size if curved else 0.0
        # The outer back curve an exclusive gate's inputs reach.
        shadow = 0.15 * size if kind in ("xor", "xnor") else 0.0

        def back_x(y: float) -> float:
            return -h + bulge * (1 - (y / h) ** 2)

        def back_edge(t: float) -> np.ndarray:
            y = -h + 2 * h * t
            return np.array([back_x(y), y, 0.0])

        def back_slope(t: float) -> np.ndarray:
            y = -h + 2 * h * t
            return np.array([-bulge * 2 * y / h**2 * 2 * h, 2 * h, 0.0])

        corner_top, corner_bottom = np.array([-h, h, 0.0]), np.array([-h, -h, 0.0])
        if kind in ("and", "nand"):
            front = h
            outline = np.concatenate(
                [
                    polyline(
                        [
                            np.array([0.0, -h, 0.0]),
                            corner_bottom,
                            corner_top,
                            np.array([0.0, h, 0.0]),
                        ]
                    ),
                    arc(np.zeros(3), h, TAU / 4, -TAU / 2),
                ]
            )
        elif single:
            front = -h + 0.9 * size
            outline = polyline(
                [corner_bottom, corner_top, np.array([front, 0.0, 0.0])], closed=True
            )
        else:
            front = 0.6 * size
            tip = np.array([front, 0.0, 0.0])
            reach = np.array([0.45 * size, 0.0, 0.0])
            lean = np.array([-0.3 * size, 0.28 * size, 0.0])
            outline = np.concatenate(
                [
                    np.array([corner_top, corner_top + reach, tip + lean, tip]),
                    np.array(
                        [
                            tip,
                            tip + lean * [1, -1, 1],
                            corner_bottom + reach,
                            corner_bottom,
                        ]
                    ),
                    hermite(back_edge, back_slope, 0.0, 1.0, 4),
                ]
            )
        self.body = VMobject(color=color, fill_opacity=fill_opacity)
        self.body.set_points(outline)
        self.add(self.body)
        strokes: list[np.ndarray] = []
        if shadow:
            offset = np.array([-shadow, 0.0, 0.0])
            strokes.append(
                hermite(lambda t: back_edge(t) + offset, back_slope, 0.0, 1.0, 4)
            )
        out = np.array([front, 0.0, 0.0])
        self.bubble: Circle | None = None
        if GATES[kind]:
            radius = 0.08 * size
            self.bubble = Circle(radius=radius, color=color).move_to(
                out + RIGHT * radius
            )
            self.add(self.bubble)
            out = out + RIGHT * 2 * radius
        heights = (
            [0.0]
            if count == 1
            else [0.6 * h - 1.2 * h * i / (count - 1) for i in range(count)]
        )
        ports: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for index, y in enumerate(heights):
            meets = np.array([back_x(y) - shadow, y, 0.0])
            outer = meets + LEFT * lead
            strokes.append(line(outer, meets))
            ports[f"in{index}"] = (outer, LEFT)
        end = out + RIGHT * lead
        strokes.append(line(out, end))
        ports["out"] = (end, RIGHT)
        self.leads = VMobject(color=color)
        self.leads.set_points(np.concatenate(strokes))
        self.add(self.leads)
        apply_style(
            [m for m in (self.body, self.leads, self.bubble) if m is not None],
            style,
        )
        self._set_ports(ports)
