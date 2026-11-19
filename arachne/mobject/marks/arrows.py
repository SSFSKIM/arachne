"""Arrow marks: what moves, what turns back, what blocks and what passes.

Each is an ordinary vector object between points the caller gives; a mark
with a head carries it as a filled submobject, so the mark recolours whole.
"""

from __future__ import annotations

__all__ = [
    "BluntArrow",
    "CycleArrow",
    "ElbowConnector",
    "FlowArrow",
    "Ray",
    "ReversibleArrows",
]

import math
from typing import Any

import numpy as np

from arachne.constants import LEFT, ORIGIN, RIGHT, TAU
from arachne.mobject.types.vectorized_mobject import VMobject
from arachne.typing import Point3DLike, Vector3DLike
from arachne.utils.color import WHITE, ParsableManimColor

from ._paths import (
    arc,
    as_point,
    direction,
    head,
    line,
    normal,
    polyline,
    rounded_polyline,
)


def _head(points: np.ndarray, color: ParsableManimColor) -> VMobject:
    """A filled arrowhead drawn from ``points`` in ``color``."""
    tip = VMobject(color=color, fill_opacity=1.0, stroke_width=0)
    tip.set_points(points)
    return tip


class FlowArrow(VMobject):
    """A broad filled arrow from ``start`` to ``end`` whose shaft is
    ``magnitude * width_per_unit`` wide, so a row of them reads as amounts:
    energy passed on, a net reaction rate, data moved between stores.

    The head is twice the shaft's width and as long as it is wide, at most
    half the arrow's length.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        magnitude: float = 1.0,
        width_per_unit: float = 0.25,
        color: ParsableManimColor = WHITE,
        fill_opacity: float = 1.0,
        stroke_width: float = 0,
        **kwargs: Any,
    ) -> None:
        if magnitude <= 0 or width_per_unit <= 0:
            raise ValueError("a flow arrow's magnitude and width per unit are positive")
        super().__init__(
            color=color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
        )
        self.magnitude = float(magnitude)
        self.width_per_unit = float(width_per_unit)
        a, b = as_point(start), as_point(end)
        u = direction(a, b)
        n = normal(u)
        shaft = self.magnitude * self.width_per_unit
        length = float(np.linalg.norm(b - a))
        head_length = min(2 * shaft, length / 2)
        neck = b - u * head_length
        self.set_points(
            polyline(
                [
                    a + n * shaft / 2,
                    neck + n * shaft / 2,
                    neck + n * shaft,
                    b,
                    neck - n * shaft,
                    neck - n * shaft / 2,
                    a - n * shaft / 2,
                ],
                closed=True,
            )
        )


def _elbow_route(
    start: np.ndarray,
    leaving: np.ndarray,
    end: np.ndarray,
    arriving_from: np.ndarray,
    stub: float,
    detour: float,
) -> list[np.ndarray]:
    """Corner points of a right-angled route that leaves ``start`` along
    ``leaving`` and reaches ``end`` from the side ``arriving_from`` points
    to (an outward direction of the end's port).

    Two ports on parallel sides that do not face each other are joined round
    the outside. When the two lie within ``detour`` of one line across their
    sides, runs along that line would lie on each other or cross the pieces
    the ports are on, so the route goes round ``detour`` past the further
    port, to the side of positive ``detour``'s sign (up, or right).
    """
    horizontal_start = abs(leaving[0]) >= abs(leaving[1])
    horizontal_end = abs(arriving_from[0]) >= abs(arriving_from[1])
    s_axis = 0 if horizontal_start else 1
    e_axis = 0 if horizontal_end else 1
    s_sign = math.copysign(1.0, leaving[s_axis])
    e_sign = math.copysign(1.0, arriving_from[e_axis])
    out = start.copy()
    out[s_axis] += s_sign * stub
    into = end.copy()
    into[e_axis] += e_sign * stub
    if s_axis == e_axis:
        other = 1 - s_axis
        ahead = (end[s_axis] - start[s_axis]) * s_sign > 0
        if ahead and s_sign == -e_sign:
            # Facing each other: turn halfway.
            middle = (start[s_axis] + end[s_axis]) / 2
            a, b = start.copy(), end.copy()
            a[s_axis] = b[s_axis] = middle
            return [start, a, b, end]
        if abs(end[other] - start[other]) < abs(detour):
            # Side by side: round the outside, past the further port.
            beyond = (
                max(start[other], end[other]) + detour
                if detour >= 0
                else min(start[other], end[other]) + detour
            )
            a, b = out.copy(), into.copy()
            a[other] = b[other] = beyond
            return [start, out, a, b, into, end]
        if s_sign == e_sign:
            # Both ports face one way: go round past the further one.
            far = max(out[s_axis] * s_sign, into[s_axis] * s_sign) * s_sign
            a, b = start.copy(), end.copy()
            a[s_axis] = b[s_axis] = far
            return [start, a, b, end]
        # Facing away from each other: out, across between the two, in.
        middle = (start[other] + end[other]) / 2
        a, b = out.copy(), into.copy()
        a[other] = b[other] = middle
        return [start, out, a, b, into, end]
    reaches = (end[s_axis] - start[s_axis]) * s_sign > 0
    arrives = (start[e_axis] - end[e_axis]) * e_sign > 0
    if reaches and arrives:
        # One turn, where the leaving run meets the arriving one.
        corner = end.copy()
        corner[e_axis] = start[e_axis]
        return [start, corner, end]
    a = out.copy()
    a[e_axis] = into[e_axis]
    return [start, out, a, into, end]


class ElbowConnector(VMobject):
    """A right-angled connector with rounded corners from ``start`` to
    ``end``: it leaves ``start`` along ``start_direction`` and comes into
    ``end`` from the side ``end_direction`` points to, the outward
    directions of the two ports it joins. With ``tip`` it ends in a filled
    head at ``end``.

    Its own points are the wire, ending at the head's base, so a token
    sent along it (:class:`~arachne.mobject.diagram.Flow`) runs on the wire.
    ``stub`` is how far it runs straight out of a port before it turns, and
    ``detour`` how far past the further port it goes round two ports that
    sit side by side on parallel sides (:func:`_elbow_route`).
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        start_direction: Vector3DLike = RIGHT,
        end_direction: Vector3DLike = LEFT,
        corner_radius: float = 0.15,
        tip: bool = True,
        tip_length: float = 0.2,
        stub: float = 0.3,
        detour: float = 0.5,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, **kwargs)
        #: The text a diagram set on it, if any.
        self.label: VMobject | None = None
        a, b = as_point(start), as_point(end)
        leaving, arriving_from = as_point(start_direction), as_point(end_direction)
        self.corner_radius = float(corner_radius)
        corners = _elbow_route(a, leaving, b, arriving_from, stub, detour)
        if tip:
            last = corners[-2]
            u = direction(last, b)
            reach = min(tip_length, float(np.linalg.norm(b - last)) * 0.9)
            corners[-1] = b - u * reach
            self.add(_head(head(b, u, reach, reach * 0.9), color))
        self.set_points(rounded_polyline(corners, self.corner_radius))


class CycleArrow(VMobject):
    """An arrow bent most of the way round a circle of ``radius`` about
    ``center``: from the angle ``start_angle`` through ``angle`` (radians,
    counterclockwise when positive), with a head at its end.
    """

    def __init__(
        self,
        radius: float = 1.0,
        start_angle: float = TAU / 12,
        angle: float = 5 * TAU / 6,
        center: Point3DLike = ORIGIN,
        tip_length: float = 0.25,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if radius <= 0 or angle == 0:
            raise ValueError("a cycle arrow has a positive radius and turns")
        super().__init__(color=color, **kwargs)
        middle = as_point(center)
        self.radius = float(radius)
        reach = min(tip_length, abs(angle) * radius / 3)
        sign = math.copysign(1.0, angle)
        drawn = angle - sign * reach / radius
        end_angle = start_angle + angle
        tip = middle + radius * np.array([math.cos(end_angle), math.sin(end_angle), 0])
        base_angle = start_angle + drawn
        base = middle + radius * np.array(
            [math.cos(base_angle), math.sin(base_angle), 0]
        )
        self.set_points(arc(middle, radius, start_angle, drawn))
        self.add(_head(head(tip, direction(base, tip), reach, reach * 0.9), color))


class ReversibleArrows(VMobject):
    """Two opposed half-headed arrows between ``start`` and ``end``, ``gap``
    apart: the forward one above (to the left of the course), the reverse
    one below, each with one barb on its outer side.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        gap: float = 0.12,
        barb: float = 0.2,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, **kwargs)
        a, b = as_point(start), as_point(end)
        u = direction(a, b)
        n = normal(u)
        back = barb * math.cos(TAU / 12)
        side = barb * math.sin(TAU / 12)
        top_a, top_b = a + n * gap / 2, b + n * gap / 2
        low_a, low_b = a - n * gap / 2, b - n * gap / 2
        self.set_points(
            np.concatenate(
                [
                    line(top_a, top_b),
                    line(top_b, top_b - u * back + n * side),
                    line(low_b, low_a),
                    line(low_a, low_a + u * back - n * side),
                ]
            )
        )


class BluntArrow(VMobject):
    """A line from ``start`` ending in a bar ``bar_width`` across at
    ``end``: what inhibits, represses or blocks.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        bar_width: float = 0.35,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, **kwargs)
        a, b = as_point(start), as_point(end)
        n = normal(direction(a, b))
        self.set_points(
            np.concatenate(
                [line(a, b), line(b + n * bar_width / 2, b - n * bar_width / 2)]
            )
        )


class Ray(VMobject):
    """A straight line from ``start`` to ``end`` with its arrowhead at its
    middle: the course of a ray of light, a beam or a signal.
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        tip_length: float = 0.25,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, **kwargs)
        a, b = as_point(start), as_point(end)
        u = direction(a, b)
        reach = min(tip_length, float(np.linalg.norm(b - a)) / 3)
        tip = (a + b) / 2 + u * reach / 2
        self.set_points(line(a, b))
        self.add(_head(head(tip, u, reach, reach * 0.9), color))
