"""A magnifier: a small view of a place and a large view of its inside."""

from __future__ import annotations

__all__ = ["Magnifier"]

import math
from typing import Any

import numpy as np

from arachne.constants import ORIGIN, UR
from arachne.mobject.geometry.arc import Circle
from arachne.mobject.types.vectorized_mobject import VGroup, VMobject
from arachne.typing import Point3DLike
from arachne.utils.color import WHITE, ParsableManimColor

from ._paths import as_point, line


class Magnifier(VGroup):
    """A circle of ``radius`` about ``point``, a circle of ``large_radius``
    about ``center``, and the two lines tangent to both that join them: a
    cell's detail, the molecular view of a sample, a component inside a
    chip. ``small`` and ``large`` are the two circles, ``lines`` the
    tangents; what the large circle shows is the caller's to put in it.
    """

    def __init__(
        self,
        point: Point3DLike = ORIGIN,
        center: Point3DLike | None = None,
        radius: float = 0.3,
        large_radius: float = 1.2,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        near = as_point(point)
        far = near + 2.5 * UR if center is None else as_point(center)
        course = far - near
        apart = float(np.linalg.norm(course))
        if apart <= radius + large_radius:
            raise ValueError("a magnifier's two circles lie apart from each other")
        self.small = Circle(radius=radius, color=color).move_to(near)
        self.large = Circle(radius=large_radius, color=color).move_to(far)
        facing = math.atan2(course[1], course[0])
        turn = math.acos((radius - large_radius) / apart)
        self.lines = VMobject(color=color)
        pieces = []
        for sign in (1, -1):
            angle = facing + sign * turn
            unit = np.array([math.cos(angle), math.sin(angle), 0.0])
            pieces.append(line(near + unit * radius, far + unit * large_radius))
        self.lines.set_points(np.concatenate(pieces))
        self.add(self.lines, self.small, self.large)
