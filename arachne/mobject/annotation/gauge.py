"""Gauges: a declared value shown as a share of a scale.

- :class:`BarGauge`: a horizontal track filled from its left end;
- :class:`ColumnGauge`: a vertical track filled from its bottom;
- :class:`DialGauge`: an arc with a needle turned through the value's share
  of its sweep.

Each holds its value as a plain number on its scale, ``value_range``,
changed by :meth:`set_value` or counted by
:class:`~.readout.CountUp`, and is drawn from its track as it stands:
moved, scaled or turned, a gauge shows the same share. Each attaches to an
anchor as every annotation does (:mod:`.attach`).
"""

from __future__ import annotations

__all__ = ["BarGauge", "ColumnGauge", "DialGauge"]

import math
from typing import TYPE_CHECKING

import numpy as np

from arachne.constants import DEGREES, ORIGIN
from arachne.mobject.geometry.arc import Arc, Dot
from arachne.mobject.geometry.polygram import Rectangle
from arachne.mobject.types.vectorized_mobject import VGroup, VMobject
from arachne.utils.color import BLUE_C, GREY_B, ParsableManimColor

from .attach import DEFAULT_OFFSET, Annotation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from arachne.assets.anchor import Anchor


class _Gauge(Annotation):
    """A gauge's scale and value; a subclass draws the value on its body."""

    def __init__(
        self,
        value: float,
        value_range: tuple[float, float],
        body: VMobject,
        *,
        at: Anchor | None,
        offset: Sequence[float] | np.ndarray,
        follow: bool,
        leader: bool,
        leader_color: ParsableManimColor,
        name: str | None,
    ) -> None:
        low, high = (float(end) for end in value_range)
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(
                f"A gauge's value_range runs from its min to a larger max, not"
                f" {tuple(value_range)!r}."
            )
        #: The scale's two ends.
        self.min, self.max = low, high
        self._value = self._checked(value)
        super().__init__(
            body,
            at=at,
            offset=offset,
            follow=follow,
            leader=leader,
            leader_color=leader_color,
            name=name,
        )
        self.set_value(self._value)

    @property
    def value(self) -> float:
        """The declared value now."""
        return self._value

    @property
    def share(self) -> float:
        """The value's share of the scale, 0 at ``min``, 1 at ``max``."""
        return (self._value - self.min) / (self.max - self.min)

    def _checked(self, value: float) -> float:
        number = float(value)
        if not (math.isfinite(number) and self.min <= number <= self.max):
            raise ValueError(
                f"A gauge shows a value on its scale, from {self.min:g} to"
                f" {self.max:g}, not {value!r}."
            )
        return number

    def set_value(self, value: float) -> Self:
        """Declare ``value`` and draw it."""
        self._value = self._checked(value)
        self._draw()
        return self

    def _draw(self) -> None:
        raise NotImplementedError


class _TrackGauge(_Gauge):
    """A rectangular track filled from one end: from its left with
    ``upright`` false, from its bottom with it true.
    """

    _upright = False

    def __init__(
        self,
        value: float = 0,
        value_range: tuple[float, float] = (0, 1),
        at: Anchor | None = None,
        *,
        length: float = 2.0,
        thickness: float = 0.2,
        color: ParsableManimColor = BLUE_C,
        track_color: ParsableManimColor = GREY_B,
        offset: Sequence[float] | np.ndarray = DEFAULT_OFFSET,
        follow: bool = True,
        leader: bool = False,
        name: str | None = None,
    ) -> None:
        wide, tall = (thickness, length) if self._upright else (length, thickness)
        #: The outline the value fills.
        self.track = Rectangle(
            width=wide, height=tall, color=track_color, stroke_width=2
        )
        #: The filled share.
        self.fill = VMobject(fill_color=color, fill_opacity=1.0, stroke_width=0)
        super().__init__(
            value,
            value_range,
            VGroup(self.fill, self.track),
            at=at,
            offset=offset,
            follow=follow,
            leader=leader,
            leader_color=track_color,
            name=name,
        )

    def _draw(self) -> None:
        if not len(self.track.points):
            # Attached to nothing, it draws nothing.
            return
        # A rectangle's corners, in order: upper right, upper left, lower
        # left, lower right; as it stands now.
        ur, ul, dl, dr = np.asarray(self.track.get_vertices())[:4]
        share = self.share
        if self._upright:
            corners = [dl, dr, dr + share * (ur - dr), dl + share * (ul - dl)]
        else:
            corners = [dl, dl + share * (dr - dl), ul + share * (ur - ul), ul]
        self.fill.set_points_as_corners(np.array([*corners, corners[0]]))


class BarGauge(_TrackGauge):
    """``value`` on a scale ``value_range`` (its min and max), shown as the
    filled share of a horizontal track ``length`` long and ``thickness`` tall,
    filled from its left end; attached like every annotation.
    """


class ColumnGauge(_TrackGauge):
    """``value`` on a scale ``value_range`` (its min and max), shown as the
    filled share of a vertical track ``length`` tall and ``thickness`` wide, filled
    from its bottom; attached like every annotation.
    """

    _upright = True


class DialGauge(_Gauge):
    """``value`` on a scale ``value_range`` (its min and max), shown by a
    needle on a
    dial of ``radius``: the arc runs from ``start_angle`` to ``end_angle``
    (by default from lower left, clockwise over the top, to lower right)
    and the needle turns through the value's share of it; attached like
    every annotation.
    """

    def __init__(
        self,
        value: float = 0,
        value_range: tuple[float, float] = (0, 1),
        at: Anchor | None = None,
        *,
        radius: float = 0.6,
        start_angle: float = 225 * DEGREES,
        end_angle: float = -45 * DEGREES,
        color: ParsableManimColor = BLUE_C,
        track_color: ParsableManimColor = GREY_B,
        offset: Sequence[float] | np.ndarray = DEFAULT_OFFSET,
        follow: bool = True,
        leader: bool = False,
        name: str | None = None,
    ) -> None:
        sweep = float(end_angle - start_angle)
        if not 0 < abs(sweep) < 2 * math.pi:
            raise ValueError(
                "A dial's arc runs less than a full turn from its start_angle to its"
                f" end_angle, not {math.degrees(sweep):g} degrees."
            )
        #: The dial's arc.
        self.track = Arc(
            radius=radius,
            start_angle=start_angle,
            angle=sweep,
            arc_center=ORIGIN,
            color=track_color,
            stroke_width=4,
        )
        #: The dial's centre.
        self.hub = Dot(ORIGIN, radius=radius * 0.08, color=color)
        #: The needle, from the centre toward the arc.
        self.needle = VMobject(color=color, stroke_width=4)
        self._sweep = abs(sweep)
        super().__init__(
            value,
            value_range,
            VGroup(self.track, self.needle, self.hub),
            at=at,
            offset=offset,
            follow=follow,
            leader=leader,
            leader_color=track_color,
            name=name,
        )

    def _draw(self) -> None:
        if not len(self.track.points):
            return
        centre = self.hub.get_center()
        start = self.track.points[0] - centre
        middle = self.track.point_from_proportion(0.5) - centre
        # The arc's turn as it stands: a gauge seen in a mirror turns back.
        turn = math.copysign(
            self._sweep, float(start[0] * middle[1] - start[1] * middle[0])
        )
        angle = turn * self.share
        cos, sin = math.cos(angle), math.sin(angle)
        tip = 0.85 * np.array(
            [start[0] * cos - start[1] * sin, start[0] * sin + start[1] * cos, 0.0]
        )
        self.needle.set_points_as_corners(np.array([centre, centre + tip]))
