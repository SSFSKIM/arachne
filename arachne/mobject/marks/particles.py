"""Particle marks: charges, a scatter, a burst, spreading wavefronts and a
trail of dots.
"""

from __future__ import annotations

__all__ = ["Burst", "ChargedParticle", "DottedTrail", "Particles", "WavefrontArcs"]

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.constants import ORIGIN, RIGHT, TAU
from arachne.mobject.geometry.arc import Circle, Dot
from arachne.mobject.types.vectorized_mobject import VGroup, VMobject
from arachne.typing import Point3DLike, Vector3DLike
from arachne.utils.color import BLUE, GREY, RED, WHITE, ParsableManimColor

from ._paths import arc, as_point, line

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject


class ChargedParticle(VGroup):
    """A disc of ``radius`` bearing a plus (``charge > 0``), a minus
    (``charge < 0``) or nothing: point charges, ions in solution, ions at a
    membrane. ``charge`` is kept as given; its sign picks the sign drawn and,
    unless ``color`` is given, the colour.
    """

    def __init__(
        self,
        charge: float = 1,
        radius: float = 0.2,
        color: ParsableManimColor | None = None,
        fill_opacity: float = 0.8,
        sign_color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.charge = charge
        shade = (
            color
            if color is not None
            else RED
            if charge > 0
            else BLUE
            if charge < 0
            else GREY
        )
        self.disc = Circle(radius=radius, color=shade, fill_opacity=fill_opacity)
        self.sign = VMobject(color=sign_color, stroke_width=max(2.0, 12 * radius))
        bar = radius * 0.55
        strokes = []
        if charge != 0:
            strokes.append(line(np.array([-bar, 0.0, 0.0]), np.array([bar, 0.0, 0.0])))
        if charge > 0:
            strokes.append(line(np.array([0.0, -bar, 0.0]), np.array([0.0, bar, 0.0])))
        if strokes:
            self.sign.set_points(np.concatenate(strokes))
        self.add(self.disc, self.sign)


class Particles(VGroup):
    """``count`` small discs of ``radius`` scattered over a ``width`` by
    ``height`` region about the origin, by a random draw seeded with
    ``seed``, no two nearer than ``min_distance`` (twice their diameter by
    default) where the region has room: a gas, a concentration, a
    population. The same arguments give the same scatter.
    """

    #: Draws tried for each particle before one is placed regardless.
    TRIES = 60

    def __init__(
        self,
        count: int = 40,
        width: float = 3.0,
        height: float = 2.0,
        radius: float = 0.05,
        seed: int = 0,
        min_distance: float | None = None,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if count < 0:
            raise ValueError("a scatter has no fewer than no particles")
        super().__init__(**kwargs)
        self.count = int(count)
        self.seed = seed
        rng = np.random.default_rng(seed)
        apart = 4 * radius if min_distance is None else min_distance
        low = np.array([-width / 2 + radius, -height / 2 + radius])
        high = np.array([width / 2 - radius, height / 2 - radius])
        placed = np.empty((0, 2))
        for _ in range(self.count):
            choice = rng.uniform(low, high)
            for _ in range(self.TRIES):
                if (
                    not len(placed)
                    or np.linalg.norm(placed - choice, axis=1).min() >= apart
                ):
                    break
                choice = rng.uniform(low, high)
            placed = np.vstack([placed, choice])
        for x, y in placed:
            self.add(Dot(np.array([x, y, 0.0]), radius=radius, color=color))


class Burst(VMobject):
    """``rays`` short strokes radiating from ``center``, each from
    ``inner_radius`` to ``outer_radius``: a light source, a collision or
    reaction event, a signal firing.
    """

    def __init__(
        self,
        rays: int = 12,
        inner_radius: float = 0.3,
        outer_radius: float = 0.6,
        center: Point3DLike = ORIGIN,
        start_angle: float = 0.0,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if rays < 1 or not 0 <= inner_radius < outer_radius:
            raise ValueError(
                "a burst has rays from an inner radius out to an outer one"
            )
        super().__init__(color=color, **kwargs)
        self.rays = int(rays)
        middle = as_point(center)
        strokes = []
        for index in range(self.rays):
            angle = start_angle + TAU * index / self.rays
            unit = np.array([math.cos(angle), math.sin(angle), 0.0])
            strokes.append(
                line(middle + unit * inner_radius, middle + unit * outer_radius)
            )
        self.set_points(np.concatenate(strokes))


class WavefrontArcs(VMobject):
    """``count`` concentric arcs about ``center``, ``spacing`` apart from
    ``start_radius`` out, each spanning ``angle`` about ``direction`` (a
    whole circle at ``TAU``): sound or light spreading, a broadcast, a
    diffusing signal.
    """

    def __init__(
        self,
        count: int = 4,
        spacing: float = 0.35,
        start_radius: float = 0.35,
        angle: float = TAU / 4,
        direction: Vector3DLike = RIGHT,
        center: Point3DLike = ORIGIN,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        if count < 1 or spacing <= 0 or start_radius <= 0 or not 0 < angle <= TAU:
            raise ValueError(
                "wavefronts have a count, a positive spacing and start radius, and"
                " an angle up to a whole turn"
            )
        super().__init__(color=color, **kwargs)
        self.count = int(count)
        self.spacing = float(spacing)
        self.start_radius = float(start_radius)
        middle = as_point(center)
        toward = as_point(direction)
        facing = math.atan2(toward[1], toward[0])
        self.set_points(
            np.concatenate(
                [
                    arc(middle, start_radius + i * spacing, facing - angle / 2, angle)
                    for i in range(self.count)
                ]
            )
        )


class DottedTrail(VMobject):
    """Dots left along ``mobject``'s path: one where its centre is when the
    trail is made, and one every ``interval`` seconds of scene time after,
    for as long as the trail is in the scene and running. A trajectory, a
    migration path, a packet's trace.

    It keeps time on the tick: its updater adds the ``dt`` it is handed, so
    a dot falls at the same scene time at any frame rate. :meth:`stop`
    leaves the dots as they are.
    """

    def __init__(
        self,
        mobject: Mobject,
        interval: float = 0.2,
        dot_radius: float = 0.04,
        color: ParsableManimColor = WHITE,
        fill_opacity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if interval <= 0:
            raise ValueError("a trail's interval is a positive number of seconds")
        super().__init__(
            color=color, fill_opacity=fill_opacity, stroke_width=0, **kwargs
        )
        self.interval = float(interval)
        self.dot_radius = float(dot_radius)
        #: Seconds of scene time the trail has run.
        self.elapsed = 0.0
        #: Where each dot fell, in order.
        self.dots: tuple[tuple[float, float, float], ...] = ()
        self._drop(mobject.get_center())

        def keep_time(trail: Mobject, dt: float) -> None:
            assert isinstance(trail, DottedTrail)
            trail.elapsed += dt
            # A dot falls at every whole interval passed (to a rounding).
            while trail.elapsed >= len(trail.dots) * trail.interval - 1e-9:
                trail._drop(mobject.get_center())

        self._keeper = keep_time
        self.add_updater(keep_time)

    def _drop(self, where: np.ndarray) -> None:
        self.dots = (*self.dots, (float(where[0]), float(where[1]), float(where[2])))
        self.set_points(
            np.concatenate(
                [arc(np.array(p), self.dot_radius, 0.0, TAU) for p in self.dots]
            )
        )

    def stop(self) -> DottedTrail:
        """Stop dropping dots; those dropped stay."""
        self.remove_updater(self._keeper)
        return self
