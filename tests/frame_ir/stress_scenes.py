"""Scenes at the snap's conditioning edge: geometry whose drawing derives a
direction or a decision from points a grid step apart or closer.

Each is drawn on the core snapped, on the core from its points as made, and
by Cairo; snapped, it must be no further from Cairo than unsnapped
(``test_canonical_stress``).
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DEGREES,
    GREEN,
    IN,
    ORANGE,
    PI,
    RED,
    TAU,
    WHITE,
    YELLOW,
    Circle,
    Cone,
    Dot,
    LineJointType,
    Scene,
    Surface,
    ThreeDScene,
    VMobject,
)

FPS = 15


def _path(anchors_and_handles: list[list[float]]) -> VMobject:
    """A path of cubics through the given points, four per cubic (CE's layout)."""
    path = VMobject()
    path.set_points(np.array(anchors_and_handles, dtype=np.float64))
    return path


def _corner(offset: float, at: tuple[float, float]) -> list[list[float]]:
    """Two cubics meeting at a right-angled corner at ``at``, each with its
    handles ``offset`` from its anchors (0: collapsed onto them).
    """
    x, y = at
    a, b, c = (x - 1.0, y, 0.0), (x, y, 0.0), (x, y + 1.0, 0.0)
    return [
        list(a),
        [a[0] + offset, a[1], 0.0],
        [b[0] - offset, b[1], 0.0],
        list(b),
        list(b),
        [b[0], b[1] + offset, 0.0],
        [c[0], c[1] - offset, 0.0],
        list(c),
    ]


class CollapsedHandles(Scene):
    """Thick polylines whose handles sit on their anchors, or 1e-14, 1e-9,
    1e-6 and 1e-4 off them (below, about and above a grid step), with miter,
    round and bevel joins and butt caps.
    """

    def construct(self):
        for row, joint in enumerate(
            [LineJointType.MITER, LineJointType.ROUND, LineJointType.BEVEL]
        ):
            for col, offset in enumerate([0.0, 1e-14, 1e-9, 1e-6, 1e-4]):
                path = _path(_corner(offset, (-5.0 + 2.4 * col, 1.6 - 1.9 * row)))
                path.set_stroke(WHITE, width=24).set_fill(opacity=0)
                path.joint_type = joint
                self.add(path)
        self.wait(1 / FPS)


class NearDegenerateSegments(Scene):
    """Paths with a cubic between anchors a grid step apart or less, inside
    thick strokes: a zero-length one, and ones 1e-12, 1e-7 and 3e-5 long.
    """

    def construct(self):
        for col, length in enumerate([0.0, 1e-12, 1e-7, 3e-5]):
            x = -4.5 + 3.0 * col
            a, b = (x - 1.0, -0.3, 0.0), (x, 0.0, 0.0)
            c, d = (x + length, 0.0, 0.0), (x + 1.0, 0.8, 0.0)
            points = []
            for p, q in [(a, b), (b, c), (c, d)]:
                p, q = np.array(p), np.array(q)
                points += [p, p + (q - p) / 3, p + 2 * (q - p) / 3, q]
            path = _path([list(p) for p in points])
            path.set_stroke(YELLOW, width=20).set_fill(opacity=0)
            self.add(path)
        self.wait(1 / FPS)


class TinyCirclesAndDots(Scene):
    """Circles and dots a few pixels across and less, stroked and filled."""

    def construct(self):
        for col, radius in enumerate([0.002, 0.008, 0.02, 0.06]):
            x = -4.0 + 2.5 * col
            self.add(
                Circle(radius=radius, color=BLUE)
                .set_stroke(width=4)
                .move_to([x, 1.0, 0])
            )
            self.add(Dot([x, -1.0, 0], radius=radius, color=ORANGE))
        self.wait(1 / FPS)


class SpherePoles(ThreeDScene):
    """A checkered sphere lit from below, both poles in view."""

    def construct(self):
        sphere = Surface(
            lambda u, v: np.array(
                [
                    1.5 * np.cos(u) * np.cos(v),
                    1.5 * np.cos(u) * np.sin(v),
                    1.5 * np.sin(u),
                ]
            ),
            v_range=[0, TAU],
            u_range=[-PI / 2, PI / 2],
            checkerboard_colors=[RED, GREEN],
            resolution=(15, 32),
        )
        self.renderer.camera.light_source.move_to(3 * IN)
        self.set_camera_orientation(phi=95 * DEGREES, theta=30 * DEGREES)
        self.add(sphere)
        self.wait(1 / FPS)


class ConeApex(ThreeDScene):
    """A cone seen from above its apex, where its faces meet in a point."""

    def construct(self):
        self.set_camera_orientation(phi=35 * DEGREES, theta=20 * DEGREES)
        self.add(Cone(base_radius=1.5, height=2.0, direction=np.array([0.0, 0.0, 1.0])))
        self.wait(1 / FPS)


SCENES = [
    CollapsedHandles,
    NearDegenerateSegments,
    TinyCirclesAndDots,
    SpherePoles,
    ConeApex,
]
