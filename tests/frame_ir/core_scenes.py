"""3D scenes for the core path: the overlay Cairo draws over the core's frame.

Each scene has a twin without its overlay (``overlay = False``), so a test can
tell the pixels the overlay changes from those the core and Cairo already
agree on beneath it.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DEGREES,
    GREEN,
    ORIGIN,
    RED,
    WHITE,
    YELLOW,
    Prism,
    Rectangle,
    Scene,
    Square,
    Surface,
    Text,
    ThreeDScene,
)


class FixedLabelOverSquare(ThreeDScene):
    """A translucent label and a translucent bar fixed in the frame, over an
    opaque square lying in space, under an orbit: the overlay drawn over core
    content that changes every frame.
    """

    overlay = True

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        square = Square(side_length=4, color=BLUE, fill_opacity=1.0)
        self.add(square)
        if self.overlay:
            label = Text("Label", font_size=72, color=YELLOW).move_to(ORIGIN)
            label.set_fill(opacity=0.6)
            bar = Rectangle(width=3, height=0.4, color=RED, fill_opacity=0.5)
            bar.set_stroke(WHITE, width=2, opacity=0.7).shift([0.0, -1.0, 0.0])
            self.add_fixed_in_frame_mobjects(label, bar)
        self.move_camera(theta=-20 * DEGREES, run_time=0.4)


class FixedLabelOverSquareBare(FixedLabelOverSquare):
    """:class:`FixedLabelOverSquare` without its overlay."""

    overlay = False


class FlatSquares(Scene):
    """A 2D scene: on ``auto`` it never goes through XF."""

    def construct(self):
        moved = Square(color=RED)
        self.add(Square(color=BLUE, fill_opacity=0.5), moved)
        self.play(moved.animate.shift([1.0, 0.0, 0.0]), run_time=0.2)


class ShadedSolids(ThreeDScene):
    """A prism and a gently curved surface lit by CE's light, each face a
    two-stop gradient, under a short orbit: nothing occludes itself, so
    CE's order and the core's depth agree inside every face.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-50 * DEGREES)
        prism = Prism(dimensions=[1.6, 1.2, 1.0], fill_color=RED).shift([-2.2, 0, 0])
        surface = Surface(
            lambda u, v: np.array([u, v, 0.15 * np.sin(u) * np.cos(v)]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(6, 6),
            checkerboard_colors=[BLUE, GREEN],
        ).shift([1.6, 0, 0])
        self.add(prism, surface)
        self.move_camera(theta=-30 * DEGREES, run_time=0.4)
