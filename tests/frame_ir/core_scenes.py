"""3D scenes for the core path: the overlay Cairo draws over the core's frame.

Each scene has a twin without its overlay (``overlay = False``), so a test can
tell the pixels the overlay changes from those the core and Cairo already
agree on beneath it.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    DEGREES,
    ORIGIN,
    RED,
    WHITE,
    YELLOW,
    Rectangle,
    Scene,
    Square,
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
