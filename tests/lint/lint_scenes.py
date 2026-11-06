"""Scenes the lint core tests execute.

A comment ``# @name`` tags a line the tests look up by name with
:func:`tests.lint.conftest.line_of`.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    Axes,
    Circle,
    Dot,
    ImageMobject,
    Line,
    MappingCamera,
    PMobject,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
    config,
    linear,
)


class Instants(Scene):
    """A trusted play, a frozen wait, a play and a frozen wait each shorter
    than a frame at every rate tested, and a ticked play and wait (a dt
    updater) after them.
    """

    def construct(self):
        dot = Dot(LEFT)
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5, rate_func=linear)
        self.wait(0.4)
        self.play(dot.animate.shift(UP), run_time=0.01)
        self.wait(0.005)
        dot.add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.play(dot.animate.set_color("#ff0000"), run_time=0.6)
        self.wait(0.3)
        dot.clear_updaters()


class Furnished(Scene):
    """A group of a thick-stroked square and a filled circle, a pair of axes,
    a text drawn twice in one place, and a group grown by a transform from
    one dot to three, which pads it.
    """

    def construct(self):
        shapes = VGroup(  # @shapes
            Square(side_length=1, stroke_width=8).shift(LEFT * 4),
            Circle(radius=0.5, fill_opacity=1, stroke_width=0).shift(LEFT * 2),
        )
        axes = Axes(x_range=[0, 2, 1], y_range=[0, 2, 1], x_length=2, y_length=2)
        label = Text("hi").shift(UP * 3)  # @label
        twin = label.copy()
        grown = VGroup(Dot(DOWN * 3))
        self.add(shapes, axes, label, twin, grown)
        self.play(
            Transform(grown, VGroup(*(Dot(DOWN * 3 + RIGHT * i) for i in range(3)))),
            run_time=0.25,
        )
        self.wait(0.25)


class Shrunk(Scene):
    """A group of three dots transformed into one: two of them end fully
    transparent, as the target's padding is, and draw nothing.
    """

    def construct(self):
        many = VGroup(*(Dot(RIGHT * i) for i in range(3)))
        self.add(many)
        self.play(Transform(many, VGroup(Dot(UP))), run_time=0.25)
        self.wait(0.25)


class Pair(Scene):
    """Two named dots, one made in a group, for the exemption tests."""

    def construct(self):
        left = Dot(LEFT)  # @left
        right = Dot(RIGHT)  # @right
        held = VGroup(Dot(DOWN))  # @held
        self.add(left, right, held)
        self.play(left.animate.shift(UP), run_time=0.5)


class RefusedInstants(Instants):
    """:class:`Instants` under a mapping camera, whose record draws nothing."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=MappingCamera, **kwargs)


class Backdropped(Scene):
    """A text on a wide background rectangle of its own."""

    def construct(self):
        label = Text("hi").add_background_rectangle(buff=1)
        self.add(label)
        self.wait(0.25)


class StyledCopy(Scene):
    """A text and, in the same place, a copy of it with a thick stroke."""

    def construct(self):
        label = Text("hi")
        thick = label.copy().set_stroke(width=20, opacity=1)
        self.add(label, thick)
        self.wait(0.25)


class InvisibleImage(Scene):
    """An image made fully transparent."""

    def construct(self):
        image = ImageMobject(np.full((4, 4, 4), 255, dtype=np.uint8)).set_opacity(0)
        self.add(image)
        self.wait(0.25)


class WideCloud(Scene):
    """One point of a cloud drawn 40 wide."""

    def construct(self):
        cloud = PMobject(stroke_width=40).add_points(np.array([[0.0, 0.0, 0.0]]))
        self.add(cloud)
        self.wait(0.25)


class Drifting(Scene):
    """A dot moved by a dt updater through a wait."""

    def construct(self):
        dot = Dot()
        dot.add_updater(lambda m, dt: m.shift(RIGHT * dt))
        self.add(dot)
        self.wait(1)


class Hooked(Scene):
    """A dot the scene's own frame hook moves on each call, and a default
    the script sets while it runs and puts back at its end.
    """

    def construct(self):
        Dot.set_default(radius=0.25)
        self.calls = 0
        self.marker = Dot()
        self.add(self.marker)
        self.play(self.marker.animate.shift(UP), run_time=1)
        self.wait(1)
        Dot.set_default()

    def update_to_time(self, t):
        super().update_to_time(t)
        self.calls += 1
        self.marker.set_x(self.calls / 100)


class Recoloured(Scene):
    """A text with a recoloured copy and a faded copy, in one place."""

    def construct(self):
        label = Text("hi")
        red = label.copy().set_color(RED)
        faint = label.copy().set_opacity(0.3)
        self.add(label, red, faint)
        self.wait(0.25)


class CornerCloud(Scene):
    """A cloud drawn 40 wide, one point on each corner of the frame."""

    def construct(self):
        w, h = config.frame_width / 2, config.frame_height / 2
        corners = [[x, y, 0.0] for x in (-w, w) for y in (-h, h)]
        cloud = PMobject(stroke_width=40).add_points(np.array(corners))
        self.add(cloud)
        self.wait(0.25)


def lines(across: int, down: int, *, side: float = 2.0, short: float = 0.0) -> VGroup:
    """``across`` horizontal and ``down`` vertical lines spaced evenly over a
    square of ``side``, each from edge to edge of it, or ``short`` of each
    edge.
    """
    half = side / 2

    def spaced(count):
        return (
            [half * (2 * i / (count - 1) - 1) for i in range(count)]
            if count > 1
            else [0.0]
        )

    reach = half - short
    rows = (
        [Line([-reach, y, 0], [reach, y, 0]) for y in spaced(across)] if across else []
    )
    cols = [Line([x, -reach, 0], [x, reach, 0]) for x in spaced(down)] if down else []
    return VGroup(*rows, *cols)


class Lattices(Scene):
    """Groups of lines side by side: a lattice of three each way, and three
    that are not one (one direction only, two lines one way, lines stopping
    short of the box their group spans).
    """

    def construct(self):
        lattice = lines(3, 3).shift(LEFT * 4.5)
        parallel = lines(5, 0).shift(LEFT * 1.5)
        sparse = lines(2, 5).shift(RIGHT * 1.5)
        short = VGroup(lines(3, 0, short=0.3), lines(0, 3)).shift(RIGHT * 4.5)
        self.add(lattice, parallel, sparse, short)
        self.wait(0.25)
