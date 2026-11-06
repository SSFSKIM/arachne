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
    Arrow,
    Axes,
    Circle,
    Dot,
    FadeIn,
    ImageMobject,
    Line,
    MappingCamera,
    ParametricFunction,
    PMobject,
    Polygon,
    Rectangle,
    Scene,
    Square,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    always_redraw,
    config,
    linear,
)
from arachne.lint.bridge import sweep


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


class FillGrownOverLattice(Scene):
    """A lattice of 31 lines each way, faint, played in; then a translucent
    fill redrawn every frame, empty at first (a group padded with a point),
    grown into a square over the lattice; then a sweep as the scene closes.
    """

    def construct(self):
        lattice = lines(31, 31, side=3.0).set_stroke(opacity=0.35)
        self.play(FadeIn(lattice), run_time=0.25)
        grown = ValueTracker(0.0)

        def fill():
            size = 3.0 * grown.get_value()
            if size <= 0:
                return VGroup()
            corner = np.array([-1.5, -1.5, 0.0])
            return Polygon(
                corner,
                corner + [size, 0, 0],
                corner + [size, size, 0],
                corner + [0, size, 0],
            ).set_style(fill_opacity=0.55, stroke_width=0)

        self.add(always_redraw(fill))
        self.play(grown.animate.set_value(1.0), run_time=0.25)
        self.found = sweep()
        self.wait(0.25)


class OverLattice(Scene):
    """A group of lines drawn first, then something drawn over it that
    crosses its lines, swept as the scene closes.
    """

    def grid(self) -> VGroup:
        return lines(5, 5)

    def over(self):
        return ParametricFunction(
            lambda t: np.array([t, 0.6 * np.sin(3 * t), 0.0]), t_range=(-1.0, 1.0)
        )

    def construct(self):
        self.add(self.grid())
        self.add(self.over())
        self.wait(0.25)
        self.found = sweep()
        self.wait(0.25)


class FilledOverLattice(OverLattice):
    """A filled square drawn over the lattice."""

    def over(self):
        return Square(side_length=1.2).set_style(fill_opacity=0.5, stroke_width=0)


class TextOverLattice(OverLattice):
    """A word laid across one of the lattice's lines."""

    def over(self):
        return Text("on a line", font_size=30)


class LatticeOverFill(OverLattice):
    """A filled square drawn first, the lattice drawn over it."""

    def construct(self):
        self.add(self.over_first())
        self.add(self.grid())
        self.wait(0.25)
        self.found = sweep()
        self.wait(0.25)

    def over_first(self):
        return FilledOverLattice.over(self)


class FillOverParallelLines(FilledOverLattice):
    """Five lines one way only: no grid."""

    def grid(self) -> VGroup:
        return lines(5, 0)


class FillOverSparseLattice(FilledOverLattice):
    """Two lines one way and five the other: no grid."""

    def grid(self) -> VGroup:
        return lines(2, 5)


class FillOverShortLattice(FilledOverLattice):
    """Lines one way stop short of the box the group spans: no grid."""

    def grid(self) -> VGroup:
        return VGroup(lines(5, 0, short=0.3), lines(0, 5))


#: One pixel of a picture 1080 pixels tall, in scene units.
REFERENCE_PIXEL = config.frame_height / 1080


class ArrowFromProbe(Scene):
    """An arrow drawn out from a probe dot, the dot drawn over its tail; the
    arrow at opacity 0.8, which fills its straight shaft too.
    """

    def construct(self):
        arrow = Arrow(LEFT, RIGHT, buff=0).set_opacity(0.8)
        probe = Dot(LEFT).set_z_index(4)
        self.add(arrow, probe)
        self.wait(0.25)
        self.found = sweep()
        self.wait(0.25)


class BarOverText(Scene):
    """A filled bar two reference pixels tall laid across a word."""

    def construct(self):
        word = Text("under a bar", font_size=36)
        bar = Rectangle(width=3.0, height=2 * REFERENCE_PIXEL)
        self.add(word, bar.set_style(fill_opacity=1.0, stroke_width=0))
        self.wait(0.25)
        self.found = sweep()
        self.wait(0.25)


class ProbeOnSliver(Scene):
    """A probe dot on a filled sliver whose area is a few reference pixels:
    a fill at 1080 pixels tall, and under one pixel of a preview's.
    """

    def construct(self):
        sliver = Polygon([-1.5, 0, 0], [1.5, 0, 0], [0, 0.002, 0])
        probe = Dot().set_z_index(4)
        self.add(sliver.set_style(fill_opacity=1.0, stroke_width=0), probe)
        self.wait(0.25)
        self.found = sweep()
        self.wait(0.25)
