"""Scenes the stale-free tests render: static mobjects a later-added driver
writes, a camera a driver turns or relights, static mobjects nothing
touches, and a live-drawn scene whose drawing reads a class attribute.

A *driver* is a point-less mobject added after the mobjects it writes, so
the play's moving list starts at it and what it writes stays static.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Cube,
    DecimalNumber,
    Dot,
    Mobject,
    Scene,
    Square,
    ThreeDScene,
    Triangle,
    VGroup,
)
from arachne.utils.color import color_to_rgba

#: Where the body starts, and how fast its driver moves it (units per second).
BODY_START = LEFT * 4
BODY_SPEED = 3.0


def driver(update) -> Mobject:
    """A point-less mobject carrying ``update`` as its updater."""
    mob = Mobject()
    mob.add_updater(update)
    return mob


class DrivenBody(Scene):
    """A body added before the driver whose updater moves it."""

    def construct(self):
        body = Square(side_length=1, color=RED, fill_opacity=1).move_to(BODY_START)
        self.add(body)
        self.add(driver(lambda m, dt: body.shift(RIGHT * BODY_SPEED * dt)))
        self.wait(2)


def readout() -> DecimalNumber:
    return DecimalNumber(0, num_decimal_places=0, font_size=96)


class DrivenReadout(Scene):
    """A readout placed before a driver that rewrites it with ``set_value``,
    which replaces its glyphs.
    """

    def construct(self):
        number = readout()
        self.add(number)
        clock = {"t": 0.0}

        def count(m, dt):
            clock["t"] += dt
            number.set_value(int(clock["t"]))

        self.add(driver(count))
        self.wait(2.5)


class ReadoutAt(Scene):
    """The readout showing ``VALUE``, set the way the driver sets it, under
    a driver that writes nothing: one frame.
    """

    VALUE = 1

    def construct(self):
        number = readout()
        number.set_value(self.VALUE)
        self.add(number)
        self.add(driver(lambda m, dt: None))
        self.wait(1 / 15)


class ReadoutAtTwo(ReadoutAt):
    VALUE = 2


class PartlyWritten(Scene):
    """Three static mobjects; a driver moves the middle one."""

    def construct(self):
        first = Square(side_length=1.5, color=BLUE, fill_opacity=1).shift(LEFT * 4)
        middle = Circle(radius=0.5, color=RED, fill_opacity=1).shift(UP)
        last = Triangle(color=GREEN, fill_opacity=1).shift(RIGHT * 4)
        self.add(first, middle, last)
        self.add(driver(lambda m, dt: middle.shift(DOWN * dt)))
        self.wait(1)


class Untouched(Scene):
    """Static mobjects nothing writes, under two plays that move a dot."""

    def construct(self):
        self.add(Square(color=BLUE, fill_opacity=0.5).shift(LEFT * 3))
        self.add(Circle(color=GREEN, fill_opacity=0.5).shift(RIGHT * 3))
        dot = Dot(radius=0.3, color=YELLOW)
        self.add(dot)
        self.play(dot.animate.shift(UP * 2), run_time=1)
        self.play(dot.animate.shift(DOWN * 4), run_time=1)


#: The frame rate the camera drivers step at: their camera input is a
#: function of the frame they are at, so a still of that input can match it.
STEP_RATE = 15
PHI, THETA = 60 * np.pi / 180, -45 * np.pi / 180
#: How far a driver turns the azimuth, and moves the light, per frame.
THETA_STEP = 0.05
LIGHT_STEP = RIGHT * 3


def steps(clock: dict[str, float]) -> int:
    """The frames of the frame grid a clock has passed."""
    return round(clock["t"] * STEP_RATE)


def turned(at: int) -> float:
    return THETA + THETA_STEP * at


def lit(start: np.ndarray, at: int) -> np.ndarray:
    return np.array([start + LIGHT_STEP * at])


class TurnedCamera(ThreeDScene):
    """A static cube, and a driver that turns the camera's azimuth tracker
    by a step each frame.
    """

    def construct(self):
        self.set_camera_orientation(phi=PHI, theta=THETA)
        self.add(Cube(side_length=3, fill_opacity=1))
        theta = self.camera.theta_tracker
        clock = {"t": 0.0}

        def turn(m, dt):
            clock["t"] += dt
            theta.set_value(turned(steps(clock)))

        self.add(driver(turn))
        self.wait(1)


class TurnedCameraAt(ThreeDScene):
    """The cube with the azimuth the driver reaches at frame ``FRAME``, under
    a driver that writes nothing: one frame.
    """

    FRAME = 7

    def construct(self):
        self.set_camera_orientation(phi=PHI, theta=turned(self.FRAME))
        self.add(Cube(side_length=3, fill_opacity=1))
        self.add(driver(lambda m, dt: None))
        self.wait(1 / STEP_RATE)


class MovedLight(ThreeDScene):
    """A static cube, and a driver that moves the camera's light source by a
    step each frame.
    """

    def construct(self):
        self.set_camera_orientation(phi=PHI, theta=THETA)
        self.add(Cube(side_length=3, fill_opacity=1))
        light = self.camera.light_source
        start = light.points[0].copy()
        clock = {"t": 0.0}

        def move(m, dt):
            clock["t"] += dt
            light.set_points(lit(start, steps(clock)))

        self.add(driver(move))
        self.wait(1)


class MovedLightAt(ThreeDScene):
    """The cube lit from where the driver puts the light at frame ``FRAME``,
    under a driver that writes nothing: one frame.
    """

    FRAME = 7

    def construct(self):
        self.set_camera_orientation(phi=PHI, theta=THETA)
        self.add(Cube(side_length=3, fill_opacity=1))
        light = self.camera.light_source
        light.set_points(lit(light.points[0].copy(), self.FRAME))
        self.add(driver(lambda m, dt: None))
        self.wait(1 / STEP_RATE)


class ZOrderedChild(Scene):
    """Under z-index order a static square's child sorts after the moving
    cut, so it moves with the play while its parent stays static; the cutting
    dot's updater moves that child. CE's static picture paints the child too,
    where it stood when the play began.
    """

    def construct(self):
        square = Square(side_length=3, color=BLUE, fill_opacity=1)
        child = Dot(radius=0.4, color=RED, fill_opacity=1).shift(LEFT).set_z_index(5)
        square.add(child)
        cutting = Dot(radius=0.1, color=YELLOW).shift(DOWN * 3).set_z_index(3)
        cutting.add_updater(lambda m, dt: child.shift(RIGHT * 2 * dt))
        self.add(square, cutting)
        self.wait(1)


#: When the holder takes the static square in, and when it lets it go.
TAKE_IN, LET_GO = 0.3, 0.6


class TakenIn(Scene):
    """A moving holder takes a translucent static square in for a while,
    then lets it go: while it holds it, the square moves with the holder.
    """

    def construct(self):
        square = Square(side_length=1.5, color=BLUE, fill_opacity=0.5).shift(LEFT * 3)
        self.add(square)
        holder = VGroup()
        clock = {"t": 0.0}

        def hold(m, dt):
            clock["t"] += dt
            if TAKE_IN < clock["t"] < LET_GO and square not in m.submobjects:
                m.add(square)
            elif clock["t"] >= LET_GO and square in m.submobjects:
                m.remove(square)

        holder.add_updater(hold)
        self.add(holder, Dot(color=YELLOW).shift(RIGHT * 3))
        self.wait(1)


class Counted(Square):
    """A square whose drawing counts its fill reads: its drawing runs the
    scene's code, so its scene is drawn as it runs.
    """

    calls = 0

    def get_fill_rgbas(self):
        type(self).calls += 1
        return super().get_fill_rgbas()


class LiveHeld(Scene):
    """A live-drawn scene's frozen wait, then a play that moves a dot."""

    def construct(self):
        Counted.calls = 0
        self.add(Counted(color=BLUE, fill_opacity=1).shift(LEFT * 3))
        self.wait(1)
        self.held_calls = Counted.calls
        dot = Dot(color=RED)
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=1)


class Tinted(Square):
    """A square whose fill is read from its class's ``tint`` when drawn: its
    drawing runs the scene's code, so its scene is drawn as it runs.
    """

    tint = RED

    def get_fill_rgbas(self):
        return np.array([color_to_rgba(type(self).tint, 1.0)])


#: When the live scene's driver switches the tint.
TINT_SWITCH = 0.5


class LiveDriven(Scene):
    """A tinted square and a body, both added before a driver that moves the
    body and, half a second in, switches the square's class's tint.
    """

    def construct(self):
        Tinted.tint = RED  # an execution of the scene starts from red
        self.add(Tinted(side_length=1.5).shift(LEFT * 4 + UP * 2))
        body = Square(side_length=1, color=GREEN, fill_opacity=1).move_to(BODY_START)
        self.add(body)
        clock = {"t": 0.0}

        def drive(m, dt):
            clock["t"] += dt
            body.shift(RIGHT * BODY_SPEED * dt)
            if clock["t"] >= TINT_SWITCH:
                Tinted.tint = BLUE

        self.add(driver(drive))
        self.wait(1.5)


class EndsFrozen(Scene):
    """A scene that ends on a frozen wait."""

    def construct(self):
        self.add(Square(color=BLUE, fill_opacity=1))
        self.wait(0.5)


class LiveEndsFrozen(Scene):
    """A live-drawn scene that ends on a frozen wait."""

    def construct(self):
        self.add(Counted(color=BLUE, fill_opacity=1))
        self.wait(0.5)
