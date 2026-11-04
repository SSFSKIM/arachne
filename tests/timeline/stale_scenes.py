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


class TurnedCamera(ThreeDScene):
    """A static cube, and a driver that turns the camera's azimuth tracker."""

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
        self.add(Cube(side_length=3, fill_opacity=1))
        theta = self.camera.theta_tracker
        self.add(driver(lambda m, dt: theta.increment_value(dt)))
        self.wait(1)


class MovedLight(ThreeDScene):
    """A static cube, and a driver that moves the camera's light source."""

    def construct(self):
        self.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
        self.add(Cube(side_length=3, fill_opacity=1))
        light = self.camera.light_source
        self.add(driver(lambda m, dt: light.shift(RIGHT * 40 * dt)))
        self.wait(1)


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
