"""Spike scenes (do not merge): a closed world of one falling body, removed
at 1/2 s after a wait at 15 fps, with scene time after the removal.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    UR,
    DecimalNumber,
    Dot,
    Readout,
    Scene,
    Square,
    ValueTracker,
    World,
    linear,
)


def _world():
    world = World(gravity=DOWN * 9.8)
    dot = Dot(UP * 2, radius=0.15)
    world.body(dot, shape=None, mass=1.0, velocity=RIGHT * 3)
    return world, dot


class Removed(Scene):
    """The world removed, a wait after: the body leaves with it."""

    def construct(self):
        world, _ = _world()
        self.add(world)
        self.wait(0.5)
        self.remove(world)
        self.wait(0.5)


class RemovedTicked(Scene):
    """As :class:`Removed`, with a scene updater reading ``dt`` throughout."""

    def construct(self):
        self.elapsed = 0.0

        def count(dt):
            self.elapsed += dt

        self.add_updater(count)
        world, _ = _world()
        self.add(world)
        self.wait(0.5)
        self.remove(world)
        self.wait(0.5)


class Returned(Scene):
    """The world removed and its body added back alone: a held wait after."""

    def construct(self):
        world, dot = _world()
        self.add(world)
        self.wait(0.5)
        self.remove(world)
        self.add(dot)
        self.wait(0.5)


class ReturnedTicked(Scene):
    """As :class:`Returned`, with a scene updater reading ``dt`` throughout."""

    def construct(self):
        self.elapsed = 0.0

        def count(dt):
            self.elapsed += dt

        self.add_updater(count)
        world, dot = _world()
        self.add(world)
        self.wait(0.5)
        self.remove(world)
        self.add(dot)
        self.wait(0.5)


class RemovedThenPlayed(Scene):
    """The world removed, then a play of something else."""

    def construct(self):
        world, _ = _world()
        square = Square(side_length=0.5).shift(RIGHT * 4)
        self.add(world, square)
        self.wait(0.5)
        self.remove(world)
        self.play(square.animate.shift(DOWN), run_time=0.5)


class ReturnedThenPlayed(Scene):
    """The world removed, its body added back and moved by a play."""

    def construct(self):
        world, dot = _world()
        self.add(world)
        self.wait(0.5)
        self.remove(world)
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)


class RemovedRead(Scene):
    """As :class:`Removed`, with a readout of the body's height kept by an
    updater, staying after the world leaves.
    """

    def construct(self):
        world, dot = _world()
        reading = Readout(0, "{:.2f}").to_corner(UL)
        reading.add_updater(lambda m: m.set_value(float(dot.get_y())))
        self.add(world, reading)
        self.wait(0.5)
        self.remove(world)
        self.wait(0.5)


class ReturnedThenState(Scene):
    """The body added back, then a state-function play: a marker kept by a
    dt-free updater at a tracker's value.
    """

    def construct(self):
        world, dot = _world()
        tracker = ValueTracker(0)
        marker = Dot(LEFT * 4).add_updater(lambda m: m.set_y(tracker.get_value()))
        self.add(world, marker)
        self.wait(0.5)
        self.remove(world)
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=0.5, rate_func=linear)


class ReturnedThenCounted(Scene):
    """As :class:`ReturnedThenState`, with a frame hook counting frames into
    a number: a play the probe may demote.
    """

    def construct(self):
        world, dot = _world()
        self.count = DecimalNumber(0, num_decimal_places=0).to_corner(UR)
        tracker = ValueTracker(0)
        marker = Dot(LEFT * 4).add_updater(lambda m: m.set_y(tracker.get_value()))
        self.add(world, marker, self.count)
        self.wait(0.5)
        self.remove(world)
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=0.5, rate_func=linear)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "count"):
            self.count.set_value(self.count.number + 1)


SCENES = (
    "Removed",
    "RemovedTicked",
    "Returned",
    "ReturnedTicked",
    "RemovedThenPlayed",
    "ReturnedThenPlayed",
    "RemovedRead",
    "ReturnedThenState",
    "ReturnedThenCounted",
)
