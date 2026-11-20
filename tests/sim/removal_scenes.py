"""Spike scenes (do not merge): a closed world of one falling body, removed
at 1/2 s after a wait at 15 fps, with scene time after the removal.
"""

from __future__ import annotations

from arachne import DOWN, RIGHT, UL, UP, Dot, Readout, Scene, Square, World


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


SCENES = (
    "Removed",
    "RemovedTicked",
    "Returned",
    "ReturnedTicked",
    "RemovedThenPlayed",
    "ReturnedThenPlayed",
    "RemovedRead",
)
