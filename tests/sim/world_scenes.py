"""Scenes the worlds' tests execute: worlds of bodies made from ordinary
mobjects, closed and open, entering and leaving the scene.
"""

from __future__ import annotations

from types import SimpleNamespace

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    BoxShape,
    Circle,
    ConstantForce,
    Cube,
    Dot,
    Rectangle,
    Scene,
    ThreeDScene,
    World,
)

#: The tick of every call of the open worlds' callbacks, in order.
CALLS: list[int] = []
#: What an open world's callback read at each call: (tick, the handed
#: state's x of a body of another world, the x of that body's mobject).
READS: list[tuple[int, float, float]] = []


class ClosedThenOpen(Scene):
    """A closed world (a ball dropping onto a floor) added before an open one
    (a point pulled toward the ball), whose force reads the ball through the
    closed world's tick-indexed read.
    """

    def construct(self):
        CALLS.clear()
        closed = World(gravity=DOWN * 9.8)
        closed.body(Rectangle(width=6, height=0.4).shift(DOWN * 2), fixed=True)
        ball = closed.body(Circle(radius=0.3).shift(UP), restitution=0.5)
        pulled = World()
        follower = pulled.body(Dot(LEFT * 3), shape=None, mass=1.0)

        def toward(state):
            CALLS.append(state.tick)
            target = closed.at_tick(state.tick).position(ball)
            return {follower: (target - state.position(follower)) * 4.0}

        pulled.add_force(toward)
        self.add(closed, pulled)
        self.wait(1)
        self.wait(0.5)


class Falling(Scene):
    """A closed world of one point falling, over a wait."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0, velocity=RIGHT)
        self.add(world)
        self.wait(1)


class Counted(Scene):
    """An open world over two plays: its callback is called once per tick."""

    def construct(self):
        CALLS.clear()
        world = World()
        point = world.body(Dot(), shape=None, mass=2.0)

        def push(state):
            CALLS.append(state.tick)
            return {point: RIGHT}

        world.add_force(push)
        self.add(world)
        self.wait(0.5)
        self.wait(0.25)


class TwoAlike(Scene):
    """Two worlds declared alike, side by side in time."""

    def construct(self):
        for _ in range(2):
            world = World(gravity=DOWN * 9.8)
            world.body(Dot(), shape=None, mass=1.0)
            self.add(world)
        self.wait(0.5)


class AwayAndBack(Scene):
    """A world removed at a time that is not a tick (0.51 s, between ticks
    30 and 31) and added again at 1.01 s.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0, velocity=RIGHT)
        self.add(world)
        self.wait(0.51)
        self.remove(world)
        self.wait(0.5)
        self.add(world)
        self.wait(0.5)


class Steered(Scene):
    """A force law declared between plays acts from the next tick."""

    def construct(self):
        world = World()
        point = world.body(Dot(), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.5)
        world.add_force(ConstantForce(point, force=RIGHT * 2))
        self.wait(0.5)


class BodyTooLate(Scene):
    """A body declared into a world in the scene, after its first row."""

    def construct(self):
        world = World()
        world.body(Dot(), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.25)
        world.body(Dot(RIGHT), shape=None, mass=1.0)


class ReadThroughAnObject(Scene):
    """An open world's force that reads another world's body both ways: the
    handed, tick-indexed state, and the body's mobject reached through an
    object (which the check cannot see, and which is not supported).
    """

    def construct(self):
        READS.clear()
        falling = World(gravity=DOWN * 9.8)
        drawn = Dot(UP * 2)
        body = falling.body(drawn, shape=None, mass=1.0, velocity=RIGHT * 3)
        holder = SimpleNamespace(mob=drawn)
        other = World()
        point = other.body(Dot(DOWN * 2), shape=None, mass=1.0)

        def read_both(state):
            handed = falling.at_tick(state.tick).position(body)
            READS.append((state.tick, float(handed[0]), float(holder.mob.get_x())))
            return {point: RIGHT * 0.0}

        other.add_force(read_both)
        self.add(falling, other)
        self.wait(0.5)


class Tumbling(ThreeDScene):
    """A box spinning free in a 3D world, with no gravity."""

    def construct(self):
        world = World(3)
        world.body(
            Cube(side_length=1.0),
            shape=BoxShape(1.0, 1.0, 1.0),
            velocity=RIGHT * 0.5,
            angular_velocity=(0.3, 1.1, 0.7),
        )
        self.add(world)
        self.wait(1)
