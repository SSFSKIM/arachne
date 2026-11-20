"""Scenes the simulation findings' tests execute: worlds whose solver finds
a body gone non-finite, a body above its world's speed limit, a body that
passed through a thin wall, and each one's twin that finds nothing; bodies
dropped on a floor whose quantities are read; a body shown after its world
left the scene.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Attraction,
    Circle,
    ConstantForce,
    Dot,
    Rectangle,
    Scene,
    World,
)

#: Where the floor's top is, and the radius of the balls dropped on it.
FLOOR_TOP = -1.0
RADIUS = 0.2
#: How far each ball falls to the floor: under 9.8 u/s², a fall of 0.817 u
#: takes 0.4083 s and one of 1.7154 u 0.5917 s, both inside a tick.
DROPS = (0.817, 1.7154)


class Collapsing(Scene):
    """Two discs that attract each other by the inverse square, declared at
    one place: the first step divides by a zero distance, and the solver
    stops both where they were.
    """

    apart = 0.0

    def construct(self):
        world = World()
        a = world.body(Circle(radius=0.2).shift(LEFT * self.apart / 2))
        b = world.body(Circle(radius=0.2).shift(RIGHT * self.apart / 2))
        world.add_force(Attraction(a, b, strength=0.01))
        self.add(world)
        self.wait(0.5)


class Attracting(Collapsing):
    """The twin of :class:`Collapsing`: the discs 2 apart."""

    apart = 2.0


class Speeding(Scene):
    """A point pushed at 10 u/s each second in a world limited to 5.08 u/s:
    5.0 at tick 30, 5.1667 at tick 31.
    """

    limit: float | None = 5.08

    def construct(self):
        world = World(speed_limit=self.limit)
        point = world.body(Dot(), shape=None, mass=1.0)
        world.add_force(ConstantForce(point, force=RIGHT * 10))
        self.add(world)
        self.wait(1)


class Pushed(Speeding):
    """The twin of :class:`Speeding`: the default limit."""

    limit = None


class ThroughTheWall(Scene):
    """A disc 0.05 in radius at 40 u/s toward a fixed wall 0.04 thick, the
    solver's continuous detection off: it crosses the wall in one step.
    """

    continuous = False

    def construct(self):
        world = World(continuous_collision=self.continuous)
        world.body(Rectangle(width=0.04, height=2), fixed=True)
        world.body(Circle(radius=0.05).shift(LEFT * 0.4), velocity=RIGHT * 40)
        self.add(world)
        self.wait(0.5)


class StoppedByTheWall(ThroughTheWall):
    """The twin of :class:`ThroughTheWall`: continuous detection on."""

    continuous = True


class Dropped(Scene):
    """Two balls dropped on a floor, half bouncing: each first meets it
    inside a tick, at 0.4083 s and 0.5917 s.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        floor = Rectangle(width=8, height=0.4).shift(UP * (FLOOR_TOP - 0.2))
        world.body(floor, fixed=True, restitution=0.5)
        for x, drop in zip((-1, 1), DROPS, strict=True):
            ball = Circle(radius=RADIUS).move_to(
                RIGHT * x + UP * (FLOOR_TOP + RADIUS + drop)
            )
            world.body(ball, restitution=0.5)
        self.add(world)
        self.wait(0.75)


class ShownAfterItsWorld(Scene):
    """A falling point's mobject shown on its own after its world left the
    scene: no bake holds its quantities then.
    """

    leaves = True

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        dot = Dot(UP * 2)
        world.body(dot, shape=None, mass=1.0)
        self.add(world)
        self.wait(0.5)
        if self.leaves:
            self.remove(world)
            self.add(dot)
        self.wait(0.5)


class ShownWithItsWorld(ShownAfterItsWorld):
    """The twin of :class:`ShownAfterItsWorld`: the world stays."""

    leaves = False
