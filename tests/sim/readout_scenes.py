"""Scenes the body readouts' tests execute: readouts a world makes of its
bodies' quantities, shown at every frame, and their twins: a plain readout
set by hand, a readout of a world that never entered the scene.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    DR,
    RIGHT,
    UL,
    UP,
    UR,
    Attraction,
    Circle,
    Dot,
    FadeIn,
    Group,
    Indicate,
    Readout,
    Rectangle,
    Scene,
    Square,
    World,
    anchor,
)
from arachne.timeline.record import DEPENDENCIES

#: Each readout of :class:`FallingRead` by its format: what it reads, its
#: part and its scale.
FALLING = {
    "{:.1f}": ("sim.velocity", "y", 1.0),
    "{:.0f} cm/s": ("sim.velocity", "norm", 100.0),
    "{:.6f}": ("sim.position", "y", 1.0),
}
#: What a scene found as it ran, by name.
FOUND: dict[str, object] = {}
#: Where the floor's top is, and the radius of the ball dropped on it.
FLOOR_TOP = -1.0
RADIUS = 0.2


class FallingRead(Scene):
    """A point falling from rest, and three readouts of it added after its
    world: its velocity along y to one decimal, following it (made before
    the world entered the scene); its speed in hundredths of a unit a
    second; its height to six decimals.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        dot = Dot(UP * 2)
        ball = world.body(dot, shape=None, mass=1.0)
        following = world.readout(
            ball, "velocity", part="y", format="{:.1f}", at=anchor(dot), offset=RIGHT
        )
        self.add(world, following)
        speed = world.readout(ball, "velocity", scale=100, format="{:.0f} cm/s")
        height = world.readout(ball, "position", part="y", format="{:.6f}")
        self.add(speed.to_corner(UL), height.to_corner(UR))
        self.wait(1)


class FallingReadByHand(Scene):
    """The twin of :class:`FallingRead`: a plain readout following the
    point, set by the scene's code between plays from the world's state.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        dot = Dot(UP * 2)
        ball = world.body(dot, shape=None, mass=1.0)
        plain = Readout(0, "{:.1f}", at=anchor(dot), offset=RIGHT)
        self.add(world, plain)
        self.wait(0.5)
        plain.set_value(float(world.at_tick(30).velocity(ball)[1]))
        self.wait(0.5)


class ReadWithoutItsWorld(Scene):
    """A readout of a body whose world never enters the scene."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world.readout(ball, "velocity"))
        self.wait(0.5)


class ReadAfterItsWorldLeft(Scene):
    """A readout of a falling point left on screen after its world left the
    scene at 1/2 s (tick 30), over a wait; the scene's code then tries to set
    it by hand.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        reading = world.readout(ball, "velocity", part="y", format="{:.3f}")
        self.add(world, reading)
        self.wait(0.5)
        self.remove(world)
        try:
            reading.set_value(100)
        except TypeError as refused:
            FOUND["set by hand"] = str(refused)
        self.wait(0.5)


class FadedInOverAFall(Scene):
    """A readout of a falling point brought in with FadeIn while the point
    falls: the animation holds its updater.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.2)
        reading = world.readout(ball, "velocity", part="y")
        self.play(FadeIn(reading), run_time=0.5)
        self.wait(0.2)


class FadedInOffTheTickGrid(Scene):
    """As :class:`FadedInOverAFall`, the fade ending the scene at 0.69 s,
    off the tick grid, with the world still in the scene: its stay's last
    row, tick 41, is a frame the fade draws at 60 fps.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.2)
        reading = world.readout(ball, "velocity", part="y")
        self.play(FadeIn(reading), run_time=0.49)


class HeldOverAFallThenLeft(Scene):
    """A readout of a falling point held by hand (its updating suspended)
    while the point falls, until its world is removed at 0.69 s, off the
    tick grid; a wait after.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.2)
        reading = world.readout(ball, "velocity", part="y")
        reading.suspend_updating()
        self.add(reading)
        self.wait(0.49)
        self.remove(world)
        self.wait(0.3)


class HeldPastItsWorldKeptByAnother(Scene):
    """A readout held by hand while a point falls, then its world removed at
    1/2 s while another mobject keeps the world as a capture dependency (as
    an effect keeps its mask or window), with a wait after.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.2)
        reading = world.readout(ball, "velocity", part="y")
        reading.suspend_updating()
        self.add(reading)
        self.wait(0.3)
        keeper = Square(side_length=0.5).to_corner(DR)
        vars(keeper)[DEPENDENCIES] = (world,)
        self.add(keeper)
        self.remove(world)
        self.wait(0.3)


class AddedOverAFall(Scene):
    """The twin of :class:`FadedInOverAFall`: the readout added."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.2)
        self.add(world.readout(ball, "velocity", part="y"))
        self.wait(0.7)


class PushedRead(Scene):
    """An open world: a point pushed along x by a callback force, a readout
    of its velocity added before the world and one added after it.
    """

    def construct(self):
        world = World()
        ball = world.body(Dot(), shape=None, mass=2.0)

        def push(state):
            return {ball: RIGHT * 3}

        world.add_force(push)
        before = world.readout(ball, "velocity", part="x", format="{:.4f}")
        after = world.readout(ball, "velocity", part="x", format="{:.5f}")
        self.add(before.to_corner(UL), world, after.to_corner(UR))
        self.wait(0.5)


class SpunAndDropped(Scene):
    """A square spinning in place at 2 rad/s and a ball dropped on a fixed
    floor, with a readout of the square's angular velocity and one of the
    ball's contacts.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        square = world.body(
            Square(side_length=0.6).shift(UP * 2 + RIGHT * 2),
            angular_velocity=2.0,
            gravity_scale=0.0,
        )
        floor = Rectangle(width=8, height=0.4).shift(UP * (FLOOR_TOP - 0.2))
        world.body(floor, fixed=True, restitution=0.5)
        ball = world.body(
            Circle(radius=RADIUS).move_to(UP * (FLOOR_TOP + RADIUS + 0.817)),
            restitution=0.5,
        )
        self.add(
            world,
            world.readout(square, "angular_velocity", format="{:.3f} rad/s"),
            world.readout(ball, "contact_count", format="{:.0f}").to_corner(UL),
        )
        self.wait(0.75)


class HeldWhileRead(Scene):
    """A point read by a readout already in the scene as its world enters by
    a fade, which holds the world's drawing: the point falls in a closed
    world; with ``open``, a callback pushes it along x. With ``grouped``,
    the fade is of a group holding the world and a square.
    """

    open = False
    grouped = False

    def construct(self):
        world = World(gravity=None if self.open else DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        if self.open:

            def push(state):
                return {ball: RIGHT * 3}

            world.add_force(push)
        part = "x" if self.open else "y"
        reading = world.readout(ball, "position", part=part, format="{:.6f}")
        self.add(reading.to_corner(UL))
        shown = (
            Group(world, Square(side_length=0.5).to_corner(DR))
            if self.grouped
            else world
        )
        self.play(FadeIn(shown), run_time=0.5)
        self.wait(0.25)


class HeldInAGroupWhileRead(HeldWhileRead):
    grouped = True


class OpenHeldWhileRead(HeldWhileRead):
    open = True


class OpenHeldInAGroupWhileRead(HeldWhileRead):
    open = True
    grouped = True


class CollapsingRead(Scene):
    """Two point masses declared at one place, attracting each other by the
    inverse square, and a readout of one's speed: the first step divides by
    a zero distance, and every number after it is not a number.
    """

    def construct(self):
        world = World()
        a = world.body(Dot(), shape=None, mass=1.0)
        b = world.body(Dot(), shape=None, mass=1.0)
        world.add_force(Attraction(a, b, strength=0.01))
        self.add(world, world.readout(a, "velocity").to_corner(UL))
        self.wait(0.5)


class FadedInWithItsReadout(Scene):
    """A world at rest faded in, its readout added with it as the fade
    begins: nothing moves, and the fade holds the world's drawing.
    """

    def construct(self):
        world = World()
        ball = world.body(Dot(UP), shape=None, mass=1.0)
        reading = world.readout(ball, "position", part="y")
        self.add(reading.to_corner(UL))
        self.play(FadeIn(world), run_time=0.5)
        self.wait(0.25)


class IndicatedOverAFall(Scene):
    """A readout added with its falling point's world, then held by an
    animation of its own (Indicate) while the point falls.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        ball = world.body(Dot(UP * 2), shape=None, mass=1.0)
        reading = world.readout(ball, "velocity", part="y")
        self.add(world, reading.to_corner(UL))
        self.wait(0.2)
        self.play(Indicate(reading), run_time=0.5)
        self.wait(0.2)
