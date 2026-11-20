"""Scenes the worlds' tests execute: worlds of bodies made from ordinary
mobjects, closed and open, entering and leaving the scene.
"""

from __future__ import annotations

import sys
import types
import weakref
from types import SimpleNamespace

from arachne import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    BoxShape,
    Circle,
    ConstantForce,
    Cube,
    Dot,
    FadeIn,
    FadeOut,
    Gravity,
    Group,
    Motor,
    PinJoint,
    Rectangle,
    Scene,
    SlideJoint,
    Square,
    ThreeDScene,
    World,
)

#: The tick of every call of the open worlds' callbacks, in order.
CALLS: list[int] = []
#: What an open world's callback read at each call: (tick, the handed
#: state's x of a body of another world, the x of that body's mobject).
READS: list[tuple[int, float, float]] = []
#: What a scene found as it ran, by name.
FOUND: dict[str, object] = {}
#: The lists a callback closes over, which a test changes after the run.
HELD: list[list[int]] = []
#: Shared with the tests across the sessions' own copies of this file: a
#: session loads it as a module of its own.
SHARED = sys.modules.setdefault("_world_scenes_shared", types.ModuleType("shared"))
#: What happened, in order: ("call", the tick a callback was handed) and
#: ("tick", the tick a scene updater ran at).
EVENTS: list[tuple[str, int]] = []


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
            target = state.position(ball)
            return {follower: (target - state.position(follower)) * 4.0}

        pulled.add_force(toward)
        self.add(closed, pulled)
        self.wait(1.5)
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
            handed = state.position(body)
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


class Blinked(Scene):
    """A closed world removed and added again with no scene time between,
    then brought to the front and to the back of the scene mid-stay.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world, Square())
        self.wait(0.5)
        self.remove(world)
        self.add(world)
        self.wait(0.25)
        self.bring_to_front(world)
        self.bring_to_back(world)
        self.wait(0.25)


class LeftBetweenFrames(Scene):
    """A closed world removed by an updater at tick 19 (a tick a 15 fps
    render reads no frame at), the play going on.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0, velocity=RIGHT)
        self.add(world)
        ticks = [0]

        def remove_at_tick_19(dt):
            if dt:
                ticks[0] += 1
                if ticks[0] == 19:
                    self.remove(world)

        self.add_updater(remove_at_tick_19)
        self.wait(0.5)


class ReadWhileAway(Scene):
    """A world read while it is away, and past what it computed."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.5)
        try:
            world.at_tick(60)
        except LookupError as error:
            FOUND["ahead"] = str(error)
        self.remove(world)
        self.wait(0.5)
        try:
            world.at_tick(30)
        except LookupError as error:
            FOUND["away"] = str(error)


class Grouped(Scene):
    """A world inside a group: the group added, the world taken out of the
    group's place (the group dissolved), the group added again, then cleared.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0)
        group = Group(world, Square())
        self.add(group)
        self.wait(0.5)
        self.remove(world)
        self.wait(0.5)
        self.add(group)
        self.wait(0.5)
        self.clear()
        self.wait(0.5)


class Replaced(Scene):
    """A world replaced by a square, and the square by the world again."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0)
        stand_in = Square()
        self.add(world)
        self.wait(0.5)
        self.replace(world, stand_in)
        self.wait(0.5)
        self.replace(stand_in, world)
        self.wait(0.5)


class Assigned(Scene):
    """A world taken out by assigning the scene's mobjects between plays."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.5)
        self.mobjects = []
        self.wait(0.5)


class AssignedDuringAPlay(Scene):
    """A world taken out by assigning the scene's mobjects in an updater:
    the instant it left is not known.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP * 2), shape=None, mass=1.0)
        self.add(world)

        def drop(dt):
            if dt and self.mobjects:
                self.mobjects = []

        self.add_updater(drop)
        self.wait(0.5)


class FadedIn(Scene):
    """A closed and an open world faded in and out: the animations' copies
    of them run their updaters.
    """

    def construct(self):
        CALLS.clear()
        closed = World(gravity=DOWN * 9.8)
        closed.body(Circle(radius=0.3).shift(UP), restitution=0.5)
        pulled = World()
        point = pulled.body(Dot(LEFT), shape=None, mass=1.0)

        def push(state):
            CALLS.append(state.tick)
            return {point: RIGHT}

        pulled.add_force(push)
        self.play(FadeIn(closed), FadeIn(pulled), run_time=0.5)
        self.wait(0.5)


class Resumed(Scene):
    """An open world whose updating is suspended and resumed between plays,
    and mid-play from an updater.
    """

    def construct(self):
        CALLS.clear()
        world = World()
        point = world.body(Dot(), shape=None, mass=1.0)

        def push(state):
            CALLS.append(state.tick)
            return {point: RIGHT}

        world.add_force(push)
        self.add(world)
        self.wait(0.25)
        world.suspend_updating()
        world.resume_updating()
        self.wait(0.25)


class SteeredMidPlay(Scene):
    """A force law added to a closed world by a ``dt`` updater at tick 19 of
    a play that animates something else too.
    """

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        body = world.body(Dot(UP * 2), shape=None, mass=1.0)
        marker = Square()
        self.add(world, marker)
        ticks = [0]

        def steer(dt):
            if dt:
                ticks[0] += 1
                if ticks[0] == 19:
                    world.add_force(ConstantForce(body, force=RIGHT * 3))

        self.add_updater(steer)
        self.play(marker.animate.shift(RIGHT), run_time=1)


class Hinged(Scene):
    """Two alike worlds of a pendulum on a limited pin and a slider driven by
    a motor: one stays in the scene, the other is away from 0.5 s to 1 s.
    """

    def construct(self):
        worlds = []
        for _ in range(2):
            world = World(gravity=DOWN * 9.8)
            anchor = world.body(Dot(ORIGIN), shape=None, fixed=True)
            bob = world.body(Circle(radius=0.2).shift(RIGHT), restitution=0.0)
            world.add_joint(PinJoint(anchor, bob, at=ORIGIN, limits=(-0.6, 0.6)))
            rail = world.body(Square(0.4).shift(DOWN * 2), fixed=True)
            slider = world.body(Square(0.3).shift(DOWN * 2), gravity_scale=0.0)
            world.add_joint(
                SlideJoint(
                    rail,
                    slider,
                    axis=RIGHT,
                    limits=(-1.0, 1.0),
                    motor=Motor(position=0.8, stiffness=20.0, damping=4.0),
                )
            )
            worlds.append(world)
        steady, away = worlds
        self.add(steady, away)
        self.wait(0.5)
        self.remove(away)
        self.wait(0.5)
        self.add(away)
        self.wait(0.5)


class GravityFirst(Scene):
    def construct(self):
        world = World()
        world.add_force(Gravity(DOWN * 9.8))
        world.body(Dot(), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.1)


class BodyFirst(Scene):
    def construct(self):
        world = World()
        world.body(Dot(), shape=None, mass=1.0)
        world.add_force(Gravity(DOWN * 9.8))
        self.add(world)
        self.wait(0.1)


class PushedOutOfPlane(Scene):
    """An open 2D world whose callback pushes out of the plane."""

    pushed = (1.0, 0.0, 1.0)

    def construct(self):
        world = World()
        point = world.body(Dot(), shape=None, mass=1.0)
        pushed = self.pushed

        def push(state):
            return {point: pushed}

        world.add_force(push)
        self.add(world)
        self.wait(0.1)


class PushedNotANumber(PushedOutOfPlane):
    pushed = (1.0, 0.0, float("nan"))


class Remembering(Scene):
    """An open world whose callback keeps what it read in a list it closes
    over.
    """

    def construct(self):
        HELD.clear()
        world = World()
        point = world.body(Dot(), shape=None, mass=1.0)
        seen: list[int] = []
        HELD.append(seen)

        def push(state):
            seen.append(state.tick)
            return {point: RIGHT}

        world.add_force(push)
        self.add(world)
        self.wait(0.25)
        self.wait(0.25)


class Overridden(Circle):
    """A circle whose family walk hides its points."""

    def family_members_with_points(self):
        return []


class DrawnByAnOverride(Scene):
    """A body drawn by a mobject whose class overrides the family walk."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Overridden(radius=0.3).shift(UP * 2), shape=None, mass=1.0)
        self.add(world)
        self.wait(0.5)


def _logged_open_world():
    """An open world whose callback logs each call in EVENTS."""
    world = World()
    point = world.body(Dot(LEFT), shape=None, mass=1.0)

    def push(state):
        EVENTS.append(("call", state.tick))
        return {point: RIGHT}

    world.add_force(push)
    return world


class _Logged(Scene):
    def log_ticks(self):
        EVENTS.clear()

        def tick(dt):
            if dt:
                EVENTS.append(("tick", int(self.time * 60)))

        self.add_updater(tick)


class FadedInLast(_Logged):
    """An open world faded in by the scene's last play."""

    def construct(self):
        self.log_ticks()
        self.play(FadeIn(_logged_open_world()), run_time=0.5)


class FadedBeside(Scene):
    """Two alike open worlds: one faded in, one added, then a wait."""

    def construct(self):
        faded, added = World(), World()
        for world in (faded, added):
            point = world.body(Dot(LEFT), shape=None, mass=1.0)
            world.add_force(_pushing(point))
        self.add(added)
        self.play(FadeIn(faded), run_time=0.5)
        self.wait(0.5)


def _pushing(point):
    def push(state):
        return {point: RIGHT * (1 + state.tick / 60)}

    return push


class FadedOut(_Logged):
    """An open world in the scene for a wait, then faded out."""

    def construct(self):
        self.log_ticks()
        world = _logged_open_world()
        self.add(world)
        self.wait(0.5)
        self.play(FadeOut(world), run_time=0.5)


class Dropped(Scene):
    """A world the scene enters, whose session is closed unread."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        point = world.body(Dot(UP), shape=None, mass=1.0)

        def push(state):
            return {point: RIGHT}

        world.add_force(push)
        SHARED.world = weakref.ref(world)
        self.add(world)
        self.wait(0.25)


class RaisesAfterEntering(Dropped):
    def construct(self):
        super().construct()
        raise RuntimeError("the scene's own error")


class Resting(Scene):
    """A ball resting on a floor, away and back."""

    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Rectangle(width=4, height=0.4).shift(DOWN * 0.2), fixed=True)
        world.body(Circle(radius=0.3).shift(UP * 0.3))
        self.add(world)
        self.wait(0.5)
        self.remove(world)
        self.wait(0.25)
        self.add(world)
        self.wait(0.5)


class CopiedOpen(Scene):
    def construct(self):
        self.add(_logged_open_world().copy())
        self.wait(0.1)


class CopiedClosed(Scene):
    def construct(self):
        world = World(gravity=DOWN * 9.8)
        world.body(Dot(UP), shape=None, mass=1.0)
        self.add(world.copy())
        self.wait(0.5)


class Heavier(Scene):
    """Falling, under another gravity: the same start, another world."""

    def construct(self):
        world = World(gravity=DOWN * 19.6)
        world.body(Dot(UP * 2), shape=None, mass=1.0, velocity=RIGHT)
        self.add(world)
        self.wait(1)


class FadedInGrouped(_Logged):
    """An open world faded in with a label, in one group, by the last play."""

    def construct(self):
        self.log_ticks()
        self.play(FadeIn(Group(_logged_open_world(), Dot(UP))), run_time=0.5)


class FadedInGroupedQuiet(Scene):
    """The grouped fade with no other updater that reads dt."""

    def construct(self):
        EVENTS.clear()
        self.play(FadeIn(Group(_logged_open_world(), Dot(UP))), run_time=0.5)


class _RemovedByAnUpdater(Scene):
    """An open world removed at tick 19 (a tick no 15 fps frame reads) by a
    dot's dt updater, the dot in the scene before the world or after it.
    """

    dot_first = True

    def construct(self):
        EVENTS.clear()
        world = _logged_open_world()
        dot = Dot(DOWN)
        ticks = [0]

        def remove_at_tick_19(mob, dt):
            if dt:
                ticks[0] += 1
                if ticks[0] == 19:
                    self.remove(world)

        dot.add_updater(remove_at_tick_19)
        self.add(*((dot, world) if self.dot_first else (world, dot)))
        self.wait(0.5)


class RemovedBeforeItsUpdater(_RemovedByAnUpdater):
    dot_first = True


class RemovedAfterItsUpdater(_RemovedByAnUpdater):
    dot_first = False


class GroupedBeside(Scene):
    """Two alike worlds of each kind: one of each faded in with a label in
    one group, the other added, then a wait.
    """

    def construct(self):
        faded, added = World(), World()
        for world in (faded, added):
            point = world.body(Dot(LEFT), shape=None, mass=1.0)
            world.add_force(_pushing(point))
        falling, dropped = World(gravity=DOWN * 9.8), World(gravity=DOWN * 9.8)
        for world in (falling, dropped):
            world.body(Dot(UP), shape=None, mass=1.0, velocity=RIGHT)
        self.add(added, dropped)
        self.play(FadeIn(Group(faded, falling, Dot(UP * 2))), run_time=0.5)
        self.wait(0.5)


class GroupedClosedLast(Scene):
    """A closed world faded in with a label by the scene's last play, an
    alike one added beside it.
    """

    def construct(self):
        faded, added = World(gravity=DOWN * 9.8), World(gravity=DOWN * 9.8)
        for world in (faded, added):
            world.body(Dot(UP), shape=None, mass=1.0, velocity=RIGHT)
        self.add(added)
        self.play(FadeIn(Group(faded, Dot(UP * 2))), run_time=0.5)


class SuspendedByHand(Scene):
    """An open world in a group whose updating the scene suspends, then a
    wait.
    """

    def construct(self):
        group = Group(_logged_open_world(), Dot(UP))
        self.add(group)
        group.suspend_updating()
        self.wait(0.5)


class WorldSuspendedByHand(_Logged):
    """An open world whose own updating the scene suspends, then a wait."""

    def construct(self):
        self.log_ticks()
        world = _logged_open_world()
        self.add(Group(world, Dot(UP)))
        world.suspend_updating()
        self.wait(0.5)
