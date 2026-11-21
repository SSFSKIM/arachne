"""Scenes whose bakes are pinned byte for byte (:mod:`.determinism`).

:class:`LiteralWorlds` holds a 2D world and a 3D world, each with contacts,
joints and force laws, over 20 s, every number a literal: no mathematical
function computes any value the worlds are given, and every drawing is a
polygon, whose points are sums and products of those literals. (A circle's
or a cube's points go through sines and cosines, whose last bits a numpy
build may choose by the CPU, and its centre would carry them into a body's
position.)

:class:`ComputedWorlds` is the same two worlds with their starting values
computed through Python's ``math`` and numpy's functions, and an open world
whose callback force computes what it pushes the same way. Two machines
that compute those functions alike declare the same bytes; two that do not
are told apart by the declaration's digest before the motion is compared.
"""

from __future__ import annotations

import math

import numpy as np

from arachne import (
    BallJoint,
    BallShape,
    BoxShape,
    ConstantForce,
    LinearDrag,
    Motor,
    PinJoint,
    QuadraticDrag,
    Rectangle,
    SlideJoint,
    Spring,
    Square,
    ThreeDScene,
    World,
)

#: How long each scene's worlds are in it, in seconds.
DURATION = 20

#: The starting values of the two closed worlds, as literals.
LITERAL: dict[str, float | tuple[float, ...]] = {
    "ball_a": (-1.0, 0.5),
    "ball_a_velocity": (0.8, 0.0),
    "ball_b": (0.5, 1.5),
    "ball_b_velocity": (-0.3, 0.0),
    "bob": (-1.5, 1.5),
    "bob_spin": 0.5,
    "slider_push": (0.6, 0.0),
    "box_a": (0.3, 0.6, 1.0),
    "box_a_spin": (0.5, -0.3, 0.8),
    "box_b": (0.7, 0.4, 2.2),
    "box_b_velocity": (0.0, 0.2, 0.0),
    "link_1": (-1.0, 1.0, 1.5),
    "link_2": (-0.4, 1.0, 1.5),
    "link_2_velocity": (0.0, 0.5, 0.0),
    "paddle_spin": (0.0, 0.0, 1.0),
}

#: Each world's declaration bytes as its solver holds them when the scene
#: ends, by world ("plane", "space", "open"): the bytes its occurrence's
#: declaration digest covers, the open world's stored pushes at every tick
#: among them.
DECLARED: dict[str, bytes] = {}


def computed() -> dict[str, float | tuple[float, ...]]:
    """The same starting values as :data:`LITERAL`, near them, each computed
    through a function of ``math`` or of numpy.
    """
    return {
        "ball_a": (-math.cos(0.1), float(np.sin(0.52))),
        "ball_a_velocity": (0.8 * math.exp(-0.01), 0.0),
        "ball_b": (float(np.cbrt(0.126)), 1.5 * float(np.cos(0.05))),
        "ball_b_velocity": (-float(np.tanh(0.31)), 0.0),
        "bob": (-2.5 + float(np.cos(0.02)), 1.5 + math.sin(0.02)),
        "bob_spin": math.atan(0.55),
        "slider_push": (float(np.log1p(0.8)), 0.0),
        "box_a": (
            float(np.arctan2(0.3, 1.0)),
            math.log(1.82),
            float(np.exp(0.0033)),
        ),
        "box_a_spin": (float(np.sin(0.52)), -math.tan(0.29), float(np.exp(-0.22))),
        "box_b": (math.hypot(0.6, 0.36), float(np.exp(-0.92)), 2.2 * math.cos(0.03)),
        "box_b_velocity": (0.0, float(np.sinh(0.2)), 0.0),
        "link_1": (-float(np.cos(0.01)), 1.0, 1.5 + math.sin(0.001)),
        "link_2": (-0.4 * math.exp(0.01), 1.0, 1.5),
        "link_2_velocity": (0.0, float(np.sin(0.5236)), 0.0),
        "paddle_spin": (0.0, 0.0, math.erf(1.2)),
    }


def keep_declarations(**worlds: World) -> None:
    """Keep each world's declaration bytes in :data:`DECLARED`, as its
    solver holds them now.
    """
    DECLARED.clear()
    for name, world in worlds.items():
        DECLARED[name] = bytes(world._crate_now().declaration_bytes())


def _at(point) -> np.ndarray:
    """A scene point from a world's two or three numbers."""
    return np.array([*point, 0.0][:3], dtype=float)


def plane_world(values) -> World:
    """A 2D world: two balls tied by a spring, dropped on a floor; a bob on a
    pin with limits, under a drag; a slider on a rail, driven by a motor and
    pushed against it.
    """
    world = World(gravity=(0.0, -9.8, 0.0))
    world.body(
        Rectangle(width=8.0, height=0.4).shift(_at((0.5, -3.0))),
        shape=BoxShape(8.0, 0.4),
        fixed=True,
        friction=0.6,
    )
    ball_a = world.body(
        Square(0.5).shift(_at(values["ball_a"])),
        shape=BallShape(0.25),
        velocity=values["ball_a_velocity"],
        friction=0.3,
        restitution=0.5,
    )
    ball_b = world.body(
        Square(0.5).shift(_at(values["ball_b"])),
        shape=BallShape(0.25),
        velocity=values["ball_b_velocity"],
        friction=0.3,
        restitution=0.5,
    )
    world.add_force(Spring(ball_a, ball_b, stiffness=2.0, rest=1.2, damping=0.1))
    pivot = world.body(Square(0.1).shift(_at((-2.5, 1.5))), shape=None, fixed=True)
    bob = world.body(
        Square(0.3).shift(_at(values["bob"])),
        shape=BoxShape(0.3, 0.3),
        angular_velocity=values["bob_spin"],
        friction=0.2,
    )
    world.add_joint(PinJoint(pivot, bob, at=(-2.5, 1.5, 0.0), limits=(-1.0, 1.0)))
    world.add_force(LinearDrag(bob, coefficient=0.05))
    rail = world.body(
        Square(0.4).shift(_at((2.5, 1.0))), shape=BoxShape(0.4, 0.4), fixed=True
    )
    slider = world.body(
        Square(0.3).shift(_at((2.5, 1.0))),
        shape=BoxShape(0.3, 0.3),
        gravity_scale=0.0,
    )
    world.add_joint(
        SlideJoint(
            rail,
            slider,
            axis=(1.0, 0.0, 0.0),
            limits=(-1.0, 1.0),
            motor=Motor(position=0.8, stiffness=20.0, damping=4.0),
        )
    )
    world.add_force(ConstantForce(slider, force=values["slider_push"]))
    return world


def space_world(values) -> World:
    """A 3D world: two boxes tied by a spring, dropped on a floor, one under
    a quadratic drag; a chain of two links on ball joints, under a drag; a
    paddle on a pin about z, driven by a motor and turned by a torque.
    """
    world = World(3, gravity=(0.0, 0.0, -9.8))
    world.body(
        Square(6.0).shift(_at((0.5, 0.5, -2.0))),
        shape=BoxShape(6.0, 6.0, 0.4),
        fixed=True,
        friction=0.5,
    )
    box_a = world.body(
        Square(0.6).shift(_at(values["box_a"])),
        shape=BoxShape(0.6, 0.6, 0.6),
        angular_velocity=values["box_a_spin"],
        friction=0.4,
        restitution=0.2,
    )
    box_b = world.body(
        Square(0.6).shift(_at(values["box_b"])),
        shape=BoxShape(0.6, 0.6, 0.6),
        velocity=values["box_b_velocity"],
        friction=0.4,
        restitution=0.2,
    )
    world.add_force(Spring(box_a, box_b, stiffness=1.0, rest=0.8, damping=0.2))
    world.add_force(QuadraticDrag(box_a, coefficient=0.02))
    anchor = world.body(
        Square(0.1).shift(_at((-1.5, 1.0, 1.5))), shape=None, fixed=True
    )
    link_1 = world.body(
        Square(0.3).shift(_at(values["link_1"])), shape=BoxShape(0.5, 0.2, 0.2)
    )
    link_2 = world.body(
        Square(0.3).shift(_at(values["link_2"])),
        shape=BoxShape(0.5, 0.2, 0.2),
        velocity=values["link_2_velocity"],
    )
    world.add_joint(BallJoint(anchor, link_1, at=(-1.5, 1.0, 1.5)))
    world.add_joint(BallJoint(link_1, link_2, at=(-0.7, 1.0, 1.5)))
    world.add_force(LinearDrag(link_2, coefficient=0.05))
    hinge = world.body(Square(0.1).shift(_at((1.5, -1.5, 0.5))), shape=None, fixed=True)
    paddle = world.body(
        Square(0.4).shift(_at((1.5, -1.5, 0.5))),
        shape=BoxShape(0.8, 0.1, 0.4),
        angular_velocity=values["paddle_spin"],
    )
    world.add_joint(
        PinJoint(
            hinge,
            paddle,
            at=(1.5, -1.5, 0.5),
            axis=(0.0, 0.0, 1.0),
            motor=Motor(velocity=1.0, damping=2.0),
        )
    )
    world.add_force(ConstantForce(paddle, torque=(0.0, 0.0, 0.05)))
    return world


def open_world() -> World:
    """A 2D open world: two points tied by a spring, each pulled toward the
    origin by a callback force whose strength and swirl are computed through
    ``math`` and numpy from the state it is handed, and damped by it.
    """
    world = World()
    a = world.body(
        Square(0.2).shift(_at((1.5 * math.cos(0.7), 1.5 * float(np.sin(0.7))))),
        shape=None,
        mass=1.0,
        velocity=(-1.2 * float(np.sin(0.7)), 1.2 * math.cos(0.7)),
    )
    b = world.body(
        Square(0.2).shift(_at((-2.0 * float(np.cos(0.3)), -2.0 * math.sin(0.3)))),
        shape=None,
        mass=1.5,
        velocity=(0.9 * math.sin(0.3), -0.9 * float(np.cos(0.3))),
    )
    world.add_force(Spring(a, b, stiffness=0.5, rest=1.5))

    def swirl(state):
        pushed = {}
        for body in (a, b):
            x, y = (float(c) for c in state.position(body)[:2])
            vx, vy = (float(c) for c in state.velocity(body)[:2])
            r = math.hypot(x, y)
            angle = float(np.arctan2(y, x))
            strength = 4.0 * (1.0 + 0.2 * float(np.sin(2.0 * angle)) * math.exp(-r))
            turn = 0.1 * math.tanh(r)
            force = (
                -strength * x - 0.05 * vx - turn * float(np.sin(angle)),
                -strength * y
                - 0.05 * vy
                + turn * math.cos(angle)
                + 0.01 * float(np.log1p(r)),
            )
            pushed[body] = force
        return pushed

    world.add_force(swirl)
    return world


class LiteralWorlds(ThreeDScene):
    """The 2D and the 3D world from literal numbers, 20 s."""

    def construct(self):
        plane, space = plane_world(LITERAL), space_world(LITERAL)
        self.add(plane, space)
        self.wait(DURATION)
        keep_declarations(plane=plane, space=space)


class ComputedWorlds(ThreeDScene):
    """The 2D and the 3D world from computed starting values, and the open
    world, 20 s.
    """

    def construct(self):
        values = computed()
        plane, space, pulled = plane_world(values), space_world(values), open_world()
        self.add(plane, space, pulled)
        self.wait(DURATION)
        keep_declarations(plane=plane, space=space, open=pulled)
