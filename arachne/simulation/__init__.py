"""Simulated motion: bodies, joints and forces that a physics engine steps on
the scene's 60 Hz tick, their result kept in the record as a bake
(:mod:`arachne.timeline.bake`).

A scene declares a :class:`World` (:mod:`.world`), its bodies (each drawn by
a mobject, read by a :class:`Body` handle), its joints and its force laws
(:mod:`.declaration`), and adds it to the scene: the world moves its bodies'
mobjects there, the same at every frame rate, and each stay of it in the
scene is one occurrence of the record's bakes (:mod:`.stay`). A closed world
runs no scene code as it steps; a callback force makes it open, and reads the
:class:`WorldState` it is handed (:mod:`.inputs`). The solver is the engine's
``sim`` extra, imported only when a world is made.

Its declared quantities (:mod:`arachne.quantities`) are named under the
prefix ``sim.``, which this package owns, in the units it registers here:
``u`` (a scene unit), ``u/s``, ``rad``, ``rad/s`` and ``s``. A
:class:`BodyReadout` (:meth:`World.readout`, :mod:`.readout`) shows one of a
body's quantities at every frame, and declares which.
"""

from __future__ import annotations

from arachne import quantities

from .declaration import (
    Attraction,
    BallJoint,
    BallShape,
    BoxShape,
    CapsuleShape,
    ConstantForce,
    Gravity,
    HullShape,
    LinearDrag,
    Motor,
    PinJoint,
    Push,
    QuadraticDrag,
    RopeJoint,
    SlideJoint,
    Spring,
    SpringJoint,
    WeldJoint,
)
from .readout import BodyReadout
from .world import Body, World, WorldState

__all__ = [
    "Attraction",
    "BallJoint",
    "BallShape",
    "Body",
    "BodyReadout",
    "BoxShape",
    "CapsuleShape",
    "ConstantForce",
    "Gravity",
    "HullShape",
    "LinearDrag",
    "Motor",
    "PinJoint",
    "Push",
    "QuadraticDrag",
    "RopeJoint",
    "SlideJoint",
    "Spring",
    "SpringJoint",
    "WeldJoint",
    "World",
    "WorldState",
]

#: The units a simulated body's quantities are in.
UNITS = ("u", "u/s", "rad", "rad/s", "s")

quantities.register("sim.", owner="arachne.simulation", units=UNITS)
