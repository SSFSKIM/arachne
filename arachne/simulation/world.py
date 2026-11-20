"""A world: the mobject a scene declares bodies, joints and forces in.

A :class:`World` holds its declaration as plain values (:mod:`.declaration`),
its bodies' mobjects as its submobjects, and, while it is in the scene, a
reference to the record's occurrence of its stay (:class:`.bake.BakeRef`). It
has one updater, one of two module-level functions: a **closed** world's
(no scene code in its step) names ``dt`` and never reads it, so a ticked
play runs it only where a frame reads, and the solver advances several ticks
in one call; an **open** world's (a callback force) reads it, so it runs at
every tick and calls its callbacks once for each new tick. Neither integrates
the ``dt`` it is handed: each asks the scene clock for the tick
(:mod:`.stay`). Naming ``dt`` is what keeps a ``wait`` over a world from
being a held frame.

Adding a world to the scene starts a stay; it ends at the last tick at or
before the world leaves (its updater no longer runs: removed, or its
updating suspended), and a world added again starts a new stay, and a new
occurrence, from the state it was left in. A body is a handle
(:class:`Body`), not a mobject: a callback force reads bodies by handle from
the :class:`WorldState` it is handed, never from a displayed mobject.
"""

from __future__ import annotations

__all__ = ["Body", "World", "WorldState"]

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.mobject.mobject import Mobject
from arachne.mobject.values import FrozenPoints, freeze

from . import inputs
from .declaration import (
    Attraction,
    BallJoint,
    BallShape,
    BodyItem,
    BoxShape,
    CallbackItem,
    CapsuleShape,
    ConstantForce,
    Declaration,
    ForceItem,
    Gravity,
    HullShape,
    JointItem,
    LinearDrag,
    MotorItem,
    PinJoint,
    QuadraticDrag,
    RopeJoint,
    ShapeItem,
    SlideJoint,
    Spring,
    SpringJoint,
    WeldJoint,
    measure,
    stored,
    stored_angular,
    stored_vector,
)
from .stay import closed_world, live_stay, open_world

if TYPE_CHECKING:
    from arachne.timeline.bake import BakeRef

    from .stay import Stay

#: What :meth:`World.body` takes a body's shape from by default.
AUTO = "auto"


def _require_extension() -> None:
    try:
        import arachne_sim  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "A World needs the simulation crate, arachne_sim: install the"
            " engine's 'sim' extra (pip install 'arachne-engine[sim]')."
        ) from error


@dataclass(frozen=True, eq=False)
class Body:
    """A body of a world, by its place among the world's bodies: what a
    callback force and a world's state read a body by. Not a mobject.
    """

    world: World
    index: int

    def __repr__(self) -> str:
        return f"Body({self.index})"


class WorldState:
    """A world's bodies at one tick, read-only, as the solver stored them:
    what a callback force is handed (the tick before the step it pushes) and
    what :meth:`World.at_tick` answers. ``tick`` is the scene's tick and
    ``time`` its instant in seconds (``tick / 60``).
    """

    __slots__ = ("_pose", "_velocity", "tick", "time", "world")

    def __init__(
        self, world: World, tick: int, pose: np.ndarray, velocity: np.ndarray
    ) -> None:
        self.world = world
        self.tick = tick
        self.time = tick / 60
        self._pose = pose
        self._velocity = velocity

    def _row(self, body: Body) -> int:
        if not isinstance(body, Body) or body.world is not self.world:
            raise ValueError(f"{body!r} is not a body of this world.")
        return body.index

    def position(self, body: Body) -> np.ndarray:
        """``body``'s position, a point of the scene (``z`` 0 in 2D)."""
        row = self._pose[self._row(body)]
        return _point(row[:2] if self.world.dimension == 2 else row[:3])

    def rotation(self, body: Body) -> float | np.ndarray:
        """``body``'s angle from its declared pose (2D), or its rotation from
        it as a unit quaternion ``x, y, z, w`` (3D).
        """
        row = self._pose[self._row(body)]
        if self.world.dimension == 2:
            return float(row[2])
        return _frozen(row[3:7].astype(float))

    def velocity(self, body: Body) -> np.ndarray:
        """``body``'s velocity (``z`` 0 in 2D)."""
        row = self._velocity[self._row(body)]
        return _point(row[:2] if self.world.dimension == 2 else row[:3])

    def angular_velocity(self, body: Body) -> float | np.ndarray:
        """``body``'s angular velocity: about the axis out of the plane (2D),
        or a vector (3D).
        """
        row = self._velocity[self._row(body)]
        if self.world.dimension == 2:
            return float(row[2])
        return _frozen(row[3:6].astype(float))


def _point(values: np.ndarray) -> np.ndarray:
    point = np.zeros(3)
    point[: len(values)] = values
    return _frozen(point)


def _frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class World(Mobject):
    """A world of bodies, joints and forces that a physics engine steps on
    the scene's 60 Hz tick, whatever the frame rate.

    ``dimension`` is 2 or 3. ``gravity``, when given, is a uniform
    acceleration on every body (:class:`Gravity`). ``integrator`` is
    ``"solver"`` (the rigid-body solver: shapes, contacts, joints) or
    ``"point_masses"`` (velocity Verlet on points with no extent: force laws
    between moving bodies, such as springs, orbits and N bodies, held exact
    through a tick); by default a world whose bodies are all points and that
    has no joint takes ``point_masses``, and any other takes ``"solver"``.
    :attr:`integrator` says which it takes, and the record's bake names it.

    Declare bodies with :meth:`body`, then joints with :meth:`add_joint` and
    force laws with :meth:`add_force`, and add the world to the scene: it
    moves its bodies' mobjects, its submobjects, while it is there. A
    callback force makes it open. A world's own numbers are read through
    ``state_at``'s declared quantities and :meth:`at_tick`, never from its
    mobjects.

    Without the engine's ``sim`` extra, making a world raises an
    :class:`ImportError` that names it.
    """

    def __init__(
        self,
        dimension: int = 2,
        *,
        gravity: Any = None,
        integrator: str | None = None,
        **kwargs: Any,
    ) -> None:
        _require_extension()
        if dimension not in (2, 3):
            raise ValueError(f"A world has 2 or 3 dimensions, not {dimension!r}.")
        if integrator not in (None, "solver", "point_masses"):
            raise ValueError(
                'A world\'s integrator is "solver" or "point_masses", not'
                f" {integrator!r}."
            )
        super().__init__(**kwargs)
        self._declaration = Declaration(dimension, integrator)
        self._body_mobjects: tuple[Mobject, ...] = ()
        self._bake: BakeRef | None = None
        self.add_updater(closed_world)  # type: ignore[arg-type]
        if gravity is not None:
            self.add_force(Gravity(gravity))

    # What it was given

    @property
    def dimension(self) -> int:
        return self._declaration.dimension

    @property
    def integrator(self) -> str:
        """The integrator the world takes: ``"solver"`` or
        ``"point_masses"``, as declared, else as routed from what it holds
        (fixed once it first enters the scene).
        """
        return self._declaration.routed()

    @property
    def is_open(self) -> bool:
        """Whether a callback force runs in the world's step."""
        return self._declaration.open

    @property
    def bodies(self) -> tuple[Body, ...]:
        """The world's bodies, in the order they were declared."""
        return tuple(Body(self, i) for i in range(len(self._declaration.bodies)))

    def body(
        self,
        mobject: Mobject,
        *,
        shape: Any = AUTO,
        mass: float | None = None,
        density: float | None = None,
        velocity: Any = None,
        angular_velocity: Any = None,
        fixed: bool = False,
        friction: float = 0.0,
        restitution: float = 0.0,
        gravity_scale: float = 1.0,
    ) -> Body:
        """Declare a body drawn by ``mobject``, which joins the world's
        submobjects, as it stands now: its centre and its drawing are read
        here, so shape it first. Returns its handle.

        ``shape`` is taken from the mobject by default (a circle or a sphere
        is a ball, anything else the convex hull of its points); a shape
        (:class:`BallShape`, :class:`BoxShape`, :class:`CapsuleShape`,
        :class:`HullShape`) or a sequence of them (a compound) declares it,
        about the mobject's centre; ``None`` makes the body a point, with no
        extent and no contact. ``mass`` is the body's mass (a point's is
        required unless ``fixed``), or else ``density`` (1 by default) gives
        its shapes theirs. ``velocity`` and ``angular_velocity`` are its
        initial ones; a ``fixed`` body never moves. ``friction`` and
        ``restitution`` are its shapes' material.
        """
        if not isinstance(mobject, Mobject) or isinstance(mobject, World):
            raise TypeError(f"A body is drawn by a mobject, not {mobject!r}.")
        if any(mobject is held for held in self._body_mobjects):
            raise ValueError("That mobject is a body of this world already.")
        if self._bake is not None:
            raise ValueError(
                "A body joins a world before the world first enters the scene:"
                " declare it with the world's other bodies, or in another world."
            )
        dimension = self.dimension
        centre = np.asarray(mobject.get_center(), dtype=float)
        position = stored_vector(centre[:dimension], dimension, "A body's position")
        shapes = self._shapes(mobject, shape, centre, friction, restitution)
        if mass is not None and density is not None:
            raise ValueError("A body takes a mass or a density, not both.")
        point_mass = 0.0
        if shapes and self._declaration.integrator == "point_masses":
            raise ValueError(
                "A world of point masses has no shapes: declare the body with"
                ' shape=None, or the world with integrator="solver".'
            )
        if not shapes:
            if mass is None and not fixed:
                raise ValueError("A point (shape=None) that moves needs a mass.")
            if angular_velocity is not None:
                raise ValueError("A point (shape=None) does not turn.")
            point_mass = 0.0 if mass is None else stored(mass, "A body's mass")
        elif mass is not None:
            whole = sum(measure(item, dimension) for item in shapes)
            shapes = tuple(
                dataclasses.replace(
                    item, density=stored(float(mass) / whole, "A body's density")
                )
                for item in shapes
            )
        elif density is not None:
            shapes = tuple(
                dataclasses.replace(item, density=stored(density, "A body's density"))
                for item in shapes
            )
        for declared in shapes:
            inputs.check_feature(f"shape.{declared.kind}")
        if fixed and (velocity is not None or angular_velocity is not None):
            raise ValueError("A fixed body does not move: give it no velocity.")
        body = BodyItem(
            fixed=bool(fixed),
            position=position,
            rotation=(0.0,) if dimension == 2 else (0.0, 0.0, 0.0, 1.0),
            velocity=stored_vector(
                (0.0,) * dimension if velocity is None else velocity,
                dimension,
                "A body's velocity",
            ),
            angular_velocity=stored_angular(
                (0.0 if dimension == 2 else (0.0,) * 3)
                if angular_velocity is None
                else angular_velocity,
                dimension,
                "A body's angular velocity",
            ),
            mass=point_mass,
            gravity_scale=stored(gravity_scale, "A body's gravity scale"),
            shapes=shapes,
            rest=tuple(
                freeze(member.points, FrozenPoints)
                for member in mobject.family_members_with_points()
            ),
        )
        self._declare(
            dataclasses.replace(
                self._declaration, bodies=(*self._declaration.bodies, body)
            )
        )
        self._body_mobjects = (*self._body_mobjects, mobject)
        self.add(mobject)
        return Body(self, len(self._declaration.bodies) - 1)

    def _shapes(
        self,
        mobject: Mobject,
        shape: Any,
        centre: np.ndarray,
        friction: float,
        restitution: float,
    ) -> tuple[ShapeItem, ...]:
        dimension = self.dimension
        material = (
            1.0,
            stored(friction, "A body's friction"),
            stored(restitution, "A body's restitution"),
        )
        if shape is None:
            return ()
        if isinstance(shape, str):
            if shape != AUTO:
                raise ValueError(
                    f'A body\'s shape is "{AUTO}", None, a shape or shapes, not'
                    f" {shape!r}."
                )
            shape = _shape_of(mobject, centre, dimension)
        declared = list(shape) if isinstance(shape, (list, tuple)) else [shape]
        if not declared:
            raise ValueError("A body's shapes are one shape or more; None for a point.")
        return tuple(_shape_item(item, dimension, material) for item in declared)

    def add_joint(
        self,
        joint: PinJoint | SlideJoint | WeldJoint | BallJoint | RopeJoint | SpringJoint,
    ) -> None:
        """Join two of the world's bodies. Declared while the world is in the
        scene, it holds from the next tick.
        """
        item = self._joint_item(joint)
        inputs.check_feature(f"joint.{item.kind}")
        if item.limits is not None:
            inputs.check_feature(f"joint.{item.kind}.limits")
        if item.motor is not None:
            inputs.check_feature(f"joint.{item.kind}.motor")
        if self._declaration.integrator == "point_masses":
            raise ValueError(
                "A world of point masses has no joints: declare the world with"
                ' integrator="solver".'
            )
        stay = self._live()
        self._declare(
            dataclasses.replace(
                self._declaration, joints=(*self._declaration.joints, item)
            )
        )
        if stay is not None:
            stay.add_joint(item)

    def add_force(self, force: Any) -> None:
        """Add a force law (:class:`Gravity`, :class:`Attraction`,
        :class:`Spring`, :class:`LinearDrag`, :class:`QuadraticDrag`,
        :class:`ConstantForce`) or a callback force, a function called once
        for each new tick with the :class:`WorldState` of the tick before,
        which returns what it pushes: a mapping from body to a force vector
        or a :class:`.Push` (a force and a torque). A callback makes the
        world open; it reads bodies only from the state it is handed, and
        another world through its :meth:`at_tick` (``sim.force_input``). A
        force law declared while the world is in the scene acts from the next
        tick; a callback joins a world before it enters the scene.
        """
        if isinstance(
            force,
            (Gravity, Attraction, Spring, LinearDrag, QuadraticDrag, ConstantForce),
        ):
            item = self._force_item(force)
            inputs.check_feature(f"force.{item.kind}")
            stay = self._live()
            self._declare(
                dataclasses.replace(
                    self._declaration, forces=(*self._declaration.forces, item)
                )
            )
            if stay is not None:
                stay.add_force(item)
            return
        inputs.check_force(force)
        if self._bake is not None:
            raise ValueError(
                "A callback force joins a world before the world first enters the"
                " scene: declare it with the world's other forces."
            )
        self._declare(
            dataclasses.replace(
                self._declaration,
                callbacks=(*self._declaration.callbacks, CallbackItem(force)),
            )
        )
        if closed_world in self.updaters:
            self.remove_updater(closed_world)
            self.add_updater(open_world)  # type: ignore[arg-type]

    def at_tick(self, tick: int) -> WorldState:
        """The world's bodies at the scene's tick ``tick`` (``t * 60``) of its
        stay now: a closed world is advanced to it; an open world's tick must
        have been stepped. The tick-indexed read of another world a callback
        force makes; such a world is added to the scene before the world
        whose force reads it.
        """
        stay = self._live()
        if stay is None:
            raise LookupError(
                "World.at_tick reads a world while it is in the scene: add it to"
                " the scene before the world whose force reads it."
            )
        return stay.state(tick)

    # Internal

    def _live(self) -> Stay | None:
        return live_stay(self)

    def _declare(self, declaration: Declaration) -> None:
        stay = self._live()
        if stay is not None:
            stay.check_change()
        self._declaration = declaration

    def _local(self, index: int, point: Any, what: str) -> tuple[float, ...]:
        """The scene point ``point`` in body ``index``'s frame, as the body
        stands now: declared, or at the current row of its stay.
        """
        dimension = self.dimension
        given = np.zeros(3)
        values = np.asarray(point, dtype=float).ravel()
        given[: len(values)] = values
        stay = self._live()
        if stay is not None:
            centre, turn = stay.pose_now(index)
        else:
            item = self._declaration.bodies[index]
            centre, turn = _pose_point(item.position), np.eye(3)
        local = turn.T @ (given - centre)
        return stored_vector(local[:dimension], dimension, what)

    def _axis(self, index: int, axis: Any, what: str) -> tuple[float, ...]:
        """The scene direction ``axis`` in body ``index``'s frame, as now."""
        dimension = self.dimension
        given = np.zeros(3)
        values = np.asarray(axis, dtype=float).ravel()
        given[: len(values)] = values
        stay = self._live()
        turn = stay.pose_now(index)[1] if stay is not None else np.eye(3)
        local = turn.T @ given
        return stored_vector(local[:dimension], dimension, what)

    def _extended(self, index: int, what: str) -> None:
        """Refuse ``what`` on body ``index`` if it is a point."""
        if not self._declaration.bodies[index].shapes:
            raise ValueError(
                f"{what} needs a body with a shape: a point has no extent."
            )

    def _index(self, body: Any) -> int:
        if not isinstance(body, Body) or body.world is not self:
            raise ValueError(f"{body!r} is not a body of this world.")
        return body.index

    def _centre(self, index: int) -> np.ndarray:
        stay = self._live()
        if stay is not None:
            return stay.pose_now(index)[0]
        return _pose_point(self._declaration.bodies[index].position)

    def _joint_item(self, joint: Any) -> JointItem:
        dimension = self.dimension
        if not isinstance(
            joint, (PinJoint, SlideJoint, WeldJoint, BallJoint, RopeJoint, SpringJoint)
        ):
            raise TypeError(f"A joint is a PinJoint, SlideJoint, ..., not {joint!r}.")
        a, b = self._index(joint.a), self._index(joint.b)
        if a == b:
            raise ValueError("A joint joins two different bodies.")
        numbers: tuple[tuple[str, float], ...] = ()
        axis = None
        limits = None
        motor = None
        if isinstance(joint, (RopeJoint, SpringJoint)):
            anchor_a = self._centre(a) if joint.anchor_a is None else joint.anchor_a
            anchor_b = self._centre(b) if joint.anchor_b is None else joint.anchor_b
            anchor1 = self._local(a, anchor_a, "A joint's anchor")
            anchor2 = self._local(b, anchor_b, "A joint's anchor")
            if isinstance(joint, RopeJoint):
                kind = "rope"
                numbers = (("length", stored(joint.length, "A rope's length")),)
            else:
                kind = "spring"
                numbers = (
                    ("rest", stored(joint.rest, "A spring's rest length")),
                    ("stiffness", stored(joint.stiffness, "A spring's stiffness")),
                    ("damping", stored(joint.damping, "A spring's damping")),
                )
        else:
            at = getattr(joint, "at", None)
            if at is None:
                at = self._centre(b)
            anchor1 = self._local(a, at, "A joint's point")
            anchor2 = self._local(b, at, "A joint's point")
            if isinstance(joint, PinJoint):
                kind = "pin"
                if dimension == 3:
                    axis = self._axis(
                        a,
                        (0, 0, 1) if joint.axis is None else joint.axis,
                        "A pin's axis",
                    )
                elif joint.axis is not None:
                    raise ValueError("A pin in a 2D world turns about no axis but z.")
            elif isinstance(joint, SlideJoint):
                kind = "slide"
                axis = self._axis(a, joint.axis, "A slide's axis")
            elif isinstance(joint, WeldJoint):
                kind = "weld"
            else:
                if dimension != 3:
                    raise ValueError(
                        "A ball joint is 3D only; in 2D, a pin turns about its point."
                    )
                kind = "ball"
            if isinstance(joint, (PinJoint, SlideJoint)):
                if joint.limits is not None:
                    low, high = joint.limits
                    limits = (
                        stored(low, "A joint's limit"),
                        stored(high, "A joint's limit"),
                    )
                if joint.motor is not None:
                    given = joint.motor
                    motor = MotorItem(
                        stored(given.position, "A motor's position"),
                        stored(given.velocity, "A motor's velocity"),
                        stored(given.stiffness, "A motor's stiffness"),
                        stored(given.damping, "A motor's damping"),
                    )
        return JointItem(
            kind,
            a,
            b,
            anchor1,
            anchor2,
            axis,
            limits,
            numbers,
            bool(joint.collide),
            motor,
        )

    def _force_item(self, force: Any) -> ForceItem:
        dimension = self.dimension
        if isinstance(force, Gravity):
            return ForceItem(
                "gravity",
                (),
                (
                    (
                        "acceleration",
                        stored_vector(force.acceleration, dimension, "Gravity"),
                    ),
                ),
            )
        if isinstance(force, Attraction):
            bodies = tuple(self._index(body) for body in force.bodies)
            return ForceItem(
                "attraction",
                bodies,
                (("strength", (stored(force.strength, "An attraction's strength"),)),),
            )
        if isinstance(force, Spring):
            a, b = self._index(force.a), self._index(force.b)
            anchors: list[tuple[str, tuple[float, ...]]] = []
            for name, index, given in (
                ("anchor_a", a, force.anchor_a),
                ("anchor_b", b, force.anchor_b),
            ):
                if given is not None:
                    self._extended(index, "A spring's anchor off its centre")
                    anchors.append(
                        (name, self._local(index, given, "A spring's anchor"))
                    )
            return ForceItem(
                "spring",
                (a, b),
                (
                    ("stiffness", (stored(force.stiffness, "A spring's stiffness"),)),
                    ("rest", (stored(force.rest, "A spring's rest length"),)),
                    ("damping", (stored(force.damping, "A spring's damping"),)),
                    *anchors,
                ),
            )
        if isinstance(force, (LinearDrag, QuadraticDrag)):
            kind = "linear_drag" if isinstance(force, LinearDrag) else "quadratic_drag"
            return ForceItem(
                kind,
                (self._index(force.body),),
                (
                    (
                        "coefficient",
                        (stored(force.coefficient, "A drag's coefficient"),),
                    ),
                ),
            )
        numbers: list[tuple[str, tuple[float, ...]]] = []
        if force.force is not None:
            numbers.append(
                ("force", stored_vector(force.force, dimension, "A constant force"))
            )
        if force.torque is not None:
            self._extended(self._index(force.body), "A torque")
            numbers.append(
                ("torque", stored_angular(force.torque, dimension, "A constant torque"))
            )
        return ForceItem("constant", (self._index(force.body),), tuple(numbers))


def _pose_point(position: Sequence[float]) -> np.ndarray:
    point = np.zeros(3)
    point[: len(position)] = position
    return point


def _shape_of(mobject: Mobject, centre: np.ndarray, dimension: int) -> Any:
    """The shape a body takes from its mobject: a ball for a circle (2D) or a
    sphere (3D) of equal extents, else the convex hull of its points.
    """
    from arachne.mobject.geometry.arc import Circle
    from arachne.mobject.three_d.three_dimensions import Sphere

    width, height = mobject.width, mobject.height
    round_kind = Circle if dimension == 2 else Sphere
    if (
        isinstance(mobject, round_kind)
        and math.isclose(width, height, rel_tol=1e-9, abs_tol=1e-12)
        and (dimension == 2 or math.isclose(width, mobject.depth, rel_tol=1e-9))
    ):
        return BallShape(width / 2)
    members = mobject.family_members_with_points()
    if not members:
        raise ValueError(
            "A body's mobject has no points to take a shape from: declare its shape,"
            " or shape=None for a point."
        )
    points = np.concatenate([member.points for member in members])
    return HullShape(points[:, :dimension] - centre[:dimension])


def _shape_item(
    shape: Any, dimension: int, material: tuple[float, float, float]
) -> ShapeItem:
    density, friction, restitution = material
    if isinstance(shape, HullShape):
        points = np.asarray(shape.points, dtype=float)
        points = points.reshape(-1, points.shape[-1] if points.ndim > 1 else dimension)
        points = points[:, :dimension]
        if len(points) < dimension + 1:
            raise ValueError(f"A convex hull takes at least {dimension + 1} points.")
        return ShapeItem(
            "convex_hull",
            tuple(
                stored(float(value), "A convex hull's point")
                for value in points.ravel()
            ),
            (0.0,) * dimension,
            _rotation(0.0, dimension),
            density,
            friction,
            restitution,
        )
    if isinstance(shape, BallShape):
        kind = "ball"
        size: tuple[float, ...] = (stored(shape.radius, "A ball's radius"),)
        angle = 0.0
    elif isinstance(shape, BoxShape):
        kind = "cuboid"
        extents = [shape.width, shape.height]
        if dimension == 3:
            if shape.depth is None:
                raise ValueError("A box in a 3D world has a depth.")
            extents.append(shape.depth)
        elif shape.depth is not None:
            raise ValueError("A box in a 2D world has no depth.")
        size = tuple(stored(extent / 2, "A box's size") for extent in extents)
        angle = shape.angle
    elif isinstance(shape, CapsuleShape):
        kind = "capsule"
        size = (
            stored(shape.length / 2, "A capsule's length"),
            stored(shape.radius, "A capsule's radius"),
        )
        angle = shape.angle
    else:
        raise TypeError(
            "A body's shape is a BallShape, BoxShape, CapsuleShape or HullShape, not"
            f" {shape!r}."
        )
    if dimension == 3 and angle:
        raise ValueError("A shape in a 3D world is not turned: angle is for 2D.")
    return ShapeItem(
        kind,
        size,
        stored_vector(shape.at, dimension, "A shape's offset"),
        _rotation(angle, dimension),
        density,
        friction,
        restitution,
    )


def _rotation(angle: float, dimension: int) -> tuple[float, ...]:
    if dimension == 2:
        return (stored(angle, "A shape's angle"),)
    return (0.0, 0.0, 0.0, 1.0)
