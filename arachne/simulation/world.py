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

A world's stays are its membership in the scene (:mod:`.stay`): one
begins where it enters the family of the scene's mobjects and ends at the
last tick at or before it leaves, a leave and an entry with no scene time
between are one stay, and a world added again later starts a new stay, and
a new occurrence, from the state it was left in. A body is a handle
(:class:`Body`), not a mobject: a callback force reads bodies by handle from
the :class:`WorldState` it is handed, another world's bodies included,
never from a displayed mobject.
"""

from __future__ import annotations

__all__ = ["Body", "World", "WorldState"]

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from arachne.mobject import mobject as mobject_module
from arachne.mobject.mobject import Mobject

from . import inputs, stay
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
    given_vector,
    measure,
    stored,
    stored_angular,
    stored_vector,
)
from .stay import closed_world, crate_of, live_stay, open_world

if TYPE_CHECKING:
    from fractions import Fraction

    from arachne.assets.anchor import Anchor
    from arachne.scene.scene import Scene
    from arachne.timeline.bake import BakeRef

    from .readout import BodyReadout

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
    ``time`` its instant in seconds (``tick / 60``). A body of another world
    in the scene is read at the same tick, from that world's own rows: how a
    callback force reads another world.
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

    def _of(self, body: Body) -> WorldState:
        """The state ``body`` is read from: this one, or its own world's at
        this tick.
        """
        if not isinstance(body, Body):
            raise ValueError(f"{body!r} is not a body.")
        if body.world is self.world:
            return self
        return body.world.at_tick(self.tick)

    def position(self, body: Body) -> np.ndarray:
        """``body``'s position, a point of the scene (``z`` 0 in 2D)."""
        state = self._of(body)
        row = state._pose[body.index]
        return _point(row[:2] if state.world.dimension == 2 else row[:3])

    def rotation(self, body: Body) -> float | np.ndarray:
        """``body``'s angle from its declared pose (2D), or its rotation from
        it as a unit quaternion ``x, y, z, w`` (3D).
        """
        state = self._of(body)
        row = state._pose[body.index]
        if state.world.dimension == 2:
            return float(row[2])
        return _frozen(row[3:7].astype(float))

    def velocity(self, body: Body) -> np.ndarray:
        """``body``'s velocity (``z`` 0 in 2D)."""
        state = self._of(body)
        row = state._velocity[body.index]
        return _point(row[:2] if state.world.dimension == 2 else row[:3])

    def angular_velocity(self, body: Body) -> float | np.ndarray:
        """``body``'s angular velocity: about the axis out of the plane (2D),
        or a vector (3D).
        """
        state = self._of(body)
        row = state._velocity[body.index]
        if state.world.dimension == 2:
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

    ``speed_limit`` (units per second) and ``angular_speed_limit`` (radians
    per second) are the speeds above which a body is unstable: the lint
    reports ``sim.unstable`` for a body that rises above one, and for a body
    whose state went non-finite. Each is positive and below the solver's own
    cap (400 u/s; 45 degrees a tick, about 47.1 rad/s), above which the
    solver would scale the motion down; by default 200 u/s and 40 rad/s.
    ``continuous_collision`` (on by default) has the solver sweep a fast body
    against the fixed ones, so it cannot pass through a thin one; two bodies
    that pass through each other in a tick are reported as
    ``sim.tunnelled``. Each body's mobject declares its position, velocity,
    angular velocity and contacts (``sim.*`` in ``state_at``'s quantities,
    :mod:`.measures`); :meth:`readout` shows one of them at every frame.

    Declare bodies with :meth:`body`, then joints with :meth:`add_joint` and
    force laws with :meth:`add_force`, and add the world to the scene: it
    moves its bodies' mobjects, its submobjects, while it is there. A
    callback force makes it open. A world's own numbers are read through
    ``state_at``'s declared quantities and :meth:`at_tick`, never from its
    mobjects.

    A body's mobject is drawn by the engine's own points: a member of its
    family whose class replaces the engine's ``points`` or the placement of
    its points is refused, since the world places each member's points
    itself.

    **Membership goes through the scene's methods.** A world's stays follow
    ``Scene.add``, ``remove``, ``replace``, ``clear``, ``bring_to_front`` and
    ``bring_to_back``, the foreground pair, the animations that introduce and
    remove, and groups changed between plays. A direct edit of
    ``scene.mobjects`` (an assignment or an in-place change) is not tracked
    during a play, where one the scene finds after scene time passed is
    refused, nor after the last play; an edit inside one play that is undone
    before the play ends is not seen at all.

    **An animation that holds the world's drawing** (``FadeIn(world)``, or
    ``FadeIn(Group(world, label))``, which suspend its updating) holds its
    bodies where they are; an open world's callbacks still run at every tick
    while it is in the scene, in the tick's order, and the bodies show the
    world's motion again once the animation ends. A closed world's rows are
    the scene clock's all the same: it computes the ticks it missed when its
    updating resumes. Suspending the world's own updating by hand does what
    an animation does. An open world in a group whose updating is suspended
    by hand, outside an animation, is refused when a play that does not step
    on the tick advances it: that play would call no callback at its ticks.

    **An open world's stay ends at the last tick its callbacks ran for.**
    Removed at tick ``k`` by an updater that runs before the world's own
    (a mobject earlier in the scene), its stay ends at ``k - 1``; removed
    after the world's updater ran at ``k`` (by a later mobject or a scene
    updater), at ``k``. The order of the updaters within a tick does not
    depend on the frame grid, so every frame rate ends it alike.

    A world's mobject keys as content by :attr:`_declared`, a digest of what
    it was given, and its bodies by their placements; the declaration
    itself, its bodies' list (its submobjects) and its occurrence's
    reference are not content (:attr:`_content_key_omits`). Its callback
    forces are held by the world apart from what a capture keeps of it.

    Without the engine's ``sim`` extra, making a world raises an
    :class:`ImportError` that names it.
    """

    #: Not content: the declaration (its digest, ``_declared``, is), the
    #: bodies' mobjects (they are the world's submobjects), and the reference
    #: to its occurrence (the record's bakes key the stay).
    _content_key_omits: ClassVar[frozenset[str]] = frozenset(
        {"_declaration", "_body_mobjects", "_bake"}
    )
    # The callback forces: held by the world, outside what a capture keeps
    # of it (its __dict__), and not copied with it.
    # Each body's drawing as it was declared (stay.drawing_of), by body:
    # held by the world apart from its captured state and not copied with
    # it, so a copy vouches for none.
    __slots__ = ("_callbacks", "_drawn_as_declared")

    def __init__(
        self,
        dimension: int = 2,
        *,
        gravity: Any = None,
        integrator: str | None = None,
        speed_limit: float | None = None,
        angular_speed_limit: float | None = None,
        continuous_collision: bool = True,
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
        if not isinstance(continuous_collision, bool):
            raise TypeError(
                "A world's continuous_collision is True or False, not"
                f" {continuous_collision!r}."
            )
        limits = _limits(speed_limit, angular_speed_limit)
        super().__init__(**kwargs)
        self._callbacks: tuple[Any, ...] = ()
        self._drawn_as_declared: tuple[object | None, ...] = ()
        self._declare(
            Declaration(
                dimension,
                integrator,
                speed_limit=limits[0],
                angular_speed_limit=limits[1],
                continuous_collision=continuous_collision,
            )
        )
        self._body_mobjects: tuple[Mobject, ...] = ()
        self._bake: BakeRef | None = None
        #: Whether an animation holds the world's drawing (its updating
        #: suspended): it steps, and places nothing.
        self._drawing_suspended = False
        self.add_updater(closed_world)  # type: ignore[arg-type]
        _take_membership_notice(self)
        if gravity is not None:
            self.add_force(Gravity(gravity))

    def __deepcopy__(self, clone_from_id: dict[int, Any]) -> World:
        """A copy is a world declared alike that never entered the scene,
        with no callback forces: a copy of an open world is refused if it
        enters the scene.
        """
        copied = super().__deepcopy__(clone_from_id)
        copied._bake = None
        _take_membership_notice(copied)
        return copied

    def _declare(self, declaration: Declaration) -> None:
        """Hold ``declaration``, and its digest as the world's content."""
        self._declaration = declaration
        #: The digest of the declaration: what a content key reads of it.
        self._declared = declaration.digest()

    def _declared_drawings(self) -> tuple[object | None, ...]:
        """Each body's drawing as it was declared; for a copy, which vouches
        for none, ``None`` for each.
        """
        found: tuple[object | None, ...] = getattr(self, "_drawn_as_declared", ())
        missing = len(self._body_mobjects) - len(found)
        return (*found, *(None,) * missing)

    def _callback_functions(self) -> tuple[Any, ...]:
        """The world's callback forces, in the order declared."""
        return getattr(self, "_callbacks", ())

    def suspend_updating(self, recursive: bool = True) -> World:
        """Suspend updating, as an animation of the world does: the bodies
        hold where they are. An open world keeps stepping (its callbacks run
        at every tick it is in the scene) and places nothing until resumed.
        """
        super().suspend_updating(recursive)
        if self._declaration.open:
            self.updating_suspended = False
            self._drawing_suspended = True
        return self

    def resume_updating(self, recursive: bool = True) -> World:
        self._drawing_suspended = False
        super().resume_updating(recursive)
        return self

    def _scene_membership(
        self, scene: Scene, present: bool, at: Fraction | None
    ) -> None:
        """The scene's notice that the world entered or left its mobjects'
        family at ``at`` (``Scene._membership_changed``).
        """
        stay.membership(self, scene, present, at)

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
        refused = sorted(
            {
                type(member).__qualname__
                for member in stay.drawn_members(mobject)
                if not stay.placeable(member)
            }
        )
        if refused:
            raise TypeError(
                "A body's mobject is drawn by the engine's own points:"
                f" {', '.join(refused)} replaces them or their placement, which"
                " the world cannot vouch for. Draw the body with a mobject of"
                " another class."
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
            rest=stay.rest_points(mobject),
        )
        self._declare(
            dataclasses.replace(
                self._declaration,
                bodies=(*self._declaration.bodies, body),
                order=(*self._declaration.order, "body"),
            )
        )
        drawings = self._declared_drawings()
        self._body_mobjects = (*self._body_mobjects, mobject)
        self.add(mobject)
        self._drawn_as_declared = (*drawings, stay.drawing_of(mobject))
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
        if self._declaration.integrator == "point_masses":
            raise ValueError(
                "A world of point masses has no joints: declare the world with"
                ' integrator="solver".'
            )
        crate = self._crate_now()
        item = self._joint_item(joint)
        inputs.check_feature(f"joint.{item.kind}")
        if item.limits is not None:
            inputs.check_feature(f"joint.{item.kind}.limits")
        if item.motor is not None:
            inputs.check_feature(f"joint.{item.kind}.motor")
        if crate is not None:
            stay.add_joint(crate, item)
        self._declare(
            dataclasses.replace(
                self._declaration,
                joints=(*self._declaration.joints, item),
                order=(*self._declaration.order, "joint"),
            )
        )

    def add_force(self, force: Any) -> None:
        """Add a force law (:class:`Gravity`, :class:`Attraction`,
        :class:`Spring`, :class:`LinearDrag`, :class:`QuadraticDrag`,
        :class:`ConstantForce`) or a callback force, a function called once
        for each new tick with the :class:`WorldState` of the tick before,
        which returns what it pushes: a mapping from body to a force vector
        or a :class:`.Push` (a force and a torque). A callback makes the
        world open; it reads bodies only from the state it is handed, another
        world's by their handles too, and holds no mobject, scene or world
        (``sim.force_input``). A force law declared while the world is in the
        scene acts from the next tick, at every frame rate; a callback joins a
        world before it enters the scene.
        """
        if isinstance(
            force,
            (Gravity, Attraction, Spring, LinearDrag, QuadraticDrag, ConstantForce),
        ):
            crate = self._crate_now()
            item = self._force_item(force)
            inputs.check_feature(f"force.{item.kind}")
            if crate is not None:
                stay.add_force(crate, item)
            self._declare(
                dataclasses.replace(
                    self._declaration,
                    forces=(*self._declaration.forces, item),
                    order=(*self._declaration.order, "force"),
                )
            )
            return
        inputs.check_force(force)
        if self._bake is not None:
            raise ValueError(
                "A callback force joins a world before the world first enters the"
                " scene: declare it with the world's other forces."
            )
        self._callbacks = (*self._callback_functions(), force)
        name = getattr(force, "__qualname__", type(force).__qualname__)
        self._declare(
            dataclasses.replace(
                self._declaration,
                callbacks=(*self._declaration.callbacks, CallbackItem(str(name))),
            )
        )
        if closed_world in self.updaters:
            self.remove_updater(closed_world)
            self.add_updater(open_world)  # type: ignore[arg-type]

    def at_tick(self, tick: int) -> WorldState:
        """The world's bodies at the scene's tick ``tick`` (``t * 60``) of its
        stay now, a tick the scene has reached: a closed world is computed to
        it; an open world's tick must have been stepped. A callback force
        reads another world through the state it is handed instead.
        """
        live = live_stay(self)
        if live is None:
            raise LookupError(
                "A world is read while it is in the scene: add it to the scene"
                " before the world whose force reads it."
            )
        return live.state(tick)

    def readout(
        self,
        body: Body,
        quantity: str,
        *,
        part: str | None = None,
        scale: float = 1.0,
        format: str = "{:.1f}",  # noqa: A002 (the readout's own word)
        at: Anchor | None = None,
        **options: Any,
    ) -> BodyReadout:
        """A readout of ``body``'s ``quantity`` (``"position"``,
        ``"velocity"``, ``"angular_velocity"`` or ``"contact_count"``),
        shown at every frame from the world's rows: its part ``part`` times
        ``scale``, in ``format``, at ``at``; the other arguments are a
        readout's (:class:`.BodyReadout`). Add it to the scene after the
        world, as anything that follows a body.
        """
        from .readout import BodyReadout

        self._index(body)
        return BodyReadout(
            body, quantity, part=part, scale=scale, format=format, at=at, **options
        )

    # Internal

    def _crate_now(self) -> Any | None:
        """The world's solver's world in the running scene, computed through
        the scene's tick if the world is in the scene: where a change
        declared now takes effect.
        """
        live = live_stay(self)
        if live is not None:
            live.compute_through(live.now_tick())
        return crate_of(self)

    def _pose(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Body ``index``'s centre and rotation as it stands now: at its
        solver's latest row, or as declared.
        """
        crate = crate_of(self)
        if crate is None:
            item = self._declaration.bodies[index]
            return _pose_point(item.position), np.eye(3)
        row = crate.pose(crate.rows - 1, crate.rows)[0][index]
        return stay.centre(row, self.dimension), stay.turn(row, self.dimension)

    def _local(self, index: int, point: Any, what: str) -> tuple[float, ...]:
        """The scene point ``point`` in body ``index``'s frame, as the body
        stands now.
        """
        dimension = self.dimension
        given = _pose_point(given_vector(point, dimension, what))
        centre, turn = self._pose(index)
        local = turn.T @ (given - centre)
        return stored_vector(local[:dimension], dimension, what)

    def _axis(self, index: int, axis: Any, what: str) -> tuple[float, ...]:
        """The scene direction ``axis`` in body ``index``'s frame, as now."""
        dimension = self.dimension
        given = _pose_point(given_vector(axis, dimension, what))
        local = self._pose(index)[1].T @ given
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
        return self._pose(index)[0]

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


def _limits(
    speed: float | None, angular: float | None
) -> tuple[float | None, float | None]:
    """A world's declared speed limits, stored: each positive and strictly
    below the solver's own cap; ``None`` for the solver's default.
    """
    import arachne_sim

    found: list[float | None] = []
    for given, cap, what, unit in (
        (speed, arachne_sim.SPEED_CAP, "speed_limit", "u/s"),
        (angular, arachne_sim.ANGULAR_SPEED_CAP, "angular_speed_limit", "rad/s"),
    ):
        if given is None:
            found.append(None)
            continue
        limit = stored(given, f"A world's {what}")
        if not 0 < limit < cap:
            raise ValueError(
                f"A world's {what} is a positive number below the solver's own cap"
                f" of {cap:.4g} {unit}, above which the solver scales a body's"
                f" motion down; not {given!r}."
            )
        found.append(limit)
    return found[0], found[1]


def _take_membership_notice(world: World) -> None:
    from arachne.scene.scene import _take_membership_notice as take

    take(world)
    if stay.held_update not in mobject_module._suspended_steppers:
        mobject_module._suspended_steppers.append(stay.held_update)


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
    members = stay.drawn_members(mobject)
    if not members:
        raise ValueError(
            "A body's mobject has no points to take a shape from: declare its shape,"
            " or shape=None for a point."
        )
    points = np.concatenate([stay.own_points(member) for member in members])
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
