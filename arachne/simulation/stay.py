"""A world's stays in the scene, each computed by the solver and kept as a
bake occurrence (:mod:`arachne.timeline.bake`).

A world advances by the tick of the scene clock, never by the ``dt`` its
updater is handed (which is zero at a play's start, for the lint's entry
state and after a play). Its updater, :func:`closed_world` or
:func:`open_world`, finds the world's stay through the running recorder's
record, whose bakes hold what computes it (:class:`Stay`, never the world's
mobject), and there:

- begins a stay where the world enters the scene: the solver's world is built
  from the declaration, from the state the last stay left it in if it had
  one, and the record reserves the stay's occurrence;
- ends the stay that was open when the world was away (its updater did not
  run since an earlier play, and time passed): sealed at the last tick at or
  before it left;
- computes the rows through the current tick: a closed world in one call; an
  open one a step at a time, calling its callbacks once for each new tick
  with the state at the tick before;
- places each moving body's mobject at the current row: its drawing as
  declared under the body's rigid motion since, so a frame between ticks
  shows the last tick at or before it.

A stay still open when the scene's script has ended is sealed when the
record's bakes are first read (:meth:`Stay.settle`): a world still in the
scene at the end has been advanced to the scene's last tick. Before that, the
stay's computed rows through its latest tick are readable through the bakes
as a sealed occurrence's are (:meth:`Stay.row`), so a lint's forward run
reads them while the script runs.
"""

from __future__ import annotations

__all__ = ["Stay", "closed_world", "live_stay", "open_world"]

import hashlib
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.mobject.mobject import Mobject
from arachne.mobject.placement import Placement
from arachne.mobject.values import FrozenPoints
from arachne.timeline import recorder as _recorder
from arachne.timeline.bake import (
    BakeContent,
    BakeIdentity,
    BakeRef,
    Binding,
    Channel,
    EventTable,
    Row,
    bakes_of,
)
from arachne.timeline.clock import TICK_RATE

from .declaration import Declaration, ForceItem, JointItem, Push, stored

if TYPE_CHECKING:
    from fractions import Fraction

    from arachne.timeline.record import Record
    from arachne.timeline.recorder import Recorder

    from .world import World, WorldState


def closed_world(world: World, dt: float) -> None:
    """A closed world's updater. It names ``dt``, so a ``wait`` over the
    world is not a held frame, and never reads it, so a ticked play runs it
    only at the ticks a frame reads (:mod:`arachne.timeline.ticking`): the
    solver computes the ticks between in one call.
    """
    reach(world)


def open_world(world: World, dt: float) -> None:
    """An open world's updater. It reads ``dt`` (and nothing of it), so a
    ticked play runs it at every tick, where it calls the world's callbacks
    for that tick.
    """
    _every_tick = dt
    reach(world)


def live_stay(world: World) -> Stay | None:
    """``world``'s open stay in the record of the scene running now, if any."""
    ref = world._bake
    recorder = _recorder.running()
    if ref is None or recorder is None or recorder.record is None:
        return None
    bakes = recorder.record.bakes
    if bakes is None:
        return None
    try:
        stay = bakes.live(ref.ordinal)
    except KeyError:
        return None
    if isinstance(stay, Stay) and stay.world is world and stay.recorder is recorder:
        return stay
    return None


def reach(world: World) -> None:
    """Bring ``world`` to the scene clock's tick: its stay begun or ended as
    it entered or left the scene, its rows computed through the tick, its
    bodies placed there.
    """
    recorder = _recorder.running()
    if recorder is None:
        return
    record = recorder.ensure_record()
    now = recorder.scene.time
    tick = math.floor(now * TICK_RATE)
    play = len(record.plays) - 1
    stay = live_stay(world)
    if stay is not None and not stay.continues(play, now):
        stay.seal()
        stay = None
    if stay is None:
        stay = Stay.begin(world, recorder, record, tick, now)
    stay.reach(tick, play, now)
    stay.place(tick)


class Stay:
    """One stay of a world in the scene while it is open: the solver's world
    that computes it, held by the record's bakes under its ordinal, never by
    a mobject.
    """

    def __init__(
        self,
        world: World,
        recorder: Recorder,
        record: Record,
        crate: Any,
        first_tick: int,
        now: Fraction,
    ) -> None:
        self.world = world
        self.recorder = recorder
        self.record = record
        self.crate = crate
        self.first_tick = first_tick
        self.ordinal = -1
        #: The play and instant the world was last brought to.
        self.last_play: int | None = None
        self.last_time = now
        #: The row the bodies were last placed at.
        self._placed = -1

    @classmethod
    def begin(
        cls,
        world: World,
        recorder: Recorder,
        record: Record,
        tick: int,
        now: Fraction,
    ) -> Stay:
        """Begin ``world``'s stay at ``tick``: from the state its last stay
        left it in, if it had one.
        """
        bakes = bakes_of(record)
        start = None
        if world._bake is not None:
            occurrence = bakes.occurrence(world._bake.ordinal)
            if occurrence is not None:
                start = bakes.row(occurrence.ordinal, occurrence.last_tick)
        declaration = world._declaration
        if declaration.integrator is None:
            declaration = world._declaration = _fixed(declaration)
        stay = cls(world, recorder, record, build(declaration, start), tick, now)
        stay.ordinal = bakes.begin(stay)
        world._bake = BakeRef(stay.ordinal)
        return stay

    def continues(self, play: int, now: Fraction) -> bool:
        """Whether the world stayed in the scene since it was last brought
        to the clock: in the same play, or at the same instant (between
        plays no time passes). Otherwise it was away while time passed.
        """
        return play == self.last_play or now == self.last_time

    @property
    def last_tick(self) -> int:
        """The stay's latest tick: the last at or before its latest instant."""
        return math.floor(self.last_time * TICK_RATE)

    def reach(self, tick: int, play: int, now: Fraction) -> None:
        """Compute the rows through ``tick``."""
        index = tick - self.first_tick
        crate = self.crate
        if not crate.open:
            crate.advance_to(index)
        else:
            if crate.rows == 0:
                crate.advance_to(0)
            while crate.rows <= index:
                self._step(crate.rows)
        self.last_play = play
        self.last_time = now

    def _step(self, row: int) -> None:
        """Compute an open world's row ``row``: call each callback once,
        in declaration order, with the state of the row before, and step
        under what they push.
        """
        world, crate = self.world, self.crate
        dimension = world.dimension
        angular = 1 if dimension == 2 else 3
        count = len(world._declaration.bodies)
        state = self._state_of(row - 1)
        forces = [[0.0] * dimension for _ in range(count)]
        torques = [[0.0] * angular for _ in range(count)]
        for callback in world._declaration.callbacks:
            pushed = callback.function(state)
            for body, push in _pushes(pushed, world):
                force, torque = (
                    (push.force, push.torque)
                    if isinstance(push, Push)
                    else (push, None)
                )
                if force is not None:
                    _add(forces[body.index], force, dimension, "A pushed force")
                if torque is not None:
                    _add(torques[body.index], torque, angular, "A pushed torque")
        crate.step(
            _floats(forces, "A pushed force"), _floats(torques, "A pushed torque")
        )

    def _state_of(self, index: int) -> WorldState:
        from .world import WorldState

        crate = self.crate
        return WorldState(
            self.world,
            self.first_tick + index,
            crate.pose(index, index + 1)[0],
            crate.velocity(index, index + 1)[0],
        )

    def state(self, tick: int) -> WorldState:
        """The world's bodies at the scene's ``tick``: a closed world is
        advanced to it, an open world's must have been stepped.
        """
        index = tick - self.first_tick
        crate = self.crate
        if index < 0:
            raise LookupError(
                f"The world's stay began at tick {self.first_tick}, after tick {tick}."
            )
        if not crate.open:
            crate.advance_to(index)
        elif index >= crate.rows:
            raise LookupError(
                f"An open world has stepped to tick {self.first_tick + crate.rows - 1},"
                f" not yet to tick {tick}: read it at a tick it has reached."
            )
        return self._state_of(index)

    def pose_now(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        """Body ``body``'s centre and rotation matrix at the stay's latest
        tick.
        """
        index = self.last_tick - self.first_tick
        pose = self.crate.pose(index, index + 1)[0][body]
        return _centre(pose, self.world.dimension), _turn(pose, self.world.dimension)

    def place(self, tick: int) -> None:
        """Place each moving body's mobject at the row of ``tick``."""
        index = tick - self.first_tick
        if index == self._placed:
            return
        self._placed = index
        world = self.world
        dimension = world.dimension
        pose = self.crate.pose(index, index + 1)[0]
        declaration = world._declaration
        for number, (item, mobject) in enumerate(
            zip(declaration.bodies, world._body_mobjects, strict=True)
        ):
            if item.fixed:
                continue
            row = pose[number]
            turn = _turn(row, dimension)
            linear = turn.T
            declared = np.zeros(3)
            declared[:dimension] = item.position
            offset = _centre(row, dimension) - declared @ linear
            _place(mobject, item.rest, linear, offset)

    def check_change(self) -> None:
        """Refuse a change to the world when rows past the scene's tick
        exist: a read ahead (another world's force) computed them without it.
        """
        now = math.floor(self.recorder.scene.time * TICK_RATE)
        computed = self.first_tick + self.crate.rows - 1
        if computed > now:
            raise ValueError(
                f"The world has computed rows to tick {computed}, past the scene's"
                f" tick {now} (another world's force read it ahead): change it"
                " before that read, or between plays where nothing reads ahead."
            )

    def add_joint(self, item: JointItem) -> None:
        _add_joint(self.crate, item)

    def add_force(self, item: ForceItem) -> None:
        _add_force(self.crate, item)

    # The record's side

    def row(self, tick: int) -> Row | None:
        """The stay's row at ``tick`` while it is open: through its latest
        tick, else ``None``.
        """
        index = tick - self.first_tick
        if not 0 <= index <= self.last_tick - self.first_tick:
            return None
        if index >= self.crate.rows:
            return None
        crate = self.crate
        return Row(
            index,
            tick,
            MappingProxyType(
                {
                    "pose": crate.pose(index, index + 1)[0],
                    "velocity": crate.velocity(index, index + 1)[0],
                }
            ),
        )

    def settle(self) -> None:
        """Seal the stay if the scene's script has ended."""
        if self.recorder not in _recorder._RUNNING:
            self.seal()

    def seal(self) -> None:
        """Seal the stay through its latest tick: its rows as the bake's
        content, each body's mobject bound to its rows.
        """
        crate = self.crate
        count = self.last_tick - self.first_tick + 1
        producer, version, solver, solver_version, integrator, build, features = (
            crate.identity()
        )
        pose_unit, velocity_unit = crate.units()
        contacts = np.asarray(crate.contacts())
        content = BakeContent(
            BakeIdentity(
                producer,
                version,
                solver,
                solver_version,
                integrator,
                build,
                tuple(features),
            ),
            count,
            {
                "pose": Channel("f32", pose_unit, crate.pose(0, count)),
                "velocity": Channel("f32", velocity_unit, crate.velocity(0, count)),
            },
            {"contact": EventTable(contacts[contacts[:, 0] < count])},
        )
        bindings = []
        for number, mobject in enumerate(self.world._body_mobjects):
            body = self.record.origin(mobject).id
            bindings.append(Binding(body, "pose", number))
            bindings.append(Binding(body, "velocity", number))
        bakes_of(self.record).seal(
            self.ordinal,
            content,
            first_tick=self.first_tick,
            closed=not crate.open,
            declaration=hashlib.blake2b(
                crate.declaration_bytes(), digest_size=16
            ).digest(),
            bindings=tuple(bindings),
        )


def _fixed(declaration: Declaration) -> Declaration:
    """``declaration`` with the integrator it takes, fixed from its first
    stay on.
    """
    import dataclasses

    return dataclasses.replace(declaration, integrator=declaration.routed())


def build(declaration: Declaration, start: Row | None) -> Any:
    """The solver's world for a stay of ``declaration``: each body as
    declared, or as ``start`` (the last row of the stay before) holds it.
    """
    import arachne_sim

    dimension = declaration.dimension
    crate = arachne_sim.World(
        dimension, integrator=declaration.routed(), open=declaration.open
    )
    left = 0 if start is None else len(start["pose"])
    if start is not None and any(
        joint.limits is not None or joint.motor is not None
        for joint in declaration.joints
    ):
        raise ValueError(
            "A world with a joint that has limits or a motor is in the scene for one"
            " stay: their angles count from the pose it was declared in. Keep it in"
            " the scene, or declare a new world."
        )
    for number, body in enumerate(declaration.bodies):
        if number < left:
            assert start is not None
            pose, velocity = start["pose"][number], start["velocity"][number]
            position = pose[:dimension]
            rotation = pose[dimension:]
            linear, turning = velocity[:dimension], velocity[dimension:]
        else:
            position = _f32(body.position)
            rotation = _f32(body.rotation)
            linear = _f32(body.velocity)
            turning = _f32(body.angular_velocity)
        crate.add_body(
            position=np.ascontiguousarray(position, dtype=np.float32),
            rotation=np.ascontiguousarray(rotation, dtype=np.float32),
            velocity=np.ascontiguousarray(linear, dtype=np.float32),
            angular_velocity=np.ascontiguousarray(turning, dtype=np.float32),
            mass=np.float32(body.mass),
            gravity_scale=np.float32(body.gravity_scale),
            fixed=body.fixed,
        )
        for shape in body.shapes:
            crate.add_shape(
                number,
                shape.kind,
                _f32(shape.size),
                position=_f32(shape.position),
                rotation=_f32(shape.rotation),
                density=np.float32(shape.density),
                friction=np.float32(shape.friction),
                restitution=np.float32(shape.restitution),
            )
    for joint in declaration.joints:
        _add_joint(crate, joint)
    for force in declaration.forces:
        _add_force(crate, force)
    return crate


def _add_joint(crate: Any, joint: JointItem) -> None:
    numbers = {name: np.float32(value) for name, value in joint.numbers}
    number = crate.add_joint(
        joint.kind,
        joint.body1,
        joint.body2,
        anchor1=_f32(joint.anchor1),
        anchor2=_f32(joint.anchor2),
        axis=None if joint.axis is None else _f32(joint.axis),
        limits=None if joint.limits is None else _f32(joint.limits),
        contacts=joint.contacts,
        **numbers,
    )
    motor = joint.motor
    if motor is not None:
        crate.add_motor(
            number,
            position=np.float32(motor.position),
            velocity=np.float32(motor.velocity),
            stiffness=np.float32(motor.stiffness),
            damping=np.float32(motor.damping),
        )


def _add_force(crate: Any, force: ForceItem) -> None:
    numbers = {name: _numbers(values) for name, values in force.numbers}
    if force.kind == "gravity":
        crate.add_gravity(numbers["acceleration"])
    elif force.kind == "attraction":
        crate.add_attraction(list(force.bodies), numbers["strength"])
    elif force.kind == "spring":
        crate.add_spring(*force.bodies, **numbers)
    elif force.kind == "linear_drag":
        crate.add_linear_drag(*force.bodies, numbers["coefficient"])
    elif force.kind == "quadratic_drag":
        crate.add_quadratic_drag(*force.bodies, numbers["coefficient"])
    else:
        crate.add_constant(*force.bodies, **numbers)


def _numbers(values: tuple[float, ...]) -> Any:
    """Stored numbers as the solver takes them: one as a ``numpy.float32``,
    several as a float32 array. Exact: each already holds a float32.
    """
    if len(values) == 1:
        return np.float32(values[0])
    return _f32(values)


def _f32(values: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Stored numbers as a float32 array, exactly."""
    return np.ascontiguousarray(values, dtype=np.float32)


def _floats(rows: list[list[float]], what: str) -> np.ndarray:
    """What callbacks pushed, summed in Python floats, through the one
    conversion to the stored type.
    """
    return np.array(
        [[stored(value, what) for value in row] for row in rows], dtype=np.float32
    )


def _add(into: list[float], given: Any, count: int, what: str) -> None:
    """Add a pushed vector (a 2D world's may carry a zero ``z``) to ``into``."""
    values = np.asarray(given, dtype=float).ravel()
    if len(values) == 3 and count == 2:
        values = values[:2]
    if len(values) != count:
        raise ValueError(f"{what} takes {count} numbers, not {len(values)}.")
    for k in range(count):
        into[k] += float(values[k])


def _pushes(pushed: Any, world: World) -> list[tuple[Any, Any]]:
    """What one callback returned, as (body, push) pairs."""
    from .world import Body

    if pushed is None:
        return []
    if not isinstance(pushed, Mapping):
        raise TypeError(
            "A callback force returns a mapping from body to a force or a Push"
            f" (or None), not {type(pushed).__name__}."
        )
    found = []
    for body, push in pushed.items():
        if not isinstance(body, Body) or body.world is not world:
            raise ValueError(
                f"A callback force pushes {body!r}, not a body of its world."
            )
        found.append((body, push))
    return found


def _centre(row: np.ndarray, dimension: int) -> np.ndarray:
    point = np.zeros(3)
    point[:dimension] = row[:dimension]
    return point


def _turn(row: np.ndarray, dimension: int) -> np.ndarray:
    """The rotation a row's pose holds, as a matrix acting on column
    vectors: by its angle (2D), or by its quaternion ``x, y, z, w`` (3D,
    normalised; a product of its numbers, no trigonometry).
    """
    if dimension == 2:
        angle = float(row[2])
        cos, sin = math.cos(angle), math.sin(angle)
        return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    x, y, z, w = (float(value) for value in row[3:7])
    scale = 2.0 / (x * x + y * y + z * z + w * w)
    return np.array(
        [
            [
                1 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1 - scale * (x * x + y * y),
            ],
        ]
    )


def _place(
    mobject: Mobject,
    rest: tuple[FrozenPoints, ...],
    linear: np.ndarray,
    offset: np.ndarray,
) -> None:
    """Place each member of ``mobject``'s family that has points: its points
    as declared, under ``points @ linear + offset``.
    """
    members = mobject.family_members_with_points()
    if len(members) != len(rest):
        raise ValueError(
            "A body's mobject has other members with points than when it was"
            " declared: declare a body once its drawing is complete."
        )
    linear.flags.writeable = False
    offset.flags.writeable = False
    for member, points in zip(members, rest, strict=True):
        placement = Placement(points, linear, offset)
        if member._places_points and type(member).points is Mobject.points:
            own = member.__dict__
            own["_placement"] = placement
            own.pop("_points", None)
        else:
            member.points = placement.points
