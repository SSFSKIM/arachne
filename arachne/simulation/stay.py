"""A world's stays in the scene, each computed by the solver and kept as a
bake occurrence (:mod:`arachne.timeline.bake`).

**A stay is the world's membership in the scene.** The scene tells a world
when it enters or leaves the family of the scene's mobjects
(``Scene._membership_changed``, which calls :func:`membership`): a stay
begins where the world enters, and ends where it leaves, at the last tick at
or before the instant it left. A world that leaves and enters again with no
scene time between (``bring_to_front``, a ``remove`` and an ``add`` between
plays) stays once. A world that left while scene time passed has its stay
sealed, and one added again later starts a new stay, and a new occurrence,
from the state it was left in. A copy of a world (an animation's) is never
told: its updater finds no stay and does nothing.

**One solver's world per world per run**, held by the run's record (its
bakes' :attr:`~.bake.Bakes.held`, released once the run's stays are sealed),
never by a mobject: built from the declaration where the
world first enters, in the order its bodies, joints and forces were
declared, and continued by every later stay, so a joint's frames, a slide's
origin and a pin's angle zero are the ones declared, whatever the world did
in between. A change declared while the world is in the scene is made at
the scene's tick: the world is first computed through that tick, at every
frame rate. One declared while it is away acts from its next stay's first
step.

A world advances by the tick of the scene clock, never by the ``dt`` its
updater is handed. Its updater (:func:`closed_world`, :func:`open_world`)
computes the rows through the current tick, a closed world in one call, an
open one a step at a time, calling its callbacks once for each new tick with
the state at the tick before, and places each moving body's mobject at the
current row: its drawing as declared under the body's rigid motion since,
so a frame between ticks shows the last tick at or before it. An evaluation
with ``dt == 0`` (a play's start, the lint's entry state, the evaluation
after a play, ``resume_updating``) steps no open world and calls no
callback; a closed world's rows are the solver's alone, and it is computed
to the tick a frame reads at any evaluation. An open world steps at every
tick it is in the scene, whether or not an animation holds its drawing
(:meth:`.World.suspend_updating`), and only there: its callbacks run once
per tick, in tick order, at that tick; a leave or the end of the script
calls none.

A stay still open when the scene's script has ended is sealed when the
record's bakes are first read (:meth:`Stay.settle`): a closed world's
through the scene's last tick, an open world's through the last tick its
callbacks were called for. Before that, its computed rows are readable through the bakes as
a sealed occurrence's are (:meth:`Stay.row`).

A stay begins only where its world left its bodies drawn: each as it was
declared, and a moving one, once placed, as the world last placed it
(:func:`check_drawn`). A body the scene's code moved or redrew since,
farther than :data:`DRAWN_TOLERANCE`, refuses the stay, since a moving body
would be drawn back at its first placement and a fixed one would part from
what the world simulates.
"""

from __future__ import annotations

__all__ = [
    "ClockRow",
    "Stay",
    "clock_row",
    "closed_world",
    "live_stay",
    "membership",
    "open_world",
]

import dataclasses
import hashlib
import inspect
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn

import numpy as np

from arachne.mobject.mobject import Mobject
from arachne.mobject.placement import Placement
from arachne.mobject.values import FrozenPoints, freeze
from arachne.timeline import recorder as _recorder
from arachne.timeline.admission import TICKED
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

from . import measures
from .declaration import (
    BodyItem,
    Declaration,
    ForceItem,
    JointItem,
    Push,
    given_angular,
    given_vector,
    stored,
)
from .findings import occurrence_findings

if TYPE_CHECKING:
    from fractions import Fraction

    from arachne.scene.scene import Scene
    from arachne.timeline.record import Record
    from arachne.timeline.recorder import Recorder

    from .world import World, WorldState


class _Run:
    """A world's part in one run, held by the run's record: its solver's
    world and its open stay, until the run's stays are sealed.
    """

    __slots__ = ("crate", "placed", "recorder", "stay", "stays")

    def __init__(self, recorder: Recorder, crate: Any) -> None:
        self.recorder = recorder
        self.crate = crate
        self.stay: Stay | None = None
        #: How many stays of the world have begun in the run.
        self.stays = 0
        #: The solver's row the world's moving bodies were last placed at, in
        #: any of its stays; ``None`` before the first placement.
        self.placed: int | None = None

    def settle(self) -> None:
        """Once the scene's script has ended and the world's stays are
        sealed, let go of its solver's world.
        """
        if self.recorder not in _recorder._RUNNING and self.stay is None:
            self.crate = None


def closed_world(world: World, dt: float) -> None:
    """A closed world's updater. It names ``dt``, so a ``wait`` over the
    world is not a held frame, and never reads it, so a ticked play runs it
    only at the ticks a frame reads (:mod:`arachne.timeline.ticking`): the
    solver computes the ticks between in one call.
    """
    reach(world, stepping=True, placing=True)


def open_world(world: World, dt: float) -> None:
    """An open world's updater. It reads ``dt``, so a ticked play runs it at
    every tick, where it calls the world's callbacks for that tick; at an
    evaluation with ``dt == 0`` it calls none. While an animation holds the
    world's drawing it steps and places nothing.
    """
    reach(world, stepping=dt != 0, placing=not world._drawing_suspended)


def held_update(mobject: Mobject, dt: float) -> None:
    """Step the open worlds in the family of ``mobject``, whose updating is
    suspended (``mobject._suspended_steppers``): their callbacks run at every
    tick they are in the scene, whatever holds an ancestor's updating, as
    they do when an animation holds the world itself. A play that holds
    such a world steps on the tick when an animation suspends the ancestor
    (its copies carry the world's updater); one that does not, the ancestor
    suspended by hand, is refused as it advances.
    """
    from .world import World

    recorder = _recorder.running()
    record = None if recorder is None else recorder.record
    if record is None or record.bakes is None or not record.bakes.held:
        return
    pending = list(reversed(mobject.__dict__.get("submobjects", ())))
    while pending:
        member = pending.pop()
        if not isinstance(member, World):
            pending.extend(reversed(member.__dict__.get("submobjects", ())))
            continue
        if not member.is_open or member.updating_suspended:
            continue
        if dt != 0 and live_stay(member) is not None and not _ticked(record):
            raise ValueError(
                f"An open world is in a {type(mobject).__name__} whose updating was"
                " suspended outside an animation of the play, so its callbacks"
                " would not run at the scene's ticks. Suspend the world's own"
                " updating (it keeps stepping and holds its bodies where they are)"
                " or a group without the world in it; or animate the group."
            )
        member.update(dt)


def _ticked(record: Any) -> bool:
    """Whether the play running now steps on the scene's tick."""
    return bool(record.plays) and record.plays[-1].route == TICKED


def _run_of(world: World) -> _Run | None:
    """``world``'s part in the run of the scene running now, if it has one."""
    recorder = _recorder.running()
    if recorder is None or recorder.record is None or recorder.record.bakes is None:
        return None
    run = recorder.record.bakes.held.get(world)
    return run if isinstance(run, _Run) else None


def live_stay(world: World) -> Stay | None:
    """``world``'s stay in the scene running now, while it is in the scene."""
    run = _run_of(world)
    stay = None if run is None else run.stay
    return stay if stay is not None and stay.left_at is None else None


def crate_of(world: World) -> Any | None:
    """``world``'s solver's world in the run of the scene running now, once
    it has entered the scene.
    """
    run = _run_of(world)
    return None if run is None else run.crate


class ClockRow(NamedTuple):
    """A world's live stay and the tick of the row it shows at the scene
    clock's tick (:func:`clock_row`).
    """

    stay: Stay
    tick: int


def clock_row(world: World, *, stepping: bool) -> ClockRow | None:
    """The row ``world`` shows at the scene clock's tick, while it is in the
    scene (``None`` outside a stay): its rows computed through the clock's
    tick (a closed world's always, an open world's only when ``stepping``),
    then the latest tick computed no later than the clock's.

    The one place the world's own updater (which places its bodies there)
    and a body readout's (:func:`.readout.show_quantity`, which shows a
    body's quantity there) get that row. A play that draws a closed world
    from its sealed bake instead of ticking it answers this function from
    the bake's rows, for both; it admits a readout's updater only for a
    closed world, and hands it the bake's row here.
    """
    stay = live_stay(world)
    if stay is None:
        return None
    tick = stay.now_tick()
    if stepping or not stay.crate.open:
        stay.compute_through(tick)
    return ClockRow(stay, min(tick, stay.computed_tick))


def reach(world: World, *, stepping: bool, placing: bool) -> None:
    """Bring ``world``, while it is in the scene, to the scene clock's tick:
    its rows computed through the tick (an open world's only when
    ``stepping``), its bodies placed at the last row computed (when
    ``placing``), the row :func:`clock_row` answers.
    """
    found = clock_row(world, stepping=stepping)
    if found is not None and placing:
        found.stay.place(found.tick)


def membership(world: World, scene: Scene, present: bool, at: Fraction | None) -> None:
    """``world`` entered (``present``) or left the family of ``scene``'s
    mobjects at the scene's instant ``at``: its stay begins, or ends there.
    """
    recorder = scene._arachne_timeline
    if recorder is not _recorder.running():
        return
    if at is None:
        raise ValueError(
            f"A world {'entered' if present else 'left'} the scene during a play"
            " other than through the scene's methods (a group's add or remove, or"
            " an assignment to the scene's mobjects): the instant it did is not"
            " known, so neither is its stay. Add and remove it with Scene.add and"
            " Scene.remove, or between plays."
        )
    record = recorder.ensure_record()
    held = bakes_of(record).held
    found = held.get(world)
    run = found if isinstance(found, _Run) else None
    stay = None if run is None else run.stay
    if not present:
        if stay is not None and stay.left_at is None:
            stay.leave(at)
        return
    if stay is not None:
        if stay.left_at is None:
            return
        if stay.left_at == at:
            stay.left_at = None
            stay.left_tick = None
            stay.left_declaration = None
            for number, mobject in enumerate(world._body_mobjects):
                measures.bind(mobject, stay.ordinal, number)
            return
        stay.seal()
    if run is None:
        declaration = world._declaration
        if declaration.integrator is None:
            declaration = dataclasses.replace(
                declaration, integrator=declaration.routed()
            )
            world._declare(declaration)
        if len(world._callback_functions()) != len(declaration.callbacks):
            raise ValueError(
                "A copy of an open world cannot run: its callback forces read the"
                " bodies of the world it was copied from, by their handles, and are"
                " not bound again to the copy's. Declare another world, with"
                " callbacks of its own, instead of copying an open one."
            )
        run = held[world] = _Run(recorder, build(declaration))
    run.stay = Stay.begin(world, recorder, record, run, at)


class Stay:
    """One stay of a world in the scene: the rows of its solver's world from
    the state the stay began in, held by the record's bakes under its
    ordinal while it is open.
    """

    def __init__(
        self,
        world: World,
        recorder: Recorder,
        record: Record,
        run: _Run,
        first_tick: int,
    ) -> None:
        self.world = world
        self.recorder = recorder
        self.record = record
        self.run = run
        self.crate = run.crate
        #: The solver's row of the stay's first tick.
        self.base: int = int(self.crate.rows) - 1
        #: Whether that row is the stay's own: it is the world's first stay
        #: in the run. A later stay begins at the row an earlier one ended
        #: at, which holds what was found there.
        self.owns_first = run.stays == 0
        run.stays += 1
        self.first_tick = first_tick
        self.ordinal = -1
        #: The instant the world left the scene, while it may enter again
        #: with no scene time between, and its last tick then.
        self.left_at: Fraction | None = None
        self.left_tick: int | None = None
        #: The declaration's bytes as it left, which a change while it is
        #: away does not reach.
        self.left_declaration: bytes | None = None
        self.sealed = False
        #: The solver's row the bodies were last placed at.
        self._placed = -1
        #: Each body's own drawing as the world last vouched for it
        #: (:func:`drawing_of`): as the stay began, where :func:`check_drawn`
        #: found it drawn as the world last drew it, then as each placement
        #: made it. A body whose drawing is another when the stay ends was
        #: moved or redrawn since, and is not of the stay's last row.
        self._drawings: list[object | None] = [
            drawing_of(mobject) for mobject in world._body_mobjects
        ]

    @classmethod
    def begin(
        cls,
        world: World,
        recorder: Recorder,
        record: Record,
        run: _Run,
        at: Fraction,
    ) -> Stay:
        """Begin ``world``'s stay at the instant ``at``: from the state its
        last stay left it in, if it had one, else as declared.
        """
        check_drawn(world, record, run, at)
        if run.crate.rows == 0:
            run.crate.advance_to(0)
        stay = cls(world, recorder, record, run, math.floor(at * TICK_RATE))
        stay.ordinal = bakes_of(record).begin(stay)
        world._bake = BakeRef(stay.ordinal)
        for number, mobject in enumerate(world._body_mobjects):
            measures.bind(mobject, stay.ordinal, number)
        return stay

    def now_tick(self) -> int:
        """The scene clock's tick."""
        return math.floor(self.recorder.scene.time * TICK_RATE)

    @property
    def computed_tick(self) -> int:
        """The latest tick the solver has computed for the stay."""
        return self.first_tick + int(self.crate.rows) - 1 - self.base

    @property
    def last_tick(self) -> int:
        """The stay's last tick: the one at or before the instant it left,
        else the latest computed.
        """
        if self.left_tick is not None:
            return self.left_tick
        return self.computed_tick

    def _solver_row(self, tick: int) -> int:
        return self.base + tick - self.first_tick

    def drawn_tick(self) -> int | None:
        """The tick of the row the moving bodies are drawn at: the one the
        run last placed them at (where the stay found them drawn, as it
        began), else, in the run's first stay before any placement, its
        first row, as declared; ``None`` when that row is an earlier stay's.
        """
        placed = 0 if self.run.placed is None else self.run.placed
        if placed < self.base:
            return None
        return self.first_tick + placed - self.base

    def compute_through(self, tick: int) -> None:
        """Compute the rows through ``tick``: a closed world's in one call,
        an open world's a step at a time.
        """
        crate = self.crate
        if not crate.open:
            crate.advance_to(self._solver_row(tick))
            return
        while self.computed_tick < tick:
            self._step(crate.rows)

    def _step(self, row: int) -> None:
        """Compute an open world's solver row ``row``: call each callback
        once, in declaration order, with the state of the row before, and
        step under what they push.
        """
        world, crate = self.world, self.crate
        dimension = world.dimension
        angular = 1 if dimension == 2 else 3
        count = len(world._declaration.bodies)
        state = self._state_of(row - 1)
        forces = [[0.0] * dimension for _ in range(count)]
        torques = [[0.0] * angular for _ in range(count)]
        for callback in world._callback_functions():
            for body, push in _pushes(callback(state), world):
                force, torque = (
                    (push.force, push.torque)
                    if isinstance(push, Push)
                    else (push, None)
                )
                if force is not None:
                    _add(
                        forces[body.index],
                        given_vector(force, dimension, "A pushed force"),
                    )
                if torque is not None:
                    _add(
                        torques[body.index],
                        given_angular(torque, dimension, "A pushed torque"),
                    )
        crate.step(
            _floats(forces, "A pushed force"), _floats(torques, "A pushed torque")
        )

    def _state_of(self, row: int) -> WorldState:
        from .world import WorldState

        crate = self.crate
        return WorldState(
            self.world,
            self.first_tick + row - self.base,
            crate.pose(row, row + 1)[0],
            crate.velocity(row, row + 1)[0],
        )

    def state(self, tick: int) -> WorldState:
        """The world's bodies at the scene's ``tick`` of the stay: a closed
        world is computed to it; an open world's must have been stepped.
        """
        if tick < self.first_tick:
            raise LookupError(
                f"The world's stay began at tick {self.first_tick}, after tick {tick}."
            )
        if tick > self.now_tick():
            raise LookupError(
                f"The scene is at tick {self.now_tick()}, before tick {tick}: a"
                " world is read at a tick the scene has reached."
            )
        if self.left_at is not None and tick > self.last_tick:
            raise LookupError(
                f"The world left the scene after tick {self.last_tick}; its stay"
                f" holds no tick {tick}."
            )
        if not self.crate.open:
            self.compute_through(tick)
        elif tick > self.computed_tick:
            raise LookupError(
                f"An open world has stepped to tick {self.computed_tick}, not yet"
                f" to tick {tick}: read it at a tick it has reached."
            )
        return self._state_of(self._solver_row(tick))

    def place(self, tick: int) -> None:
        """Place each moving body's mobject at the row of ``tick``."""
        row = self._solver_row(tick)
        if row == self._placed:
            return
        self._placed = row
        self.run.placed = row
        world = self.world
        pose = self.crate.pose(row, row + 1)[0]
        for number, (item, mobject) in enumerate(
            zip(world._declaration.bodies, world._body_mobjects, strict=True)
        ):
            if not item.fixed:
                _place_body(mobject, item, pose[number], world.dimension)
                self._drawings[number] = drawing_of(mobject)

    def leave(self, at: Fraction) -> None:
        """The world left the scene at ``at``: a closed world's stay is
        computed through the tick at or before it, an open world's holds the
        ticks its callbacks ran for (none is called now), and neither holds a
        later one.
        """
        tick = math.floor(at * TICK_RATE)
        if not self.crate.open:
            self.compute_through(tick)
        self.left_at = at
        self.left_tick = min(tick, self.computed_tick)
        self.left_declaration = self.crate.declaration_bytes()
        # Each body's drawing as it is left, when it is the one its world's
        # placement at the stay's last row made: a body shown unchanged after
        # this reads that row (:mod:`.measures`). One its world last placed at
        # an older row (its updating held until it left), or that the scene's
        # code moved or redrew since its placement, is not of the last row,
        # and reads nothing past the stay.
        placed = self._placed == self._solver_row(self.left_tick)
        for number, mobject in enumerate(self.world._body_mobjects):
            kept = placed and drawing_of(mobject) is self._drawings[number]
            measures.bind(
                mobject,
                self.ordinal,
                number,
                measures.drawn_key(mobject.__dict__) if kept else None,
                self.left_tick,
            )

    # The record's side

    def row(self, tick: int) -> Row | None:
        """The stay's row at ``tick`` while it is open: through its last
        tick, of the rows computed, else ``None``.
        """
        if not self.first_tick <= tick <= min(self.last_tick, self.computed_tick):
            return None
        row = self._solver_row(tick)
        crate = self.crate
        return Row(
            tick - self.first_tick,
            tick,
            MappingProxyType(
                {
                    "pose": crate.pose(row, row + 1)[0],
                    "velocity": crate.velocity(row, row + 1)[0],
                }
            ),
        )

    def events(self, name: str) -> np.ndarray | None:
        """The stay's event table ``name`` while it is open, of the rows
        computed through its last tick: ``contact``, else ``None``.
        """
        if name != "contact":
            return None
        count = min(self.last_tick, self.computed_tick) - self.first_tick + 1
        events = _window_contacts(self.crate.contacts(), self.base, self.base + count)
        events.flags.writeable = False
        return events

    def settle(self) -> None:
        """Seal the stay if the scene's script has ended."""
        if self.recorder not in _recorder._RUNNING:
            self.seal()

    def seal(self) -> None:
        """Seal the stay through its last tick (a closed world still in the
        scene is computed to the scene's; an open one's callbacks ran while
        the script did): its rows as the bake's content, each body's mobject
        bound to its rows.
        """
        if self.left_at is None and not self.crate.open:
            self.compute_through(self.now_tick())
        crate = self.crate
        count = self.last_tick - self.first_tick + 1
        start, stop = self.base, self.base + count
        producer, version, solver, solver_version, integrator, build, features = (
            crate.identity()
        )
        pose_unit, velocity_unit = crate.units()
        events = _window_contacts(crate.contacts(), start, stop)
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
                "pose": Channel("f32", pose_unit, crate.pose(start, stop)),
                "velocity": Channel("f32", velocity_unit, crate.velocity(start, stop)),
            },
            {"contact": EventTable(events)},
        )
        bindings = []
        ids = [self.record.origin(mobject).id for mobject in self.world._body_mobjects]
        for number, body in enumerate(ids):
            bindings.append(Binding(body, "pose", number))
            bindings.append(Binding(body, "velocity", number))
        found = occurrence_findings(
            crate, start, stop, self.first_tick, ids, owns_first=self.owns_first
        )
        declared = hashlib.blake2b(digest_size=16)
        for part in (
            crate.declaration_bytes()
            if self.left_declaration is None
            else self.left_declaration,
            f"window:{start}:{stop}".encode(),
        ):
            declared.update(len(part).to_bytes(8, "big"))
            declared.update(part)
        self.sealed = True
        if self.run.stay is self:
            self.run.stay = None
        bakes_of(self.record).seal(
            self.ordinal,
            content,
            first_tick=self.first_tick,
            closed=not crate.open,
            declaration=declared.digest(),
            bindings=tuple(bindings),
            findings=found,
        )


def _window_contacts(contacts: np.ndarray, start: int, stop: int) -> np.ndarray:
    """The contact events of the solver's rows ``start`` to ``stop`` as an
    occurrence holds them, by its own rows: a pair touching at its first row
    begins there, and every later event of the window follows; no pair ends
    that did not begin in it.
    """
    touching: dict[tuple[int, int], bool] = {}
    for _row, a, b, kind in contacts[contacts[:, 0] <= start].tolist():
        touching[(a, b)] = bool(kind)
    began = [(0, a, b, 1) for (a, b), on in sorted(touching.items()) if on]
    later = contacts[(contacts[:, 0] > start) & (contacts[:, 0] < stop)].astype(
        np.int64
    )
    later[:, 0] -= start
    events = np.array(began + [tuple(row) for row in later.tolist()], dtype=np.uint32)
    return events.reshape(-1, 4)


def build(declaration: Declaration) -> Any:
    """The solver's world for ``declaration``: its bodies, joints and force
    laws added in the order they were declared.
    """
    import arachne_sim

    crate = arachne_sim.World(
        declaration.dimension,
        integrator=declaration.routed(),
        open=declaration.open,
        speed_limit=_limit(declaration.speed_limit),
        angular_speed_limit=_limit(declaration.angular_speed_limit),
        continuous_collision=declaration.continuous_collision,
    )
    bodies, joints, forces = (
        iter(declaration.bodies),
        iter(declaration.joints),
        iter(declaration.forces),
    )
    for kind in declaration.order:
        if kind == "body":
            add_body(crate, next(bodies))
        elif kind == "joint":
            add_joint(crate, next(joints))
        else:
            add_force(crate, next(forces))
    return crate


def add_body(crate: Any, body: BodyItem) -> None:
    number = crate.bodies
    crate.add_body(
        position=_f32(body.position),
        rotation=_f32(body.rotation),
        velocity=_f32(body.velocity),
        angular_velocity=_f32(body.angular_velocity),
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


def add_joint(crate: Any, joint: JointItem) -> None:
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


def add_force(crate: Any, force: ForceItem) -> None:
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


def _limit(value: float | None) -> np.float32 | None:
    """A declared limit as the solver takes it; ``None`` for its default."""
    return None if value is None else np.float32(value)


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


def _add(into: list[float], given: tuple[float, ...]) -> None:
    """Add a pushed vector, its numbers already checked, to ``into``."""
    for k, value in enumerate(given):
        into[k] += value


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


def centre(row: np.ndarray, dimension: int) -> np.ndarray:
    """A row's position, as a point of the scene."""
    point = np.zeros(3)
    point[:dimension] = row[:dimension]
    return point


def turn(row: np.ndarray, dimension: int) -> np.ndarray:
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


# A body's drawing is read and placed through the engine's own walk of its
# family and its own placement of each member's points, which no subclass of
# the body's mobject overrides; World.body refuses a drawing whose members'
# class replaces the engine's points or its placement.


def family(mobject: Mobject) -> list[Mobject]:
    """``mobject``'s family, through the submobject lists its members hold."""
    found: list[Mobject] = []
    seen: set[int] = set()
    stack = [mobject]
    while stack:
        member = stack.pop()
        if id(member) in seen:
            continue
        seen.add(id(member))
        found.append(member)
        stack.extend(reversed(member.__dict__.get("submobjects", ())))
    return found


def own_points(member: Mobject) -> np.ndarray:
    """``member``'s points, as the engine reads them."""
    points: np.ndarray = Mobject.points.fget(member)  # type: ignore[attr-defined]
    return points


def drawn_members(mobject: Mobject) -> list[Mobject]:
    """The members of ``mobject``'s family that have points."""
    return [member for member in family(mobject) if len(own_points(member))]


def placeable(member: Mobject) -> bool:
    """Whether the engine places ``member``'s points itself: its class keeps
    the engine's ``points`` and places them.
    """
    kind = type(member)
    return (
        all(
            inspect.getattr_static(kind, name) is Mobject.__dict__[name]
            for name in ("points", "_points")
        )
        and inspect.getattr_static(kind, "_places_points") is True
    )


def rest_points(mobject: Mobject) -> tuple[FrozenPoints, ...]:
    """The drawing a body moves, as declared."""
    return tuple(freeze(own_points(m), FrozenPoints) for m in drawn_members(mobject))


def drawing_of(mobject: Mobject) -> object | None:
    """What holds a mobject's own points: its placement, or else its points
    array; a move or a redraw replaces it.
    """
    own = mobject.__dict__
    placement = own.get("_placement")
    points: object | None = own.get("_points")
    if placement is not None and (points is None or points is placement.materialized):
        return placement  # type: ignore[no-any-return]
    return points


def _motion(
    item: BodyItem, row: np.ndarray, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    """A body's rigid motion since its declaration, at a row's pose, as a
    placement's ``linear`` and ``offset``.
    """
    linear = turn(row, dimension).T
    declared = np.zeros(3)
    declared[:dimension] = item.position
    offset = centre(row, dimension) - declared @ linear
    linear.flags.writeable = False
    offset.flags.writeable = False
    return linear, offset


def _place_body(
    mobject: Mobject, item: BodyItem, row: np.ndarray, dimension: int
) -> None:
    """Place each member of a body's drawing: its points as declared, under
    the body's rigid motion since (``points @ linear + offset``).
    """
    members = drawn_members(mobject)
    if len(members) != len(item.rest):
        raise ValueError(
            "A body's mobject has other members with points than when it was"
            " declared: declare a body once its drawing is complete."
        )
    linear, offset = _motion(item, row, dimension)
    for member, points in zip(members, item.rest, strict=True):
        own = member.__dict__
        own["_placement"] = Placement(points, linear, offset)
        own.pop("_points", None)


#: How far, in scene units, a point of a body's drawing may lie from where
#: its world last drew it as a stay begins (:func:`check_drawn`). A move
#: undone (a shift and its reverse, a move to the centre a body has) leaves
#: only rounding, under 1e-12 units at the coordinates a world serves
#: (within about 1000 units of the origin); a pixel of a 4K frame is about
#: 0.004 units. 1e-6 is far above the one and far below the other.
DRAWN_TOLERANCE = 1e-6


def check_drawn(world: World, record: Record, run: _Run, at: Fraction) -> None:
    """Refuse a stay of ``world`` beginning at ``at`` while a body is drawn
    other than where its world last drew it: as declared, or, for a moving
    body the run has placed, at the run's last placement. Points are
    compared, not the objects holding them, within :data:`DRAWN_TOLERANCE`,
    over the members :func:`_place_body` places.
    """
    placed = run.placed
    pose = None if placed is None else run.crate.pose(placed, placed + 1)[0]
    for number, (item, mobject) in enumerate(
        zip(world._declaration.bodies, world._body_mobjects, strict=True)
    ):
        vouched: tuple[np.ndarray, ...] = item.rest
        if pose is not None and not item.fixed:
            linear, offset = _motion(item, pose[number], world.dimension)
            vouched = tuple(
                Placement(points, linear, offset).points for points in item.rest
            )
        # As floats: a mobject's points keep the type they were given, and an
        # integer's absolute value can overflow (-128 stays negative in int8).
        drawn = [
            np.asarray(own_points(member), dtype=np.float64)
            for member in drawn_members(mobject)
        ]
        vouched = tuple(np.asarray(points, dtype=np.float64) for points in vouched)
        if [own.shape for own in drawn] != [points.shape for points in vouched]:
            _refuse_drawn(
                record,
                mobject,
                item,
                at,
                "has other members or points than when its world last drew it",
            )
        gap = max(
            (
                float(np.max(np.abs(own - points)))
                for own, points in zip(drawn, vouched, strict=True)
            ),
            default=0.0,
        )
        if gap > DRAWN_TOLERANCE:
            _refuse_drawn(
                record,
                mobject,
                item,
                at,
                f"is drawn up to {gap:.3g} {'unit' if gap == 1 else 'units'} from"
                " where its world last drew it",
            )


def _refuse_drawn(
    record: Record, mobject: Mobject, item: BodyItem, at: Fraction, what: str
) -> NoReturn:
    origin = record.origin(mobject)
    where = (
        f"made at {origin.source_line}"
        if origin.source_line is not None
        else f"with the id {origin.id}"
    )
    then = (
        "it is fixed, so its world would simulate it apart from its drawing"
        if item.fixed
        else "it moves, and the simulation drives the stay's first placement, not"
        " where the scene's code put it"
    )
    raise ValueError(
        f"The {type(mobject).__name__} {where}, a body of a world, {what} as its"
        f" world enters the scene at {float(at):g} s: {then}. Arrange the drawing"
        " first, then declare the body; or, between stays, leave a body where the"
        " world left it."
    )
