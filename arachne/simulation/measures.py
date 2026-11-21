"""What a simulated body declares: its quantities, read from its world's bake.

Each body's mobject carries one attached source (:mod:`arachne.quantities`),
a :class:`SimBody`: the ordinal of the occurrence of its world's stay in the
record's bakes, and the body's number among the world's bodies (its row on
the first axis of the bake's channels). The world sets it as each stay
begins (:func:`bind`), so the body's capture at an instant names the stay
that held it then. The source reads, at the instant's tick (the last tick at
or before it), the bake's row and its contact events, and nothing else:

- ``sim.position``, in ``u``: the body's position, three numbers (``z`` is
  0 in a 2D world);
- ``sim.velocity``, in ``u/s``: its velocity, three numbers;
- ``sim.angular_velocity``, in ``rad/s``: one number in a 2D world (about
  the axis out of the plane), three in a 3D one;
- ``sim.contact_times``, in ``s``: the instants of the ticks at which a
  contact of the body began, at or before the instant, the latest 16, one
  per contact (two contacts begun at one tick give it twice), to within a
  tick: a contact began during the step that ended there;
- ``sim.contact_count``, in ``1``: the contacts of the body begun at or
  before the instant.

Each is exact as baked, the contact instants to within a tick (their
tolerance, 1/60 s). While an animation holds its world's drawing (one of
the world, or of a group holding it), a body reads the row it is drawn at,
the one its world last placed it at, not the instant's: its quantities and
its placement share a tick. A body left on screen after its world left reads the
row its placement shows, the stay's last, at every instant after it while
its own drawing stays as the stay left it, whichever route answers the
instant: a held wait's one capture or a ticked play's own. That holds for
a body drawn by its own points alone; a body with submobjects (a group)
is drawn by them too, which its capture does not hold, so past its stay
it is unreadable, moved or not. Outside its stay otherwise (moved or
redrawn, at the instant its world left as later, or before the stay)
every one is unreadable, with the reason, and the lint reports
``sim.quantity_unreadable``. A copy of a body's mobject is another
mobject, not the body, and declares none of them. A contact begins where
two shapes come within the world's allowed linear error, touching or
overlapping (:meth:`arachne_sim.World.contacts`).
"""

from __future__ import annotations

__all__ = ["SimBody", "bind", "body_quantities", "drawn_key"]

import hashlib
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne import quantities
from arachne.quantities import MAX_VECTOR, SOURCES, At, Quantity
from arachne.timeline.clock import TICK_RATE

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.bake import Bakes

#: Each quantity a body declares, and its unit.
UNITS: Mapping[str, str] = {
    "sim.angular_velocity": "rad/s",
    "sim.contact_count": "1",
    "sim.contact_times": "s",
    "sim.position": "u",
    "sim.velocity": "u/s",
}
#: How exact a contact instant is: one tick.
CONTACT_TOLERANCE = 1 / TICK_RATE
#: What :attr:`SimBody.held` names when the body is held drawn at a row of an
#: earlier stay: no row of its own stay.
EARLIER = -1


@dataclass(frozen=True, slots=True)
class SimBody:
    """A simulated body's source: the occurrence ``ordinal`` of its world's
    stay in the record's bakes, and its number ``body`` among the world's
    bodies; once its world has left the scene, ``left``, the stay's last
    tick, and ``drawn``, the key of the body's own drawing as the stay left
    it (:func:`drawn_key`), ``None`` when that drawing is not of the last
    row. While its world is in the scene with its drawing held (an
    animation of the world or of a group holding it, or its updating
    suspended), ``held``, the tick of the row the body is drawn at, the one
    its world last placed it at (:data:`EARLIER` when that row is an earlier
    stay's). A copy of the body's mobject is not the body: its source is
    ``copied`` and answers nothing (the world that owns a copy binds it as
    the copy's stay begins).
    """

    ordinal: int
    body: int
    drawn: bytes | None = None
    left: int | None = None
    copied: bool = False
    held: int | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> SimBody:
        return replace(self, copied=True)


def bind(
    mobject: Mobject,
    ordinal: int,
    body: int,
    drawn: bytes | None = None,
    left: int | None = None,
    *,
    held: int | None = None,
) -> None:
    """Make ``mobject``'s source the body ``body`` of the stay ``ordinal``
    (once its world has left, at the stay's last tick ``left``, with its
    drawing as the stay left it ``drawn``; while its world's drawing is
    held, drawn at the row of the tick ``held``), in place of the one an
    earlier stay set; its other sources stay.
    """
    sources = mobject.__dict__.get(SOURCES, ())
    kept = tuple(each for each in sources if type(each) is not SimBody)
    setattr(mobject, SOURCES, kept)
    quantities.attach(mobject, SimBody(ordinal, body, drawn, left, held=held))


def drawn_key(own: Mapping[str, Any]) -> bytes | None:
    """The key of what a mobject's own attributes ``own`` draw, when they
    draw all of it: its placement's, or its points'; ``None`` for one with
    submobjects (its drawing is also theirs, which its capture does not
    hold) or with no points of its own.
    """
    if own.get("submobjects"):
        return None
    placement = own.get("_placement")
    points = own.get("_points")
    if placement is not None and (points is None or points is placement.materialized):
        return bytes(placement.key)
    if points is None or not len(points):
        return None
    array = np.ascontiguousarray(points)
    return hashlib.blake2b(array.tobytes(), digest_size=16).digest()


@quantities.source(SimBody)
def _sim_body(found: SimBody, held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    if found.copied:
        return ()
    tick = _tick(found, held, at)
    if tick is None:
        why = (
            "its world holds it drawn at a row of an earlier stay"
            if found.left is None
            else f"its drawing is not the one its world's stay left at tick {found.left}"
        )
        return tuple(
            Quantity(name, None, unit, unreadable=why) for name, unit in UNITS.items()
        )
    return body_quantities(at.bakes, found.ordinal, found.body, tick)


def _tick(found: SimBody, held: Mapping[str, Any], at: At) -> int | None:
    """The tick a body's capture shows: in its world's stay, the instant's,
    or, while the world's drawing is held, the row's it is drawn at
    (``None`` for an earlier stay's); once its world has left, the stay's
    last while its drawing is the one the stay left. Otherwise (moved,
    redrawn, drawn by its submobjects, or not placed at the last row) no row
    of the stay is what it shows: at a later tick, which no bake of the stay
    holds, the instant's; at the stay's own last, ``None``.
    """
    if found.left is None:
        if found.held is None:
            return at.tick
        return None if found.held == EARLIER else found.held
    if found.drawn is not None and drawn_key(held) == found.drawn:
        return found.left
    return at.tick if at.tick > found.left else None


def body_quantities(
    bakes: Bakes | None,
    ordinal: int,
    body: int,
    tick: int,
    names: Collection[str] | None = None,
) -> tuple[Quantity, ...]:
    """The quantities of the body ``body`` of the stay ``ordinal`` at
    ``tick``, read from ``bakes``: what the body's source answers, and what a
    readout of it shows (:mod:`.readout`). With ``names``, only those, in
    the source's order: the contact events are read only for a contact's
    quantity.
    """
    wanted = tuple(name for name in UNITS if names is None or name in names)
    row = None if bakes is None else bakes.row(ordinal, tick)
    if row is None:
        why = f"no bake holds tick {tick} of this body's stay"
        return tuple(
            Quantity(name, None, UNITS[name], unreadable=why) for name in wanted
        )
    assert bakes is not None
    found: dict[str, Quantity] = {}
    pose, velocity = row["pose"][body], row["velocity"][body]
    planar = len(pose) == 3
    if "sim.position" in wanted:
        position = (*pose[:2], 0.0) if planar else tuple(pose[:3])
        found["sim.position"] = Quantity(
            "sim.position", _floats(position), UNITS["sim.position"]
        )
    if "sim.velocity" in wanted:
        linear = (*velocity[:2], 0.0) if planar else tuple(velocity[:3])
        found["sim.velocity"] = Quantity(
            "sim.velocity", _floats(linear), UNITS["sim.velocity"]
        )
    if "sim.angular_velocity" in wanted:
        angular: float | tuple[float, ...] = (
            float(velocity[2]) if planar else _floats(velocity[3:6])
        )
        found["sim.angular_velocity"] = Quantity(
            "sim.angular_velocity", angular, UNITS["sim.angular_velocity"]
        )
    if "sim.contact_times" in wanted or "sim.contact_count" in wanted:
        first_tick = row.tick - row.index
        began = _began(bakes, ordinal, body, row.index)
        times = tuple(
            float(Fraction(first_tick + int(index), TICK_RATE)) for index in began
        )
        found["sim.contact_times"] = Quantity(
            "sim.contact_times",
            times[-MAX_VECTOR:],
            UNITS["sim.contact_times"],
            tolerance=CONTACT_TOLERANCE,
        )
        found["sim.contact_count"] = Quantity(
            "sim.contact_count", len(times), UNITS["sim.contact_count"]
        )
    return tuple(found[name] for name in wanted)


def _began(bakes: Bakes, ordinal: int, body: int, index: int) -> np.ndarray:
    """The rows of the stay ``ordinal``, at or before ``index``, at which a
    contact of the body ``body`` began, one per contact, in order.
    """
    table = bakes.events(ordinal, "contact")
    if table is None or not len(table):
        return np.zeros(0, dtype=np.uint32)
    rows, a, b, kind = table.T
    mine = (kind == 1) & (rows <= index) & ((a == body) | (b == body))
    return np.asarray(rows[mine])


def _floats(values: Iterable[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)
