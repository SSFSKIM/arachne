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
tolerance, 1/60 s). Outside the stay its occurrence covers, every one is
unreadable, with the reason, and the lint reports
``sim.quantity_unreadable``. A contact begins where two shapes come within
the world's allowed linear error, touching or overlapping
(:meth:`arachne_sim.World.contacts`).
"""

from __future__ import annotations

__all__ = ["SimBody", "bind", "body_quantities"]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SimBody:
    """A simulated body's source: the occurrence ``ordinal`` of its world's
    stay in the record's bakes, and its number ``body`` among the world's
    bodies.
    """

    ordinal: int
    body: int


def bind(mobject: Mobject, ordinal: int, body: int) -> None:
    """Make ``mobject``'s source the body ``body`` of the stay ``ordinal``,
    in place of the one an earlier stay set; its other sources stay.
    """
    held = mobject.__dict__.get(SOURCES, ())
    kept = tuple(each for each in held if type(each) is not SimBody)
    setattr(mobject, SOURCES, kept)
    quantities.attach(mobject, SimBody(ordinal, body))


@quantities.source(SimBody)
def _sim_body(found: SimBody, held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    return body_quantities(at.bakes, found.ordinal, found.body, at.tick)


def body_quantities(
    bakes: Bakes | None, ordinal: int, body: int, tick: int
) -> tuple[Quantity, ...]:
    """The quantities of the body ``body`` of the stay ``ordinal`` at
    ``tick``, read from ``bakes``: what the body's source answers, and what a
    readout of it shows (:mod:`.readout`).
    """
    row = None if bakes is None else bakes.row(ordinal, tick)
    if row is None:
        why = f"no bake holds tick {tick} of this body's stay"
        return tuple(
            Quantity(name, None, unit, unreadable=why) for name, unit in UNITS.items()
        )
    assert bakes is not None
    pose, velocity = row["pose"][body], row["velocity"][body]
    planar = len(pose) == 3
    position = (*pose[:2], 0.0) if planar else tuple(pose[:3])
    linear = (*velocity[:2], 0.0) if planar else tuple(velocity[:3])
    angular: float | tuple[float, ...] = (
        float(velocity[2]) if planar else _floats(velocity[3:6])
    )
    first_tick = row.tick - row.index
    began = _began(bakes, ordinal, body, row.index)
    times = tuple(
        float(Fraction(first_tick + int(index), TICK_RATE)) for index in began
    )
    return (
        Quantity("sim.position", _floats(position), UNITS["sim.position"]),
        Quantity("sim.velocity", _floats(linear), UNITS["sim.velocity"]),
        Quantity("sim.angular_velocity", angular, UNITS["sim.angular_velocity"]),
        Quantity(
            "sim.contact_times",
            times[-MAX_VECTOR:],
            UNITS["sim.contact_times"],
            tolerance=CONTACT_TOLERANCE,
        ),
        Quantity("sim.contact_count", len(times), UNITS["sim.contact_count"]),
    )


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
