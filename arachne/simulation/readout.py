"""A readout of a simulated body's quantity, shown at every frame.

:meth:`.World.readout` makes a :class:`BodyReadout`: a
:class:`~arachne.mobject.annotation.Readout` whose value an engine updater,
:func:`show_quantity`, sets from the body's row of its world's stay at the
scene clock's tick, the tick the world places its bodies at, read as the
body's own quantities are (:func:`.measures.body_quantities`). The value is
one part of the quantity (its norm, one axis, or the whole of a single
number) times a scale, so a readout can show another unit.

It declares its value as every readout does, ``annotation.count``, unrounded
(its text is the value in its format), and under ``sim.`` what it shows, so
that a check reads the number shown and the body's quantity at one capture
and compares them:

- ``sim.reads``, a name: the quantity, ``sim.position``, ``sim.velocity``,
  ``sim.angular_velocity`` or ``sim.contact_count``;
- ``sim.reads_part``, a name: ``norm``, ``x``, ``y``, ``z``, or ``value``
  (the whole of a quantity that is one number);
- ``sim.reads_scale``, in ``1``: what the part is multiplied by;
- ``sim.reads_body``, in ``1``: two numbers, the ordinal of the stay's
  occurrence in the record's bakes and the body's number among its world's
  bodies (the occurrence's bindings name the body's mobject by stable id);
- ``sim.reads_tick``, in ``1``: the tick whose row the value is of.

Each is exact. ``annotation.count`` is the part of the body's quantity at
``sim.reads_tick`` times the scale, as computed here in double precision
from the baked numbers; the text shown is that value in the readout's
format, so it lies within half a unit of the format's last digit of it.

**The tick it shows.** At each evaluation its updater shows the latest
tick its world has computed no later than the scene clock's
(:func:`.stay.clock_row`, the row the world places its bodies at). A
closed world is computed to the clock's tick there, so its readout shows
that tick wherever it stands among the scene's mobjects; at the instant
the world enters, or at the evaluation that ends a play, that row may
already exist and is shown. An open world steps in its own updater: its
readout shows the tick the world has stepped to, the clock's when the
readout comes after the world in the scene's update order (added after
it), the tick before when it comes before.

Outside its world's stays (the world removed from the scene) it holds the
last value its updater set, which may come from a play's closing
evaluation, and ``sim.reads_tick`` and ``sim.reads_body`` keep naming the
row it is. Before it has read any stay, ``sim.reads_body`` and
``sim.reads_tick`` are unreadable (the lint reports
``sim.quantity_unreadable``) and it shows 0. An animation of the readout
itself (``FadeIn``) holds its updating, so it shows an old tick while its
body moves: the lint reports ``sim.readout_stale`` when that is more than
a tick behind the scene while its world is in the scene.

**Only the world sets it.** :meth:`BodyReadout.set_value`, and so
:class:`~arachne.mobject.annotation.CountUp`, are refused, so the number
shown is never one the declared row does not hold: a number chosen by hand
is a plain :class:`~arachne.mobject.annotation.Readout`, set between plays
from :meth:`.World.at_tick`.
"""

from __future__ import annotations

__all__ = ["BodyReadout", "ShownQuantity", "show_quantity"]

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from arachne import quantities
from arachne.mobject.annotation.readout import Readout
from arachne.quantities import SOURCES, At, Quantity
from arachne.timeline.bake import bakes_of

from . import measures
from .stay import clock_row
from .world import Body

if TYPE_CHECKING:
    from typing import Self

    from arachne.assets.anchor import Anchor

    from .world import World

#: The quantities a readout shows, by the name :meth:`.World.readout` takes.
READABLE: Mapping[str, str] = {
    "position": "sim.position",
    "velocity": "sim.velocity",
    "angular_velocity": "sim.angular_velocity",
    "contact_count": "sim.contact_count",
}
#: The parts of a quantity that is one number.
WHOLE = ("value",)


@dataclass(frozen=True, eq=False)
class _Shows:
    """What a body readout reads: its world, the body's number among the
    world's bodies, the quantity's name, the part and the scale. A handle,
    not content: a copy of the readout reads the same world.
    """

    world: World
    body: int
    quantity: str
    part: str
    scale: float

    def __deepcopy__(self, memo: dict[int, Any]) -> _Shows:
        return self


@dataclass(frozen=True, slots=True)
class ShownQuantity:
    """A body readout's source of declared quantities: the quantity, part
    and scale it shows, the body's number, and, once it has read a stay, the
    stay's occurrence and the tick its value is of.
    """

    quantity: str
    part: str
    scale: float
    body: int
    occurrence: int | None = None
    tick: int | None = None


@quantities.source(ShownQuantity)
def _shown_quantity(
    found: ShownQuantity, held: Mapping[str, Any], at: At
) -> Iterable[Quantity]:
    shows = (
        Quantity("sim.reads", found.quantity, "name"),
        Quantity("sim.reads_part", found.part, "name"),
        Quantity("sim.reads_scale", found.scale, "1"),
    )
    if found.occurrence is None or found.tick is None:
        why = "the readout has read no stay of its body's world"
        return (
            *shows,
            Quantity("sim.reads_body", None, "1", unreadable=why),
            Quantity("sim.reads_tick", None, "1", unreadable=why),
        )
    return (
        *shows,
        Quantity("sim.reads_body", (found.occurrence, found.body), "1"),
        Quantity("sim.reads_tick", found.tick, "1"),
    )


def show_quantity(readout: BodyReadout, dt: float) -> None:
    """A body readout's updater: show the body's quantity at the row its
    world shows at the scene clock's tick (:func:`.stay.clock_row`: a closed
    world computed to the clock's tick, as the world's own updater does),
    and declare which tick it is of. Outside a stay it does nothing. It
    names ``dt``, as the world's updater does, and never reads it: a ticked
    play runs it only where a frame reads. It reads the scene clock and the
    record's bakes, and writes the readout's own family. A play drawn from
    a sealed bake admits it only for a closed world, handing it the bake's
    row through :func:`.stay.clock_row`.
    """
    shows = readout._shows
    found = clock_row(shows.world, stepping=False)
    if found is None:
        return
    stay, tick = found
    (quantity,) = measures.body_quantities(
        bakes_of(stay.record), stay.ordinal, shows.body, tick, (shows.quantity,)
    )
    if quantity.value is None:
        return
    readout._show(shows.scale * _part(quantity.value, shows.part))
    shown = ShownQuantity(
        shows.quantity, shows.part, shows.scale, shows.body, stay.ordinal, tick
    )
    held = readout.__dict__.get(SOURCES, ())
    if shown not in held:
        kept = tuple(each for each in held if type(each) is not ShownQuantity)
        setattr(readout, SOURCES, (*kept, shown))


def _part(value: Any, part: str) -> float:
    """The part ``part`` of a quantity's value."""
    if part == "value":
        return float(value)
    if part == "norm":
        return math.hypot(*value)
    return float(value["xyz".index(part)])


def parts_of(dimension: int, quantity: str) -> tuple[str, ...]:
    """The parts a readout shows of ``quantity`` (``sim.velocity``) in a
    world of ``dimension``, the default first.
    """
    if quantity == "sim.contact_count" or (
        quantity == "sim.angular_velocity" and dimension == 2
    ):
        return WHOLE
    return ("norm", "x", "y", "z") if dimension == 3 else ("norm", "x", "y")


class BodyReadout(Readout):
    """A readout of the quantity ``quantity`` of the body ``body`` (a
    :class:`.Body` handle), shown at every frame: ``"position"``,
    ``"velocity"``, ``"angular_velocity"`` or ``"contact_count"``, its part
    ``part`` (``"norm"``, the default for a vector, ``"x"``, ``"y"``, or
    ``"z"`` in 3D; ``"value"`` for a quantity that is one number: a count, a
    2D world's angular velocity) times ``scale`` (100 to show centimetres of
    a world in metres; -1 for a speed downward), in ``format``. The other
    arguments are :class:`~arachne.mobject.annotation.Readout`'s.

    :meth:`.World.readout` makes one. Its updater, :func:`show_quantity`,
    sets its value; what it declares and the tick it shows are this
    module's.

    Not content (:attr:`_content_key_omits`): what it reads (its world and
    body, a handle) and its declared source; its value, text, format and
    hold key as every readout's do.
    """

    _content_key_omits: ClassVar[frozenset[str]] = frozenset({"_shows", SOURCES})

    def __init__(
        self,
        body: Body,
        quantity: str,
        *,
        part: str | None = None,
        scale: float = 1.0,
        format: str = "{:.1f}",  # noqa: A002 (the readout's own word)
        at: Anchor | None = None,
        **options: Any,
    ) -> None:
        if not isinstance(body, Body):
            raise TypeError(f"A body readout reads a body (a Body), not {body!r}.")
        name = READABLE.get(quantity) if isinstance(quantity, str) else None
        if name is None:
            raise ValueError(
                f"A readout shows a body's {', '.join(READABLE)}; not {quantity!r}."
            )
        parts = parts_of(body.world.dimension, name)
        chosen = parts[0] if part is None else part
        if chosen not in parts:
            raise ValueError(
                f"A readout of {quantity} in a {body.world.dimension}D world shows"
                f" its {', '.join(parts)}; not {part!r}."
            )
        factor = float(scale)
        if not math.isfinite(factor) or factor == 0:
            raise ValueError(
                f"A readout's scale is a finite number other than 0, not {scale!r}."
            )
        super().__init__(0.0, format, at, **options)
        self._shows = _Shows(body.world, body.index, name, chosen, factor)
        quantities.attach(self, ShownQuantity(name, chosen, factor, body.index))
        # Before the attachment's own updater, so that it places the text
        # this one sets.
        self.add_updater(show_quantity, index=0)  # type: ignore[arg-type]
        show_quantity(self, 0)

    def set_value(self, value: float) -> Self:
        """Refused: a body readout shows its body's quantity, set by its
        world at every frame (a :class:`~arachne.mobject.annotation.CountUp`
        of it is refused as it begins, for the same reason).
        """
        raise TypeError(
            "A body readout shows its body's quantity, which its world sets at"
            " every frame: it is not set by hand (set_value) or counted"
            " (CountUp). To show a number of your own, use a Readout and set"
            " it between plays, from world.at_tick(...)."
        )

    def _show(self, value: float) -> None:
        """Show ``value``: its updater's own path to the readout's."""
        Readout.set_value(self, value)
