"""The numbers the marks carry, declared (:mod:`arachne.quantities`): each
mark that carries one answers it as ``mark.<attribute>``, read from the
attribute a capture holds, in its unit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from arachne import quantities
from arachne.quantities import At, Quantity

from .arrows import FlowArrow
from .lines import WavyLine
from .particles import ChargedParticle, DottedTrail, Particles, WavefrontArcs
from .shapes import Vessel

quantities.register("mark.", owner="arachne.mobject.marks", units=("u", "s"))

#: Each mark that carries a number: the attribute it keeps it in, and the
#: number's unit (``u``, a scene unit as the mark was built; ``s``, seconds
#: of scene time).
CARRIED: dict[type, tuple[str, str]] = {
    Vessel: ("level", "u"),
    Particles: ("count", "1"),
    WavefrontArcs: ("spacing", "u"),
    DottedTrail: ("interval", "s"),
    WavyLine: ("wavelengths", "1"),
    FlowArrow: ("magnitude", "1"),
    ChargedParticle: ("charge", "1"),
}


def _carried(
    attribute: str, unit: str
) -> Callable[[Mapping[str, Any], At], Iterable[Quantity]]:
    """The provider of a mark keeping its number in ``attribute``."""
    name = f"mark.{attribute}"

    def declared(held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
        return (Quantity(name, held[attribute], unit),)

    return declared


for _mark, (_attribute, _unit) in CARRIED.items():
    quantities.provider(_mark)(_carried(_attribute, _unit))
