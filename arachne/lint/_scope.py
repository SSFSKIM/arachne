"""What the temporal rules scope their questions by: the mobjects a play
animates, as stable ids at a capture, and which units those name.

A unit and an animated mobject name each other when either holds the
other, or they are one: the mobject is in what the unit is made from, or
the unit is in the mobject's family. Under a ThreeDCamera only what is
fixed in the frame is asked about: everything else is projected through
the camera's motion, which the rules' tolerances were not set for.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .plays import Played, family_at

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, PlayRecord, Record

    from .parts import Parts
    from .rules._units import Measured


@dataclass(frozen=True)
class Animated:
    """A mobject an animation names: its stable id and its family's."""

    id: str
    family: frozenset[str]


def animated(record: Record, mob: Mobject, capture: Capture | None) -> Animated:
    """``mob`` as stable ids, its family as it was at ``capture``."""
    members = family_at(record, mob, capture) if capture is not None else [mob]
    return Animated(
        record.origin(mob).id, frozenset(record.origin(m).id for m in members)
    )


def animated_by(
    record: Record,
    play: PlayRecord,
    leaves: Iterable[Played],
    *,
    at: str = "start",
) -> list[Animated]:
    """The mobjects ``leaves`` of ``play`` animate, their families read at
    the play's first evaluated instant (``at="start"``) or its last
    (``"end"``).
    """
    capture = _capture(play, at)
    return [
        animated(record, leaf.mobject, capture)
        for leaf in leaves
        if leaf.mobject is not None
    ]


def names(unit: Measured, mob: Animated) -> bool:
    """Whether ``unit`` and ``mob`` name each other (one holds the other)."""
    return unit.id in mob.family or any(h.id == mob.id for h in unit.unit.family)


def named_by(unit: Measured, mobs: Iterable[Animated]) -> bool:
    return any(names(unit, mob) for mob in mobs)


def asked(parts: Parts, units: Iterable[Measured]) -> list[Measured]:
    """``units``, less what a ThreeDCamera projects: only what it keeps
    fixed in the frame (the unit, or a leaf of it, fixed) is asked there.
    """
    if parts.view.camera != "three_d":
        return list(units)
    fixed = parts.fixed
    return [
        u for u in units if u.id in fixed or any(leaf.id in fixed for leaf in u.leaves)
    ]


def _capture(play: PlayRecord, at: str) -> Capture | None:
    if not play.samples:
        return None
    return play.samples[0] if at == "start" else play.samples[-1]
