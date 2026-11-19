"""The lint's reading of declared quantities (:mod:`arachne.quantities`).

At each lint instant every mobject the frame's capture reaches is read for
its declared quantities, as ``state_at`` reads them there; each one that
cannot be read (its value answered as null with the reason) is a
``sim.quantity_unreadable`` finding about its object, over the run of
consecutive lint instants at which it stays unreadable for that reason.
"""

from __future__ import annotations

__all__ = ["request_quantities", "unreadable_findings"]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from arachne.quantities import SOURCES, At, declares, read
from arachne.timeline.draw import Drawing
from arachne.timeline.state import _scene_box

from .findings import Finding, FindingObject

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, PlayRecord, Record
    from arachne.timeline.source import SourceLine

    from .frames import Frames
    from .instants import LintInstant

#: Who the quantities read at the lint instants are asked for by.
OWNER = "quantities"


@dataclass(frozen=True)
class _Unreadable:
    """A quantity of an object that could not be read at an instant."""

    thing: FindingObject
    line: SourceLine | None
    name: str
    unit: str
    reason: str


def request_quantities(frames: Frames, instants: Iterable[LintInstant]) -> None:
    """Ask for the quantities read at each lint instant."""
    frames.request(OWNER, [instant.t for instant in instants], _read)


def _read(
    record: Record,
    t: Fraction,
    play: PlayRecord | None,
    capture: Capture,
    live: bool,
) -> tuple[_Unreadable, ...]:
    """The quantities at ``capture`` that cannot be read: from the states
    the record holds (a forward run's scene holds them too, ``live``), read
    at the capture's instant.
    """
    at = At.of(capture.instant, record.bakes)
    faults: list[tuple[Mobject, Mapping[str, Any], str, str, str]] = []
    for mob, state in record._walk(capture, capture.draw_list()):
        if not declares(type(mob), state.get(SOURCES)):
            continue
        held = MappingProxyType(dict(zip(state.keys, state.values, strict=True)))
        faults.extend(
            (mob, held, quantity.name, quantity.unit, quantity.unreadable)
            for quantity in read(type(mob), held, at)
            if quantity.unreadable is not None
        )
    if not faults:
        return ()
    names = play.names if play is not None else record.final_names
    if live:
        return tuple(_unreadable(record, names, *fault) for fault in faults)
    with Drawing(record) as drawing:
        # The boxes are of the capture's points, as state_at gives them.
        drawing.show(capture, capture.draw_list())
        return tuple(_unreadable(record, names, *fault) for fault in faults)


def _unreadable(
    record: Record,
    names: Mapping[int, str],
    mob: Mobject,
    held: Mapping[str, Any],
    name: str,
    unit: str,
    reason: str,
) -> _Unreadable:
    origin = record.origin(mob)
    thing = FindingObject(
        id=origin.id,
        source_line=origin.scene_line,
        kind=type(mob).__name__,
        label_or_text=names.get(id(mob)),
        bbox=_scene_box(mob),
        screen_bbox=None,
        name=names.get(id(mob)),
    )
    return _Unreadable(thing, origin.source_line, name, unit, reason)


def unreadable_findings(
    frames: Frames, instants: Iterable[LintInstant]
) -> list[Finding]:
    """A ``sim.quantity_unreadable`` finding for each quantity that could
    not be read, over each run of consecutive lint ``instants`` (those
    :func:`request_quantities` asked for) at which it stayed so for one
    reason.
    """
    request = frames.result(OWNER)
    times = [instant.t for instant in instants]
    open_runs: dict[tuple[str, str, str], tuple[_Unreadable, Fraction, Fraction]] = {}
    found: list[Finding] = []
    for t, faults in zip(times, request, strict=True):
        here = {(f.thing.id, f.name, f.reason): f for f in faults}
        ended = [key for key in open_runs if key not in here]
        found.extend(_finding(*open_runs.pop(key)) for key in ended)
        for key, fault in here.items():
            held = open_runs.get(key)
            open_runs[key] = (fault, held[1], t) if held else (fault, t, t)
    found.extend(_finding(*run) for run in open_runs.values())
    return found


def _finding(fault: _Unreadable, start: Fraction, end: Fraction) -> Finding:
    return Finding(
        "sim.quantity_unreadable",
        start,
        end,
        objects=(fault.thing,),
        message=(
            f"{fault.name} of {fault.thing.id} cannot be read: {fault.reason}; it"
            " is answered as null, never as a number"
        ),
        line=fault.line,
        detail={"name": fault.name, "unit": fault.unit, "reason": fault.reason},
    )
