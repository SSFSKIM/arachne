"""The lint's reading of declared quantities (:mod:`arachne.quantities`).

At each lint instant every mobject the frame's capture reaches is read for
its declared quantities, as ``state_at`` reads them there; each one that
cannot be read (its value answered as null with the reason) is a
``sim.quantity_unreadable`` finding about its object, over the run of
consecutive lint instants at which it stays unreadable for that reason. The
object's boxes are those of the parts the lint read of it at the run's first
instant (its own, or those of its family): in scene units and in pixels of
the target resolution, null when nothing of it is drawn there.
"""

from __future__ import annotations

__all__ = ["request_quantities", "unreadable_findings"]

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING

from arachne.quantities import SOURCES, At, declares, read

from .findings import Finding, FindingObject
from .parts import target_pixels
from .rules._geometry import union

if TYPE_CHECKING:
    from arachne.timeline.record import Capture, PlayRecord, Record
    from arachne.timeline.source import SourceLine

    from .frames import Frames
    from .instants import LintInstant
    from .parts import Parts

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
    names = play.names if play is not None else record.final_names
    found: list[_Unreadable] = []
    for mob, state in record._walk(capture, capture.draw_list()):
        if not declares(type(mob), state.get(SOURCES)):
            continue
        held = MappingProxyType(dict(zip(state.keys, state.values, strict=True)))
        faults = [q for q in read(type(mob), held, at) if q.unreadable is not None]
        if not faults:
            continue
        origin = record.origin(mob)
        thing = FindingObject(
            id=origin.id,
            source_line=origin.scene_line,
            kind=type(mob).__name__,
            label_or_text=names.get(id(mob)),
            bbox=None,
            screen_bbox=None,
            name=names.get(id(mob)),
        )
        found.extend(
            _Unreadable(thing, origin.source_line, q.name, q.unit, str(q.unreadable))
            for q in faults
        )
    return tuple(found)


def _boxed(thing: FindingObject, parts: Parts) -> FindingObject:
    """``thing`` with the boxes of what ``parts`` drew of it: the parts that
    are it or that it holds.
    """
    drawn = [
        part
        for part in parts.parts
        if part.id == thing.id or any(holder.id == thing.id for holder in part.within)
    ]
    if not drawn:
        return thing
    return replace(
        thing,
        bbox=union([part.scene_box for part in drawn]),
        screen_bbox=target_pixels(parts, union([part.screen_box for part in drawn])),
    )


def unreadable_findings(
    frames: Frames, instants: Iterable[LintInstant], parts: Sequence[Parts]
) -> list[Finding]:
    """A ``sim.quantity_unreadable`` finding for each quantity that could
    not be read, over each run of consecutive lint ``instants`` (those
    :func:`request_quantities` asked for, whose parts are ``parts``) at which
    it stayed so for one reason.
    """
    request = frames.result(OWNER)
    times = [instant.t for instant in instants]
    open_runs: dict[tuple[str, str, str], tuple[_Unreadable, Fraction, Fraction]] = {}
    found: list[Finding] = []
    for t, faults, drawn in zip(times, request, parts, strict=True):
        here: Mapping[tuple[str, str, str], _Unreadable] = {
            (f.thing.id, f.name, f.reason): f for f in faults
        }
        ended = [key for key in open_runs if key not in here]
        found.extend(_finding(*open_runs.pop(key)) for key in ended)
        for key, fault in here.items():
            held = open_runs.get(key)
            if held is None:
                open_runs[key] = (
                    replace(fault, thing=_boxed(fault.thing, drawn)),
                    t,
                    t,
                )
            else:
                open_runs[key] = (held[0], held[1], t)
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
