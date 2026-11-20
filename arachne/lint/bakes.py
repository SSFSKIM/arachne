"""The lint's reading of what the record's bakes found
(:class:`~arachne.timeline.bake.BakeFinding`).

A simulation's occurrence holds what its solver found at its rows (a body
that went non-finite or above its world's speed limits, ``sim.unstable``;
two bodies that passed through each other, ``sim.tunnelled``). Each becomes
a finding at the instant of its tick, about its bodies by stable id, each
with its boxes in the frame on screen at that instant (null when nothing of
it is drawn there), and with the line of the play running then. They join
the lint's findings as the session's divergences do: whatever the rules,
never ending the process.
"""

from __future__ import annotations

__all__ = ["bake_findings", "request_bake_frames"]

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from arachne.timeline.clock import TICK_RATE

from .findings import Finding, FindingObject
from .instants import render_grid
from .quantities import _boxed

if TYPE_CHECKING:
    from arachne.timeline.bake import BakeFinding
    from arachne.timeline.record import Capture, PlayRecord, Record

    from .frames import FrameReader, Frames
    from .parts import Parts

#: Who the frames read for the bakes' findings are asked for by.
OWNER = "bakes"

#: How to fix each signal of ``sim.unstable``: one line.
_FIXES = {
    "non_finite": (
        "Keep every force on the body finite: two attracting bodies that meet, a"
        " spring or callback that divides by a vanishing distance, or a pushed"
        " number that overflows makes it infinite."
    ),
    "speed": (
        "Calm what drives the body (a force that grows without bound, a joint or"
        " spring too stiff or too soft for the tick); if it is meant to move that"
        " fast, raise the world's speed_limit, below the solver's cap of 400 u/s."
    ),
    "angular_speed": (
        "Calm what spins the body (a torque that grows without bound, a joint or"
        " spring too stiff or too soft for the tick); if it is meant to spin that"
        " fast, raise the world's angular_speed_limit, below the solver's cap of"
        " 45 degrees a tick."
    ),
}


@dataclass(frozen=True)
class Pending:
    """A bake's finding, at its instant, and the frame on screen then."""

    found: BakeFinding
    t: Fraction
    frame: Fraction


def request_bake_frames(frames: Frames, record: Record) -> tuple[Pending, ...]:
    """The findings ``record``'s bakes hold, each with the frame on screen at
    its instant, which is asked of ``frames`` for the boxes of its bodies.
    """
    bakes = record.bakes
    if bakes is None:
        return ()
    found = [
        (each, Fraction(occurrence.first_tick + each.row, TICK_RATE))
        for occurrence in bakes.occurrences
        for each in occurrence.findings
    ]
    if not found:
        return ()
    pending = tuple(Pending(each, t, _on_screen(record, t)) for each, t in found)
    ids = frozenset(id_ for each in pending for id_ in each.found.ids)
    frames.request(
        OWNER, sorted({each.frame for each in pending}), _reader(frames, ids)
    )
    return pending


def _on_screen(record: Record, t: Fraction) -> Fraction:
    """The instant of the frame on screen at ``t``: the last frame of the
    render's grid at or before it, or the scene's end.
    """
    if t >= record.duration:
        return record.duration
    grid = render_grid(record)
    return Fraction(math.floor(t * grid)) / grid


def _reader(
    frames: Frames, ids: frozenset[str]
) -> FrameReader[Mapping[str, FindingObject]]:
    """A reader of the bodies ``ids`` as a frame shows them: each as a
    finding's object, with its boxes there.
    """
    from .core import parts_reader

    parts_of = parts_reader(frames.session, frames.parts)

    def read(
        record: Record,
        t: Fraction,
        play: PlayRecord | None,
        capture: Capture,
        live: bool,
    ) -> Mapping[str, FindingObject]:
        drawn: Parts = parts_of(record, t, play, capture, live)
        names = play.names if play is not None else record.final_names
        found: dict[str, FindingObject] = {}
        for mob, _state in record._walk(capture, capture.draw_list()):
            origin = record.origin(mob)
            if origin.id not in ids or origin.id in found:
                continue
            thing = FindingObject(
                id=origin.id,
                source_line=origin.scene_line,
                kind=type(mob).__name__,
                label_or_text=names.get(id(mob)),
                bbox=None,
                screen_bbox=None,
                name=names.get(id(mob)),
            )
            found[origin.id] = _boxed(thing, drawn)
        return found

    return read


def bake_findings(
    frames: Frames, record: Record, pending: Sequence[Pending]
) -> list[Finding]:
    """Each of ``pending`` as a finding, its bodies boxed as the frame on
    screen at its instant shows them.
    """
    if not pending:
        return []
    shown = dict(
        zip(sorted({each.frame for each in pending}), frames.result(OWNER), strict=True)
    )
    out = []
    for each in pending:
        found = each.found
        objects = shown.get(each.frame, {})
        things = tuple(
            objects.get(id_) or FindingObject(id_, None, "Mobject", None, None, None)
            for id_ in found.ids
        )
        play = record.play_at(each.t)
        detail = dict(found.detail)
        out.append(
            Finding(
                found.code,
                each.t,
                each.t,
                objects=things,
                message=_message(found, things),
                line=play.source_line if play is not None else None,
                detail=detail,
                fix=_FIXES.get(str(detail.get("signal")), ""),
            )
        )
    return out


def _message(found: BakeFinding, things: Sequence[FindingObject]) -> str:
    detail = found.detail
    at = f"at tick {detail.get('tick')}"
    if found.code == "sim.tunnelled":
        a, b = (thing.id for thing in things)
        return (
            f"{a} and {b} passed through each other in one tick, {at}, with no"
            " contact between them: the picture shows them crossing"
        )
    body = things[0].id
    signal = detail.get("signal")
    if signal == "non_finite":
        return (
            f"{body}'s simulated state went non-finite {at}: the solver stopped it"
            " where it last was, so its motion from there is not computed"
        )
    which = "speed" if signal == "speed" else "angular speed"
    unit = "u/s" if signal == "speed" else "rad/s"
    return (
        f"{body}'s {which} rose to {detail.get('value'):.6g} {unit} {at}, above its"
        f" world's limit of {detail.get('limit'):.6g} {unit}"
    )
