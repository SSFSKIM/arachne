"""The lint's reading of body readouts (:mod:`arachne.simulation.readout`).

At each lint instant every body readout the frame's capture draws is read
for the tick its value is of (its declared ``sim.reads_tick``). One more
than a tick behind the capture's tick, while its body's world is in the
scene there (a stay of the world holds the capture's tick, and goes on
past it), is a ``sim.readout_stale`` finding about the readout, over each
run of consecutive lint instants at which it stays so: something held its
updater (an animation of the readout itself, such as ``FadeIn``) while the
body moved. A tick behind is the lag a readout added before an open world
shows by design, and is not reported.
"""

from __future__ import annotations

__all__ = ["readout_findings", "request_readouts"]

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING

from arachne.quantities import SOURCES
from arachne.timeline.clock import TICK_RATE

from .findings import Finding, FindingObject
from .quantities import _boxed

if TYPE_CHECKING:
    from arachne.timeline.bake import Bakes
    from arachne.timeline.record import Capture, PlayRecord, Record
    from arachne.timeline.source import SourceLine

    from .frames import Frames
    from .instants import LintInstant
    from .parts import Parts

#: Who the readouts read at the lint instants are asked for by.
OWNER = "readouts"
#: How many ticks behind its world a readout may show: an open world's
#: readout added before the world shows the tick before.
LAG = 1


@dataclass(frozen=True)
class _Stale:
    """A readout showing an old tick at an instant."""

    thing: FindingObject
    line: SourceLine | None
    body: str
    tick: int
    scene_tick: int


def request_readouts(frames: Frames, instants: Iterable[LintInstant]) -> None:
    """Ask for the body readouts read at each lint instant."""
    frames.request(OWNER, [instant.t for instant in instants], _read)


def _read(
    record: Record,
    t: Fraction,
    play: PlayRecord | None,
    capture: Capture,
    live: bool,
) -> tuple[_Stale, ...]:
    """The body readouts ``capture`` draws that show a tick more than
    :data:`LAG` behind its own while their bodies' worlds are in the scene.
    """
    bakes = record.bakes
    if bakes is None:
        return ()
    from arachne.simulation.readout import BodyReadout, ShownQuantity

    scene_tick = math.floor(capture.instant * TICK_RATE)
    names = play.names if play is not None else record.final_names
    found: list[_Stale] = []
    for mob, state in record._walk(capture, capture.draw_list()):
        if not isinstance(mob, BodyReadout):
            continue
        sources = state.get(SOURCES) or ()
        shown = next((s for s in sources if type(s) is ShownQuantity), None)
        if shown is None or shown.occurrence is None or shown.tick is None:
            continue
        if scene_tick - shown.tick <= LAG:
            continue
        body = _body_id(bakes, shown.occurrence, shown.body)
        if body is None or not _in_the_scene(bakes, body, scene_tick):
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
        found.append(_Stale(thing, origin.source_line, body, shown.tick, scene_tick))
    return tuple(found)


def _body_id(bakes: Bakes, ordinal: int, body: int) -> str | None:
    """The stable id of the body ``body`` of the occurrence ``ordinal``."""
    occurrence = bakes.occurrence(ordinal)
    if occurrence is None:
        return None
    return next(
        (
            binding.id
            for binding in occurrence.bindings
            if binding.channel == "pose" and binding.index == body
        ),
        None,
    )


def _in_the_scene(bakes: Bakes, body: str, tick: int) -> bool:
    """Whether a stay of the world of the body ``body`` holds ``tick`` and
    goes on past it.
    """
    return any(
        occurrence.first_tick <= tick < occurrence.last_tick
        and any(
            binding.id == body and binding.channel == "pose"
            for binding in occurrence.bindings
        )
        for occurrence in bakes.occurrences
    )


def readout_findings(
    frames: Frames, instants: Iterable[LintInstant], parts: Sequence[Parts]
) -> list[Finding]:
    """A ``sim.readout_stale`` finding for each body readout that showed an
    old tick, over each run of consecutive lint ``instants`` (those
    :func:`request_readouts` asked for, whose parts are ``parts``) at which
    it stayed so.
    """
    request = frames.result(OWNER)
    times = [instant.t for instant in instants]
    open_runs: dict[str, tuple[_Stale, Fraction, Fraction]] = {}
    found: list[Finding] = []
    for t, stale, drawn in zip(times, request, parts, strict=True):
        here = {each.thing.id: each for each in stale}
        ended = [key for key in open_runs if key not in here]
        found.extend(_finding(*open_runs.pop(key)) for key in ended)
        for key, each in here.items():
            held = open_runs.get(key)
            if held is None:
                open_runs[key] = (replace(each, thing=_boxed(each.thing, drawn)), t, t)
            else:
                open_runs[key] = (held[0], held[1], t)
    found.extend(_finding(*run) for run in open_runs.values())
    return found


def _finding(stale: _Stale, start: Fraction, end: Fraction) -> Finding:
    return Finding(
        "sim.readout_stale",
        start,
        end,
        objects=(stale.thing,),
        message=(
            f"the readout of {stale.body} shows tick {stale.tick} while the scene"
            f" is at tick {stale.scene_tick}: an animation of the readout (e.g."
            " FadeIn) suspended its updater"
        ),
        line=stale.line,
        detail={
            "body": stale.body,
            "tick": stale.tick,
            "scene_tick": stale.scene_tick,
        },
    )
