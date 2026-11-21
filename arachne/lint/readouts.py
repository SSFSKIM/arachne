"""The lint's reading of body readouts (:mod:`arachne.simulation.readout`).

At each lint instant every body readout the frame's capture draws is read
for the tick its value is of (its declared ``sim.reads_tick``). One more
than a tick behind the tick of the row its body is drawn at there, while
its body's world is in the scene (the capture holds the world: not at the
instant it leaves, which the next play's capture answers without it), is a
``sim.readout_stale`` finding about the readout, over each
run of consecutive lint instants at which it stays so: something held its
updater (an animation of the readout itself, such as ``FadeIn``) while the
body moved. The body is drawn at the capture's tick, unless an animation
holds its world's drawing: then at the row the world last placed it at,
which its capture's source names (:attr:`.measures.SimBody.held`), so a
world faded in with its readout is not reported. A tick behind is the lag
a readout added before an open world shows by design, and is not reported.
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
    drawn_tick: int


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
    from arachne.simulation.measures import EARLIER, SimBody
    from arachne.simulation.readout import BodyReadout, ShownQuantity

    scene_tick = math.floor(capture.instant * TICK_RATE)
    names = play.names if play is not None else record.final_names
    walked = list(record._walk(capture, capture.draw_list()))
    present = _members(record, capture)
    found: list[_Stale] = []
    for mob, state in walked:
        if not isinstance(mob, BodyReadout):
            continue
        sources = state.get(SOURCES) or ()
        shown = next((s for s in sources if type(s) is ShownQuantity), None)
        if shown is None or shown.occurrence is None or shown.tick is None:
            continue
        if scene_tick - shown.tick <= LAG:
            continue
        # Its world in the scene at the capture: a member of its mobjects'
        # families, as a world's stay counts it (not a capture dependency).
        shows = state.get("_shows")
        if shows is None or id(shows.world) not in present:
            continue
        # The row its body is drawn at: the capture's, unless the world's
        # drawing is held, as the body's source names.
        drawn_tick = scene_tick
        bodies = shows.world._body_mobjects
        drawn = (
            record.node_state(bodies[shown.body], capture)
            if 0 <= shown.body < len(bodies)
            else None
        )
        source = next(
            (
                s
                for s in (drawn.get(SOURCES) or () if drawn else ())
                if type(s) is SimBody
            ),
            None,
        )
        if (
            source is not None
            and source.ordinal == shown.occurrence
            and source.left is None
            and source.held is not None
            and source.held != EARLIER
        ):
            drawn_tick = source.held
        if drawn_tick - shown.tick <= LAG:
            continue
        body = _body_id(record, bakes, shown.occurrence, shown.body)
        if body is None:
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
        found.append(_Stale(thing, origin.source_line, body, shown.tick, drawn_tick))
    return tuple(found)


def _members(record: Record, capture: Capture) -> set[int]:
    """The ids of the scene's mobjects at ``capture`` and their families,
    through submobjects only: what is in the scene, as a world's stay counts
    it (``Scene._membership_changed``), and not what a mobject keeps as a
    capture dependency (an effect's mask or window).
    """
    seen: set[int] = set()
    pending = list(capture.draw_list())
    while pending:
        mob = pending.pop()
        if id(mob) in seen:
            continue
        seen.add(id(mob))
        state = record.node_state(mob, capture)
        if state is not None:
            pending.extend(state.submobjects)
    return seen


def _body_id(record: Record, bakes: Bakes, ordinal: int, body: int) -> str | None:
    """The stable id of the body ``body`` of the occurrence ``ordinal``: from
    its bindings once sealed, else from the stay that computes it (a forward
    run's, read while it runs).
    """
    occurrence = bakes.occurrence(ordinal)
    if occurrence is None:
        try:
            live = bakes.live(ordinal)
        except KeyError:
            return None
        mobjects = getattr(getattr(live, "world", None), "_body_mobjects", ())
        if not 0 <= body < len(mobjects):
            return None
        return record.origin(mobjects[body]).id
    return next(
        (
            binding.id
            for binding in occurrence.bindings
            if binding.channel == "pose" and binding.index == body
        ),
        None,
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
            f"the readout of {stale.body} shows tick {stale.tick} while its body"
            f" is drawn at tick {stale.drawn_tick}: an animation of the readout"
            " (e.g. FadeIn) suspended its updater"
        ),
        line=stale.line,
        detail={
            "body": stale.body,
            "tick": stale.tick,
            "drawn_tick": stale.drawn_tick,
        },
    )
