"""Departures: the frame a settled thing leaves, swept as it goes.

When what the viewer had settled on departs (a play removes it, as a
fade-out does, or transforms it onto another mobject, or the script takes
it out of the scene) the frame it leaves is swept, scoped to what departs
(:mod:`.plays` says which animations depart). Only findings with a text in
them are kept: a text cut by the frame's edge, or a text touching,
crowding or covered by something. In a frame the scene moves on from, a
text on a filled shape and a text over a stroke inside its own halo read
clean, as the instant rules say (``Terms.mid_scene``).

Settled means on screen and untouched by any animation for
:data:`SETTLED_WAIT` of the scene's time: since the later of when the top
level holding it entered the scene and when any animation last touched
anything in its family. A hold made of consecutive plays counts as a wait
does. Settledness is judged per member: animating one line of a panel
leaves its other lines settled, and the panel that holds the line in
motion is not. What a play departs is swept at the play's first frame;
what the script removes between two plays, at the last frame of the play
before. A thing a play itself took out (a fade-out's cleanup) was swept
when that play began, and is not swept again.

The scene's time is the engine's exact clock. A decision made within one
frame of the render's grid of :data:`SETTLED_WAIT` says so in its
finding's ``detail.clock_margin`` (seconds past it): another engine whose
plays last whole numbers of frames can decide it the other way.
"""

from __future__ import annotations

__all__ = ["SETTLED_WAIT", "departures"]

from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING

from arachne.timeline.clock import frames_in

from ._scope import Animated, animated, asked, names
from .armed import ArmedRule
from .findings import Finding
from .instants import render_grid
from .plays import family_at, played
from .rules._pairs import pair_findings
from .rules._units import measured_units
from .rules.frame_edge import edge_findings

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, Record
    from arachne.timeline.source import SourceLine

    from .core import LintContext
    from .parts import Parts
    from .rules._rule import Terms

#: How long a thing is on screen, untouched, before the viewer has read it.
SETTLED_WAIT = Fraction(1, 2)


def _frames(record: Record, since: Fraction) -> list[Fraction]:
    """Each play's first and last frame, from the play armed at on."""
    grid = render_grid(record)
    times = []
    for play in record.plays:
        frames = frames_in(play.start, play.end, grid)
        if play.start >= since and frames:
            times += [frames[0], frames[-1]]
    return times


def _departures(context: LintContext, rule: ArmedRule) -> list[Finding]:
    record = context.record
    terms, since = rule.terms, rule.since
    frame = 1 / render_grid(record)
    read = iter(rule.frames(context))
    shown = {
        play.index: (next(read), next(read))
        for play in record.plays
        if play.start >= since and frames_in(play.start, play.end, render_grid(record))
    }
    entered: dict[str, Fraction] = {}
    touched: dict[str, Fraction] = {}
    found: list[Finding] = []
    plays = record.plays
    for index, play in enumerate(plays):
        if index not in shown or not play.samples:
            continue
        first, last = shown[index]
        start = play.samples[0]
        tops = _tops(record, start)
        entered = {key: entered.get(key, play.start) for key in tops}
        leaves = played(play)
        going = [leaf.mobject for leaf in leaves if leaf.departs and leaf.mobject]
        settled = _settled(record, start, tops, entered, touched, play.start, going)
        if settled:
            found += _sweep(first, settled, terms, play.source_line, frame)
        end = play.samples[-1]
        moved: set[str] = set()
        for leaf in leaves:
            if leaf.mobject is not None:
                ids = animated(record, leaf.mobject, end).family
                moved |= ids
                touched.update(dict.fromkeys(ids, play.end))
        following = plays[index + 1] if index + 1 < len(plays) else None
        after = (
            following.samples[0]
            if following is not None and following.samples
            else record.final
        )
        if after is None:
            continue
        kept = {record.origin(m).id for m in after.roots}
        before = _tops(record, end)
        removed = [
            mob for key, mob in before.items() if key not in kept and key not in moved
        ]
        if removed:
            entered = {key: entered.get(key, play.end) for key in before}
            settled = _settled(record, end, before, entered, touched, play.end, removed)
            if settled:
                found += _sweep(last, settled, terms, None, frame)
    return found


def _tops(record: Record, capture: Capture) -> dict[str, Mobject]:
    return {record.origin(m).id: m for m in capture.roots}


def _settled(
    record: Record,
    capture: Capture,
    tops: dict[str, Mobject],
    entered: dict[str, Fraction],
    touched: dict[str, Fraction],
    now: Fraction,
    going: list[Mobject],
) -> dict[str, tuple[Animated, Fraction]]:
    """The topmost settled members of ``going`` at ``now`` (families as at
    ``capture``), by stable id, each with how long past settling it was.
    """
    owned: dict[str, Fraction] = {}
    for key, top in tops.items():
        for member in family_at(record, top, capture):
            mid = record.origin(member).id
            owned[mid] = min(owned.get(mid, entered[key]), entered[key])
    found: dict[str, tuple[Animated, Fraction]] = {}

    def walk(mob: Mobject) -> None:
        key = record.origin(mob).id
        entry = owned.get(key)
        if entry is not None:
            members = family_at(record, mob, capture)
            touches = [
                touched[i] for m in members if (i := record.origin(m).id) in touched
            ]
            since = max(entry, *touches) if touches else entry
            if now - since >= SETTLED_WAIT:
                found.setdefault(
                    key, (animated(record, mob, capture), now - since - SETTLED_WAIT)
                )
                return
        state = record.node_state(mob, capture)
        for sub in state.submobjects if state is not None else ():
            walk(sub)

    for mob in going:
        walk(mob)
    return found


def _sweep(
    parts: Parts,
    settled: dict[str, tuple[Animated, Fraction]],
    terms: Terms,
    line: SourceLine | None,
    frame: Fraction,
) -> list[Finding]:
    """The findings with a text in them that the departing ``settled``
    members are party to, in the frame ``parts`` shows.
    """
    units = asked(parts, measured_units(parts))
    margins: dict[str, Fraction] = {}
    for unit in units:
        for mob, margin in settled.values():
            if names(unit, mob):
                margins[unit.id] = min(margins.get(unit.id, margin), margin)
    focus = [u for u in units if u.id in margins]
    if not focus:
        return []
    found = [
        *edge_findings(parts, replace(terms, mid_scene=False), among=focus),
        *pair_findings(parts, terms, among=units, focus=margins),
    ]
    out = []
    for finding in found:
        if finding.code == "contact.held":
            continue
        margin = min(margins[o.id] for o in finding.objects if o.id in margins)
        detail = dict(finding.detail)
        if margin < frame:
            detail["clock_margin"] = float(margin)
        out.append(replace(finding, line=line, detail=detail))
    return out


#: The rule: what settled things leave, swept as they depart.
departures = ArmedRule(_departures, reads=_frames)
