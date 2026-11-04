"""Departures: the frame a settled thing leaves, swept as it goes.

When what the viewer had settled on departs (a play removes it, as a
fade-out does, or transforms it onto another mobject, or the script takes
it out of the scene with ``remove`` or ``clear``) the frame it leaves is
swept, scoped to what departs (:mod:`.plays` says which animations depart).
Only findings with a text in them are kept: a text cut by the frame's
edge, or a text touching, crowding or covered by something. In a frame the
scene moves on from, a text on a filled shape and a text over a stroke
inside its own halo read clean, as the instant rules say
(``Terms.mid_scene``).

Settled means on screen and untouched by any animation for
:data:`SETTLED_WAIT` of the scene's time: since the later of when the top
level holding it entered the scene and when any animation last touched
anything in its family. A top level enters when the script first calls a
play or a removal with it in the scene; one that leaves the top level
(taken out, or taken into a group made for a play) enters again when it is
back. A hold made of consecutive plays counts as a wait does. Settledness
is judged per member: animating one line of a panel leaves its other lines
settled, and the panel that holds the line in motion is not.

What a play departs is swept in the scene the script called it on, before
the play added or began anything, at the play's start and line (a play too
short to show a frame included); what the script removes, in the scene as
it stood just before the call, at the call's line. Both are states the
execution keeps (``Record.states``). A thing a play
itself took out (a fade-out's cleanup) was swept when that play began, and
is not swept again. A clear forgets every entry and touch.

The scene's time is the engine's exact clock. A decision made within one
frame of the render's grid of :data:`SETTLED_WAIT` (that frame included)
says so in its finding's ``detail.clock_margin`` (seconds past it):
another engine whose plays last whole numbers of frames can decide it the
other way.
"""

from __future__ import annotations

__all__ = ["SETTLED_WAIT", "departures"]

from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING

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
    from arachne.timeline.record import Capture, Record, Removal
    from arachne.timeline.source import SourceLine

    from .core import LintContext
    from .parts import Parts
    from .rules._rule import Terms

#: How long a thing is on screen, untouched, before the viewer has read it.
SETTLED_WAIT = Fraction(1, 2)


def _departures(context: LintContext, rule: ArmedRule) -> list[Finding]:
    # Plays and states, from the one execution the lint's frames come from.
    record = context.frames.execution
    states = rule.states(context)
    grid = render_grid(record)
    clock = _Clock(record)
    found: list[Finding] = []
    removals: dict[int, list[Removal]] = {}
    for removal in states.removals:
        if removal.t >= rule.since:
            removals.setdefault(removal.plays, []).append(removal)

    def removed(removal: Removal) -> None:
        clock.stamp(removal.capture.roots, removal.t)
        settled = clock.settled(removal.capture, removal.t, removal.operands)
        if settled:
            parts = rule.state(context, removal.capture)
            found.extend(
                _sweep(parts, settled, rule.terms, removal.source_line, removal.t, grid)
            )
        if removal.kind == "clear":
            clock.clear()
        else:
            clock.prune(removal.after, removal.capture)

    for index, play in enumerate(record.plays):
        for removal in removals.get(index, ()):
            removed(removal)
        if play.start < rule.since:
            continue
        # The scene as the script called the play, before it added or began
        # anything, and as the play left it, finished and cleaned up.
        called, left = states.calls[play.index], states.exits[play.index]
        clock.stamp(called.roots, play.start)
        leaves = played(play)
        going = [leaf.mobject for leaf in leaves if leaf.departs and leaf.mobject]
        settled = clock.settled(called, play.start, going) if going else {}
        if settled:
            parts = rule.state(context, called)
            found.extend(
                _sweep(parts, settled, rule.terms, play.source_line, play.start, grid)
            )
        touched = [leaf.mobject for leaf in leaves if leaf.mobject is not None]
        clock.touch(touched, left, play.end)
        clock.prune(left.roots, left)
    for removal in removals.get(len(record.plays), ()):
        removed(removal)
    return found


class _Clock:
    """When each top level entered the scene and when an animation last
    touched each member, by stable id, as the scene runs.
    """

    def __init__(self, record: Record) -> None:
        self.record = record
        self.entered: dict[str, tuple[Mobject, Fraction]] = {}
        self.touched: dict[str, Fraction] = {}

    def _id(self, mob: Mobject) -> str:
        return self.record.origin(mob).id

    def stamp(self, tops: Iterable[Mobject], now: Fraction) -> None:
        """The top levels ``tops`` not entered yet enter at ``now``."""
        for top in tops:
            self.entered.setdefault(self._id(top), (top, now))

    def touch(self, mobs: Iterable[Mobject], capture: Capture, now: Fraction) -> None:
        """Every member of ``mobs`` (their families at ``capture``) is touched
        at ``now``.
        """
        for mob in mobs:
            for member in family_at(self.record, mob, capture):
                self.touched[self._id(member)] = now

    def prune(self, tops: Iterable[Mobject], capture: Capture) -> None:
        """Forget the entries of what is no longer a top level (``tops``),
        and the touches of what no entered top level holds (at ``capture``).
        """
        present = {self._id(top) for top in tops}
        for key in [k for k in self.entered if k not in present]:
            del self.entered[key]
        owned = self._owned(capture)
        for key in [k for k in self.touched if k not in owned]:
            del self.touched[key]

    def clear(self) -> None:
        self.entered.clear()
        self.touched.clear()

    def _owned(self, capture: Capture) -> dict[str, Fraction]:
        """Each member of an entered top level (its family at ``capture``),
        with the earliest entry of a top level holding it.
        """
        owned: dict[str, Fraction] = {}
        for top, entry in self.entered.values():
            for member in family_at(self.record, top, capture):
                key = self._id(member)
                owned[key] = min(owned.get(key, entry), entry)
        return owned

    def settled(
        self, capture: Capture, now: Fraction, going: Iterable[Mobject]
    ) -> dict[str, tuple[Animated, Fraction]]:
        """The topmost settled members of ``going`` at ``now`` (families as at
        ``capture``), by stable id, each with how long past settling it was.
        """
        record = self.record
        owned = self._owned(capture)
        found: dict[str, tuple[Animated, Fraction]] = {}

        def walk(mob: Mobject) -> None:
            entry = owned.get(self._id(mob))
            if entry is not None:
                touches = [
                    self.touched[i]
                    for m in family_at(record, mob, capture)
                    if (i := self._id(m)) in self.touched
                ]
                since = max(entry, *touches) if touches else entry
                if now - since >= SETTLED_WAIT:
                    found.setdefault(
                        self._id(mob),
                        (animated(record, mob, capture), now - since - SETTLED_WAIT),
                    )
                    return
            # An unentered wrapper (a play's own group) passes through to
            # what it wraps.
            state = record.node_state(mob, capture)
            for sub in state.submobjects if state is not None else mob.submobjects:
                walk(sub)

        for mob in going:
            walk(mob)
        return found


def _sweep(
    parts: Parts,
    settled: dict[str, tuple[Animated, Fraction]],
    terms: Terms,
    line: SourceLine | None,
    at: Fraction,
    grid: Fraction,
) -> list[Finding]:
    """The findings with a text in them that the departing ``settled``
    members are party to, in the scene ``parts`` shows, found at ``at``.
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
        if margin <= 1 / grid:
            detail["clock_margin"] = float(margin)
        out.append(replace(finding, t_start=at, t_end=at, line=line, detail=detail))
    return out


#: The rule: what settled things leave, swept as they depart.
departures = ArmedRule(_departures)
