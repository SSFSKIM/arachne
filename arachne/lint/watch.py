"""The hold watch: what stayed wrong on screen long enough to be watched.

Armed, the watch samples the scene from then on: as each play begins,
every :data:`HOLD_SAMPLE` of the scene's time while it plays (the first
frame of the render's grid at least that long after the sample before,
counted from the play's start; a frozen frame changes nothing and has no
such samples), and as it ends. What it finds at a sample stays open while
it is found again, and once it has been on screen for :data:`HOLD_PERSIST`
from the sample it was first found at to the last it was found at, it is
reported, ``held``, with that span. It asks four things:

- **a text at the frame's edge**: its glyphs within :data:`TEXT_MARGIN` of
  an edge, or past it (``clip.frame_edge``);
- **a shape that has left**: a thing on screen (filled, closed, or an
  arrow; an open stroke is not asked) that was wholly inside the frame
  earlier and now runs more than :data:`SHAPE_EXIT_TOL` past an edge,
  unless it has grown past :data:`SHAPE_GROWTH` times its size when last
  wholly inside (``clip.frame_edge``);
- **a text struck or covered**: the instant rules' contacts and occlusions
  for every pair with a text in it, under the scene's terms, a text on a
  filled shape and a text over a stroke inside its own halo reading clean
  mid-scene (``overlap.text``, ``gap.text``, ``occlusion.glyph``);
- **a label left behind**: a short text (:data:`LABEL_GLYPHS` glyphs or
  fewer) that sat still within :data:`LABEL_ATTACH` of a still shape, and
  stays where it was while that shape moves off, unchanged in size, until
  it lies :data:`LABEL_LEFT` further from it than it began
  (``label.left_behind``). A label grouped with shapes names one of its
  own group; while a shape in reach is still being animated the label is
  asked again later; a closed outline around the label and another text
  frames them, and names neither.

What the current play takes away or pulses (:mod:`.plays`) is not asked
while it plays, nor is a contact between two things the play carries
together. What a play carries wholly off the frame left on purpose: the
edge it crossed on the way out was transit, and it is not asked again until
it is back on the frame. A camera away from its home frame is asked only
about texts its edge cuts. Under a ThreeDCamera only what it keeps fixed in
the frame is asked.

A play's first frame is read as the scene when it called the play
(nothing in it is in motion yet). As it ends, the scene is its last frame
less what it removes, standing for the play's end: the frame after it
already shows what the script changed between the two plays.

A finding is keyed by the unit it names (:class:`~arachne.lint.parts.Unit`),
so one rebuilt every frame in place is one finding. A held span decided
within one frame of the render's grid of :data:`HOLD_PERSIST` says so in
``detail.clock_margin`` (seconds past it).
"""

from __future__ import annotations

__all__ = [
    "HOLD_PERSIST",
    "HOLD_SAMPLE",
    "LABEL_ATTACH",
    "LABEL_GLYPHS",
    "LABEL_LEFT",
    "SHAPE_EXIT_TOL",
    "SHAPE_GROWTH",
    "TEXT_MARGIN",
    "hold_watch",
]

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.timeline.clock import frames_in

from ._scope import Animated, animated_by, asked, named_by
from .armed import ArmedRule
from .findings import Finding
from .instants import render_grid
from .plays import played
from .rules._excuse import muted, paired
from .rules._geometry import box_gap
from .rules._pairs import _measure, pair_findings
from .rules._units import Measured, measured_units
from .rules.frame_edge import edge_findings

if TYPE_CHECKING:
    from arachne.camera.projection import View
    from arachne.timeline.record import PlayRecord, Record

    from .core import LintContext
    from .parts import Box, Parts
    from .rules._rule import Terms

#: A defect on screen this long, in seconds, was watched.
HOLD_PERSIST = Fraction(1)
#: Seconds between the watch's samples while a play runs.
HOLD_SAMPLE = Fraction(1, 4)
#: How far inside the frame's edge a text keeps (about 7 px at 1080).
TEXT_MARGIN = 0.05
#: How far past an edge a shape seen wholly inside runs before it has left.
SHAPE_EXIT_TOL = 0.05
#: ...unless its box's diagonal has grown this many times since.
SHAPE_GROWTH = 1.2
#: A text of this many glyphs or fewer is a label: it names a thing.
LABEL_GLYPHS = 6
#: A label sat this near a still shape names that shape...
LABEL_ATTACH = 0.3
#: ...and is left behind once the shape has drawn this much further off.
LABEL_LEFT = 0.75
#: A centre that moved less than this since the instant before is still.
_STILL = 0.02
#: A label that moved this far since it was placed was carried.
_CARRIED = 0.1
#: A referent that moved less than this has not moved away.
_AWAY = 0.5
#: A referent whose size changed by this share changed shape, not place.
_RESIZED = 0.05
#: How near a camera's frame is to its home to be at home.
_HOME = 1e-3


@dataclass
class _Open:
    """A defect open since ``first``, last seen at ``last``, and the worst
    finding seen of it.
    """

    first: Fraction
    last: Fraction
    finding: Finding


@dataclass(frozen=True)
class _Step:
    """One sample of the watch: what it is (``"entry"``, ``"inside"``,
    ``"exit"`` or ``"final"``), the play it is of, the frame it reads and
    the time it stands for.
    """

    kind: str
    play: int | None
    frame: Fraction
    at: Fraction


def _schedule(record: Record, since: Fraction) -> list[_Step]:
    """When the watch samples, from ``since`` on: as each play begins, then
    at the first frame at least :data:`HOLD_SAMPLE` after the sample before
    while it plays (a frozen frame has none), and as it ends; and the scene
    as it closes.
    """
    grid = render_grid(record)
    steps = []
    for play in record.plays:
        frames = frames_in(play.start, play.end, grid)
        if play.start < since or not frames:
            continue
        steps.append(_Step("entry", play.index, frames[0], frames[0]))
        if not play.held:
            last = play.start
            for frame in frames:
                if frame - last >= HOLD_SAMPLE:
                    steps.append(_Step("inside", play.index, frame, frame))
                    last = frame
        # The play's last frame, read for the scene as the play leaves it.
        steps.append(_Step("exit", play.index, frames[-1], play.end))
    steps.append(_Step("final", None, record.duration, record.duration))
    return steps


def _frames(record: Record, since: Fraction) -> list[Fraction]:
    return [step.frame for step in _schedule(record, since)]


def _watch(context: LintContext, rule: ArmedRule) -> list[Finding]:
    record = context.record
    watch = _Watch(record, rule.terms)
    plays = record.plays
    steps = _schedule(record, rule.since)
    shown = rule.frames(context)
    for index, (step, parts) in enumerate(zip(steps, shown, strict=True)):
        play = plays[step.play] if step.play is not None else None
        if play is None or step.kind == "entry":
            watch.sample(parts, step.at, (), ())
        elif step.kind == "inside":
            watch.sample(parts, step.at, watch.moved(play), watch.going(play))
        else:
            # The frame after the play shows what it carried off; its last
            # frame stands for how it left the scene, less what it removed.
            watch.after_play(play, shown[index + 1])
            watch.sample(parts, step.at, (), watch.removed(play))
    return watch.report(1 / render_grid(record))


class _Watch:
    """The watch's state as it reads its samples in order."""

    def __init__(self, record: Record, terms: Terms) -> None:
        self.record = record
        self.terms = terms
        self.open: dict[tuple[Any, ...], _Open] = {}
        self.held: list[_Open] = []
        self.framed: dict[str, float] = {}
        self.carried: set[str] = set()
        self.centres: dict[str, np.ndarray] = {}
        self.labels: dict[
            str, tuple[str, float, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self.named: set[str] = set()
        self._animated: dict[tuple[int, str], list[Animated]] = {}

    # What a play animates

    def moved(self, play: PlayRecord, at: str = "start") -> list[Animated]:
        """What ``play`` animates."""
        return self._of(play, "moved", at)

    def going(self, play: PlayRecord) -> list[Animated]:
        """What ``play`` takes away or pulses: not asked while it plays."""
        return self._of(play, "going", "start")

    def removed(self, play: PlayRecord) -> list[Animated]:
        """What ``play`` takes out of the scene as it ends."""
        return self._of(play, "removed", "start")

    def _of(self, play: PlayRecord, which: str, at: str) -> list[Animated]:
        key = (play.index, f"{which}:{at}")
        found = self._animated.get(key)
        if found is None:
            leaves = played(play)
            if which == "going":
                leaves = tuple(leaf for leaf in leaves if leaf.departs or leaf.emphasis)
            elif which == "removed":
                leaves = tuple(leaf for leaf in leaves if leaf.remover)
            found = animated_by(self.record, play, leaves, at=at)
            self._animated[key] = found
        return found

    # Reading a sample

    def sample(
        self,
        parts: Parts,
        at: Fraction,
        moved: list[Animated] | tuple[()],
        going: list[Animated] | tuple[()],
    ) -> None:
        """Read the frame ``parts`` shows, standing for the time ``at``, with
        ``moved`` animated and ``going`` leaving.
        """
        frame = parts.frame
        every = asked(parts, measured_units(parts))
        leaving = set().union(*(g.family for g in going)) if going else set()
        staying = [u for u in every if u.root not in leaving]
        for unit in staying:
            if unit.id in self.carried and unit.on(frame) and not named_by(unit, moved):
                self.carried.discard(unit.id)
        live = [
            u for u in staying if not named_by(u, going) and u.id not in self.carried
        ]
        home = _at_home(parts.view)
        by_id = {u.id: u for u in live}
        found: dict[tuple[Any, ...], Finding] = {}
        edged = live if home else [u for u in live if _meets(u.glyph_box, frame)]
        edge_terms = replace(self.terms, mid_scene=False)
        for f in edge_findings(parts, edge_terms, margin=TEXT_MARGIN, among=edged):
            found[("edge", f.objects[0].id)] = f
        if home:
            for f in self._exits(parts, live):
                found[("exit", f.objects[0].id)] = f
            for f in self._left_behind(parts, live, moved):
                found[("strand", f.objects[0].id)] = f
            for f in pair_findings(parts, self.terms, among=live, text_only=True):
                # Two things the play carries together are its own geometry
                # in motion, not an arrangement anyone placed.
                if all(named_by(by_id[o.id], moved) for o in f.objects):
                    continue
                kind = "cover" if f.code == "occlusion.glyph" else "touch"
                found[(kind, *(o.id for o in f.objects))] = f
        for key, finding in found.items():
            held = self.open.get(key)
            if held is None:
                self.open[key] = _Open(at, at, finding)
                continue
            held.last = at
            if _badness(finding) > _badness(held.finding):
                held.finding = finding
        for key in [k for k in self.open if k not in found]:
            self._close(key)

    def after_play(self, play: PlayRecord, parts: Parts) -> None:
        """After ``play``, at the frame that follows it: what it carried wholly
        off the frame left on purpose, and its way out was transit.
        """
        moved = self.moved(play, at="end")
        for unit in asked(parts, measured_units(parts)):
            if unit.on(parts.frame):
                self.carried.discard(unit.id)
            elif moved and named_by(unit, moved):
                self.carried.add(unit.id)
                for key in [
                    k
                    for k, held in self.open.items()
                    if k in (("edge", unit.id), ("exit", unit.id))
                    and held.first >= play.start
                ]:
                    del self.open[key]

    # The questions only a frame at home asks

    def _exits(self, parts: Parts, live: list[Measured]) -> list[Finding]:
        left, bottom, right, top = parts.frame
        found = []
        for unit in live:
            if unit.text or unit.furniture or not _object_like(unit):
                continue
            x0, y0, x1, y1 = unit.outer
            size = math.hypot(x1 - x0, y1 - y0)
            if size < 1e-3:
                continue
            over = {
                "left": left - x0,
                "right": x1 - right,
                "bottom": bottom - y0,
                "top": y1 - top,
            }
            edge = max(over, key=lambda e: over[e])
            past = over[edge]
            if past <= 0:
                self.framed[unit.id] = size
                continue
            framed = self.framed.get(unit.id)
            if (
                framed is None
                or past <= SHAPE_EXIT_TOL
                or size > SHAPE_GROWTH * max(framed, 1e-6)
                or muted(self.terms.exempt, unit)
            ):
                continue
            gone = not unit.on(parts.frame)
            found.append(
                _finding(
                    parts,
                    "clip.frame_edge",
                    (unit,),
                    f"{_called(unit)} has left the frame: it was wholly inside it,"
                    f" and now {'lies beyond' if gone else 'runs'} {past:.3f} units"
                    f" past the {edge} edge while still in the scene.",
                    {"edge": edge, "past": round(past, 4)},
                )
            )
        return found

    def _left_behind(
        self, parts: Parts, live: list[Measured], moved: Iterable[Animated]
    ) -> list[Finding]:
        moved = list(moved)
        by_id = {u.id: u for u in live}
        centre = {k: _centre(u.outer) for k, u in by_id.items()}
        still = {
            k
            for k, c in centre.items()
            if k in self.centres and float(np.linalg.norm(c - self.centres[k])) < _STILL
        }
        found = []
        for key, label in by_id.items():
            if not label.text or len(label.leaves) > LABEL_GLYPHS:
                continue
            if key not in self.named:
                if key not in still:
                    continue
                near = [
                    (_measure(label, shape), other)
                    for other, shape in by_id.items()
                    if other in still
                    and not shape.text
                    and not shape.marker
                    and box_gap(label.outer, shape.outer) < LABEL_ATTACH
                    and not _frames_texts(shape, label, live)
                ]
                near = [(gap, other) for gap, other in near if gap < LABEL_ATTACH]
                near = _own_group(label, near, live)
                if any(named_by(by_id[other], moved) for _, other in near):
                    continue
                self.named.add(key)
                if near:
                    gap, other = min(near)
                    shape = by_id[other]
                    self.labels[key] = (
                        other,
                        gap,
                        centre[key],
                        centre[other],
                        _size(shape.outer),
                    )
                continue
            if key not in self.labels:
                continue
            other, gap0, placed, referent, size0 = self.labels[key]
            thing = by_id.get(other)
            if (
                thing is None
                or float(np.linalg.norm(centre[key] - placed)) > _CARRIED
                or float(np.linalg.norm(centre[other] - referent)) < _AWAY
            ):
                continue
            if np.any(
                np.abs(_size(thing.outer) - size0) > _RESIZED * np.maximum(size0, 0.1)
            ):
                continue
            gap = _measure(label, thing)
            if gap < gap0 + LABEL_LEFT:
                continue
            exempt = self.terms.exempt
            if (
                muted(exempt, label)
                or muted(exempt, thing)
                or paired(exempt, label, thing)
            ):
                continue
            away = float(np.linalg.norm(centre[other] - referent))
            found.append(
                _finding(
                    parts,
                    "label.left_behind",
                    (label, thing),
                    f"label {label.object.label_or_text!r} stayed {gap:.2f} units"
                    f" from {_called(thing)}, which it was placed {gap0:.2f} from:"
                    f" the {thing.unit.type} moved {away:.2f} units without it.",
                    {"referent": thing.id, "distance": round(gap, 4)},
                )
            )
        self.centres = centre
        return found

    # Reporting

    def _close(self, key: tuple[Any, ...]) -> None:
        held = self.open.pop(key)
        if held.last - held.first >= HOLD_PERSIST:
            self.held.append(held)

    def report(self, frame: Fraction) -> list[Finding]:
        for key in list(self.open):
            self._close(key)
        found = []
        for held in sorted(self.held, key=lambda h: h.first):
            finding = held.finding
            detail = dict(finding.detail)
            margin = held.last - held.first - HOLD_PERSIST
            if margin < frame:
                detail["clock_margin"] = float(margin)
            found.append(
                replace(
                    finding,
                    t_start=held.first,
                    t_end=held.last,
                    held=True,
                    line=None,
                    message=(
                        f"on screen {float(held.first):.2f}-{float(held.last):.2f} s:"
                        f" {finding.message}"
                    ),
                    detail=detail,
                )
            )
        return found


def _finding(
    parts: Parts,
    code: str,
    units: tuple[Measured, ...],
    message: str,
    detail: dict[str, Any],
) -> Finding:
    t = Fraction(parts.t)
    return Finding(
        code, t, t, tuple(u.object for u in units), message=message, detail=detail
    )


def _badness(finding: Finding) -> float:
    """How bad a finding is, to keep the worst seen of a defect."""
    detail = finding.detail
    for field, sign in (
        ("past", 1),
        ("overlap", 1),
        ("covered", 1),
        ("gap", -1),
        ("distance", -1),
    ):
        if field in detail:
            return sign * float(detail[field])
    return 0.0


def _at_home(view: View) -> bool:
    x, y = view.frame_center[0], view.frame_center[1]
    return (
        abs(x) < _HOME
        and abs(y) < _HOME
        and abs(view.frame_width - view.home_width) < _HOME
    )


def _meets(box: Box, frame: Box) -> bool:
    return (
        box[2] >= frame[0]
        and box[0] <= frame[2]
        and box[3] >= frame[1]
        and box[1] <= frame[3]
    )


def _object_like(unit: Measured) -> bool:
    """A thing on screen rather than a stroke through it: filled, closed,
    or an arrow.
    """
    return unit.region or unit.ring or unit.unit.tip


def _frames_texts(shape: Measured, label: Measured, live: list[Measured]) -> bool:
    """A closed outline round the label and another text: it frames them."""
    if shape.region or not shape.ring or not _inside(label.outer, shape.outer):
        return False
    return any(
        t is not label and t.text and _inside(t.outer, shape.outer) for t in live
    )


def _own_group(
    label: Measured, near: list[tuple[float, str]], live: list[Measured]
) -> list[tuple[float, str]]:
    """The shapes a label may name: its own group's, when its group holds
    any shape.
    """
    shapes = [u for u in live if not u.text]
    for holder in reversed(label.unit.within):
        held = {
            s.id
            for s in shapes
            if s.id == holder.id or any(h.id == holder.id for h in s.unit.within)
        }
        if held:
            return [(gap, other) for gap, other in near if other in held]
    return near


def _inside(inner: Box, outer: Box) -> bool:
    """``inner`` strictly inside ``outer``."""
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def _centre(box: Box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _size(box: Box) -> np.ndarray:
    return np.array([box[2] - box[0], box[3] - box[1]])


def _called(unit: Measured) -> str:
    said = unit.object.label_or_text
    return f"{unit.unit.type} {said!r}" if said else unit.unit.type


#: The rule: what stays wrong on screen, watched from where it is armed.
hold_watch = ArmedRule(_watch, reads=_frames)
