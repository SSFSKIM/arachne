"""``motion.jump``: a body seen elsewhere across the cut between two plays.

A still frame can't show a teleport: a body that slides during one play,
is set down somewhere else by the script, and is seen there at the next
play's first frame passes every check of a single instant. This rule reads
the frames the render shows at each instant of a fixed reference clock
(:data:`~arachne.lint.codes.MOTION_REFERENCE_FPS`, asked of the lint's
frames, :mod:`.frames`), so its verdict is one whatever rate the scene is
rendered at, and, at each cut, compares where each body stood as the play ended (the
record's exit of the play, :mod:`arachne.timeline.bodies`: its animations
finished and cleaned up, the script not yet run on) with where it stood as
the next play began (its entry: evaluated at its exact start, its parts
begun and its updaters run once, as CE's first frame of a play shows it,
wherever the next frame of the grid falls).

**Bodies** are as :mod:`arachne.timeline.bodies` reads them: every
top-level mobject that paints something and every direct child of one (but
a text's glyphs and a surface's faces), keyed by its stable id, at its
centre in the scene and as the viewer sees it (``view``: relative to a
moving camera's frame and scaled by its zoom; under every other camera, a
three-dimensional one too, the scene's x and y).

**Moving.** A body moves during a play when its centre travels at least
:data:`TRAVEL_MIN` in the scene across that play's frames and its exit, by
steps that keep its box's size within :data:`RIGID` (drawing, growth and
reshaping move a centre too, and are not travel).

**The cut.** A body visible (some member lays ink) as the play ended and as
the next play began, which moves during either play, jumps when its
centre stands more than the tolerance apart across the cut: the scene's
displacement scaled by the later frame's zoom, so a camera cut moves the
picture and no body. A body that moved during the play before the cut has
``detail.side`` ``"before"``; one that moved only after it (set down
elsewhere, then thrown) ``"after"``.

**What a step across the cut may be instead.** None of these is reported:

* *rebuilt*: a child of a group ``become`` rebuilt across the cut
  (:mod:`arachne.timeline.rebuilds`), which takes whatever item the new list
  puts at its place: one that lands within the tolerance of where one of
  the group's visible children stood as the play ended, whatever its size,
  or one born at the cut. The born are those landing where none stood, at
  most as many as the group's count of children rose by (``become``'s
  padding counted), the farthest from every such place first and, among
  those as far, the one first seen first; a group that
  gained no child has none born, so a child set back (a rig snapped back)
  is still judged. A child that left the list within the play keeps its
  padded slot, so a newcomer refilling it at the cut is no birth;
* *reshaped*: a single shape at another size (an arc restarted), or a body
  whose number of drawn members changed: another shape, not the same one
  elsewhere;
* *churn*: a step no longer than :data:`CHURN` times the body's largest
  step, as the viewer saw it (a camera's pan included), between two of the
  frames and the exit of the play that moved it: within its own motion;
* *relabelled*: a look-alike sibling (same group, same size) stood where it
  lands as the play ended, or stands where it left just after: the picture
  shows a body in each place, not one crossing the gap;
* *redrawn*: a top-level group drawing nothing of its own whose box moved
  while none of its members jumped: a part changed under it.

**One finding per cause.** A group that jumped whole is the finding, not
its members; a group whose box moved because a member jumped defers to the
member. A finding names its body's stable id, creating line and name (the
scene's name for it at the cut, or the first it is given after), at the
next play's first frame; its ``detail`` holds its centres as viewed before
and after, the distance and the two plays. A jump object has no box
(``bbox`` and ``screen_bbox`` are ``None``), as the moment of the jump has
none.

A scene that arms the rule (:meth:`.Recorder.arm_motion`) has the cuts
measured from the first play to end after the arming on
(``Record.armed_at``), and its session's record keeps those plays' exits;
the cuts before are not seen. The plays, their exits and the arming are
those of the execution the lint reads: a forward run that armed later than
the session's record measures only the cuts after its own arming, and one
that armed earlier the cuts from its own (every frame of the reference
clock the record's plays leave out is asked for from a forward run, should the lint
make one: one whose plays are split or frozen otherwise is read whole). A
scene that never armed it has every cut measured. A record that keeps no
exit of a play it measures (the scene never armed the rule) is read
through the lint's forward run, which keeps every play's; it is never
measured another way. A play of the execution whose frames were not read
(it plays otherwise than the session's record, which they were asked for
by) is not measured: the two executions differ, and the lint says so with
a ``replay.diverged``. The rule is not among the defaults: it runs when
passed to :func:`.lint`.
"""

from __future__ import annotations

__all__ = [
    "CHURN",
    "JUMP_TOL",
    "RESHAPE",
    "RIGID",
    "TRAVEL_MIN",
    "MotionJump",
    "motion_jump",
]

from bisect import bisect_left
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from arachne.timeline import rebuilds
from arachne.timeline.bodies import Body, Exit, bodies
from arachne.timeline.clock import frames_in
from arachne.timeline.draw import Drawing

from .codes import MOTION_REFERENCE_FPS
from .findings import Finding, FindingObject, Holder
from .instants import render_grid

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.session import Session
    from arachne.timeline.record import Capture, PlayRecord, Record

    from .core import LintContext
    from .frames import Frames

#: Home-frame units a body may stand from where it was across a cut.
JUMP_TOL = 0.25
#: Scene units a centre must travel within a play for the body to move.
TRAVEL_MIN = 0.05
#: The share of its size a box may change by in a step that is travel.
RIGID = 0.05
#: The share of its size a single shape may change by across a cut and
#: still be the same shape.
RESHAPE = 0.25
#: A step across a cut within this many times the body's largest step
#: between two frames of the play that moved it is its own motion.
CHURN = 2.0
#: The smallest size, in scene units, a share of a size is taken of.
_SIZE_FLOOR = 0.1


@dataclass(frozen=True, eq=False)
class _Frame:
    """One frame: its time, the zoom of its view and its bodies, each group
    before its members.
    """

    t: Fraction
    zoom: float
    bodies: dict[str, Body]


@dataclass(eq=False)
class _Play:
    """What one play shows of its bodies, its frames and then its exit: how
    far each travelled, its largest step as viewed, and where it was last
    seen (as the play ended, when it was there).
    """

    record: PlayRecord
    frames: list[_Frame]
    exit: Exit
    #: Where its bodies stood as it began; ``None`` for the first play
    #: measured, which may have begun before the scene armed the rule.
    entry: Exit | None
    travel: dict[str, float] = field(default_factory=dict)
    largest: dict[str, float] = field(default_factory=dict)
    last: dict[str, Body] = field(default_factory=dict)

    @property
    def first_frame(self) -> _Frame:
        return self.frames[0]

    def moved(self, body: str) -> bool:
        return self.travel.get(body, 0.0) >= TRAVEL_MIN


@dataclass(frozen=True)
class MotionJump:
    """The ``motion.jump`` rule at a tolerance, in home-frame units."""

    jump_tol: float = JUMP_TOL

    def request_frames(self, frames: Frames) -> None:
        """Ask the lint for the reference clock's frames of each play from
        the record's arming on; for every other one of the scene too, from
        a forward run if the lint makes one (whose own plays and arming are
        known only once it has run); and for that run, when the record keeps
        no exit of one of them.
        """
        record = frames.session.record
        wanted = _wanted(record)
        if not _has_exits(record, wanted):
            frames.require_forward()

        def read(
            source: Record,
            t: Fraction,
            play: PlayRecord | None,
            capture: Capture,
            live: bool,
        ) -> tuple[float, dict[str, Body]]:
            return _read(source, play, capture, live=live)

        frames.request(self, _times(wanted), read)
        others = _others(record, wanted)
        if others:
            frames.request(self._others, others, read, forward_only=True)

    def __call__(self, context: LintContext) -> list[Finding]:
        frames = context.frames
        record = context.record
        wanted = _wanted(record)
        seen = dict(zip(_times(wanted), frames.result(self), strict=True))
        if frames.answered(self._others):
            others = _others(record, wanted)
            seen.update(zip(others, frames.result(self._others), strict=True))
        # The plays, exits, arming and names of the execution read: the
        # session's record when it keeps them all, else the forward run,
        # which keeps every play's exit.
        execution = frames.execution
        exits = execution.exits or {}
        entries = execution.entries or {}
        plays: list[_Play | None] = []
        unread: PlayRecord | None = None
        for play, shown in _wanted(execution):
            if play.index in exits and all(t in seen for t in shown):
                read = [_Frame(t, *seen[t]) for t in shown]
                entry = entries.get(play.index)
                plays.append(_measure(_Play(play, read, exits[play.index], entry)))
            else:
                # A play other than those the frames were asked for: the
                # two executions differ.
                plays.append(None)
                unread = unread or play
        names = _Names(execution)
        groups = _Rebuilt(execution)
        found: list[Finding] = []
        for before, after in zip(plays, plays[1:], strict=False):
            if before is not None and after is not None and after.entry is not None:
                cut = _Cut(self.jump_tol, before, after, names, groups)
                found.extend(cut.findings())
        if unread is not None:
            found.extend(_unmeasured(context.session, execution, unread))
        return found

    @property
    def _others(self) -> tuple[str, int]:
        """Who the frames only a forward run reads are asked for by."""
        return ("motion from a forward run", id(self))


#: The rule at the default tolerance.
motion_jump = MotionJump()


# Reading the frames


def _wanted(record: Record) -> list[tuple[PlayRecord, list[Fraction]]]:
    """Every play that shows a frame, with the frames it shows: from the
    play the scene armed the rule before on, when it did.
    """
    grid = render_grid(record)
    first = record.armed_at or 0
    wanted = []
    for play in record.plays:
        if play.index < first:
            continue
        shown = _reference_frames(play.start, play.end, grid)
        if shown:
            # A frozen play shows one picture all along.
            wanted.append((play, shown[:1] if play.held else shown))
    return wanted


def _times(wanted: list[tuple[PlayRecord, list[Fraction]]]) -> list[Fraction]:
    """The frames ``wanted`` shows, in order."""
    return [t for _, shown in wanted for t in shown]


def _others(
    record: Record, wanted: list[tuple[PlayRecord, list[Fraction]]]
) -> list[Fraction]:
    """Every reference frame over ``record``'s scene but those ``wanted``
    shows: what a forward execution may play otherwise (armed
    earlier, its plays split or frozen otherwise) over the same span.
    """
    shown = set(_times(wanted))
    every = _reference_frames(Fraction(0), record.duration, render_grid(record))
    return [t for t in every if t not in shown]


def _reference_frames(start: Fraction, end: Fraction, grid: Fraction) -> list[Fraction]:
    """The frames of the render's ``grid`` in ``[start, end)`` the rule
    reads: at each instant of the reference clock
    (:data:`~arachne.lint.codes.MOTION_REFERENCE_FPS`) there, the frame the
    render shows from then on (the first of the grid at or after it), each
    once. On a grid that holds the reference clock's, those instants.
    """
    shown = frames_in(start, end, grid)
    picked: dict[Fraction, None] = {}
    for instant in frames_in(start, end, Fraction(MOTION_REFERENCE_FPS)):
        at = bisect_left(shown, instant)
        if at < len(shown):
            picked[shown[at]] = None
    return list(picked)


def _unmeasured(session: Session, execution: Record, play: PlayRecord) -> list[Finding]:
    """What a play of the execution read is, whose frames or exit were not
    read (the frames are asked for by the session's record, which plays it
    otherwise): a ``replay.diverged`` of the two executions, unless the
    session holds one already, which the lint reports.
    """
    from arachne.timeline.execution import Divergence

    from .core import divergence_finding

    if any(divergence.kind == "execution" for divergence in session.divergences):
        return []
    reason = (
        f"the session's record does not play play {play.index}"
        f" ({float(play.start):g} s to {float(play.end):g} s) as the execution"
        " the lint read does, so its motion across that play's cuts is not"
        " measured"
    )
    divergence = Divergence("execution", play.index, play.source_line, reason)
    return [divergence_finding(execution, divergence)]


def _has_exits(record: Record, wanted: list[tuple[PlayRecord, list[Fraction]]]) -> bool:
    """Whether ``record`` keeps the exit of every play it shows, and the
    entry of every one but the first.
    """
    exits, entries = record.exits, record.entries
    return (
        exits is not None
        and entries is not None
        and all(play.index in exits for play, _ in wanted)
        and all(play.index in entries for play, _ in wanted[1:])
    )


def _read(
    record: Record,
    play: PlayRecord | None,
    capture: Capture,
    *,
    live: bool,
) -> tuple[float, dict[str, Body]]:
    """The zoom of one frame's view and its bodies: of the live scene of a
    forward run, or of ``capture`` shown from the record.
    """
    scene = record._scene
    camera, timeline = scene.renderer.camera, scene._arachne_timeline

    def read() -> tuple[float, dict[str, Body]]:
        return bodies(
            capture.draw_list(),
            camera,
            record.origin,
            home=timeline.home,
            configured_pixels=timeline.configured_pixels,
        )

    if live:
        return read()
    static = list(play.static) if play is not None and capture.moving else []
    with Drawing(record) as drawing:
        drawing.show(capture, [*capture.draw_list(), *static, *capture.moving])
        return read()


def _measure(play: _Play) -> _Play:
    """``play`` with each body's travel, largest step and last sight, over
    its frames and then its exit.
    """
    seen = [(frame.zoom, frame.bodies) for frame in play.frames]
    seen.append((play.exit.zoom, play.exit.bodies))
    previous: dict[str, Body] | None = None
    for _, shown in seen:
        for key, body in shown.items():
            play.last[key] = body
            if previous is None or key not in previous:
                continue
            was = previous[key]
            viewed = float(np.hypot(*(body.view - was.view)))
            play.largest[key] = max(play.largest.get(key, 0.0), viewed)
            if _kept_size(was.size, body.size, RIGID):
                play.travel[key] = play.travel.get(key, 0.0) + float(
                    np.hypot(*(body.centre - was.centre))
                )
        previous = shown
    return play


def _kept_size(was: np.ndarray, now: np.ndarray, share: float) -> bool:
    """Whether a box's width and height each changed by at most ``share``
    of what they were: a step that moves a body without reshaping it.
    """
    return all(
        abs(float(now[k]) - float(was[k])) <= share * max(float(was[k]), _SIZE_FLOOR)
        for k in (0, 1)
    )


def _same_size(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two boxes' widths and heights each agree within
    :data:`RESHAPE` of the larger: the same shape, wherever it stands.
    """
    return all(
        abs(float(a[k]) - float(b[k]))
        <= RESHAPE * max(float(a[k]), float(b[k]), _SIZE_FLOOR)
        for k in (0, 1)
    )


def _nearest(centre: np.ndarray, places: list[np.ndarray]) -> float:
    """How far ``centre`` stands from the nearest of ``places``, in scene
    units.
    """
    return min(
        (float(np.hypot(*(centre - place))) for place in places), default=float("inf")
    )


# Judging a cut


class _Rebuilt:
    """Which of an execution's groups ``become`` rebuilt across a cut
    (:mod:`arachne.timeline.rebuilds`), and in what order its mobjects were
    first seen, by stable id.
    """

    def __init__(self, record: Record) -> None:
        self._record = record
        self._mobjects: dict[str, Mobject] | None = None
        self._seen: dict[str, int] = {}

    def first_seen(self, key: str) -> int:
        """Where ``key`` stands in the order the execution first saw its
        mobjects (the record's first captures, in the order each walks the
        scene): a mobject ``become``'s padding made between plays comes after
        every one seen before it, wherever in its group's list it stands.
        """
        self._read()
        return self._seen.get(key, len(self._seen))

    def _read(self) -> None:
        if self._mobjects is not None:
            return
        record = self._record
        self._mobjects = {}
        for mob in record.mobjects():
            key = record.origin(mob).id
            if key not in self._mobjects:
                self._mobjects[key] = mob
                self._seen[key] = len(self._seen)

    def across(self, group: str, before: _Play, after: _Play) -> bool:
        """Whether ``become`` rebuilt ``group`` from the scene's call of
        the play ``before`` the cut to the state the play ``after`` it
        began in.
        """
        start, end = before.record.rebuilds_called, after.record.rebuilds_entered
        if end <= start:
            return False
        self._read()
        assert self._mobjects is not None
        held = self._mobjects.get(group)
        return held is not None and rebuilds.rebuilt_between(held, start, end)


class _Names:
    """The scene's names for its mobjects, by stable id: per play, as each
    began, and at the end.
    """

    def __init__(self, record: Record) -> None:
        nodes = record._nodes

        def by_id(names: dict[int, str]) -> dict[str, str]:
            out: dict[str, str] = {}
            for key, name in names.items():
                node = nodes.get(key)
                if node is not None:
                    out.setdefault(node.origin.id, name)
            return out

        self.plays = {play.index: by_id(play.names) for play in record.plays}
        self.final = by_id(record.final_names)

    def at_cut(self, body: str, before: int, after: int) -> str | None:
        """The name ``body`` has across the cut, or the first it is given
        after it.
        """
        for index in (after, before):
            name = self.plays.get(index, {}).get(body)
            if name is not None:
                return name
        for index in sorted(i for i in self.plays if i > after):
            name = self.plays[index].get(body)
            if name is not None:
                return name
        return self.final.get(body)


@dataclass(frozen=True, eq=False)
class _Step:
    """A body's step across a cut, as found."""

    body: Body
    shift: np.ndarray
    distance: float
    side: str


class _Cut:
    """The cut between play ``before`` and play ``after``: which of the
    bodies that moved on either side of it jumped.
    """

    def __init__(
        self,
        tol: float,
        before: _Play,
        after: _Play,
        names: _Names,
        groups: _Rebuilt,
    ) -> None:
        self.tol = tol
        self.before, self.after = before, after
        self.names = names
        self.groups = groups
        #: Of each group rebuilt across the cut: where its children stood as
        #: the play ended, and which of them were born at the cut.
        self._places: dict[str, list[np.ndarray]] = {}
        self._born: dict[str, set[str]] = {}
        assert after.entry is not None
        self.left = before.exit.bodies
        # As the next play began, CE's first frame of it: not the next frame
        # of the grid, which may be well into the play.
        self.came = after.entry.bodies
        self.zoom = after.entry.zoom

    def shift(self, key: str) -> np.ndarray:
        return (self.came[key].centre - self.left[key].centre) * self.zoom

    def seen_across(self, key: str) -> bool:
        return (
            key in self.left
            and key in self.came
            and self.left[key].visible
            and self.came[key].visible
        )

    def findings(self) -> list[Finding]:
        moving = [
            key
            for key in self.came
            if self.seen_across(key)
            and (self.before.moved(key) or self.after.moved(key))
        ]
        found: dict[str, _Step] = {}
        for key in moving:
            shift = self.shift(key)
            distance = float(np.hypot(*shift))
            if distance <= self.tol:
                continue
            side = "before" if self.before.moved(key) else "after"
            mover = self.before if side == "before" else self.after
            if self.passes(key, distance, mover) is None:
                found[key] = _Step(self.came[key], shift, distance, side)
        batch = set(moving)
        redrawn = [key for key in found if self.redrawn(key, batch, found)]
        for key in redrawn:
            del found[key]
        whole = {
            key
            for key, step in found.items()
            if step.body.parent in found
            and float(np.hypot(*(step.shift - found[step.body.parent].shift)))
            <= self.tol
        }
        out = []
        for key, step in found.items():
            if key in whole or any(
                other.body.parent == key and j not in whole
                for j, other in found.items()
            ):
                continue
            out.append(self.finding(step))
        return out

    def passes(self, key: str, distance: float, mover: _Play) -> str | None:
        """Why a step beyond the tolerance is not a jump, or ``None``."""
        if self.rebuilt(key):
            return "rebuilt"
        was, now = self.left[key], self.came[key]
        if was.members != now.members or (
            now.members == 1 and not _same_size(was.size, now.size)
        ):
            return "reshaped"
        if distance <= CHURN * mover.largest.get(key, 0.0):
            return "churn"
        if self.relabelled(key):
            return "relabelled"
        return None

    def rebuilt(self, key: str) -> bool:
        """Whether its group's rebuilt list explains ``key``'s step: ``become``
        rebuilt the group across the cut (from the call of the play before it
        to the state the next play began in), and ``key`` now stands within
        the tolerance of where one of the group's children stood as the play
        ended (it took another child's item), or is one of the children born
        at the cut (:meth:`born`).
        """
        parent = self.came[key].parent
        if parent is None or not self.groups.across(parent, self.before, self.after):
            return False
        places = self.places(parent)
        return self.lands_on(self.came[key].centre, places) or key in self.born(parent)

    def places(self, group: str) -> list[np.ndarray]:
        """Where ``group``'s children stood, seen, as the play ended."""
        found = self._places.get(group)
        if found is None:
            found = self._places[group] = [
                body.centre
                for body in self.left.values()
                if body.parent == group and body.visible
            ]
        return found

    def lands_on(self, centre: np.ndarray, places: list[np.ndarray]) -> bool:
        """Whether ``centre`` stands within the tolerance of one of
        ``places``, measured as a cut's step is.
        """
        return any(
            float(np.hypot(*(centre - place))) * self.zoom <= self.tol
            for place in places
        )

    def born(self, group: str) -> set[str]:
        """``group``'s children born at the cut: of those seen on both sides
        that stand where none of its children stood, at most as many as its
        count of children rose by across the cut (padding counted, as
        ``become`` leaves it). A group that gained no child has none, so its
        children that stand where none stood are bodies: a rig's atoms
        snapped back, one or all. When more stand so than the count rose by,
        the ones farthest from every place a child stood are the born (a
        newcomer appears at its source, away from the rest; a jumper is
        caught nearer its old place), the one first seen first breaking a
        tie (:meth:`_Rebuilt.first_seen`).
        """
        found = self._born.get(group)
        if found is not None:
            return found
        was, now = self.left.get(group), self.came.get(group)
        grew = now.children - was.children if was is not None and now is not None else 0
        found = set()
        if grew > 0:
            places = self.places(group)
            unmatched = [
                key
                for key, body in self.came.items()
                if body.parent == group
                and self.seen_across(key)
                and not self.lands_on(body.centre, places)
            ]
            unmatched.sort(
                key=lambda key: (
                    -_nearest(self.came[key].centre, places),
                    self.groups.first_seen(key),
                )
            )
            found = set(unmatched[:grew])
        self._born[group] = found
        return found

    def relabelled(self, key: str) -> bool:
        """Whether a look-alike sibling stood where ``key`` lands just
        before the cut, or stands where it left just after.
        """
        parent = self.came[key].parent
        if parent is None:
            return False
        landing, leaving = self.came[key].view, self.left[key].view
        siblings = {
            other
            for bodies in (self.came, self.before.last)
            for other, body in bodies.items()
            if body.parent == parent and other != key
        }
        for other in siblings:
            latest = self.came.get(other) or self.before.last[other]
            if not _same_size(latest.size, self.came[key].size):
                continue
            stood = self.before.last.get(other)
            if stood is not None and np.hypot(*(stood.view - landing)) <= self.tol:
                return True
            stands = self.came.get(other)
            if stands is not None and np.hypot(*(stands.view - leaving)) <= self.tol:
                return True
        return False

    def redrawn(self, key: str, batch: set[str], found: dict[str, _Step]) -> bool:
        """Whether ``key`` is a top-level group drawing nothing of its own
        none of whose members jumped across the cut.
        """
        body = self.came[key]
        if body.parent is not None or body.own:
            return False
        members = [other for other, b in self.came.items() if b.parent == key]
        return bool(members) and not any(
            self.member_jumped(member, batch, found) for member in members
        )

    def member_jumped(self, key: str, batch: set[str], found: dict[str, _Step]) -> bool:
        if key in batch:
            return key in found
        if not self.seen_across(key):
            return False
        distance = float(np.hypot(*self.shift(key)))
        return distance > self.tol and self.passes(key, distance, self.before) is None

    def finding(self, step: _Step) -> Finding:
        body, key = step.body, step.body.id
        before, after = self.before.record, self.after.record
        name = self.names.at_cut(key, before.index, after.index)
        holders: tuple[Holder, ...] = ()
        if body.parent is not None:
            group = self.came[body.parent]
            holders = (
                Holder(
                    group.id,
                    self.names.at_cut(group.id, before.index, after.index),
                    group.scene_line,
                    group.type,
                ),
            )
        start = self.left[key].view
        end = body.view
        label = body.text if body.text is not None else name
        what = f"{label or body.type} [{key}]"
        mover = self.before if step.side == "before" else self.after
        how = (
            f"it moved {mover.travel.get(key, 0.0):.2f} units during play {before.index}"
            if step.side == "before"
            else f"it was set down there and then moved during play {after.index}"
        )
        return Finding(
            "motion.jump",
            self.after.first_frame.t,
            self.after.first_frame.t,
            objects=(
                FindingObject(
                    id=key,
                    source_line=body.scene_line,
                    kind=body.type,
                    label_or_text=label,
                    bbox=None,
                    screen_bbox=None,
                    name=name,
                    within=holders,
                ),
            ),
            message=(
                f"{what} jumped {step.distance:.2f} units between play"
                f" {before.index} and play {after.index}, from {_where(start)} to"
                f" {_where(end)}, with nothing carrying it; {how}."
            ),
            line=after.source_line,
            detail={
                "side": step.side,
                "distance": round(step.distance, 4),
                "from": _point(start),
                "to": _point(end),
                "boundary": {"from_play": before.index, "to_play": after.index},
            },
        )


def _point(xy: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in xy]


def _where(xy: np.ndarray) -> str:
    return f"({xy[0]:.2f}, {xy[1]:.2f})"
