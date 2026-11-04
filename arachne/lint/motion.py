"""``motion.jump``: a body seen elsewhere across the cut between two plays.

A still frame can't show a teleport: a body that slides during one play,
is set down somewhere else by the script, and is seen there at the next
play's first frame passes every check of a single instant. This rule reads
every frame the render shows (asked of the lint's frames, :mod:`.frames`)
and, at each cut, compares where each body stood as the play ended (the
record's exit of the play, :mod:`arachne.timeline.bodies`: its animations
finished and cleaned up, the script not yet run on) with the next play's
first frame.

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

**The cut.** A body visible (some member lays ink) as the play ended and at
the next play's first frame, which moves during either play, jumps when its
centre stands more than the tolerance apart across the cut: the scene's
displacement scaled by the later frame's zoom, so a camera cut moves the
picture and no body. A body that moved during the play before the cut has
``detail.side`` ``"before"``; one that moved only after it (set down
elsewhere, then thrown) ``"after"``.

**What a step across the cut may be instead.** None of these is reported:

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

A session's record keeps the exits of the plays that end once its scene
has armed the rule (:meth:`.Recorder.arm_motion`). A record that keeps no
exit of a play it shows (the scene never armed the rule, or armed it late)
is read through the lint's forward run, which keeps every play's; it is
never measured another way. The rule is not among the defaults: it runs
when passed to :func:`.lint`.
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

from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from arachne.timeline.bodies import Body, Exit, bodies
from arachne.timeline.clock import frames_in
from arachne.timeline.draw import Drawing

from .findings import Finding, FindingObject, Holder
from .instants import render_grid

if TYPE_CHECKING:
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
        """Ask the lint for every frame the render shows of each play; and
        for its forward run, when the record keeps no exit of one of them.
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

        frames.request(self, [t for _, shown in wanted for t in shown], read)

    def __call__(self, context: LintContext) -> list[Finding]:
        record = context.record
        wanted = _wanted(record)
        execution = context.frames.execution
        # The session's record when it keeps them all, else the forward run,
        # which keeps every play's.
        assert _has_exits(execution, wanted), "an execution read without its exits"
        assert execution.exits is not None
        read = iter(context.frames.result(self))
        plays = [
            _measure(
                _Play(
                    play,
                    [_Frame(t, *next(read)) for t in shown],
                    execution.exits[play.index],
                )
            )
            for play, shown in wanted
        ]
        names = _Names(record)
        found: list[Finding] = []
        for before, after in zip(plays, plays[1:], strict=False):
            found.extend(_Cut(self.jump_tol, before, after, names).findings())
        return found


#: The rule at the default tolerance.
motion_jump = MotionJump()


# Reading the frames


def _wanted(record: Record) -> list[tuple[PlayRecord, list[Fraction]]]:
    """Every play that shows a frame, with the frames it shows."""
    grid = render_grid(record)
    wanted = []
    for play in record.plays:
        shown = frames_in(play.start, play.end, grid)
        if shown:
            # A frozen play shows one picture all along.
            wanted.append((play, shown[:1] if play.held else shown))
    return wanted


def _has_exits(record: Record, wanted: list[tuple[PlayRecord, list[Fraction]]]) -> bool:
    """Whether ``record`` keeps the exit of every play it shows."""
    exits = record.exits
    return exits is not None and all(play.index in exits for play, _ in wanted)


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
    camera, timeline = scene.renderer.camera, scene.timeline

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


# Judging a cut


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

    def __init__(self, tol: float, before: _Play, after: _Play, names: _Names) -> None:
        self.tol = tol
        self.before, self.after = before, after
        self.names = names
        self.left = before.exit.bodies
        self.came = after.first_frame.bodies
        self.zoom = after.first_frame.zoom

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
