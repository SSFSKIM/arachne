"""``motion.jump``: a body seen elsewhere across the cut between two plays.

A still frame can't show a teleport: a body that slides during one play,
is set down somewhere else by the script, and is seen there at the next
play's first frame passes every check of a single instant. This rule reads
every frame the render shows (:mod:`.frames`) and compares, at each cut, a
play's last frame with the next play's first.

**Bodies.** At each frame, every top-level mobject that paints something,
and every direct child of one (but a text's glyphs and a surface's faces),
is a body, keyed by its stable id. Its position is the centre of the
points of its members that paint (a path or image with extent, a point
cloud with a visible stroke), in scene units; its size is their box's.
Something that paints nothing (alignment padding, a tracker) is no body
there.

**Moving.** A body moves during a play when its centre travels at least
:data:`TRAVEL_MIN` across that play's consecutive frames by steps that keep
its box's size within :data:`RIGID` (drawing, growth and reshaping move a
centre too, and are not travel).

**The cut.** A body on screen and visible (some member lays ink) at both
the play's last frame and the next play's first, which moves during either
play, jumps when its centre stands more than the tolerance apart across the
cut. The distance is measured in the scene and scaled to the home frame by
the later frame's zoom, so a camera cut moves the picture and no body; under
a three-dimensional camera it is the scene's x and y, z left out, and only
the centres a finding reports are projected. A body
that moved during the play before the cut has ``detail.side`` ``"before"``;
one that moved only after it (set down elsewhere, then thrown) ``"after"``.

**What a step across the cut may be instead.** None of these is reported:

* *reshaped*: a single shape at another size (an arc restarted), or a body
  whose number of drawn members changed: another shape, not the same one
  elsewhere;
* *churn*: a step no longer than :data:`CHURN` times the body's largest
  step between two frames of the play that moved it, both measured in the
  scene and scaled by the later frame's zoom, so a camera moving with the
  body cancels none of its steps: within its own motion (a play's last
  frame is up to one of its steps short of where the play leaves the body);
* *relabelled*: a look-alike sibling (same group, same size) stood where it
  lands just before the cut, or stands where it left just after: the
  picture shows a body in each place, not one crossing the gap;
* *redrawn*: a top-level group drawing nothing of its own whose box moved
  while none of its members jumped: a part changed under it.

**One finding per cause.** A group that jumped whole is the finding, not
its members; a group whose box moved because a member jumped defers to the
member. A finding names its body's stable id, creating line and name (the
scene's name for it at the cut, or the first it is given after), at the
next play's first frame; its ``detail`` holds the centres on screen before
and after (home-frame units, through the projection), the distance and the
two plays. A jump object has no box (``bbox`` and ``screen_bbox`` are
``None``), as the moment of the jump has none.

Nothing arms the rule by default: it runs when passed to :func:`.lint`.
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

from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.camera.projection import (
    View,
    draw_kind,
    member_box,
    record_view,
    view_of,
)
from arachne.timeline.clock import frames_in
from arachne.timeline.draw import Drawing

from .findings import Finding, FindingObject, Holder
from .instants import render_grid
from .parts import _paint, _text_types, _viewing_nodes

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
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
#: Scene units under which two points are one place.
_ONE_PLACE = 1e-3


@dataclass(frozen=True, eq=False)
class _Body:
    """A body as one frame shows it. ``view`` is its centre as the viewer
    sees it under a moving camera (scene units of the home frame);
    ``visible``, ``screen`` and the naming fields are read at a play's
    first and last frames only.
    """

    id: str
    parent: str | None
    centre: np.ndarray
    size: np.ndarray
    view: np.ndarray
    members: int
    own: bool
    visible: bool = False
    screen: tuple[float, float] | None = None
    type: str = ""
    text: str | None = None
    scene_line: int | None = None


@dataclass(frozen=True, eq=False)
class _Frame:
    """One frame: its time, its play, the zoom of its view and its bodies,
    each group before its members.
    """

    t: Fraction
    play: int
    zoom: float
    bodies: dict[str, _Body]


@dataclass(eq=False)
class _Play:
    """What one play's frames show of its bodies: how far each travelled,
    its largest step between two frames (in the scene, whatever the camera
    did) and where it was last seen.
    """

    record: PlayRecord
    frames: list[_Frame]
    travel: dict[str, float] = field(default_factory=dict)
    largest: dict[str, float] = field(default_factory=dict)
    last: dict[str, _Body] = field(default_factory=dict)

    @property
    def first_frame(self) -> _Frame:
        return self.frames[0]

    @property
    def last_frame(self) -> _Frame:
        return self.frames[-1]

    def moved(self, body: str) -> bool:
        return self.travel.get(body, 0.0) >= TRAVEL_MIN


@dataclass(frozen=True)
class MotionJump:
    """The ``motion.jump`` rule at a tolerance, in home-frame units."""

    jump_tol: float = JUMP_TOL

    def request_frames(self, frames: Frames) -> None:
        """Ask the lint for every frame the render shows of each play."""
        wanted = _wanted(frames.session.record)
        edges = {t for _, shown in wanted for t in (shown[0], shown[-1])}

        def read(
            source: Record,
            t: Fraction,
            play: PlayRecord | None,
            capture: Capture,
            live: bool,
        ) -> tuple[float, dict[str, _Body]]:
            return _read(source, play, capture, live=live, edge=t in edges)

        frames.request(self, [t for _, shown in wanted for t in shown], read)

    def __call__(self, context: LintContext) -> list[Finding]:
        record = context.record
        read = iter(context.frames.result(self))
        plays = [
            _measure(_Play(play, [_Frame(t, play.index, *next(read)) for t in shown]))
            for play, shown in _wanted(record)
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


def _read(
    record: Record,
    play: PlayRecord | None,
    capture: Capture,
    *,
    live: bool,
    edge: bool,
) -> tuple[float, dict[str, _Body]]:
    """The zoom of one frame's view and its bodies, each group before its
    members: of the live scene of a forward run, or of ``capture`` shown
    from the record.
    """
    if live:
        return _bodies(record, capture, edge=edge, live=True)
    static = list(play.static) if play is not None and capture.moving else []
    with Drawing(record) as drawing:
        drawing.show(capture, [*capture.draw_list(), *static, *capture.moving])
        return _bodies(record, capture, edge=edge, live=False)


def _bodies(
    record: Record, capture: Capture, *, edge: bool, live: bool
) -> tuple[float, dict[str, _Body]]:
    camera = record._scene.renderer.camera
    timeline = record._scene.timeline
    view = view_of(
        camera, home=timeline.home, configured_pixels=timeline.configured_pixels
    )
    zoom_x = view.home_width / view.frame_width
    zoom_y = view.home_height / view.frame_height
    # As the viewer sees a centre under a moving camera; every other camera
    # shows the home frame.
    cx, cy = view.frame_center[:2] if view.camera == "moving" else (0.0, 0.0)
    # At a play's edges, the projection places each body on screen: a live
    # run's camera holds the map it draws with, a shown capture maps paths
    # around the centre the render froze.
    projected = (view if live else record_view(record)) if edge else None
    bodies: dict[str, _Body] = {}
    for mob, parent in _candidates(capture.draw_list(), _viewing_nodes(camera)):
        origin = record.origin(mob)
        if origin.id in bodies:
            continue
        leaves = [m for m in mob.get_family() if len(m.points)]
        lit = [m for m in leaves if _lays(m)]
        if not lit:
            continue
        points = np.concatenate([np.asarray(m.points, dtype=float)[:, :2] for m in lit])
        low, high = points.min(axis=0), points.max(axis=0)
        centre = (low + high) / 2
        named: dict[str, Any] = {}
        if projected is not None:
            named = {
                "visible": any(_visible(m) for m in leaves),
                "screen": _screen_centre(camera, projected, lit),
                "type": type(mob).__name__,
                "text": _label(mob),
                "scene_line": origin.scene_line,
            }
        bodies[origin.id] = _Body(
            id=origin.id,
            parent=record.origin(parent).id if parent is not None else None,
            centre=centre,
            size=high - low,
            view=np.array([(centre[0] - cx) * zoom_x, (centre[1] - cy) * zoom_y]),
            members=len(leaves),
            own=_lays(mob),
            **named,
        )
    return zoom_x, bodies


def _screen_centre(
    camera: Any, view: View, lit: list[Mobject]
) -> tuple[float, float] | None:
    """The centre, in home-frame units, of the box the projection places
    ``lit``'s points in.
    """
    boxes = [box for m in lit if (box := member_box(camera, view, m, home=True))]
    if not boxes:
        return None
    return (
        (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2,
        (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2,
    )


def _candidates(
    roots: list[Mobject], hidden: set[int]
) -> Iterator[tuple[Mobject, Mobject | None]]:
    """The scene's top-level mobjects with points, each followed by its
    direct children with points (none of a text's or a surface's), with the
    top-level holding each child; the camera's own are none of them.
    """
    from arachne.mobject.three_d.three_dimensions import Surface
    from arachne.mobject.value_tracker import ValueTracker

    whole = (*_text_types(), Surface)

    def drawn(mob: Mobject) -> bool:
        return (
            id(mob) not in hidden
            and not isinstance(mob, ValueTracker)
            and any(len(m.points) for m in mob.get_family())
        )

    seen: set[int] = set()
    for top in roots:
        if id(top) in seen or id(top) in hidden:
            continue
        seen.add(id(top))
        if drawn(top):
            yield top, None
        if isinstance(top, whole):
            continue
        for child in top.submobjects:
            if id(child) not in seen and drawn(child):
                seen.add(id(child))
                yield child, top


def _lays(mob: Mobject) -> bool:
    """Whether one member's own draw paints anything: a path or an image
    whose points are not all in one place, or a point cloud with a visible
    stroke.
    """
    kind = draw_kind(mob)
    points = np.asarray(mob.points)
    if not len(points) or kind is None:
        return False
    if kind == "points":
        _, alpha, _ = _paint(mob, kind)
        return alpha > 0 and float(np.max(getattr(mob, "stroke_width", 0))) > 0
    return bool((np.abs(points - points[0]) >= _ONE_PLACE).any())


def _visible(mob: Mobject) -> bool:
    """Whether a member lays ink a viewer can see move: a fill, or a stroke
    with width and opacity, over more than one place; a visible point
    cloud; an image not wholly transparent.
    """
    kind = draw_kind(mob)
    if kind is None or not len(mob.points):
        return False
    if kind == "image":
        raster = np.asarray(getattr(mob, "pixel_array", np.zeros((0, 0, 4))))
        return not (raster.ndim == 3 and raster.shape[-1] == 4) or bool(
            raster[..., 3].any()
        )
    if kind == "points":
        return _lays(mob)
    fill, stroke, _ = _paint(mob, kind)
    spread = np.ptp(np.asarray(mob.points)[:, :2], axis=0).max()
    return (fill > 0 or stroke > 0) and float(spread) > 0


def _label(mob: Mobject) -> str | None:
    from arachne.timeline.state import _text

    return _text(mob) if isinstance(mob, _text_types()) else None


def _measure(play: _Play) -> _Play:
    """``play`` with each body's travel, largest step and last sight."""
    previous: _Frame | None = None
    for frame in play.frames:
        for key, body in frame.bodies.items():
            play.last[key] = body
            if previous is None or key not in previous.bodies:
                continue
            was = previous.bodies[key]
            # The body's own step, in the scene: a camera moving with it
            # cancels none of it, and a cut scales it as it does the cut.
            step = float(np.hypot(*(body.centre - was.centre)))
            play.largest[key] = max(play.largest.get(key, 0.0), step)
            if _kept_size(was.size, body.size, RIGID):
                play.travel[key] = play.travel.get(key, 0.0) + step
        previous = frame
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

    body: _Body
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
        self.left = before.last_frame.bodies
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
        # Both the scene's, at the later frame's zoom: the step from the last
        # frame to where the play left the body is one of the play's own.
        if distance <= CHURN * mover.largest.get(key, 0.0) * self.zoom:
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
        start = self.left[key].screen
        end = body.screen
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


def _point(xy: tuple[float, float] | None) -> list[float] | None:
    return None if xy is None else [round(float(v), 4) for v in xy]


def _where(xy: tuple[float, float] | None) -> str:
    if xy is None:
        return "off the picture"
    return f"({xy[0]:.2f}, {xy[1]:.2f})"
