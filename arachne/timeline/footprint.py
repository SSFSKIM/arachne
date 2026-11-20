"""Where each object of a recorded frame draws, found by drawing it again.

A comparison of two renders asks, on sampled frames, which pixels each object
covers and which objects changed between two frames (:mod:`arachne_compare`).
Cairo reports neither, so a frame is drawn again from the record with named
objects left out of every capture (:class:`.Drawing`'s capture filter): an
object's footprint is where the frame without it differs from the frame with
it. Nothing recorded moves and no callback runs differently; only what is
painted changes.

The objects are **drawables**, what the Frame IR gives a slot
(:class:`~arachne.frame_ir.extract.FrameStream`): each leaf a capture paints,
or the surface whose faces it paints as one group. A leaf that holds
submobjects of its own is one drawable, for its own painting. A grouping
node's id is accepted wherever ids are asked and stands for the drawables
under it; it is never an entry of a map or of a changed set.

Cairo only: on another draw path, under a camera that draws into displays
(a ``MultiCamera``), or for a scene or an instant the record cannot draw,
every call raises :class:`ReplayUnsupported`, and so does making a
:class:`SessionFrames` or :class:`ReplayMaps` of such a session. A frame on
the ``gpu`` route is drawn again as the session draws it, its runs by Cairo.

An object drawn only through another, as its capture dependency (a layer's
mask, a window's content; :data:`~.record.DEPENDENCIES`), is no drawable: it
is in no map and in no changed set, and asking for it by id raises
:class:`ReplayUnsupported`. Its versions count as its holder's: when it
changes, every drawable under its holder has changed.
"""

from __future__ import annotations

__all__ = [
    "ReplayMaps",
    "ReplayUnsupported",
    "SessionFrames",
    "changed_between",
    "drawables",
    "frame_without",
    "object_map",
]

import bisect
import contextlib
import hashlib
import math
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from arachne_compare import Box, ChangedSet, Footprint, ObjectMap

from arachne import config
from arachne.camera.multi_camera import MultiCamera
from arachne.camera.projection import (
    draw_kind,
    member_box,
    pixel_box,
    record_view,
    render_mapping_centre,
)
from arachne.frame_ir.extract import FrameStream
from arachne.frame_ir.painter import camera_path
from arachne.mobject.mobject import Mobject
from arachne.utils.family import extract_mobject_family_members

from .clock import exact, frames_in
from .draw import CameraExtras, _constant, _same_value
from .record import SAMPLE_DECIMALS, NodeState, content_fields, dependencies
from .restore import Unrestorable
from .routes import RecordRefused, _resolve

if TYPE_CHECKING:
    from arachne.session import Session

    from .record import Capture, Record
    from .routes import _Resolved


class ReplayUnsupported(Exception):
    """The frame can't be drawn again from the record on Cairo: the session
    draws on another path, or the record can't draw the scene or the instant.
    """


#: A drawable's box when it paints nothing.
_NO_BOX: Box = (0, 0, 0, 0)


@dataclass
class _Seen:
    """What one draw of an instant painted.

    ``boxes`` holds each drawable painted, in paint order, with its box in
    the frame's pixels; ``sources`` its source line where known; ``order``
    the drawables in the order painted, a drawable painted twice twice.
    ``held`` are the asked ids the frame holds, ``groups`` those of them that
    name a group (painting nothing of its own, holding submobjects), and
    ``resolved`` the drawables each asked id stands for. ``versions`` holds each drawable's recorded
    states (itself, then the leaves it paints), and ``evaluated`` the
    content key of those an evaluated instant changed, as evaluated there.
    """

    pixels: np.ndarray | None = None
    boxes: dict[str, Box] = field(default_factory=dict)
    sources: dict[str, str | None] = field(default_factory=dict)
    held: set[str] = field(default_factory=set)
    groups: set[str] = field(default_factory=set)
    resolved: dict[str, set[str]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    versions: dict[str, tuple[NodeState | None, ...]] = field(default_factory=dict)
    evaluated: dict[str, bytes] = field(default_factory=dict)
    camera: _CameraView | None = None


@dataclass(frozen=True)
class _CameraView:
    """The camera's effective inputs at a drawn instant, by value.

    ``inputs`` are its plain values and background as the draw installed
    them, read calling no centre function. ``view`` is its projection as the
    draw holds it (:class:`~arachne.camera.projection.View`: the frame a
    MovingCamera's frame mobject gives, a ThreeDCamera's angles, distance
    and zoom), with its light's place, and ``centers`` the fixed-orientation
    centres the draw installed (a restore's evaluated ones where it
    restored), numbers rounded to
    :data:`~.record.SAMPLE_DECIMALS` places: a mobject animated to where it
    is moves by ulps and by nothing a frame shows.
    """

    inputs: CameraExtras | None
    view: tuple[object, ...]
    centers: tuple[tuple[int, object], ...]

    def same(self, other: _CameraView) -> bool:
        return (
            _same_inputs(self.inputs, other.inputs)
            and self.view == other.view
            and self.centers == other.centers
        )


def _same_inputs(a: CameraExtras | None, b: CameraExtras | None) -> bool:
    """Whether two cameras' plain values, fixed-in-frame mobjects and
    backgrounds are equal by value.
    """
    if a is None or b is None:
        return a is b
    return (
        len(a.values) == len(b.values)
        and all(
            k == m and _same_value(v, w)
            for (k, v), (m, w) in zip(a.values, b.values, strict=True)
        )
        and a.fixed_in_frame == b.fixed_in_frame
        and _same_background(a.background, b.background)
    )


def _rounded(value: object) -> object:
    """``value`` with every number in it rounded, arrays and tuples as tuples."""
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, SAMPLE_DECIMALS) + 0.0
    if isinstance(value, np.ndarray):
        return tuple(_rounded(float(x)) for x in value.ravel())
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(_rounded(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, (tuple, list)):
        return tuple(_rounded(x) for x in value)
    if isinstance(value, (np.floating, np.integer)):
        return _rounded(value.item())
    return repr(value)


#: The code of the centre functions a drawing installs, each returning a
#: recorded or evaluated value (:meth:`.CameraExtras.install`).
_CONSTANT_CODE = _constant(np.zeros(3)).__code__


def _installed_centers(
    camera: object, capture: Capture
) -> tuple[tuple[Mobject, np.ndarray], ...]:
    """The fixed-orientation centres the camera draws with now, as a drawing
    installed them: its capture's, or a restore's evaluated ones. Read from
    the installed functions when every one of them is a drawing's, which
    only return a value; else the capture's own record, calling no
    function of the scene's.
    """
    orientations = getattr(camera, "fixed_orientation_mobjects", None)
    if orientations and all(
        getattr(func, "__code__", None) is _CONSTANT_CODE
        for func in orientations.values()
    ):
        return tuple((mob, func()) for mob, func in orientations.items())
    recorded = capture.camera
    return recorded.centers if recorded is not None else ()


def _same_background(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.shape == b.shape and bool(np.array_equal(a, b))


class _Filter:
    """The capture filter of one draw: it leaves out the drawables the asked
    ``ids`` stand for when ``remove`` is set, and notes what it paints.
    """

    def __init__(
        self,
        session: Session,
        resolved: _Resolved,
        ids: Collection[str],
        *,
        remove: bool,
        watch: bool = False,
    ) -> None:
        self.record = session.record
        self.camera = session.scene.renderer.camera
        self.capture: Capture = resolved.capture
        self.evaluated = resolved.prepare is not None
        self.ids = frozenset(ids)
        self.remove = remove
        self.watch = watch
        self.seen = _Seen()
        #: The tree's members the asked ids name, found at the first capture.
        self._named: list[Mobject] | None = None
        #: The nodes the capture dependencies of each member's holders
        #: reach, by member, found at the first capture.
        self._depends: dict[int, list[Mobject]] = {}
        #: The centre the render froze, found before the frame is drawn:
        #: finding it draws, which a frame being drawn reads nothing for.
        self._frozen = render_mapping_centre(self.record) if watch else None
        if self.ids:
            _refuse_dependencies(self.record, self.capture, self.ids)

    def __call__(
        self, handed: Sequence[Mobject], leaves: list[Mobject]
    ) -> list[Mobject]:
        if self._named is None:
            self._begin()
        assert self._named is not None
        camera = self.camera
        faces = FrameStream()._faces(handed, leaves)
        owners: dict[int, Mobject] = {}
        for leaf in leaves:
            if camera.type_or_raise(leaf) is not Mobject:
                face = faces.get(id(leaf))
                owners[id(leaf)] = face[0] if face is not None else leaf
        drawn = {id(owner): owner for owner in owners.values()}
        out: set[int] = set()
        for mob in self._named:
            name = self.record.origin(mob).id
            if id(mob) in drawn or _paints_itself(mob):
                # A drawable stands for itself, not for what it holds.
                found = [mob] if id(mob) in drawn else []
            else:
                family = {id(member) for member in mob.get_family()}
                found = [owner for key, owner in drawn.items() if key in family]
            for owner in found:
                self.seen.resolved.setdefault(name, set()).add(
                    self.record.origin(owner).id
                )
                out.add(id(owner))
        if self.watch:
            self._note(leaves, owners, drawn)
        if not self.remove:
            return leaves
        return [
            leaf
            for leaf in leaves
            if id(leaf) not in owners or id(owners[id(leaf)]) not in out
        ]

    def _begin(self) -> None:
        """Find what the asked ids name in the frame's tree, as the draw
        shows it, and note the camera's own state.
        """
        named: list[Mobject] = []
        if self.ids:
            # What the frame paints: the scene's mobjects, and the moving ones
            # a play still paints once the scene removed them.
            capture = self.capture
            members = extract_mobject_family_members(
                [*capture.draw_list(), *capture.moving]
            )
            seen: set[int] = set()
            for mob in members:
                name = self.record.origin(mob).id
                if name in self.ids and id(mob) not in seen:
                    seen.add(id(mob))
                    self.seen.held.add(name)
                    named.append(mob)
                    if mob.submobjects and not _paints_itself(mob):
                        self.seen.groups.add(name)
        self._named = named
        if self.watch:
            self._depends = _held_through(self.record, self.capture)
            camera = self.camera
            light = getattr(camera, "light_source", None)
            self.seen.camera = _CameraView(
                CameraExtras.observe(camera, None, evaluate=False),
                (
                    _rounded(record_view(self.record, frozen=self._frozen)),
                    None if light is None else _rounded(light.get_center()),
                ),
                tuple(
                    (id(mob), _rounded(center))
                    for mob, center in _installed_centers(camera, self.capture)
                ),
            )

    def _note(
        self,
        leaves: list[Mobject],
        owners: Mapping[int, Mobject],
        drawn: Mapping[int, Mobject],
    ) -> None:
        """Note each drawable this capture paints: its box, source line and
        states.
        """
        camera = self.camera
        view = record_view(self.record, frozen=self._frozen)
        painted: dict[int, list[Mobject]] = {}
        for leaf in leaves:
            owner = owners.get(id(leaf))
            if owner is not None:
                painted.setdefault(id(owner), []).append(leaf)
        seen = self.seen
        for key, owner in drawn.items():
            origin = self.record.origin(owner)
            name = origin.id
            if owner is painted[key][0]:
                found = member_box(camera, view, owner, ink=True)
            else:
                found = pixel_box(camera, view, owner, ink=True)
            box = _pixel_box(found, camera.pixel_width, camera.pixel_height)
            seen.boxes[name] = _union(seen.boxes.get(name), box)
            seen.order.append(name)
            seen.sources[name] = (
                None if origin.source_line is None else str(origin.source_line)
            )
            mobs = [owner, *(leaf for leaf in painted[key] if leaf is not owner)]
            # What it is drawn through changes it too.
            mobs.extend(self._depends.get(key, ()))
            seen.versions[name] = self._states(mobs)
            evaluated = self._evaluated_key(mobs)
            if evaluated is not None:
                seen.evaluated[name] = evaluated

    def _states(self, mobs: Sequence[Mobject]) -> tuple[NodeState | None, ...]:
        return tuple(self.record.node_state(mob, self.capture) for mob in mobs)

    def _evaluated_key(self, mobs: Sequence[Mobject]) -> bytes | None:
        """The content key of ``mobs`` as the evaluated instant holds them,
        when it changed any of them from its capture; else ``None``.
        """
        if not self.evaluated:
            return None
        states = self._states(mobs)
        if all(
            state is not None and state.matches(vars(mob))
            for mob, state in zip(mobs, states, strict=True)
        ):
            return None
        return _content_key(NodeState.of(mob) for mob in mobs)


def _holders(record: Record, capture: Capture) -> Iterator[tuple[Mobject, Mobject]]:
    """Each member of the frame's tree at ``capture`` under a holder of
    capture dependencies, with that holder, nearest first.
    """
    stack: list[tuple[Mobject, tuple[Mobject, ...]]] = [
        (mob, ()) for mob in reversed([*capture.draw_list(), *capture.moving])
    ]
    seen: set[int] = set()
    while stack:
        mob, above = stack.pop()
        if id(mob) in seen:
            continue
        seen.add(id(mob))
        state = record.node_state(mob, capture)
        if state is None:
            continue
        if dependencies(state):
            above = (mob, *above)
        for holder in above:
            yield mob, holder
        stack.extend((sub, above) for sub in reversed(state.submobjects))


def _reached(record: Record, capture: Capture, holder: Mobject) -> list[Mobject]:
    """The nodes ``holder``'s capture dependencies reach at ``capture``."""
    state = record.node_state(holder, capture)
    assert state is not None
    return [mob for mob, _ in record._walk(capture, dependencies(state))]


def _held_through(record: Record, capture: Capture) -> dict[int, list[Mobject]]:
    """For each member of the frame's tree under a holder of capture
    dependencies, the nodes they reach, by the member's id.
    """
    reached: dict[int, list[Mobject]] = {}
    found: dict[int, list[Mobject]] = {}
    for mob, holder in _holders(record, capture):
        if id(holder) not in reached:
            reached[id(holder)] = _reached(record, capture, holder)
        found.setdefault(id(mob), []).extend(reached[id(holder)])
    return found


def _refuse_dependencies(
    record: Record, capture: Capture, ids: Collection[str]
) -> None:
    """Raise :class:`ReplayUnsupported` for an id of ``ids`` naming a node
    drawn only through its holder, as a capture dependency.
    """
    done: set[int] = set()
    for _, holder in _holders(record, capture):
        if id(holder) in done:
            continue
        done.add(id(holder))
        for mob in _reached(record, capture, holder):
            name = record.origin(mob).id
            if name in ids:
                raise ReplayUnsupported(
                    f"{name} is drawn only through {record.origin(holder).id}, as"
                    " what its layer reads (a mask, or a window's content): it has"
                    " no footprint of its own and is not left out alone"
                )


def _paints_itself(mob: Mobject) -> bool:
    """Whether ``mob`` is a leaf a camera paints: it has points of its own and
    a kind of draw.
    """
    return bool(len(mob.points)) and draw_kind(mob) is not None


def _content_key(states: Iterable[NodeState | None]) -> bytes:
    """A digest of the values ``states`` hold (:func:`.record.content_fields`)."""
    digest = hashlib.blake2b(digest_size=16)
    for state in states:
        if state is None:
            digest.update(b"\x00")
            continue
        digest.update(b"\x01")
        for key, part, _ in content_fields(state):
            for data in (key.encode(), part):
                digest.update(len(data).to_bytes(4, "big"))
                digest.update(data)
    return digest.digest()


def _pixel_box(
    found: tuple[float, float, float, float] | None, width: int, height: int
) -> Box:
    """A box of floats as whole pixels holding it, one more on every side
    for antialiasing, kept to the frame; :data:`_NO_BOX` for none.
    """
    if found is None:
        return _NO_BOX

    def clamp(value: float, top: int) -> int:
        return min(max(int(value), 0), top)

    x0 = clamp(math.floor(found[0]) - 1, width)
    y0 = clamp(math.floor(found[1]) - 1, height)
    x1 = max(clamp(math.ceil(found[2]) + 1, width), x0)
    y1 = max(clamp(math.ceil(found[3]) + 1, height), y0)
    return (x0, y0, x1, y1)


def _union(a: Box | None, b: Box) -> Box:
    if a is None or a == _NO_BOX:
        return b
    if b == _NO_BOX:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


@contextlib.contextmanager
def _replayable(session: Session) -> Iterator[None]:
    """Raise :class:`ReplayUnsupported` for a session that does not draw on
    Cairo, and for an instant the record can't draw.
    """
    _check(session)
    try:
        yield
    except (RecordRefused, Unrestorable) as error:
        raise ReplayUnsupported(str(error)) from error


def _check(session: Session) -> None:
    """Raise :class:`ReplayUnsupported` unless the session's frames can be
    drawn again on Cairo, one camera drawing them.
    """
    camera = session.scene.renderer.camera
    path = session.draw_path if session.draw_path is not None else config.draw_path
    if camera_path(camera, path) != "cairo":
        raise ReplayUnsupported(f"the session draws on {path!r}, not on 'cairo'")
    if isinstance(camera, MultiCamera):
        raise ReplayUnsupported(
            "the session's camera draws into displays of other cameras"
        )
    if session.record.refusal is not None:
        raise ReplayUnsupported(session.record.refusal)


def _draw(
    session: Session,
    t: Fraction,
    quality: str | None,
    ids: Collection[str] = (),
    *,
    remove: bool = False,
    watch: bool = False,
) -> _Seen:
    """Draw the frame at ``t`` whole, leaving out what ``ids`` stand for when
    ``remove`` is set, noting what it paints when ``watch`` is.
    """
    with _replayable(session):
        resolved = _resolve(session.record, t)
        leave_out = _Filter(session, resolved, ids, remove=remove, watch=watch)
        pixels = session._frame_leaving_out(t, quality, leave_out)
    seen = leave_out.seen
    seen.pixels = pixels
    return seen


def drawables(
    session: Session, t: Fraction | float, quality: str | None = None
) -> Mapping[str, Box]:
    """The drawables of the frame at ``t``, in paint order, each with its
    box (``x0, y0, x1, y1``, half-open, y down) in pixels of the frame drawn
    at ``quality``: what it paints, with a pixel more on every side;
    ``(0, 0, 0, 0)`` for one that paints nothing.
    """
    return _draw(session, exact(t), quality, watch=True).boxes


def frame_without(
    session: Session,
    t: Fraction | float,
    ids: Collection[str],
    quality: str | None = None,
) -> np.ndarray:
    """The frame at ``t`` drawn whole without the drawables ``ids`` stand
    for, as the session draws it otherwise (:meth:`.Session.frame`), as an
    RGBA array. An id the frame does not hold leaves it as it is.
    """
    seen = _draw(session, exact(t), quality, ids, remove=True)
    assert seen.pixels is not None
    return seen.pixels


def object_map(
    session: Session,
    t: Fraction | float,
    ids: Sequence[str],
    quality: str | None = None,
) -> ObjectMap:
    """Where each of ``ids`` draws in the frame at ``t``: for each drawable
    an id stands for, the pixels the frame without it differs at, in any
    channel. An id the scene does not hold there is examined and empty. The
    map is not complete: a drawable no id stands for is untested.
    """
    instant = exact(t)
    return _map(
        session,
        instant,
        quality,
        _draw(session, instant, quality, ids, watch=True),
        ids,
    )


def _map(
    session: Session,
    t: Fraction,
    quality: str | None,
    seen: _Seen,
    ids: Sequence[str],
) -> ObjectMap:
    """The map of ``ids`` at ``t``, from ``seen``, the whole frame's draw
    there (which resolved ``ids``).
    """
    assert seen.pixels is not None
    whole = seen.pixels
    height, width = whole.shape[:2]
    objects: dict[str, Footprint] = {}
    for name in ids:
        if not seen.resolved.get(name) and name not in seen.groups:
            # Not held, or held painting nothing (points cleared): it draws
            # nothing here. A group holding nothing painted gets no entry.
            objects[name] = Footprint(
                _NO_BOX, np.zeros((0, 0), dtype=bool), source=seen.sources.get(name)
            )
    wanted = sorted(set().union(*(seen.resolved.get(name, ()) for name in ids)))
    for name in wanted:
        without = _draw(session, t, quality, [name], remove=True).pixels
        assert without is not None
        changed = np.any(without != whole, axis=2)
        rows = np.flatnonzero(changed.any(axis=1))
        if not rows.size:
            objects[name] = Footprint(
                _NO_BOX, np.zeros((0, 0), dtype=bool), source=seen.sources.get(name)
            )
            continue
        columns = np.flatnonzero(changed.any(axis=0))
        box = (int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)
        objects[name] = Footprint(
            box,
            changed[box[1] : box[3], box[0] : box[2]],
            source=seen.sources.get(name),
        )
    return ObjectMap((height, width), objects, complete=False)


def changed_between(
    session: Session, t0: Fraction | float, t1: Fraction | float
) -> ChangedSet:
    """The drawables that changed between the frames at ``t0`` and ``t1``.

    A drawable changed when the record holds another version of it, or of a
    leaf it paints, at one capture than at the other, or when it is drawn
    at one instant only. Where an instant is evaluated between captures,
    the drawables the evaluation changed are ``undecided`` unless their
    evaluated values are equal at both. ``camera`` is set when the camera's
    effective inputs differ by value (:class:`_CameraView`): its plain
    values and background, its projection and light, and the centres it
    holds fixed-orientation mobjects at.
    """
    quality = session._current_draw_quality()
    a = _draw(session, exact(t0), quality, watch=True)
    b = _draw(session, exact(t1), quality, watch=True)
    changed = set(a.versions).symmetric_difference(b.versions)
    undecided: set[str] = set()
    for name in set(a.versions) & set(b.versions):
        if name in a.evaluated or name in b.evaluated:
            if _key(a, name) != _key(b, name):
                undecided.add(name)
        elif not _same_versions(a.versions[name], b.versions[name]):
            changed.add(name)
    assert a.camera is not None
    assert b.camera is not None
    changed |= _reordered(a.order, b.order)
    camera = not a.camera.same(b.camera)
    return ChangedSet(frozenset(changed), frozenset(undecided), camera)


def _reordered(first: Sequence[str], second: Sequence[str]) -> set[str]:
    """The drawables of both paint orders whose place among the others
    changed: those outside a longest run the two orders share, each drawable
    at the place it is painted last (on top).
    """
    common = set(first) & set(second)

    def last(order: Sequence[str]) -> list[str]:
        places = {name: k for k, name in enumerate(order) if name in common}
        return sorted(places, key=places.__getitem__)

    a, b = last(first), last(second)
    if a == b:
        return set()
    place = {name: k for k, name in enumerate(a)}
    sequence = [place[name] for name in b]
    # A longest increasing run of ``sequence`` (patience sorting).
    tails: list[int] = []
    tail_at: list[int] = []
    before = [-1] * len(sequence)
    for k, value in enumerate(sequence):
        slot = bisect.bisect_left(tails, value)
        if slot:
            before[k] = tail_at[slot - 1]
        if slot == len(tails):
            tails.append(value)
            tail_at.append(k)
        else:
            tails[slot] = value
            tail_at[slot] = k
    kept: set[str] = set()
    k = tail_at[-1]
    while k >= 0:
        kept.add(b[k])
        k = before[k]
    return set(b) - kept


def _key(seen: _Seen, name: str) -> bytes:
    found = seen.evaluated.get(name)
    return found if found is not None else _content_key(seen.versions[name])


def _same_versions(
    a: tuple[NodeState | None, ...], b: tuple[NodeState | None, ...]
) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


class SessionFrames:
    """A session's frames as a render writes them (``arachne_compare.Frames``):
    each play's frames of the global grid at the frame rate the render draws
    at, but a skipped play's, drawn at ``quality`` (:meth:`.Session.frame`);
    a scene that plays nothing is its final frame alone, as a render saves
    it. Only a session whose frames can be drawn again on Cairo is taken.

    ``plays`` are each play's frames as a half-open span of frame numbers;
    ``fps`` is the frame rate.
    """

    def __init__(self, session: Session, quality: str | None = None) -> None:
        _check(session)
        self.session = session
        self.quality = quality
        rates = {
            play.frame_rate
            for play in session.record.plays
            if not play.skipped and play.frame_rate is not None
        }
        if len(rates) > 1:
            raise ReplayUnsupported(
                "the scene's plays were drawn at different frame rates: "
                + ", ".join(str(rate) for rate in sorted(rates))
            )
        self.fps: Fraction = rates.pop() if rates else session._render_frame_rate()
        instants: list[Fraction] = []
        plays: list[tuple[int, int]] = []
        for play in session.record.plays:
            if play.skipped:
                continue
            found = frames_in(play.start, play.end, self.fps)
            if found:
                plays.append((len(instants), len(instants) + len(found)))
                instants.extend(found)
        if not session.record.plays:
            final = session.record.final
            assert final is not None, "the scene has not finished"
            instants, plays = [final.instant], [(0, 1)]
        self.instants: tuple[Fraction, ...] = tuple(instants)
        self.plays: Sequence[tuple[int, int]] = tuple(plays)

    def __len__(self) -> int:
        return len(self.instants)

    def instant(self, i: int) -> Fraction:
        return self.instants[i]

    def frame(self, i: int) -> np.ndarray:
        return np.asarray(self.session.frame(self.instants[i], self.quality))


class ReplayMaps:
    """The object maps of two sessions' frames, drawn again on Cairo: the
    comparator's map source (``arachne_compare.MapSource``) for a reference
    and a candidate drawn at ``quality``, frame ``i`` being the ``i``-th
    frame of each one's :class:`SessionFrames`.

    :meth:`boxes` gives each side's drawables, from which the comparator
    chooses the ids it examines; :meth:`map` examines them on one side, so
    an id chosen for what one side shows is examined on the other too.
    :meth:`changed` is the candidate's changed set between two of its
    frames, the comparator's ``changed``.
    """

    def __init__(
        self, reference: Session, candidate: Session, quality: str | None = None
    ) -> None:
        self.quality = quality
        self.sessions = {"reference": reference, "candidate": candidate}
        self.frames = {
            "reference": SessionFrames(reference, quality),
            "candidate": SessionFrames(candidate, quality),
        }

    def boxes(self, frame: int) -> tuple[Mapping[str, Box], Mapping[str, Box]]:
        return (
            drawables(
                self.sessions["reference"],
                self.frames["reference"].instant(frame),
                self.quality,
            ),
            drawables(
                self.sessions["candidate"],
                self.frames["candidate"].instant(frame),
                self.quality,
            ),
        )

    def map(self, side: str, frame: int, ids: Sequence[str] | None) -> ObjectMap:
        session = self.sessions[side]
        t = self.frames[side].instant(frame)
        if ids is None:
            ids = list(drawables(session, t, self.quality))
        return object_map(session, t, ids, self.quality)

    def changed(self, i: int, j: int) -> ChangedSet:
        frames = self.frames["candidate"]
        return changed_between(
            self.sessions["candidate"], frames.instant(i), frames.instant(j)
        )
