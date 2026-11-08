"""The probe: a state-function play restored at instants its record kept,
against what the record kept there.

Admission proves each callback of a state-function play (route 2) a
function of the instant, within what its proofs vouch for (:mod:`.purity`).
A callback outside that, or a frame hook writing what the play's frames
show, keeps history no proof saw, and a restore between the play's samples
would answer from a state the forward run never had. The probe is the
backstop: for each state play it restores two instants the record kept and
a frame read, the first and the last inside the play (neither its start nor
an event), each from the play's start or its last event before it, never
from the instant itself, and compares the restored scene with the capture
kept there:

- the draw lists, the tree (every node's type and its children, in order)
  exactly;
- every field a content key covers (:func:`.record.covered_fields`):
  paint exactly (a paint value by its key, an array of a paint kind such as
  a point cloud's colours, a number in a field that holds paint), geometry
  (points, placements, other numbers) within :data:`.execution.TOLERANCE`,
  anything else by its content key (:func:`.keys.value_key`), with the
  mobjects it refers to by their places in the walk, as children are; a
  value of a type the key does not know never agrees;
- the camera's inputs: its paint (background colour and opacity, line
  width, a supplied background) exactly, its other values and its
  fixed-orientation centres within the tolerance, and the same for every
  camera drawing into one of its displays.

A play that differs is found ``demoted`` and reported as a
``replay.diverged`` divergence of kind ``probe`` at its line; the session
demotes it (:func:`.admission.demote`) and records its scene again with the
play on route 3 (:class:`~arachne.session.Session`).
A play with fewer than two such instants is ``unprobed``, and one whose
restore refuses at them (:class:`.restore.Unrestorable`) is
``unrestorable``: neither is demoted. A record that refuses drawing answers
no instant, and is not probed.

The probe never runs in a render.
"""

from __future__ import annotations

__all__ = ["PlayProbe", "Probe", "probe"]

import re
import reprlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .clock import TICK_RATE
from .draw import CameraExtras, camera_nodes
from .execution import KEY_DECIMALS, TOLERANCE, Divergence
from .keys import marker_reason, value_key
from .record import DepthReference, NodeState, covered_fields
from .restore import Unrestorable, restored

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject

    from .record import Capture, PlayRecord, Record


@dataclass(frozen=True)
class PlayProbe:
    """What the probe found for one state-function play.

    ``outcome`` is ``"agreed"``, ``"demoted"`` (a restore differed from the
    record; ``reason`` says how), ``"unprobed"`` (fewer than two instants to
    restore) or ``"unrestorable"`` (``reason`` is the restore's refusal
    code). ``instants`` are the instants restored, and ``bases`` the
    instant each was restored from.
    """

    play: int
    outcome: str
    instants: tuple[Fraction, ...] = ()
    bases: tuple[Fraction, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class Probe:
    """What the probe found for every state-function play of a record, in
    play order, and the divergences it reported.
    """

    plays: tuple[PlayProbe, ...] = ()
    divergences: tuple[Divergence, ...] = ()
    #: Whether the session holds the record the probe last checked, every
    #: play it demoted ticked there; false when the session stopped
    #: recording again first, because an execution played other plays than
    #: the one before it or its passes reached their cap.
    settled: bool = True

    @property
    def demoted(self) -> tuple[int, ...]:
        """The plays demoted, by index."""
        return tuple(p.play for p in self.plays if p.outcome == "demoted")

    @property
    def unprobed(self) -> int:
        """How many plays had fewer than two instants to restore."""
        return sum(p.outcome == "unprobed" for p in self.plays)

    def moved(self, places: Mapping[int, int | None]) -> Probe:
        """This probe, its plays moved to the indices ``places`` gives them
        in another execution of the scene: a play that one does not play is
        left out, and a divergence of it names no play.
        """
        plays = sorted(
            (
                replace(each, play=place)
                for each in self.plays
                if (place := places.get(each.play)) is not None
            ),
            key=lambda each: each.play,
        )
        divergences = tuple(
            replace(d, play=places.get(d.play)) if d.play is not None else d
            for d in self.divergences
        )
        return Probe(tuple(plays), divergences, self.settled)

    def then(self, later: Probe) -> Probe:
        """This probe followed by ``later``'s, of the scene recorded again:
        a play found in both as ``later`` found it, unless this one demoted
        it; every divergence of both, each cause once.
        """
        found = {p.play: p for p in self.plays}
        for each in later.plays:
            earlier = found.get(each.play)
            if earlier is None or earlier.outcome != "demoted":
                found[each.play] = each
        divergences: list[Divergence] = []
        for divergence in (*self.divergences, *later.divergences):
            if all(kept.cause != divergence.cause for kept in divergences):
                divergences.append(divergence)
        return Probe(tuple(found[index] for index in sorted(found)), tuple(divergences))


def probe(record: Record) -> Probe:
    """Probe every state-function play of ``record``."""
    if record.refusal is not None:
        return Probe()
    plays: list[PlayProbe] = []
    divergences: list[Divergence] = []
    for play in record.plays:
        if play.route != "state" or play.held:
            continue
        found = _probe_play(record, play)
        plays.append(found)
        if found.outcome == "demoted":
            assert found.reason is not None
            divergences.append(
                Divergence("probe", play.index, play.source_line, found.reason)
            )
    return Probe(tuple(plays), tuple(divergences))


def _probe_play(record: Record, play: PlayRecord) -> PlayProbe:
    inside = _interior(record, play)
    if len(inside) < 2:
        return PlayProbe(play.index, "unprobed")
    instants = (inside[0], inside[-1])
    bases = tuple(_base(play, t) for t in instants)
    starts = tuple(base.instant for base in bases)
    for t, base in zip(instants, bases, strict=True):
        try:
            why = _difference(record, play, t, base)
        except Unrestorable as refused:
            return PlayProbe(
                play.index, "unrestorable", instants, starts, refused.reason
            )
        if why is not None:
            return PlayProbe(
                play.index,
                "demoted",
                instants,
                starts,
                f"restored at {float(t):g} s from {float(base.instant):g} s, {why}",
            )
    return PlayProbe(play.index, "agreed", instants, starts)


def _interior(record: Record, play: PlayRecord) -> list[Fraction]:
    """The instants inside ``play`` its record kept and a frame read: on the
    record's grid (every 60 Hz tick without one), neither its start nor an
    event.
    """
    grid = record.grid if record.grid is not None else Fraction(TICK_RATE)
    events = set(play.events)
    return [
        capture.instant
        for capture in play.samples
        if capture.instant != play.start
        and capture.instant not in events
        and (capture.instant * grid).denominator == 1
    ]


def _base(play: PlayRecord, t: Fraction) -> Capture:
    """The capture of the play's start, or of its last event before ``t``."""
    instant = max((play.start, *(e for e in play.events if e < t)))
    capture = play.sample_at(instant)
    assert capture is not None, f"play {play.index} kept no capture at {instant}"
    return capture


# Comparing a restored scene with a kept capture


def _difference(
    record: Record, play: PlayRecord, t: Fraction, base: Capture
) -> str | None:
    """How ``play`` restored at ``t`` from ``base`` differs from the capture
    its record kept at ``t``, or ``None``.
    """
    kept = play.sample_at(t)
    assert kept is not None
    with restored(record, t, base=base) as scene:
        camera = scene.renderer.camera
        extra = [*play.moving, *camera_nodes(camera)]
        lists = (scene.mobjects, scene.foreground_mobjects, scene.moving_mobjects)
        mine = _Walk(
            _walk([*(m for each in lists for m in each), *extra], NodeState.of)
        )
        theirs = _Walk(
            _walk(
                [*kept.roots, *kept.foreground, *kept.moving, *extra],
                lambda mob: record.node_state(mob, kept),
            )
        )
        why = _tree_difference(
            mine,
            lists,
            theirs,
            (kept.roots, kept.foreground, kept.moving),
            {id(mob): reference.center for mob, reference in kept.references},
        )
        if why is not None:
            return why
        return _camera_difference(
            mine,
            CameraExtras.observe(camera, None),
            theirs,
            kept.camera,
            "the camera's",
        )


def _walk(
    roots: Sequence[Mobject], state_of: Callable[[Mobject], NodeState | None]
) -> Iterator[tuple[Mobject, NodeState]]:
    """The mobjects reachable from ``roots``, depth first, each once, with
    their states by ``state_of`` (as :meth:`.Record._walk` reads a capture).
    """
    seen: set[int] = set()
    stack = list(reversed(roots))
    while stack:
        mob = stack.pop()
        if id(mob) in seen:
            continue
        seen.add(id(mob))
        state = state_of(mob)
        if state is None:
            continue
        yield mob, state
        stack.extend(reversed(state.submobjects))


def _places(walk: Sequence[tuple[Mobject, NodeState]]) -> dict[int, int]:
    """Each walked mobject's place in ``walk``, by its id."""
    return {id(mob): index for index, (mob, _) in enumerate(walk)}


class _Walk(list["tuple[Mobject, NodeState]"]):
    """A walk's nodes in order, with each mobject's place in it indexed once
    for every comparison that reads it.
    """

    def __init__(self, nodes: Iterable[tuple[Mobject, NodeState]]) -> None:
        super().__init__(nodes)
        self.places = _places(self)


def _positions(walk: _Walk, members: Iterable[Mobject]) -> tuple[int, ...]:
    return tuple(walk.places.get(id(mob), -1) for mob in members)


_LISTS = ("scene's mobjects", "foreground", "moving mobjects")


def _tree_difference(
    mine: _Walk,
    my_lists: Sequence[Sequence[Mobject]],
    theirs: _Walk,
    their_lists: Sequence[Sequence[Mobject]],
    centers: Mapping[int, np.ndarray] = {},  # noqa: B006 (read only)
) -> str | None:
    """How the restored draw lists, tree and nodes differ from the kept.
    ``centers`` are the depth reference points the kept capture found, by
    mobject.
    """
    for name, members, others in zip(_LISTS, my_lists, their_lists, strict=True):
        if _positions(mine, members) != _positions(theirs, others):
            return f"the {name} differ from those kept"
    if len(mine) != len(theirs):
        return f"it holds {len(mine)} mobjects, where the record kept {len(theirs)}"
    for (mob, state), (twin, kept) in zip(mine, theirs, strict=True):
        if type(mob) is not type(twin) or len(state.submobjects) != len(
            kept.submobjects
        ):
            return (
                f"a {type(mob).__name__} with {len(state.submobjects)} submobjects is"
                f" a {type(twin).__name__} with {len(kept.submobjects)} in the record"
            )
        # Its edges, not only how many: a child two parents share is walked
        # once, so pointing one of them at another child keeps every count.
        if _positions(mine, state.submobjects) != _positions(theirs, kept.submobjects):
            return (
                f"a {type(mob).__name__}'s submobjects are other mobjects than"
                " in the record"
            )
        why = _fields_difference(
            state, kept, (mine.places, theirs.places), centers.get(id(twin))
        )
        if why is not None:
            return f"its {type(mob).__name__}'s {why}"
    return None


#: Each walked mobject's place in the restored walk and in the kept one, by
#: id: a field's references to mobjects are compared by them.
_Places = tuple[Mapping[int, int], Mapping[int, int]]


def _fields_difference(
    state: NodeState,
    kept: NodeState,
    places: _Places,
    center: np.ndarray | None,
) -> str | None:
    """How the fields of a restored node differ from those of the kept one.
    ``center`` is the depth reference point the kept capture found for the
    node, which a restore's stand-in for its ``z_index_group``
    (:class:`.record.DepthReference`) is compared with.
    """
    mine, theirs = dict(covered_fields(state)), dict(covered_fields(kept))
    if mine.keys() != theirs.keys():
        return f"fields {', '.join(sorted(mine.keys() ^ theirs.keys()))} differ"
    for name in sorted(mine):
        if name == "z_index_group" and isinstance(mine[name], DepthReference):
            # The stand-in is the point the depth sort reads of the group the
            # record kept; the kept capture's own point is its value.
            if center is None or not np.allclose(
                mine[name].center, center, rtol=0, atol=TOLERANCE
            ):
                return "z_index_group's reference point differs from the kept"
            continue
        why = _value_difference(
            mine[name], theirs[name], paint=_paint_name(name), places=places
        )
        if why is not None:
            return f"{name} {why}"
    return None


#: What names a field that holds paint as a plain number or a colour,
#: outside any :class:`~arachne.mobject.values.Paint`: every opacity (a
#: VMobject's beside its paint, an image's, an SVG's ``opacity``, a
#: camera's background), every stroke or line width (a point cloud's, the
#: widths a mobject keeps to restyle with, a camera's line-width multiple),
#: a sheen, a colour and a camera's background. A sweep of the engine's
#: assignments found no style number it misses.
_PAINT_NAME = re.compile(r"opacity|stroke_width|line_width|sheen|colou?r|background")


def _paint_name(name: str) -> bool:
    """Whether the field ``name`` holds paint, compared exactly."""
    return _PAINT_NAME.search(name) is not None


def _paint_arrays() -> tuple[type, ...]:
    """The array kinds that hold paint, inside a :class:`Paint` or outside
    it (a point cloud's colours, an image's pixels, a camera's background).
    """
    from arachne.mobject.values import (
        FrozenBackground,
        FrozenColors,
        FrozenPixels,
        FrozenPointColors,
        FrozenVector,
    )

    return (
        FrozenColors,
        FrozenPointColors,
        FrozenPixels,
        FrozenVector,
        FrozenBackground,
    )


def _value_difference(
    a: Any, b: Any, *, paint: bool = False, places: _Places | None = None
) -> str | None:
    """How a restored value differs from the kept one: paint exactly (a
    :class:`Paint` by its key, an array of a paint kind, a number of a field
    that holds paint, ``paint``), geometry within
    :data:`.execution.TOLERANCE`, the rest by content key
    (:func:`.keys.value_key`), its numbers to :data:`.execution.KEY_DECIMALS`
    unless it is paint, and the mobjects it refers to by their ``places`` in
    the two walks (by identity without them). A value of a type the key does
    not know always differs.
    """
    from arachne.mobject.placement import Placement
    from arachne.mobject.values import Paint
    from arachne.utils.color import ManimColor

    if isinstance(a, _paint_arrays()) or isinstance(b, _paint_arrays()):
        paint = True
    if isinstance(a, ManimColor) or isinstance(b, ManimColor):
        # A colour by its numbers, exactly.
        paint = True
        a = np.asarray(a.to_rgba(), dtype=float) if isinstance(a, ManimColor) else a
        b = np.asarray(b.to_rgba(), dtype=float) if isinstance(b, ManimColor) else b
    if isinstance(a, Paint) or isinstance(b, Paint):
        if isinstance(a, Paint) and isinstance(b, Paint):
            if a.key == b.key:
                return None
            for part in Paint.FIELDS:
                if not _same(vars(a).get(part), vars(b).get(part)):
                    return f"{part} differs from the kept paint"
        return "differs from the kept paint"
    if isinstance(a, Placement):
        a = a.points
    if isinstance(b, Placement):
        b = b.points
    if (
        isinstance(a, np.ndarray)
        and isinstance(b, np.ndarray)
        and not (a.dtype.hasobject or b.dtype.hasobject)
    ):
        if a.dtype != b.dtype or a.shape != b.shape:
            return f"is {a.dtype}{list(a.shape)}, kept {b.dtype}{list(b.shape)}"
        if paint or a.dtype.kind not in "fc":
            if np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
                return None
            return "differs from the kept, where it is paint, compared exactly"
        if np.allclose(a, b, rtol=0, atol=TOLERANCE, equal_nan=True):
            return None
        return f"differs from the kept by up to {np.nanmax(np.abs(a - b)):.3g}"
    if isinstance(a, float) and isinstance(b, float) and not paint:
        if abs(a - b) <= TOLERANCE or (a != a and b != b):
            return None
        return f"differs from the kept by {abs(a - b):.3g}"
    if type(a) is type(b):
        decimals = None if paint else KEY_DECIMALS
        (key, refers), (kept, kept_refers) = (
            value_key(a, decimals),
            value_key(b, decimals),
        )
        if key == kept and _referred(refers, places, 0) == _referred(
            kept_refers, places, 1
        ):
            return None
        reason = marker_reason(key)
        if reason is not None:
            return f"holds {reason}, which agrees with nothing kept"
    return f"is {reprlib.repr(a)}, kept {reprlib.repr(b)}"


def _referred(
    mobs: tuple[Mobject, ...], places: _Places | None, side: int
) -> tuple[int, ...]:
    """Which mobjects ``mobs`` are: their places in the walk of ``side``,
    ``-1`` for one outside it; their identities without ``places``.
    """
    if places is None:
        return tuple(map(id, mobs))
    return tuple(places[side].get(id(mob), -1) for mob in mobs)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and bool(np.array_equal(a, b))
        )
    return type(a) is type(b) and bool(a == b)


def _camera_difference(
    mine: _Walk,
    extras: CameraExtras | None,
    theirs: _Walk,
    kept: CameraExtras | None,
    which: str,
) -> str | None:
    """How a camera's restored inputs differ from the kept ones: its values
    (paint exactly, the rest as :func:`_value_difference` compares them),
    its background exactly, its fixed-orientation centres, and the same of
    every camera drawing into one of its displays.
    """
    if extras is None or kept is None:
        if extras is kept:
            return None
        return f"{which} inputs differ from those kept"
    names = [name for name, _ in extras.values]
    if names != [name for name, _ in kept.values]:
        return f"{which} values differ from those kept"
    for (name, value), (_, twin) in zip(extras.values, kept.values, strict=True):
        why = _value_difference(value, twin, paint=_paint_name(name))
        if why is not None:
            return f"{which} {name} {why}"
    background, twin_background = extras.background, kept.background
    if (background is None) != (twin_background is None) or (
        background is not None and not _same(background, twin_background)
    ):
        return f"{which} background differs from the kept one"
    why = _centre_difference(mine, extras.centers, theirs, kept.centers)
    if why is not None:
        return f"{which} {why}"
    if len(extras.sub_cameras) != len(kept.sub_cameras):
        return f"{which} displays draw other cameras than those kept"
    for number, ((_, sub), (_, twin_sub)) in enumerate(
        zip(extras.sub_cameras, kept.sub_cameras, strict=True)
    ):
        why = _camera_difference(
            mine, sub, theirs, twin_sub, f"{which} display camera {number}'s"
        )
        if why is not None:
            return why
    return None


def _centre_difference(
    mine: _Walk,
    centers: Sequence[tuple[Mobject, np.ndarray]],
    theirs: _Walk,
    kept: Sequence[tuple[Mobject, np.ndarray]],
) -> str | None:
    """How the camera's restored fixed-orientation centres differ from the
    kept ones, by their mobjects' places in the walk.
    """
    if _positions(mine, (m for m, _ in centers)) != _positions(
        theirs, (m for m, _ in kept)
    ):
        return "fixed-orientation mobjects differ from those kept"
    for (_, center), (_, twin) in zip(centers, kept, strict=True):
        why = _value_difference(
            np.asarray(center, dtype=float), np.asarray(twin, dtype=float)
        )
        if why is not None:
            return f"a fixed-orientation centre {why}"
    return None
