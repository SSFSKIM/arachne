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

- the draw lists, the tree and every node's type, exactly;
- every field a content key covers (:func:`.record.content_fields`): a
  paint by its exact key, geometry (points, placements, numbers) within
  :data:`.execution.TOLERANCE`, anything else exactly;
- the camera's fixed-orientation centres, within the tolerance.

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

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .clock import TICK_RATE
from .draw import CameraExtras, camera_nodes
from .execution import TOLERANCE, Divergence
from .record import NodeState, _value_key, content_fields
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

    @property
    def demoted(self) -> tuple[int, ...]:
        """The plays demoted, by index."""
        return tuple(p.play for p in self.plays if p.outcome == "demoted")

    @property
    def unprobed(self) -> int:
        """How many plays had fewer than two instants to restore."""
        return sum(p.outcome == "unprobed" for p in self.plays)

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
        mine = list(_walk([*(m for each in lists for m in each), *extra], NodeState.of))
        theirs = list(
            _walk(
                [*kept.roots, *kept.foreground, *kept.moving, *extra],
                lambda mob: record.node_state(mob, kept),
            )
        )
        why = _tree_difference(
            mine, lists, theirs, (kept.roots, kept.foreground, kept.moving)
        )
        if why is not None:
            return why
        extras = CameraExtras.observe(camera, None)
        centers = () if extras is None else extras.centers
        return _centre_difference(
            mine, centers, theirs, kept.camera.centers if kept.camera else ()
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


def _positions(
    walk: Sequence[tuple[Mobject, NodeState]], members: Iterable[Mobject]
) -> tuple[int, ...]:
    where = {id(mob): index for index, (mob, _) in enumerate(walk)}
    return tuple(where.get(id(mob), -1) for mob in members)


_LISTS = ("scene's mobjects", "foreground", "moving mobjects")


def _tree_difference(
    mine: Sequence[tuple[Mobject, NodeState]],
    my_lists: Sequence[Sequence[Mobject]],
    theirs: Sequence[tuple[Mobject, NodeState]],
    their_lists: Sequence[Sequence[Mobject]],
) -> str | None:
    """How the restored draw lists, tree and nodes differ from the kept."""
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
        why = _fields_difference(state, kept)
        if why is not None:
            return f"its {type(mob).__name__}'s {why}"
    return None


def _fields_difference(state: NodeState, kept: NodeState) -> str | None:
    mine, theirs = _keyed(state), _keyed(kept)
    if mine.keys() != theirs.keys():
        return f"fields {', '.join(sorted(mine.keys() ^ theirs.keys()))} differ"
    for name in sorted(mine):
        why = _value_difference(mine[name], theirs[name])
        if why is not None:
            return f"{name} {why}"
    return None


def _keyed(state: NodeState) -> dict[str, Any]:
    """The fields of ``state`` a content key covers, with their values."""
    names = {name for name, _ in content_fields(state)}
    return {
        name: value
        for name, value in zip(state.keys, state.values, strict=True)
        if name in names
    }


def _value_difference(a: Any, b: Any) -> str | None:
    """How a restored value differs from the kept one: a paint by its exact
    key, geometry within :data:`.execution.TOLERANCE`, the rest exactly.
    """
    from arachne.mobject.placement import Placement
    from arachne.mobject.values import Paint

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
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return f"is {a.dtype}{list(a.shape)}, kept {b.dtype}{list(b.shape)}"
        if a.dtype.kind not in "fc":
            return None if np.array_equal(a, b) else "differs from the kept"
        if np.allclose(a, b, rtol=0, atol=TOLERANCE, equal_nan=True):
            return None
        return f"differs from the kept by up to {np.nanmax(np.abs(a - b)):.3g}"
    if isinstance(a, float) and isinstance(b, float):
        if abs(a - b) <= TOLERANCE or (a != a and b != b):
            return None
        return f"differs from the kept by {abs(a - b):.3g}"
    if type(a) is type(b) and _value_key(a) == _value_key(b):
        return None
    return f"is {a!r}, kept {b!r}"


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


def _centre_difference(
    mine: Sequence[tuple[Mobject, NodeState]],
    centers: Sequence[tuple[Mobject, np.ndarray]],
    theirs: Sequence[tuple[Mobject, NodeState]],
    kept: Sequence[tuple[Mobject, np.ndarray]],
) -> str | None:
    """How the camera's restored fixed-orientation centres differ from the
    kept ones, by their mobjects' places in the walk.
    """
    if _positions(mine, (m for m, _ in centers)) != _positions(
        theirs, (m for m, _ in kept)
    ):
        return "the camera's fixed-orientation mobjects differ from those kept"
    for (_, center), (_, twin) in zip(centers, kept, strict=True):
        why = _value_difference(
            np.asarray(center, dtype=float), np.asarray(twin, dtype=float)
        )
        if why is not None:
            return f"a fixed-orientation centre {why}"
    return None
