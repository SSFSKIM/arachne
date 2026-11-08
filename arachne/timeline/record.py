"""The forward record of one scene execution.

A :class:`Record` is what one run of a scene script leaves: per play
(:class:`PlayRecord`) its exact span, kind, source line, route, held flag and
markers, the state the scene was in when it began, and its samples. Every
answer about a time t is read from it.

States are kept as references to the mobjects' immutable values. Each
mobject has a history of :class:`NodeState` versions, one per capture at
which anything in its ``__dict__`` (or its list of submobjects or updaters)
changed by identity; a capture of an unchanged mobject costs a comparison
and stores nothing.
"""

from __future__ import annotations

__all__ = [
    "Capture",
    "DepthReference",
    "Identity",
    "Marker",
    "NodeState",
    "PlayRecord",
    "Record",
    "Removal",
    "Sample",
    "States",
]

import bisect
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from operator import is_
from typing import TYPE_CHECKING, Any

import numpy as np

from .clock import exact, frames_in
from .dispatch import draw_member_refusal, draw_names
from .execution import Execution
from .origins import Origin, origin_of
from .source import SourceLine

if TYPE_CHECKING:
    from arachne.animation.animation import Animation
    from arachne.frame_ir.extract import Term
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene

    from .admission import Admission
    from .bodies import Exit
    from .camera_code import Replaced
    from .draw import CameraExtras
    from .recorder import Bridge
    from .state import State
    from .ticking import TickPlan

#: How many decimals :meth:`PlayRecord.sample_keys` compares numbers to: a
#: callback stepping skips computes the same values along another float path
#: (``next_to`` moves a mobject by where it was), ulps apart.
SAMPLE_DECIMALS = 9


class NodeState:
    """One version of a mobject's state: its ``__dict__``'s keys and values,
    and the contents of its submobject and updater lists, by reference.
    """

    __slots__ = ("keys", "values", "submobjects", "updaters")

    keys: tuple[str, ...]
    values: tuple[Any, ...]
    submobjects: tuple[Mobject, ...]
    updaters: tuple[Any, ...]

    @classmethod
    def of(cls, mob: Mobject, previous: NodeState | None = None) -> NodeState:
        """``mob``'s state now: ``previous`` itself when nothing changed."""
        own = mob.__dict__
        if previous is not None and previous.matches(own):
            return previous
        state = cls.__new__(cls)
        keys = tuple(own)
        state.keys = (
            previous.keys if previous is not None and previous.keys == keys else keys
        )
        state.values = tuple(own.values())
        state.submobjects = tuple(own.get("submobjects", ()))
        state.updaters = tuple(own.get("updaters", ()))
        return state

    def matches(self, own: dict[str, Any]) -> bool:
        """Whether ``own`` (a mobject's ``__dict__``) holds exactly this state."""
        keys = self.keys
        if len(own) != len(keys):
            return False
        if not all(map(is_, own, keys)) or not all(map(is_, own.values(), self.values)):
            return False
        for name, held in (
            ("submobjects", self.submobjects),
            ("updaters", self.updaters),
        ):
            live = own.get(name)
            if live is not None and (
                len(live) != len(held) or not all(map(is_, live, held))
            ):
                return False
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """The value this state holds for the attribute ``key``."""
        try:
            return self.values[self.keys.index(key)]
        except ValueError:
            return default

    def install(self, mob: Mobject) -> None:
        """Make ``mob`` hold this state."""
        own = mob.__dict__
        own.clear()
        own.update(zip(self.keys, self.values, strict=True))
        submobjects = own.get("submobjects")
        if submobjects is not None:
            submobjects[:] = self.submobjects
        updaters = own.get("updaters")
        if updaters is not None:
            updaters[:] = self.updaters


class _Node:
    """A mobject the record has seen, its origin and its versions."""

    __slots__ = ("mob", "origin", "seqs", "states")

    def __init__(self, mob: Mobject) -> None:
        self.mob = mob
        self.origin: Origin = origin_of(mob)
        self.seqs: list[int | Fraction] = []
        self.states: list[NodeState] = []

    def at(self, seq: int | Fraction) -> NodeState | None:
        index = bisect.bisect_right(self.seqs, seq) - 1
        return self.states[index] if index >= 0 else None


@dataclass(frozen=True)
class Capture:
    """One captured instant: what the scene held there.

    ``kind`` is ``"static"`` (a play's state when its static picture is
    drawn, right after its animations began), ``"sample"`` (an evaluated
    instant of a play) or ``"final"`` (the scene after its script ended).
    ``until`` is the next instant the play evaluated, so the capture
    answers every time in ``[instant, until)``. Those are drawn, and make
    the record's history of captures, by ``seq``.

    The states a record keeps for the lint (``Record.states``) are captures
    too, never drawn and outside that history: ``"call"`` (the scene as the
    script called a play), ``"exit"`` (as a play ended) and ``"removal"``
    (as the script removed from it). Each has a ``seq`` between those of
    the drawn captures either side, so that a mobject's state at it is read
    as at any capture, and no drawn capture reads anything of it.

    ``moving`` is the scene's list of moving mobjects there, which a frame
    of the play draws over its static picture: flattened when the play
    began, and grown by whatever the scene added since (CE's ``Scene.add``
    appends to it during a play).

    ``references`` are the depth sort's reference points there: a mobject
    with a ``z_index_group`` is sorted by that group's centre, which is
    another mobject's state (one the scene may not even hold), so the point
    itself is kept, as the sort reads it, and a drawing gives it back.

    ``pictures`` are what the displays of other cameras show: an
    ``ImageMobjectFromCamera`` reads its camera's framebuffer as it is drawn,
    and a camera the scene's camera does not draw into (one of the scene's
    own, or a MultiCamera's display before it joins) holds no recorded
    state. Its picture is kept once a play, shared with the play before
    while it is the same; a picture that changes during a play is not kept
    (:data:`DISPLAY_REDRAWN`), which keeps one copy a play at most.
    """

    seq: int | Fraction
    instant: Fraction
    kind: str
    roots: tuple[Mobject, ...]
    foreground: tuple[Mobject, ...]
    moving: tuple[Mobject, ...]
    scene_updaters: tuple[Any, ...]
    camera: CameraExtras | None
    until: Fraction | None = None
    #: Each mobject a ThreeDCamera sorts by a point other than its own centre
    #: (its ``z_index_group``), with that point here (:class:`DepthReference`).
    references: tuple[tuple[Mobject, DepthReference], ...] = ()
    #: Each display on screen of a camera the scene's camera does not draw
    #: into, with that camera's picture as the play found it, read-only.
    pictures: tuple[tuple[Mobject, np.ndarray], ...] = ()

    @property
    def drawn_seq(self) -> int:
        """The place of a capture a frame draws in the record's history of
        drawn captures (a state no frame draws has none).
        """
        assert isinstance(self.seq, int), "a capture no frame draws"
        return self.seq

    def draw_list(self) -> list[Mobject]:
        """The scene's mobjects with the foreground last, as CE draws a frame
        when no list is given (``list_update(mobjects, foreground_mobjects)``).
        """
        foreground = set(map(id, self.foreground))
        return [m for m in self.roots if id(m) not in foreground] + list(
            self.foreground
        )


class DepthReference:
    """The point a ThreeDCamera's depth sort placed a mobject by at a
    capture, standing in for its ``z_index_group`` while a frame is drawn:
    what ``Mobject.get_z_index_reference_point`` reads of the group.
    """

    __slots__ = ("center",)

    def __init__(self, center: np.ndarray) -> None:
        self.center = center

    def get_center(self) -> np.ndarray:
        return self.center


#: Why a record can't draw a play during which a display of another camera
#: showed a new picture: keeping one a capture costs a framebuffer each.
DISPLAY_REDRAWN = (
    "camera.display_redrawn: a display shows a camera the scene's camera does not"
    " draw into, and that camera's picture changed during the play, so its frames"
    " are drawn while the scene runs and its record answers no instant"
)


#: Why a record can't draw a play from which on the configured pixel shape
#: is not the one its first capture found: a point's thickness is drawn by it
#: (``Camera.adjusted_thickness``), and a frame drawn after the play reads it
#: as it is then.
PIXEL_SHAPE_CHANGED = (
    "config.pixel_shape: the scene changed the configured pixel shape"
    " (config.pixel_width, config.pixel_height), which a point is drawn as thick"
    " as, so its frames from this play on are drawn while the scene runs and its"
    " record answers no instant there"
)

#: Why a record can't draw a play during which a camera's own mobject was
#: rebound: its frames would each need the binding of their own instant,
#: which an instant between captures does not have.
CAMERA_REBOUND = (
    "camera.rebound: a camera's own mobject (its light, frame centre, a tracker"
    " or its frame) was rebound to another during the play, so its frames are"
    " drawn while the scene runs and its record answers no instant"
)


def _pictures(
    displays: Sequence[Mobject],
    previous: tuple[tuple[Mobject, np.ndarray], ...],
    kept: dict[int, tuple[Mobject, np.ndarray, int]],
    play: int,
) -> tuple[tuple[tuple[Mobject, np.ndarray], ...], bool]:
    """The picture each of ``displays`` shows now, its camera's framebuffer,
    and whether one changed during ``play`` since the play first showed it.
    ``kept`` holds, per display (by id), the last picture kept and the play
    it was last shown in, whether or not it was on screen at every capture
    between: a picture is shared while it is the same, copied anew in a new
    play, and not copied for a change within the play (that play's is kept).
    ``previous`` itself, or its pictures, where they are the same.
    """
    found: list[tuple[Mobject, np.ndarray]] = []
    redrawn = False
    for mob in displays:
        live = mob.camera.pixel_array  # type: ignore[attr-defined]
        entry = kept.get(id(mob))
        picture = None
        if entry is not None and entry[0] is mob:
            held, shown_in = entry[1], entry[2]
            if held.shape == live.shape and np.array_equal(held, live):
                picture = held
            elif shown_in == play:
                redrawn = True
                picture = held
        if picture is None:
            picture = live.copy()
            picture.flags.writeable = False
        kept[id(mob)] = (mob, picture, play)
        found.append((mob, picture))
    if len(found) == len(previous) and all(
        a is c and b is d for (a, b), (c, d) in zip(found, previous, strict=True)
    ):
        return previous, redrawn
    return tuple(found), redrawn


def _references(
    mobs: Sequence[Mobject],
    previous: tuple[tuple[Mobject, DepthReference], ...],
) -> tuple[tuple[Mobject, DepthReference], ...]:
    """The depth reference point of each of ``mobs`` now; ``previous``
    itself, or its points, where they are the same.
    """
    earlier = {id(mob): reference for mob, reference in previous}
    found: list[tuple[Mobject, DepthReference]] = []
    for mob in mobs:
        center = np.array(mob.get_z_index_reference_point(), dtype=float)
        reference = earlier.get(id(mob))
        if reference is None or not np.array_equal(reference.center, center):
            center.flags.writeable = False
            reference = DepthReference(center)
        found.append((mob, reference))
    if len(found) == len(previous) and all(
        a is c and b is d for (a, b), (c, d) in zip(found, previous, strict=True)
    ):
        return previous
    return tuple(found)


@dataclass(frozen=True)
class Marker:
    """Something the script placed on the timeline at an instant.

    ``kind`` is one of ``"section"``, ``"subcaption"``, ``"sound"``,
    ``"mark"`` (:meth:`~.Scene.mark`) and ``"wait_until"`` (a wait whose stop
    condition ended it before its requested duration).
    """

    kind: str
    t: Fraction
    data: Mapping[str, Any]
    source_line: SourceLine | None = None


@dataclass(frozen=True)
class Removal:
    """A ``remove`` or ``clear`` the script called between two plays (a
    play's own cleanup is not one): when, what it named (``operands``; the
    whole scene for a clear), the scene just before it (``capture``), the
    scene's top level just after it (``after``), its call site, and how
    many plays had been recorded before it.
    """

    t: Fraction
    kind: str
    operands: tuple[Mobject, ...]
    capture: Capture
    after: tuple[Mobject, ...]
    source_line: SourceLine | None
    plays: int


@dataclass
class States:
    """The states a record keeps for the lint's temporal rules (``Record.states``):
    the scene as the script called each play, before the play added or began
    anything (``calls``, by play index), as each play ended, its animations
    finished and cleaned up and the script not yet run on (``exits``), and
    as the script removed from it (:class:`Removal`); from the scene time
    ``armed`` on, the first play kept being ``since``. None is drawn, and
    none is in the record's history of drawn captures.
    """

    since: int
    armed: Fraction
    calls: dict[int, Capture] = field(default_factory=dict)
    exits: dict[int, Capture] = field(default_factory=dict)
    removals: list[Removal] = field(default_factory=list)

    def covers(self, plays: Sequence[PlayRecord], since: Fraction) -> bool:
        """Whether every state of the scene from the time ``since`` on is
        kept: it was armed by then, every play starting then or later has its
        call and its exit, and a play running then has its exit.
        """
        return self.armed <= since and all(
            (play.start < since or play.index in self.calls)
            and play.index in self.exits
            for play in plays
            if play.end > since or play.start >= since
        )


@dataclass(frozen=True)
class Identity:
    """What a record carries about the execution that made it: the digest of
    its source, the digests of the files it read, and its seed.
    """

    source_digest: str | None
    input_digests: tuple[tuple[str, str], ...] = ()
    seed: int | None = None


@dataclass
class PlayRecord:
    """One ``play`` or ``wait`` of the scene, as the forward run made it."""

    index: int
    start: Fraction
    end: Fraction
    kind: str
    route: str
    source_line: SourceLine | None
    held: bool
    #: Whether its frames were left out of the render (a skipped section,
    #: ``-n`` or a still output); it was executed all the same.
    skipped: bool
    #: Where it would have ended had no stop condition ended it first.
    requested_end: Fraction
    #: The animations' names, in the order they were played.
    animations: tuple[str, ...]
    markers: list[Marker] = field(default_factory=list)
    # What drawing it needs: the draw lists of CE 0.22.1's split (#4936),
    # flattened when it began (a frame draws its capture's moving list,
    # which can have grown since).
    moving: tuple[Mobject, ...] = ()
    static: tuple[Mobject, ...] = ()
    static_seq: int = -1
    #: The captures of its evaluated instants, by instant.
    samples: list[Capture] = field(default_factory=list)
    _instants: list[Fraction] = field(default_factory=list, repr=False)
    #: The names the scene's code had for its mobjects when it began, by id.
    names: dict[int, str] = field(default_factory=dict, repr=False)
    #: ``random``'s and ``numpy.random``'s state when it began.
    random_state: tuple[Any, Any] | None = field(default=None, repr=False)
    #: The partial movie key its frames are cached under, if any.
    cache_key: str | None = None
    #: Why its frames can't be drawn from the record though they were to be:
    #: a mobject joined while it ran whose drawing runs the scene's code.
    draw_refusal: str | None = None
    #: Why it took its route, and its callbacks with their proofs.
    admission: Admission | None = field(default=None, repr=False)
    #: A ticked play's stepping plan (:mod:`.ticking`): what its stepping
    #: ticks leave out, or why it ticks everything. ``None`` when every tick
    #: is full anyway (no grid, a frozen frame, another route).
    tick_plan: TickPlan | None = field(default=None, repr=False)
    #: Every plan the play stepped by, in order, each with the first tick it
    #: governs: its first, then one for each plan made again (an event tick,
    #: a full tick whose skipped callbacks changed the updaters) or for the
    #: fall back to ticking everything (a plan whose ``unbounded`` is set).
    #: :attr:`tick_plan` is the last.
    tick_plans: tuple[tuple[Fraction, TickPlan], ...] = field(default=(), repr=False)
    #: How many of its ticks were full and how many only stepped.
    full_ticks: int = 0
    stepping_ticks: int = 0
    #: How many ``become`` calls the process had made as the scene called it
    #: and once it stood as it began (:func:`.rebuilds.count`): a group
    #: rebuilt between the one of a play and the other of the next was
    #: rebuilt across their cut.
    rebuilds_called: int = 0
    rebuilds_entered: int = 0
    #: The instants of its composition events (a Succession part ending and
    #: the next beginning), exact or found by bisection, in order.
    events: tuple[Fraction, ...] = ()
    #: The frame hooks the scene's code held when it began (``update_to_time``,
    #: ``update_mobjects``, ``update_meshes``, ``update_self``: an override, a
    #: patch of the engine's ``Scene`` or a wrapper on the scene object), which
    #: run at its frames only (:mod:`.hooks`), and whether ``play_internal`` was.
    frame_hooks: tuple[str, ...] = ()
    play_internal_overridden: bool = False
    _record: Record | None = field(default=None, repr=False)
    #: The animations it played, as objects.
    _animations: tuple[Animation, ...] = field(default=(), repr=False)
    #: What every animation of it held, by capture seq, where it can be
    #: evaluated from: each capture of a trusted play; the start and event
    #: captures of a state-function play.
    _animation_states: dict[int, Any] = field(default_factory=dict, repr=False)

    @property
    def duration(self) -> Fraction:
        return self.end - self.start

    @property
    def start_state(self) -> RegisteredState:
        """The registered state when the play began: the scene's mobject tree
        after its animations began and before its first evaluation.
        """
        assert self._record is not None
        return RegisteredState(self._record, self._record._captures[self.static_seq])

    def set_tick_plan(self, plan: TickPlan, first_tick: Fraction) -> None:
        """Step by ``plan`` from the tick ``first_tick`` on."""
        self.tick_plan = plan
        self.tick_plans = (*self.tick_plans, (first_tick, plan))

    def sample_keys(
        self, decimals: int = SAMPLE_DECIMALS
    ) -> tuple[tuple[Fraction, bytes], ...]:
        """Each kept capture's instant and content key, its numbers rounded
        to ``decimals`` (:meth:`RegisteredState.content_key`): equal between
        a stepping run and one that ticks everything
        (``Recorder(stepping=False)``) when stepping changed nothing a frame
        reads. A callback that keeps history differs by design: stepping
        runs it at the frames, as CE does.
        """
        assert self._record is not None
        return tuple(
            (
                capture.instant,
                RegisteredState(self._record, capture).content_key(decimals),
            )
            for capture in self.samples
        )

    def leaf_terms(self, t: Fraction) -> dict[Mobject, Term]:
        """The leaves whose state at ``t`` a term of their animation states,
        for a trusted play: of the transform family on a straight path, each
        leaf the animation interpolates, with its start and end mobjects and
        its own alpha at ``t`` (the play's alpha through the animation's rate
        function and lag), a blend; of a partial reveal (``Create``), each
        leaf with its start and its window at that alpha, a reveal. Empty for
        another play, and at an instant whose animations the record did not
        keep. A term is what the forward run computed there only if the leaf
        holds it; a drawer checks that.
        """
        from arachne.animation.creation import ShowPartial
        from arachne.animation.transform import Transform
        from arachne.frame_ir.extract import Blend, Reveal
        from arachne.utils.bezier import interpolate

        from .frozen import engine_class
        from .routes import _alpha

        if self.route != "trusted" or self.held:
            return {}
        base = self.base_at(t)
        states = {
            id(animation): state
            for animation, state in self._animation_states.get(base.drawn_seq, ())
        }
        found: dict[Mobject, Term] = {}
        for animation in self._animations:
            state = states.get(id(animation))
            if state is None or not engine_class(type(animation)):
                continue
            blends = (
                isinstance(animation, Transform)
                and state.get("_path_func") is interpolate
            )
            if not blends and not isinstance(animation, ShowPartial):
                continue
            saved = dict(vars(animation))
            try:
                vars(animation).clear()
                vars(animation).update(state)
                alpha = _alpha(t - self.start, animation.run_time)
                families = list(animation.get_all_families_zipped())
                for index, (leaf, start, *end) in enumerate(families):
                    sub = animation.get_sub_alpha(alpha, index, len(families))
                    if blends:
                        found[leaf] = Blend(start, end[0], sub)
                    else:
                        lo, hi = animation._get_bounds(sub)  # type: ignore[attr-defined]
                        found[leaf] = Reveal(start, lo, hi)
            finally:
                vars(animation).clear()
                vars(animation).update(saved)
        return found

    def base_at(self, t: Fraction) -> Capture:
        """The capture of the last instant the play was evaluated at, at or
        before ``t``.
        """
        index = bisect.bisect_right(self._instants, t) - 1
        if index < 0:
            raise LookupError(f"{t} is before play {self.index} began.")
        return self.samples[index]

    def sample_at(self, t: Fraction) -> Capture | None:
        """The capture of the instant ``t`` itself, if the play was evaluated
        there and the record kept it.
        """
        index = bisect.bisect_left(self._instants, t)
        if index < len(self._instants) and self._instants[index] == t:
            return self.samples[index]
        return None

    def capture_at(self, t: Fraction) -> Capture:
        """The capture that answers ``t`` as a ticked play's: the last
        evaluated instant at or before it. (Other routes answer through
        :mod:`.routes`.)
        """
        index = bisect.bisect_right(self._instants, t) - 1
        if index < 0:
            raise LookupError(f"{t} is before play {self.index} began.")
        capture = self.samples[index]
        if capture.until is not None and t >= capture.until:
            raise LookupError(
                f"The record did not keep play {self.index}'s state at {t}: it kept "
                "only the instants its frame grid reads."
            )
        return capture


class RegisteredState:
    """A play's start state: the mobject tree and every node's values, as
    references, with a content key computed on demand.
    """

    def __init__(self, record: Record, capture: Capture) -> None:
        self._record = record
        self.capture = capture

    def nodes(self) -> Iterator[tuple[Mobject, NodeState]]:
        """Every node reachable from the scene, with its state, depth first."""
        yield from self._record._walk(self.capture, self.capture.roots)

    def content_key(self, decimals: int | None = None) -> bytes:
        """A digest of the tree's shape and its nodes' values, equal across
        two executions when what they hold is the same; with ``decimals``,
        when their numbers agree to that many decimals.

        The walk visits a node held twice once, so the shape is keyed as
        which node each root and each parent's child is, by its place in
        the walk: two trees of the same nodes in which a parent holds
        another of them key apart.
        """
        nodes = list(self.nodes())
        place = {id(mob): index for index, (mob, _) in enumerate(nodes)}
        digest = hashlib.blake2b(digest_size=16)

        def held(mobs: Sequence[Mobject]) -> None:
            digest.update(len(mobs).to_bytes(4, "big"))
            for mob in mobs:
                # A mobject with no state at this capture is no node: 0.
                digest.update((place.get(id(mob), -1) + 1).to_bytes(4, "big"))

        held(self.capture.roots)
        for mob, state in nodes:
            digest.update(_framed(type(mob).__qualname__.encode()))
            held(state.submobjects)
            for key, part in content_fields(state, decimals):
                digest.update(_framed(key.encode()))
                digest.update(_framed(part))
        return digest.digest()


def _framed(data: bytes) -> bytes:
    """``data`` with its length before it, so that two runs of parts key
    apart wherever they split (``"AB"`` then ``"C"`` against ``"A"`` then
    ``"BC"``).
    """
    return len(data).to_bytes(4, "big") + data


#: The fields of a mobject that name an object of this process, not content:
#: ``original_id`` is ``str(id(...))`` of what a copy was made from (CE's
#: caching filters it for the same reason).
IDENTITY_FIELDS = frozenset({"original_id"})


def content_fields(
    state: NodeState, decimals: int | None = None
) -> Iterator[tuple[str, bytes]]:
    """Each field of ``state`` a content key covers, by name in sorted order,
    with its value's key (:func:`_value_key`): none naming an object of this
    process (:data:`IDENTITY_FIELDS`), so that two executions in two
    processes key the same content alike.
    """
    for key, value in sorted(zip(state.keys, state.values, strict=True)):
        if key in IDENTITY_FIELDS:
            continue
        part = _value_key(value, decimals)
        if part is not None:
            yield key, part


def _value_key(value: Any, decimals: int | None = None) -> bytes | None:
    """A content key for a value a content key covers, else ``None``; its
    floats rounded to ``decimals`` when given.
    """
    from arachne.mobject.placement import Placement
    from arachne.mobject.values import FrozenArray, Paint

    if isinstance(value, (Paint, Placement)):
        return value.key
    if isinstance(value, FrozenArray):
        if decimals is not None and value.dtype.kind in "fc":
            # Adding zero turns -0.0, which rounding makes, into 0.0.
            value = np.round(value, decimals) + 0.0
        return hashlib.blake2b(value.tobytes(), digest_size=16).digest()
    if isinstance(value, float) and decimals is not None:
        return repr(round(value, decimals) + 0.0).encode()
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value).encode()
    return None


@dataclass(frozen=True)
class Sample:
    """An instant of a frame grid, and the play and capture that answer it."""

    t: Fraction
    play: int | None
    seq: int


class Record:
    """What one forward execution of a scene produced."""

    #: Where each body stood as each play ended (:class:`.bodies.Exit`, by
    #: play index), given by the recorder that makes the record; ``None`` on
    #: a record made without them.
    exits: dict[int, Exit] | None = None
    #: Where each body stood as each play began (:class:`.bodies.Exit`, by
    #: play index): evaluated at its exact start, its parts begun and its
    #: updaters handed ``dt == 0``, as CE's first frame of a play shows it.
    #: Kept beside :attr:`exits`, for every play that starts while they are.
    entries: dict[int, Exit] | None = None
    #: The first play to end once the scene armed the motion rule
    #: (:meth:`.Recorder.arm_motion`), by index; ``None`` if it never did.
    armed_at: int | None = None
    #: The scene as each play ended and as the script removed from it
    #: (:class:`States`), given by a recorder keeping them; ``None`` on a
    #: record made without them.
    states: States | None = None
    #: The assertion bridge's calls, per play (:class:`.recorder.Bridge`),
    #: given by the recorder that makes the record.
    bridge: Bridge

    def __init__(
        self,
        scene: Scene,
        *,
        grid: Fraction | None,
        scene_file: str | None,
        identity: Identity,
    ) -> None:
        self._scene = scene
        #: The frame rate whose instants the record keeps, or ``None`` when it
        #: keeps every evaluated instant.
        self.grid = grid
        self.scene_file = scene_file
        self.scene_class = type(scene).__name__
        self.identity = identity
        self.plays: list[PlayRecord] = []
        self.markers: list[Marker] = []
        self.duration = Fraction(0)
        #: Why no instant can be drawn or answered from this record, if so:
        #: its camera draws through inputs the record does not hold, or a
        #: mobject runs the scene's code as it is drawn, and its frames were
        #: drawn while the scene ran.
        self.refusal: str | None = None
        #: The camera methods the scene's code replaced while it ran, each
        #: once, in the order found (:func:`.camera_code.replaced_methods`, on
        #: every camera a frame is drawn with). The record still draws its
        #: frames; what measures the picture without the camera reads them.
        self.camera_replaced: tuple[Replaced, ...] = ()
        #: Which execution of the scene made it.
        self.execution = Execution()
        #: The first mobject found as it entered a capture since
        #: :meth:`entry_refusal` was last asked whose drawing runs the
        #: scene's code, and the mobjects checked since (by id).
        self._entry_refusal: str | None = None
        self._entry_checked: set[int] = set()
        #: The configured pixel shape the first drawn capture found, and the
        #: seq of the first drawn capture that found another (``None`` while
        #: none has).
        self._pixel_shape: tuple[int, int] | None = None
        #: Per display of another camera (by id): its last picture kept, and
        #: the play that last showed it (:func:`_pictures`).
        self._kept_pictures: dict[int, tuple[Mobject, np.ndarray, int]] = {}
        self.pixel_shape_changed: int | None = None
        self.final: Capture | None = None
        #: The names the scene's code had for its mobjects when it ended.
        self.final_names: dict[int, str] = {}
        self._nodes: dict[int, _Node] = {}
        self._captures: list[Capture] = []
        #: Undrawn captures since the last drawn one (:meth:`_capture`).
        self._undrawn = 0
        #: The seq of the last capture taken, drawn or not; -1 before any.
        #: What the script does after it and before the next is placed
        #: there (``BridgeCall.seq``).
        self.last_seq: int | Fraction = -1
        #: The play the last drawn capture was taken in (how many plays had
        #: begun then; -1 for the final capture).
        self._captured_play = -1

    # Capturing

    def _capture(
        self,
        kind: str,
        instant: Fraction,
        extra_roots: Sequence[Mobject],
        camera: CameraExtras | None,
        until: Fraction | None = None,
        *,
        drawn: bool = True,
    ) -> Capture:
        """Capture the scene at ``instant``: a new version of every reachable
        mobject that changed since its last capture. A capture no frame
        draws (``drawn`` false) stays out of the record's history of drawn
        captures, which no reader of it then meets, checks nothing drawing a
        mobject runs and joins no display: what a drawn capture finds is as
        without it.
        """
        scene = self._scene
        seq: int | Fraction
        if drawn:
            seq = len(self._captures)
            self._undrawn = 0
        else:
            # Between the last drawn capture and the next, after any other
            # undrawn one since.
            self._undrawn += 1
            seq = len(self._captures) - Fraction(1, self._undrawn + 1)
        self.last_seq = seq
        previous = self._captures[-1] if self._captures else None
        roots = _reuse(tuple(scene.mobjects), previous.roots if previous else None)
        foreground = _reuse(
            tuple(scene.foreground_mobjects), previous.foreground if previous else None
        )
        moving = _reuse(
            tuple(scene.moving_mobjects), previous.moving if previous else None
        )
        updaters = _reuse(
            tuple(scene.updaters), previous.scene_updaters if previous else None
        )
        nodes = self._nodes
        seen: set[int] = set()
        # A ThreeDCamera sorts a shaded mobject by its z_index_group's centre.
        sorts_by_depth = drawn and _sorts_by_depth(scene.renderer.camera)
        grouped: list[Mobject] = []
        # The cameras the scene's camera draws into at this capture.
        drawing = _drawn_cameras(scene.renderer.camera) if drawn else frozenset()
        shown: list[Mobject] = []
        checked = self._entry_checked
        names: tuple[frozenset[str], frozenset[str]] | None = None
        classes: dict[type, str | None] = {}
        stack = [
            *reversed(extra_roots),
            *reversed(moving),
            *reversed(foreground),
            *reversed(roots),
        ]
        while stack:
            mob = stack.pop()
            key = id(mob)
            if key in seen:
                continue
            seen.add(key)
            node = nodes.get(key)
            if node is None:
                node = nodes[key] = _Node(mob)
            last = node.states[-1] if node.states else None
            state = NodeState.of(mob, last)
            if state is not last:
                node.seqs.append(seq)
                node.states.append(state)
            if sorts_by_depth:
                group = mob.__dict__.get("z_index_group")
                if (
                    group is not None
                    and group is not mob
                    and getattr(mob, "shade_in_3d", False)
                ):
                    grouped.append(mob)
            if drawn:
                other = mob.__dict__.get("camera")
                if other is not None and id(other) not in drawing and _shows(mob):
                    shown.append(mob)
            if not drawn:
                # Changed here, it is checked by the next capture that draws.
                if state is not last:
                    checked.discard(key)
            elif self._entry_refusal is None and (
                key not in checked or state is not last
            ):
                # A mobject entering the capture, or changed since it did,
                # is drawn from it: check what drawing it runs as it enters.
                checked.add(key)
                if names is None:
                    names = draw_names(scene.renderer.camera)
                self._entry_refusal = draw_member_refusal(names, mob, classes)
            stack.extend(reversed(state.submobjects))
        if not drawn:
            return Capture(
                seq, instant, kind, roots, foreground, moving, updaters, camera, until
            )
        references = (
            _references(grouped, previous.references if previous else ())
            if grouped
            else ()
        )
        play = len(self.plays) if kind != "final" else -1
        pictures: tuple[tuple[Mobject, np.ndarray], ...] = ()
        if shown:
            pictures, redrawn = _pictures(
                shown,
                previous.pictures if previous else (),
                self._kept_pictures,
                play,
            )
            if redrawn and self._entry_refusal is None:
                self._entry_refusal = DISPLAY_REDRAWN
        if (
            camera is not None
            and previous is not None
            and previous.camera is not None
            and self._captured_play == play
            and camera.bound() != previous.camera.bound()
            and self._entry_refusal is None
        ):
            self._entry_refusal = CAMERA_REBOUND
        self._note_pixel_shape(seq)
        self._captured_play = play
        capture = Capture(
            seq,
            instant,
            kind,
            roots,
            foreground,
            moving,
            updaters,
            camera,
            until,
            references,
            pictures,
        )
        self._captures.append(capture)
        return capture

    def _note_pixel_shape(self, seq: int) -> None:
        """Compare the configured pixel shape now with the first drawn
        capture's; from the first capture that finds another, every capture
        refuses its play (:data:`PIXEL_SHAPE_CHANGED`).
        """
        from arachne import config

        shape = (config.pixel_width, config.pixel_height)
        if self._pixel_shape is None:
            self._pixel_shape = shape
        elif shape != self._pixel_shape and self.pixel_shape_changed is None:
            self.pixel_shape_changed = seq
        if self.pixel_shape_changed is not None and self._entry_refusal is None:
            self._entry_refusal = PIXEL_SHAPE_CHANGED

    def entry_refusal(self) -> str | None:
        """Why a mobject found as it entered a capture since this was last
        asked can't be drawn from the record, if one can't; and check every
        mobject afresh from now on.
        """
        reason = self._entry_refusal
        self._entry_refusal = None
        self._entry_checked.clear()
        return reason

    def display_left(self, displays: Sequence[Mobject]) -> Mobject | None:
        """A display that was drawn into before and is not among ``displays``
        (a MultiCamera's now), if any: it shows the last picture its camera
        drew, which only drawing as the scene runs knows.
        """
        current = {id(display) for display in displays}
        for capture in reversed(self._captures):
            extras = capture.camera
            if extras is not None and extras.displays is not None:
                for display in extras.displays:
                    if id(display) not in current:
                        return display
                return None
        return None

    # Reading

    def mobjects(self) -> Iterator[Mobject]:
        """Every mobject the record holds a version of, each once, in the
        order the record first captured them: submobjects never added on
        their own and mobjects since removed included.
        """
        return (node.mob for node in self._nodes.values())

    def node_state(self, mob: Mobject, capture: Capture) -> NodeState | None:
        """``mob``'s state at ``capture``."""
        node = self._nodes.get(id(mob))
        return node.at(capture.seq) if node is not None else None

    def first_change(
        self, mobs: Iterable[Mobject], capture: Capture
    ) -> int | Fraction | None:
        """The seq of the earliest capture after ``capture`` at which one of
        ``mobs`` holds another version than it held there, or ``None`` when
        none does in any capture made so far. A mobject's state is the same
        at every capture before that seq: the record keeps one version per
        change, so this compares versions, not values.
        """
        first: int | Fraction | None = None
        for mob in mobs:
            node = self._nodes.get(id(mob))
            if node is None:
                continue
            index = bisect.bisect_right(node.seqs, capture.seq)
            if index < len(node.seqs) and (first is None or node.seqs[index] < first):
                first = node.seqs[index]
        return first

    def origin(self, mob: Mobject) -> Origin:
        """``mob``'s creating line and stable id."""
        node = self._nodes.get(id(mob))
        return node.origin if node is not None else origin_of(mob)

    def _walk(
        self, capture: Capture, roots: Iterable[Mobject]
    ) -> Iterator[tuple[Mobject, NodeState]]:
        """The nodes reachable from ``roots`` at ``capture``, depth first,
        each once, with their states there.
        """
        seen: set[int] = set()
        stack = list(roots)
        stack.reverse()
        while stack:
            mob = stack.pop()
            if id(mob) in seen:
                continue
            seen.add(id(mob))
            state = self.node_state(mob, capture)
            if state is None:
                continue
            yield mob, state
            stack.extend(reversed(state.submobjects))

    def play_at(self, t: Fraction) -> PlayRecord | None:
        """The play whose span ``[start, end)`` holds ``t``, if any."""
        starts = [play.start for play in self.plays]
        index = bisect.bisect_right(starts, t) - 1
        while index >= 0:
            play = self.plays[index]
            if play.start <= t < play.end:
                return play
            if play.end <= t:
                return None
            index -= 1
        return None

    def capture_at(self, t: Fraction) -> tuple[PlayRecord | None, Capture]:
        """The play holding ``t`` and the capture that answers it there."""
        play = self.play_at(t)
        if play is not None:
            return play, play.capture_at(t)
        if t < 0:
            raise LookupError(f"{t} is before the scene began.")
        if t < self.duration:
            raise LookupError(f"No play holds {t}.")
        assert self.final is not None, "the scene has not finished"
        return None, self.final

    def state_at(self, t: Fraction | float) -> State:
        """The scene at ``t``: every mobject in it, with its stable id,
        creating line, name, type, box, z, colour and text, and the route
        that answered.
        """
        from .routes import answer

        return answer(self, exact(t))

    def samples(self, fps: Fraction | float) -> list[Sample]:
        """The instants of the global frame grid at ``fps`` over the scene,
        each with the play and capture that answer it.
        """
        rate = exact(fps)
        found: list[Sample] = []
        for t in frames_in(Fraction(0), self.duration, rate):
            play, capture = self.capture_at(t)
            found.append(Sample(t, play.index if play else None, capture.drawn_seq))
        return found

    def markers_in(self, start: Fraction, end: Fraction) -> list[Marker]:
        """The markers placed at an instant in ``[start, end)``."""
        return [marker for marker in self.markers if start <= marker.t < end]


def _sorts_by_depth(camera: Any) -> bool:
    """Whether ``camera`` sorts the mobjects it draws by depth: a
    ThreeDCamera, which reads a shaded mobject's reference point.
    """
    from arachne.camera.three_d_camera import ThreeDCamera

    return isinstance(camera, ThreeDCamera)


def _drawn_cameras(camera: Any) -> frozenset[int]:
    """The cameras (by id) ``camera`` draws into as it captures: itself, and
    a MultiCamera's displays' cameras.
    """
    found = {id(camera)}
    for display in vars(camera).get("image_mobjects_from_cameras", ()):
        found.add(id(display.camera))
    return frozenset(found)


def _shows(mob: Any) -> bool:
    """Whether ``mob`` is a display of a camera's picture."""
    from arachne.mobject.types.image_mobject import ImageMobjectFromCamera

    return isinstance(mob, ImageMobjectFromCamera)


def _reuse(value: tuple[Any, ...], previous: tuple[Any, ...] | None) -> tuple[Any, ...]:
    """``previous`` when it holds the same objects as ``value``, so unchanged
    lists cost nothing to keep.
    """
    if (
        previous is not None
        and len(previous) == len(value)
        and all(map(is_, previous, value))
    ):
        return previous
    return value
