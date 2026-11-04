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
    "Identity",
    "Marker",
    "NodeState",
    "PlayRecord",
    "Record",
    "Sample",
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
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene

    from .admission import Admission
    from .bodies import Exit
    from .draw import CameraExtras
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
        self.seqs: list[int] = []
        self.states: list[NodeState] = []

    def at(self, seq: int) -> NodeState | None:
        index = bisect.bisect_right(self.seqs, seq) - 1
        return self.states[index] if index >= 0 else None


@dataclass(frozen=True)
class Capture:
    """One captured instant: what the scene held there.

    ``kind`` is ``"static"`` (a play's state when its static picture is
    drawn, right after its animations began), ``"sample"`` (an evaluated
    instant of a play) or ``"final"`` (the scene after its script ended).
    ``until`` is the next instant the play evaluated, so the capture
    answers every time in ``[instant, until)``.

    ``moving`` is the scene's list of moving mobjects there, which a frame
    of the play draws over its static picture: flattened when the play
    began, and grown by whatever the scene added since (CE's ``Scene.add``
    appends to it during a play).
    """

    seq: int
    instant: Fraction
    kind: str
    roots: tuple[Mobject, ...]
    foreground: tuple[Mobject, ...]
    moving: tuple[Mobject, ...]
    scene_updaters: tuple[Any, ...]
    camera: CameraExtras | None
    until: Fraction | None = None

    def draw_list(self) -> list[Mobject]:
        """The scene's mobjects with the foreground last, as CE draws a frame
        when no list is given (``list_update(mobjects, foreground_mobjects)``).
        """
        foreground = set(map(id, self.foreground))
        return [m for m in self.roots if id(m) not in foreground] + list(
            self.foreground
        )


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
    # What drawing it needs: CE's draw lists, flattened when it began (a
    # frame draws its capture's moving list, which can have grown since).
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
        """
        digest = hashlib.blake2b(digest_size=16)
        for mob, state in self.nodes():
            digest.update(type(mob).__qualname__.encode())
            digest.update(len(state.submobjects).to_bytes(4, "big"))
            for key, part in content_fields(state, decimals):
                digest.update(key.encode())
                digest.update(part)
        return digest.digest()


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
    #: The first play to end once the scene armed the motion rule
    #: (:meth:`.Recorder.arm_motion`), by index; ``None`` if it never did.
    armed_at: int | None = None

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
        #: Which execution of the scene made it.
        self.execution = Execution()
        #: The first mobject found as it entered a capture since
        #: :meth:`entry_refusal` was last asked whose drawing runs the
        #: scene's code, and the mobjects checked since (by id).
        self._entry_refusal: str | None = None
        self._entry_checked: set[int] = set()
        self.final: Capture | None = None
        #: The names the scene's code had for its mobjects when it ended.
        self.final_names: dict[int, str] = {}
        self._nodes: dict[int, _Node] = {}
        self._captures: list[Capture] = []
        #: Per MultiCamera display (by id): the first capture its camera was
        #: drawn into at, and that camera's picture before it, which the
        #: display shows wherever it is on screen before then.
        self._display_joins: dict[int, tuple[int, np.ndarray]] = {}

    # Capturing

    def _capture(
        self,
        kind: str,
        instant: Fraction,
        extra_roots: Sequence[Mobject],
        camera: CameraExtras | None,
        until: Fraction | None = None,
    ) -> Capture:
        """Capture the scene at ``instant``: a new version of every reachable
        mobject that changed since its last capture.
        """
        scene = self._scene
        seq = len(self._captures)
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
            if self._entry_refusal is None and (
                key not in checked or state is not last
            ):
                # A mobject entering the capture, or changed since it did,
                # is drawn from it: check what drawing it runs as it enters.
                checked.add(key)
                if names is None:
                    names = draw_names(scene.renderer.camera)
                self._entry_refusal = draw_member_refusal(names, mob, classes)
            stack.extend(reversed(state.submobjects))
        capture = Capture(
            seq, instant, kind, roots, foreground, moving, updaters, camera, until
        )
        self._captures.append(capture)
        if camera is not None and camera.displays:
            for display in camera.displays:
                if id(display) not in self._display_joins:
                    picture = display.camera.pixel_array.copy()  # type: ignore[attr-defined]
                    picture.flags.writeable = False
                    self._display_joins[id(display)] = (seq, picture)
        return capture

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

    def display_picture(self, display: Mobject, capture: Capture) -> np.ndarray | None:
        """The picture ``display`` shows at ``capture`` when its camera is not
        drawn into there: its camera's first picture, before it ever was.
        """
        joined = self._display_joins.get(id(display))
        if joined is None or joined[0] < capture.seq:
            return None
        return joined[1]

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
            found.append(Sample(t, play.index if play else None, capture.seq))
        return found

    def markers_in(self, start: Fraction, end: Fraction) -> list[Marker]:
        """The markers placed at an instant in ``[start, end)``."""
        return [marker for marker in self.markers if start <= marker.t < end]


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
