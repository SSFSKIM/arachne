"""Executing a scene again when a play it meant to draw from its record
turns out, mid-play, to need drawing as it runs.

A play is drawn from the record unless, when it begins, its camera or a
mobject it holds runs scene code while it draws (:mod:`.dispatch`). A
mobject can still join while the play runs and leave before it ends; the
record finds it as it enters a capture (:meth:`.Record._capture`), and by
then the play is no longer being drawn as it runs. The render does not
stop there: it executes the scene again from the start (:class:`Reexecute`),
knowing that play's refusal in advance, and draws that play and every play
after it as it runs.

Each re-execution is an execution of its own: its record says which
(:class:`Execution`) and whether it began each play its predecessors ran
from the same state. From the earliest play known to refuse on, nothing is
drawn from the record, so another execution is needed only for a refusal
found before it, which moves that threshold earlier: a refusal costs at
most one extra execution, and the executions end, however the plays an
execution runs shift from one to the next.

Any two executions of one source, in one process or two, compare through
their summaries (:func:`summarize`, :class:`ExecutionSummary`): at every
instant both kept and where their scripts ended, by key, and where the keys
differ by the raw values within :data:`TOLERANCE`. The first play they differ in is a
:class:`Divergence`, naming the input files whose digests differ.
"""

from __future__ import annotations

__all__ = [
    "TOLERANCE",
    "Divergence",
    "Execution",
    "ExecutionSummary",
    "Incomparable",
    "NodeValues",
    "PlaySummary",
    "Reexecute",
    "StateValues",
    "changed_inputs",
    "compare",
    "divergence",
    "play_keys",
    "summarize",
]

import base64
import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .source import SourceLine

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject

    from .draw import CameraExtras
    from .record import Capture, Identity, NodeState, Record


class Reexecute(BaseException):  # noqa: N818 (an instruction, not an error)
    """Play ``play``, drawn from the record, met ``reason`` mid-play: the
    scene is to be executed again with that play drawn as it runs. Not an
    ``Exception``, so that a scene catching its own errors lets it through.
    """

    def __init__(self, play: int, reason: str) -> None:
        super().__init__(
            f"{reason}; it joined the scene during play {play}, whose frames were"
            " to be drawn from the record, so the scene is executed again with"
            " that play drawn as it runs"
        )
        self.play = play
        self.reason = reason


@dataclass(frozen=True)
class Execution:
    """Which execution of its scene a record is."""

    #: 1 for the first; each re-execution counts on.
    number: int = 1
    #: The plays an earlier execution found, mid-play, that the record
    #: can't draw, with why: from the earliest of them on, every play is
    #: drawn as the scene runs (:attr:`live_from`).
    known: Mapping[int, str] = field(default_factory=dict)
    #: Where this execution began a play from another state than an earlier
    #: one did, or ``None`` when every play both ran began the same.
    divergence: str | None = None

    @property
    def live_from(self) -> int | None:
        """The first play drawn as the scene runs because a refusal is
        known, or ``None``.
        """
        return min(self.known) if self.known else None


def play_keys(record: Record) -> tuple[bytes, ...]:
    """The content key of the state each of the record's plays began from."""
    return tuple(play.start_state.content_key() for play in record.plays)


def divergence(
    record: Record, earlier: Sequence[tuple[int, tuple[bytes, ...]]]
) -> str | None:
    """Where ``record`` began a play from another state than an earlier
    execution (``(number, play_keys)``) did, if anywhere.
    """
    keys = play_keys(record)
    for number, theirs in earlier:
        if len(keys) < len(theirs):
            return (
                f"it ran {len(keys)} plays, where execution {number} ran {len(theirs)}"
            )
        for index, (mine, other) in enumerate(zip(keys, theirs, strict=False)):
            if mine != other:
                return (
                    f"play {index} began from another state than in execution {number}"
                )
    return None


# Comparing two executions of one source

#: How far apart two numbers of one field may be, in scene units, and still
#: agree: keys round numbers to 9 decimals, and two values either side of a
#: rounding boundary key apart however close they are.
TOLERANCE = 1e-9

#: How many decimals a state's key rounds numbers to.
KEY_DECIMALS = 9

#: The version of an execution summary's serialised form.
SUMMARY_VERSION = 2

#: A raw value of a field: an array, or a scalar.
Leaf: TypeAlias = np.ndarray | bool | int | float | str | None


class Incomparable(ValueError):  # noqa: N818 (a refusal, named as one)
    """Two executions :func:`compare` refuses: of different sources, or of a
    source not known for one of them.
    """


@dataclass(frozen=True)
class Divergence:
    """Where two executions differ, or a play's restored instant differs
    from what the record kept, as the lint reports it (``replay.diverged``).
    """

    #: ``"execution"``: two executions of one source differ;
    #: ``"probe"``: a play restored at a recorded instant differs from it.
    kind: str
    #: The first play they differ in; ``None`` when they differ only in what
    #: the script left after its last play (the record's final capture). For
    #: a probe's, the play it demoted, by its index in the execution the
    #: session holds; ``None`` when that execution does not play it, or
    #: plays its line otherwise, so which play it is there is unknown.
    play: int | None
    #: The line of the scene's code that played it.
    source_line: SourceLine | None
    #: How it differs.
    reason: str
    #: The input files whose digests differ between the executions, by path
    #: relative to the scene's directory.
    changed_inputs: tuple[str, ...] = ()
    code: str = "replay.diverged"

    @property
    def cause(self) -> tuple[Any, ...]:
        """What the divergence is a finding of, alike however many
        comparisons found it: its kind, the play and line, and the inputs
        that changed. Not its reason, whose amounts vary from one execution
        to the next.
        """
        return (self.kind, self.play, self.source_line, self.changed_inputs)


@dataclass(frozen=True, eq=False)
class NodeValues:
    """One mobject's version as a content key covers it: its type, how many
    submobjects it has, and each field's raw value, by name.
    """

    type: str
    children: int
    fields: tuple[tuple[str, Leaf], ...]


@dataclass(frozen=True, eq=False)
class StateValues:
    """What a frame drawn at one captured instant reads: the mobjects the
    capture reaches, from the scene's list, its foreground and moving lists,
    the play's draw lists and the camera's own mobjects, in walking order
    (as indices into :attr:`ExecutionSummary.nodes`); and ``drawing``, the
    capture's drawing inputs that are not mobjects: which of those nodes
    each draw list holds, in order (as positions in ``nodes``, ``-1`` for a
    mobject the capture does not reach), and the camera's values.

    ``key`` digests all of it, every number rounded to 9 decimals and every
    array's dtype and shape included: equal keys mean equal states.
    """

    key: bytes
    nodes: tuple[int, ...]
    drawing: tuple[tuple[str, Leaf], ...]


@dataclass(frozen=True, eq=False)
class PlaySummary:
    """One play: its span, its line, the state it began from and the state
    at each instant the record kept, by instant.
    """

    index: int
    start: Fraction
    end: Fraction
    source_line: SourceLine | None
    start_state: StateValues
    instants: dict[Fraction, StateValues]


@dataclass(frozen=True, eq=False)
class ExecutionSummary:
    """What another process needs to compare an execution with this one: its
    identity, each play's states and the state the script left at its end
    (each distinct node version once, in :attr:`nodes`).
    """

    identity: Identity
    plays: tuple[PlaySummary, ...] = field(repr=False)
    nodes: tuple[NodeValues, ...] = field(repr=False)
    #: The scene when its script ended, and when that was; ``None`` for a
    #: record without a final capture.
    final: StateValues | None = field(default=None, repr=False)
    end: Fraction | None = None

    def to_json(self) -> str:
        """The summary as JSON text; :meth:`from_json` reads it back."""
        identity = self.identity
        return json.dumps(
            {
                "version": SUMMARY_VERSION,
                "identity": {
                    "source_digest": identity.source_digest,
                    "input_digests": [list(pair) for pair in identity.input_digests],
                    "seed": identity.seed,
                },
                "nodes": [
                    [node.type, node.children, _fields_json(node.fields)]
                    for node in self.nodes
                ],
                "plays": [
                    {
                        "index": play.index,
                        "start": str(play.start),
                        "end": str(play.end),
                        "source_line": None
                        if play.source_line is None
                        else [play.source_line.path, play.source_line.line],
                        "start_state": _state_json(play.start_state),
                        "instants": [
                            [str(t), _state_json(state)]
                            for t, state in play.instants.items()
                        ],
                    }
                    for play in self.plays
                ],
                "final": None if self.final is None else _state_json(self.final),
                "end": None if self.end is None else str(self.end),
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> ExecutionSummary:
        """The summary :meth:`to_json` wrote."""
        from .record import Identity

        data = json.loads(text)
        if data.get("version") != SUMMARY_VERSION:
            raise ValueError(
                f"An execution summary of version {data.get('version')!r}; this"
                f" engine reads version {SUMMARY_VERSION}."
            )
        identity = data["identity"]
        return cls(
            identity=Identity(
                source_digest=identity["source_digest"],
                input_digests=tuple(
                    (path, digest) for path, digest in identity["input_digests"]
                ),
                seed=identity["seed"],
            ),
            nodes=tuple(
                NodeValues(kind, children, _fields_from_json(fields))
                for kind, children, fields in data["nodes"]
            ),
            plays=tuple(
                PlaySummary(
                    index=play["index"],
                    start=Fraction(play["start"]),
                    end=Fraction(play["end"]),
                    source_line=None
                    if play["source_line"] is None
                    else SourceLine(*play["source_line"]),
                    start_state=_state_from_json(play["start_state"]),
                    instants={
                        Fraction(t): _state_from_json(state)
                        for t, state in play["instants"]
                    },
                )
                for play in data["plays"]
            ),
            final=None if data["final"] is None else _state_from_json(data["final"]),
            end=None if data["end"] is None else Fraction(data["end"]),
        )


def summarize(record: Record) -> ExecutionSummary:
    """The summary of the execution ``record`` is the record of.

    The camera's own mobjects are those the camera holds when the summary
    is made, as the record's frames are drawn with it.
    """
    from .draw import camera_nodes

    nodes: list[NodeValues] = []
    node_keys: list[bytes] = []
    numbers: dict[tuple[int, int], int] = {}
    camera = record._scene.renderer.camera
    owned = tuple(camera_nodes(camera))

    def state_values(
        capture: Capture, lists: tuple[tuple[str, Sequence[Mobject]], ...]
    ) -> StateValues:
        roots = [
            *capture.roots,
            *capture.foreground,
            *capture.moving,
            *(mob for _, members in lists for mob in members),
            *owned,
        ]
        indices: list[int] = []
        positions: dict[int, int] = {}
        for mob, node in record._walk(capture, roots):
            number = numbers.get((id(mob), id(node)))
            if number is None:
                number = numbers[id(mob), id(node)] = len(nodes)
                values = NodeValues(
                    type(mob).__qualname__, len(node.submobjects), _fields(node)
                )
                nodes.append(values)
                node_keys.append(_node_key(values))
            positions[id(mob)] = len(indices)
            indices.append(number)

        def at(members: Iterable[Mobject], *, ordered: bool = True) -> np.ndarray:
            found = [positions.get(id(mob), -1) for mob in members]
            return np.array(found if ordered else sorted(found), dtype=np.int64)

        drawing: list[tuple[str, Leaf]] = [
            ("roots", at(capture.roots)),
            ("foreground", at(capture.foreground)),
            ("moving", at(capture.moving)),
            *((name, at(members)) for name, members in lists),
            ("camera.mobjects", at(owned)),
        ]
        if capture.camera is not None:
            drawing.extend(_camera_leaves("camera", capture.camera, at))
        digest = hashlib.blake2b(digest_size=16)
        for number in indices:
            digest.update(node_keys[number])
        digest.update(_fields_key(drawing))
        return StateValues(digest.digest(), tuple(indices), tuple(drawing))

    plays = []
    for play in record.plays:
        moving = (("play.moving", play.moving),)
        plays.append(
            PlaySummary(
                index=play.index,
                start=play.start,
                end=play.end,
                source_line=play.source_line,
                start_state=state_values(
                    play.start_state.capture, (*moving, ("play.static", play.static))
                ),
                instants={
                    capture.instant: state_values(capture, moving)
                    for capture in play.samples
                },
            )
        )
    final = None if record.final is None else state_values(record.final, ())
    return ExecutionSummary(
        record.identity,
        tuple(plays),
        tuple(nodes),
        final,
        None if record.final is None else record.final.instant,
    )


def compare(
    mine: ExecutionSummary,
    theirs: ExecutionSummary,
    *,
    until: Fraction | None = None,
    before_play: int | None = None,
) -> Divergence | None:
    """The first play where ``mine`` differs from ``theirs``, an execution
    of the same source, or ``None``. Each play's span, the state it began
    from and the states at every instant both kept are compared, then the
    state the script left at its end: by key, and where keys differ, by raw
    value within :data:`TOLERANCE` (an array's dtype and shape exactly).
    Raises :class:`Incomparable` for two executions of different sources,
    and for one whose source is unknown (a ``None`` digest: a source file
    gone before its run ended, a scene defined in no file), which is the
    same source as none.

    With ``until``, only what both did up to that instant is compared: the
    plays that began by then, the instants up to it, a play's end where
    either reached it, and not where the scripts ended. An execution
    stopped once past ``until`` (a forward run, :mod:`.fallback`) is so
    compared with a whole one. With ``before_play``, only the plays before
    that one are compared, and not where the scripts ended: what one
    execution drew while it ran is written into it from there on.
    """
    digests = (mine.identity.source_digest, theirs.identity.source_digest)
    if None in digests:
        raise Incomparable(
            "An execution whose source is unknown is not compared: source digest"
            f" {digests[0]} and {digests[1]}."
        )
    if digests[0] != digests[1]:
        raise Incomparable(
            "Two executions of different sources are not compared: source digest"
            f" {digests[0]} and {digests[1]}."
        )
    changed = changed_inputs(mine.identity, theirs.identity)
    plays = _begun(mine.plays, until)[:before_play]
    others = _begun(theirs.plays, until)[:before_play]
    for play, other in zip(plays, others, strict=False):
        reason = _play_difference(mine, play, theirs, other, until)
        if reason is not None:
            return Divergence(
                "execution", play.index, play.source_line, reason, changed
            )
    if len(plays) != len(others):
        index = min(len(plays), len(others))
        longer = plays if len(plays) > index else others
        return Divergence(
            "execution",
            index,
            longer[index].source_line,
            f"this execution ran {len(plays)} plays, and the other {len(others)}"
            + ("" if until is None else f" by {float(until):g} s"),
            changed,
        )
    if until is not None or before_play is not None:
        return None
    reason = _end_difference(mine, theirs)
    if reason is not None:
        return Divergence("execution", None, None, reason, changed)
    return None


def changed_inputs(mine: Identity, theirs: Identity) -> tuple[str, ...]:
    """The input files whose digests differ between two identities, a file
    only one of them read included.
    """
    a, b = dict(mine.input_digests), dict(theirs.input_digests)
    return tuple(
        sorted(path for path in a.keys() | b.keys() if a.get(path) != b.get(path))
    )


def _begun(
    plays: tuple[PlaySummary, ...], until: Fraction | None
) -> tuple[PlaySummary, ...]:
    """The plays that began by ``until`` (all of them without it)."""
    if until is None:
        return plays
    return tuple(play for play in plays if play.start <= until)


def _play_difference(
    mine: ExecutionSummary,
    play: PlaySummary,
    theirs: ExecutionSummary,
    other: PlaySummary,
    until: Fraction | None = None,
) -> str | None:
    if play.start != other.start:
        return (
            f"it began at {float(play.start):g} s, and at {float(other.start):g} s"
            " in the other execution"
        )
    why = _state_difference(mine, play.start_state, theirs, other.start_state)
    if why is not None:
        return f"it began from another state: {why}"
    for t, state in play.instants.items():
        shared = other.instants.get(t)
        if shared is None or (until is not None and t > until):
            continue
        why = _state_difference(mine, state, theirs, shared)
        if why is not None:
            return f"at {float(t):g} s, {why}"
    if play.end != other.end and (until is None or min(play.end, other.end) <= until):
        return (
            f"it ended at {float(play.end):g} s, and at {float(other.end):g} s"
            " in the other execution"
        )
    return None


def _end_difference(mine: ExecutionSummary, theirs: ExecutionSummary) -> str | None:
    """How the states the two scripts left at their ends differ, if they
    do: what the scripts did after their last plays.
    """
    if mine.final is None or theirs.final is None:
        return None
    if mine.end != theirs.end:
        return (
            f"its script ended at {float(mine.end or 0):g} s, and at"
            f" {float(theirs.end or 0):g} s in the other execution"
        )
    why = _state_difference(mine, mine.final, theirs, theirs.final)
    if why is None:
        return None
    return f"after its last play, when its script ended, {why}"


def _state_difference(
    mine: ExecutionSummary,
    state: StateValues,
    theirs: ExecutionSummary,
    other: StateValues,
) -> str | None:
    """How two states differ beyond :data:`TOLERANCE`, or ``None``."""
    if state.key == other.key:
        return None
    # A draw list holding more or fewer mobjects is what moves the nodes it
    # reaches, so it is named first.
    for (name, value), (other_name, other_value) in zip(
        state.drawing, other.drawing, strict=False
    ):
        if (
            name == other_name
            and isinstance(value, np.ndarray)
            and isinstance(other_value, np.ndarray)
            and value.dtype.kind == other_value.dtype.kind == "i"
            and value.shape != other_value.shape
        ):
            return (
                f"its drawing's {name} list holds {len(value)} mobjects, and"
                f" {len(other_value)} in the other execution"
            )
    if len(state.nodes) != len(other.nodes):
        return (
            f"it holds {len(state.nodes)} mobjects, and {len(other.nodes)} in the"
            " other execution"
        )
    for a, b in zip(state.nodes, other.nodes, strict=True):
        node, twin = mine.nodes[a], theirs.nodes[b]
        if node.type != twin.type or node.children != twin.children:
            return (
                f"a {node.type} with {node.children} submobjects is a {twin.type}"
                f" with {twin.children} in the other execution"
            )
        why = _fields_difference(node.fields, twin.fields)
        if why is not None:
            return f"its {node.type}'s {why}"
    why = _fields_difference(state.drawing, other.drawing)
    if why is not None:
        return f"its drawing's {why}"
    return None


def _fields_difference(
    fields: tuple[tuple[str, Leaf], ...], others: tuple[tuple[str, Leaf], ...]
) -> str | None:
    names = [name for name, _ in fields]
    other_names = [name for name, _ in others]
    if names != other_names:
        only = sorted(set(names) ^ set(other_names))
        if not only:
            return "fields come in another order in the other execution"
        return f"fields {', '.join(only)} are in one execution only"
    for (name, value), (_, other) in zip(fields, others, strict=True):
        why = _leaf_difference(value, other)
        if why is not None:
            return f"{name} {why}"
    return None


def _leaf_difference(a: Leaf, b: Leaf) -> str | None:
    """How two raw values of one field differ beyond :data:`TOLERANCE`."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return (
                f"is {a.dtype}{list(a.shape)}, and {b.dtype}{list(b.shape)} in the"
                " other execution"
            )
        if a.dtype.kind not in "fc":
            return None if np.array_equal(a, b) else "differs"
        if np.allclose(a, b, rtol=0, atol=TOLERANCE, equal_nan=True):
            return None
        return f"differs by up to {np.nanmax(np.abs(a - b)):.3g}"
    if type(a) is float and type(b) is float:
        if abs(a - b) <= TOLERANCE or (a != a and b != b):
            return None
        return f"differs by {abs(a - b):.3g}"
    if type(a) is type(b) and a == b:
        return None
    return f"is {_brief(a)}, and {_brief(b)} in the other execution"


def _brief(value: Leaf) -> str:
    if isinstance(value, np.ndarray):
        return f"an array of {value.dtype}{list(value.shape)}"
    return repr(value)


def _fields(node: NodeState) -> tuple[tuple[str, Leaf], ...]:
    """The raw values of the fields a content key covers
    (:func:`.content_fields`), a paint's and a placement's spelled out: the
    paint's fields (its numbers as floats, as its key takes them), the
    placement's points.
    """
    from arachne.mobject.placement import Placement
    from arachne.mobject.values import Paint

    from .record import content_fields

    keyed = {name for name, _ in content_fields(node)}
    found: list[tuple[str, Leaf]] = []
    for name, value in sorted(zip(node.keys, node.values, strict=True)):
        if name not in keyed:
            continue
        if isinstance(value, Paint):
            own = vars(value)
            found.extend(
                (f"{name}.{part}", _paint_leaf(own[part]))
                for part in Paint.FIELDS
                if part in own
            )
        elif isinstance(value, Placement):
            found.append((name, _leaf(value.points)))
        else:
            found.append((name, _leaf(value)))
    return tuple(found)


def _camera_leaves(
    prefix: str, extras: CameraExtras, at: Callable[..., np.ndarray]
) -> Iterator[tuple[str, Leaf]]:
    """A camera's drawing inputs that are not mobjects, and which mobjects
    its sets and lists hold (positions, by ``at``).
    """
    for name, value in extras.values:
        yield f"{prefix}.{name}", _leaf(value)
    if extras.background is not None:
        # A picture: compared exactly, by its digest.
        background = _contiguous(extras.background)
        yield (
            f"{prefix}.background",
            hashlib.sha256(
                f"{background.dtype.str}{background.shape}".encode()
                + background.tobytes()
            ).hexdigest(),
        )
    if extras.fixed_in_frame:
        yield f"{prefix}.fixed_in_frame", at(extras.fixed_in_frame, ordered=False)
    if extras.centers:
        # A lookup from each mobject to its centre, not a draw order: the
        # pairs are taken by the mobjects' places in the walk.
        positions = at(mob for mob, _ in extras.centers)
        centers = np.array([center for _, center in extras.centers], dtype=float)
        order = sorted(
            range(len(positions)),
            key=lambda k: (int(positions[k]), tuple(centers[k].ravel().tolist())),
        )
        yield f"{prefix}.fixed_orientation", positions[order]
        yield f"{prefix}.fixed_orientation_centers", centers[order]
    if extras.displays is not None:
        yield f"{prefix}.displays", at(extras.displays)
    for number, (_, sub) in enumerate(extras.sub_cameras):
        yield from _camera_leaves(f"{prefix}.sub{number}", sub, at)


def _paint_leaf(value: Any) -> Leaf:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return float(value)
    return _leaf(value)


def _leaf(value: Any) -> Leaf:
    from arachne.utils.color import ManimColor

    if isinstance(value, np.ndarray):
        array = _contiguous(value)
        # An object array's bytes are addresses; its text is its content.
        return repr(array.tolist()) if array.dtype.hasobject else array
    if isinstance(value, ManimColor):
        # Its numbers, not its text: the hex string keeps 8 bits a channel.
        return _contiguous(value.to_rgba()).astype(float)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _contiguous(value: np.ndarray) -> np.ndarray:
    """``value`` as a plain C-ordered array of its own shape, a
    zero-dimensional one included (``np.ascontiguousarray`` makes it 1-d).
    """
    array = np.asarray(value).view(np.ndarray)
    return array if array.flags.c_contiguous else array.copy(order="C")


def _node_key(node: NodeValues) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{node.type}\0{node.children}\0".encode())
    digest.update(_fields_key(node.fields))
    return digest.digest()


def _fields_key(fields: Iterable[tuple[str, Leaf]]) -> bytes:
    """A digest of named raw values, numbers rounded to
    :data:`KEY_DECIMALS`, each array's dtype and shape included.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, value in fields:
        if isinstance(value, np.ndarray):
            head = f"{name}\0A{value.dtype.str}{value.shape}\0"
            if value.dtype.kind in "fc":
                # Adding zero turns -0.0, which rounding makes, into 0.0.
                value = np.asarray(np.round(value, KEY_DECIMALS) + 0.0)
            digest.update(head.encode())
            digest.update(_contiguous(value).tobytes())
        elif type(value) is float:
            digest.update(f"{name}\0F{round(value, KEY_DECIMALS) + 0.0!r}\0".encode())
        else:
            digest.update(f"{name}\0{type(value).__name__}:{value!r}\0".encode())
    return digest.digest()


def _leaf_json(value: Leaf) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    return value


def _leaf_from_json(value: Any) -> Leaf:
    if isinstance(value, dict):
        return np.frombuffer(
            base64.b64decode(value["data"]), dtype=np.dtype(value["dtype"])
        ).reshape(value["shape"])
    return value


def _fields_json(fields: tuple[tuple[str, Leaf], ...]) -> list[Any]:
    return [[name, _leaf_json(value)] for name, value in fields]


def _fields_from_json(fields: list[Any]) -> tuple[tuple[str, Leaf], ...]:
    return tuple((name, _leaf_from_json(value)) for name, value in fields)


def _state_json(state: StateValues) -> list[Any]:
    return [state.key.hex(), list(state.nodes), _fields_json(state.drawing)]


def _state_from_json(data: list[Any]) -> StateValues:
    key, nodes, drawing = data
    return StateValues(bytes.fromhex(key), tuple(nodes), _fields_from_json(drawing))
