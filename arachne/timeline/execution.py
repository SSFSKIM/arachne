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
instant both kept, by content key, and where the keys differ by the raw
values within :data:`TOLERANCE`. The first play they differ in is a
:class:`Divergence`, naming the input files whose digests differ.
"""

from __future__ import annotations

__all__ = [
    "TOLERANCE",
    "Divergence",
    "Execution",
    "ExecutionSummary",
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
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .source import SourceLine

if TYPE_CHECKING:
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
#: agree: content keys round numbers to 9 decimals, and two values either
#: side of a rounding boundary key apart however close they are.
TOLERANCE = 1e-9

#: The version of an execution summary's serialised form.
SUMMARY_VERSION = 1

#: A raw value of a field: an array, or a scalar.
Leaf: TypeAlias = np.ndarray | bool | int | float | str | None


@dataclass(frozen=True)
class Divergence:
    """Where two executions differ, or a play's restored instant differs
    from what the record kept, as the lint reports it (``replay.diverged``).
    """

    #: ``"execution"``: two executions of one source differ;
    #: ``"probe"``: a play restored at a recorded instant differs from it.
    kind: str
    #: The first play they differ in.
    play: int
    #: The line of the scene's code that played it.
    source_line: SourceLine | None
    #: How it differs.
    reason: str
    #: The input files whose digests differ between the executions, by path
    #: relative to the scene's directory.
    changed_inputs: tuple[str, ...] = ()
    code: str = "replay.diverged"


@dataclass(frozen=True, eq=False)
class NodeValues:
    """One mobject's version as a content key covers it: its type, how many
    submobjects it has, and each field's raw value, by name.
    """

    type: str
    children: int
    fields: tuple[tuple[str, Leaf], ...]


@dataclass(frozen=True)
class StateValues:
    """The scene's tree at one instant: its content key
    (:meth:`.RegisteredState.content_key`, numbers to 9 decimals) and its
    nodes in walking order, as indices into :attr:`ExecutionSummary.nodes`.
    """

    key: bytes
    nodes: tuple[int, ...]


@dataclass(frozen=True)
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
    identity, and each play's states with their raw values (each distinct
    node version once, in :attr:`nodes`).
    """

    identity: Identity
    plays: tuple[PlaySummary, ...] = field(repr=False)
    nodes: tuple[NodeValues, ...] = field(repr=False)

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
                    [
                        node.type,
                        node.children,
                        [[name, _leaf_json(value)] for name, value in node.fields],
                    ]
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
                            [str(t), *_state_json(state)]
                            for t, state in play.instants.items()
                        ],
                    }
                    for play in self.plays
                ],
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
                NodeValues(
                    kind,
                    children,
                    tuple((name, _leaf_from_json(value)) for name, value in fields),
                )
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
                    start_state=_state_from_json(*play["start_state"]),
                    instants={
                        Fraction(t): _state_from_json(key, nodes)
                        for t, key, nodes in play["instants"]
                    },
                )
                for play in data["plays"]
            ),
        )


def summarize(record: Record) -> ExecutionSummary:
    """The summary of the execution ``record`` is the record of."""
    from .record import SAMPLE_DECIMALS, RegisteredState

    nodes: list[NodeValues] = []
    numbers: dict[tuple[int, int], int] = {}

    def state_values(capture: Capture) -> StateValues:
        state = RegisteredState(record, capture)
        indices = []
        for mob, node in state.nodes():
            number = numbers.get((id(mob), id(node)))
            if number is None:
                number = numbers[id(mob), id(node)] = len(nodes)
                nodes.append(
                    NodeValues(
                        type(mob).__qualname__, len(node.submobjects), _fields(node)
                    )
                )
            indices.append(number)
        return StateValues(state.content_key(SAMPLE_DECIMALS), tuple(indices))

    plays = tuple(
        PlaySummary(
            index=play.index,
            start=play.start,
            end=play.end,
            source_line=play.source_line,
            start_state=state_values(play.start_state.capture),
            instants={
                capture.instant: state_values(capture) for capture in play.samples
            },
        )
        for play in record.plays
    )
    return ExecutionSummary(record.identity, plays, tuple(nodes))


def compare(mine: ExecutionSummary, theirs: ExecutionSummary) -> Divergence | None:
    """The first play where ``mine`` differs from ``theirs``, an execution
    of the same source, or ``None``. Each play's span, the state it began
    from and the states at every instant both kept are compared by content
    key, and where the keys differ, by raw value within :data:`TOLERANCE`
    (an array's dtype and shape exactly). Raises :class:`ValueError` for two
    executions of different sources.
    """
    if mine.identity.source_digest != theirs.identity.source_digest:
        raise ValueError(
            "Two executions of different sources are not compared: source digest"
            f" {mine.identity.source_digest} and {theirs.identity.source_digest}."
        )
    changed = changed_inputs(mine.identity, theirs.identity)
    for play, other in zip(mine.plays, theirs.plays, strict=False):
        reason = _play_difference(mine, play, theirs, other)
        if reason is not None:
            return Divergence(
                "execution", play.index, play.source_line, reason, changed
            )
    if len(mine.plays) != len(theirs.plays):
        index = min(len(mine.plays), len(theirs.plays))
        longer = mine.plays if len(mine.plays) > index else theirs.plays
        return Divergence(
            "execution",
            index,
            longer[index].source_line,
            f"this execution ran {len(mine.plays)} plays, and the other"
            f" {len(theirs.plays)}",
            changed,
        )
    return None


def changed_inputs(mine: Identity, theirs: Identity) -> tuple[str, ...]:
    """The input files whose digests differ between two identities, a file
    only one of them read included.
    """
    a, b = dict(mine.input_digests), dict(theirs.input_digests)
    return tuple(
        sorted(path for path in a.keys() | b.keys() if a.get(path) != b.get(path))
    )


def _play_difference(
    mine: ExecutionSummary,
    play: PlaySummary,
    theirs: ExecutionSummary,
    other: PlaySummary,
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
        if shared is None:
            continue
        why = _state_difference(mine, state, theirs, shared)
        if why is not None:
            return f"at {float(t):g} s, {why}"
    if play.end != other.end:
        return (
            f"it ended at {float(play.end):g} s, and at {float(other.end):g} s"
            " in the other execution"
        )
    return None


def _state_difference(
    mine: ExecutionSummary,
    state: StateValues,
    theirs: ExecutionSummary,
    other: StateValues,
) -> str | None:
    """How two states differ beyond :data:`TOLERANCE`, or ``None``."""
    if state.key == other.key:
        return None
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
        names = [name for name, _ in node.fields]
        twin_names = [name for name, _ in twin.fields]
        if names != twin_names:
            only = sorted(set(names) ^ set(twin_names))
            return (
                f"its {node.type}'s fields {', '.join(only)} are in one execution only"
            )
        for (name, value), (_, twin_value) in zip(
            node.fields, twin.fields, strict=True
        ):
            why = _leaf_difference(value, twin_value)
            if why is not None:
                return f"its {node.type}'s {name} {why}"
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
    paint's fields, the placement's points.
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
                (f"{name}.{field}", _leaf(own[field]))
                for field in Paint.FIELDS
                if field in own
            )
        elif isinstance(value, Placement):
            found.append((name, _leaf(value.points)))
        else:
            found.append((name, _leaf(value)))
    return tuple(found)


def _leaf(value: Any) -> Leaf:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value).view(np.ndarray)
        # An object array's bytes are addresses; its text is its content.
        return repr(array.tolist()) if array.dtype.hasobject else array
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


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


def _state_json(state: StateValues) -> list[Any]:
    return [state.key.hex(), list(state.nodes)]


def _state_from_json(key: str, nodes: list[int]) -> StateValues:
    return StateValues(bytes.fromhex(key), tuple(nodes))
