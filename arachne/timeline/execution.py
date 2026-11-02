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
"""

from __future__ import annotations

__all__ = ["Execution", "Reexecute", "divergence", "play_keys"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .record import Record


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
