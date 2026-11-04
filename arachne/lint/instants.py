"""The instants the lint reads: frames of the render's grid.

By default (``"held+sampled"``) they are the first frame of the grid at or
after each ``k/fps`` of the scene, each play's first and last frame (a
frozen play's frame among them) and the final capture, the scene as its
script left it. The grid is the render's frame rate, whatever grid the
record keeps instants at (none, for a record that keeps every instant). Asked for at times, the lint reads the frame on screen at
each: the last frame at or before it, or the final capture after the last
play. Every one is a frame the render shows, so a finding's time names a
frame a viewer sees; a play no frame falls in is read at none.
"""

from __future__ import annotations

__all__ = ["DEFAULT", "LintInstant", "lint_instants", "render_grid"]

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from arachne.timeline.clock import exact, frames_in

if TYPE_CHECKING:
    from arachne.timeline.record import Record

#: The default instants: every play's frames that hold, and samples between.
DEFAULT = "held+sampled"


@dataclass(frozen=True)
class LintInstant:
    """An instant the lint reads: the play holding it (``None`` for the
    final capture) and why it is read: ``"sampled"``, ``"first"``,
    ``"last"``, ``"frozen"``, ``"final"`` or ``"asked"``.
    """

    t: Fraction
    play: int | None
    why: frozenset[str]


def render_grid(record: Record) -> Fraction:
    """The frame rate of the render ``record``'s execution describes."""
    return exact(record._scene.renderer.camera.frame_rate)


def lint_instants(
    record: Record,
    fps: Fraction | float = 4,
    *,
    at: str | Iterable[Fraction | float] | None = None,
) -> tuple[LintInstant, ...]:
    """The instants of ``record`` the lint reads, in time order: at ``fps``
    by default, or the frames on screen at the times ``at`` lists.
    """
    grid = render_grid(record)
    duration = record.duration
    found: dict[Fraction, set[str]] = {}

    def add(t: Fraction, why: str) -> None:
        found.setdefault(t, set()).add(why)

    if at is None or isinstance(at, str):
        if at not in (None, DEFAULT):
            raise ValueError(f"Lint at {DEFAULT!r} or at a list of times, not {at!r}.")
        rate = exact(fps)
        if rate <= 0:
            raise ValueError(f"The lint rate must be positive, not {fps!r}.")
        for k in range(math.ceil(duration * rate)):
            frame = Fraction(math.ceil(k / rate * grid)) / grid
            if frame < duration:
                add(frame, "sampled")
        for play in record.plays:
            frames = frames_in(play.start, play.end, grid)
            if not frames:
                continue
            add(frames[0], "first")
            add(frames[-1], "last")
            if play.held:
                add(frames[0], "frozen")
        add(duration, "final")
    else:
        for time in at:
            t = exact(time)
            if t >= duration:
                add(duration, "asked")
                add(duration, "final")
            else:
                add(Fraction(math.floor(max(t, Fraction(0)) * grid)) / grid, "asked")
    instants = []
    for t in sorted(found):
        holder = record.play_at(t)
        index = holder.index if holder is not None else None
        instants.append(LintInstant(t, index, frozenset(found[t])))
    return tuple(instants)
