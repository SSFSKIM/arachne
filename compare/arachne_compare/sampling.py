"""Which frames the structural pass reads, and which objects a replay examines there."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Mapping, Sequence
from fractions import Fraction

from .model import Box, Group
from .pixels import boxes_meet


def _draw(seed: int, key: str, item: object) -> int:
    """A number in [0, 2**64) from the seed, the key and the item, the same on every Python."""
    digest = hashlib.sha256(f"{seed}:{key}:{item}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sample(
    count: int,
    fps: Fraction,
    plays: Sequence[tuple[int, int]],
    flagged: Collection[int],
    *,
    seed: int,
    key: str,
    rate: float = 0.05,
) -> tuple[int, ...]:
    """The frames the structural pass reads, in order.

    The first and last frame of every play (``plays`` are half-open frame spans), the
    first frame of every second, every flagged frame, and a seeded share ``rate`` of
    the rest: frame i joins when the first 8 bytes of ``sha256("<seed>:<key>:<i>")``,
    as a big-endian integer, are below ``rate * 2**64``.
    """
    fps = Fraction(fps)
    if fps <= 0:
        raise ValueError("the frame rate must be positive")
    chosen: set[int] = set()
    for start, stop in plays:
        if not 0 <= start <= stop <= count:
            raise ValueError(f"play {(start, stop)} lies outside {count} frames")
        if stop > start:
            chosen.update((start, stop - 1))
    second = -1
    for i in range(count):
        at = math.floor(Fraction(i) / fps)
        if at != second:
            chosen.add(i)
            second = at
    for i in flagged:
        if not 0 <= i < count:
            raise ValueError(f"flagged frame {i} lies outside {count} frames")
        chosen.add(i)
    limit = rate * 2**64
    chosen.update(i for i in range(count) if _draw(seed, key, i) < limit)
    return tuple(sorted(chosen))


def choose_objects(
    reference: Mapping[str, Box],
    candidate: Mapping[str, Box],
    groups: Sequence[Group],
    *,
    cap: int,
    reserve: int,
    seed: int,
    key: str,
) -> list[str]:
    """The drawables a replay examines in one sampled frame, on both sides.

    In order: drawables on one side only; those whose box (on either side) meets a
    cluster, largest cluster first; those whose box meets a group under the
    thresholds, largest group first; then a seeded sample of the others. The first
    three classes take at most ``cap - reserve``, the sample fills up to ``cap``, and
    room the sample leaves goes back to the first three in their order.
    """
    if not 0 <= reserve <= cap:
        raise ValueError("the reserve is a part of the cap")

    def meets(name: str, box: Box) -> bool:
        return any(
            boxes_meet(side[name], box)
            for side in (reference, candidate)
            if name in side
        )

    every = sorted(set(reference) | set(candidate))
    ranked: list[str] = sorted(set(reference) ^ set(candidate))
    taken = set(ranked)
    for want_cluster in (True, False):
        for group in sorted(groups, key=lambda g: (-g.pixels, g.box)):
            if group.cluster != want_cluster:
                continue
            for name in every:
                if name not in taken and meets(name, group.box):
                    ranked.append(name)
                    taken.add(name)
    others = sorted(
        (n for n in every if n not in taken), key=lambda n: (_draw(seed, key, n), n)
    )
    first = ranked[: cap - reserve]
    chosen = first + others[: cap - len(first)]
    chosen += ranked[len(first) : len(first) + cap - len(chosen)]
    return chosen
