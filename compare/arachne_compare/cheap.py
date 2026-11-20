"""The cheap pass: every frame, frames only."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .model import Frames, Group, Mode, Thresholds
from .pixels import check_frame, label_groups, peak_difference


@dataclass(frozen=True)
class CheapFrame:
    """What the cheap pass found in one frame.

    ``clusters`` are the groups over the thresholds, ``below`` the groups under
    them, largest first. ``flicker`` holds candidate flicker: clusters among pixels
    that changed from the previous frame where they should not have. A frame with
    a cluster or a candidate is flagged.
    """

    index: int
    t: Fraction
    changed_pixels: int
    clusters: tuple[Group, ...]
    below: tuple[Group, ...]
    flicker: tuple[Group, ...]

    @property
    def flagged(self) -> bool:
        return bool(self.clusters or self.flicker)


def check_instants(reference: Frames, candidate: Frames) -> None:
    """Two sequences are compared only at the same instants, frame for frame."""
    if len(reference) != len(candidate):
        raise ValueError(
            f"the sequences hold {len(reference)} and {len(candidate)} frames; "
            "sequences of different lengths are not compared"
        )
    for i in range(len(reference)):
        if reference.instant(i) != candidate.instant(i):
            raise ValueError(
                f"frame {i} is at {reference.instant(i)} in the reference and at "
                f"{candidate.instant(i)} in the candidate; frames at different instants "
                "are not compared"
            )


def _clusters_only(groups: list[Group]) -> tuple[Group, ...]:
    return tuple(g for g in groups if g.cluster)


def cheap(
    reference: Frames,
    candidate: Frames,
    *,
    mode: Mode,
    thresholds: Thresholds = Thresholds(),
) -> tuple[CheapFrame, ...]:
    """Compare two sequences frame by frame.

    A pixel is changed by any difference in ``same`` mode and by more than
    ``cross_floor`` in ``cross`` mode. Candidate flicker at frame i is a cluster among
    the pixels that changed in the candidate from frame i - 1 where the reference's did
    not (a frame against the previous one of its own sequence: any difference counts).
    """
    if mode not in ("same", "cross"):
        raise ValueError(f"mode is 'same' or 'cross', not {mode!r}")
    check_instants(reference, candidate)
    floor = thresholds.member_floor(mode)
    found = []
    previous: tuple[np.ndarray, np.ndarray] | None = None
    for i in range(len(reference)):
        ref = check_frame(reference.frame(i), f"reference frame {i}")
        cand = check_frame(candidate.frame(i), f"candidate frame {i}")
        peak = peak_difference(ref, cand, f"frame {i}")
        members = peak > floor
        groups = label_groups(members, peak, thresholds)
        flicker: tuple[Group, ...] = ()
        if previous is not None:
            moved_cand = peak_difference(
                previous[1], cand, f"candidate frames {i - 1}, {i}"
            )
            moved_ref = peak_difference(
                previous[0], ref, f"reference frames {i - 1}, {i}"
            )
            flicker = _clusters_only(
                label_groups(
                    (moved_cand > 0) & (moved_ref == 0), moved_cand, thresholds
                )
            )
        found.append(
            CheapFrame(
                index=i,
                t=reference.instant(i),
                changed_pixels=int(members.sum()),
                clusters=_clusters_only(groups),
                below=tuple(g for g in groups if not g.cluster),
                flicker=flicker,
            )
        )
        previous = (ref, cand)
    return tuple(found)


def cheap_self(
    frames: Frames, *, thresholds: Thresholds = Thresholds()
) -> tuple[CheapFrame, ...]:
    """Candidate flicker in one sequence alone.

    At frame i: a cluster among the pixels that changed from frame i - 1 and are back
    at frame i + 1. ``changed_pixels`` counts the pixels that changed from frame i - 1.
    """
    found = []
    count = len(frames)
    previous: np.ndarray | None = None
    current = check_frame(frames.frame(0), "frame 0") if count else None
    for i in range(count):
        following = (
            check_frame(frames.frame(i + 1), f"frame {i + 1}")
            if i + 1 < count
            else None
        )
        changed = 0
        flicker: tuple[Group, ...] = ()
        if previous is not None and current is not None:
            moved = peak_difference(previous, current, f"frames {i - 1}, {i}")
            changed = int((moved > 0).sum())
            if following is not None:
                back = (
                    peak_difference(previous, following, f"frames {i - 1}, {i + 1}")
                    == 0
                )
                flicker = _clusters_only(
                    label_groups((moved > 0) & back, moved, thresholds)
                )
        found.append(
            CheapFrame(
                index=i,
                t=frames.instant(i),
                changed_pixels=changed,
                clusters=(),
                below=(),
                flicker=flicker,
            )
        )
        previous, current = current, following
    return tuple(found)
