"""One case: both passes over two sequences, and its verdict."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

import numpy as np

from .cheap import cheap
from .model import (
    Box,
    ChangedSet,
    Footprint,
    Frames,
    Group,
    MapSource,
    Mode,
    ObjectMap,
    Thresholds,
)
from .sampling import choose_objects, sample
from .structural import flicker, structural
from .verdict import CLUSTER, COUNTS, FLICKER, Finding, Verdict


def _as_footprints(
    boxes: Mapping[str, Box], names: frozenset[str]
) -> dict[str, Footprint]:
    return {name: Footprint(boxes[name]) for name in names if name in boxes}


def _folded(
    clusters: Sequence[Finding], groups: Sequence[Group], flickers: Sequence[np.ndarray]
) -> list[Finding]:
    """A cluster sharing pixels with a flicker of the same frame is that flicker.

    ``groups`` are the clusters' pixel groups, in the clusters' order.
    """
    kept = []
    for finding, group in zip(clusters, groups, strict=True):
        x0, y0, x1, y1 = group.box
        if any(union[y0:y1, x0:x1][group.mask].any() for union in flickers):
            continue
        kept.append(finding)
    return kept


def compare(
    reference: Frames,
    candidate: Frames,
    *,
    mode: Mode,
    maps: MapSource,
    changed: Callable[[int, int], ChangedSet],
    plays: Sequence[tuple[int, int]],
    fps: Fraction,
    seed: int,
    key: str,
    thresholds: Thresholds = Thresholds(),
) -> Verdict:
    """Compare a candidate sequence with its reference and give one verdict.

    Every frame goes through the cheap pass. The structural pass reads the sampled
    frames (``sample``): there it takes each side's object map (for the objects
    ``choose_objects`` picks, when the source lists boxes), decides presence, clusters
    outside the reference's edges (``cross`` mode), edge pixels per object, and
    flicker in the candidate against the previous frame. Sequences of different
    lengths, instants or sizes are refused, never compared.
    """
    frames = cheap(reference, candidate, mode=mode, thresholds=thresholds)
    count = len(frames)
    sampled = sample(
        count,
        fps,
        plays,
        [f.index for f in frames if f.flagged],
        seed=seed,
        key=key,
        rate=thresholds.sample_rate,
    )
    findings: list[Finding] = []
    totals = dict.fromkeys(COUNTS, 0)
    totals["changed_pixels"] = sum(f.changed_pixels for f in frames)
    totals["groups_below"] = sum(len(f.below) for f in frames)
    totals["largest_below"] = max(
        (f.below[0].pixels for f in frames if f.below), default=0
    )
    edges: dict[str, dict[str, int]] = {}

    for i in sampled:
        listed = maps.boxes(i)
        ids = None
        if listed is not None:
            ids = choose_objects(
                listed[0],
                listed[1],
                frames[i].clusters + frames[i].below,
                cap=thresholds.replay_cap,
                reserve=thresholds.replay_reserve,
                seed=seed,
                key=f"{key}:{i}",
            )
            listed_ids = set(listed[0]) | set(listed[1])
            totals["objects_unexamined"] += len(listed_ids) - len(ids)
        ref_map = maps.map("reference", i, ids)
        cand_map = maps.map("candidate", i, ids)
        current = candidate.frame(i)
        found = structural(
            reference.frame(i),
            current,
            ref_map,
            cand_map,
            mode=mode,
            thresholds=thresholds,
            frame=i,
            t=frames[i].t,
            ids=ids,
        )
        for name, value in found.counts.items():
            totals[name] += value
        for name, value in found.edge_counts.items():
            into = edges.setdefault(name, {"pixels": 0, "changed": 0, "strong": 0})
            for what, n in value.items():
                into[what] += n
        own = [f for f in found.findings if f.code != CLUSTER]
        clusters = [f for f in found.findings if f.code == CLUSTER]
        flickers: list[np.ndarray] = []
        if i > 0:
            pair = changed(i - 1, i)
            totals["undecided_objects"] += len(pair.undecided)
            moving = pair.ids | pair.undecided
            at_current = _at(cand_map, listed[1] if listed else None, moving)
            if listed is not None:
                before = maps.boxes(i - 1)
                at_previous = (
                    _as_footprints(before[1], moving) if before is not None else {}
                )
            else:
                earlier = maps.map("candidate", i - 1, None)
                at_previous = dict(earlier.objects) if earlier is not None else {}
            groups = flicker(
                candidate.frame(i - 1),
                current,
                pair,
                at_previous,
                at_current,
                exempt=found.edges,
                thresholds=thresholds,
            )
            if groups is None:
                totals["flicker_unjudged"] += 1
            else:
                for group in groups:
                    union = np.zeros(current.shape[:2], dtype=bool)
                    x0, y0, x1, y1 = group.box
                    union[y0:y1, x0:x1] = group.mask
                    flickers.append(union)
                    own.append(
                        Finding(
                            code=FLICKER,
                            frame=i,
                            t=frames[i].t,
                            box=group.box,
                            detail={"pixels": group.pixels, "peak": group.peak},
                        )
                    )
        findings += own + _folded(clusters, found.clusters, flickers)
    findings.sort(key=Finding.sort_key)
    return Verdict(
        mode=mode,
        thresholds=thresholds.as_dict(),
        frames=count,
        sampled=sampled,
        findings=tuple(findings),
        counts=totals,
        edges={name: edges[name] for name in sorted(edges)},
    )


def _at(
    found: ObjectMap | None, boxes: Mapping[str, Box] | None, moving: frozenset[str]
) -> dict[str, Footprint]:
    """The moving objects' footprints at a frame: from its map, else from its boxes."""
    at = dict(_as_footprints(boxes, moving)) if boxes is not None else {}
    if found is not None:
        at.update(
            {name: found.objects[name] for name in moving if name in found.objects}
        )
    return at
