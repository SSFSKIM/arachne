"""The structural pass: one sampled frame with each side's object map, and flicker."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .model import ChangedSet, Footprint, Group, Mode, ObjectMap, Thresholds
from .pixels import (
    boxes_meet,
    footprints_union,
    label_groups,
    map_edges,
    peak_difference,
    tight_box,
)
from .verdict import (
    CLUSTER,
    EDGE_OVER_BUDGET,
    PRESENCE_EXTRA,
    PRESENCE_MISSING,
    Finding,
)


@dataclass(frozen=True)
class Structural:
    """What the structural pass found in one frame.

    ``clusters`` keep their masks. ``edges`` is the union of the reference's edge
    pixels in two-renderer mode with a reference map, else ``None``. ``edge_counts``
    give each object's edge pixels, how many changed and how many changed strongly.
    """

    findings: tuple[Finding, ...]
    clusters: tuple[Group, ...]
    edges: np.ndarray | None = field(repr=False)
    edge_counts: Mapping[str, Mapping[str, int]]
    counts: Mapping[str, int]


def _state(found: ObjectMap | None, name: str) -> str:
    if found is None:
        return "untested"
    footprint = found.objects.get(name)
    if footprint is None:
        return "empty" if found.complete else "untested"
    if footprint.mask is None:
        return "untested"
    return "empty" if footprint.empty else "drawn"


def _is_part(name: str, maps: tuple[ObjectMap | None, ...]) -> bool:
    return any(
        m is not None and name in m.objects and m.objects[name].parent for m in maps
    )


def _presence(
    reference: ObjectMap | None,
    candidate: ObjectMap | None,
    ids: Collection[str] | None,
    frame: int,
    t: Fraction,
) -> tuple[list[Finding], int]:
    """Presence from the maps alone: drawn on one side, examined and empty on the other."""
    names = set(ids or ())
    for found in (reference, candidate):
        if found is not None:
            names.update(found.objects)
    findings = []
    untested = 0
    for name in sorted(names):
        if _is_part(name, (reference, candidate)):
            continue
        ref, cand = _state(reference, name), _state(candidate, name)
        if "untested" in (ref, cand):
            untested += 1
            continue
        if ref == cand:
            continue
        drawn = reference if ref == "drawn" else candidate
        assert drawn is not None
        footprint = drawn.objects[name]
        assert footprint.mask is not None
        findings.append(
            Finding(
                code=PRESENCE_MISSING if ref == "drawn" else PRESENCE_EXTRA,
                frame=frame,
                t=t,
                box=tight_box(footprint.mask, footprint.box[:2]),
                object=name,
                source_line=footprint.source,
                detail={"pixels": int(footprint.mask.sum())},
            )
        )
    return findings, untested


def _drawn_by(
    group: Group, maps: tuple[ObjectMap | None, ...]
) -> tuple[str | None, str | None]:
    """The drawable whose footprint holds most of the group's pixels, with its source line."""
    assert group.mask is not None
    best: tuple[int, str, str | None] | None = None
    gx0, gy0, gx1, gy1 = group.box
    for found in maps:
        if found is None:
            continue
        for name in sorted(found.objects):
            footprint = found.objects[name]
            if footprint.mask is None or not boxes_meet(footprint.box, group.box):
                continue
            fx0, fy0, fx1, fy1 = footprint.box
            x0, y0, x1, y1 = max(gx0, fx0), max(gy0, fy0), min(gx1, fx1), min(gy1, fy1)
            overlap = int(
                (
                    group.mask[y0 - gy0 : y1 - gy0, x0 - gx0 : x1 - gx0]
                    & footprint.mask[y0 - fy0 : y1 - fy0, x0 - fx0 : x1 - fx0]
                ).sum()
            )
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, footprint.parent or name, footprint.source)
    return (None, None) if best is None else (best[1], best[2])


def structural(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_map: ObjectMap | None,
    candidate_map: ObjectMap | None,
    *,
    mode: Mode,
    thresholds: Thresholds = Thresholds(),
    frame: int = 0,
    t: Fraction = Fraction(0),
    ids: Collection[str] | None = None,
) -> Structural:
    """Presence, clusters outside edges, and edge pixels per object, in one frame.

    Presence is decided first and from the maps alone, for every listed or asked id;
    no pixel is exempt from it. In ``cross`` mode the reference map's edge pixels are
    taken out of the cluster test, counted per object, and held to the edge budget
    where one is set. Without a reference map nothing is exempt.
    """
    peak = peak_difference(reference, candidate, f"frame {frame}")
    size = peak.shape
    for found, side in ((reference_map, "reference"), (candidate_map, "candidate")):
        if found is not None and tuple(found.size) != size:
            raise ValueError(
                f"frame {frame}: the {side} map is {found.size[1]} x {found.size[0]}, the "
                f"frame {size[1]} x {size[0]}; a map of another size is not used"
            )
    findings, untested = _presence(reference_map, candidate_map, ids, frame, t)
    floor = thresholds.member_floor(mode)
    members = peak > floor
    edges = None
    edge_counts: dict[str, dict[str, int]] = {}
    counts = {
        "presence_untested": untested,
        "edge_pixels": 0,
        "edge_changed": 0,
        "edge_strong": 0,
    }
    if mode == "cross" and reference_map is not None:
        edges, each = map_edges(reference_map.objects, size)
        members &= ~edges
        counts["edge_pixels"] = int(edges.sum())
        counts["edge_changed"] = int((edges & (peak > floor)).sum())
        counts["edge_strong"] = int((edges & (peak > thresholds.strong_delta)).sum())
        budget = thresholds.edge_budget
        for name, ((x0, y0, x1, y1), mask) in each.items():
            local = peak[y0:y1, x0:x1]
            pixels = int(mask.sum())
            edge_counts[name] = {
                "pixels": pixels,
                "changed": int((mask & (local > floor)).sum()),
                "strong": int((mask & (local > thresholds.strong_delta)).sum()),
            }
            if budget is None:
                continue
            differing = int((mask & (local > budget.delta)).sum())
            allowed = budget.allowed(pixels)
            if differing > allowed:
                footprint = reference_map.objects[name]
                findings.append(
                    Finding(
                        code=EDGE_OVER_BUDGET,
                        frame=frame,
                        t=t,
                        box=tight_box(mask, (x0, y0)),
                        object=name,
                        source_line=footprint.source,
                        detail={
                            "edge_pixels": pixels,
                            "differing": differing,
                            "allowed": allowed,
                        },
                    )
                )
    clusters = tuple(
        g for g in label_groups(members, peak, thresholds, keep_masks=True) if g.cluster
    )
    for group in clusters:
        name, source = _drawn_by(group, (reference_map, candidate_map))
        findings.append(
            Finding(
                code=CLUSTER,
                frame=frame,
                t=t,
                box=group.box,
                object=name,
                source_line=source,
                detail={"pixels": group.pixels, "peak": group.peak},
            )
        )
    return Structural(tuple(findings), clusters, edges, edge_counts, counts)


def flicker(
    previous: np.ndarray,
    current: np.ndarray,
    changed: ChangedSet,
    at_previous: Mapping[str, Footprint],
    at_current: Mapping[str, Footprint],
    *,
    exempt: np.ndarray | None = None,
    thresholds: Thresholds = Thresholds(),
) -> tuple[Group, ...] | None:
    """Clusters that change between two consecutive frames of one sequence where no
    object that changed draws.

    Every changed or undecided object's footprint at either frame (its mask, or its
    box where only the box is known; grown by its ``grow``) is taken out, and so are
    the ``exempt`` pixels (the reference's edges, in two-renderer mode). Any
    difference makes a pixel changed. An id with no footprint in either mapping is
    taken as drawn at neither frame. ``None`` when the camera changed: the pair is
    not judged.
    """
    if changed.camera:
        return None
    peak = peak_difference(previous, current, "consecutive frames")
    moving = changed.ids | changed.undecided
    footprints = [
        found[name]
        for found in (at_previous, at_current)
        for name in sorted(moving)
        if name in found
    ]
    members = (peak > 0) & ~footprints_union(footprints, peak.shape, grow=True)
    if exempt is not None:
        members &= ~exempt
    return tuple(
        g for g in label_groups(members, peak, thresholds, keep_masks=True) if g.cluster
    )
