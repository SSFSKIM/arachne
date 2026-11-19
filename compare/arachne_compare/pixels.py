"""Pixel operations every pass shares: differences, groups, footprints and edges."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
from scipy import ndimage

from .model import Box, Footprint, Group, Thresholds

EIGHT = np.ones((3, 3), dtype=bool)
FOUR = ndimage.generate_binary_structure(2, 1)


def check_frame(frame: np.ndarray, what: str) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(
            f"{what}: a frame is a uint8 array of height x width x 3 or 4, "
            f"not {frame.dtype} {frame.shape}"
        )
    return frame


def peak_difference(a: np.ndarray, b: np.ndarray, what: str = "frames") -> np.ndarray:
    """Per pixel, the largest difference in any channel.

    Frames of different sizes, or with different channels (RGB against RGBA), are
    refused, never compared.
    """
    a = check_frame(a, what)
    b = check_frame(b, what)
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            f"{what}: sizes differ ({a.shape[1]} x {a.shape[0]} against "
            f"{b.shape[1]} x {b.shape[0]}); frames of different sizes are not compared"
        )
    if a.shape[2] != b.shape[2]:
        raise ValueError(
            f"{what}: one frame has {a.shape[2]} channels, the other {b.shape[2]}; "
            "frames with different channels are not compared"
        )
    difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return difference.max(axis=2).astype(np.uint8)


def label_groups(
    members: np.ndarray,
    peak: np.ndarray,
    thresholds: Thresholds,
    *,
    keep_masks: bool = False,
) -> list[Group]:
    """The 8-connected groups of ``members``, each marked a cluster or not.

    A cluster covers more than ``cluster_share`` of the frame and holds a pixel that
    changed by more than ``strong_delta``. Groups come largest first.
    """
    if not members.any():
        return []
    labels, count = ndimage.label(members, structure=EIGHT)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    peaks = ndimage.maximum(peak, labels, index=np.arange(1, count + 1))
    slices = ndimage.find_objects(labels)
    limit = thresholds.cluster_pixels(*members.shape)
    groups = []
    for index, where in enumerate(slices, start=1):
        rows, cols = where
        box = (cols.start, rows.start, cols.stop, rows.stop)
        pixels = int(sizes[index])
        strongest = int(peaks[index - 1])
        mask = labels[where] == index if keep_masks else None
        cluster = pixels > limit and strongest > thresholds.strong_delta
        groups.append(Group(box, pixels, strongest, cluster, mask))
    groups.sort(key=lambda g: (-g.pixels, g.box))
    return groups


def boxes_meet(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def grown_box(box: Box, by: int, size: tuple[int, int]) -> Box:
    height, width = size
    return (
        max(box[0] - by, 0),
        max(box[1] - by, 0),
        min(box[2] + by, width),
        min(box[3] + by, height),
    )


def paint(target: np.ndarray, footprint: Footprint, *, grow: bool = False) -> None:
    """Set ``target`` true where the footprint covers, or over its box with no mask."""
    size = target.shape
    by = math.ceil(footprint.grow) if grow else 0
    x0, y0, x1, y1 = footprint.box
    if footprint.mask is None:
        gx0, gy0, gx1, gy1 = grown_box(footprint.box, by, size)
        target[gy0:gy1, gx0:gx1] = True
        return
    if by == 0:
        target[y0:y1, x0:x1] |= footprint.mask
        return
    gx0, gy0, gx1, gy1 = grown_box(footprint.box, by, size)
    local = np.zeros((gy1 - gy0, gx1 - gx0), dtype=bool)
    local[y0 - gy0 : y1 - gy0, x0 - gx0 : x1 - gx0] = footprint.mask
    local = ndimage.binary_dilation(
        local, structure=np.ones((2 * by + 1, 2 * by + 1), bool)
    )
    target[gy0:gy1, gx0:gx1] |= local


def tight_box(mask: np.ndarray, origin: tuple[int, int] = (0, 0)) -> Box:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    x, y = origin
    return (
        x + int(cols[0]),
        y + int(rows[0]),
        x + int(cols[-1]) + 1,
        y + int(rows[-1]) + 1,
    )


def edge_ring(height: int) -> int:
    """How far beyond the boundary pixels an edge reaches: one pixel per 1080 of height."""
    return height // 1080 if height >= 1080 else 0


def object_edges(
    footprint: Footprint, size: tuple[int, int]
) -> tuple[Box, np.ndarray] | None:
    """One footprint's edge pixels, as a region of the frame and a mask over it.

    The boundary pixels are the footprint's pixels with one of their four neighbours
    outside it; the frame's own border is not a boundary. From a height of 1080 every
    pixel within ``height // 1080`` of a boundary pixel, on either side, is an edge too.
    A footprint known only by its box has no edges.
    """
    if footprint.mask is None or not footprint.mask.any():
        return None
    ring = edge_ring(size[0])
    region = grown_box(footprint.box, ring + 1, size)
    rx0, ry0, rx1, ry1 = region
    x0, y0, x1, y1 = footprint.box
    local = np.zeros((ry1 - ry0, rx1 - rx0), dtype=bool)
    local[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = footprint.mask
    # Pixels beyond the frame count as inside: a clip at the frame's border is no edge.
    pad = (
        (int(ry0 == 0), int(ry1 == size[0])),
        (int(rx0 == 0), int(rx1 == size[1])),
    )
    padded = np.pad(local, pad, mode="constant", constant_values=True)
    inner = ndimage.binary_erosion(padded, structure=FOUR, border_value=0)
    inner = inner[
        pad[0][0] : inner.shape[0] - pad[0][1], pad[1][0] : inner.shape[1] - pad[1][1]
    ]
    boundary = local & ~inner
    if ring:
        boundary = ndimage.binary_dilation(
            boundary, structure=np.ones((2 * ring + 1, 2 * ring + 1), dtype=bool)
        )
    return region, boundary


def map_edges(
    objects: Mapping[str, Footprint], size: tuple[int, int]
) -> tuple[np.ndarray, dict[str, tuple[Box, np.ndarray]]]:
    """Every footprint's edges, parts included: their union and each object's own."""
    union = np.zeros(size, dtype=bool)
    each: dict[str, tuple[Box, np.ndarray]] = {}
    for name in sorted(objects):
        edges = object_edges(objects[name], size)
        if edges is None:
            continue
        (x0, y0, x1, y1), mask = edges
        union[y0:y1, x0:x1] |= mask
        each[name] = edges
    return union, each


def footprints_union(
    footprints: Iterable[Footprint], size: tuple[int, int], *, grow: bool
) -> np.ndarray:
    union = np.zeros(size, dtype=bool)
    for footprint in footprints:
        paint(union, footprint, grow=grow)
    return union
