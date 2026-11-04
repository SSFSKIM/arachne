"""Plane geometry the instant rules measure with, in home-frame units.

Boxes are ``(x_min, y_min, x_max, y_max)``. Signed distances are positive
outside and negative inside. Pairwise distances are taken in blocks, so a
dense trace against a long outline costs bounded memory.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..parts import Box, Leaf

#: The spacing a stroke is sampled at along its curves; never fewer than
#: ``FEWEST_SAMPLES`` per curve, nor more than ``MOST_SAMPLES``.
SAMPLE_STEP = 0.02
FEWEST_SAMPLES = 8
MOST_SAMPLES = 512
#: Samples per curve of a glyph's outline when its ink is traced.
OUTLINE_SAMPLES = 8
#: Two points closer than this are where one curve hands on to the next.
_JOINED = 1e-6
#: The working set one block of a pairwise distance may take, in bytes.
_BLOCK = 64 * 1024 * 1024


def _cubics(leaf: Leaf) -> np.ndarray | None:
    """The leaf's whole cubic curves, ``(curves, 4, 2)``, or ``None``."""
    if leaf.curve_size != 4 or len(leaf.points) < 4:
        return None
    usable = len(leaf.points) - len(leaf.points) % 4
    return leaf.points[:usable].reshape(-1, 4, 2)


def _bernstein(t: np.ndarray) -> np.ndarray:
    t = t.reshape(-1, 1)
    u = 1 - t
    return np.hstack([u**3, 3 * t * u**2, 3 * t**2 * u, t**3])


def stroke_samples(leaf: Leaf) -> np.ndarray:
    """Points along the leaf's curves about :data:`SAMPLE_STEP` apart (by
    its longest curve's control polygon), ``(n, 2)``; its points as they
    are when they are no cubic curves.
    """
    cubics = _cubics(leaf)
    if cubics is None:
        return leaf.points
    polygon = np.linalg.norm(np.diff(cubics, axis=1), axis=2).sum(axis=1)
    count = math.ceil(float(polygon.max()) / SAMPLE_STEP)
    count = int(np.clip(count, FEWEST_SAMPLES, MOST_SAMPLES))
    basis = _bernstein(np.linspace(0.0, 1.0, count))
    return np.einsum("sk,ckd->csd", basis, cubics).reshape(-1, 2)


def outline_edges(leaf: Leaf) -> tuple[np.ndarray, np.ndarray] | None:
    """The leaf's outline as straight edges ``(starts, ends)``, each curve
    cut into :data:`OUTLINE_SAMPLES`, each run of joined curves closed on
    itself; ``None`` when it is no cubic curves.
    """
    cubics = _cubics(leaf)
    if cubics is None:
        return None
    basis = _bernstein(np.arange(OUTLINE_SAMPLES) / OUTLINE_SAMPLES)
    gaps = np.linalg.norm(cubics[1:, 0] - cubics[:-1, 3], axis=1)
    starts, ends = [], []
    for run in np.split(cubics, np.flatnonzero(gaps > _JOINED) + 1):
        ring = np.einsum("sk,ckd->csd", basis, run).reshape(-1, 2)
        starts.append(ring)
        ends.append(np.roll(ring, -1, axis=0))
    return np.vstack(starts), np.vstack(ends)


def _rows(columns: int, per_cell: int) -> int:
    return max(1, min(2048, _BLOCK // (max(columns, 1) * per_cell)))


def signed_to_outline(
    points: np.ndarray, edges: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Each point's distance to the outline, negative inside the area it
    fills. Inside is a nonzero winding number, as Cairo fills: a counter
    wound against its letter is a hole, two contours wound alike over one
    another are filled.
    """
    a, b = edges
    ab = b - a
    lengths = np.maximum((ab * ab).sum(axis=1), 1e-18)
    rising, falling = ab[:, 1] > 0, ab[:, 1] < 0
    out = np.empty(len(points))
    step = _rows(len(a), 6 * 8)
    for start in range(0, len(points), step):
        p = points[start : start + step]
        rel = p[:, None, :] - a[None, :, :]
        # Where each edge crosses the point's height, the cross product says
        # which side of the edge the point is on.
        cross = ab[None, :, 0] * rel[:, :, 1] - ab[None, :, 1] * rel[:, :, 0]
        y = p[:, 1:2]
        up = (a[None, :, 1] <= y) & (b[None, :, 1] > y) & rising & (cross > 0)
        down = (a[None, :, 1] > y) & (b[None, :, 1] <= y) & falling & (cross < 0)
        winding = up.sum(axis=1) - down.sum(axis=1)
        along = np.clip((rel * ab[None]).sum(axis=2) / lengths, 0.0, 1.0)
        rel -= along[:, :, None] * ab[None]
        distance = np.sqrt((rel * rel).sum(axis=2).min(axis=1))
        out[start : start + step] = np.where(winding != 0, -distance, distance)
    return out


def signed_to_box(points: np.ndarray, box: Box) -> np.ndarray:
    """Each point's distance to ``box``: outside, to its edge; inside, how
    deep, negated.
    """
    low = np.array(box[:2])
    high = np.array(box[2:])
    beyond = np.maximum(np.maximum(low - points, points - high), 0.0)
    outside = np.linalg.norm(beyond, axis=1)
    depth = np.minimum(points - low, high - points).min(axis=1)
    inside = outside == 0
    return np.where(inside, -np.clip(depth, 0.0, None), outside)


def box_gap(a: Box, b: Box) -> float:
    """The gap between two boxes along the axis they are furthest apart on;
    below zero, how far they overlap.
    """
    return float(max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3]))


def least_box_gap(first: Sequence[Box], second: Sequence[Box]) -> float:
    """The smallest :func:`box_gap` between a box of ``first`` and one of
    ``second``.
    """
    a = np.array(first, dtype=float)[:, None, :]
    b = np.array(second, dtype=float)[None, :, :]
    gaps = np.stack(
        [
            a[..., 0] - b[..., 2],
            b[..., 0] - a[..., 2],
            a[..., 1] - b[..., 3],
            b[..., 1] - a[..., 3],
        ]
    )
    return float(gaps.max(axis=0).min())


def nearest(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """For each of ``points``, the distance to the nearest of ``others``."""
    out = np.empty(len(points))
    step = _rows(len(others), 3 * 8)
    for start in range(0, len(points), step):
        block = points[start : start + step, None, :] - others[None, :, :]
        out[start : start + step] = np.sqrt((block * block).sum(axis=2)).min(axis=1)
    return out


def closest(points: np.ndarray, others: np.ndarray) -> float:
    """The least distance between a point of one set and one of the other."""
    if not len(points) or not len(others):
        return math.inf
    return float(nearest(points, others).min())


def anchor_box(leaves: Sequence[Leaf]) -> Box | None:
    """The box of the leaves' anchors (the ends of their curves; a lone
    point for itself), as a mobject's corners are found.
    """
    anchors = []
    for leaf in leaves:
        points, n = leaf.points, leaf.curve_size
        if len(points) == 1 or n < 1:
            anchors.append(points)
        elif len(points):
            anchors.extend((points[::n], points[n - 1 :: n]))
    if not anchors:
        return None
    every = np.vstack(anchors)
    low, high = every.min(axis=0), every.max(axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def union(boxes: Sequence[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def within(inner: Box, outer: Box) -> bool:
    """Whether ``inner`` lies strictly inside ``outer``."""
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def meets(box: Box, frame: Box) -> bool:
    """Whether ``box`` touches the frame at all."""
    return (
        box[2] >= frame[0]
        and box[0] <= frame[2]
        and box[3] >= frame[1]
        and box[1] <= frame[3]
    )
