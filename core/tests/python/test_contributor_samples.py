"""The contributor report's per-sub-sample sequences and the edges the rasterizer reads.

``contributor_samples`` gives each sub-sample's own sequence (``contributors``' ``orders`` are
their distinct ones); ``contributor_edges`` gives, per leaf drawn with exact area, the edges
each of its layers is rasterized from. The exact nonzero-winding area of those edges inside a
sub-sample's cell is the cell coverage the core composites, to the byte: a reader of the edges
reproduces the core's coverage.

The frame and view are ``test_contributors``': seen straight down, 10 pixels a unit, the
origin at pixel (80, 45).
"""

from __future__ import annotations

import itertools

import arachne_core as ac
import numpy as np
from test_contributors import DOWN, FRAME, segment, stream_of

# ``composite.rs`` ``SAMPLE_OFFSETS``: sample ``s`` sits at the centre of the 1/16 px cell
# ``offset`` from the pixel's centre; its 1/4 px cell is ``(offset + 8) // 4``.
SAMPLE_OFFSETS = [
    (-8, -7), (-1, -8), (3, -6), (5, -5), (-6, -1), (-4, -4), (0, -3), (6, -2),
    (-7, 2), (-2, 0), (2, 3), (4, 1), (-5, 5), (-3, 4), (1, 7), (7, 6),
]  # fmt: skip


def cell_area(edges, x0: float, y0: float, x1: float, y1: float) -> float:
    """The exact area, in ``0..=1`` of the box, the edges fill by the nonzero rule inside the box
    ``x0..x1`` by ``y0..y1``: between the heights where an edge ends, crosses the box's sides or
    crosses another edge, each edge is linear and their order fixed, so each span where the
    winding is nonzero is a trapezoid clipped to the box. Edges right of the box are left out:
    a span still open at the last edge is filled on to the box's right side.
    """
    near = [e for e in edges if e[1] < y1 and e[3] > y0 and min(e[0], e[2]) < x1]
    cuts = {y0, y1}
    for ax, ay, bx, by, _ in near:
        cuts.update(v for v in (ay, by) if y0 < v < y1)
        for side in (x0, x1):
            if (ax - side) * (bx - side) < 0:
                cuts.add(ay + (side - ax) * (by - ay) / (bx - ax))
    for (ax, ay, bx, by, _), (cx, cy, dx, dy, _) in itertools.combinations(near, 2):
        # Where the two lines meet, if within both edges' heights.
        den = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
        if den == 0:
            continue
        t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / den
        if 0 < t < 1:
            y = ay + t * (by - ay)
            if max(ay, cy) < y < min(by, dy):
                cuts.add(y)
    ys = sorted(v for v in cuts if y0 <= v <= y1)

    def x_at(e, y):
        ax, ay, bx, by, _ = e
        return ax + (y - ay) * (bx - ax) / (by - ay)

    total = 0.0
    for ya, yb in itertools.pairwise(ys):
        if yb <= ya:
            continue
        ym = (ya + yb) / 2
        crossing = sorted(
            (e for e in near if e[1] <= ya and e[3] >= yb), key=lambda e: x_at(e, ym)
        )
        winding, start = 0, None
        for e in crossing:
            before = winding
            winding += e[4]
            if before == 0 and winding != 0:
                start = e
            elif before != 0 and winding == 0:
                width = [
                    min(max(x_at(e, y), x0), x1) - min(max(x_at(start, y), x0), x1)
                    for y in (ya, yb)
                ]
                total += (width[0] + width[1]) / 2 * (yb - ya)
        if (
            winding != 0
        ):  # filled on to the box's right side, where the edges closing it lie beyond
            width = [x1 - min(max(x_at(start, y), x0), x1) for y in (ya, yb)]
            total += (width[0] + width[1]) / 2 * (yb - ya)
    return total / ((x1 - x0) * (y1 - y0))


def one_path(points: np.ndarray, **record) -> ac.Stream:
    delta = ac.FrameDelta()
    delta.set_view(background=(0, 0, 0, 1), **FRAME, **DOWN)
    shape = delta.upload_path(points)
    delta.set_record("path", 1, 11, shape=shape, space="world", **record)
    delta.set_order([("moving", 1, 0)])
    return stream_of(delta)


def circle(radius: float, centre=(0.0, 0.0)) -> np.ndarray:
    """Four cubic quarter arcs."""
    k = 0.5522847498 * radius
    cx, cy = centre
    out = []
    for i in range(4):
        a, b = i * np.pi / 2, (i + 1) * np.pi / 2
        p0 = np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), 0.0])
        p3 = np.array([cx + radius * np.cos(b), cy + radius * np.sin(b), 0.0])
        t0 = np.array([-np.sin(a), np.cos(a), 0.0])
        t3 = np.array([-np.sin(b), np.cos(b), 0.0])
        out += [p0, p0 + k * t0, p3 - k * t3, p3]
    return np.array(out)


def check_cells(stream: ac.Stream, box=(0, 0, 160, 90)) -> int:
    """Every sub-sample of every pixel in ``box``: each leaf's coverage there against the exact
    area its edges fill in the sub-sample's cell, the greatest over its layers (a leaf's entry
    is its run's greatest coverage). Returns the number of (sub-sample, leaf) pairs read.
    """
    report = stream.contributors(*box)
    samples = stream.contributor_samples(*box)
    edges = stream.contributor_edges(*box)
    assert set(samples) == set(report)
    read = 0
    for (x, y), (leaves, _orders, _crossing, _decided) in report.items():
        keys = [(obj, part) for obj, part, *_ in leaves]
        for s, sequence in enumerate(samples[(x, y)]):
            dx, dy = ((o + 8) // 4 for o in SAMPLE_OFFSETS[s])
            cx, cy = x + dx / 4, y + dy / 4
            core = {}
            for i, c in sequence:
                core[keys[i]] = max(core.get(keys[i], 0), c)
            for key in set(keys) | set(edges):
                layers = edges.get(key, [])
                area = max(
                    (
                        cell_area(layer, cx, cy, cx + 0.25, cy + 0.25)
                        for layer in layers
                    ),
                    default=0.0,
                )
                got = core.get(key, 0) / 255
                assert abs(area - got) <= 1 / 510 + 1e-6, ((x, y), s, key, area, got)
                read += 1
    return read


def test_each_sub_sample_s_sequence_is_reported_and_the_orders_are_their_distinct_ones():
    stream = one_path(
        circle(1.0), fill=(1, 0, 0, 0.5), stroke=(0, 0, 1, 0.5), stroke_width=20.0
    )
    report = stream.contributors(60, 25, 100, 65)
    samples = stream.contributor_samples(60, 25, 100, 65)
    assert set(samples) == set(report)
    for at, per_sample in samples.items():
        assert len(per_sample) == 16
        distinct = []
        for seq in per_sample:
            seq = [tuple(e) for e in seq]
            if seq and seq not in distinct:
                distinct.append(seq)
        assert distinct == [[tuple(e) for e in order] for order in report[at][1]], at


def test_a_curved_fill_s_edges_fill_each_cell_by_its_coverage():
    stream = one_path(circle(1.3), fill=(1, 0, 0, 1))
    assert check_cells(stream, (64, 29, 96, 61)) > 0


def test_a_stroke_s_edges_with_its_joins_and_caps_fill_each_cell_by_its_coverage():
    # An open polyline turning twice, mitred and bevelled corners, round caps, 0.3 units wide.
    bend = np.concatenate(
        [
            segment((-1.5, -1.0, 0), (0.0, 0.8, 0)),
            segment((0.0, 0.8, 0), (1.4, -0.6, 0)),
            segment((1.4, -0.6, 0), (0.2, -1.2, 0)),
        ]
    )
    for joint, cap in (("miter", "round"), ("bevel", "butt"), ("round", "square")):
        stream = one_path(
            bend, stroke=(0, 0, 1, 1), stroke_width=30.0, joint=joint, cap=cap
        )
        assert check_cells(stream, (60, 33, 100, 60)) > 0, (joint, cap)


def test_a_dashed_stroke_s_edges_fill_each_cell_by_its_coverage():
    stream = one_path(
        circle(1.2), stroke=(0, 1, 0, 1), stroke_width=20.0, dash=(0.4, 0.5, 0.1)
    )
    assert check_cells(stream, (66, 31, 94, 59)) > 0


def test_a_mesh_has_no_edges():
    delta = ac.FrameDelta()
    delta.set_view(background=(0, 0, 0, 1), **FRAME, **DOWN)
    positions = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0)], dtype=float)
    mesh = delta.upload_mesh(positions, np.array([[0, 1, 2]], dtype=np.uint32))
    delta.set_record("mesh", 1, 11, shape=mesh, fill=(1, 1, 1, 1), space="world")
    delta.set_order([("moving", 1, 0)])
    stream = stream_of(delta)
    assert stream.contributors(70, 35, 90, 55)
    assert stream.contributor_edges(70, 35, 90, 55) == {}
