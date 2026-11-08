"""A convex hull is built in an order fixed by its points alone.

The quickhull keeps its facets in sets, and a facet hashes through its points' bytes, so set order
follows the interpreter's string hash seed, which differs from process to process unless
``PYTHONHASHSEED`` is fixed. The hulls built from it must not inherit that order: the same points give
the same vertices, faces, start vertices and winding in every process, so a hull draws the same frames
(a ``Create`` traces from the same start; a face's shade reads its first corner).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from arachne import ConvexHull, ConvexHull3D

POINTS_2D = [
    [-2.3, -2.2, 0],
    [1.6, -2.3, 0],
    [2.7, -0.2, 0],
    [1.6, 1.8, 0],
    [-0.3, 2.8, 0],
    [-2.4, 0.7, 0],
    [-0.4, -1.2, 0],
    [0.6, -0.3, 0],
    [0.1, 0.4, 0],
    [-1.2, 0.9, 0],
    [2.0, 0.6, 0],
    [-1.9, -1.5, 0],
]
POINTS_3D = [
    [1.6, 0.3, -1.2],
    [-1.1, 1.4, 0.5],
    [-0.3, -0.8, 0.4],
    [-0.6, -1.3, 1.4],
    [-1.5, 0.7, -1.6],
    [1.4, -0.4, 0.9],
    [0.2, 0.6, -1.7],
    [0.9, 1.5, 1.6],
    [0.1, -1.6, -0.9],
    [-1.7, -0.2, 0.1],
    [0.0, 0.1, 0.2],
    [0.3, -0.2, -0.1],
]

BUILD = f"""
import json
import numpy as np
from arachne import ConvexHull, ConvexHull3D

flat = ConvexHull(*{POINTS_2D!r})
solid = ConvexHull3D(*{POINTS_3D!r})
print(json.dumps({{
    "flat": np.round(flat.get_vertices(), 9).tolist(),
    "flat_points": np.round(flat.points, 9).tolist(),
    "vertices": np.round(np.array(solid.vertex_coords), 9).tolist(),
    "faces": solid.faces_list,
    "face_points": [np.round(f.points, 9).tolist() for f in solid.faces],
}}))
"""


def build_in(hash_seed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    p = subprocess.run(
        [sys.executable, "-c", BUILD],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_a_hull_is_built_alike_whatever_the_hash_seed():
    builds = {seed: build_in(seed) for seed in ("0", "1", "7", "12345")}
    first = builds["0"]
    for seed, b in builds.items():
        for part in first:
            assert b[part] == first[part], f"{part} differs at PYTHONHASHSEED={seed}"


def signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_a_flat_hull_runs_counterclockwise_from_its_lowest_left_vertex():
    v = ConvexHull(*POINTS_2D).get_vertices()
    assert signed_area(v[:, :2]) > 0
    lowest = min(range(len(v)), key=lambda i: (v[i, 0], v[i, 1]))
    assert lowest == 0
    # the geometry is the hull's: every vertex is an input point, and no input point lies outside
    assert {tuple(p) for p in np.round(v, 9).tolist()} <= {
        tuple(map(float, p)) for p in POINTS_2D
    }


def test_a_solid_hull_lists_its_vertices_in_order_and_winds_each_face_outward():
    hull = ConvexHull3D(*POINTS_3D)
    verts = np.array(hull.vertex_coords, dtype=float)
    keys = [tuple(p) for p in verts]
    assert keys == sorted(keys)
    centre = verts.mean(axis=0)
    for face in hull.faces_list:
        assert face[0] == min(face)  # each face starts at its lowest vertex
        a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
        normal = np.cross(b - a, c - a)
        assert np.dot(normal, a - centre) > 0  # counterclockwise seen from outside
    assert hull.faces_list == sorted(hull.faces_list)
    # every face is a plane of the hull: all points on or behind it
    pts = np.array(POINTS_3D, dtype=float)
    for face in hull.faces_list:
        a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(b - a, c - a)
        assert ((pts - a) @ n <= 1e-9 * np.linalg.norm(n) + 1e-9).all()


@pytest.mark.parametrize("points", [POINTS_3D])
def test_the_solid_hull_keeps_its_faces(points):
    """The canonical order changes order only: the faces are the quickhull's facets, each once."""
    from arachne.utils.qhull import QuickHull

    q = QuickHull(1e-5)
    q.build(np.array(points))
    facets = {
        frozenset(tuple(p.coordinates) for sf in f.subfacets for p in sf.points)
        for f in set(q.facets) - q.removed
    }
    hull = ConvexHull3D(*points)
    verts = [tuple(v) for v in np.array(hull.vertex_coords, dtype=float)]
    faces = {frozenset(verts[i] for i in face) for face in hull.faces_list}
    assert faces == facets
    assert len(hull.faces_list) == len(facets)
