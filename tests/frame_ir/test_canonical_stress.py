"""Geometry at the snap's conditioning edge draws no further from Cairo snapped
than from its points as made.

Each stress scene (``stress_scenes``) derives a direction or a decision from
points a grid step apart or closer: handles on or near their anchors under
thick joins, near-degenerate cubics, tiny circles and dots, a sphere's poles,
a cone's apex. Its pixels more than 24 levels off Cairo's frame are counted
apart in two regions:

- The seams: a 1-pixel band around every face's and path's outline, where
  faces share an edge and the snap settles which of the two antialiases over
  the other (class E/C). Snapped, they are no more than unsnapped by 4%, and a
  few pixels.
- Everywhere else, the discs around degenerate corners (a pole, an apex)
  included: snapped, they are no more than unsnapped by a few pixels. A
  quantity the snap broke (a normal, a plane, a tangent) moves thousands; the
  degenerate-corner rule switched off moves a sphere's poles past the bound.

How a degenerate corner is lit is proved directly: its decided normal is zero,
and the core lights it by its face's own normal.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from arachne.frame_ir.canonical import Canonical
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.xf import Delta, Record

from ..timeline.conftest import render
from .stress_scenes import FPS, SCENES, ConeApex, SpherePoles

pytest.importorskip("arachne_core")

FULL = {"pixel_width": 854, "pixel_height": 480}
#: How far from a degenerate corner (pixels) the seam band is cut away, so
#: the pixels around a pole or an apex stay in the strict count.
CORNER_RADIUS = 3.0


class StressReport(UserWarning):
    """A stress scene's counts (off Cairo by region; its degenerate corners),
    reported on every run, passing or not: only CI holds the other platforms'
    Cairo (macOS's among them), and their margins are read from these lines.
    """


def _far(frame: np.ndarray, cairo: np.ndarray) -> np.ndarray:
    """Where ``frame`` is more than 24 levels off Cairo's in any channel."""
    d = np.abs(frame[..., :3].astype(int) - cairo[..., :3].astype(int)).max(-1)
    return d > 24


def _pixels(camera, points: np.ndarray) -> np.ndarray:
    """``points`` where ``camera`` puts them, in pixels (x, y)."""
    points = np.asarray(points, dtype=float)
    if hasattr(camera, "project_points"):
        camera.reset_rotation_matrix()
        points = camera.project_points(points)
    x = (points[:, 0] / camera.frame_width + 0.5) * camera.pixel_width
    y = (0.5 - points[:, 1] / camera.frame_height) * camera.pixel_height
    return np.stack([x, y], axis=1)


def _seams(scene, shape: tuple[int, int]) -> np.ndarray:
    """The seam band of ``scene`` as drawn last: a pixel around every outline
    of what it shows, less the discs around its degenerate corners (a cubic
    whose ends coincide: a pole's, an apex's).
    """
    camera = scene.renderer.camera
    height, width = shape
    marks = np.zeros(shape, bool)
    corners = []
    ts = np.linspace(0.0, 1.0, 65)[:, None]
    for mobject in scene.mobjects:
        for leaf in mobject.family_members_with_points():
            points = np.asarray(leaf.points, dtype=float)
            cubics = points[: len(points) // 4 * 4].reshape(-1, 4, 3)
            for a, h1, h2, b in cubics:
                curve = (
                    (1 - ts) ** 3 * a
                    + 3 * (1 - ts) ** 2 * ts * h1
                    + 3 * (1 - ts) * ts**2 * h2
                    + ts**3 * b
                )
                x, y = np.floor(_pixels(camera, curve)).astype(int).T
                inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                marks[y[inside], x[inside]] = True
                if np.abs(a - b).max() < 1e-6:
                    corners.append(a)
    padded = np.pad(marks, 1)
    band = np.zeros(shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            band |= padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    yy, xx = np.mgrid[0:height, 0:width]
    if corners:
        for x, y in _pixels(camera, np.array(corners)):
            band &= (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 > CORNER_RADIUS**2
    return band


@pytest.mark.parametrize("scene", SCENES, ids=[s.__name__ for s in SCENES])
def test_snapped_is_no_further_from_cairo_than_unsnapped(scene, monkeypatch):
    _, cairo = render(scene, FPS, draw_path="ir_cairo", **FULL)
    shown, snapped = render(scene, FPS, draw_path="core", **FULL)
    # The core handed the deltas as made: the reference, past the guard.
    monkeypatch.setattr(Canonical, "__call__", lambda self, delta: delta)
    draw = CorePainter.draw
    monkeypatch.setattr(
        CorePainter, "draw", lambda self, delta: draw(self, delta, _unguarded=True)
    )
    _, unsnapped = render(scene, FPS, draw_path="core", **FULL)
    seams = _seams(shown, cairo[-1].shape[:2])
    counts = {}
    for name, frames in (("snapped", snapped), ("unsnapped", unsnapped)):
        far = _far(frames[-1], cairo[-1])
        counts[name] = (int((far & ~seams).sum()), int((far & seams).sum()))
    split = (
        f"{scene.__name__}: off Cairo, strict snapped {counts['snapped'][0]}"
        f" unsnapped {counts['unsnapped'][0]}; seams (E/C, {int(seams.sum())} px)"
        f" snapped {counts['snapped'][1]} unsnapped {counts['unsnapped'][1]}"
    )
    warnings.warn(split, StressReport, stacklevel=1)
    assert counts["snapped"][0] <= counts["unsnapped"][0] + 4, split
    assert counts["snapped"][1] <= counts["unsnapped"][1] * 1.04 + 8, split


def _kept_painters(monkeypatch) -> list[CorePainter]:
    """Every core painter made from now on, each keeping the deltas it sent."""
    made: list[CorePainter] = []
    init = CorePainter.__init__

    def keeping(self, **kwargs):
        init(self, **{**kwargs, "keep": True})
        made.append(self)

    monkeypatch.setattr(CorePainter, "__init__", keeping)
    return made


def _newell(points: list[tuple[float, ...]]) -> tuple[float, ...]:
    """A face's own unit normal, as the core finds it (``depth::newell`` over
    the points less the first, normalised), in its order of operations.
    """
    first = points[0]
    n = [0.0, 0.0, 0.0]
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        a = (a[0] - first[0], a[1] - first[1], a[2] - first[2])
        b = (b[0] - first[0], b[1] - first[1], b[2] - first[2])
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) ** 0.5
    assert norm > 0.0, "a degenerate face whose own normal cancels"
    return (n[0] / norm, n[1] / norm, n[2] / norm)


def _faces(record: Record, points: np.ndarray) -> list[np.ndarray]:
    """The faces a path or face-group record draws, placed."""
    place = record.fields.get("place")
    if place is not None:
        m = np.reshape(np.asarray(place, dtype=float), (3, 4))
        points = points @ m[:, :3].T + m[:, 3]
    if record.kind == "path":
        return [points]
    count = len(record.fields["uv"])
    size = len(points) // count
    return [points[i * size : (i + 1) * size] for i in range(count)]


def _degenerate(face: np.ndarray, grid: float) -> list[int]:
    """Which of the face's two lit corners (CE's first point and point
    ``(n - 1) // 6 * 3``) are degenerate: a point CE's normal there is found
    from lies less than ``grid`` off it in every coordinate.
    """
    n = len(face)
    if n < 8:  # two anchors or fewer: CE lights the face UP, not by a corner
        return []
    found = []
    for k, i in enumerate((0, (n - 1) // 6 * 3)):
        back = i - 3 if i > 2 else n - 4
        on = i + 3 if i < n - 3 else 3
        short = (np.abs(face[on] - face[i]).max(), np.abs(face[back] - face[i]).max())
        if min(short) < grid:
            found.append(k)
    return found


def _lit_by_newell(delta: Delta, shapes: dict[bytes, np.ndarray], grid: float):
    """``delta`` with each degenerate corner's normal passed as its face's own
    normal; and how many degenerate corners it set, each checked decided zero.
    """
    found = Delta(**{**vars(delta), "set": []})
    corners = 0
    for slot, record in delta.set:
        normals = record.fields.get("normals")
        if normals is None or record.kind not in ("path", "face_group"):
            found.set.append((slot, record))
            continue
        normals = [tuple(n) for n in normals]
        faces = _faces(record, shapes[record.fields["shape"]])
        for f, face in enumerate(faces):
            for k in _degenerate(face, grid):
                assert not any(normals[2 * f + k]), (slot, f, k, normals[2 * f + k])
                normals[2 * f + k] = _newell([tuple(p) for p in face])
                corners += 1
        found.set.append(
            (slot, Record(record.kind, {**record.fields, "normals": normals}))
        )
    return found, corners


@pytest.mark.parametrize("scene", [SpherePoles, ConeApex], ids=lambda s: s.__name__)
def test_a_degenerate_corner_is_decided_zero_and_lit_by_its_face(scene, monkeypatch):
    """At a corner a grid step from its neighbour or closer (a pole, an apex)
    the writer decides the zero normal, and the core lights the face there by
    the face's own normal: the frame the scene's core render draws is, byte
    for byte, the same records drawn with those normals passed.
    """
    import arachne_core

    painters = _kept_painters(monkeypatch)
    render(scene, FPS, draw_path="core", **FULL)
    drawn = [p for p in painters if p.deltas]
    assert drawn
    corners = 0
    for painter in drawn:
        reference = arachne_core.Stream()
        shapes: dict[bytes, np.ndarray] = {}
        grid = None
        for delta in painter.deltas:
            shapes.update((shape.key, shape.data) for shape in delta.upload)
            if delta.view is not None:
                grid = delta.view["input_grid"]
            assert grid is not None
            lit, found = _lit_by_newell(delta, shapes, grid)
            corners += found
            reference.apply(
                arachne_core.FrameDelta.from_bytes(lit.to_core().to_bytes())
            )
        view = painter.state.view
        size = (view["pixel_width"], view["pixel_height"])
        made = np.array(painter.core.draw(*size, threads=1, overlay=False))
        passed = np.array(reference.draw(*size, threads=1, overlay=False))
        differ = np.abs(made.astype(int) - passed.astype(int)).max(-1)
        assert not differ.any(), int((differ > 0).sum())
    # Not vacuous: the scene has its degenerate corners.
    warnings.warn(
        f"{scene.__name__}: {corners} degenerate corners", StressReport, stacklevel=1
    )
    assert corners > 0, f"{scene.__name__}: no degenerate corner"
