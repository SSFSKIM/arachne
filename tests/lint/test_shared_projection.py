"""A leaf's points are projected once per read: the box its draw paints and
its points on the screen come from one pass through the camera's pre-display
code, where the camera traces a path with the engine's own code, which makes
that same pass. Both come out as reading them apart gives them.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.camera.camera import Camera
from arachne.camera.projection import member_box, pixel_points, record_view
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.lint.parts import PartsCache
from arachne.timeline.draw import Drawing
from arachne.timeline.routes import _resolve

from . import memo_scenes
from .test_leaf_memo import _is_probe

SCENES = Path(memo_scenes.__file__)
FPS = 15


def _projections(monkeypatch) -> list[int]:
    """Each pass of the probe's points through a camera's pre-display code,
    from now on.
    """
    passes: list[int] = []
    inside = [0]
    for kind in (Camera, ThreeDCamera):
        engine = kind.transform_points_pre_display

        def counted(self, mob, points, _engine=engine):
            # A ThreeDCamera's pass runs Camera's within it: one pass.
            if _is_probe(mob) and not inside[0]:
                passes.append(1)
            inside[0] += 1
            try:
                return _engine(self, mob, points)
            finally:
                inside[0] -= 1

        monkeypatch.setattr(kind, "transform_points_pre_display", counted)
    return passes


def _read_fresh(session, t: Fraction):
    record = session.record
    resolved = _resolve(record, t)
    return PartsCache(memo=False).build(
        record, t, resolved.play, resolved.capture, answer="recorded", target=(96, 54)
    )


@pytest.mark.parametrize(
    ("scene", "passes"),
    [
        ("HeldProbe2D", 1),
        ("HeldProbe3D", 1),
        ("MovingFrameShifts", 1),
        ("ReplacedProjection", 1),
        # Its own path trace: traced as it traces, and placed apart.
        ("ReplacedPathTrace", 2),
    ],
)
def test_a_leaf_s_points_are_projected_once_per_read(
    session_of, monkeypatch, scene, passes
):
    session = session_of(scene, SCENES, frame_rate=FPS)
    counted = _projections(monkeypatch)
    _read_fresh(session, Fraction(2, FPS))
    assert len(counted) == passes


@pytest.mark.parametrize(
    "scene",
    [
        "HeldProbe2D",
        "HeldProbe3D",
        "TurningProbe3D",
        "ProbeFixedInFrame",
        "FixedOrientationCentreMoves",
        "MovingFrameWidens",
        "CameraFocalDistance",
        "ReplacedProjection",
        "ReplacedPathTrace",
    ],
)
def test_a_leaf_is_placed_as_reading_its_ink_and_points_apart_places_it(
    session_of, scene
):
    from arachne.camera.projection import member_place

    session = session_of(scene, SCENES, frame_rate=FPS)
    record = session.record
    camera = record._scene.renderer.camera
    compared = 0
    for k in (2, 9):
        capture = _resolve(record, Fraction(k, FPS)).capture
        with Drawing(record) as drawing:
            drawing.show(capture, capture.draw_list())
            view = record_view(record)
            for root in capture.draw_list():
                for mob in root.family_members_with_points():
                    points = np.asarray(mob.points, dtype=float)
                    apart = member_box(camera, view, mob, ink=True, home=True)
                    placed = member_place(camera, view, mob)
                    if apart is None:
                        assert placed is None
                        continue
                    assert placed is not None
                    ink, pixels = placed
                    assert ink == apart
                    expected = pixel_points(camera, view, mob, points)
                    assert pixels.dtype == expected.dtype
                    assert pixels.tobytes() == expected.tobytes()
                    compared += 1
    assert compared >= 2
