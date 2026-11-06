"""What one lint does per instant, not per leaf: the work of reading a frame
that its leaves share is done once for the instant. Each test counts the
operation; none times one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arachne.camera import three_d_camera
from arachne.camera.projection import _rotate
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.lint import lint

from . import read_scenes

SCENES = Path(read_scenes.__file__)


def _rotations_made(monkeypatch) -> list[float]:
    """The angles a ThreeDCamera's rotation matrix is made from, from now on:
    two for every matrix made.
    """
    made: list[float] = []
    about_z = three_d_camera.rotation_about_z

    def counted(angle):
        made.append(angle)
        return about_z(angle)

    monkeypatch.setattr(three_d_camera, "rotation_about_z", counted)
    return made


def test_a_held_3d_camera_s_rotation_is_made_once_for_the_lint(session_of, monkeypatch):
    """Every leaf, at every instant, is projected through the one rotation
    a camera that never turns holds: it is not made again per leaf.
    """
    session = session_of("HeldSquares3D", SCENES, frame_rate=15)
    made = _rotations_made(monkeypatch)
    result = lint(session)
    assert len(result.instants) > 1
    assert len(made) / 2 <= 1


def test_a_turning_3d_camera_s_rotation_is_made_once_per_lint_instant(
    session_of, monkeypatch
):
    """The camera turns between every two instants: each is projected
    through its own rotation, made once for all its leaves.
    """
    session = session_of("TurningSquares3D", SCENES, frame_rate=15)
    made = _rotations_made(monkeypatch)
    result = lint(session)
    assert len(made) / 2 == len(result.instants)


def test_a_projection_s_rotation_is_made_again_when_anything_it_reads_moved(
    monkeypatch,
):
    """Made again when the camera's angles moved or something else set its
    matrix since; never kept for a camera that makes its matrix its own way.
    """
    camera = ThreeDCamera()
    made = _rotations_made(monkeypatch)
    _rotate(camera)
    _rotate(camera)
    assert len(made) == 2
    camera.set_phi(1.0)
    _rotate(camera)
    assert len(made) == 4
    camera.rotation_matrix = np.identity(3)
    _rotate(camera)
    assert len(made) == 6
    assert np.array_equal(camera.rotation_matrix, camera.generate_rotation_matrix())

    class OwnRotation(ThreeDCamera):
        def generate_rotation_matrix(self):
            return super().generate_rotation_matrix()

    own = OwnRotation()
    made.clear()
    _rotate(own)
    _rotate(own)
    assert len(made) == 4
