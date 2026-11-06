"""A lint reads a leaf once for each version of it under each view of the
camera, not once per instant: what it lays once for each version of the leaf,
where the camera puts it once for each version of the leaf and the camera.
Anything a read depends on, changed alone, is read again; a read the memo's
key can't see all of is never kept. Each test counts reads, never times one,
and a lint with the memo answers as one without it
(``test_leaf_memo_answers.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint import parts as parts_module

from . import memo_scenes

SCENES = Path(memo_scenes.__file__)
FPS = 15


def _is_probe(mob) -> bool:
    """Whether ``mob`` is the scene's probe (the session loads the scene
    file as a module of its own, so its class is not this one's import).
    """
    return type(mob).__name__ == "Probe"


def _counted(monkeypatch) -> dict[str, int]:
    """How often, from now on, the probe is read, its lay is worked out (its
    closure stands for that) and its place under the camera.
    """
    counts = {"reads": 0, "lays": 0, "placed": 0}
    leaf = parts_module._Reader.leaf
    closed = parts_module._closed
    place = parts_module.member_place

    def read(self, mob, order, chain=()):
        counts["reads"] += _is_probe(mob)
        return leaf(self, mob, order, chain)

    def lays(mob, points):
        counts["lays"] += _is_probe(mob)
        return closed(mob, points)

    def placed(camera, view, mob):
        counts["placed"] += _is_probe(mob)
        return place(camera, view, mob)

    monkeypatch.setattr(parts_module._Reader, "leaf", read)
    monkeypatch.setattr(parts_module, "_closed", lays)
    monkeypatch.setattr(parts_module, "member_place", placed)
    return counts


def _lint(session_of, monkeypatch, scene: str) -> dict[str, int]:
    session = session_of(scene, SCENES, frame_rate=FPS)
    counts = _counted(monkeypatch)
    lint(session)
    assert counts["reads"] > 2
    return counts


@pytest.mark.parametrize("scene", ["HeldProbe3D", "HeldProbe2D"])
def test_a_held_leaf_under_a_still_camera_is_read_once_for_the_lint(
    session_of, monkeypatch, scene
):
    counts = _lint(session_of, monkeypatch, scene)
    assert (counts["lays"], counts["placed"]) == (1, 1)


def test_under_a_turning_camera_only_where_it_is_placed_is_read_again(
    session_of, monkeypatch
):
    counts = _lint(session_of, monkeypatch, "TurningProbe3D")
    assert counts["lays"] == 1
    assert counts["placed"] == counts["reads"]


@pytest.mark.parametrize(
    "scene",
    ["ProbeShifts", "ProbeFillChanges", "ProbeStrokeWidens", "ProbeStrokeFades"],
)
def test_a_change_to_the_leaf_alone_reads_it_again(session_of, monkeypatch, scene):
    counts = _lint(session_of, monkeypatch, scene)
    assert (counts["lays"], counts["placed"]) == (2, 2)


@pytest.mark.parametrize(
    "scene",
    [
        "CameraPhi",
        "CameraTheta",
        "CameraGamma",
        "CameraFocalDistance",
        "CameraZoom",
        "CameraFrameCentre",
        "ProbeFixedInFrame",
        "FixedOrientationCentreMoves",
        "MovingFrameShifts",
        "MovingFrameWidens",
    ],
)
def test_a_change_to_the_camera_alone_places_the_leaf_again(
    session_of, monkeypatch, scene
):
    counts = _lint(session_of, monkeypatch, scene)
    assert (counts["lays"], counts["placed"]) == (1, 2)


@pytest.mark.parametrize(
    "scene",
    [
        "ProbeWithUpdater",
        "ProbeInUpdatedGroup",
        "RedrawnProbe",
        "SceneUpdaterBeside",
        "ReplacedProjection",
    ],
)
def test_a_read_the_key_cannot_see_all_of_is_never_kept(session_of, monkeypatch, scene):
    """An updater on the leaf or what holds it, a new leaf at every frame, a
    scene updater, or a camera running the scene's code: read every time.
    """
    counts = _lint(session_of, monkeypatch, scene)
    assert counts["lays"] == counts["placed"] == counts["reads"]
