"""A play is evaluated at its exact start, as CE evaluates its first frame:
its parts begun, its updaters handed ``dt == 0`` once, whether the start is
on the frame grid or between two of its frames. Its end hands them
``dt == 0`` once more, as CE's ``update_mobjects(0)`` after a play does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arachne.camera import three_d_camera
from arachne.timeline import recorder as recorder_module
from arachne.timeline.draw import CameraExtras, camera_nodes
from arachne.timeline.record import NodeState

from . import entry_scenes
from .conftest import render

SCENES = Path(entry_scenes.__file__)
NAMES = ["CountedStarts", "CountedStartsOffGrid"]


def _expected(scene_class) -> list[float]:
    """Each play's start and end, three plays one after the other."""
    run = scene_class.run_time
    return sorted(
        [round(i * run, 9) for i in range(3)]
        + [round((i + 1) * run, 9) for i in range(3)]
    )


@pytest.mark.parametrize("fps", [15, 30])
@pytest.mark.parametrize("name", NAMES)
def test_a_render_hands_dt_0_once_at_each_play_s_start(fps, name):
    scene_class = getattr(entry_scenes, name)
    scene, _ = render(scene_class, fps)
    assert sorted(round(t, 9) for t in scene.zero_dt) == _expected(scene_class)


@pytest.mark.parametrize("fps", [15, 30])
@pytest.mark.parametrize("name", NAMES)
def test_a_session_hands_dt_0_once_at_each_play_s_start(session_of, fps, name):
    session = session_of(name, SCENES, frame_rate=fps, probe=False)
    found = session.record._scene.zero_dt
    assert sorted(round(t, 9) for t in found) == _expected(getattr(entry_scenes, name))


def test_a_record_keeps_each_play_s_entry_beside_its_exit(session_of):
    """Each play's entry is the scene at its exact start: between two plays
    nothing moves, so it stands where the play before left every body,
    though the start falls between two frames.
    """
    record = session_of("ArmedOffGridStarts", SCENES, frame_rate=15, probe=False).record
    assert record.entries is not None
    assert record.exits is not None
    assert sorted(record.entries) == sorted(record.exits) == [0, 1, 2]
    for index in (1, 2):
        entry, left = record.entries[index].bodies, record.exits[index - 1].bodies
        assert entry.keys() == left.keys()
        for key, body in entry.items():
            assert np.allclose(body.centre, left[key].centre, atol=1e-9), key


def test_reading_an_entry_leaves_every_node_and_the_camera_s_inputs_as_they_were(
    session_of, monkeypatch
):
    """Reading where each body stands as a play begins changes no node a
    capture versions and none of the camera inputs it records, under a 3D
    camera that turns between plays and during one: no static picture is
    redrawn for it.
    """
    entered = recorder_module.PlayRun.entered
    read = []
    remade: list[bool] = []
    rotations: list[float] = []
    about_z = three_d_camera.rotation_about_z

    def counted(angle):
        rotations.append(angle)
        return about_z(angle)

    monkeypatch.setattr(three_d_camera, "rotation_about_z", counted)

    def watched(run):
        scene = run.scene
        camera = scene.renderer.camera
        mobs = {
            id(m): m
            for root in [*scene.mobjects, *camera_nodes(camera)]
            for m in root.get_family()
        }
        states = {key: NodeState.of(mob) for key, mob in mobs.items()}
        inputs = CameraExtras.observe(camera, None)
        assert inputs is not None
        made = len(rotations)
        entered(run)
        read.append(
            (
                [
                    key
                    for key, mob in mobs.items()
                    if NodeState.of(mob, states[key]) is not states[key]
                ],
                CameraExtras.observe(camera, inputs) is inputs,
            )
        )
        remade.append(len(rotations) > made)

    monkeypatch.setattr(recorder_module.PlayRun, "entered", watched)
    record = session_of("ArmedTurningCamera", SCENES, frame_rate=15, probe=False).record
    assert record.entries is not None
    assert len(record.entries) == len(record.plays) == len(read) == 3
    assert read == [([], True)] * 3
    # Under a 3D camera a body stands at the scene's own x and y: reading
    # one projects nothing, so the camera's rotation is not made again.
    assert remade == [False] * 3
