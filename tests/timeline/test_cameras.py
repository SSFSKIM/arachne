"""Cameras: callbacks recorded as values and classified with the play's, a
MultiCamera's displays recorded per capture, and a MappingCamera drawn while
the scene runs.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import LEFT, RIGHT, Dot, tempconfig
from arachne.timeline.routes import RecordRefused

from . import camera_scenes
from .conftest import SMALL, render

CAMERA_SCENES = Path(camera_scenes.__file__)
RATES = (4, 15, 30, 60)


def _shared_frames_agree(scene_class) -> None:
    frames = {fps: render(scene_class, fps)[1] for fps in RATES}
    for slow, fast in itertools.combinations(RATES, 2):
        for k in range(len(frames[slow])):
            if k * fast % slow == 0:
                assert np.array_equal(
                    frames[slow][k], frames[fast][k * fast // slow]
                ), (
                    slow,
                    fast,
                    k,
                )


def _render_equals_session(session_of, scene_class, fps=6) -> None:
    _, frames = render(scene_class, fps)
    session = session_of(scene_class.__name__, CAMERA_SCENES)
    assert len(frames) == len(session.record.samples(fps))
    for k, frame in enumerate(frames):
        assert np.array_equal(frame, np.asarray(session.frame(Fraction(k, fps)))), k


def test_a_pure_centre_function_makes_its_plays_state_functions(session_of):
    record = session_of("PureCentre", CAMERA_SCENES).record
    assert [play.route for play in record.plays] == ["state", "state"]
    (callback,) = record.plays[0].admission.callbacks
    assert callback.kind == "center"
    assert callback.proof is not None
    _shared_frames_agree(camera_scenes.PureCentre)
    _render_equals_session(session_of, camera_scenes.PureCentre)


def test_a_stateful_centre_function_ticks_its_play(session_of):
    record = session_of("StatefulCentre", CAMERA_SCENES).record
    (play,) = record.plays
    assert play.route == "ticked"
    assert any(
        "center at camera_scenes.py" in reason for reason in play.admission.reasons
    )
    _shared_frames_agree(camera_scenes.StatefulCentre)
    _render_equals_session(session_of, camera_scenes.StatefulCentre)


@pytest.mark.parametrize(
    "scene_class", [camera_scenes.ZoomStill, camera_scenes.ZoomAnimated]
)
def test_a_zoomed_scene_is_drawn_from_its_record_frame_for_frame(
    session_of, scene_class
):
    _render_equals_session(session_of, scene_class, fps=15)


def _display_brightness(frame: np.ndarray) -> float:
    """The mean brightness inside the zoomed display, which sits in the
    frame's upper right corner.
    """
    height, width = frame.shape[:2]
    corner = frame[height // 10 : height // 2, width * 3 // 5 : width * 9 // 10, :3]
    return float(corner.mean())


def test_a_display_whose_camera_joins_mid_scene_is_empty_until_it_joins(session_of):
    """The display's camera joins at 1/2 s. A frame drawn after the scene ran
    shows the membership of its own instant: before 1/2 the display is its
    camera's empty picture, after it the zoomed dot. Drawn with the
    membership at the scene's end instead, every frame shows the dot.
    """
    session = session_of("DisplayLater", CAMERA_SCENES)
    after = np.asarray(session.frame(Fraction(3, 4)))
    before = np.asarray(session.frame(Fraction(1, 4)))
    assert _display_brightness(after) > _display_brightness(before) + 10
    _, frames = render(camera_scenes.DisplayLater, 8)
    assert np.array_equal(frames[2], before)
    assert np.array_equal(frames[6], after)


def test_a_display_left_by_its_camera_is_drawn_as_the_scene_runs(session_of):
    """Once a display's camera leaves, the display shows the last picture
    drawn into it, which only drawing as the scene runs knows: the record
    refuses from then on, and the render draws the rest live.
    """
    session = session_of("DisplayLeaves", CAMERA_SCENES)
    record = session.record
    assert record.refusal.startswith("camera.display_left:")
    assert [play.route for play in record.plays] == ["trusted", "ticked"]
    with pytest.raises(RecordRefused):
        session.frame(Fraction(1, 4))
    _, frames = render(camera_scenes.DisplayLeaves, 8)
    # The display keeps the zoomed dot as it was at the last frame before
    # its camera left, while the dot itself moves away.
    last_joined = frames[3]
    for frame in frames[4:]:
        height, width = frame.shape[:2]
        corner = (slice(0, height // 2), slice(width * 3 // 5, width))
        assert np.array_equal(frame[corner], last_joined[corner])
    assert not np.array_equal(frames[-1], last_joined)


def test_a_mapping_camera_record_refuses_with_its_reason(session_of):
    session = session_of("Mapped", CAMERA_SCENES)
    record = session.record
    assert record.refusal.startswith("camera.mapping_func:")
    assert all(play.route == "ticked" for play in record.plays)
    with pytest.raises(RecordRefused, match="mapping_func"):
        session.frame(Fraction(1, 2))
    with pytest.raises(RecordRefused):
        session.state_at(Fraction(1, 2))


def _drawn_as_ce_draws(fps: int) -> list[np.ndarray]:
    """``Mapped``'s frames drawn as CE's play loop draws them: the animation
    interpolated at each frame's alpha and the camera capturing the scene,
    then the held wait's one picture.
    """
    frames: list[np.ndarray] = []
    with tempconfig(
        {**SMALL, "frame_rate": fps, "dry_run": True, "disable_caching": True}
    ):
        camera = camera_scenes.Mapped().renderer.camera
        dot = Dot(LEFT * 3, radius=0.3)
        animation = dot.animate.shift(RIGHT * 2).build()
        animation.begin()
        for k in range(fps):
            animation.interpolate(k / fps)
            camera.reset()
            camera.capture_mobjects([dot])
            frames.append(camera.pixel_array.copy())
        animation.finish()
        camera.reset()
        camera.capture_mobjects([dot])
        frames.extend([camera.pixel_array.copy()] * (math.ceil(fps * 1.25) - fps))
    return frames


@pytest.mark.parametrize("fps", [6, 15])
def test_a_mapping_camera_scene_is_drawn_as_it_runs(fps):
    _, frames = render(camera_scenes.Mapped, fps)
    expected = _drawn_as_ce_draws(fps)
    assert len(frames) == len(expected)
    for k, (frame, reference) in enumerate(zip(frames, expected, strict=True)):
        assert np.array_equal(frame, reference), k
