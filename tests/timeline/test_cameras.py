"""Cameras: callbacks recorded as values and classified with the play's, a
MultiCamera's displays recorded per capture, and a MappingCamera drawn while
the scene runs.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from arachne import BLUE, LEFT, RED, RIGHT, Dot, config, tempconfig
from arachne.camera.camera import Camera
from arachne.timeline import recorder
from arachne.timeline.routes import RecordRefused
from arachne.utils.color import ManimColor

from . import camera_scenes
from .conftest import SMALL, render

CAMERA_SCENES = Path(camera_scenes.__file__)
#: Opened in a session only: importing it changes the process's configuration.
CONFIG_SCENES = Path(__file__).parent / "config_scenes.py"
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


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_display_repainted_before_its_camera_joins_shows_each_plays_picture(
    session_of, draw_path
):
    """Before its camera joins, the display shows that camera's picture as
    each play found it: red in the first wait, blue in the second. Drawn
    with the picture its camera held as it joined, both waits show blue.
    """
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.DisplayRepaintedBeforeItJoins
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    assert _display_brightness(live[1]) != _display_brightness(live[4])
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    assert scene.timeline.record.refusal is None
    _equal_frames(drawn, live)
    session = session_of(scene_class.__name__, CAMERA_SCENES, draw_path=draw_path)
    _equal_frames(
        [np.asarray(session.frame(Fraction(k, 15))) for k in range(len(live))], live
    )


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


def _drawn_live(scene_class, fps, **settings) -> list[np.ndarray]:
    """The frames of ``scene_class`` drawn as the scene runs, each from the
    live scene at its instant, as CE draws them.
    """
    with mock.patch.object(
        recorder, "camera_refusal", return_value="drawn live, to compare"
    ):
        scene, frames = render(scene_class, fps, **settings)
    assert scene.timeline.record.refusal is not None
    return frames


def _centre(frame: np.ndarray) -> tuple[int, ...]:
    height, width = frame.shape[:2]
    return tuple(int(v) for v in frame[height // 2, width // 2, :3])


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
@pytest.mark.parametrize(
    "scene_class",
    [camera_scenes.DepthReferenceMoves, camera_scenes.StaticDepthReferenceMoves],
)
def test_a_depth_reference_the_scene_does_not_hold_sorts_each_frame_where_it_was(
    session_of, scene_class, draw_path
):
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    # The point moves partway through: blue in front, then red.
    red, blue = (tuple(int(v) for v in ManimColor(c).to_int_rgb()) for c in (RED, BLUE))
    centres = [_centre(frame) for frame in live]
    assert centres[0] == blue
    assert centres[-1] == red
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    assert scene.timeline.record.refusal is None
    assert len(drawn) == len(live)
    for k, (frame, reference) in enumerate(zip(drawn, live, strict=True)):
        np.testing.assert_array_equal(frame, reference, err_msg=str(k))
    session = session_of(scene_class.__name__, CAMERA_SCENES, draw_path=draw_path)
    for k, reference in enumerate(live):
        frame = np.asarray(session.frame(Fraction(k, 15)))
        np.testing.assert_array_equal(frame, reference, err_msg=f"session {k}")


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_restored_instant_sorts_by_the_depth_reference_its_capture_found(
    session_of, draw_path
):
    """Between two frames of the state play, the restore starts from the
    capture before and sorts the red square by the reference point that
    capture found, behind the blue one, as a forward run evaluated there
    does; the point the scene moved in front after the play is not it.
    """
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    name = "RestoredDepthReference"
    settings = {"frame_rate": 15, "probe": False, "draw_path": draw_path}
    session = session_of(name, CAMERA_SCENES, **settings)
    play = session.record.plays[0]
    assert play.route == "state"
    t = play.start + Fraction(3, 30)
    assert session.state_at(t).answer == "restored"
    frame = np.asarray(session.frame(t))
    blue = tuple(int(v) for v in ManimColor(BLUE).to_int_rgb())
    assert _centre(frame) == blue
    oracle = session_of(name, CAMERA_SCENES, instants=[t], **settings)
    np.testing.assert_array_equal(frame, np.asarray(oracle.frame(t)))
    # The probe's restores find the play as it ran: nothing demotes it.
    probed = session_of(name, CAMERA_SCENES, frame_rate=15, draw_path=draw_path)
    assert probed.record.plays[0].route == "state"


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_restored_instant_sorts_by_a_depth_group_the_restore_moves(
    session_of, draw_path
):
    """The scene holds the depth group, and the restore evaluates it at the
    instant: by then it is in front of the blue square, as a forward run
    evaluated there finds it, though the capture before found it behind.
    """
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    name = "RestoredMovingDepthGroup"
    settings = {"frame_rate": 15, "probe": False, "draw_path": draw_path}
    session = session_of(name, CAMERA_SCENES, **settings)
    assert session.record.plays[0].route == "state"
    t = Fraction(19, 100)
    assert session.state_at(t).answer == "restored"
    frame = np.asarray(session.frame(t))
    red = tuple(int(v) for v in ManimColor(RED).to_int_rgb())
    assert _centre(frame) == red
    oracle = session_of(name, CAMERA_SCENES, instants=[t], **settings)
    np.testing.assert_array_equal(frame, np.asarray(oracle.frame(t)))


def _kept_pictures(record) -> list[np.ndarray]:
    """Every picture of another camera the record keeps, each once."""
    kept: dict[int, np.ndarray] = {}
    for capture in record._captures:
        for _, picture in capture.pictures:
            kept[id(picture)] = picture
    return list(kept.values())


def _equal_frames(drawn, reference) -> None:
    assert len(drawn) == len(reference)
    for k, (frame, expected) in enumerate(zip(drawn, reference, strict=True)):
        np.testing.assert_array_equal(frame, expected, err_msg=str(k))


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_still_display_of_another_camera_keeps_one_picture(session_of, draw_path):
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.OtherCameraStill
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    record = scene.timeline.record
    assert record.refusal is None
    assert len(record.plays) == 3
    assert len(_kept_pictures(record)) == 1
    _equal_frames(drawn, live)
    session = session_of(scene_class.__name__, CAMERA_SCENES, draw_path=draw_path)
    _equal_frames(
        [np.asarray(session.frame(Fraction(k, 15))) for k in range(len(live))], live
    )


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_display_repainted_between_plays_keeps_a_picture_a_play(
    session_of, draw_path
):
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.OtherCameraBetweenPlays
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    red, blue = (tuple(int(v) for v in ManimColor(c).to_int_rgb()) for c in (RED, BLUE))
    assert [_centre(live[k]) for k in (0, 4, 8)] == [red, blue, red]
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    record = scene.timeline.record
    assert record.refusal is None
    # Red, blue, red: one copy each play, the last not the first's.
    assert len(_kept_pictures(record)) == 3
    _equal_frames(drawn, live)
    session = session_of(scene_class.__name__, CAMERA_SCENES, draw_path=draw_path)
    _equal_frames(
        [np.asarray(session.frame(Fraction(k, 15))) for k in range(len(live))], live
    )


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_display_repainted_while_off_screen_during_a_play_draws_that_play_live(
    session_of, draw_path
):
    """The play's picture of the display is the one it showed first, also
    when the display leaves the screen between: shown again repainted, it
    refuses the play rather than keep a second picture.
    """
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.OtherCameraHidden
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    red, blue = (tuple(int(v) for v in ManimColor(c).to_int_rgb()) for c in (RED, BLUE))
    assert _centre(live[0]) == red
    assert _centre(live[-1]) == blue
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    refusal = scene.timeline.record.refusal
    assert refusal is not None
    assert refusal.startswith("camera.display_redrawn:")
    _equal_frames(drawn, live)


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_display_repainted_during_a_play_draws_that_play_live(session_of, draw_path):
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.OtherCameraShown
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    red, blue = (tuple(int(v) for v in ManimColor(c).to_int_rgb()) for c in (RED, BLUE))
    centres = [_centre(frame) for frame in live]
    assert centres[0] == red
    assert centres[-1] == blue
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    assert scene.timeline.record.refusal.startswith("camera.display_redrawn:")
    _equal_frames(drawn, live)
    session = session_of(scene_class.__name__, CAMERA_SCENES, draw_path=draw_path)
    with pytest.raises(RecordRefused, match="display_redrawn"):
        session.frame(Fraction(1, 15))


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_pixel_shape_changed_during_a_play_draws_from_that_play_live(draw_path):
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.PixelShapeChanged
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    assert not np.array_equal(live[1], live[-1])
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    refusal = scene.timeline.record.refusal
    assert refusal is not None
    assert refusal.startswith("config.pixel_shape:")
    _equal_frames(drawn, live)


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_light_rebound_during_a_play_draws_that_play_live(draw_path):
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.LightReboundInPlay
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    assert _centre(live[1]) != _centre(live[-1])
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    refusal = scene.timeline.record.refusal
    assert refusal is not None
    assert refusal.startswith("camera.rebound:")
    _equal_frames(drawn, live)


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_light_rebound_between_plays_is_drawn_from_the_record(session_of, draw_path):
    """Each play's frames are lit by the light bound as it ran, also when
    drawn after the scene ended, with the other light bound.
    """
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.LightReboundBetweenPlays
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    assert _centre(live[1]) != _centre(live[-1])
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    assert scene.timeline.record.refusal is None
    _equal_frames(drawn, live)
    session = session_of(scene_class.__name__, CAMERA_SCENES, draw_path=draw_path)
    _equal_frames(
        [np.asarray(session.frame(Fraction(k, 15))) for k in range(len(live))], live
    )


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_an_image_painted_in_place_is_drawn_as_the_camera_first_read_it(
    session_of, draw_path
):
    """The camera reads a fill's image once and keeps it: painted later in
    place, it is drawn as it was then, also when drawn after the scene ran.
    """
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    scene_class = camera_scenes.ImageFillPainted
    live = _drawn_live(scene_class, 15, draw_path=draw_path)
    assert _centre(live[0]) == _centre(live[-1]) == (255, 0, 0)
    scene, drawn = render(scene_class, 15, draw_path=draw_path)
    assert scene.timeline.record.refusal is None
    _equal_frames(drawn, live)
    session = session_of(scene_class.__name__, CAMERA_SCENES, draw_path=draw_path)
    _equal_frames(
        [np.asarray(session.frame(Fraction(k, 15))) for k in range(len(live))], live
    )


def _cloud_as_ce_draws(camera_width: int, configured_width: int) -> np.ndarray:
    """The cloud of the pixel-shape scenes drawn as CE draws it: by a camera
    ``camera_width`` wide while the configured width is ``configured_width``.
    """
    with tempconfig({**SMALL, "pixel_width": camera_width, "dry_run": True}):
        camera = Camera()
        config.pixel_width = configured_width
        camera.capture_mobjects([camera_scenes._cloud()])
        return camera.pixel_array.copy()


def _answered(session, instants) -> list[np.ndarray]:
    return [np.asarray(found.frame) for found in session.answers(instants)]


def test_a_pixel_shape_changed_after_the_last_play_is_drawn_as_the_scene_left_it(
    session_of,
):
    scene, _ = render(camera_scenes.PixelShapeChangedAtTheEnd, 15)
    assert scene.timeline.record.refusal.startswith("config.pixel_shape:")
    session = session_of("PixelShapeChangedAtTheEnd", CAMERA_SCENES, frame_rate=15)
    assert session.record.refusal.startswith("config.pixel_shape:")
    width = SMALL["pixel_width"]
    (final,) = _answered(session, [session.record.duration])
    np.testing.assert_array_equal(final, _cloud_as_ce_draws(width, 2 * width))
    (first,) = _answered(session, [0])
    np.testing.assert_array_equal(first, _cloud_as_ce_draws(width, width))


def test_a_pixel_shape_changed_in_setup_is_drawn_as_the_scene_runs(session_of):
    scene_class = camera_scenes.PixelShapeChangedInSetup
    live = _drawn_live(scene_class, 15)
    width = SMALL["pixel_width"]
    np.testing.assert_array_equal(live[0], _cloud_as_ce_draws(width, 2 * width))
    scene, drawn = render(scene_class, 15)
    assert scene.timeline.record.refusal.startswith("config.pixel_shape:")
    _equal_frames(drawn, live)
    session = session_of(scene_class.__name__, CAMERA_SCENES, frame_rate=15)
    assert session.record.refusal.startswith("config.pixel_shape:")
    _equal_frames(_answered(session, [Fraction(k, 15) for k in range(len(live))]), live)


def test_a_pixel_shape_changed_as_the_file_is_imported_is_drawn_as_it_runs(
    session_of,
):
    session = session_of("PixelShapeChangedOnImport", CONFIG_SCENES, frame_rate=15)
    assert session.record.refusal.startswith("config.pixel_shape:")
    width = SMALL["pixel_width"]
    expected = _cloud_as_ce_draws(2 * width, 2 * width)
    for frame in _answered(session, [Fraction(k, 15) for k in range(3)]):
        np.testing.assert_array_equal(frame, expected)
