"""Every frame shows each mobject at its state at that frame.

CE paints a play's static mobjects once, when it begins, and reuses that
picture under every frame; what a driver added after them writes while the
play runs (a body it moves, a readout whose glyphs it replaces, the camera
it turns or relights) stays frozen until the next play. Arachne keeps CE's
decision of what moves, and rebuilds each frame's static list from the
frame's capture: the cached picture covers the longest prefix of that list
that is unchanged since the play began, and the rest is drawn at the
frame's state. Live frames (a record that refuses) cache nothing.

``static_layer=False`` draws every frame from the background, and is the
oracle each scene is compared with byte for byte. It shares the rebuilt
list, so the tests also assert the expected pixels.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import config, tempconfig
from arachne.camera.camera import Camera
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.manager import Manager
from arachne.mobject.geometry.polygram import Square
from arachne.timeline.clock import frames_in
from arachne.timeline.draw import Drawing, _depths_ordered
from arachne.timeline.routes import draw

from . import stale_scenes
from .conftest import SMALL, render
from .stale_scenes import (
    BODY_SPEED,
    BODY_START,
    LET_GO,
    TAKE_IN,
    TINT_SWITCH,
    DrivenBody,
    DrivenReadout,
    LiveDriven,
    LiveHeld,
    MovedLight,
    MovedLightAt,
    PartlyWritten,
    ReadoutAt,
    ReadoutAtTwo,
    TakenIn,
    TurnedCamera,
    TurnedCameraAt,
    Untouched,
    ZOrderedChild,
)

STALE_SCENES = Path(stale_scenes.__file__)
FPS = 15
#: A frame large enough for a readout's glyphs.
MEDIUM = {"pixel_width": 192, "pixel_height": 108}

RED = (252, 98, 85)
GREEN = (131, 193, 103)
BLUE = (88, 196, 221)
BLACK = (0, 0, 0)


def pixel(scene, frame: np.ndarray, x: float, y: float) -> tuple[int, ...]:
    """The RGB of ``frame`` at the scene point ``(x, y)`` of a 2D camera."""
    camera = scene.renderer.camera
    column = int((x / camera.frame_width + 0.5) * camera.pixel_width)
    row = int((0.5 - y / camera.frame_height) * camera.pixel_height)
    return tuple(int(v) for v in frame[row, column, :3])


def assert_same_frames(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    assert len(a) == len(b)
    for index, (one, other) in enumerate(zip(a, b, strict=True)):
        np.testing.assert_array_equal(one, other, err_msg=f"frame {index}")


def without_static_layer(scene_class: type, **settings) -> list[np.ndarray]:
    return render(scene_class, FPS, static_layer=False, **settings)[1]


def render_counted(
    scene_class: type, **settings
) -> tuple[object, list[np.ndarray], Counter[object]]:
    """Render ``scene_class`` as :func:`.conftest.render` does, counting how
    many times each vectorized mobject is drawn (by id), and how many frames
    start from a picture rather than from the background (``"picture"``).
    """
    counts: Counter[object] = Counter()
    frames: list[np.ndarray] = []
    with tempconfig(
        {
            **SMALL,
            "frame_rate": FPS,
            "dry_run": True,
            "disable_caching": True,
            **settings,
        }
    ):
        scene = scene_class()
        Manager(scene)
        # On the camera instance: a method of a class the engine defines,
        # patched, would change what the record reads its drawing code to call.
        camera = scene.renderer.camera
        display = camera.display_vectorized

        def counting(vmobject, ctx):
            counts[id(vmobject)] += 1
            return display(vmobject, ctx)

        camera.display_vectorized = counting
        to_background = camera.set_frame_to_background

        def from_picture(picture):
            counts["picture"] += 1
            return to_background(picture)

        camera.set_frame_to_background = from_picture

        def write_frame(pixels, *, repeat=1):
            frames.extend([pixels] * repeat)

        scene.renderer.file_writer.write_frame = write_frame
        scene.render()
    return scene, frames, counts


# ── a body moved by a later-added driver ─────────────────────────────────────


def test_a_body_a_later_driver_moves_travels_with_it():
    scene, frames = render(DrivenBody, FPS)
    assert len(frames) == 30
    start = tuple(BODY_START[:2])
    for index in (8, 15, 24):
        x = start[0] + BODY_SPEED * index / FPS
        assert pixel(scene, frames[index], x, 0) == RED, index
        assert pixel(scene, frames[index], *start) == BLACK, index


def test_a_driven_body_draws_as_without_the_static_layer():
    assert_same_frames(render(DrivenBody, FPS)[1], without_static_layer(DrivenBody))


# ── a readout a later-added driver rewrites ──────────────────────────────────


def test_a_readout_shows_the_digits_its_driver_writes():
    _, frames = render(DrivenReadout, FPS, **MEDIUM)
    one = render(ReadoutAt, FPS, **MEDIUM)[1][0]
    two = render(ReadoutAtTwo, FPS, **MEDIUM)[1][0]
    # 1.5 s and 2.2 s in: the driver has set 1, then 2.
    np.testing.assert_array_equal(frames[22], one)
    np.testing.assert_array_equal(frames[33], two)
    assert not np.array_equal(frames[0], one)


def test_a_driven_readout_draws_as_without_the_static_layer():
    assert_same_frames(
        render(DrivenReadout, FPS, **MEDIUM)[1],
        without_static_layer(DrivenReadout, **MEDIUM),
    )


# ── what the cached picture still covers ─────────────────────────────────────


def test_static_mobjects_nothing_writes_are_painted_once_per_play():
    scene, frames, counts = render_counted(Untouched)
    square, circle, dot = scene.mobjects
    assert len(frames) == 30
    assert counts[id(square)] == 2
    assert counts[id(circle)] == 2
    assert counts[id(dot)] >= 30


def test_the_picture_covers_the_static_mobjects_before_the_written_one():
    scene, frames, counts = render_counted(PartlyWritten)
    first, middle, last = scene.mobjects[:3]
    assert len(frames) == 15
    # The play's picture, then one picture of what comes before the
    # written circle; the circle and what follows it are drawn every frame.
    assert counts[id(first)] == 2
    assert counts[id(middle)] >= 15
    assert counts[id(last)] >= 15
    # 0.6 s in, the circle has moved 0.6 down.
    assert pixel(scene, frames[9], 0, 0.4) == RED
    assert pixel(scene, frames[9], 0, 1.3) == BLACK
    assert_same_frames(frames, without_static_layer(PartlyWritten))


# ── the camera is a drawing input too ────────────────────────────────────────


@pytest.mark.parametrize(
    ("scene_class", "still"),
    [(TurnedCamera, TurnedCameraAt), (MovedLight, MovedLightAt)],
)
def test_a_driver_that_changes_the_camera_redraws_the_static_mobjects(
    scene_class, still
):
    _, frames = render(scene_class, FPS)
    assert len(frames) == 15
    # The cube as a still of the camera input the driver set at that frame.
    np.testing.assert_array_equal(frames[still.FRAME], render(still, FPS)[1][0])
    assert not np.array_equal(frames[0], frames[still.FRAME])
    assert_same_frames(frames, without_static_layer(scene_class))


# ── what the static picture painted and the frame no longer has ─────────────


def test_a_static_mobjects_moving_child_leaves_no_trace_where_it_began():
    scene, frames = render(ZOrderedChild, FPS)
    assert len(frames) == 15
    # Half a second in, the child has moved a unit right of where it began:
    # the square shows where it was.
    assert pixel(scene, frames[8], -1, 0) == BLUE
    assert pixel(scene, frames[8], -1 + 2 * 8 / FPS, 0) == RED
    assert_same_frames(frames, without_static_layer(ZOrderedChild))


def test_a_static_mobject_a_moving_family_takes_in_is_drawn_once():
    scene, frames = render(TakenIn, FPS)
    assert len(frames) == 15
    square = (-3, 0)
    before = pixel(scene, frames[2], *square)
    held = [i for i in range(15) if TAKE_IN + 0.05 < i / FPS < LET_GO - 0.05]
    assert held
    for index in held:
        # Translucent: drawn twice (in the static picture and with the
        # holder), it would show more opaque.
        assert pixel(scene, frames[index], *square) == before, index
    assert pixel(scene, frames[12], *square) == before
    assert_same_frames(frames, without_static_layer(TakenIn))


def test_a_depth_that_is_no_number_keeps_the_list_whole():
    # ThreeDCamera sorts by depth; with a NaN key, a list sorted in two parts
    # may not keep the order the whole list sorted to, so it is never cut.
    camera = ThreeDCamera()
    camera.reset_rotation_matrix()
    flat = Square().set_shade_in_3d(True)
    assert _depths_ordered(camera, [flat, Square()])
    undepthed = Square().set_shade_in_3d(True)
    undepthed.points = np.full_like(undepthed.points, np.nan)
    assert not _depths_ordered(camera, [flat, undepthed])
    assert _depths_ordered(Camera(), [flat, undepthed])


# ── frozen frames drawn live ─────────────────────────────────────────────────


def test_a_live_frozen_frame_runs_the_scenes_drawing_code_as_ce_does():
    scene, frames, counts = render_counted(LiveHeld)
    assert scene.timeline.record.refusal is not None, "drawn from the record"
    # Its static picture, then the scene over it, as CE draws a frozen frame.
    assert scene.held_calls == 2
    assert counts["picture"] == 1
    off, off_frames, off_counts = render_counted(LiveHeld, static_layer=False)
    assert off.held_calls == 2
    # Without the static layer the frozen frame starts from the background.
    assert off_counts["picture"] == 0
    assert_same_frames(frames, off_frames)


# ── frames drawn live cache nothing ──────────────────────────────────────────


def test_a_live_drawn_scene_shows_what_its_driver_writes():
    scene, frames = render(LiveDriven, FPS)
    assert scene.timeline.record.refusal is not None, "drawn from the record"
    assert len(frames) == 23
    square = (-4, 2)
    assert pixel(scene, frames[3], *square) == RED
    # The class attribute the square's fill reads switched to blue.
    after = int(TINT_SWITCH * FPS) + 3
    assert pixel(scene, frames[after], *square) == BLUE
    assert pixel(scene, frames[18], BODY_START[0] + BODY_SPEED * 18 / FPS, 0) == GREEN
    assert_same_frames(frames, without_static_layer(LiveDriven))


# ── one record, drawn with the static layer and without ──────────────────────


def drawn_both_ways(scene) -> tuple[list[np.ndarray], list[np.ndarray]]:
    record = scene.timeline.record
    fps = Fraction(FPS)
    instants = [
        t for play in record.plays for t in frames_in(play.start, play.end, fps)
    ]
    found = []
    for static_layer in (True, False):
        with Drawing(record, static_layer=static_layer) as drawing:
            pictures: dict[int, object] = {}

            def static(play, drawing=drawing, pictures=pictures):
                if play.index not in pictures:
                    pictures[play.index] = drawing.static(play)
                return pictures[play.index]

            found.append([draw(drawing, record, t, static) for t in instants])
    return found[0], found[1]


@pytest.mark.parametrize(
    "scene_class",
    [
        DrivenBody,
        DrivenReadout,
        PartlyWritten,
        Untouched,
        TurnedCamera,
        ZOrderedChild,
        TakenIn,
    ],
)
def test_one_record_draws_alike_with_and_without_the_static_layer(scene_class):
    scene, rendered = render(scene_class, FPS)
    with_layer, without = drawn_both_ways(scene)
    assert_same_frames(with_layer, without)
    assert_same_frames(rendered, with_layer)


@pytest.mark.parametrize("scene_class", [PartlyWritten, TurnedCamera])
def test_a_session_draws_the_frames_the_render_does(session_of, scene_class):
    # A session keeps each play's static picture across its frames, which
    # are each drawn by a drawing of their own.
    _, frames = render(scene_class, FPS)
    session = session_of(scene_class.__name__, STALE_SCENES)
    for index in (0, 4, 9, 14):
        frame = np.asarray(session.frame(Fraction(index, FPS)))
        np.testing.assert_array_equal(frame, frames[index], err_msg=str(index))


# ── the option ───────────────────────────────────────────────────────────────


def test_the_static_layer_is_an_option_on_by_default():
    assert config.static_layer is True
    with tempconfig({"static_layer": False}):
        assert config.static_layer is False
    with pytest.raises(ValueError, match="static_layer"):
        config.static_layer = "no"


def test_drawing_reads_the_option_unless_told():
    scene, _ = render(Untouched, FPS)
    record = scene.timeline.record
    assert Drawing(record).static_layer is True
    assert Drawing(record, static_layer=False).static_layer is False
    with tempconfig({"static_layer": False}):
        assert Drawing(record).static_layer is False
        assert Drawing(record, static_layer=True).static_layer is True
