"""A render to video writes exactly the frames it drew, on every route and
on every way a play comes to be drawn as it runs.

The other timeline tests render with nothing written; these write an MP4,
decode it and compare it with the frames the renderer handed its writer:
losslessly (PNG frames in the MP4 container) to compare pixels, and with
the default encoder to count frames.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from arachne import (
    BLUE,
    LEFT,
    RIGHT,
    UP,
    Dot,
    FadeIn,
    Line,
    MappingCamera,
    Scene,
    Square,
    ValueTracker,
    always_redraw,
    tempconfig,
)
from arachne.manager import Manager

from .camera_scenes import DisplayLeaves, Mapped
from .conftest import SMALL
from .soundness_scenes import OverriddenFill, TwoTransientFills

FPS = 15
#: Frames stored as PNG images: what is decoded is what was drawn.
LOSSLESS = {"video_codec": "png", "pixel_format": "rgb24"}


class Moves(Scene):
    """Trusted plays only, and a wait."""

    def construct(self):
        dot = Dot()
        self.play(dot.animate.shift(RIGHT * 2))
        self.play(FadeIn(Square().shift(LEFT * 2)), run_time=0.5)
        self.wait(0.5)


class Follows(Scene):
    """A line redrawn from a tracker: a state-function play."""

    def construct(self):
        t = ValueTracker(0)
        dot = Dot(color=BLUE)
        dot.add_updater(lambda m: m.move_to(RIGHT * t.get_value() + UP))
        line = always_redraw(lambda: Line(LEFT * 3, dot.get_center()))
        self.add(dot, line)
        self.play(t.animate.set_value(2), run_time=1)


class MappedStops(Scene):
    """A mapping camera (every play drawn as it runs) and two waits a stop
    condition ends: one at once, before any frame, and one partway.
    """

    def __init__(self, **kwargs):
        super().__init__(camera_class=MappingCamera, **kwargs)
        self.renderer.camera.mapping_func = lambda p: p + RIGHT

    def construct(self):
        dot = Dot(LEFT * 3, radius=0.3)
        self.add(dot)
        self.play(dot.animate.shift(RIGHT * 2))
        self.wait_until(lambda: True, max_time=1)
        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(t)
        self.wait_until(lambda: t.get_value() > 0.2, max_time=1)
        self.wait(0.25)


def render_to_video(
    scene_class: type, media_dir: Path, **settings
) -> tuple[Scene, list[np.ndarray], Path]:
    """Render ``scene_class`` to an MP4 under ``media_dir``; return the
    scene, every frame its writer was handed (repeats expanded) in the
    execution that ended the render, and the video's path.
    """
    executions: list[list[np.ndarray]] = []
    with tempconfig(
        {
            **SMALL,
            "frame_rate": FPS,
            "media_dir": str(media_dir),
            "disable_caching": True,
            "format": "mp4",
            **settings,
        }
    ):
        scene = scene_class()
        manager = Manager(scene)

        def collect() -> None:
            # On the writer instance, passing every frame on to it.
            frames: list[np.ndarray] = []
            executions.append(frames)
            writer = scene.renderer.file_writer
            write = writer.write_frame

            def write_frame(pixels, *, repeat=1):
                frames.extend([pixels.copy()] * repeat)
                write(pixels, repeat=repeat)

            writer.write_frame = write_frame

        renew = manager._renew_scene

        def renewed() -> None:
            renew()
            collect()

        manager._renew_scene = renewed
        collect()
        scene.render()
        video = Path(scene.renderer.file_writer.movie_file_path)
    assert video.suffix == ".mp4"
    return scene, executions[-1], video


def decoded(video: Path) -> list[np.ndarray]:
    with av.open(str(video)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def live_plays(scene: Scene) -> list[str | None]:
    """Per play, why it was drawn as it ran: ``"refused"`` (the record
    can't draw it), ``"after_refusal"`` (it follows a mid-play refusal known
    from an earlier execution) or ``None`` (drawn from the record).
    """
    record = scene.timeline.ensure_record()
    live_from = record.execution.live_from
    kinds: list[str | None] = []
    for play in record.plays:
        if play.index in record.execution.known:
            kinds.append("refused")
        elif live_from is not None and play.index > live_from:
            kinds.append("after_refusal")
        elif record.refusal is not None:
            kinds.append("refused")
        else:
            kinds.append(None)
    return kinds


#: Each route, and each way a play comes to be drawn as it runs: the scene,
#: its frame count at 15 fps, and per play its route (trusted and state
#: scenes) or why it was drawn live.
CASES = [
    pytest.param(Moves, 30, ["trusted"] * 3, id="trusted"),
    pytest.param(Follows, 15, ["state"], id="state"),
    pytest.param(OverriddenFill, 15, ["refused"] * 2, id="user_override"),
    pytest.param(Mapped, 19, ["refused"] * 2, id="mapping_camera"),
    pytest.param(DisplayLeaves, 15, ["refused"] * 2, id="display_left"),
    pytest.param(
        TwoTransientFills,
        38,
        ["refused", "after_refusal", "after_refusal"],
        id="after_refusal",
    ),
    pytest.param(MappedStops, 22, ["refused"] * 4, id="live_stop_condition"),
]

#: The record's whole-record refusal, by case.
REFUSALS = {
    OverriddenFill: "draw.user_override",
    Mapped: "camera.mapping_func",
    DisplayLeaves: "camera.display_left",
    MappedStops: "camera.mapping_func",
}


def check_routes(scene: Scene, expected: list[str]) -> None:
    record = scene.timeline.ensure_record()
    refusal = REFUSALS.get(type(scene))
    if refusal is not None:
        assert record.refusal is not None
        assert record.refusal.startswith(refusal), record.refusal
    if expected[0] in ("trusted", "state", "ticked"):
        assert record.refusal is None
        assert [play.route for play in record.plays] == expected
        assert live_plays(scene) == [None] * len(expected)
    else:
        assert live_plays(scene) == expected
        if type(scene) is TwoTransientFills:
            assert record.execution.number == 2


@pytest.mark.parametrize(("scene_class", "frames", "plays"), CASES)
def test_the_video_holds_the_frames_drawn(tmp_path, scene_class, frames, plays):
    scene, drawn, video = render_to_video(scene_class, tmp_path, **LOSSLESS)
    check_routes(scene, plays)
    assert len(drawn) == frames
    shown = decoded(video)
    assert len(shown) == frames
    for index, (pixels, frame) in enumerate(zip(drawn, shown, strict=True)):
        np.testing.assert_array_equal(frame, pixels[..., :3], err_msg=f"frame {index}")


@pytest.mark.parametrize(("scene_class", "frames", "plays"), CASES)
def test_the_default_encoder_writes_every_frame(tmp_path, scene_class, frames, plays):
    scene, drawn, video = render_to_video(scene_class, tmp_path)
    assert len(drawn) == frames
    assert len(decoded(video)) == frames
