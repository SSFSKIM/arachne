"""A render to video writes exactly the frames it drew, on every route and
on every way a play comes to be drawn as it runs.

The other timeline tests render with nothing written; these write an MP4,
decode it and compare it with the frames the renderer handed its writer:
losslessly (PNG frames in the MP4 container) to compare pixels, and with
the default encoder to count frames. With ``hold_gaps`` a run of equal frames
is one decoded frame that lasts the run: decoded frames are expanded on the
frame grid by their times before they are compared.
"""

from __future__ import annotations

from fractions import Fraction
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
    assert video.suffix == f".{settings.get('format', 'mp4')}"
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


# ── one stream, and holds as gaps ────────────────────────────────────────────

#: Lossless encodings: PNG frames in MP4, and QuickTime RLE in MOV.
GAPPED_LOSSLESS = {
    "png": LOSSLESS,
    "qtrle": {"format": "mov", "video_codec": "qtrle", "pixel_format": "rgb24"},
}

CLICK = Path(__file__).parents[2] / "docs" / "source" / "_static" / "click.wav"


class HeldAround(Scene):
    """A hold first, a play, and a hold to the end: the first frame, a middle
    run and the last frame each last several frames.
    """

    def construct(self):
        dot = Dot(LEFT)
        self.add(dot)
        self.wait(0.4)
        self.play(dot.animate.shift(RIGHT * 2), run_time=0.4)
        self.wait(0.6)


class Narrated(Scene):
    """A sound that ends inside the scene's last hold."""

    def construct(self):
        dot = Dot()
        self.add(dot)
        self.add_sound(CLICK, time_offset=1.35)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
        self.wait(1.5)


GAPPED_CASES = [*CASES, pytest.param(HeldAround, 21, ["trusted"] * 3, id="held_around")]


def stream_of(video: Path) -> dict:
    """The video stream decoded: its frames (RGB) and their times in frames
    of the grid, the time its last frame ends, and its durations.
    """
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        base = Fraction(stream.time_base)
        packets = [
            (p.pts, p.duration) for p in container.demux(stream) if p.pts is not None
        ]
        stream_duration = stream.duration * base * FPS if stream.duration else None
        microseconds = container.duration
    with av.open(str(video)) as container:
        frames = list(container.decode(video=0))
    origin = min(pts for pts, _ in packets)
    # The container's duration is in whole microseconds: the nearest whole
    # frame when the duration is that frame's, rounded.
    nearest = round(Fraction(microseconds * FPS, av.time_base))
    rounded = round(Fraction(nearest * av.time_base, FPS)) == microseconds
    return {
        "times": [(f.pts - origin) * base * FPS for f in frames],
        "pictures": [f.to_ndarray(format="rgb24") for f in frames],
        "end": (max(p + d for p, d in packets) - origin) * base * FPS,
        "stream_duration": stream_duration,
        "container_duration": (
            nearest if rounded else Fraction(microseconds * FPS, av.time_base)
        ),
    }


def on_grid(movie: dict) -> list[np.ndarray]:
    """Decoded frames expanded on the frame grid: each from its time to the
    next frame's, the last to the stream's end. Every time is on the grid.
    """
    ends = [*movie["times"][1:], movie["end"]]
    shown: list[np.ndarray] = []
    for start, end, pixels in zip(movie["times"], ends, movie["pictures"], strict=True):
        assert start.denominator == 1, start
        assert end.denominator == 1, end
        shown += [pixels] * int(end - start)
    return shown


def run_starts(drawn: list[np.ndarray]) -> list[int]:
    """Where each run of equal frames in ``drawn`` starts."""
    return [
        index
        for index, pixels in enumerate(drawn)
        if index == 0 or not np.array_equal(pixels, drawn[index - 1])
    ]


def runs(drawn: list[np.ndarray]) -> int:
    """How many runs of equal frames ``drawn`` holds."""
    return len(run_starts(drawn))


@pytest.mark.parametrize("encoding", sorted(GAPPED_LOSSLESS))
@pytest.mark.parametrize(("scene_class", "frames", "plays"), GAPPED_CASES)
def test_a_gapped_video_shows_the_frames_drawn(
    tmp_path, scene_class, frames, plays, encoding
):
    scene, drawn, video = render_to_video(
        scene_class, tmp_path, hold_gaps=True, **GAPPED_LOSSLESS[encoding]
    )
    check_routes(scene, plays)
    assert len(drawn) == frames
    movie = stream_of(video)
    assert len(movie["pictures"]) == runs(drawn)
    assert movie["end"] == frames
    shown = on_grid(movie)
    assert len(shown) == frames
    for index, (pixels, frame) in enumerate(zip(drawn, shown, strict=True)):
        np.testing.assert_array_equal(frame, pixels[..., :3], err_msg=f"frame {index}")


def test_holds_are_written_once(tmp_path):
    _, drawn, video = render_to_video(HeldAround, tmp_path, hold_gaps=True, **LOSSLESS)
    # The leading hold (6 frames, and the play's first frame, which shows
    # the same), the play's other frames, and the last hold (9 frames).
    starts = run_starts(drawn)
    assert starts[0] == 0
    assert starts[1] >= 6
    assert starts[-1] == 12
    assert stream_of(video)["times"] == starts


def test_without_gaps_holds_are_written_frame_by_frame(tmp_path):
    _, drawn, video = render_to_video(HeldAround, tmp_path, **LOSSLESS)
    assert stream_of(video)["times"] == list(range(21))


def test_a_render_with_caching_off_writes_one_stream(tmp_path):
    scene, _, _ = render_to_video(Moves, tmp_path, **LOSSLESS)
    directory = scene.renderer.file_writer.partial_movie_directory
    assert len([p for p in directory.iterdir() if p.suffix == ".mp4"]) == 1


def test_a_render_with_caching_on_writes_a_segment_per_play(tmp_path):
    scene, _, _ = render_to_video(Moves, tmp_path, disable_caching=False, **LOSSLESS)
    directory = scene.renderer.file_writer.partial_movie_directory
    assert len([p for p in directory.iterdir() if p.suffix == ".mp4"]) == 3


def test_the_default_encoder_writes_gaps(tmp_path):
    _, drawn, video = render_to_video(HeldAround, tmp_path, hold_gaps=True)
    movie = stream_of(video)
    assert movie["times"] == run_starts(drawn)
    assert len(movie["times"]) < 21
    assert movie["stream_duration"] == movie["container_duration"] == 21


def test_gaps_refuse_x264_b_frames(tmp_path):
    options = {"x264-params": "bframes=3:b-adapt=0:scenecut=0"}
    with pytest.raises(ValueError, match="B-frames"):
        render_to_video(
            HeldAround, tmp_path, hold_gaps=True, video_encoder_options=options
        )


def test_gaps_take_other_x264_settings(tmp_path):
    options = {"x264-params": "b-adapt=0:scenecut=0"}
    _, drawn, video = render_to_video(
        HeldAround, tmp_path, hold_gaps=True, video_encoder_options=options
    )
    movie = stream_of(video)
    assert movie["times"] == run_starts(drawn)
    assert movie["stream_duration"] == 21


def test_a_sound_lasts_to_the_end_of_the_last_hold(tmp_path):
    _, drawn, video = render_to_video(Narrated, tmp_path, hold_gaps=True)
    assert len(drawn) == 30
    movie = stream_of(video)
    assert len(movie["pictures"]) == runs(drawn) < 30
    assert movie["stream_duration"] == 30
    with av.open(str(video)) as container:
        audio = container.streams.audio[0]
        heard = audio.duration * audio.time_base
    # The click starts at 1.35 s and lasts 0.6 s: it ends within the hold.
    assert Fraction(19, 10) <= heard <= Fraction(2)


def test_sections_keep_a_segment_per_play_and_their_gaps(tmp_path):
    scene, drawn, _ = render_to_video(
        HeldAround, tmp_path, hold_gaps=True, save_sections=True, **LOSSLESS
    )
    writer = scene.renderer.file_writer
    directory = writer.partial_movie_directory
    assert len([p for p in directory.iterdir() if p.suffix == ".mp4"]) == 3
    (section,) = [s for s in writer.sections if s.video is not None]
    movie = stream_of(writer.sections_output_dir / section.video)
    # Each play's segment holds its own runs: the hold's picture, which the
    # play's first frame shows too, is one frame in each.
    per_play = runs(drawn[:6]) + runs(drawn[6:12]) + runs(drawn[12:])
    assert len(movie["pictures"]) == per_play == runs(drawn) + 1
    shown = on_grid(movie)
    assert len(shown) == len(drawn)
    for index, (pixels, frame) in enumerate(zip(drawn, shown, strict=True)):
        np.testing.assert_array_equal(frame, pixels[..., :3], err_msg=f"frame {index}")


def test_a_gif_keeps_every_frame(tmp_path):
    settings = {
        **SMALL,
        "frame_rate": FPS,
        "media_dir": str(tmp_path),
        "disable_caching": True,
        "format": "gif",
        "hold_gaps": True,
    }
    with tempconfig(settings):
        scene = HeldAround()
        scene.render()
        gif = Path(scene.renderer.file_writer.gif_file_path)
    with av.open(str(gif)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 21
