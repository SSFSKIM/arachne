"""The writer's movie: one stream for the plays drawn with caching off, and a
run of equal frames written as one frame that lasts the run (``hold_gaps``).

These tests hand the writer frames directly (no scene is drawn) and decode
what it wrote: each frame's presentation time on the frame grid, the time
the stream ends, and the frames expanded on the grid.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from arachne import Scene, tempconfig
from arachne._config.output import OutputFormat, OutputSpec
from arachne._config.video_encoder import (
    resolve_video_encoder,
    video_encoder_fingerprint,
    with_thread_share,
)
from arachne.scene.scene_file_writer import uncached_key
from arachne.scene.video_segment_encoder import VideoSegmentEncoder

WIDTH, HEIGHT, FPS = 32, 24, 15

#: Encodings whose decoded colours are the ones written.
LOSSLESS = {
    "png": {"format": "mp4", "video_codec": "png", "pixel_format": "rgb24"},
    "qtrle": {"format": "mov", "video_codec": "qtrle", "pixel_format": "rgb24"},
}


def picture(index: int) -> np.ndarray:
    """A distinct opaque RGBA picture for each ``index``."""
    rng = np.random.default_rng(index)
    pixels = rng.integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return pixels


def writer_for(media_dir: Path, **settings):
    """A scene's file writer, from the configuration ``settings`` amend."""
    with tempconfig(
        {
            "media_dir": str(media_dir),
            "pixel_width": WIDTH,
            "pixel_height": HEIGHT,
            "frame_rate": FPS,
            "disable_caching": True,
            "format": "mp4",
            **settings,
        }
    ):
        return Scene().renderer.file_writer


def write_plays(writer, plays: list[list[tuple[np.ndarray, int]]], *, cut_before=()):
    """Write each play's ``(pixels, repeat)`` writes as a play drawn with
    caching off, cutting the stream before the plays ``cut_before`` names;
    then finish the movie.
    """
    for index, writes in enumerate(plays):
        if index in cut_before:
            writer.cut_stream()
        writer.add_partial_movie_file(uncached_key(index))
        writer.begin_animation(True, animation_index=index)
        for pixels, repeat in writes:
            writer.write_frame(pixels.copy(), repeat=repeat)
        writer.end_animation(True)
    writer.finish()
    return Path(writer.movie_file_path)


def decoded(video: Path) -> dict:
    """The video stream's frames (RGB), their times in frames of the grid,
    the time its last frame ends, and its stream and format durations.
    """
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        base = Fraction(stream.time_base)
        packets = [
            (p.pts, p.duration) for p in container.demux(stream) if p.pts is not None
        ]
        duration = None if stream.duration is None else stream.duration * base
        container_duration = in_frames(container.duration)
    with av.open(str(video)) as container:
        frames = list(container.decode(video=0))
        rate = container.streams.video[0].codec_context.framerate
    origin = min(pts for pts, _ in packets)
    return {
        "times": [(f.pts - origin) * base * FPS for f in frames],
        "pictures": [f.to_ndarray(format="rgb24") for f in frames],
        "end": (max(p + d for p, d in packets) - origin) * base * FPS,
        "stream_duration": None if duration is None else duration * FPS,
        "container_duration": container_duration,
        "declared_rate": rate,
    }


def in_frames(microseconds: int) -> Fraction:
    """A container's duration (whole microseconds) in frames of the grid:
    the nearest whole frame when the duration is that frame's, rounded.
    """
    frames = Fraction(microseconds * FPS, av.time_base)
    nearest = round(frames)
    return (
        Fraction(nearest)
        if round(Fraction(nearest * av.time_base, FPS)) == microseconds
        else frames
    )


def on_grid(movie: dict) -> list[np.ndarray]:
    """The decoded frames expanded on the frame grid: each holds from its
    time to the next frame's, the last to the stream's end.
    """
    ends = [*movie["times"][1:], movie["end"]]
    shown = []
    for start, end, pixels in zip(movie["times"], ends, movie["pictures"], strict=True):
        assert start.denominator == 1, start
        assert end.denominator == 1, end
        shown += [pixels] * int(end - start)
    return shown


def expanded(plays) -> list[np.ndarray]:
    return [
        pixels[..., :3]
        for writes in plays
        for pixels, repeat in writes
        for _ in range(repeat)
    ]


def assert_frames(shown, written):
    assert len(shown) == len(written)
    for index, (a, b) in enumerate(zip(shown, written, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"grid frame {index}")


#: Two plays: a frozen frame written once with a repeat, then frames whose
#: equal neighbours are separate writes of equal (not identical) arrays, the
#: play ending in a hold.
P0 = [(picture(0), 4)]
P1 = [
    (picture(1), 1),
    (picture(1), 1),
    (picture(2), 1),
    (picture(3), 2),
    (picture(3), 1),
    (picture(3), 1),
]


@pytest.mark.parametrize("encoding", sorted(LOSSLESS))
def test_a_run_of_equal_frames_is_one_frame_that_lasts_the_run(tmp_path, encoding):
    plays = [P0, P1]
    movie = decoded(
        write_plays(writer_for(tmp_path, hold_gaps=True, **LOSSLESS[encoding]), plays)
    )

    # One frame per run: 4 | 2, 1, 4.
    assert movie["times"] == [0, 4, 6, 7]
    assert movie["end"] == 11  # the trailing hold's duration is kept
    assert movie["container_duration"] == 11
    assert_frames(on_grid(movie), expanded(plays))


@pytest.mark.parametrize("encoding", sorted(LOSSLESS))
def test_without_gaps_every_frame_of_the_grid_is_written(tmp_path, encoding):
    plays = [P0, P1]
    movie = decoded(write_plays(writer_for(tmp_path, **LOSSLESS[encoding]), plays))

    assert movie["times"] == list(range(11))
    assert movie["end"] == 11
    assert_frames(movie["pictures"], expanded(plays))


def test_frames_differing_in_one_byte_are_two_frames(tmp_path):
    first = picture(4)
    alpha = first.copy()
    alpha[0, 0, 3] = 254  # differs in alpha only
    colour = first.copy()
    colour[-1, -1, 0] ^= 1  # differs in one colour byte
    plays = [[(first, 1), (first, 1), (alpha, 1), (colour, 2)]]
    movie = decoded(
        write_plays(writer_for(tmp_path, hold_gaps=True, **LOSSLESS["qtrle"]), plays)
    )

    assert movie["times"] == [0, 2, 3]
    assert movie["end"] == 5


def test_a_stream_that_ends_on_a_single_frame_ends_one_frame_after_it(tmp_path):
    plays = [[(picture(5), 3), (picture(6), 1)]]
    movie = decoded(
        write_plays(writer_for(tmp_path, hold_gaps=True, **LOSSLESS["qtrle"]), plays)
    )

    assert movie["times"] == [0, 3]
    assert movie["end"] == 4


@pytest.mark.parametrize("encoding", sorted(LOSSLESS))
def test_plays_drawn_with_caching_off_share_one_stream(tmp_path, encoding):
    writer = writer_for(tmp_path, hold_gaps=True, **LOSSLESS[encoding])
    write_plays(writer, [P0, P1, P0])

    segments = sorted(
        p.name
        for p in writer.partial_movie_directory.iterdir()
        if not p.name.endswith(".txt")
    )
    assert segments == [f"{uncached_key(0)}_on{writer.output_plan.segment_extension}"]


def test_plays_drawn_with_caching_on_keep_a_segment_each(tmp_path):
    writer = writer_for(
        tmp_path, hold_gaps=True, disable_caching=False, **LOSSLESS["qtrle"]
    )
    for index, writes in enumerate([P0, P1]):
        writer.add_partial_movie_file(f"key{index}")
        writer.begin_animation(True, animation_index=index)
        for pixels, repeat in writes:
            writer.write_frame(pixels.copy(), repeat=repeat)
        writer.end_animation(True)
    writer.finish()

    segments = sorted(
        p.name
        for p in writer.partial_movie_directory.iterdir()
        if not p.name.endswith(".txt")
    )
    assert segments == ["key0.mov", "key1.mov"]
    movie = decoded(Path(writer.movie_file_path))
    assert movie["times"] == [0, 4, 6, 7]
    assert movie["end"] == 11


@pytest.mark.parametrize("encoding", sorted(LOSSLESS))
def test_a_stream_cut_in_two_keeps_every_frame_s_time(tmp_path, encoding):
    """Each piece ends in a hold: the second starts where the first's hold
    ends, and the movie ends where the second's does.
    """
    plays = [P1, P0, P1]
    writer = writer_for(tmp_path, hold_gaps=True, **LOSSLESS[encoding])
    movie = decoded(write_plays(writer, plays, cut_before={1, 2}))

    pieces = [
        p
        for p in writer.partial_movie_directory.iterdir()
        if not p.name.endswith(".txt")
    ]
    assert len(pieces) == 3
    assert movie["times"] == [0, 2, 3, 7, 11, 13, 14]
    assert movie["end"] == 18
    assert_frames(on_grid(movie), expanded(plays))


def test_a_cached_play_between_uncached_plays_splits_the_stream(tmp_path):
    writer = writer_for(tmp_path, hold_gaps=True, **LOSSLESS["qtrle"])
    for index, (key, writes) in enumerate(
        [(uncached_key(0), P1), ("cached", P0), (uncached_key(2), P1)]
    ):
        writer.add_partial_movie_file(key)
        writer.begin_animation(True, animation_index=index)
        for pixels, repeat in writes:
            writer.write_frame(pixels.copy(), repeat=repeat)
        writer.end_animation(True)
    writer.finish()

    segments = sorted(
        p.name
        for p in writer.partial_movie_directory.iterdir()
        if not p.name.endswith(".txt")
    )
    assert segments == [
        "cached.mov",
        f"{uncached_key(0)}_on.mov",
        f"{uncached_key(2)}_on.mov",
    ]
    movie = decoded(Path(writer.movie_file_path))
    assert movie["times"] == [0, 2, 3, 7, 11, 13, 14]
    assert movie["end"] == 18


@pytest.mark.parametrize("save_sections", [False, True])
def test_a_play_drawn_with_caching_off_is_never_found_cached(tmp_path, save_sections):
    writer = writer_for(tmp_path, save_sections=save_sections, **LOSSLESS["qtrle"])
    stale = writer.output_plan.segment_path(uncached_key(0))
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"left by another render")

    assert not writer.is_already_cached(uncached_key(0))


def test_a_play_drawn_with_caching_on_is_found_cached(tmp_path):
    writer = writer_for(tmp_path, **LOSSLESS["qtrle"])
    kept = writer.output_plan.segment_path("key")
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"written by an earlier render")

    assert writer.is_already_cached("key")


def test_the_default_encoder_writes_gaps_without_b_frames(tmp_path):
    """Under ``hold_gaps`` x264 writes no B-frames (see
    ``test_b_frames_would_cut_a_trailing_hold_short``). Every run lasts two
    frames, so no frame time is one frame
    from the next, and the rate the stream declares is still the scene's.
    """
    plays = [[(picture(i), 2) for i in range(12)]]
    writer = writer_for(tmp_path, hold_gaps=True)
    assert writer.video_encoder.codec == "libx264"
    movie = decoded(write_plays(writer, plays))

    assert movie["times"] == list(range(0, 24, 2))
    assert movie["stream_duration"] == 24
    assert movie["container_duration"] == 24
    assert movie["declared_rate"] == FPS


def test_without_gaps_the_default_encoder_writes_every_frame(tmp_path):
    plays = [[(picture(i), 2) for i in range(12)]]
    movie = decoded(write_plays(writer_for(tmp_path), plays))

    assert movie["times"] == list(range(24))
    assert movie["stream_duration"] == 24
    assert movie["declared_rate"] == FPS


def test_a_gif_is_written_at_a_constant_rate(tmp_path):
    plays = [P0, P1]
    writer = writer_for(tmp_path, hold_gaps=True, format="gif")
    assert not writer.video_encoder.gaps
    video = write_plays(writer, plays)
    with av.open(str(video)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 11


def test_rgb_h264_segments_are_joined(tmp_path):
    plays = [P0, P1]
    writer = writer_for(
        tmp_path,
        hold_gaps=True,
        video_codec="libx264rgb",
        pixel_format="rgb24",
        video_encoder_options={"qp": "0"},  # lossless
    )
    movie = decoded(write_plays(writer, plays, cut_before={1}))

    assert movie["times"] == [0, 4, 6, 7]
    assert movie["container_duration"] == 11
    assert_frames(on_grid(movie), expanded(plays))


# ── the encoder settings ─────────────────────────────────────────────────────


def _mp4(**kwargs):
    output = OutputSpec(
        OutputFormat.MP4,
        transparent=False,
        save_sections=False,
        fallback_to_still=False,
    )
    return resolve_video_encoder(
        output, width=WIDTH, height=HEIGHT, frame_rate=FPS, **kwargs
    )


def test_gaps_turn_b_frames_off_and_change_the_cache_identity():
    plain, gapped = _mp4(), _mp4(gaps=True)

    assert gapped.gaps
    assert not plain.gaps
    assert dict(gapped.options) == {**dict(plain.options), "bf": "0"}
    assert video_encoder_fingerprint(gapped) != video_encoder_fingerprint(plain)


def test_without_gaps_the_encoder_settings_are_unchanged():
    assert _mp4() == _mp4(gaps=False)
    assert dict(_mp4().options) == {"crf": "23"}


@pytest.mark.parametrize(
    "asked",
    [
        {"bf": "2"},
        {"x264-params": "bframes=3:b-adapt=0:scenecut=0"},
        {"x264opts": "scenecut=0:bframes=3"},
        {"x264-params": "b_adapt=0:bframes=1"},
    ],
    ids=["bf", "x264-params", "x264opts", "x264-params-last"],
)
def test_gaps_refuse_b_frames_asked_for_in_any_spelling(asked):
    with pytest.raises(ValueError, match="B-frames"):
        _mp4(gaps=True, options=asked)


@pytest.mark.parametrize(
    "options",
    [
        {"bf": "0"},
        {"x264-params": "bframes=0:b-adapt=0:scenecut=0"},
        {"x264opts": "scenecut=0"},
    ],
    ids=["bf", "x264-params", "x264opts"],
)
def test_gaps_take_settings_that_ask_for_no_b_frames(options):
    spec = _mp4(gaps=True, options=options)
    assert dict(spec.options) == {"crf": "23", **options, "bf": "0"}


def test_without_gaps_b_frames_may_be_asked_for():
    asked = {"x264-params": "bframes=3:b-adapt=0:scenecut=0"}
    assert dict(_mp4(options=asked).options) == {"crf": "23", **asked}


def test_b_frames_would_cut_a_trailing_hold_short(tmp_path):
    """Why gaps take no B-frames: with them, this run pattern (pictures and
    size as here) comes out of libx264 into MP4 with a stream two frames
    short of its last frame's end; without, exact. Should this ever pass
    with B-frames, ``bf=0`` could be revisited.
    """
    runs = [1, 4, 1, 1, 6]

    def stream_duration(spec, target):
        encoder = VideoSegmentEncoder(target=target, spec=spec)
        for index, repeat in enumerate(runs):
            encoder.write_frame(picture(index), repeat=repeat)
        encoder.finish()
        return decoded(target)["stream_duration"]

    gapped = _mp4(gaps=True)
    with_b_frames = replace(
        gapped, options=tuple(sorted({**dict(gapped.options), "bf": "3"}.items()))
    )
    assert stream_duration(gapped, tmp_path / "without.mp4") == 13
    assert stream_duration(with_b_frames, tmp_path / "with.mp4") == 11


def test_a_gif_s_segments_have_no_gaps():
    output = OutputSpec(
        OutputFormat.GIF,
        transparent=False,
        save_sections=False,
        fallback_to_still=False,
    )
    spec = resolve_video_encoder(
        output, width=WIDTH, height=HEIGHT, frame_rate=FPS, gaps=True
    )
    assert not spec.gaps
    assert "bf" not in dict(spec.options)


def test_an_encoder_s_threads_are_capped_at_its_share_of_the_cores():
    spec = _mp4()
    assert dict(with_thread_share(spec, workers=4, cores=8).options)["threads"] == "2"
    assert dict(with_thread_share(spec, workers=3, cores=8).options)["threads"] == "2"
    assert dict(with_thread_share(spec, workers=16, cores=8).options)["threads"] == "1"


@pytest.mark.parametrize(
    ("asked", "shared"),
    [
        ({"threads": "6"}, {"threads": "2"}),
        ({"threads": "0"}, {"threads": "2"}),
        ({"threads": "auto"}, {"threads": "2"}),
        (
            {"x264-params": "threads=8:bframes=0"},
            {"threads": "2", "x264-params": "threads=2:bframes=0"},
        ),
        (
            {"x264opts": "scenecut=0:threads=auto"},
            {"threads": "2", "x264opts": "scenecut=0:threads=2"},
        ),
    ],
    ids=["above", "zero", "auto", "x264-params", "x264opts"],
)
def test_threads_asked_for_are_capped_at_the_share(asked, shared):
    spec = with_thread_share(_mp4(options=asked), workers=4, cores=8)
    assert dict(spec.options) == {"crf": "23", **shared}


def test_threads_asked_for_within_the_share_are_kept():
    spec = with_thread_share(
        _mp4(options={"threads": "1", "x264-params": "threads=1:scenecut=0"}),
        workers=4,
        cores=8,
    )
    assert dict(spec.options)["threads"] == "1"
    assert dict(spec.options)["x264-params"] == "threads=1:scenecut=0"


def test_one_encoder_keeps_the_settings():
    spec = _mp4()
    assert with_thread_share(spec, workers=1, cores=8) == spec
    asked = _mp4(options={"threads": "6"})
    assert with_thread_share(asked, workers=1, cores=8) == asked
