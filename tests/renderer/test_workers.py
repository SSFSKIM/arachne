"""A render's frames drawn and encoded by worker processes
(``render_workers``): the movie decodes to the frames one process draws,
frame for frame, at every boundary between two workers' ranges, with and
without gaps; a render the workers can't draw is drawn in its own process,
saying why; a worker that fails fails the render and leaves nothing behind.

Each render runs in a process of its own (``render_child.py``): a test
runner's worker keeps threads, beside which a render forks no worker. How
many processes a render forked is counted there as a check outside the
engine counts it, by ``os.register_at_fork``: a render drawn in its own
process forks none.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from arachne.renderer import workers

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CHILD = HERE / "render_child.py"
SCENES = HERE / "workers_scenes.py"
TIMELINE = ROOT / "tests" / "timeline"

FPS = 15
#: PNG frames in MP4: what is decoded is what was drawn.
BASE = {
    "pixel_width": 96,
    "pixel_height": 54,
    "frame_rate": FPS,
    "disable_caching": True,
    "format": "mp4",
    "video_codec": "png",
    "pixel_format": "rgb24",
    "progress_bar": "none",
}
#: QuickTime RLE in MOV, the other lossless encoding.
QTRLE = {"format": "mov", "video_codec": "qtrle", "pixel_format": "rgb24"}
ENCODINGS = {"png": {}, "qtrle": QTRLE}


def renders(
    tmp_path: Path, scene: str, runs: list[dict], *, file: Path = SCENES
) -> list[dict]:
    """Each of ``runs`` of ``scene`` rendered in a process of its own
    (``render_child.py``), what each did and wrote.
    """
    spec = tmp_path / "spec.json"
    out = tmp_path / "out.json"
    spec.write_text(
        json.dumps(
            {
                "file": str(file),
                "scene": scene,
                "config": BASE,
                "runs": runs,
                "media": str(tmp_path / "media"),
            }
        )
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    done = subprocess.run(
        [sys.executable, str(CHILD), str(spec), str(out)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-6000:]
    return json.loads(out.read_text())


def at(workers: int, **config) -> dict:
    return {"config": {"render_workers": workers, **config}}


def drawn_by_workers(found: dict, count: int) -> None:
    """``found`` forked ``count`` workers, which drew every frame."""
    assert found["error"] is None, found["error"]
    assert found["fallback"] is None
    assert found["forks"] == count == len(found["workers"])
    # Nothing was drawn in the render's own process.
    assert found["drawn"] == []
    assert found["movie"]["off_grid"] == 0


def drawn_here(found: dict) -> None:
    """``found`` drew every frame in its own process and forked nothing."""
    assert found["error"] is None, found["error"]
    assert found["forks"] == 0
    assert found["workers"] == []
    assert found["movie"]["frames"] == found["drawn"]


# ── the ranges ───────────────────────────────────────────────────────────────


def test_ranges_are_contiguous_and_balanced_by_cost():
    assert workers.ranges([6, 9, 6, 12, 6], 2) == [(0, 3), (3, 5)]
    assert workers.ranges([6, 9, 6, 12, 6, 9], 3) == [(0, 2), (2, 4), (4, 6)]
    # A long play outweighs the short ones before it.
    assert workers.ranges([1, 1, 1, 1, 20], 2) == [(0, 4), (4, 5)]


def test_ranges_are_never_more_than_the_items():
    assert workers.ranges([5, 5, 5], 8) == [(0, 1), (1, 2), (2, 3)]
    assert workers.ranges([5, 5, 5], 3) == [(0, 1), (1, 2), (2, 3)]
    assert workers.ranges([10], 4) == [(0, 1)]
    assert workers.ranges([], 4) == []
    # Fewer items than workers, but one outweighs the two after it, which
    # share a range.
    assert workers.ranges([100, 1, 1], 4) == [(0, 1), (1, 3)]


# ── equal to one process ─────────────────────────────────────────────────────


@pytest.mark.parametrize("gaps", [False, True], ids=["every_frame", "gaps"])
@pytest.mark.parametrize("encoding", sorted(ENCODINGS))
def test_workers_write_what_one_process_draws(tmp_path, encoding, gaps):
    config = {**ENCODINGS[encoding], "hold_gaps": gaps}
    one, two, three = renders(tmp_path, "Steps", [at(n, **config) for n in (1, 2, 3)])
    assert one["fallback"] is None
    drawn_here(one)
    assert len(one["drawn"]) == 48
    for found, count in ((two, 2), (three, 3)):
        drawn_by_workers(found, count)
        assert found["movie"]["frames"] == one["drawn"]
        assert sum(w["frames"] for w in found["workers"]) == len(one["drawn"])
        assert found["frame_runs"] == one["frame_runs"]
        if gaps:
            # Each worker's stream starts a run of its own at its first frame.
            decoded = found["movie"]["decoded"]
            assert one["movie"]["decoded"] <= decoded < one["movie"]["decoded"] + count
            assert decoded < len(one["drawn"])
        else:
            assert found["movie"]["decoded"] == len(one["drawn"])


def test_a_fixed_orientation_scene_drawn_by_workers_is_the_session_s(tmp_path):
    one, two = renders(
        tmp_path,
        "FixedOrientation",
        [{**at(1), "session": True}, at(2)],
        file=TIMELINE / "scenes.py",
    )
    drawn_here(one)
    assert one["session"] == one["drawn"]
    drawn_by_workers(two, 2)
    assert two["movie"]["frames"] == one["session"]


# ── where the workers' ranges meet ───────────────────────────────────────────


@pytest.mark.parametrize("gaps", [False, True], ids=["every_frame", "gaps"])
def test_a_frozen_frame_and_the_play_after_it_meet_between_two_workers(tmp_path, gaps):
    one, two, each = renders(
        tmp_path, "FrozenThenPlays", [at(n, hold_gaps=gaps) for n in (1, 2, 8)]
    )
    drawn_here(one)
    drawn_by_workers(each, 3)
    # Every play its own worker: the frozen frame's ends at the play after it.
    assert [w["first"] for w in each["workers"]] == [0, 1, 2]
    assert each["movie"]["frames"] == one["drawn"]
    # Two workers: the cut falls one play later.
    drawn_by_workers(two, 2)
    assert [(w["first"], w["last"]) for w in two["workers"]] == [(0, 1), (2, 2)]
    assert two["movie"]["frames"] == one["drawn"]


@pytest.mark.parametrize("scene", ["Zoomed", "CentreMoved"])
def test_a_worker_starting_after_the_frame_moved_maps_as_one_process_does(
    tmp_path, scene
):
    """A MovingCamera maps each frame around its frame as it is; a flat
    camera, around the centre of its first frame, which a worker starting
    after the centre moved must keep.
    """
    one, two, three = renders(tmp_path, scene, [at(n) for n in (1, 2, 3)])
    drawn_here(one)
    for found, count in ((two, 2), (three, 3)):
        drawn_by_workers(found, count)
        assert found["workers"][-1]["first"] > 0
        assert found["movie"]["frames"] == one["drawn"]
    # Each of the three plays in a worker of its own.
    assert [w["first"] for w in three["workers"]] == [0, 1, 2]


def test_workers_draw_on_the_ir_cairo_path(tmp_path):
    pytest.importorskip("arachne_core")
    one, three = renders(
        tmp_path, "Steps", [at(n, draw_path="ir_cairo") for n in (1, 3)]
    )
    drawn_here(one)
    drawn_by_workers(three, 3)
    assert three["movie"]["frames"] == one["drawn"]


def test_more_workers_than_plays_take_one_play_each(tmp_path):
    one, eight = renders(tmp_path, "Pairs", [at(1), at(8)])
    drawn_by_workers(eight, 2)
    assert [(w["first"], w["last"]) for w in eight["workers"]] == [(0, 0), (1, 1)]
    assert eight["movie"]["frames"] == one["drawn"]


# ── drawn in the render's own process ────────────────────────────────────────

FALLBACKS = [
    pytest.param("Steps", {"disable_caching": False}, "caching", id="caching_on"),
    pytest.param("Steps", {"save_sections": True}, "sections", id="sections"),
    pytest.param("OnePlay", {}, "one play", id="one_play"),
    pytest.param("LiveLater", {}, "drawn as it runs", id="live_later"),
]


@pytest.mark.parametrize(("scene", "config", "reason"), FALLBACKS)
def test_a_render_the_workers_can_t_draw_is_drawn_in_its_own_process(
    tmp_path, scene, config, reason
):
    one, two = renders(tmp_path, scene, [at(1, **config), at(2, **config)])
    drawn_here(one)
    assert one["fallback"] is None
    drawn_here(two)
    assert reason in two["fallback"]
    assert two["drawn"] == one["drawn"]


def test_a_live_camera_is_drawn_in_its_own_process(tmp_path):
    (found,) = renders(tmp_path, "Mapped", [at(2)], file=TIMELINE / "camera_scenes.py")
    drawn_here(found)
    assert "drawn as it runs" in found["fallback"]


def test_a_gif_is_drawn_in_its_own_process(tmp_path):
    (found,) = renders(tmp_path, "Steps", [at(2, format="gif")])
    assert found["error"] is None
    assert found["forks"] == 0
    assert "GIF" in found["fallback"]


def test_a_thread_alive_at_fork_time_is_refused(tmp_path):
    held, free = renders(tmp_path, "Steps", [{**at(2), "thread": True}, at(2)])
    drawn_here(held)
    assert "thread" in held["fallback"]
    assert "held" in held["fallback"]
    # Without the thread, the same render forks its workers.
    drawn_by_workers(free, 2)
    assert free["movie"]["frames"] == held["drawn"]


#: Each way a render whose plays wait for its workers draws them in its own
#: process instead, on a scene that drew a skipped frame's static picture
#: while they waited, after its frame centre moved.
SKIPPED_CENTRE = [
    pytest.param("SkippedCentre", {"thread": True}, "thread", id="thread"),
    pytest.param("SkippedCentreOnePlay", {}, "one play", id="one_play"),
    pytest.param("SkippedCentreLiveLater", {}, "drawn as it runs", id="live_later"),
    pytest.param("SkippedCentreZoomed", {}, "MultiCamera", id="multi_camera"),
    pytest.param("SkippedCentreCameraCode", {}, "draws through", id="camera_code"),
    pytest.param(
        "SkippedCentreCameraCodeUndone",
        {},
        "replaced its camera's",
        id="camera_code_undone",
    ),
]


@pytest.mark.parametrize(("scene", "run", "reason"), SKIPPED_CENTRE)
def test_plays_drawn_here_after_waiting_map_as_one_process_does(
    tmp_path, scene, run, reason
):
    one, two = renders(tmp_path, scene, [at(1), {**at(2), **run}])
    drawn_here(one)
    drawn_here(two)
    assert reason in two["fallback"]
    assert two["movie"]["frames"] == one["drawn"]


def test_plays_drawn_by_workers_after_a_skipped_static_map_as_one_process_does(
    tmp_path,
):
    one, two = renders(tmp_path, "SkippedCentre", [at(1), at(2)])
    drawn_here(one)
    drawn_by_workers(two, 2)
    assert two["movie"]["frames"] == one["drawn"]


#: Why a render whose script reads a frame through the renderer draws every
#: play in its own process from then on.
READ = "reads a frame"

#: Scenes whose script reads a frame through the renderer, each beside the
#: same scene reading none.
READS = [
    pytest.param("FrameRead", "FrameNotRead", id="drawn_after_a_play"),
    pytest.param("FrameGot", "FrameNotRead", id="held_after_a_play"),
    pytest.param("FrameReadFirst", "FrameNotReadFirst", id="drawn_before_any_play"),
    pytest.param(
        "SkippedStaticFrameGot", "SkippedStaticNotRead", id="held_after_a_skipped_one"
    ),
    pytest.param(
        "SkippedCentreSnapshot", "SkippedCentre", id="drawn_over_a_skipped_one"
    ),
]


@pytest.mark.parametrize(("scene", "unread"), READS)
def test_a_frame_the_script_reads_is_one_process_s(tmp_path, scene, unread):
    """Drawn now (``update_frame``) or as the renderer holds it
    (``get_frame``), whenever it is read, the frame the script reads is one
    process's, and so is every frame after it.
    """
    one, two = renders(tmp_path, scene, [at(1), at(2)])
    drawn_here(one)
    drawn_here(two)
    assert READ in two["fallback"]
    assert two["movie"]["frames"] == one["drawn"]
    # The same scene reading no frame is drawn by the workers.
    one, two = renders(tmp_path, unread, [at(1), at(2)])
    drawn_by_workers(two, 2)
    assert two["movie"]["frames"] == one["drawn"]


@pytest.mark.xfail(
    strict=True,
    reason="a frame read from the camera itself does not reach the renderer"
    " (tech-debt-tracker.md, Render workers)",
)
def test_a_frame_read_from_the_camera_is_one_process_s(tmp_path):
    one, two = renders(tmp_path, "CameraImageGot", [at(1), at(2)])
    drawn_here(one)
    assert two["movie"]["frames"] == one["drawn"]


# ── the configuration a waiting play is drawn under ──────────────────────────

#: Read through ``draw_reuse``'s getter (with the levers off, every lever
#: reads as its scalar); the snapshot holds ``draw_reuse`` resolved.
THROUGH_DRAW_REUSE = frozenset({"speed_levers"})
#: The configured pixel shape, which sets how thick a point is drawn: a
#: change refuses the record's plays from then on (``PIXEL_SHAPE_CHANGED``),
#: which are drawn as they run.
PIXEL_SHAPE = frozenset({"pixel_width", "pixel_height"})

#: Scenes drawn by the invariant below: flat, moving and 3D cameras, the
#: static layer off, gaps, a skipped section, text and an image, points.
DRAWN = [
    pytest.param("Steps", {}, id="flat"),
    pytest.param("Steps", {"static_layer": False}, id="no_static_layer"),
    pytest.param("Steps", {"hold_gaps": True}, id="gaps"),
    pytest.param("Zoomed", {}, id="moving_camera"),
    pytest.param("Turned3D", {}, id="three_d"),
    pytest.param("SkippedCentre", {}, id="skipped_section"),
    pytest.param("TextAndImage", {}, id="text_and_image"),
    pytest.param("Points", {}, id="points"),
]


@pytest.mark.parametrize(("scene", "config"), DRAWN)
def test_a_waiting_play_is_drawn_under_the_configuration_it_ran_under(
    tmp_path, scene, config
):
    """Drawn after the scene ran, in this process or by a worker, a play
    reads no configuration but what it took as it ran (``_DrawConfig``) and
    what the record guards.
    """
    from dataclasses import fields

    from arachne.renderer.cairo_renderer import _DrawConfig

    here, forked = renders(
        tmp_path,
        scene,
        [
            {**at(2, **config), "thread": True, "config_reads": True},
            {**at(2, **config), "config_reads": True},
        ],
    )
    drawn_here(here)
    drawn_by_workers(forked, 2)
    taken = frozenset(f.name for f in fields(_DrawConfig))
    allowed = taken | THROUGH_DRAW_REUSE | PIXEL_SHAPE
    reads = {"this process": here["config_reads"]["here"]} | {
        f"worker {n}": keys for n, keys in enumerate(forked["config_reads"]["workers"])
    }
    assert len(reads) == 3
    for where, keys in reads.items():
        unknown = sorted(set(keys) - allowed)
        assert not unknown, (
            f"Drawing a play that waited for the workers, {where} read the"
            f" configuration's {unknown}, which the play did not take as it"
            " ran. Add each to _DrawConfig (cairo_renderer.py), the"
            " configuration a waiting play carries, or make a change of it"
            " during the render a reason to draw the render in its own"
            " process."
        )


#: A frame read, or not, after the static layer is turned off midway.
LAYER_CHANGED = [
    pytest.param("LayerChanged", "LayerChangedUnread", id="after_a_play"),
    pytest.param(
        "LayerChangedSkipped", "LayerChangedSkippedUnread", id="after_a_skipped_one"
    ),
]


@pytest.mark.parametrize(("scene", "unread"), LAYER_CHANGED)
def test_each_play_is_drawn_under_the_configuration_it_ran_under(
    tmp_path, scene, unread
):
    one, two = renders(tmp_path, scene, [at(1), at(2)])
    drawn_here(one)
    drawn_here(two)
    assert READ in two["fallback"]
    assert two["movie"]["frames"] == one["drawn"]
    # Read no frame, the same scene is drawn by the workers.
    one, two = renders(tmp_path, unread, [at(1), at(2)])
    drawn_by_workers(two, 2)
    assert two["movie"]["frames"] == one["drawn"]


def test_plays_drawn_on_another_path_midway_are_drawn_on_it_by_workers(tmp_path):
    pytest.importorskip("arachne_core")
    one, two, three = renders(tmp_path, "DrawPathChanged", [at(1), at(2), at(3)])
    drawn_here(one)
    for found, count in ((two, 2), (three, 3)):
        drawn_by_workers(found, count)
        assert found["movie"]["frames"] == one["drawn"]


# ── a worker that fails ──────────────────────────────────────────────────────


@pytest.mark.parametrize("how", ["raise", "kill"])
@pytest.mark.parametrize("play", [0, 5], ids=["first_range", "last_range"])
def test_a_failed_worker_fails_the_render_and_leaves_nothing(tmp_path, how, play):
    (found,) = renders(
        tmp_path, "Steps", [{**at(3), "inject": {"how": how, "play": play}}]
    )
    assert found["error_type"] == "WorkerFailed", found["error"]
    assert "worker" in found["error"]
    if how == "raise":
        assert "the worker's drawing failed" in found["error"]
    else:
        assert "signal" in found["error"]
    assert found["forks"] == 3
    assert not found["movie_exists"]
    assert found["left"] == []
    assert not found["children"]


#: x264 asked for B-frames, which a video with gaps refuses once its encoder
#: is opened: in each worker, on the worker's own stream.
B_FRAMES = {
    "video_codec": "libx264",
    "pixel_format": "yuv420p",
    "video_encoder_options": {"bf": "2"},
}


def test_a_worker_s_encoder_that_writes_b_frames_under_gaps_fails_the_render(
    tmp_path,
):
    refused, taken = renders(
        tmp_path,
        "Steps",
        [at(3, hold_gaps=True, **B_FRAMES), at(3, hold_gaps=False, **B_FRAMES)],
    )
    assert refused["error_type"] == "WorkerFailed", refused["error"]
    assert "B-frames" in refused["error"]
    assert refused["forks"] == 3
    assert not refused["movie_exists"]
    assert refused["left"] == []
    assert not refused["children"]
    # Without gaps the same encoder is taken, by the same workers.
    assert taken["error"] is None, taken["error"]
    assert taken["fallback"] is None
    assert taken["forks"] == 3 == len(taken["workers"])
    assert taken["movie_exists"]


def test_a_frame_the_core_cannot_draw_in_a_worker_draws_the_scene_on_cairo(
    tmp_path,
):
    pytest.importorskip("arachne_core")
    one, refused = renders(
        tmp_path,
        "Steps",
        [
            at(1, draw_path="cairo"),
            {**at(3, draw_path="core"), "inject": {"how": "unable", "play": 3}},
        ],
    )
    # Executed again on Cairo, by workers again: three forked for each
    # execution.
    assert refused["error"] is None, refused["error"]
    assert refused["forks"] == 6
    assert len(refused["workers"]) == 3
    assert refused["movie"]["frames"] == one["drawn"]


# ── what a worker reports ────────────────────────────────────────────────────


def test_each_worker_reports_its_own_peak_memory(tmp_path):
    read, real = renders(tmp_path, "Steps", [{**at(3), "rss_pid": True}, at(3)])
    drawn_by_workers(read, 3)
    # The reader ran in each worker: each read its own process.
    peaks = [w["peak_mb"] for w in read["workers"]]
    assert len(set(peaks)) == 3
    assert read["pid"] not in peaks
    drawn_by_workers(real, 3)
    assert all(w["peak_mb"] > 0 for w in real["workers"])


# ── many forked renders ──────────────────────────────────────────────────────


def test_fifty_forked_renders_write_the_same_movie(tmp_path):
    """Forking after the scene's drawing libraries are loaded neither
    crashes nor hangs a worker, and every render's movie is the same.
    """
    found = renders(tmp_path, "Pairs", [{**at(2), "repeat": 50}])
    assert len(found) == 50
    for each in found:
        drawn_by_workers(each, 2)
    assert {each["movie_sha"] for each in found} == {found[0]["movie_sha"]}


# ── the command line ─────────────────────────────────────────────────────────

#: The command line run in this process, counting the forks it makes.
COMMAND = textwrap.dedent(
    """
    import json, os, sys
    forks = [0]
    os.register_at_fork(before=lambda: forks.__setitem__(0, forks[0] + 1))
    from arachne.__main__ import main
    try:
        main(sys.argv[2:], standalone_mode=False)
    finally:
        with open(sys.argv[1], "w") as out:
            json.dump(forks[0], out)
    """
)


def test_the_command_line_renders_with_workers(tmp_path):
    movies = {}
    for count in ("1", "2"):
        media = tmp_path / f"media-{count}"
        forks = tmp_path / f"forks-{count}.json"
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                COMMAND,
                str(forks),
                "render",
                "-ql",
                "--disable_caching",
                "--format",
                "mp4",
                "--media_dir",
                str(media),
                "--workers",
                count,
                str(SCENES),
                "Pairs",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert done.returncode == 0, done.stderr[-4000:]
        assert json.loads(forks.read_text()) == (0 if count == "1" else 2)
        (movie,) = media.rglob("Pairs.mp4")
        movies[count] = movie
    from tests.timeline.test_video_output import decoded

    one, two = (decoded(movies[c]) for c in ("1", "2"))
    assert len(one) == len(two) > 0
