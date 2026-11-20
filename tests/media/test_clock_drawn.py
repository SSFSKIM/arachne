"""The media clock drawn on a device, the launch mode's: the stamp read from
a session's frame at an instant is the frame the clip's clock program names
there, for the five programs, at every tick and between ticks, in any
order; a clip runs on through its fade and a move, and through a window; a
frame first needed late that decodes otherwise is refused, and so is a
frame showing a clip whose clock ran out. Collected only where the GPU
build is installed.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arachne.assets.reader import FIXTURES
from arachne.lint.codes import CODES
from arachne.mobject.media import frames
from arachne.mobject.media.frames import FrameCache
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError

from .clock_programs import CLIP, PROGRAMS, START, clip_time, frame_of
from .conftest import SMALL, copied, line_of

SCENES = Path(__file__).parent / "clock_scenes.py"
#: The test clip's stamp: six cells, least significant first, each this
#: wide and tall, the first this far from the clip's left and top edges,
#: each the next's this far apart (the fixture generator's numbers).
CELL, AT, PITCH, BITS = (10, 8), (2, 2), 12, 6
CLIP_PIXELS = (96, 54)
#: A session's frames on every tick: its frame rate the tick's.
EVERY_TICK = {"frame_rate": 60}


def ticks(n: int) -> list[Fraction]:
    """The ticks from 0 to ``n`` sixtieths."""
    return [Fraction(k, 60) for k in range(n + 1)]


def stamp(picture: np.ndarray, box: tuple[float, float, float, float]) -> int | None:
    """The number the clip's stamp shows in ``picture``, where the clip
    fills ``box`` (left, top, right, bottom, in pixels); ``None`` where the
    clip is too faint to read. A set cell is brighter than an unset one in
    any fade; a stamp whose cells all read alike is 0, the one such number
    a clip of 48 frames has, read only where its strip is opaque.
    """
    left, top, right, bottom = box
    across = (right - left) / CLIP_PIXELS[0]
    down = (bottom - top) / CLIP_PIXELS[1]
    values = []
    for bit in range(BITS):
        x = AT[0] + bit * PITCH + CELL[0] / 2
        y = AT[1] + CELL[1] / 2
        column = math.floor(left + x * across)
        row = math.floor(top + y * down)
        values.append(float(picture[row, column, :3].astype(int).mean()))
    low, high = min(values), max(values)
    if high - low < 40:
        return 0 if abs(low - 32) <= 4 else None
    middle = (low + high) / 2
    return sum(1 << bit for bit, value in enumerate(values) if value > middle)


def _stamps(session, instants, box=None) -> list[tuple[Fraction, int | None]]:
    """Each instant's answer: the instant it is of, and the clip's stamp in
    its frame (in ``box``, else where the clip is drawn).
    """
    found = []
    for answer in session.answers(list(instants)):
        if box is None:
            (node,) = [n for n in answer.state.nodes if n.type == "VideoMobject"]
            where = node.screen_box
        else:
            where = box
        found.append((answer.state.as_of, stamp(np.asarray(answer.frame), where)))
    return found


#: The scenes' end: their last play, a wait from 0.37 s, ends between two
#: ticks.
END = START + 2
#: The last tick before it.
LAST_TICK = Fraction(math.floor(END * 60), 60)


def answered(t: Fraction) -> Fraction:
    """The instant a session at 60 frames a second answers a request for
    ``t`` with, from how the scenes are built: the end's own capture from
    the end on; else the last tick at or before ``t``, where that tick lies
    in the play that starts at 0.37 s; else the first play's one state, its
    start's (a still wait, before any clock runs).
    """
    if t >= END:
        return END
    tick = Fraction(math.floor(t * 60), 60)
    return tick if tick > START else Fraction(0)


#: Instants between ticks, among them the first and last instants of the
#: plays that start at 0.37 s.
OFF_TICK = [START, END, Fraction(1234, 1000), Fraction(61, 120)]


@pytest.mark.parametrize("program", sorted(PROGRAMS))
def test_the_stamp_is_the_frame_the_program_names(session_of, door, program):
    session = session_of(program, SCENES, **EVERY_TICK)
    instants = ticks(142) + OFF_TICK
    found = _stamps(session, instants, box=(0, 0, *CLIP_PIXELS))
    for t, (as_of, read) in zip(instants, found, strict=True):
        assert as_of == answered(t), (t, as_of)
        # At the end, the stated limit (test_clock.py): a play ending between
        # two ticks ends on its last tick's frame.
        drawn_at = LAST_TICK if as_of == END else as_of
        assert read == frame_of(clip_time(program, drawn_at), CLIP), (t, as_of)
    # Asked out of order, each instant gives what it gave in order.
    backward = _stamps(session, reversed(instants), box=(0, 0, *CLIP_PIXELS))
    assert backward == found[::-1]


def test_a_clip_at_24_a_second_shows_frame_k_times_24_over_60_at_tick_k(
    session_of, door
):
    session = session_of("ClipRuns", SCENES, **EVERY_TICK)
    found = _stamps(session, ticks(120), box=(0, 0, *CLIP_PIXELS))
    assert [read for _, read in found] == [
        min(math.floor(k * 24 / 60), 47) for k in range(121)
    ]


def test_a_clip_looped_at_rate_minus_one_wraps_downward(session_of, door):
    session = session_of("LoopsBackward", SCENES, **EVERY_TICK)
    found = _stamps(session, ticks(60), box=(0, 0, *CLIP_PIXELS))
    expected = [math.floor((Fraction(-k, 60) % 2) * 24) for k in range(61)]
    assert [read for _, read in found] == expected
    assert expected[:3] == [0, 47, 47]


def test_a_program_changed_between_plays_leaves_every_earlier_instant(session_of, door):
    instants = ticks(120)
    runs = session_of("ClipRuns", SCENES, **EVERY_TICK).answers(instants)
    changes = session_of("ClipRunsThenChanges", SCENES, **EVERY_TICK).answers(instants)
    for t, before, after in zip(instants, runs, changes, strict=True):
        if t <= 1:
            assert np.array_equal(np.asarray(before.frame), np.asarray(after.frame)), t
        else:
            media = (1 - (t - 1)) % 2
            read = stamp(np.asarray(after.frame), (0, 0, *CLIP_PIXELS))
            assert read == math.floor(media * 24), t


def test_a_running_clip_runs_through_its_fade_in_and_a_move(session_of, door):
    session = session_of("ClipFadedInAndMoved", SCENES, **EVERY_TICK)
    found = _stamps(session, ticks(60))
    for k, (as_of, read) in enumerate(found):
        assert as_of == Fraction(k, 60)
        if as_of >= Fraction(1, 4):
            # Half faded in or more: the stamp reads.
            assert read is not None, as_of
        if read is not None:
            assert read == math.floor(k * 24 / 60), as_of


@pytest.mark.parametrize("scene", ["HeldBeforeTheFade", "RunThenHold"])
def test_a_held_clip_shows_its_last_frame_through_its_fade(session_of, door, scene):
    session = session_of(scene, SCENES, **EVERY_TICK)
    instants = [Fraction(2) + Fraction(k, 60) for k in range(0, 13)]
    found = _stamps(session, instants)
    assert [read for _, read in found] == [47] * len(instants)


def test_a_clip_the_scene_holds_and_a_window_shows_is_drawn_once_inside_it(
    session_of, door
):
    session = session_of("ClipInWindow", SCENES, **EVERY_TICK)
    half = CLIP_PIXELS[0] // 2
    for k, answer in enumerate(session.answers(ticks(60))):
        picture = np.asarray(answer.frame)
        assert stamp(picture, (0, 0, half, CLIP_PIXELS[1])) == math.floor(k * 24 / 60)
        # Nothing of it outside the window: the frame's right half is the
        # camera's black background.
        assert not picture[:, half:, :3].any(), k
        quantities = answer.state.quantities
        (node,) = [n for n in answer.state.nodes if n.type == "VideoMobject"]
        numbers = {q.name: q.value for q in quantities[node.id]}
        assert numbers == {
            "media.frame": math.floor(k * 24 / 60),
            "media.time": float(Fraction(k, 60)),
        }


def test_a_later_frame_that_decodes_otherwise_is_refused_when_first_needed(
    tmp_path, monkeypatch, door
):
    """The frames before it draw; the first request that needs frame 20 is
    refused with the line that made the clip, and draws nothing.
    """
    monkeypatch.setattr(frames, "cache", FrameCache())
    twentieth = (FIXTURES / "numbered-clip" / "frames" / "0020.png").read_bytes()
    decoded = frames.decode

    def changed(data: bytes) -> np.ndarray:
        pixels = decoded(data)
        if data == twentieth:
            pixels[30, 50, 0] ^= 2
        return pixels

    monkeypatch.setattr(frames, "decode", changed)
    server = Dispatcher()
    refused_at = tmp_path / "refused.png"
    try:
        server.handle(
            "open",
            {
                "file": str(SCENES),
                "scene": "ClipRuns",
                "config": {**SMALL, **EVERY_TICK},
            },
        )
        for k in range(20):
            drawn = tmp_path / f"{k}.png"
            # Frame k of the clip shows from k/24 s on: the first tick on or
            # after that shows it.
            tick = Fraction(math.ceil(Fraction(k, 24) * 60), 60)
            server.handle("frame", {"t": float(tick), "path": str(drawn)})
            with Image.open(drawn) as picture:
                assert stamp(np.asarray(picture), (0, 0, *CLIP_PIXELS)) == k
        before = door.draws
        assert before == 20
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.85, "path": str(refused_at)})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    finding = refused.value.data["finding"]
    assert finding["code"] == "media.inexact"
    assert finding["detail"]["reason"] == "digest"
    assert finding["detail"]["file"] == "frames/0020.png"
    assert finding["line"] == line_of("clip-runs", SCENES).line
    assert door.draws == before
    assert not refused_at.exists()


def test_a_clip_added_back_after_its_clock_ran_out_refuses_its_frame(tmp_path, door):
    """Added back after the last play, with no instant evaluated after it:
    the scene builds and answers its states, and a frame showing the clip
    is refused with the line that made it.
    """
    server = Dispatcher()
    picture = tmp_path / "frame.png"
    try:
        server.handle(
            "open",
            {"file": str(SCENES), "scene": "RanOutAddedLast", "config": SMALL},
        )
        state = server.handle("state_at", {"t": 10})["state"]
        assert "VideoMobject" in {node["kind"] for node in state["nodes"]}
        assert server.handle("frame", {"t": 0.5, "path": str(tmp_path / "a.png")})
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 10, "path": str(picture)})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    finding = refused.value.data["finding"]
    assert finding["code"] == "media.clock_past_end"
    assert finding["line"] == line_of("added-last", SCENES).line
    assert finding["fix"] == CODES["media.clock_past_end"].fix
    assert 'run(then="hold")' in finding["fix"]
    assert finding["detail"]["media_time"] == 2.5
    assert not picture.exists()


EDITED = """\
from arachne import *


class Edited(Scene):
    def construct(self):
        clip = VideoMobject({asset!r}, resampling_algorithm=0)  # @edited
        clip.stretch_to_fit_width(config.frame_width)
        clip.stretch_to_fit_height(config.frame_height)
        self.add(clip)
        clip.hold(frame=5)
        self.wait(0.5)
"""


def _edited_scene(tmp_path) -> tuple[Path, Path]:
    """A scene showing a copy of the test clip held on frame 5, and the
    copy's directory.
    """
    asset = copied(tmp_path, "numbered-clip")
    scene = tmp_path / "edited.py"
    scene.write_text(EDITED.format(asset=str(asset)))
    return scene, asset


def _overwrite_frame_5(asset: Path) -> None:
    """Frame 5's file, overwritten with other pixels in the admitted form:
    frame 6's file.
    """
    frames_dir = asset / "frames"
    (frames_dir / "0005.png").write_bytes((frames_dir / "0006.png").read_bytes())


def test_a_frame_file_edited_on_disk_after_the_object_is_built_is_refused(
    tmp_path, monkeypatch, session_of, door
):
    """Frame 5 draws; its file is then overwritten and let go by the cache:
    the next request for it decodes the file again and is refused with the
    line that made the clip, drawing nothing.
    """
    from arachne.lint.raised import RaisedFinding

    kept = FrameCache()
    monkeypatch.setattr(frames, "cache", kept)
    scene, asset = _edited_scene(tmp_path)
    session = session_of("Edited", scene, **EVERY_TICK)
    picture = np.asarray(session.frame(Fraction(1, 4)))
    assert stamp(picture, (0, 0, *CLIP_PIXELS)) == 5
    before = door.draws
    kept.clear()
    _overwrite_frame_5(asset)
    with pytest.raises(RaisedFinding) as refused:
        session.frame(Fraction(2, 5))
    assert refused.value.code == "media.inexact"
    assert refused.value.detail["reason"] == "digest"
    assert refused.value.detail["file"] == "frames/0005.png"
    assert refused.value.line == line_of("edited", scene).line
    assert door.draws == before


def test_at_the_server_an_edited_frame_file_refuses_the_asset_on_reload(
    tmp_path, monkeypatch, door
):
    """The server runs a scene again when a file it read has changed, and
    the asset then fails its admission (``asset.unadmitted``) at the line
    that made the clip, before any frame of it is decoded or drawn.
    """
    monkeypatch.setattr(frames, "cache", FrameCache())
    scene, asset = _edited_scene(tmp_path)
    server = Dispatcher()
    drawn, refused_at = tmp_path / "drawn.png", tmp_path / "refused.png"
    try:
        server.handle(
            "open",
            {"file": str(scene), "scene": "Edited", "config": {**SMALL, **EVERY_TICK}},
        )
        server.handle("frame", {"t": 0.25, "path": str(drawn)})
        before = door.draws
        _overwrite_frame_5(asset)
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.4, "path": str(refused_at)})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    finding = refused.value.data["finding"]
    assert finding["code"] == "asset.unadmitted"
    assert finding["line"] == line_of("edited", scene).line
    assert door.draws == before
    assert not refused_at.exists()
