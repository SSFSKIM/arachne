"""Key frames and storyboard sheets: one picture per play or wait, the last
frame of the render's grid it holds, drawn as the session draws a frame.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arachne.lint.keyframes import COLUMNS, LABEL_HEIGHT, ROWS, TILE

from ..timeline import scenes as timeline_scenes
from . import query_scenes
from .conftest import line_of

QUERIES = Path(query_scenes.__file__)
TIMELINE_SCENES = Path(timeline_scenes.__file__)


def _pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def test_key_frame_count_and_times(session_of, tmp_path):
    """At 15 fps: a 0.5 s play, a frozen 0.4 s wait, a 0.01 s play and a
    0.005 s wait that hold no frame, a ticked 0.6 s play and a 0.3 s wait.
    """
    session = session_of("Instants", frame_rate=15)
    record = session.record
    assert [p.held for p in record.plays] == [False, True, False, True, False, False]
    result = session.key_frames(tmp_path)
    assert [(k.play, k.kind, k.t) for k in result.frames] == [
        (0, "play", Fraction(7, 15)),
        (1, "wait", Fraction(13, 15)),
        (4, "play", Fraction(22, 15)),
        (5, "wait", Fraction(27, 15)),
    ]
    for key in result.frames:
        play = record.plays[key.play]
        assert play.start <= key.t < play.end
        assert key.line == play.source_line
        assert key.answer == "recorded"
    assert [(u.play, u.kind) for u in result.unshown] == [(2, "play"), (3, "wait")]
    for unshown in result.unshown:
        assert "no frame" in unshown.reason
    assert result.findings == ()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "play-000.png",
        "play-001.png",
        "play-004.png",
        "play-005.png",
    ]


def test_a_key_frame_is_the_session_s_frame(session_of, tmp_path):
    session = session_of("Instants", frame_rate=15)
    result = session.key_frames(tmp_path, quality="low_quality")
    for key in result.frames:
        with Image.open(key.path) as image:
            assert image.size == (854, 480)
            drawn = _pixels(image)
        assert np.array_equal(drawn, _pixels(session.frame(key.t, "low_quality")))


def test_a_draw_the_painter_refuses_is_a_render_unsupported_finding(
    session_of, tmp_path, monkeypatch
):
    session = session_of("Instants", frame_rate=15)

    class Refused(Exception):
        """What a painter raises for a frame it does not draw."""

        code = "render.unsupported"
        kind = "mesh"

    drawn = session.frame

    def frame(t, quality=None):
        if t == Fraction(13, 15):
            raise Refused("render.unsupported: slot 3 is a mesh record")
        return drawn(t, quality)

    monkeypatch.setattr(session, "frame", frame)
    result = session.key_frames(tmp_path)
    assert [k.play for k in result.frames] == [0, 4, 5]
    (finding,) = result.findings
    assert finding.code == "render.unsupported"
    assert finding.severity == "error"
    assert finding.t_start == finding.t_end == Fraction(13, 15)
    assert finding.line == session.record.plays[1].source_line
    assert finding.detail == {"subject": "mesh"}
    assert "slot 3" in finding.message
    refused = [u for u in result.unshown if u.play == 1]
    assert refused
    assert "render.unsupported" in refused[0].reason
    assert not (tmp_path / "play-001.png").exists()


def test_another_failure_of_a_draw_is_not_a_finding(session_of, tmp_path, monkeypatch):
    session = session_of("Instants", frame_rate=15)

    def frame(t, quality=None):
        raise ZeroDivisionError("a bug")

    monkeypatch.setattr(session, "frame", frame)
    with pytest.raises(ZeroDivisionError):
        session.key_frames(tmp_path)


def test_a_refused_record_s_key_frames_come_from_one_forward_run(session_of, tmp_path):
    session = session_of("RefusedInstants", frame_rate=15)
    assert session.record.refusal is not None
    result = session.key_frames(tmp_path)
    assert [k.play for k in result.frames] == [0, 1, 4, 5]
    assert {k.answer for k in result.frames} == {"forward"}
    assert len(session.forward_runs) == 1
    for key in result.frames:
        with Image.open(key.path) as image:
            assert image.size == (96, 54)


def test_a_play_the_render_skips_has_no_key_frame(session_of, tmp_path):
    session = session_of("Sections", TIMELINE_SCENES, frame_rate=15)
    assert [p.skipped for p in session.record.plays] == [False, True, False]
    result = session.key_frames(tmp_path)
    assert [k.play for k in result.frames] == [0, 2]
    (unshown,) = result.unshown
    assert unshown.play == 1
    assert "skipped" in unshown.reason


# Storyboard sheets ---------------------------------------------------------------


def test_sheet_geometry_is_six_tiles_of_960_by_540(session_of, tmp_path):
    session = session_of("ManyPlays", QUERIES, frame_rate=15)
    assert len(session.record.plays) == 7
    board = session.storyboard(tmp_path)
    assert TILE == (960, 540)
    assert COLUMNS * ROWS == 6
    assert len(board.sheets) == 2
    assert [len(sheet.tiles) for sheet in board.sheets] == [6, 1]
    width, height = TILE
    for sheet in board.sheets:
        with Image.open(sheet.path) as image:
            assert image.size == (COLUMNS * width, ROWS * (height + LABEL_HEIGHT))
            pixels = _pixels(image)
        for tile in sheet.tiles:
            left, top, right, bottom = tile.box
            assert (right - left, bottom - top) == TILE
            expected = _pixels(
                session.frame(tile.key.t)
                .convert("RGB")
                .resize(TILE, Image.Resampling.LANCZOS)
            )
            assert np.array_equal(pixels[top:bottom, left:right], expected)
            # Its label is written in the band under it.
            band = pixels[bottom : bottom + LABEL_HEIGHT, left:right]
            assert len(np.unique(band.reshape(-1, 3), axis=0)) > 1


def test_each_tile_is_labelled_with_its_time_play_and_line(session_of, tmp_path):
    session = session_of("ManyPlays", QUERIES, frame_rate=15)
    board = session.storyboard(tmp_path)
    tiles = [tile for sheet in board.sheets for tile in sheet.tiles]
    assert [tile.key.play for tile in tiles] == list(range(7))
    step, last = line_of("step", QUERIES), line_of("last", QUERIES)
    for tile in tiles:
        key = tile.key
        assert f"t={float(key.t):.3f}s" in tile.label
        assert f"play {key.play}" in tile.label
        line = step if key.play < 6 else last
        assert f"query_scenes.py:{line}" in tile.label


def test_a_tile_keeps_the_frame_s_shape(session_of, tmp_path):
    """A picture of another shape than 16:9 is fitted into its tile whole,
    centred, the rest of the tile left dark.
    """
    session = session_of(
        "ManyPlays", QUERIES, frame_rate=15, pixel_width=54, pixel_height=54
    )
    board = session.storyboard(tmp_path)
    tile = board.sheets[0].tiles[0]
    with Image.open(board.sheets[0].path) as image:
        pixels = _pixels(image)
    left, top, right, bottom = tile.box
    cell = pixels[top:bottom, left:right]
    width, height = TILE
    side = height
    margin = (width - side) // 2
    assert not cell[:, : margin - 1].any()
    assert not cell[:, margin + side + 1 :].any()
    expected = _pixels(
        session.frame(tile.key.t)
        .convert("RGB")
        .resize((side, side), Image.Resampling.LANCZOS)
    )
    assert np.array_equal(cell[:, margin : margin + side], expected)
