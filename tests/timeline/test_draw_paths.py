"""The frames a render writes are the frames ``routes.draw`` gives from the
record, at every instant of the render's frame grid.

The renderer draws a play from its record through its own loop
(``CairoRenderer._draw_play``); whatever reads frames from a record later
(a session, a capture of the render's record) goes through
:func:`arachne.timeline.routes.draw`. These tests hold the two to one
picture: the render's decoded MP4 against ``routes.draw`` over the same
run's record, and against a session's record, which keeps only the
instants its routes evaluate and so draws trusted plays between captures
analytically.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.session import Session
from arachne.timeline.clock import frames_in
from arachne.timeline.draw import Drawing
from arachne.timeline.routes import _resolve, draw

from . import draw_path_scenes
from .conftest import SMALL
from .draw_path_scenes import (
    EndWindow,
    Held,
    StateReadout,
    Succeeding,
    Ticked,
    TrustedOffGrid,
)
from .test_video_output import FPS, LOSSLESS, decoded, render_to_video

SCENES_FILE = Path(draw_path_scenes.__file__)

#: Each scene, its frame count at 15 fps, and the routes of its plays.
CASES = [
    pytest.param(TrustedOffGrid, 32, ["trusted"] * 3, id="trusted_off_grid"),
    pytest.param(StateReadout, 26, ["state"] * 2, id="state"),
    pytest.param(Ticked, 19, ["ticked"] * 2, id="ticked"),
    pytest.param(Held, 26, ["trusted"] * 4, id="held_and_frozen"),
    pytest.param(Succeeding, 30, ["trusted"] * 2, id="succession"),
    pytest.param(EndWindow, 43, ["trusted"] * 5, id="end_window"),
]


def grid_instants(record, fps: Fraction) -> list[Fraction]:
    """The instants of the frame grid the render wrote, play by play."""
    return [
        t
        for play in record.plays
        if not play.skipped
        for t in frames_in(play.start, play.end, fps)
    ]


def drawn_from(record, instants: list[Fraction]) -> tuple[list[np.ndarray], set[str]]:
    """``routes.draw`` at each instant, and the kinds of answer it gave."""
    kinds = {_resolve(record, t)[3] for t in instants}
    with Drawing(record) as drawing:
        frames = [
            draw(drawing, record, t, lambda play: drawing.static(play))
            for t in instants
        ]
    return frames, kinds


def assert_frames_equal(shown: list[np.ndarray], drawn: list[np.ndarray]) -> None:
    assert len(shown) == len(drawn)
    for index, (frame, pixels) in enumerate(zip(shown, drawn, strict=True)):
        np.testing.assert_array_equal(frame, pixels[..., :3], err_msg=f"frame {index}")


@pytest.mark.parametrize(("scene_class", "frames", "routes"), CASES)
def test_the_render_writes_what_routes_draw_gives_from_its_record(
    tmp_path, scene_class, frames, routes
):
    scene, written, video = render_to_video(scene_class, tmp_path, **LOSSLESS)
    record = scene.timeline.ensure_record()
    assert [play.route for play in record.plays] == routes
    instants = grid_instants(record, Fraction(FPS))
    assert len(instants) == frames
    shown = decoded(video)
    assert len(shown) == frames
    assert len(written) == frames
    drawn, _ = drawn_from(record, instants)
    assert_frames_equal(shown, drawn)


@pytest.mark.parametrize(("scene_class", "frames", "routes"), CASES)
def test_the_render_writes_what_a_session_draws(tmp_path, scene_class, frames, routes):
    scene, _, video = render_to_video(scene_class, tmp_path, **LOSSLESS)
    instants = grid_instants(scene.timeline.ensure_record(), Fraction(FPS))
    session = Session.open(
        SCENES_FILE, scene_class.__name__, config={**SMALL, "frame_rate": FPS}
    )
    try:
        record = session.record
        assert [play.route for play in record.plays] == routes
        drawn, kinds = drawn_from(record, instants)
    finally:
        session.close()
    if "trusted" in routes and not all(
        play.held for play in record.plays if play.route == "trusted"
    ):
        # A session keeps a trusted play's start and event instants only, so
        # the grid's frames between them are evaluated.
        assert "analytic" in kinds
    assert_frames_equal(decoded(video), drawn)
