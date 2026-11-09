"""A frame described from the camera's own captures finds a drawn face's
surface among the scene's roots: each root it walks holds its state at the
frame, however the frame is drawn. A play drawn after the scene has run (its
plays waited for the render's workers), and a session's frame drawn over a
static picture it kept from an earlier frame, see the still surface as the
play held it, not as the scene left it.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import tempconfig
from arachne.frame_ir import live as live_module
from arachne.session import Session

from ..timeline.conftest import SMALL
from . import captured_surface_scenes as scenes

pytest.importorskip("arachne_core")

FPS = 15
SCENES = Path(scenes.__file__)
CAMERA_CODE = "the scene's code draws through its camera"


def _seen(monkeypatch) -> list[tuple[object, int]]:
    """Each surface the live description walks, with the uv faces it holds
    then, in order.
    """
    seen: list[tuple[object, int]] = []
    found = live_module._uv_faces

    def counted(surface):
        faces = found(surface)
        seen.append((surface, len(faces)))
        return faces

    monkeypatch.setattr(live_module, "_uv_faces", counted)
    return seen


def _still(scene, seen) -> list[int]:
    return [count for surface, count in seen if surface is scene.still]


def _rendered(scene_class, workers, tmp_path, monkeypatch):
    """The scene rendered as an uncached video with ``workers``, and what
    the description saw of its still surface: with more than one, its plays
    wait for them until the scene has run, then are drawn here (the scene's
    code draws through its camera).
    """
    seen = _seen(monkeypatch)
    settings = {
        **SMALL,
        "frame_rate": FPS,
        "disable_caching": True,
        "format": "mp4",
        "progress_bar": "none",
        "media_dir": str(tmp_path / f"media-{workers}"),
        "draw_path": "ir_cairo",
        "render_workers": workers,
    }
    with tempconfig(settings):
        scene = scene_class()
        scene.render()
    monkeypatch.undo()
    return scene, _still(scene, seen)


@pytest.mark.parametrize(
    "scene_class",
    [
        scenes.StillSurfaceCapturedCut,
        scenes.StillSurfaceCutMidPlay,
        scenes.StillSurfaceCaptured,
    ],
    ids=["cut between plays", "cut mid play", "uncut"],
)
def test_plays_drawn_after_the_scene_see_the_still_surface_as_each_held_it(
    scene_class, tmp_path, monkeypatch
):
    here, as_run = _rendered(scene_class, 1, tmp_path, monkeypatch)
    later, after = _rendered(scene_class, 2, tmp_path, monkeypatch)
    assert here.renderer.worker_fallback is None
    assert CAMERA_CODE in later.renderer.worker_fallback
    assert as_run
    assert after == as_run


def _session_frames(name, monkeypatch):
    """Two frames of the first play asked of a session one after the other
    (the second over the static picture the first kept), and what the
    description saw of the still surface.
    """
    session = Session.open(
        SCENES, name, config={**SMALL, "frame_rate": FPS, "draw_path": "ir_cairo"}
    )
    try:
        play = session.record.plays[0]
        step = Fraction(1, FPS)
        seen = _seen(monkeypatch)
        frames = [np.asarray(session.frame(play.start + k * step)) for k in (1, 2)]
        counts = _still(session.scene, seen)
        monkeypatch.undo()
        return frames, counts
    finally:
        session.close()


def test_a_session_frame_over_a_kept_static_picture_sees_the_play_s_surface(
    monkeypatch,
):
    _, counts = _session_frames("StillSurfaceCapturedCut", monkeypatch)
    assert counts
    assert set(counts) == {64}, counts


def test_an_uncut_surface_s_session_frames_are_unchanged(monkeypatch):
    frames, counts = _session_frames("StillSurfaceCaptured", monkeypatch)
    assert counts
    assert set(counts) == {64}, counts
    cut, _ = _session_frames("StillSurfaceCapturedCut", monkeypatch)
    # The first play is the same in both scenes: so are its frames.
    for a, b in zip(frames, cut, strict=True):
        np.testing.assert_array_equal(a, b)
