"""What renders deliver of a declared rig, read off their frames (a PNG
sequence) at 15, 30 and 60 frames per second, each frame matched against
drawings made here of the rig at every pose it could show:

2. every delivered frame draws the pose true at the last boundary ``k / 12``
   at or before its instant, the same at every rate that delivers that
   instant, never the pose true at the instant itself;
3. a render's last frame is its last frame instant before the scene's end:
   a play's final state, first shown at the boundary after the play ends,
   is in a delivered frame only when a frame instant at or after that
   boundary exists.

The pose true at ``t`` is computed here from the scene's own terms (a
linear pose), and the instants a render delivers from its frame rate
(``k / fps`` before the scene's end); every drawing is made by a camera
here, as a render draws a frame.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arachne import Rig, tempconfig
from arachne.camera.camera import Camera
from arachne.session import Session

from . import timing_scenes
from .reading import RATES, SCENES, boundary, scene_box_of
from .timing_scenes import OPEN

#: The rendered picture: wide enough that one pose's frame is told from the
#: next's.
PICTURE = {"pixel_width": 192, "pixel_height": 108}
#: How far, in mean levels per channel, a frame may lie from the drawing it
#: shows, made here: rounding, never another pose.
SAME = 1.0


def _delivered(end: Fraction, fps: int) -> list[Fraction]:
    """The instants a render at ``fps`` of a scene ending at ``end``
    delivers frames at: each ``k / fps`` before the end.
    """
    return [Fraction(k, fps) for k in range(math.ceil(end * fps))]


def _rendered(name: str, fps: int, media: Path) -> list[np.ndarray]:
    """The frames a render of the scene ``name`` at ``fps`` delivers, in
    order.
    """
    scene = getattr(timing_scenes, name)
    with tempconfig(
        {
            **PICTURE,
            "frame_rate": fps,
            "media_dir": str(media),
            "format": "png-sequence",
            "disable_caching": True,
        }
    ):
        scene().render()
    files = sorted(media.rglob("*.png"))
    return [np.asarray(Image.open(f).convert("RGB"), dtype=float) for f in files]


class _Poses:
    """The box drawn here at each opening, as a render draws a frame."""

    def __init__(self) -> None:
        self._drawn: dict[float, np.ndarray] = {}

    def __call__(self, value: float) -> np.ndarray:
        key = round(value, 9)
        found = self._drawn.get(key)
        if found is None:
            with tempconfig(PICTURE):
                camera = Camera()
                camera.capture_mobjects(
                    [Rig("fixture:lidded-box").pose(lid_open=value)]
                )
                found = np.asarray(camera.pixel_array[..., :3], dtype=float)
            self._drawn[key] = found
        return found


def _shows(frame: np.ndarray, candidates: set[float], poses: _Poses) -> float:
    """The opening among ``candidates`` that ``frame`` draws: the nearest,
    which must lie within :data:`SAME` of it.
    """
    distance = {v: float(np.abs(frame - poses(v)).mean()) for v in candidates}
    nearest = sorted(distance, key=lambda v: distance[v])
    found = nearest[0]
    assert distance[found] <= SAME, [(v, distance[v]) for v in nearest[:2]]
    return found


def _posed_box(t: Fraction) -> float:
    """``PosedBox``'s opening true at ``t``: a linear pose over a second."""
    return float(OPEN * min(t, Fraction(1)))


def _off_grid(t: Fraction) -> float:
    """The off-grid scenes' opening true at ``t``: from 0.37 s to 0.87 s."""
    start, end = Fraction(37, 100), Fraction(87, 100)
    return float(OPEN * min(max((t - start) / (end - start), Fraction(0)), Fraction(1)))


def _candidates(true) -> set[float]:
    """Every opening a frame could show: the true one at each tick, of which
    the drawings at the boundaries are some.
    """
    return {true(Fraction(k, 60)) for k in range(0, 121)}


def test_every_delivered_frame_draws_its_last_boundary_s_pose(tmp_path):
    poses = _Poses()
    candidates = _candidates(_posed_box)
    drawn: dict[int, dict[Fraction, np.ndarray]] = {}
    for fps in RATES:
        frames = _rendered("PosedBox", fps, tmp_path / f"{fps}")
        times = _delivered(Fraction(5, 4), fps)
        assert len(frames) == len(times), fps
        for t, frame in zip(times, frames, strict=True):
            shown = _posed_box(boundary(t))
            assert _shows(frame, candidates, poses) == shown, (fps, t)
        drawn[fps] = dict(zip(times, frames, strict=True))
    # The same drawing wherever two rates deliver the same instant.
    for t, frame in drawn[15].items():
        assert np.abs(drawn[30][t] - frame).max() <= 1, t
        assert np.abs(drawn[60][t] - frame).max() <= 1, t
    # And between two boundaries, nothing moves.
    for t, frame in drawn[60].items():
        assert np.abs(drawn[60][boundary(t)] - frame).max() <= 1, t


@pytest.mark.parametrize("fps", RATES)
def test_an_off_grid_end_is_delivered_only_when_a_frame_follows_its_boundary(
    tmp_path, fps
):
    poses = _Poses()
    candidates = _candidates(_off_grid)
    end = Fraction(87, 100)
    first_shown = Fraction(11, 12)

    ends = _rendered("OffGridEnding", fps, tmp_path / "ends")
    times = _delivered(end, fps)
    assert len(ends) == len(times)
    for t, frame in zip(times, ends, strict=True):
        assert _shows(frame, candidates, poses) == _off_grid(boundary(t)), t
    # The last frame is the last frame instant before the scene's end, and
    # shows the drawing on display there: short of the final pose, which no
    # delivered frame shows.
    assert _shows(ends[-1], candidates, poses) == _off_grid(Fraction(10, 12))
    assert _off_grid(Fraction(10, 12)) != OPEN

    waits = _rendered("OffGridThenWait", fps, tmp_path / "waits")
    times = _delivered(end + Fraction(1, 2), fps)
    assert len(waits) == len(times)
    for t, frame in zip(times, waits, strict=True):
        shown = _shows(frame, candidates, poses)
        assert shown == _off_grid(boundary(t)), t
        # The final pose is first delivered at the first frame at or after
        # the boundary that follows the play's end.
        assert (shown == OPEN) == (t >= first_shown), t


@pytest.mark.parametrize("fps", RATES)
def test_an_off_grid_ending_s_last_frame_is_held_and_its_end_answers_true(
    tmp_path, fps
):
    """The scene's last play ends at 0.87 s, between the boundaries 10/12
    and 11/12. The movie's last frame shows the drawing held from 10/12;
    the record asked at the scene's end, ``state_at(duration)``, answers
    the pose true there, the final one. The difference is intended: at the
    scene's end the record answers the script's final state, as it does for
    every scene, and no movie frame is drawn at an instant equal to the
    duration.
    """
    end = Fraction(87, 100)
    poses = _Poses()
    frames = _rendered("OffGridEnding", fps, tmp_path / "media")
    assert len(frames) == len(_delivered(end, fps))
    held = _off_grid(Fraction(10, 12))
    assert _shows(frames[-1], _candidates(_off_grid), poses) == held
    assert held != OPEN

    session = Session.open(
        SCENES, "OffGridEnding", config={**PICTURE, "frame_rate": fps}
    )
    try:
        assert session.record.duration == end
        state = session.state_at(session.record.duration)
        final = Rig("fixture:lidded-box").pose(lid_open=OPEN)
        assert state.named("box").scene_box == pytest.approx(
            scene_box_of(final), abs=1e-9
        )
    finally:
        session.close()
