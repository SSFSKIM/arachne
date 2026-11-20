"""Media objects beside stepped timing, with no GPU.

The media driver is a scene driver: a scene that declares timing runs it
from its clock, whether the clip ran before the declaration or after, and
nothing is refused for it. A media object declared with stepped timing
shows its drawing, and stands where it stands, only on its grid, while its
own time, ``media.time``, runs on: the program's time at the instant asked,
against the frame on display, ``media.frame``, held from the last boundary.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import RasterPart, Rig, Sprite, Timing, VideoMobject, timing_of
from arachne.assets import timing
from arachne.mobject.media import media_driver
from arachne.session import Session
from arachne.timeline import admission

from . import media_scenes
from .media_scenes import HALF, MOVE

SCENES = Path(media_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54}
TICK = Fraction(1, 60)

#: The test sprite: its duration, rate and drawing count; the test clip's.
SPRITE = (Fraction(2, 3), Fraction(12), 8)
CLIP = (Fraction(2), Fraction(24), 48)


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, fps: int = 60) -> Session:
        session = Session.open(SCENES, name, config={**SMALL, "frame_rate": fps})
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _of(session: Session, kind: type):
    return next(m for m in session.record.mobjects() if type(m) is kind)


def _ticks(session: Session):
    """Every 60 Hz tick of the scene before its end."""
    end = session.record.duration
    return [k * TICK for k in range(math.ceil(end / TICK))]


def _held(session: Session, mobject, t: Fraction):
    """``mobject``'s recorded state at the scene instant ``t``."""
    record = session.record
    return record.node_state(mobject, record.capture_at(t)[1])


def _frame_of(media: Fraction, of: tuple[Fraction, Fraction, int]) -> int:
    _, rate, count = of
    return min(math.floor(media * rate), count - 1)


def _boundary(t: Fraction, rate: int) -> Fraction:
    return Fraction(math.floor(t * rate), rate)


def _media(session: Session, t: Fraction) -> tuple[Fraction, dict[str, float]]:
    """The instant ``state_at(t)`` answers for, and the one media object's
    quantities then.
    """
    state = session.state_at(t)
    (node,) = [n for n in state.nodes if n.type in ("Sprite", "VideoMobject")]
    numbers = {q.name: q.value for q in state.quantities[node.id]}
    return state.as_of, numbers


def _scene_updaters(session: Session) -> list[tuple[Fraction, tuple]]:
    return [
        (capture.instant, tuple(capture.scene_updaters))
        for play in session.record.plays
        for capture in play.samples
    ]


# The driver under the clock


@pytest.mark.parametrize("scene", ["ClipRunAfterTiming", "ClipRunBeforeTiming"])
def test_a_clip_run_beside_declared_timing_builds_and_runs_from_the_clock(
    opened, scene
):
    session = opened(scene)
    clip = _of(session, VideoMobject)
    clock = timing.clock_of(session.scene)
    driver = media_driver(session.scene)
    assert clock is not None
    assert driver is not None
    assert driver.holds(clip)
    # The clock runs the driver, and is the scene's one updater once there.
    assert [d for d in clock.drivers if d is driver] == [driver]
    declared = Fraction(1, 4) if scene == "ClipRunBeforeTiming" else Fraction(0)
    for t, updaters in _scene_updaters(session):
        assert driver.updater not in updaters or t <= declared, t
        if t > declared:
            assert updaters == (clock.updater,), t
    # The clip runs from its run(), frame for frame.
    started = Fraction(1, 4) - declared
    for t in _ticks(session):
        media = max(t - started, Fraction(0))
        assert _held(session, clip, t).get("_frame") == _frame_of(media, CLIP), t


def test_a_clip_run_before_the_declaration_had_its_driver_as_the_scene_s_updater(
    opened,
):
    session = opened("ClipRunBeforeTiming")
    driver = media_driver(session.scene)
    before = [u for t, u in _scene_updaters(session) if t < Fraction(1, 4)]
    assert before
    assert all(updaters == (driver.updater,) for updaters in before)


def test_stated_limit_the_clock_does_not_run_the_driver_at_an_off_tick_end(opened):
    # The timeline evaluates no scene updater at a play's end between two
    # ticks, and the clock's end of a play (its restorer's, a mobject
    # updater handed 0) holds the declared families only: the driver is
    # not run there either. So the frame on display is the last tick's,
    # as with no clock (tests/media/test_clock.py, its stated limit).
    session = opened("ClipEndingOffTickBesideTiming")
    end = Fraction(3, 8)
    as_of, numbers = _media(session, 10)
    assert as_of == end
    last_tick = Fraction(math.floor(end * 60), 60)
    assert numbers == {
        "media.frame": _frame_of(last_tick, CLIP),
        "media.time": float(end),
    }
    assert (_frame_of(last_tick, CLIP), _frame_of(end, CLIP)) == (8, 9)


# Unequal rates: program time runs on, the drawing waits


def test_a_sprite_drawn_at_half_its_rate_reports_its_time_running_on(opened):
    session = opened("SpriteAtHalfItsRate")
    duration = SPRITE[0]
    disagreed = 0
    for t in _ticks(session):
        as_of, numbers = _media(session, t)
        assert as_of == t
        assert numbers["media.time"] == float(t % duration), t
        shown = _frame_of(_boundary(t, HALF) % duration, SPRITE)
        assert numbers["media.frame"] == shown, t
        disagreed += shown != _frame_of(t % duration, SPRITE)
    # Between two boundaries the frame the program names moves on: here at
    # every other drawing of the sprite.
    assert disagreed == len(_ticks(session)) // 2


def test_the_drawing_changes_only_at_the_declared_boundaries(opened):
    session = opened("SpriteAtHalfItsRate")
    sprite = _of(session, Sprite)
    changes = []
    last = None
    for t in _ticks(session):
        frame = _held(session, sprite, t).get("_frame")
        if frame != last:
            changes.append(t)
        last = frame
    assert changes == [Fraction(k, HALF) for k in range(len(changes))]
    assert len(changes) == 9


def test_a_sprite_drawn_at_its_own_rate_shows_what_its_time_names(opened):
    session = opened("SpriteAtItsRate")
    duration = SPRITE[0]
    for t in _ticks(session):
        _, numbers = _media(session, t)
        media = t % duration
        assert numbers["media.time"] == float(media), t
        assert numbers["media.frame"] == _frame_of(media, SPRITE), t


def test_a_moved_sprite_stands_and_draws_only_on_its_grid(opened):
    session = opened("MovedSteppedSprite")
    sprite = _of(session, Sprite)
    start = np.asarray(_held(session, sprite, Fraction(0)).get("_points"))
    moved = 0
    for t in _ticks(session):
        state = _held(session, sprite, t)
        at = _boundary(t, HALF)
        # Where the linear move put it at the last boundary, and the drawing
        # its loop named there.
        share = float(min(at / Fraction(9, 10), Fraction(1)))
        np.testing.assert_allclose(
            np.asarray(state.get("_points")),
            start + share * MOVE,
            rtol=0,
            atol=1e-9,
            err_msg=str(t),
        )
        assert state.get("_frame") == _frame_of(at % SPRITE[0], SPRITE), t
        moved += share > 0
    assert moved


# What a declared media object is


def test_a_clip_is_declared_without_a_refusal_and_held_on_its_grid(opened):
    session = opened("DeclaredClip")
    clip = _of(session, VideoMobject)
    assert timing_of(clip) == Timing(HALF)
    for t in _ticks(session):
        assert _held(session, clip, t).get("_frame") == _frame_of(
            _boundary(t, HALF), CLIP
        ), t


def test_timing_of_reads_the_mark_of_an_installed_capture(opened):
    session = opened("DeclaredClip")
    clip = _of(session, VideoMobject)
    state = _held(session, clip, Fraction(1, 2))
    clip.__dict__.pop(timing.MARK)
    assert timing_of(clip) is None
    state.install(clip)
    assert timing_of(clip) == Timing(HALF)


def test_a_rig_of_stills_declared_with_timing_is_posed_only_on_its_grid(opened):
    session = opened("PosedSteppedFlap")
    rig = _of(session, Rig)
    quads = [m for m in rig.get_family() if isinstance(m, RasterPart)]
    assert quads
    index = rig.model.index["lift"]
    lifted = 0
    for t in _ticks(session):
        at = _boundary(t, HALF)
        values = _held(session, rig, t).get("_values")
        expected = 60 * float(min(at, Fraction(1)))
        assert float(values[index]) == pytest.approx(expected, abs=1e-9), t
        for quad in quads:
            np.testing.assert_array_equal(
                np.asarray(_held(session, quad, t).get("_points")),
                np.asarray(_held(session, quad, at).get("_points")),
                err_msg=str(t),
            )
        lifted += expected > 0
    assert lifted
