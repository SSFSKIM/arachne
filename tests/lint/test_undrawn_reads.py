"""What the lint reads between and beside the frames runs none of the
scene's code: a state the record keeps for the temporal rules, and the
parts read off a forward run's live scene, take a camera callback's values
as the run recorded them. With and without them, the scene's callback runs
as often, and every frame is the same.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from arachne.lint.core import parts_reader
from arachne.timeline.fallback import visit_frames

from . import callback_scenes, run_counter

SCENES = Path(callback_scenes.__file__)
FPS = 15
#: Frames of the render's grid inside each play and at the scene's end.
FRAMES = [Fraction(k, FPS) for k in (1, 4, 8, 11, 13, 17)]


def _centres(record):
    """Every centre the record's drawn captures hold for the callback."""
    return [
        np.asarray(value)
        for play in record.plays
        for capture in play.samples
        if capture.camera is not None
        for _, value in capture.camera.centers
    ]


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_keeping_states_runs_no_camera_callback(session_of):
    run_counter.centre_calls = 0
    plain = session_of("CountedCentre", SCENES, frame_rate=FPS)
    plain_calls = run_counter.centre_calls
    run_counter.centre_calls = 0
    armed = session_of("CountedCentreArmed", SCENES, frame_rate=FPS)
    assert armed.record.states is not None
    assert len(armed.record.states.calls) == 2
    assert len(armed.record.states.removals) == 1
    assert run_counter.centre_calls == plain_calls > 0
    _same(_centres(armed.record), _centres(plain.record))
    for t in FRAMES:
        np.testing.assert_array_equal(
            np.asarray(armed.frame(t)), np.asarray(plain.frame(t))
        )
    # Drawing from the record runs it no more either.
    assert run_counter.centre_calls == plain_calls


def test_reading_a_forward_run_s_live_parts_runs_no_camera_callback(session_of):
    session = session_of("CountedCentre", SCENES, frame_rate=FPS)

    def run(visit, *, states):
        run_counter.centre_calls = 0
        forward = visit_frames(session, FRAMES, visit, to_end=True, states=states)
        return run_counter.centre_calls, _centres(forward)

    read = parts_reader(session)
    seen = []

    def parts(t, forward, play, capture):
        seen.append(read(forward, t, play, capture, True))

    calls, centres = run(lambda *_: None, states=False)
    read_calls, read_centres = run(parts, states=True)
    assert len(seen) == len(FRAMES)
    assert read_calls == calls > 0
    _same(read_centres, centres)
