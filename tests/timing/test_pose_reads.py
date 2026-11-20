"""A pose of a declared rig moves its own parameters from what is true and
leaves the others as they are true, not as the drawing on display holds
them: a parameter the scene's code set between two boundaries, or another
pose ended earlier in the play, is shown from the next boundary.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from arachne import Rig

from .reading import boundary, frame_captures
from .timing_scenes import (
    FIGURE_RATE,
    FIGURE_RUN,
    FIGURE_START,
    HEAD,
    REACH,
    REACHED,
)

START = Fraction(FIGURE_START).limit_denominator(100)
RUN = Fraction(FIGURE_RUN).limit_denominator(100)


def _figure(session) -> Rig:
    return next(m for m in session.record.mobjects() if isinstance(m, Rig))


def _shown(session, fps: int):
    """Each delivered frame's instant, with the figure's reach and head
    turn then, and the last boundary at or before it.
    """
    figure = _figure(session)
    index = figure.model.index
    for t, capture in frame_captures(session, fps):
        values = session.record.node_state(figure, capture).get("_values")
        yield (
            t,
            boundary(t, FIGURE_RATE),
            float(values[index["reach"]]),
            float(values[index["head_turn"]]),
        )


def _along(t: Fraction, start: Fraction, run: Fraction) -> float:
    """How far a linear pose over ``[start, start + run]`` is at ``t``."""
    return float(min(max((t - start) / run, 0), 1))


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_parameter_set_between_drawings_stays_through_a_pose_of_another(opened, fps):
    session = opened("PosedBetweenDrawings", fps)
    for t, at, reach, head in _shown(session, fps):
        assert reach == pytest.approx(REACH * _along(at, START, RUN)), t
        # Set at START, between the boundaries 1/3 and 2/3.
        assert head == pytest.approx(HEAD if at > START else 0), t


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_pose_ended_earlier_in_the_play_stays_through_the_next(opened, fps):
    session = opened("PosesInTurn", fps)
    first = Fraction(REACHED).limit_denominator(100)
    for t, at, reach, head in _shown(session, fps):
        assert reach == pytest.approx(_along(at, START, first)), t
        turned = _along(at, START + first, RUN - first)
        assert head == pytest.approx(2 * HEAD * turned), t


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_two_poses_at_once_each_move_their_own(opened, fps):
    session = opened("PosesTogether", fps)
    for t, at, reach, head in _shown(session, fps):
        assert reach == pytest.approx(_along(at, START, RUN)), t
        assert head == pytest.approx(2 * HEAD * _along(at, START, RUN)), t
