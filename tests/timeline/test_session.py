"""The in-process session, and frames drawn from the record."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from .conftest import render
from .scenes import Drawn, FixedOrientation


def _pixels(image) -> np.ndarray:
    return np.asarray(image)


@pytest.mark.parametrize("scene_class", [Drawn, FixedOrientation])
def test_a_render_draws_what_the_session_draws_at_each_instant(session_of, scene_class):
    _, frames = render(scene_class, 6)
    session = session_of(scene_class.__name__)
    assert len(frames) == len(session.record.samples(6))
    for k, frame in enumerate(frames):
        assert np.array_equal(frame, _pixels(session.frame(Fraction(k, 6)))), k


def test_frames_agree_at_the_instants_two_rates_share():
    _, slow = render(Drawn, 4)
    _, fast = render(Drawn, 30)
    shared = [(k, k * 30 // 4) for k in range(len(slow)) if (k * 30) % 4 == 0]
    assert len(shared) > 3
    for slow_index, fast_index in shared:
        assert np.array_equal(slow[slow_index], fast[fast_index]), slow_index


def _dot_x(image) -> float:
    """The mean column of the frame's bright pixels."""
    pixels = _pixels(image)[..., :3].max(axis=-1)
    columns = np.nonzero(pixels > 128)[1]
    return float(columns.mean())


def test_a_centre_function_is_drawn_at_each_wait_binding(session_of):
    """The script rebinds the variable the centre function reads between two
    waits; frames drawn after the script ended still show each wait at its
    own binding, from the values the forward run recorded.
    """
    session = session_of("FixedOrientation")
    first = session.frame(Fraction(1, 4))
    second = session.frame(Fraction(3, 4))
    # Drawn with the live binding, both would show the second. (That each
    # equals the frame the render drew during its wait is
    # test_a_render_draws_what_the_session_draws_at_each_instant.)
    first_box = session.state_at(Fraction(1, 4)).named("dot").box
    second_box = session.state_at(Fraction(3, 4)).named("dot").box
    first_centre = (first_box[0] + first_box[2]) / 2
    second_centre = (second_box[0] + second_box[2]) / 2
    assert abs(first_centre - second_centre) > 1
    # The frames put the dot where the boxes say.
    first_x, second_x = _dot_x(first), _dot_x(second)
    assert abs(first_x - second_x) > 10
    assert (first_x < second_x) == (first_centre < second_centre)
    # Asked again, in the other order, the answer is the same.
    assert np.array_equal(_pixels(session.frame(Fraction(1, 4))), _pixels(first))


def test_frame_draws_at_a_quality(session_of):
    session = session_of("Drawn")
    image = session.frame(Fraction(1, 2), "low_quality")
    assert image.size == (854, 480)
    assert session.frame(Fraction(1, 2)).size == (96, 54)
    with pytest.raises(ValueError, match="Unknown quality"):
        session.frame(0, "draft")


def test_frame_after_the_last_play_is_the_final_state(session_of):
    session = session_of("Drawn")
    end = session.record.duration
    final = _pixels(session.frame(end))
    assert final.shape == (54, 96, 4)
    assert session.state_at(end).play is None


def test_open_names_a_missing_scene():
    from arachne.session import Session

    from .conftest import SCENES

    with pytest.raises(LookupError, match="no scene named"):
        Session.open(SCENES, "Missing")
