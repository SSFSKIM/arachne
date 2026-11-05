"""The in-process session, and frames drawn from the record."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne import BLUE, RED, Camera

from .conftest import render
from .scenes import (
    Background,
    CentreMovedFlat,
    CentreMovedThreeD,
    Drawn,
    FixedOrientation,
    Panned,
    SuppliedBackground,
    WrittenBackground,
)


def _pixels(image) -> np.ndarray:
    return np.asarray(image)


@pytest.mark.parametrize(
    "scene_class", [Drawn, FixedOrientation, Background, SuppliedBackground]
)
def test_a_render_draws_what_the_session_draws_at_each_instant(session_of, scene_class):
    _, frames = render(scene_class, 6)
    session = session_of(scene_class.__name__)
    assert len(frames) == len(session.record.samples(6))
    for k, frame in enumerate(frames):
        assert np.array_equal(frame, _pixels(session.frame(Fraction(k, 6)))), k


#: Scenes whose camera's frame centre moves after the first frame. A Camera
#: or ThreeDCamera maps paths around the centre its first path draw froze; a
#: MovingCamera around its frame as it is.
CENTRE_MOVES = [CentreMovedThreeD, CentreMovedFlat, Panned]


@pytest.mark.parametrize("scene_class", CENTRE_MOVES)
def test_a_late_frame_asked_first_is_the_render_s(session_of, scene_class):
    """The session's first frame is drawn with the centre the render froze
    at its first frame, not the centre at the instant asked.
    """
    _, frames = render(scene_class, 6)
    session = session_of(scene_class.__name__)
    last = len(frames) - 1
    assert np.array_equal(frames[last], _pixels(session.frame(Fraction(last, 6))))


@pytest.mark.parametrize("scene_class", CENTRE_MOVES)
def test_frames_asked_in_reverse_are_the_render_s(session_of, scene_class):
    _, frames = render(scene_class, 6)
    session = session_of(scene_class.__name__)
    # A frame at another quality first: the pixel array is replaced, and the
    # frames after it are drawn at the executed size again.
    session.frame(Fraction(len(frames) - 1, 6), "low_quality")
    for k in reversed(range(len(frames))):
        assert np.array_equal(frames[k], _pixels(session.frame(Fraction(k, 6)))), k


def test_answers_draw_the_render_s_frames(session_of):
    """``answers`` draws through the same seam, from a late instant first."""
    _, frames = render(CentreMovedThreeD, 6)
    session = session_of("CentreMovedThreeD")
    instants = [Fraction(k, 6) for k in reversed(range(len(frames)))]
    for answer in session.answers(instants):
        k = int(answer.t * 6)
        assert np.array_equal(frames[k], _pixels(answer.frame)), k


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


def test_a_camera_value_is_drawn_as_it_was_at_each_instant(session_of):
    """The script turns the background from red to blue between two waits;
    frames drawn after it ended show red in the first and blue in the second,
    from the camera values the forward run recorded, and the camera keeps
    its live value afterwards.
    """
    session = session_of("Background")
    first = _pixels(session.frame(Fraction(1, 4)))[0, 0, :3]
    second = _pixels(session.frame(Fraction(3, 4)))[0, 0, :3]
    assert first.argmax() == 0  # red
    assert first[0] > 200
    assert second.argmax() == 2  # blue
    assert second[2] > 200
    again = _pixels(session.frame(Fraction(1, 4)))[0, 0, :3]
    assert np.array_equal(again, first)
    assert session.scene.camera.background_color.to_hex() == BLUE.to_hex()


def _corner(image) -> list[int]:
    return _pixels(image)[0, 0, :3].tolist()


def test_a_supplied_background_is_drawn_as_it_was_at_each_instant(session_of):
    """The script sets a red, then a blue background array between two
    waits; frames drawn after it ended show red, then blue, also at another
    quality, and the camera keeps its live background afterwards.
    """
    session = session_of("SuppliedBackground")
    live = session.scene.camera.background
    assert _corner(session.frame(Fraction(1, 4))) == [255, 0, 0]
    assert _corner(session.frame(Fraction(3, 4))) == [0, 0, 255]
    assert _corner(session.frame(Fraction(1, 4))) == [255, 0, 0]
    assert session.scene.camera.background is live
    low = session.frame(Fraction(1, 4), "low_quality")
    assert low.size == (854, 480)
    assert _corner(low) == [255, 0, 0]
    assert _corner(session.frame(Fraction(3, 4), "low_quality")) == [0, 0, 255]


def test_a_background_is_replaced_never_written_in_place():
    """A camera's background is read-only, the one it builds and one supplied
    alike: a write into it would change frames the record already holds, so
    it raises naming ``set_background``, and the caller's own array stays
    theirs. (That a supplied background is drawn from the record as the
    render drew it is test_a_render_draws_what_the_session_draws_at_each_instant.)
    """
    with pytest.raises(ValueError, match=r"set_background"):
        render(WrittenBackground, 6)

    camera = Camera()
    with pytest.raises(ValueError, match=r"set_background"):
        camera.background[:] = (255, 0, 0, 255)
    built = camera.background
    camera.background_color = RED
    assert camera.background is not built
    with pytest.raises(ValueError, match=r"set_background"):
        camera.background[:] = (0, 0, 255, 255)

    pixels = np.full(camera.background.shape, (255, 0, 0, 255), dtype=np.uint8)
    camera.set_background(pixels)
    pixels[:] = 0
    assert camera.background[0, 0].tolist() == [255, 0, 0, 255]
    with pytest.raises(ValueError, match=r"set_background"):
        camera.background[0, 0] = (0, 0, 255, 255)


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


def test_a_frozen_frame_no_frame_reads_is_answered_through_its_span(session_of):
    """A frozen frame between two instants of the grid is still the scene's
    state over its span: its start is kept whether a frame reads it or not.
    """
    session = session_of("HeldBetweenFrames", grid=15)
    held = session.record.plays[1]
    assert held.held and held.start == Fraction(23, 10)
    assert Fraction(23, 10) < held.end < Fraction(35, 15)  # before the next frame
    for t in (held.start, (held.start + held.end) / 2):
        assert session.state_at(t).answer == "recorded", t


def test_a_scene_whose_code_names_a_method_record_opens(session_of):
    """The session reads its record from the engine, never from a name the
    scene's own code can take.
    """
    session = session_of("DefinesRecord", grid=15)
    assert len(session.record.plays) == 1
    assert session.state_at(Fraction(1, 5)).answer in ("analytic", "recorded")
