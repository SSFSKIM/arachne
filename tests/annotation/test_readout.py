"""A readout's number, counted by :class:`CountUp`, held to three statements
about time, kept apart, at 15, 30 and 60 frames per second:

1. ``state_at(t)`` answers the last frame instant of the record's grid at
   or before ``t``, reports it as ``as_of``, and its text is the format of
   the declared value there;
2. at every delivered frame instant, the frame draws the format of the
   declared value at that instant, the same at every rate that delivers it;
3. a render's last frame is its last frame instant before the scene's end:
   the exact target is in a delivered frame only when a frame instant at or
   after the count's end exists.

The declared value at ``t`` is computed here, from the count's own terms:
``start + (to - start) * rate_func((t - t0) / run_time)``.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Readout, Text
from arachne.lint.frames import read_frames
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.clock import frames_in
from arachne.timeline.draw import Drawing

from . import annotation_scenes
from .annotation_scenes import COUNTS

SCENES = Path(annotation_scenes.__file__)
RATES = (15, 30, 60)


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, fps: int) -> Session:
        session = Session.open(
            SCENES,
            name,
            config={"pixel_width": 96, "pixel_height": 54, "frame_rate": fps},
        )
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _declared(key: str, t: Fraction) -> float:
    start, to, _, rate = COUNTS[key]
    alpha = min(max(float(t), 0.0), 1.0)
    return start + (to - start) * rate(alpha)


def _shown(key: str, t: Fraction) -> str:
    return COUNTS[key][2].format(_declared(key, t))


def _readout(session: Session) -> Readout:
    return next(m for m in session.record.mobjects() if isinstance(m, Readout))


def _frame_read(session: Session):
    """What each frame draws of the scene's readout: its declared value, the
    string its label says, and its glyphs' points.
    """
    readout = _readout(session)

    def read(record, t, play, capture, live):
        with Drawing(record) as drawing:
            if not live:
                drawing.show(capture, capture.draw_list())
            label = readout.label
            glyphs = [np.array(m.points) for m in label.family_members_with_points()]
            return (
                readout.value,
                label.original_text,
                glyphs,
                label.get_center(),
                label.font_size,
            )

    return read


def _drawn_as(said: str, center: np.ndarray, font_size: float) -> list[np.ndarray]:
    """The glyphs a text saying ``said`` draws, centred at ``center``."""
    text = Text(said, font_size=font_size).move_to(center)
    return [np.array(m.points) for m in text.family_members_with_points()]


@pytest.mark.parametrize("key", list(COUNTS))
def test_as_of_answers_the_last_frame_of_the_record_s_grid(opened, key):
    asked = Fraction(1, 12)
    answers = {}
    for fps in (15, 60):
        session = opened(f"Count{key}", fps)
        state = session.state_at(asked)
        label = state.node(state.node(state.roots[0]).children[0])
        answers[fps] = (state.as_of, label.text)
    # A 15 fps record answers from 1/15, a 60 from 1/12 itself: both right.
    assert answers[15] == (Fraction(1, 15), _shown(key, Fraction(1, 15)))
    assert answers[60] == (Fraction(1, 12), _shown(key, Fraction(1, 12)))


@pytest.mark.parametrize("key", list(COUNTS))
@pytest.mark.parametrize("waits", [False, True])
def test_every_delivered_frame_draws_the_declared_value_at_its_instant(
    opened, key, waits
):
    name = f"Count{key}{'ThenWait' if waits else ''}"
    drawn: dict[int, dict[Fraction, str]] = {}
    for fps in RATES:
        session = opened(name, fps)
        times = frames_in(Fraction(0), session.record.duration, Fraction(fps))
        for t, (value, said, glyphs, center, size) in zip(
            times, read_frames(session, times, _frame_read(session)), strict=True
        ):
            assert value == pytest.approx(_declared(key, t), abs=1e-9), (fps, t)
            assert said == _shown(key, t), (fps, t)
            expected = _drawn_as(said, center, size)
            assert len(glyphs) == len(expected), (fps, t)
            for points, reference in zip(glyphs, expected, strict=True):
                np.testing.assert_allclose(points, reference, atol=1e-9)
            drawn.setdefault(fps, {})[t] = said
    # The same drawing wherever two rates deliver the same instant.
    for t, said in drawn[15].items():
        assert drawn[30][t] == drawn[60][t] == said


@pytest.mark.parametrize("key", list(COUNTS))
@pytest.mark.parametrize("fps", RATES)
def test_the_target_is_seen_only_when_a_frame_follows_the_count(opened, key, fps):
    start, to, fmt, _ = COUNTS[key]
    end = Fraction(1)
    rate = Fraction(fps)

    ends = opened(f"Count{key}", fps)
    times = frames_in(Fraction(0), ends.record.duration, rate)
    # The last frame is the last frame instant before the scene's end.
    assert times[-1] == end - 1 / rate
    *_, (value, said, *_) = read_frames(ends, times[-1:], _frame_read(ends))
    assert said == _shown(key, times[-1])
    if key == "Rounded":
        # Counted on a straight line, the last frame shows a number short of
        # the target.
        assert said != fmt.format(to)
    # The count's value is exact once it ends, whether or not it is seen.
    assert _readout(ends).value == to
    assert ends.state_at(end).as_of == end

    waits = opened(f"Count{key}ThenWait", fps)
    times = frames_in(Fraction(0), waits.record.duration, rate)
    after = [t for t in times if t >= end]
    assert after
    assert after[0] == end
    for value, said, *_ in read_frames(waits, after, _frame_read(waits)):
        assert value == to
        assert said == fmt.format(to)


def test_a_readout_s_format_and_value_are_its_own():
    reading = Readout(2.345, format="{:.2f} s")
    assert reading.value == 2.345
    assert reading.label.original_text == "2.35 s"
    reading.set_value(10)
    assert reading.value == 10
    assert reading.label.original_text == "10.00 s"
    # A colour given since is kept by the next value.
    reading.set_color("#FF0000")
    reading.set_value(11)
    assert reading.label.submobjects[0].get_color().to_hex() == "#FF0000"
    with pytest.raises(ValueError, match="finite"):
        reading.set_value(float("nan"))
    with pytest.raises(ValueError, match="format"):
        Readout(1, format="{:.1f} and {:.1f}")
