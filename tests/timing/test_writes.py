"""A writer that puts back on a declared family the value on display, while
what is true has moved on, still writes: the hold keeps it as true, and
the next boundary shows it.

A scene driver writes against what is true: it runs, and reads, after the
clock has put back what is true, so a value equal to the one on display is
a write like any other (a small whole number, which Python shares, as much
as a new one). An animation writes before the clock runs, onto the drawing
on display, and may write the very object that drawing holds (a value
shared with a copy of the mobject made before the play): that is a write
too.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from arachne import Sprite, Square

from .reading import boundary
from .timing_scenes import SLOW, SWAPPED

TICK = Fraction(1, 60)
#: The test sprite: its duration, its drawings a second, its drawings.
SPRITE = (Fraction(2, 3), Fraction(12), 8)


def _of(session, kind: type):
    return next(m for m in session.record.mobjects() if type(m) is kind)


def _ticks(session):
    """Every 60 Hz tick of the scene before its end."""
    end = session.record.duration
    return [k * TICK for k in range(math.ceil(end / TICK))]


def _held(session, mobject, t: Fraction):
    """``mobject``'s recorded state at the scene instant ``t``."""
    record = session.record
    return record.node_state(mobject, record.capture_at(t)[1])


def _drawing(media: Fraction) -> int:
    """The sprite's drawing at its media time ``media``."""
    duration, rate, count = SPRITE
    return min(math.floor((media % duration) * rate), count - 1)


def test_a_driver_s_small_number_equal_to_the_one_on_display_is_written(opened):
    session = opened("BeatOnItsBoundaries", 60)
    square = _of(session, Square)
    ticks = _ticks(session)
    assert len(ticks) == 90
    for t in ticks:
        # floor(12 t) % 4 at the last boundary k / 3: always 0.
        assert math.floor(boundary(t, SLOW) * 12) % 4 == 0
        assert _held(session, square, t).get("beat") == 0, t


def test_every_boundary_shows_the_drawing_its_instant_names(opened):
    # The loop comes round to the drawing on display exactly at each
    # boundary: what the boundary installs is the drawing the program
    # names there, not the one before it.
    session = opened("SpriteLoopedOnItsBoundaries", 60)
    sprite = _of(session, Sprite)
    for t in _ticks(session):
        assert _drawing(2 * boundary(t, SLOW)) == 0
        assert _held(session, sprite, t).get("_frame") == 0, t


def test_a_program_changed_between_boundaries_is_shown_from_the_next(opened):
    # The driver reads the program that is true, not the one on display:
    # the first boundary after the change shows the new program's drawing.
    session = opened("SpriteProgramSwapped", 60)
    sprite = _of(session, Sprite)
    swapped = Fraction(SWAPPED)
    for t in _ticks(session):
        at = boundary(t, SLOW)
        media = at if at < swapped else at - swapped
        want = _drawing(media)
        assert _held(session, sprite, t).get("_frame") == want, (t, want)
    # The two programs name other drawings at the first boundary after the
    # change, 2/3.
    assert _drawing(Fraction(2, 3)) != _drawing(Fraction(2, 3) - swapped)


def test_an_animation_ending_on_the_points_on_display_is_written(opened):
    # The Create begins and ends between the boundaries 1/3 and 2/3, so the
    # square on display stays whole throughout; its last write is the
    # square's own points, shared with its starting copy, which the drawing
    # on display holds. From 2/3 on the square is whole again, not as the
    # Create stood at its last tick.
    session = opened("RedrawnWithinOneDrawing", 60)
    square = _of(session, Square)
    whole = np.asarray(Square().points)
    # Every tick, and the scene's end, which the record answers with what
    # is true.
    for t in [*_ticks(session), session.record.duration]:
        points = np.asarray(_held(session, square, t).get("_points"))
        assert points.shape == whole.shape, t
        assert np.abs(points - whole).max() < 1e-9, t
