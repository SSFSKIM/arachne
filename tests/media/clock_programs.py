"""What each clock program of ``clock_scenes.py`` gives, written out here
from its declaration and nothing of the engine's: the clip's media time at
a scene instant, and the frame a media time shows.
"""

from __future__ import annotations

import math
from fractions import Fraction

#: The instant every program starts at (``START`` of the scenes, 0.37 s).
START = Fraction(37, 100)
#: The test clip: its duration, rate and frame count; the sprite's.
CLIP = (Fraction(2), Fraction(24), 48)
SPRITE = (Fraction(2, 3), Fraction(12), 8)


def _play(s: Fraction) -> Fraction:
    return s


def _loop(s: Fraction) -> Fraction:
    return (1 + s) % 2


def _hold(s: Fraction) -> Fraction:
    return min(s, Fraction(1, 2))


def _scrub(s: Fraction) -> Fraction:
    return Fraction(7, 4) - min(s, Fraction(3, 2))


def _half(s: Fraction) -> Fraction:
    return s / 2


#: Each program's clip media time, ``s`` scene seconds after it starts.
PROGRAMS = {
    "ClockPlay": _play,
    "ClockLoop": _loop,
    "ClockHold": _hold,
    "ClockScrub": _scrub,
    "ClockHalf": _half,
}


def clip_time(program: str, t: Fraction) -> Fraction:
    """The clip's media time at the scene instant ``t`` under ``program``:
    a new clip stands at 0 until the program starts.
    """
    if t < START:
        return Fraction(0)
    return PROGRAMS[program](t - START)


def sprite_time(program: str, t: Fraction) -> Fraction:
    """The sprite's: the clip's, at a third of its times."""
    return clip_time(program, t) / 3


def frame_of(media: Fraction, of: tuple[Fraction, Fraction, int]) -> int:
    """The frame the media time ``media`` shows in a clip of ``of``
    (duration, rate, frame count).
    """
    _, rate, count = of
    return min(math.floor(media * rate), count - 1)
