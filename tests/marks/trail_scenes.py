"""A trail left by a dot moving at a constant speed, through a play and a
wait.
"""

from __future__ import annotations

from arachne import RIGHT, Dot, DottedTrail, Scene, linear

#: Seconds between two dots of the trail: off the 60 Hz tick's grid.
INTERVAL = 0.12
#: Where the dot starts, and its speed in scene units per second.
START = -3.0
SPEED = 2.0


class TrailBehindADot(Scene):
    """The dot moves for 2 s and rests for 1 s."""

    def construct(self):
        dot = Dot(RIGHT * START)
        self.trail = DottedTrail(dot, interval=INTERVAL)
        self.add(self.trail, dot)
        self.play(dot.animate.shift(RIGHT * SPEED * 2), run_time=2, rate_func=linear)
        self.wait(1)


#: When the off-grid trail is made, and its interval: neither on any frame
#: rate's grid.
LATE_START = 0.37
LATE_INTERVAL = 0.016


class TrailMadeOffTheGrid(Scene):
    """A dot moving at ``SPEED`` from ``START`` for 1 s; its trail is made at
    ``LATE_START`` s, between two ticks.
    """

    def construct(self):
        dot = Dot(RIGHT * START)
        self.add(dot)
        self.play(
            dot.animate.shift(RIGHT * SPEED * LATE_START),
            run_time=LATE_START,
            rate_func=linear,
        )
        #: The scene time the trail was made at.
        self.made_at = self.time
        self.trail = DottedTrail(dot, interval=LATE_INTERVAL)
        self.add(self.trail)
        rest = 1 - LATE_START
        self.play(
            dot.animate.shift(RIGHT * SPEED * rest), run_time=rest, rate_func=linear
        )
