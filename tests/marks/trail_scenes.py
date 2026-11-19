"""A trail left by a dot moving at a constant speed, through a play and a
wait.
"""

from __future__ import annotations

from arachne import RIGHT, Dot, DottedTrail, Scene, linear

#: Seconds between two dots of the trail.
INTERVAL = 0.25
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
