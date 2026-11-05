"""Scenes with a camera callback that is the scene's own code: a
fixed-orientation centre function that counts its calls and drifts with
the count, so a call more anywhere moves every frame after it.
"""

from __future__ import annotations

from arachne import DEGREES, DOWN, LEFT, RIGHT, UP, Dot, Square, ThreeDScene
from tests.lint import run_counter


class CountedCentre(ThreeDScene):
    """A label oriented about a moving dot; the scene keeps the temporal
    rules' states when ``armed`` (its calls, exits and a removal).
    """

    armed = False

    def construct(self):
        if self.armed:
            self.timeline.arm_states()
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        dot = Dot(LEFT * 2)
        spare = Dot(DOWN * 2)
        label = Square(side_length=0.4)
        calls = []

        def centre():
            calls.append(1)
            run_counter.centre_calls = len(calls)
            return dot.get_center() + UP * min(len(calls), 90) / 60

        self.add(dot, spare)
        self.add_fixed_orientation_mobjects(label, center_func=centre)
        self.play(dot.animate.shift(RIGHT * 3), run_time=0.6)
        self.remove(spare)
        self.play(dot.animate.shift(DOWN * 2), run_time=0.6)


class CountedCentreArmed(CountedCentre):
    """...keeping the states."""

    armed = True
