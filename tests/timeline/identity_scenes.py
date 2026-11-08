"""Scenes for the identity tests: a dot class of the scene's own whose
metaclass makes it equal to ``Dot``, drawn beside a ``Dot``.
"""

from __future__ import annotations

import numpy as np

from arachne import RED, Dot, Scene, ValueTracker, VGroup, linear
from arachne.utils.color import color_to_rgb


class EqualToDot(type(Dot)):
    def __hash__(cls):
        return hash(Dot)

    def __eq__(cls, other):
        return other is Dot or other is cls


class OwnFill(Dot, metaclass=EqualToDot):
    """A dot whose fill the camera reads from the scene's code."""

    def get_fill_rgbas(self):
        return np.array([[*color_to_rgb(RED), 1.0]])


class OwnFillBesideDot(Scene):
    """A ``Dot`` and the scene's own dot, on screen from the start."""

    def construct(self):
        self.add(Dot(), OwnFill(radius=0.3))
        self.wait(0.5)


class OwnFillVisits(Scene):
    """A ``Dot`` and the scene's own dot, held together by an updater while
    the play's tracker is in [0.3, 0.6): they join and leave within the play,
    so only the captures between see them.
    """

    def construct(self):
        t = ValueTracker(0)
        holder = VGroup()

        def show(m):
            inside = 0.3 <= t.get_value() < 0.6
            if inside and not m.submobjects:
                m.add(Dot(), OwnFill(radius=0.3))
            elif not inside and m.submobjects:
                m.remove(*m.submobjects)

        holder.add_updater(show)
        self.add(holder)
        self.play(t.animate.set_value(1), run_time=1, rate_func=linear)
        self.remove(holder)
        self.wait(0.5)
