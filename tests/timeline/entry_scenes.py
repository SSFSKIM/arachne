"""Scenes whose plays start on the frame grid or between its frames, with an
updater on a mobject no play animates, counting the calls it is handed
``dt == 0`` and the scene time of each.
"""

from __future__ import annotations

from arachne import RIGHT, Dot, Scene


class CountedStarts(Scene):
    """Three plays, each ``run_time`` long, of a dot beside the counter."""

    run_time = 0.4

    def construct(self):
        self.zero_dt = []
        counter = Dot(RIGHT * 3)

        def count(mob, dt):
            if dt == 0 and mob is counter:
                self.zero_dt.append(float(self.renderer.time))

        counter.add_updater(count)
        dot = Dot()
        self.add(counter, dot)
        for _ in range(3):
            self.play(dot.animate.shift(RIGHT * 0.5), run_time=self.run_time)


class CountedStartsOffGrid(CountedStarts):
    """The plays end, and the next start, between frames at 15 and 30 fps."""

    run_time = 0.37


class ArmedOffGridStarts(CountedStartsOffGrid):
    """The same plays, the motion rule armed first: the record keeps where
    each body stands as each play begins and ends.
    """

    def construct(self):
        self.timeline.arm_motion()
        super().construct()
