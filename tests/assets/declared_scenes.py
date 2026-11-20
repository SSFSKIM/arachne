"""Scenes whose rigs and illustrations declare values: a figure posed and
its slots switched in one play; a box held to stepped timing through a pose
that ends off its grid, as the scene ends; a group declared an illustration
part way through, with line boil, then given another rate.
"""

from __future__ import annotations

from arachne import Boil, Circle, Pose, Rig, Scene, Square, VGroup, set_timing
from arachne.utils.rate_functions import linear

#: The figure's targets, reached at the end of its pose.
REACH = 1.0
TURN = 20.0
#: How often the stepped box is drawn, and how wide it opens.
RATE = 12
OPEN = 110
#: The boil of the declared group.
BOIL = Boil(rate=6, amplitude=2.0, seed=3)


class PosedFigure(Scene):
    """The figure at rest, then posed linearly over one second, its hand
    switched to pointing as the pose begins, then held.
    """

    def construct(self) -> None:
        figure = Rig("fixture:figure")
        self.add(figure)
        self.wait(0.5)
        self.play(
            Pose(figure, reach=REACH, head_turn=TURN, hand="point", rate_func=linear),
            run_time=1,
        )
        figure.show(eyes="closed")
        self.wait(0.5)


class SteppedBoxEndingOffGrid(Scene):
    """A box drawn 12 times a second, opened linearly from 0.37 s to
    0.87 s; the scene ends as the pose does, between two boundaries.
    """

    def construct(self) -> None:
        box = Rig("fixture:lidded-box")
        self.add(box)
        set_timing(box, RATE)
        self.wait(0.37)
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=0.5)


class DeclaredLater(Scene):
    """A group of plain shapes declared an illustration with boil at
    0.5 s, given another rate without boil at 1 s.
    """

    def construct(self) -> None:
        shapes = VGroup(Square(), Circle())
        self.add(shapes)
        self.wait(0.5)
        set_timing(shapes, RATE, boil=BOIL)
        self.wait(0.5)
        set_timing(shapes, 4)
        self.wait(0.5)
