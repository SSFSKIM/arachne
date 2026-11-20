"""Scenes of rigs: posed in a play with a whole-rig move, in either order;
a pose through the keys of a reshaped part; a slot switched by a pose, alone
and while the rig moves; two poses of one rig in one play; one rig morphed
into another.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    Flow,
    Line,
    MappingCamera,
    MorphParts,
    Pose,
    Rig,
    Scene,
    Transform,
    VGroup,
)
from arachne.utils.rate_functions import ease_out_back

#: How far the box moves while it opens.
MOVE = 2 * RIGHT + 0.5 * UP
#: How wide the box opens.
OPEN = 110


class PoseThenMove(Scene):
    """A pose and a whole-rig move in one play, the pose first."""

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        self.add(box)
        self.play(Pose(box, lid_open=OPEN), box.animate.shift(MOVE), run_time=1)
        self.wait(0.25)


class MoveThenPose(Scene):
    """A pose and a whole-rig move in one play, the move first."""

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        self.add(box)
        self.play(box.animate.shift(MOVE), Pose(box, lid_open=OPEN), run_time=1)
        self.wait(0.25)


class Swelling(Scene):
    """A part reshaped through its three stored outlines, while the rig
    turns as a whole.
    """

    def construct(self) -> None:
        section = Rig("fixture:layered-section", name="section").shift(LEFT)
        self.add(section)
        self.play(Pose(section, swell=1), run_time=1)
        self.play(section.animate.rotate(0.4).shift(DOWN), run_time=0.5)


class Switching(Scene):
    """A pose that switches a slot as it begins, and a slot switched between
    plays.
    """

    def construct(self) -> None:
        dial = Rig("fixture:dial", name="dial")
        self.add(dial)
        self.play(Pose(dial, reading=90, lamp="on"), run_time=0.5)
        dial.show(lamp="off")
        self.play(dial.animate.shift(UP), run_time=0.5)


class SwitchThenMove(Scene):
    """A pose that switches a slot, and a whole-rig move, in one play: the
    pose first.
    """

    def construct(self) -> None:
        dial = Rig("fixture:dial", name="dial")
        self.add(dial)
        self.play(
            Pose(dial, reading=90, lamp="on"), dial.animate.shift(UP), run_time=0.5
        )


class MoveThenSwitch(Scene):
    """A whole-rig move and a pose that switches a slot, in one play: the
    move first.
    """

    def construct(self) -> None:
        dial = Rig("fixture:dial", name="dial")
        self.add(dial)
        self.play(
            dial.animate.shift(UP), Pose(dial, reading=90, lamp="on"), run_time=0.5
        )


class GroupMoveThenSwitch(Scene):
    """A move of a group that holds a rig, then a pose that switches the
    rig's slot, in one play.
    """

    def construct(self) -> None:
        dial = Rig("fixture:dial", name="dial")
        group = VGroup(dial, Dot(2 * LEFT))
        self.add(group)
        self.play(
            group.animate.shift(UP), Pose(dial, reading=90, lamp="on"), run_time=0.5
        )


class SwitchThenGroupMove(Scene):
    """:class:`GroupMoveThenSwitch` with the pose first."""

    def construct(self) -> None:
        dial = Rig("fixture:dial", name="dial")
        group = VGroup(dial, Dot(2 * LEFT))
        self.add(group)
        self.play(
            Pose(dial, reading=90, lamp="on"), group.animate.shift(UP), run_time=0.5
        )


class FadedThenSwitched(Scene):
    """A rig faded by an animation, then a slot switched to a drawing hidden
    while it faded.
    """

    def construct(self) -> None:
        dial = Rig("fixture:dial", name="dial")
        self.add(dial)
        self.play(dial.animate.set_opacity(0.4), run_time=0.25)
        dial.show(lamp="on")
        self.wait(0.25)


class TransformedThenSwitched(Scene):
    """A rig transformed into a faded copy of itself, then a slot switched
    to a drawing hidden while it faded.
    """

    def construct(self) -> None:
        dial = Rig("fixture:dial", name="dial")
        self.add(dial)
        self.play(Transform(dial, dial.copy().set_opacity(0.4)), run_time=0.25)
        dial.show(lamp="on")
        self.wait(0.25)


class LiveWithUnknownAnchor(Scene):
    """An anchor the asset does not have, asked before the play, under a
    camera the record cannot draw.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(camera_class=MappingCamera, **kwargs)  # type: ignore[arg-type]

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        self.add(box)
        box.anchor("lid_tip")
        self.play(Pose(box, lid_open=30), run_time=0.25)


class TwoPoses(Scene):
    """Two poses of one rig's two parameters in one play."""

    def construct(self) -> None:
        figure = Rig("fixture:figure", name="figure")
        self.add(figure)
        self.play(Pose(figure, reach=1), Pose(figure, head_turn=25), run_time=0.5)


class TwoPosesReversed(Scene):
    """:class:`TwoPoses` with its poses in the other order."""

    def construct(self) -> None:
        figure = Rig("fixture:figure", name="figure")
        self.add(figure)
        self.play(Pose(figure, head_turn=25), Pose(figure, reach=1), run_time=0.5)


class States(Scene):
    """A rig posed twice between plays, the second time after a wait."""

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        self.add(box)
        self.play(Pose(box, lid_open=40), run_time=0.5)
        self.wait(0.25)
        box.pose(lid_open=90)
        self.wait(0.25)


class Morphing(Scene):
    """A box morphed into a crate."""

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box").shift(LEFT)
        box.pose(lid_open=30)
        crate = Rig("fixture:lidded-crate", name="crate")
        self.add(box)
        self.play(MorphParts(box, crate), run_time=0.5)
        self.play(Pose(crate, lid_open=80), run_time=0.5)


class StillWithUnknownAnchor(Scene):
    """An anchor the asset does not have, asked after the last play."""

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        self.add(box)
        self.play(Pose(box, lid_open=30), run_time=0.25)
        box.anchor("lid_tip")


class PoseBesideFlow(Scene):
    """A pose in a scene whose flow keeps the scene's clock."""

    def construct(self) -> None:
        wire = Line(3 * LEFT + 2 * DOWN, 3 * RIGHT + 2 * DOWN)
        self.add(wire, Flow(wire, speed=1.0))
        box = Rig("fixture:lidded-box", name="box")
        self.add(box)
        self.play(Pose(box, lid_open=OPEN), run_time=1)
        self.wait(0.5)


class Overshooting(Scene):
    """A pose whose rate function runs past its target, well inside the
    parameter's tested range.
    """

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        self.add(box)
        self.play(Pose(box, lid_open=50, rate_func=ease_out_back), run_time=1)
