"""Scenes for the motion rule: bodies that cross the cut between two plays.

A comment ``# @name`` tags a line the tests look up with
:func:`tests.lint.conftest.line_of`.
"""

from __future__ import annotations

from arachne import (
    DEGREES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    MappingCamera,
    MovingCameraScene,
    Scene,
    Square,
    ThreeDScene,
    VGroup,
    linear,
)
from tests.lint import run_counter


class Teleport(Scene):
    """A puck slides right for a second, then is set down 4 units further
    on before a wait: the viewer sees it vanish and reappear.
    """

    def construct(self):
        puck = Dot(LEFT * 3, radius=0.2)  # @puck
        self.add(puck)
        self.play(puck.animate.shift(RIGHT * 2))
        puck.move_to(RIGHT * 3)
        self.wait()  # @after-teleport


class RefusedTeleport(Teleport):
    """The same teleport, through a camera the record can't draw with."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=MappingCamera, **kwargs)


class TiltedTeleport(ThreeDScene):
    """The teleport under a tilted three-dimensional camera."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT * 2))
        puck.move_to(RIGHT * 3)
        self.wait()


class Nudge(Scene):
    """A slow slide, then a set-down 0.2 units on: under the default
    tolerance, and beyond twice the slide's steps.
    """

    def construct(self):
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), run_time=3)
        puck.shift(RIGHT * 0.2)
        self.wait()


class StaticReset(Scene):
    """A label nothing moves is rearranged between two waits: a layout
    change, not motion.
    """

    def construct(self):
        tag = Square(side_length=0.6).shift(LEFT * 2)
        self.add(tag)
        self.wait()
        tag.shift(RIGHT * 4)
        self.wait()


class PlacedThenThrown(Scene):
    """A stone rests through a wait, is set down 3 units away, and is then
    thrown from there: it moves only after the cut.
    """

    def construct(self):
        stone = Dot(LEFT * 3, radius=0.2)  # @stone
        self.add(stone)
        self.wait()
        stone.shift(RIGHT * 3)
        self.play(stone.animate.shift(UP * 2))  # @throw


class CameraCut(MovingCameraScene):
    """A kite drifts, then the camera cuts away and zooms in between plays:
    the picture moves, the kite does not.
    """

    def construct(self):
        kite = Square(side_length=0.5).shift(LEFT * 2)
        self.add(kite)
        self.play(kite.animate.shift(RIGHT))
        self.camera.frame.shift(RIGHT * 3).scale(0.5)
        self.wait()
        self.play(kite.animate.shift(UP * 0.5))


class FadedAcross(Scene):
    """A comet fades out as it moves, is moved while unseen, and fades in
    as it moves on: never visible on both sides of a cut while elsewhere.
    """

    def construct(self):
        comet = Dot(LEFT * 3, radius=0.2)
        self.add(comet)
        self.play(FadeOut(comet, shift=RIGHT))
        comet.move_to(RIGHT * 2)
        self.play(FadeIn(comet, shift=RIGHT))


class RedrawnRig(Scene):
    """A rig of a pin and a ring slides up; between plays the ring is redrawn
    much larger around where it stood. The rig's box moves, its parts don't.
    """

    def construct(self):
        rig = VGroup(Dot(LEFT * 2), Circle(radius=0.2).move_to(RIGHT * 2))
        self.add(rig)
        self.play(rig.animate.shift(UP))
        ring = rig[1]
        ring.become(Circle(radius=1.5).move_to(ring.get_center()))
        self.wait()


class Restarted(Scene):
    """A bar slides up; between plays it is redrawn three times as long
    lower down: another shape, not the same one elsewhere.
    """

    def construct(self):
        bar = Line(LEFT, RIGHT)
        self.add(bar)
        self.play(bar.animate.shift(UP))
        bar.put_start_and_end_on(LEFT * 3 + DOWN, RIGHT * 3 + DOWN)
        self.wait()


class FastReset(Scene):
    """A dart crosses the frame fast; between plays it is set back 1.2
    units, less than twice its own steps between frames.
    """

    def construct(self):
        dart = Dot(LEFT * 4, radius=0.1)
        self.add(dart)
        self.play(dart.animate.shift(RIGHT * 6), run_time=0.5, rate_func=linear)
        dart.shift(LEFT * 1.2)
        self.wait()


class SlowReset(Scene):
    """The same set-back after a slow crossing: its steps were small."""

    def construct(self):
        dart = Dot(LEFT * 4, radius=0.1)
        self.add(dart)
        self.play(dart.animate.shift(RIGHT * 1), run_time=2, rate_func=linear)
        dart.shift(LEFT * 1.2)
        self.wait()


class Conveyor(Scene):
    """Two crates ride a belt two units right; between plays each is moved
    on two more, the first onto where the second stood: the picture shows
    a crate in each place, not one crossing the gap.
    """

    def construct(self):
        belt = VGroup(
            Square(side_length=0.4).move_to(LEFT * 3),
            Square(side_length=0.4).move_to(LEFT * 1),
        )
        self.add(belt)
        self.play(belt.animate.shift(RIGHT * 2))
        belt[1].shift(RIGHT * 2)
        belt[0].shift(RIGHT * 2)
        self.wait()


class PairJumps(Scene):
    """A pair slides, then is set down 3 units on, whole."""

    def construct(self):
        pair = VGroup(Dot(LEFT * 3), Square(side_length=0.5).move_to(LEFT * 2))  # @pair
        self.add(pair)
        self.play(pair.animate.shift(UP))
        pair.shift(RIGHT * 3)
        self.wait()


class MemberJumps(Scene):
    """A pair slides; then only its square is set down 3 units on."""

    def construct(self):
        pair = VGroup(Dot(LEFT * 3), Square(side_length=0.5).move_to(LEFT * 2))
        self.add(pair)
        self.play(pair.animate.shift(UP))
        pair[1].shift(RIGHT * 3)
        self.wait()


class NamedLate(Scene):
    """A puck the code has no name for slides and is set down elsewhere; the
    code names it only in a later play.
    """

    def construct(self):
        self.add(Dot(LEFT * 3, radius=0.2))  # @unnamed
        self.play(self.mobjects[0].animate.shift(RIGHT * 2))
        self.mobjects[0].move_to(RIGHT * 3)
        self.wait()
        puck = self.mobjects[0]
        self.play(puck.animate.shift(UP))


class ZoomCut(MovingCameraScene):
    """A puck slides steadily; between plays only the camera zooms in tenfold.
    Its last frame is one step short of where it stops, which the zoom
    makes long on screen: still no move of the puck.
    """

    def construct(self):
        puck = Dot(LEFT * 0.5)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), rate_func=linear)
        self.camera.frame.scale(0.1)
        self.wait()


class ZoomedTeleport(MovingCameraScene):
    """A puck slides, is set down 2 units on, and the camera zooms in."""

    def construct(self):
        puck = Dot(LEFT * 1.5)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), rate_func=linear)
        puck.shift(RIGHT * 2)
        self.camera.frame.scale(0.5)
        self.wait()


class Repeated(Scene):
    """A puck slides and is put back five times over."""

    def construct(self):
        puck = Dot(LEFT * 3)
        self.add(puck)
        for _ in range(5):
            self.play(puck.animate.shift(RIGHT), rate_func=linear)
            puck.move_to(LEFT * 3)
            self.wait(0.5)


class CountedRuns(Scene):
    """A teleport whose length is how many times the scene has run, through
    a camera the record can't draw with: every forward run of it differs.
    """

    def __init__(self, **kwargs):
        super().__init__(camera_class=MappingCamera, **kwargs)

    def construct(self):
        run_counter.executions += 1
        puck = Dot(LEFT * 3)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), rate_func=linear)
        puck.shift(RIGHT * run_counter.executions)
        self.wait()
