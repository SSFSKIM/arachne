"""Scenes for the motion rule: bodies that cross the cut between two plays.

A comment ``# @name`` tags a line the tests look up with
:func:`tests.lint.conftest.line_of`.
"""

from __future__ import annotations

from arachne import (
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    PI,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    ImageMobjectFromCamera,
    Line,
    MappingCamera,
    MovingCamera,
    MovingCameraScene,
    Rotate,
    Scene,
    Square,
    ThreeDScene,
    ValueTracker,
    VGroup,
    always_redraw,
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


class ArmedTeleport(Teleport):
    """The same teleport, the motion rule armed before the first play."""

    def construct(self):
        self.timeline.arm_motion()
        super().construct()


class ArmedAtTwo(Scene):
    """Two teleports, the motion rule armed between them, before play 2."""

    armed = True

    def construct(self):
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT * 2))
        puck.move_to(RIGHT * 3)
        self.wait()
        if self.armed:
            self.timeline.arm_motion()
        self.play(puck.animate.shift(LEFT * 2))
        puck.move_to(LEFT * 3)
        self.wait()  # @after-armed-teleport


class TwoTeleports(ArmedAtTwo):
    """The same two teleports, the rule never armed."""

    armed = False


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


class FollowedZoomCut(MovingCameraScene):
    """The camera follows a sliding puck, then only zooms in tenfold between
    plays: the puck stands still on screen all along, and does not jump.
    """

    def construct(self):
        puck = Dot()
        self.add(puck)
        self.play(
            puck.animate.shift(RIGHT),
            self.camera.frame.animate.shift(RIGHT),
            rate_func=linear,
        )
        self.camera.frame.scale(0.1)
        self.wait()


class ZoomedReset(MovingCameraScene):
    """A puck slides, is set back half a unit, and the camera zooms in
    tenfold: a jump the zoom makes large.
    """

    def construct(self):
        puck = Dot(LEFT * 0.5)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), rate_func=linear)
        puck.shift(LEFT * 0.5)
        self.camera.frame.scale(0.1)
        self.wait()


class PanReset(MovingCameraScene):
    """A puck slides one unit right while the camera pans three left; set
    back 0.4 between plays: within twice its steps as the viewer saw them.
    """

    set_back = 0.4

    def construct(self):
        puck = Dot()
        self.add(puck)
        self.play(
            puck.animate.shift(RIGHT),
            self.camera.frame.animate.shift(LEFT * 3),
            rate_func=linear,
        )
        puck.shift(LEFT * self.set_back)
        self.wait()


class PanResetFar(PanReset):
    """The same pan, set back 0.7: past twice its steps on screen."""

    set_back = 0.7


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


class LateComer(Scene):
    """A comet joins only after two plays, slides, and is set down 3 units
    on before a wait.
    """

    def construct(self):
        self.wait(0.5)
        self.play(Create(Circle(radius=0.3).shift(UP * 2)))
        comet = Dot(LEFT * 3, radius=0.15)  # @late-comer
        self.play(FadeIn(comet))
        self.play(comet.animate.shift(RIGHT))
        comet.shift(RIGHT * 3)
        self.wait()


class LiveImageJump(Scene):
    """A display of another camera's picture, transparent until that camera's
    pixels turn opaque under the same view, travels and returns, then is set
    down 3 units on.
    """

    def construct(self):
        self.timeline.arm_motion()
        source = MovingCamera(pixel_width=16, pixel_height=16, background_opacity=0)
        display = ImageMobjectFromCamera(source)
        self.add(display)
        self.wait(0.25)
        source.pixel_array[:, :, 3] = 255
        home = display.points.copy()
        start = float(self.time)

        def travel(mob, dt):
            elapsed = float(self.time) - start
            if elapsed >= 0.5 - 1e-9:
                mob.points = home
            else:
                mob.move_to(RIGHT * elapsed * 2)

        display.add_updater(travel)
        self.wait(0.5)
        display.clear_updaters()
        display.shift(RIGHT * 3)
        self.wait(0.5)  # @after-live-jump


class _CountsFrames(Scene):
    """A scene whose frame hook counts its calls: an execution on a finer
    grid than the render's counts more of them by the end of a play.
    """

    def update_to_time(self, t):
        super().update_to_time(t)
        self.count += 1

    def on_render_grid(self):
        """Whether the count so far is the render's (15 fps for a second)."""
        return self.count < 20


class ArmedByTheCount(_CountsFrames):
    """Four plays, the same on any grid; the scene arms the motion rule
    before play 1 on the render's grid and before play 3 on a finer one.
    Play 1 slides the puck, which is set down 4 units on before play 2.
    """

    def construct(self):
        self.count = 0
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), run_time=1)
        render = self.on_render_grid()
        if render:
            self.timeline.arm_motion()
        self.play(puck.animate.shift(RIGHT), run_time=1)
        puck.move_to(RIGHT * 3)
        self.wait()  # @after-counted-teleport
        if not render:
            self.timeline.arm_motion()
        self.play(puck.animate.shift(DOWN), run_time=1)


class PlaysByTheCount(_CountsFrames):
    """A puck slid for a second and set down 3 units on; then, on the
    render's grid, one play of a second, and on a finer one two of half a
    second each: the same two seconds, split otherwise.
    """

    def construct(self):
        self.count = 0
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), run_time=1)
        puck.shift(RIGHT * 3)
        if self.on_render_grid():
            self.play(puck.animate.shift(UP), run_time=1)
        else:
            self.play(puck.animate.shift(UP * 0.5), run_time=0.5)
            self.play(puck.animate.shift(UP * 0.5), run_time=0.5)


class FrozenByTheCount(_CountsFrames):
    """A puck slid for a second and set down 3 units on, then a wait of a
    second, frozen on a finer grid and not on the render's: the same spans
    and the same scene, one picture or every frame.
    """

    def construct(self):
        self.count = 0
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), run_time=1)
        puck.shift(RIGHT * 3)
        self.wait(1, frozen_frame=not self.on_render_grid())
        self.play(puck.animate.shift(UP), run_time=1)


class LongerByTheCount(_CountsFrames):
    """A puck slid for a second and set down 3 units on, then slid again,
    for a second on the render's grid and for half a second on a finer one.
    """

    def construct(self):
        self.count = 0
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT), run_time=1)
        puck.shift(RIGHT * 3)
        run_time = 1 if self.on_render_grid() else 0.5
        self.play(puck.animate.shift(UP), run_time=run_time)


class SwapsAfterAnOffGridStart(Scene):
    """An arrow slides for 0.37 s, so the next play starts between two
    frames at 15 fps; that play's tracker leaves 0 at once, and the arrow,
    redrawn from it, points the other way for any value above 0. At the
    play's start the tracker is still 0: the arrow stands where the slide
    left it, and turns within the play.
    """

    def construct(self):
        base = ValueTracker(0)
        s = ValueTracker(0)

        def arrow():
            way = RIGHT if s.get_value() <= 0 else LEFT
            start = RIGHT * base.get_value()
            return Arrow(start, start + way * 2, buff=0)

        self.add(always_redraw(arrow))
        self.play(base.animate.set_value(1), run_time=0.37, rate_func=linear)
        self.play(s.animate.set_value(1), run_time=1, rate_func=linear)


class SwingsAfterASlowSlide(Scene):
    """A dot slides slowly for 0.36 s, so the next play starts 0.6 of a
    frame before the next frame at 15 fps; that play swings it half a turn
    about the origin in 0.4 s. At the play's start it stands where the
    slide left it.
    """

    def construct(self):
        dot = Dot(RIGHT * 2)
        self.add(dot)
        self.play(dot.animate.shift(UP * 0.1), run_time=0.36, rate_func=linear)
        self.play(
            Rotate(dot, angle=PI, about_point=ORIGIN), run_time=0.4, rate_func=linear
        )


class TeleportBeforeAnOffGridStart(Scene):
    """The puck slides for 0.37 s and is set down 4 units on before the
    next play, which starts between two frames at 15 fps.
    """

    def construct(self):
        puck = Dot(LEFT * 3, radius=0.2)
        self.add(puck)
        self.play(puck.animate.shift(RIGHT * 2), run_time=0.37, rate_func=linear)
        puck.move_to(RIGHT * 3)
        self.play(puck.animate.shift(UP), run_time=1, rate_func=linear)
