"""Scenes of annotation: callouts, readouts and gauges attached to points
of rigs and of ordinary shapes, while those move and the camera moves; a
callout built in the lagging order; counts that end with the scene and
twins that wait; clipped and overlapping labels.
"""

from __future__ import annotations

from fractions import Fraction

from arachne import (
    BLUE_E,
    DOWN,
    LEFT,
    ORIGIN,
    RED_E,
    RIGHT,
    UL,
    UP,
    UR,
    BarGauge,
    Callout,
    CountUp,
    DialGauge,
    Dot,
    Line,
    MovingCameraScene,
    Pose,
    Readout,
    Rig,
    Scene,
    Square,
    VGroup,
    VMobject,
    anchor,
)
from arachne.lint.bridge import sweep as sweep_now
from arachne.utils.rate_functions import linear, smooth, there_and_back

#: How the camera's frame moves while the annotated thing moves.
CAMERA_MOVE = 0.6 * RIGHT + 0.3 * DOWN
#: How much the camera's frame narrows: it zooms in.
CAMERA_ZOOM = 0.8


class _CameraMoves(MovingCameraScene):
    def camera_moves(self):
        return self.camera.frame.animate.shift(CAMERA_MOVE).scale(CAMERA_ZOOM)


class CalloutOnMovingBox(_CameraMoves):
    """A callout on a rig's outline anchor, the rig moved as a whole by an
    animation while the camera moves.
    """

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box").pose(lid_open=40)
        knob = Callout("knob", at=box.anchor("knob"), offset=UR, name="callout")
        self.add(box, knob)
        self.play(box.animate.shift(2 * RIGHT + 0.5 * DOWN), self.camera_moves())
        self.wait(0.25)


class CalloutOnPosedMovingBox(_CameraMoves):
    """A callout on a rig's outline anchor, the rig posed while it moves
    and the camera moves.
    """

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        knob = Callout("knob", at=box.anchor("knob"), offset=UR, name="callout")
        self.add(box, knob)
        self.play(
            Pose(box, lid_open=110),
            box.animate.shift(1.5 * LEFT),
            self.camera_moves(),
        )
        self.wait(0.25)


class CalloutOnSwellingTip(_CameraMoves):
    """A callout on the outline anchor of a part that reshapes through its
    stored outlines, while the camera moves.
    """

    def construct(self) -> None:
        section = Rig("fixture:layered-section", name="section")
        tip = Callout("tip", at=section.anchor("tip"), offset=UL, name="callout")
        self.add(section, tip)
        self.play(Pose(section, swell=1), self.camera_moves())
        self.wait(0.25)


class AnnotatedShape(_CameraMoves):
    """A callout, a readout and a gauge on points of an ordinary shape (a
    corner of its box, a point along its outline, one of its points),
    while it turns and moves and the camera moves.
    """

    def construct(self) -> None:
        square = Square(1.5, name="square")
        corner = Callout("corner", at=anchor(square, at=UR), offset=UR, name="callout")
        reading = Readout(
            3.5,
            format="{:.1f} N",
            at=anchor(square, along=0.375),
            offset=0.8 * DOWN,
            name="readout",
        )
        gauge = BarGauge(
            0.4, at=anchor(square, index=0), offset=1.2 * RIGHT, name="gauge"
        )
        self.add(square, corner, reading, gauge)
        self.play(
            square.animate.rotate(0.6).shift(1.5 * LEFT + 0.5 * UP),
            self.camera_moves(),
        )
        self.wait(0.25)


def _moving_dot() -> Dot:
    dot = Dot(2 * LEFT, name="dot")
    dot.add_updater(lambda m, dt: m.shift(2.4 * RIGHT * dt))
    return dot


class CalloutAfterMovingDot(Scene):
    """A callout added after the dot its anchor is on, which the dot's own
    updater moves: the callout reads the dot where it is now.
    """

    def construct(self) -> None:
        dot = _moving_dot()
        label = Callout("dot", at=anchor(dot), offset=UP, name="callout")
        self.add(dot, label)
        self.wait(1)


class CalloutBeforeMovingDot(Scene):
    """:class:`CalloutAfterMovingDot` in the lagging order: the callout
    added before the dot, so its updater reads the dot a tick before the
    dot's own moves it.
    """

    def construct(self) -> None:
        dot = _moving_dot()
        label = Callout("dot", at=anchor(dot), offset=UP, name="callout")
        self.add(label, dot)
        self.wait(1)


class FollowingReadoutAndGauge(Scene):
    """A readout counting and a dial gauge filling while the rig they are
    attached to moves: both kept at their anchors.
    """

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        reading = Readout(0, format="{:.0f}", at=box.anchor("knob"), offset=UP)
        dial = DialGauge(0, at=box.anchor("knob"), offset=2 * RIGHT, leader=True)
        self.add(box, reading, dial)
        self.play(
            CountUp(reading, 90),
            CountUp(dial, 0.75),
            Pose(box, lid_open=90),
            box.animate.shift(RIGHT),
        )
        self.wait(0.25)


#: What each count runs: the start, the target, the format, the rate
#: function; a format that rounds, one with a unit, and one whose rate
#: function ends where it began.
COUNTS = {
    "Rounded": (0.0, 100.0, "{:.1f}", linear),
    "WithUnit": (20.0, 120.0, "{:.0f} mV", smooth),
    "ThereAndBack": (0.0, 100.0, "{:.1f}", there_and_back),
}
#: How large a count's digits are drawn: large enough that a rendered frame
#: tells one number from the next.
COUNT_FONT_SIZE = 160


class _Count(Scene):
    """A readout counted from ``start`` to ``to`` over one second, then a
    wait of half a second when ``waits``.
    """

    key = "Rounded"
    waits = False

    def construct(self) -> None:
        start, to, fmt, rate = COUNTS[self.key]
        reading = Readout(start, format=fmt, font_size=COUNT_FONT_SIZE, name="readout")
        self.add(reading)
        self.play(CountUp(reading, to, rate_func=rate), run_time=1)
        if self.waits:
            self.wait(0.5)


class CountRounded(_Count):
    key = "Rounded"


class CountRoundedThenWait(_Count):
    key = "Rounded"
    waits = True


class CountWithUnit(_Count):
    key = "WithUnit"


class CountWithUnitThenWait(_Count):
    key = "WithUnit"
    waits = True


class CountThereAndBack(_Count):
    key = "ThereAndBack"


class CountThereAndBackThenWait(_Count):
    key = "ThereAndBack"
    waits = True


class _Step(Scene):
    """A callout added before a dot that its own updater moves once, at
    the 60 Hz tick ``at``: the callout reads the dot a tick late at that
    instant alone.
    """

    at = Fraction(1, 4)

    def construct(self) -> None:
        dot = Dot(2 * LEFT, name="dot")

        def step(mob, dt):
            mob.move_to(2 * RIGHT if self.time >= self.at - 1e-9 else 2 * LEFT)

        dot.add_updater(step)
        label = Callout("dot", at=anchor(dot), offset=UP, name="callout")
        self.add(label, dot)
        self.wait(1)


class StepAtAQuarter(_Step):
    """The step at 1/4 s: a frame the lint's sampling reads at 60 fps, and
    not one of the rule's own 15 Hz instants.
    """

    at = Fraction(1, 4)


class StepAtThirtySevenSixtieths(_Step):
    """The step at 37/60 s: neither a sampled instant nor one of the rule's
    own, read only when asked for.
    """

    at = Fraction(37, 60)


class CalloutOnOffSceneDot(Scene):
    """A callout on a dot the scene never holds, moved by the scene's code
    between two waits.
    """

    def construct(self) -> None:
        dot = Dot(2 * LEFT)
        self.add(Callout("away", at=anchor(dot), offset=UP))
        self.wait(0.5)
        dot.shift(4 * RIGHT)
        self.wait(0.5)


def _tilted_box() -> tuple[Rig, Callout]:
    box = Rig("fixture:lidded-box", name="box").pose(lid_open=40)
    return box, Callout("knob", at=box.anchor("knob"), offset=UR, name="callout")


class LeaderOnTiltedLid(Scene):
    """A callout's leader landing on a tilted lid's outline anchor: inside
    the boxes of the rig's parts, on no part but at its anchor.
    """

    def construct(self) -> None:
        self.add(*_tilted_box())
        self.wait(0.25)


class LineOnTiltedLid(Scene):
    """:class:`LeaderOnTiltedLid`'s geometry with no annotation: the pill
    and a plain line where the leader was, declaring nothing.
    """

    def construct(self) -> None:
        box, callout = _tilted_box()
        start, end = callout.leader.points[0], callout.leader.points[-1]
        self.add(box, Line(start, end, stroke_width=2), callout.pill.copy())
        self.wait(0.25)


def _target() -> Square:
    return Square(2, fill_opacity=1, color=BLUE_E, name="target")


class LeaderThroughTarget(Scene):
    """A leader that crosses its whole target to reach an anchor on its far
    side.
    """

    def construct(self) -> None:
        target = _target()
        self.add(
            target, Callout("far side", at=anchor(target, at=LEFT), offset=4 * RIGHT)
        )
        self.wait(0.25)


class LeaderPastItsOutlineAnchor(Scene):
    """A leader drawn on half a unit past its outline anchor, into its
    target: deeper than a landing.
    """

    def construct(self) -> None:
        target = _target()
        callout = Callout("edge", at=anchor(target, at=LEFT), offset=3 * LEFT)
        callout.clear_updaters()
        callout.leader.put_between(callout.leader.points[0], 0.5 * LEFT)
        self.add(target, callout)
        self.wait(0.25)


class LeaderToItsCentre(Scene):
    """A leader entering its target once, to an anchor at its centre."""

    def construct(self) -> None:
        target = _target()
        self.add(target, Callout("centre", at=anchor(target), offset=3 * LEFT))
        self.wait(0.25)


class LeaderOverAnotherObject(Scene):
    """:class:`LeaderToItsCentre` with a second shape lying across the
    leader, which is not its anchor's.
    """

    def construct(self) -> None:
        target = _target()
        other = Square(0.6, fill_opacity=1, color=RED_E, name="other").move_to(
            1.7 * LEFT
        )
        self.add(target, other, Callout("centre", at=anchor(target), offset=3 * LEFT))
        self.wait(0.25)


class LeaderThroughOpenFill(Scene):
    """A filled shape drawn as an open path, its left, top and right sides
    (its fill closes the bottom), and a leader declared at its centre but
    planted from below the shape to above it: through it, not landing.
    """

    def construct(self) -> None:
        target = VMobject(fill_opacity=1, color=BLUE_E, name="target")
        target.set_points_as_corners([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
        callout = Callout("centre", at=anchor(target), offset=3 * DOWN)
        callout.clear_updaters()
        callout.leader.put_between(callout.leader.points[0], 2 * UP)
        self.add(target, callout)
        self.wait(0.25)


class ObstacleJoinsTheTarget(Scene):
    """A leader declared at its target's centre ends inside a second shape
    lying over that centre; a quarter second in, the second shape joins the
    group the anchor belongs to.
    """

    def construct(self) -> None:
        group = VGroup(_target(), name="group")
        other = Square(0.6, fill_opacity=1, color=RED_E, name="other").move_to(
            0.2 * LEFT
        )
        self.add(group, other, Callout("centre", at=anchor(group), offset=3 * LEFT))
        self.wait(0.25)
        self.remove(other)
        group.add(other)
        self.wait(0.25)


class ObstacleJoinsAnOffSceneGroup(Scene):
    """:class:`ObstacleJoinsTheTarget` with the anchor's group never added
    to the scene: its target and the obstacle are drawn on their own.
    """

    def construct(self) -> None:
        target = _target()
        group = VGroup(target, name="group")
        other = Square(0.6, fill_opacity=1, color=RED_E, name="other").move_to(
            0.2 * LEFT
        )
        self.add(target, other, Callout("centre", at=anchor(group), offset=3 * LEFT))
        self.wait(0.25)
        group.add(other)
        self.wait(0.25)


class SweptLanding(Scene):
    """:class:`LeaderOverAnotherObject`, swept from the scene's code a
    quarter second in.
    """

    def construct(self) -> None:
        target = _target()
        other = Square(0.6, fill_opacity=1, color=RED_E, name="other").move_to(
            1.7 * LEFT
        )
        self.add(target, other, Callout("centre", at=anchor(target), offset=3 * LEFT))
        self.wait(0.25)
        self.found = sweep_now()
        self.wait(0.25)


class OverlappingCallouts(Scene):
    """Two callouts whose pills cover each other."""

    def construct(self) -> None:
        square = Square(2)
        first = Callout("first", at=anchor(square, at=UR), offset=UR)
        second = Callout("second", at=anchor(square, at=UR), offset=UR + 0.1 * RIGHT)
        self.add(square, first, second)
        self.wait(0.25)


class CalloutPastTheEdge(Scene):
    """A callout whose pill runs off the frame's right edge."""

    def construct(self) -> None:
        dot = Dot(6 * RIGHT)
        self.add(dot, Callout("far out there", at=anchor(dot), offset=RIGHT))
        self.wait(0.25)


class CalloutsApart(Scene):
    """Two callouts well apart, both inside the frame."""

    def construct(self) -> None:
        square = Square(2)
        self.add(
            square,
            Callout("left", at=anchor(square, at=UL), offset=UL),
            Callout("right", at=anchor(square, at=UR), offset=UR),
        )
        self.wait(0.25)


class StillReadout(Scene):
    """A readout at the origin, unattached, that does not change."""

    def construct(self) -> None:
        self.add(Readout(1.5, format="{:.2f}", name="readout").move_to(ORIGIN))
        self.wait(0.25)


class CountOnAPlate(Scene):
    """A readout on its plate, with a leading word, counted over one
    second.
    """

    def construct(self) -> None:
        reading = Readout(
            0, format="swell {:.0f}%", pill=True, font_size=COUNT_FONT_SIZE
        )
        self.add(reading)
        self.play(CountUp(reading, 100, rate_func=linear), run_time=1)


class CountAfterAWord(Scene):
    """A readout with a leading word counted over one second, its digits
    widening after the word.
    """

    def construct(self) -> None:
        reading = Readout(0, format="swell {:.0f}%", font_size=COUNT_FONT_SIZE)
        self.add(reading)
        self.play(CountUp(reading, 100, rate_func=linear), run_time=1)
