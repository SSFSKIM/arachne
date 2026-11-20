"""Scenes of annotation: callouts, readouts and gauges attached to points
of rigs and of ordinary shapes, while those move and the camera moves; a
callout built in the lagging order; counts that end with the scene and
twins that wait; clipped and overlapping labels.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    UR,
    BarGauge,
    Callout,
    CountUp,
    DialGauge,
    Dot,
    MovingCameraScene,
    Pose,
    Readout,
    Rig,
    Scene,
    Square,
    anchor,
)
from arachne.utils.rate_functions import linear, smooth

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
#: function; a format that rounds, and one with a unit.
COUNTS = {
    "Rounded": (0.0, 100.0, "{:.1f}", linear),
    "WithUnit": (20.0, 120.0, "{:.0f} mV", smooth),
}


class _Count(Scene):
    """A readout counted from ``start`` to ``to`` over one second, then a
    wait of half a second when ``waits``.
    """

    key = "Rounded"
    waits = False

    def construct(self) -> None:
        start, to, fmt, rate = COUNTS[self.key]
        reading = Readout(start, format=fmt, name="readout")
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
