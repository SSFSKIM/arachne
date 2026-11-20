"""Scenes of stepped timing: a declared rig posed, moved, faded in, a part of
it animated, beside smooth text, off the grid, with line boil, posed by the
scene's code between plays, with a callout on it; scene drivers beside a
clock; what is refused; and scenes that declare nothing.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    UR,
    Animation,
    AnimationGroup,
    Axes,
    Boil,
    Callout,
    Capsule,
    DecimalNumber,
    Diagram,
    Dot,
    FadeIn,
    Flow,
    Line,
    MaintainPositionRelativeTo,
    MoveAlongPath,
    MovingCameraScene,
    PhaseFlow,
    Pose,
    Rig,
    Scene,
    Square,
    Succession,
    Text,
    UpdateFromFunc,
    VGroup,
    Write,
    set_timing,
)
from arachne.assets import timing
from arachne.utils.rate_functions import linear

#: The drawings a second every declared rig here is held to.
RATE = 12
#: How wide the box opens.
OPEN = 110
#: How far the box moves.
MOVE = 2 * RIGHT + 0.5 * UP
#: The boil the boiling scene declares.
BOIL = Boil(rate=6, amplitude=2.0, seed=3)


def _box() -> Rig:
    return Rig("fixture:lidded-box", name="box")


class PosedBox(Scene):
    """A declared rig posed over a second, then a wait."""

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=1)
        self.wait(0.25)


class MovedBox(Scene):
    """A declared rig moved as a whole."""

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(box.animate(rate_func=linear).shift(MOVE), run_time=1)


class FadedInBox(Scene):
    """A declared rig faded in, declared before it is in the scene."""

    def construct(self) -> None:
        box = _box()
        set_timing(box, RATE)
        self.play(FadeIn(box, rate_func=linear), run_time=1)


class PartOfBox(Scene):
    """An animation of a part of a declared rig: the mark is on its
    ancestor.
    """

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(box.part("lid").animate(rate_func=linear).set_color(RED), run_time=1)


class PartsInTurn(Scene):
    """One part recoloured by a play that ends off the grid, then another
    part by the next play: the first part's final colour is shown from the
    next boundary, though the second play never writes it.
    """

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(
            box.part("lid").animate(rate_func=linear).set_color(RED), run_time=0.45
        )
        self.play(
            box.part("body").animate(rate_func=linear).set_color(BLUE), run_time=0.5
        )


class MovedThenPosed(Scene):
    """A move that ends off the grid, then a pose alone: the pose is drawn
    where the move left the rig, though the drawing on display between
    them stood short of it.
    """

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(box.animate(rate_func=linear).shift(RIGHT), run_time=0.45)
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=0.5)


class ShortWaitThenPose(Scene):
    """A pose that ends off the grid, a wait too short to reach the next
    boundary, then the scene's code reads the rig and poses it back.
    """

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.wait(0.37)
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=0.5)
        self.wait(0.02)
        self.read = box.value("lid_open")
        self.play(Pose(box, lid_open=0, rate_func=linear), run_time=0.5)


class BoxBesideText(Scene):
    """A declared rig and a text, animated in one group: the rig steps, the
    text is written smoothly.
    """

    def construct(self) -> None:
        box = _box().shift(2 * LEFT)
        words = Text("smooth", font_size=36).shift(2 * RIGHT)
        set_timing(box, RATE)
        self.play(
            AnimationGroup(
                FadeIn(box, rate_func=linear), Write(words, rate_func=linear)
            ),
            run_time=1,
        )


class OffGridThenWait(Scene):
    """A pose that starts and ends off the drawing grid (0.37 s to 0.87 s),
    followed by a wait.
    """

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.wait(0.37)
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=0.5)
        self.wait(0.5)


class OffGridEnding(Scene):
    """The same pose, with the scene ending as it ends."""

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.wait(0.37)
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=0.5)


class Boiling(Scene):
    """A declared rig with line boil through a play, a wait and a play."""

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE, boil=BOIL)
        self.play(Pose(box, lid_open=60, rate_func=linear), run_time=0.5)
        self.wait(1)
        self.play(box.animate(rate_func=linear).shift(RIGHT), run_time=0.5)


class SuccessionOfMoves(Scene):
    """Two moves of a boiling declared rig in one Succession, the first
    ending off the grid: the second begins in the middle of the play.
    """

    declared = True

    def construct(self) -> None:
        box = _box()
        self.add(box)
        if self.declared:
            set_timing(box, RATE, boil=BOIL)
        self.play(
            Succession(
                box.animate(rate_func=linear, run_time=0.45).shift(RIGHT),
                box.animate(rate_func=linear, run_time=0.5).shift(UP),
            )
        )


class SuccessionOfMovesUndeclared(SuccessionOfMoves):
    """The same moves, declared nothing: what is true at each instant."""

    declared = False


class PosedBetweenPlays(Scene):
    """The scene's code poses a declared rig between two plays, the first
    ending off the grid.
    """

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(box.animate(rate_func=linear).shift(RIGHT), run_time=0.45)
        box.pose(lid_open=80)
        self.play(box.animate(rate_func=linear).shift(LEFT), run_time=0.5)


class CalloutOnSteppedBox(Scene):
    """A callout on the knob of a declared rig, posed while it moves."""

    def construct(self) -> None:
        box = _box()
        knob = Callout("knob", at=box.anchor("knob"), offset=UR, name="callout")
        self.add(box, knob)
        set_timing(box, RATE)
        self.play(
            Pose(box, lid_open=OPEN, rate_func=linear),
            box.animate(rate_func=linear).shift(MOVE),
            run_time=1,
        )
        self.wait(0.25)


class BoiledCallout(Scene):
    """A callout on the knob of a boiling declared rig, posed while it
    moves.
    """

    def construct(self) -> None:
        box = _box()
        knob = Callout("knob", at=box.anchor("knob"), offset=UR, name="callout")
        self.add(box, knob)
        set_timing(box, RATE, boil=BOIL)
        self.play(
            Pose(box, lid_open=OPEN, rate_func=linear),
            box.animate(rate_func=linear).shift(MOVE),
            run_time=1,
        )


class FlowBesideSteppedBox(Scene):
    """A flow made before the timing is declared, and one made after: both
    keep the scene's time through its clock.
    """

    def construct(self) -> None:
        early = Line(3 * LEFT + 2 * DOWN, 3 * RIGHT + 2 * DOWN)
        late = Line(3 * LEFT + 2.5 * DOWN, 3 * RIGHT + 2.5 * DOWN)
        self.add(early, Flow(early, speed=1.5, name="early"))
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.add(late, Flow(late, speed=1.5, name="late"))
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=1)
        self.wait(0.5)


class CopiedBox(Scene):
    """A copy of a declared rig, which carries its mark, brought into the
    scene and posed.
    """

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        twin = box.copy().shift(2 * DOWN)
        twin.name = "twin"
        self.add(twin)
        self.play(Pose(twin, lid_open=OPEN, rate_func=linear), run_time=1)


class NestedDeclarations(Scene):
    def construct(self) -> None:
        box = _box()
        group = VGroup(box)
        self.add(group)
        set_timing(box, RATE)
        set_timing(group, 6)


class SharedMember(Scene):
    """Two declared groups holding one rig."""

    def construct(self) -> None:
        box = _box()
        first, second = VGroup(Square(), box), VGroup(Square(), box)
        self.add(first, second)
        set_timing(first, RATE)
        set_timing(second, RATE)


class MemberJoinsSecondFamily(Scene):
    """A declared rig joins a second declared group between two waits."""

    def construct(self) -> None:
        box = _box()
        first, second = VGroup(box), VGroup(Square())
        self.add(first, second)
        set_timing(first, RATE)
        set_timing(second, RATE)
        self.wait(0.25)
        second.add(box)
        self.wait(0.25)


class Redeclared(Scene):
    """A declared rig given another rate between two plays."""

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(Pose(box, lid_open=50, rate_func=linear), run_time=0.5)
        set_timing(box, 4)
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=0.5)


#: What the stub drivers saw, by scene name: each call's instant and the
#: box's value then.
SEEN: dict[str, list[tuple[str, object, float]]] = {}


class StubDriver:
    """A scene driver of the contract the clock takes over: it reads the
    scene's time, never adds up ``dt``, is found on its scene, and accepts
    later objects without adding its updater again. It holds its scene.
    """

    def __init__(self, scene: Scene, name: str, box: Rig) -> None:
        self.scene, self.name, self.box = scene, name, box
        self.held: list[object] = []

        def drive(dt: float) -> None:
            self()

        self.updater = drive

    def hold(self, thing: object) -> None:
        self.held.append(thing)

    def __call__(self) -> None:
        SEEN.setdefault(type(self.scene).__name__, []).append(
            (self.name, self.scene.time, self.box.value("lid_open"))
        )


class BoundDriver(StubDriver):
    """A stub driver whose updater is a bound method: a new object at each
    lookup, equal to the one before.
    """

    def __init__(self, scene: Scene, name: str, box: Rig) -> None:
        super().__init__(scene, name, box)
        del self.updater

    def updater(self, dt: float) -> None:
        self()


def _found(kind: str):
    def find(scene: Scene) -> StubDriver | None:
        return scene.__dict__.get("stubs", {}).get(kind)

    return find


def _register_stubs(scene: Scene) -> None:
    """Register the finders of the stub drivers, first then second, for the
    test that runs the scene (its fixture lets them go), and give the scene
    a place to keep its own.
    """
    scene.stubs = {}
    timing.register_driver(_found("first"))
    timing.register_driver(_found("second"))


class StubDrivers(Scene):
    """Two stub drivers: one made before the timing is declared, one made
    after it.
    """

    def construct(self) -> None:
        _register_stubs(self)
        box = _box()
        self.add(box)
        first = self.stubs["first"] = StubDriver(self, "first", box)
        timing.drive(self, first)
        set_timing(box, RATE)
        second = self.stubs["second"] = StubDriver(self, "second", box)
        timing.drive(self, second)
        first.hold("later")
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=0.25)


class BoundDrivers(Scene):
    """Two drivers with bound-method updaters, both made before the timing
    is declared, put in the scene in the reverse of their finders' order.
    """

    def construct(self) -> None:
        _register_stubs(self)
        box = _box()
        self.add(box)
        second = self.stubs["second"] = BoundDriver(self, "second", box)
        timing.drive(self, second)
        first = self.stubs["first"] = BoundDriver(self, "first", box)
        timing.drive(self, first)
        set_timing(box, RATE)
        self.play(Pose(box, lid_open=OPEN, rate_func=linear), run_time=0.25)


# Refused


class UpdaterBeforeDeclaring(Scene):
    def construct(self) -> None:
        box = _box()
        self.add(box)
        box.add_updater(lambda m, dt: m.shift(dt * RIGHT))
        set_timing(box, RATE)  # @refused
        self.wait(0.5)


class UpdaterAfterDeclaring(Scene):
    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        box.part("lid").add_updater(lambda m: m)
        self.wait(0.5)  # @refused


class SceneUpdaterAfterDeclaring(Scene):
    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.add_updater(lambda dt: None)
        self.wait(0.5)  # @refused


class UpdateFromFuncOnPart(Scene):
    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(UpdateFromFunc(box.part("lid"), lambda m: m))  # @refused


class Nudge(Animation):
    """An animation of the scene's own."""

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.shift(0.01 * RIGHT)


class OwnAnimationOnBox(Scene):
    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        self.play(Nudge(box))  # @refused


class _Moved(Scene):
    """Plays what :meth:`move` makes of a boiling declared rig."""

    def move(self, box: Rig) -> Animation:
        raise NotImplementedError

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE, boil=BOIL)
        self.play(self.move(box), run_time=0.5)  # @refused


class AlongPath(_Moved):
    def move(self, box: Rig) -> Animation:
        return MoveAlongPath(box, Line(LEFT, RIGHT), rate_func=linear)


class ByPhaseFlow(_Moved):
    def move(self, box: Rig) -> Animation:
        return PhaseFlow(lambda p: 0.1 * RIGHT + 0 * p, box)


class KeptBesideDot(_Moved):
    def move(self, box: Rig) -> Animation:
        dot = Dot(2 * LEFT)
        self.add(dot)
        return AnimationGroup(
            dot.animate.shift(RIGHT), MaintainPositionRelativeTo(box, dot)
        )


class SuccessionAfterAnotherPart(_Moved):
    """A Succession whose second part moves the rig, after a part that
    animates something else.
    """

    def move(self, box: Rig) -> Animation:
        dot = Dot(2 * LEFT)
        return Succession(
            FadeIn(dot, run_time=0.45), box.animate(run_time=0.5).shift(UP)
        )


class SuccessionShowingAgain(_Moved):
    """The rig on screen faded in again by a Succession's second part: a
    part whose group leaves its introducers out.
    """

    def move(self, box: Rig) -> Animation:
        return Succession(
            FadeIn(Dot(2 * LEFT), run_time=0.45), FadeIn(box, run_time=0.5)
        )


class SuccessionNestedPart(_Moved):
    """The same, the rig's fade-in inside a group inside the second part."""

    def move(self, box: Rig) -> Animation:
        return Succession(
            FadeIn(Dot(2 * LEFT), run_time=0.45),
            AnimationGroup(FadeIn(box), FadeIn(Dot(2 * RIGHT)), run_time=0.5),
        )


class SuspendedByHand(Scene):
    """The scene's code suspends a declared rig's updating."""

    def construct(self) -> None:
        box = _box()
        self.add(box)
        set_timing(box, RATE)
        box.suspend_updating()
        self.wait(0.5)  # @refused


class _Declared(Scene):
    """Declares timing on what :meth:`make` makes."""

    def make(self):
        raise NotImplementedError

    def construct(self) -> None:
        thing = self.make()
        self.add(thing)
        set_timing(thing, RATE)  # @refused
        self.wait(0.5)


class TimedText(_Declared):
    def make(self):
        return Text("label")


class TimedNumber(_Declared):
    def make(self):
        return DecimalNumber(1.5)


class TimedAxes(_Declared):
    def make(self):
        return Axes(x_range=(0, 2), y_range=(0, 2))


class TimedCallout(_Declared):
    def make(self):
        return Callout("label", at=None)


class TimedMark(_Declared):
    def make(self):
        return Capsule()


class TimedDiagram(_Declared):
    def make(self):
        return Diagram(Square())


class TimedBoxWithText(_Declared):
    def make(self):
        return VGroup(_box(), Text("label"))


class TimedCameraFrame(MovingCameraScene):
    def construct(self) -> None:
        self.add(Square())
        set_timing(self.camera.frame, RATE)  # @refused
        self.wait(0.5)


class TextJoinsDeclaredGroup(Scene):
    def construct(self) -> None:
        group = VGroup(_box())
        self.add(group)
        set_timing(group, RATE)
        self.wait(0.25)
        group.add(Text("late"))
        self.wait(0.25)  # @refused


# Declaring nothing


class Undeclared(Scene):
    """A rig posed and moved, with a flow beside it: the constructs this
    timing touches, in a scene that declares nothing.
    """

    def construct(self) -> None:
        box = _box()
        edge = Line(3 * LEFT + 2 * DOWN, 3 * RIGHT + 2 * DOWN)
        self.add(box, edge, Flow(edge, speed=1.5))
        self.play(Pose(box, lid_open=OPEN), box.animate.shift(MOVE), run_time=0.5)
        box.pose(lid_open=30)
        self.wait(0.25)
        self.play(box.animate.shift(DOWN), run_time=0.5)
