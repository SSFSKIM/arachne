"""Scenes the route tests execute: one per entry of the trusted algebra, and
scenes whose callbacks route 2 admits or refuses.

A comment ``# @name`` tags a line the tests look up by name.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DOWN,
    LEFT,
    ORIGIN,
    RED,
    RIGHT,
    UP,
    Add,
    Animation,
    AnimationGroup,
    ApplyComplexFunction,
    ApplyFunction,
    ApplyMatrix,
    ApplyMethod,
    ApplyPointwiseFunction,
    Arrow,
    Axes,
    Blink,
    Broadcast,
    ChangeSpeed,
    Circle,
    Circumscribe,
    ClockwiseTransform,
    CounterclockwiseTransform,
    Create,
    CyclicReplace,
    DecimalNumber,
    Dot,
    DrawBorderThenFill,
    FadeIn,
    FadeOut,
    FadeToColor,
    Flash,
    FocusOn,
    GrowArrow,
    GrowFromCenter,
    GrowFromEdge,
    GrowFromPoint,
    Indicate,
    LaggedStart,
    LaggedStartMap,
    Line,
    MoveAlongPath,
    MoveToTarget,
    ReplacementTransform,
    Restore,
    Rotate,
    Rotating,
    ScaleInPlace,
    Scene,
    ShowPassingFlash,
    ShowPassingFlashWithThinningStrokeWidth,
    ShrinkToCenter,
    SpinInFromNothing,
    Square,
    Succession,
    Swap,
    Text,
    ThreeDScene,
    Transform,
    TransformFromCopy,
    TransformMatchingShapes,
    Uncreate,
    Unwrite,
    UpdateFromAlphaFunc,
    ValueTracker,
    VGroup,
    VMobject,
    Wait,
    Wiggle,
    Write,
    always_redraw,
    linear,
    smooth,
    there_and_back,
)
from arachne.utils.rate_functions import ease_in_out_cubic, rush_into


def _square(scene, *at):
    square = Square(side_length=1).move_to(*at) if at else Square(side_length=1)
    scene.add(square)
    return square


def _targeted(mob):
    mob.generate_target()
    mob.target.shift(RIGHT).scale(0.5)
    return mob


#: One play per entry of the trusted algebra, built from its scene.
TRUSTED_PLAYS = {
    "Transform": lambda s: Transform(_square(s, LEFT), Circle().shift(RIGHT)),
    "ReplacementTransform": lambda s: ReplacementTransform(
        _square(s), Circle(color=RED)
    ),
    "TransformFromCopy": lambda s: TransformFromCopy(
        _square(s, LEFT), Circle().shift(RIGHT)
    ),
    "ClockwiseTransform": lambda s: ClockwiseTransform(
        _square(s, LEFT), Circle().shift(RIGHT)
    ),
    "MethodAnimation": lambda s: _square(s).animate.shift(RIGHT * 2).scale(0.5),
    "MethodPathArc": lambda s: _square(s).animate(path_arc=np.pi / 2).shift(RIGHT * 2),
    "FadeIn": lambda s: FadeIn(Square(), shift=UP),
    "FadeOut": lambda s: FadeOut(_square(s), scale=0.5),
    "FadeToColor": lambda s: FadeToColor(_square(s), RED),
    "GrowFromCenter": lambda s: GrowFromCenter(Circle()),
    "GrowArrow": lambda s: GrowArrow(Arrow(LEFT, RIGHT)),
    "SpinInFromNothing": lambda s: SpinInFromNothing(Square()),
    "ShrinkToCenter": lambda s: ShrinkToCenter(_square(s)),
    "Restore": lambda s: Restore(_square(s).save_state().shift(UP).scale(2)),
    "ApplyMatrix": lambda s: ApplyMatrix([[1, 1], [0, 1]], _square(s)),
    "Swap": lambda s: Swap(_square(s, LEFT * 2), Circle().shift(RIGHT * 2)),
    "CyclicReplace": lambda s: CyclicReplace(
        _square(s, LEFT), Circle(), Dot(RIGHT * 2)
    ),
    "Rotate": lambda s: Rotate(_square(s), angle=np.pi / 3),
    "Rotating": lambda s: Rotating(_square(s), angle=np.pi / 2),
    "Indicate": lambda s: Indicate(_square(s)),
    "Create": lambda s: Create(Circle()),
    "Uncreate": lambda s: Uncreate(_square(s)),
    "DrawBorderThenFill": lambda s: DrawBorderThenFill(
        Square(fill_opacity=1, color=BLUE)
    ),
    "Write": lambda s: Write(Text("ab")),
    "Unwrite": lambda s: Unwrite(s.add(Text("ab")).mobjects[-1]),
    "ShowPassingFlash": lambda s: ShowPassingFlash(Circle(color=RED), time_width=0.4),
    "Wait": lambda s: (s.add(Square()), Wait())[1],
    "AnimationGroup": lambda s: AnimationGroup(
        Create(Circle()), _square(s).animate.shift(RIGHT), lag_ratio=0.3
    ),
    "LaggedStart": lambda s: LaggedStart(
        *(FadeIn(Dot(RIGHT * k)) for k in range(4)), lag_ratio=0.5
    ),
    "LaggedStartMap": lambda s: LaggedStartMap(
        FadeIn, VGroup(*(Dot(UP * k) for k in range(3)))
    ),
    "Succession": lambda s: Succession(
        Create(Circle()), _square(s).animate.shift(RIGHT), rate_func=smooth
    ),
    "SuccessionThirds": lambda s: Succession(
        *(Dot(RIGHT * k).animate.shift(UP) for k in range(3))
    ),
    "Broadcast": lambda s: Broadcast(Circle(), n_mobs=3),
    "Flash": lambda s: Flash(ORIGIN),
    "Circumscribe": lambda s: Circumscribe(_square(s)),
    "Animation": lambda s: Animation(_square(s)),
    "Add": lambda s: Add(Square()),
    "CounterclockwiseTransform": lambda s: CounterclockwiseTransform(
        _square(s, LEFT), Circle().shift(RIGHT)
    ),
    "MoveToTarget": lambda s: MoveToTarget(_targeted(_square(s))),
    "ApplyMethod": lambda s: ApplyMethod(_square(s).shift, RIGHT),
    "ApplyPointwiseFunction": lambda s: ApplyPointwiseFunction(
        lambda p: p + np.array([p[1], 0, 0]) * 0.3, _square(s)
    ),
    "ScaleInPlace": lambda s: ScaleInPlace(_square(s), 0.5),
    "ApplyFunction": lambda s: ApplyFunction(
        lambda m: m.shift(UP).scale(0.5), _square(s)
    ),
    "ApplyComplexFunction": lambda s: ApplyComplexFunction(
        lambda z: z * complex(0.8, 0.3), _square(s)
    ),
    "GrowFromPoint": lambda s: GrowFromPoint(Circle(), LEFT),
    "GrowFromEdge": lambda s: GrowFromEdge(Square(), DOWN),
    "ShowPassingFlashWithThinningStrokeWidth": lambda s: (
        ShowPassingFlashWithThinningStrokeWidth(Circle(color=RED))
    ),
    "TransformMatchingShapes": lambda s: TransformMatchingShapes(
        s.add(VGroup(Square(), Circle().shift(RIGHT))).mobjects[-1],
        VGroup(Circle().shift(LEFT), Square().shift(DOWN)),
    ),
}


def _trusted_scene(name: str, build):
    class TrustedPlay(Scene):
        def construct(self):
            self.play(build(self), run_time=0.75)
            self.wait(0.25)

    TrustedPlay.__name__ = TrustedPlay.__qualname__ = f"Trusted{name}"
    return TrustedPlay


for _name, _build in TRUSTED_PLAYS.items():
    globals()[f"Trusted{_name}"] = _trusted_scene(_name, _build)


#: Plays the trusted algebra refuses, and the route each takes.
REFUSED_PLAYS = {
    "OverlappingTransforms": lambda s: AnimationGroup(
        Transform(s.add(Dot()).mobjects[-1], Dot(RIGHT), run_time=2),
        Transform(s.mobjects[-1], Dot(UP), run_time=0.8),
    ),
    "Wiggle": lambda s: Wiggle(_square(s)),
    "FocusOn": lambda s: FocusOn(ORIGIN),
    "MoveAlongPath": lambda s: MoveAlongPath(Dot(), Circle()),
    "ChangeSpeed": lambda s: ChangeSpeed(
        _square(s).animate.shift(RIGHT), speedinfo={0.5: 2, 1: 0.5}
    ),
    "UserRate": lambda s: _square(s).animate(rate_func=lambda t: t * t).shift(RIGHT),
    "UserSubclass": lambda s: _Shifting(_square(s), Circle()),
    "BackwardGroup": lambda s: Succession(
        _square(s).animate.shift(RIGHT), FadeIn(Dot()), rate_func=there_and_back
    ),
}


class _Shifting(Transform):
    """A user's subclass of an admitted animation."""


def _refused_scene(name: str, build):
    class RefusedPlay(Scene):
        def construct(self):
            self.play(build(self))

    RefusedPlay.__name__ = RefusedPlay.__qualname__ = f"Refused{name}"
    return RefusedPlay


for _name, _build in REFUSED_PLAYS.items():
    globals()[f"Refused{_name}"] = _refused_scene(_name, _build)


class Compositions(Scene):
    """Composition boundaries on and off the tick grid, linear and nonlinear
    group rates: Succession parts at 1/3 and 2/3 of a 1 s play (on the tick),
    at 1/7 (off it), and where ``smooth`` and ``rush_into`` cross a part
    boundary (found by bisection).
    """

    def construct(self):
        a, b, c = Square().shift(LEFT * 3), Circle(), Dot(RIGHT * 3)
        self.add(a, b, c)
        self.play(
            Succession(
                a.animate.shift(UP),
                Transform(b, a.copy().shift(DOWN * 2)),
                c.animate.shift(LEFT),
            ),
            run_time=1,
        )
        self.play(
            Succession(
                b.animate.shift(UP),
                Transform(a, b),
                run_time=7 / 6,
            ),
            rate_func=linear,
        )
        self.play(
            Succession(
                Transform(a, Circle().shift(DOWN)),
                c.animate.move_to(a),
                b.animate.scale(0.5),
                rate_func=smooth,
            ),
            run_time=1.1,
        )
        self.play(
            AnimationGroup(
                Succession(
                    a.animate.shift(RIGHT), Transform(c, a), rate_func=rush_into
                ),
                LaggedStart(b.animate.shift(UP), FadeIn(Dot()), lag_ratio=0.6),
                rate_func=ease_in_out_cubic,
            ),
            run_time=0.9,
        )


class Readouts(Scene):
    """Admitted callbacks only: ``always_redraw``, absolute setters, a
    readout of a tracker and of another mobject, and ``DecimalNumber``.
    """

    def construct(self):
        t = ValueTracker(0)
        axes = Axes(x_range=[0, 3], y_range=[0, 3], x_length=5, y_length=3)
        dot = Dot(color=RED)
        dot.add_updater(
            lambda m: m.move_to(axes.c2p(t.get_value(), t.get_value() ** 2 / 3))
        )
        line = always_redraw(lambda: Line(axes.c2p(0, 0), dot.get_center(), color=BLUE))
        graph = always_redraw(
            lambda: axes.plot(lambda x: t.get_value() * x / 3, x_range=[0, 2])
        )
        number = DecimalNumber(0).to_corner(UP + LEFT)
        number.add_updater(lambda m: m.set_value(t.get_value()))
        tag = Dot(radius=0.05).add_updater(lambda m: m.next_to(dot, UP))  # @readout
        self.add(axes, dot, line, graph, number, tag)
        self.play(t.animate.set_value(3), run_time=1.1)
        self.play(t.animate.set_value(1), rate_func=smooth)
        mark = Dot(DOWN * 2)
        self.add(mark)
        self.play(
            UpdateFromAlphaFunc(mark, lambda m, a: m.set_x(3 * a - 1.5)), run_time=0.9
        )
        self.play(Blink(mark, blinks=2, time_on=0.2, time_off=0.15))


class Collapse(Scene):
    """An updater that puts a path's ends on two points that meet at alpha
    0.3: CE collapses the path for good there, so whether a run sees the
    collapse depends on the instants it samples.
    """

    def construct(self):
        t = ValueTracker(-0.3)
        path = VMobject().set_points_as_corners([LEFT, 0.5 * UP, RIGHT])
        path.add_updater(
            lambda m: m.put_start_and_end_on(LEFT, LEFT + RIGHT * abs(t.get_value()))
        )
        self.add(path)
        self.play(t.animate.set_value(0.7), rate_func=linear)


class Conditional(Scene):
    """The conditional alpha function: it assigns nothing below 0.3 or
    between 0.4 and 0.5.
    """

    def construct(self):
        d = Dot()
        self.add(d)
        self.play(
            UpdateFromAlphaFunc(
                d,
                lambda m, a: (
                    m.set_x(1) if 0.3 < a < 0.4 else m.set_x(0) if a >= 0.5 else None
                ),
            )
        )


class LateReader(Scene):
    """A callback that reads a mobject another admitted callback writes after
    it in CE's update order: it would see the previous sample's value.
    """

    def construct(self):
        t = ValueTracker(0)
        tag = Dot(radius=0.05)
        dot = Dot()
        tag.add_updater(lambda m: m.next_to(dot, UP))  # @late
        dot.add_updater(lambda m: m.set_x(t.get_value()))
        self.add(tag, dot)
        self.play(t.animate.set_value(2))


class EarlyReader(Scene):
    """The same callbacks with the writer first: admitted."""

    def construct(self):
        t = ValueTracker(0)
        tag = Dot(radius=0.05)
        dot = Dot()
        tag.add_updater(lambda m: m.next_to(dot, UP))
        dot.add_updater(lambda m: m.set_x(t.get_value()))
        self.add(dot, tag)
        self.play(t.animate.set_value(2))


class AnimationAsUpdater(Scene):
    """``turn_animation_into_updater`` keeps its own clock."""

    def construct(self):
        from arachne import turn_animation_into_updater

        square = Square()
        self.add(square)
        turn_animation_into_updater(Rotate(square, angle=np.pi), cycle=True)
        self.wait(0.5)


class IllusionRotation(ThreeDScene):
    """``begin_3dillusion_camera_rotation``: a camera dt updater."""

    def construct(self):
        self.add(Square())
        self.set_camera_orientation(phi=1)
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(0.5)


# Callbacks with a sign of impurity, each in its own scene.

_counter = [0]


def _impure_scene(name: str, attach):
    class Impure(Scene):
        def construct(self):
            t = ValueTracker(0)
            dot = Dot()
            attach(dot, t)
            self.add(dot)
            self.play(t.animate.set_value(1), run_time=0.5)

    Impure.__name__ = Impure.__qualname__ = f"Impure{name}"
    return Impure


def _nonlocal(dot, t):
    seen = 0

    def count(m):
        nonlocal seen
        seen += 1
        m.set_x(t.get_value())

    dot.add_updater(count)


def _append(dot, t):
    seen = []
    dot.add_updater(lambda m: (seen.append(1), m.set_x(t.get_value())))


def _random(dot, t):
    dot.add_updater(lambda m: m.set_x(np.random.random()))  # noqa: NPY002


def _print(dot, t):
    def show(m):
        print(m.get_x())
        m.set_x(t.get_value())

    dot.add_updater(show)


def _attribute(dot, t):
    def write(m):
        m.seen = t.get_value()
        m.set_x(t.get_value())

    dot.add_updater(write)


def _shift(dot, t):
    dot.add_updater(lambda m: m.shift(RIGHT * t.get_value()))


def _global(dot, t):
    dot.add_updater(lambda m: m.set_x(_counter[0] + t.get_value()))


def _pure(dot, t):
    dot.add_updater(lambda m: m.set_x(t.get_value()))


#: Each impure callback, and what its refusal says.
IMPURE = {
    "Nonlocal": (_nonlocal, "it declares nonlocal names"),
    "Append": (_append, "it calls append"),
    "Random": (_random, "numpy.random"),
    "Print": (_print, "its body is not a sequence of setter calls"),
    "Attribute": (_attribute, "it assigns to an attribute"),
    "Shift": (_shift, "Dot.shift is not an admitted setter"),
}
for _name, (_attach, _) in IMPURE.items():
    globals()[f"Impure{_name}"] = _impure_scene(_name, _attach)
ImpureGlobalList = _impure_scene("GlobalList", _global)
ImpurePure = _impure_scene("Pure", _pure)
