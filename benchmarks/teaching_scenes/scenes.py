"""Ten teaching scenes for warm-process profiles.

They are written for the engine, to exercise what teaching scenes lean on:
axes and plots with labels, text and formula layout, transforms between
formulas, ``always_redraw`` over trackers, readouts, ``dt`` updaters with a
traced path, a 3D surface under a moving camera, matrices and a linear map on
a plane, bar charts that grow, and a moving camera over a grid. Each lasts a
few seconds.
"""

from __future__ import annotations

import numpy as np

from arachne import *


class PlotWithTangent(Scene):
    """Axes, a plot, axis labels, and a tangent line, a dot and a readout that
    follow a tracker through ``always_redraw``.
    """

    def construct(self):
        axes = Axes(x_range=[-3, 3, 1], y_range=[-2, 2, 1], x_length=8, y_length=5)
        labels = axes.get_axis_labels(MathTex("x"), MathTex("f(x)"))
        graph = axes.plot(lambda x: np.sin(2 * x) + 0.3 * x, color=BLUE)
        t = ValueTracker(-2.5)
        dot = always_redraw(lambda: Dot(axes.i2gp(t.get_value(), graph), color=YELLOW))
        tangent = always_redraw(
            lambda: TangentLine(
                graph, alpha=(t.get_value() + 3) / 6, length=3, color=RED
            )
        )
        readout = always_redraw(
            lambda: (
                DecimalNumber(t.get_value(), num_decimal_places=2)
                .next_to(dot, UR, buff=0.1)
                .scale(0.7)
            )
        )
        self.add(axes, labels)
        self.play(Create(graph), run_time=1)
        self.add(dot, tangent, readout)
        self.play(t.animate.set_value(2.5), run_time=3, rate_func=smooth)
        self.wait(0.5)


class BulletLayout(Scene):
    """A title and bullets laid out with ``arrange`` and ``next_to``, revealed
    with ``LaggedStart``, one bullet indicated and transformed.
    """

    def construct(self):
        title = Text("Why the derivative?", font_size=44).to_edge(UP)
        bullets = VGroup(
            *(
                Text(line, font_size=30)
                for line in (
                    "Slopes of secant lines",
                    "shrink toward one tangent",
                    "whose slope is the rate",
                    "of change at a point",
                )
            )
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        bullets.next_to(title, DOWN, buff=0.6).align_to(title, LEFT)
        dots = VGroup(*(Dot(radius=0.06).next_to(b, LEFT, buff=0.2) for b in bullets))
        self.play(Write(title), run_time=1)
        self.play(
            LaggedStart(
                *(
                    FadeIn(VGroup(d, b), shift=RIGHT * 0.3)
                    for d, b in zip(dots, bullets, strict=True)
                ),
                lag_ratio=0.3,
            ),
            run_time=2,
        )
        self.play(Indicate(bullets[2]))
        replacement = Text("whose slope is the derivative", font_size=30, color=YELLOW)
        replacement.move_to(bullets[2], aligned_edge=LEFT)
        self.play(Transform(bullets[2], replacement))
        self.wait(0.5)


class FormulaSteps(Scene):
    """A chain of formulas: written, boxed, transformed step by step."""

    def construct(self):
        steps = [
            MathTex(r"(a+b)^2"),
            MathTex(r"(a+b)(a+b)"),
            MathTex(r"a^2 + ab + ba + b^2"),
            MathTex(r"a^2 + 2ab + b^2"),
        ]
        for step in steps:
            step.scale(1.3)
        current = steps[0]
        self.play(Write(current))
        for step in steps[1:]:
            self.play(TransformMatchingShapes(current, step), run_time=1)
            current = step
        box = SurroundingRectangle(current, color=YELLOW, buff=0.2)
        note = MathTex(r"\text{expanded}").next_to(box, DOWN)
        self.play(Create(box), FadeIn(note, shift=UP * 0.2))
        self.wait(0.5)


class AreaUnderCurve(Scene):
    """Riemann rectangles and an area redrawn every frame from a tracker, with
    a formula readout rebuilt by ``always_redraw``.
    """

    def construct(self):
        axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 5, 1], x_length=7, y_length=4.5
        ).to_edge(DOWN)
        graph = axes.plot(lambda x: 0.25 * x**2 + 0.5, x_range=[0, 4], color=GREEN)
        right = ValueTracker(1.0)
        area = always_redraw(
            lambda: axes.get_area(
                graph, x_range=[0, right.get_value()], color=BLUE, opacity=0.4
            )
        )
        rects = always_redraw(
            lambda: axes.get_riemann_rectangles(
                graph, x_range=[0, right.get_value()], dx=0.25, fill_opacity=0.5
            )
        )
        label = always_redraw(
            lambda: MathTex(rf"\int_0^{{{right.get_value():.1f}}} f\,dx").to_corner(UL)
        )
        self.add(axes, graph, area, rects, label)
        self.play(right.animate.set_value(4.0), run_time=3, rate_func=linear)
        self.wait(0.5)


class NumberLineJumps(Scene):
    """A number line with labels, a dot jumping along it, braces with labels
    and integers counting up.
    """

    def construct(self):
        line = NumberLine(x_range=[0, 10, 1], length=11, include_numbers=True)
        dot = Dot(line.n2p(0), color=YELLOW)
        counter = Integer(0).next_to(line, UP, buff=1.2)
        self.play(Create(line), FadeIn(dot), FadeIn(counter))
        for k in (3, 7, 10):
            brace = BraceBetweenPoints(line.n2p(0), line.n2p(k), direction=DOWN)
            label = brace.get_tex(f"{k}")
            self.play(
                dot.animate.move_to(line.n2p(k)),
                ChangeDecimalToValue(counter, k),
                GrowFromCenter(brace),
                FadeIn(label),
                run_time=0.8,
            )
            self.play(FadeOut(brace), FadeOut(label), run_time=0.3)
        self.wait(0.3)


class SpinningPhasor(Scene):
    """A vector turned by a ``dt`` updater, its tip traced, its projections
    redrawn every frame.
    """

    def construct(self):
        plane = NumberPlane(x_range=[-4, 4], y_range=[-3, 3])
        vector = Arrow(ORIGIN, 2 * RIGHT, buff=0, color=YELLOW)
        vector.add_updater(lambda m, dt: m.rotate(1.5 * dt, about_point=ORIGIN))
        tip = always_redraw(lambda: Dot(vector.get_end(), radius=0.05))
        trace = TracedPath(vector.get_end, stroke_color=RED, stroke_width=3)
        projection = always_redraw(
            lambda: DashedLine(
                vector.get_end(), [vector.get_end()[0], 0, 0], color=GREY
            )
        )
        self.add(plane, vector, tip, trace, projection)
        self.wait(3)
        vector.clear_updaters()
        self.play(FadeOut(projection), run_time=0.5)


class SurfaceOrbit(ThreeDScene):
    """A parametric surface on 3D axes, a title fixed in frame, the camera
    moved and rotated.
    """

    def construct(self):
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-2, 2])
        surface = Surface(
            lambda u, v: axes.c2p(u, v, 0.6 * np.sin(u) * np.cos(v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(12, 12),
            fill_opacity=0.7,
        )
        surface.set_fill_by_checkerboard(BLUE_D, BLUE_E)
        title = Text("A saddle-like surface", font_size=32).to_corner(UL)
        self.set_camera_orientation(phi=1.1, theta=-0.85)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Create(axes), Write(title), run_time=1)
        self.play(Create(surface), run_time=1.5)
        self.move_camera(theta=20 * DEGREES, run_time=2)
        self.wait(0.5)


class LinearMap(LinearTransformationScene):
    """A linear map applied to the plane and its basis vectors, with its
    matrix shown.
    """

    def __init__(self, **kwargs):
        super().__init__(show_coordinates=True, leave_ghost_vectors=True, **kwargs)

    def construct(self):
        matrix = [[1, 1], [0, 1]]
        shown = Matrix(matrix).to_corner(UL).add_background_rectangle()
        self.add_foreground_mobject(shown)
        self.apply_matrix(matrix, run_time=2)
        self.wait(0.5)


class GrowingBars(Scene):
    """Bars that grow from a common baseline, values counted up beside them,
    then reordered.
    """

    def construct(self):
        values = [3, 5, 2, 6, 4]
        bars = (
            VGroup(
                *(
                    Rectangle(width=0.8, height=0.1, fill_opacity=0.8, color=c)
                    for c in (BLUE, GREEN, YELLOW, RED, PURPLE)
                )
            )
            .arrange(RIGHT, buff=0.4, aligned_edge=DOWN)
            .to_edge(DOWN, buff=1)
        )
        numbers = VGroup(*(Integer(0).scale(0.7) for _ in values))
        for number, bar in zip(numbers, bars, strict=True):
            number.add_updater(lambda n, b=bar: n.next_to(b, UP, buff=0.1))
        self.add(bars, numbers)
        self.play(
            *(
                bar.animate.stretch_to_fit_height(v, about_edge=DOWN)
                for bar, v in zip(bars, values, strict=True)
            ),
            *(ChangeDecimalToValue(n, v) for n, v in zip(numbers, values, strict=True)),
            run_time=2,
        )
        order = np.argsort(values)
        targets = [bars[i].get_x() for i in range(len(values))]
        self.play(
            *(bars[i].animate.set_x(targets[rank]) for rank, i in enumerate(order)),
            run_time=1.5,
        )
        self.wait(0.5)


class CameraOverGrid(MovingCameraScene):
    """A grid of dots, one circumscribed, the camera frame zooming in and
    panning across.
    """

    def construct(self):
        dots = VGroup(*(Dot(radius=0.08) for _ in range(120))).arrange_in_grid(
            10, 12, buff=0.35
        )
        labels = VGroup(
            *(
                MathTex(f"{k}").scale(0.4).next_to(dots[k], UP, buff=0.05)
                for k in range(0, 120, 11)
            )
        )
        self.add(dots, labels)
        self.play(self.camera.frame.animate.scale(0.5).move_to(dots[37]), run_time=1.5)
        self.play(Circumscribe(dots[37], color=YELLOW), run_time=1)
        self.play(self.camera.frame.animate.move_to(dots[82]), run_time=1.5)
        self.play(self.camera.frame.animate.scale(2).move_to(ORIGIN), run_time=1)
        self.wait(0.3)


SCENES = [
    PlotWithTangent,
    BulletLayout,
    FormulaSteps,
    AreaUnderCurve,
    NumberLineJumps,
    SpinningPhasor,
    SurfaceOrbit,
    LinearMap,
    GrowingBars,
    CameraOverGrid,
]
