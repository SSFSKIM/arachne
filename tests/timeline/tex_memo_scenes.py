"""A scene the TeX construction memo's record tests execute: formulas built
once and again (finding the parsed image and not), split and coloured
parts, a number whose glyphs are formulas, and a formula an updater
rebuilds as a value changes.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    DOWN,
    UP,
    ChangeDecimalToValue,
    DecimalNumber,
    MathTex,
    Scene,
    Tex,
    Transform,
    ValueTracker,
    Write,
    always_redraw,
    linear,
)


class Formulas(Scene):
    def construct(self):
        sum_ = MathTex("a^2", "+", "b^2", "=", "c^2", substrings_to_isolate=["a"])
        self.add(sum_)
        half = MathTex(r"\frac{1}{2}").next_to(sum_, DOWN)
        self.play(Write(half), run_time=0.4)
        words = Tex("Hello", " world", tex_to_color_map={" world": BLUE}).to_edge(UP)
        self.add(words)
        again = MathTex("a^2", "+", "b^2", "=", "c^2", substrings_to_isolate=["a"])
        self.play(Transform(sum_, again.shift(DOWN * 2)), run_time=0.4)
        number = DecimalNumber(0, num_decimal_places=1).to_corner(UP)
        self.add(number)
        self.play(ChangeDecimalToValue(number, 2.5), run_time=0.4, rate_func=linear)
        tracker = ValueTracker(0)
        label = always_redraw(lambda: MathTex(str(round(tracker.get_value()))))
        self.add(label)
        self.play(tracker.animate.set_value(2), run_time=0.4, rate_func=linear)
        self.wait(0.2)
