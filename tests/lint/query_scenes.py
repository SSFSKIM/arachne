"""Scenes the query and key-frame tests execute.

A comment ``# @name`` tags a line the tests look up by name with
:func:`tests.lint.conftest.line_of`.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    FadeIn,
    Line,
    Scene,
    Square,
    Text,
    linear,
)


class DotsOnALine(Scene):
    """Two dots riding a line, a filled square too big to mark a point, and
    a word: the dots are the scene's markers.
    """

    def construct(self):
        line = Line(LEFT * 3, RIGHT * 3)
        first = Dot(LEFT * 2)  # @first
        second = Dot(RIGHT * 1.5)  # @second
        block = Square(side_length=1.0, fill_opacity=1.0).shift(UP * 2)
        word = Text("word", font_size=36).shift(DOWN * 2)
        self.add(line, first, second, block, word)
        self.play(second.animate.shift(RIGHT * 0.5), run_time=0.5, rate_func=linear)


class ManyPlays(Scene):
    """Seven plays, so a storyboard takes two sheets."""

    def construct(self):
        dot = Dot(LEFT * 3)
        self.add(dot)
        for _ in range(6):
            self.play(dot.animate.shift(RIGHT), run_time=0.2)  # @step
        self.play(FadeIn(Square(side_length=0.5)), run_time=0.2)  # @last
