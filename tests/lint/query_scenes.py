"""Scenes the query and key-frame tests execute.

A comment ``# @name`` tags a line the tests look up by name with
:func:`tests.lint.conftest.line_of`.
"""

from __future__ import annotations

import struct
import tempfile
import wave
from fractions import Fraction
from pathlib import Path

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Dot,
    FadeIn,
    Line,
    MovingCameraScene,
    Scene,
    Square,
    Text,
    linear,
)
from arachne.lint.bridge import declare


class DotsOnALine(Scene):
    """Two dots riding a line, a filled square and a word."""

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


def _click() -> Path:
    """A twentieth of a second of sound, written once."""
    path = Path(tempfile.gettempdir()) / "arachne-query-click.wav"
    if not path.exists():
        with wave.open(str(path), "w") as out:
            out.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            out.writeframes(struct.pack("h", 1000) * 400)
    return path


class Narrated(Scene):
    """Two sections; in the first a moment tag, a sound and a 0.4 s
    subcaption, all placed at 0.5 s as the wait (0.5-1.0 s) begins.
    """

    def construct(self):
        self.next_section("intro")  # @intro
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.5)
        self.mark("beat", n=1)  # @beat
        self.add_sound(str(_click()))  # @sound
        self.add_subcaption("hello", duration=0.4)  # @caption
        self.wait(0.5)
        self.next_section("second")  # @second
        self.play(dot.animate.shift(UP), run_time=0.5)


class SlidingWords(MovingCameraScene):
    """Under a widened view, one word slides through another during 0-1 s,
    a text gap declared: at 4/15 s they are apart, at 7/15 s they touch.
    """

    def construct(self):
        self.camera.frame.set(width=10)
        moving = Text("ONE", font_size=36).set(width=1).move_to(LEFT * 3)
        still = Text("TWO", font_size=36).set(width=1)
        self.add(moving, still)
        declare(text_gap=0.3)
        self.play(moving.animate.shift(RIGHT * 6), run_time=1, rate_func=linear)


class LateSound(Scene):
    """A sound placed at the start of a 1 s wait, to sound half-way in."""

    def construct(self):
        self.next_section("intro")
        self.add(Circle())
        self.add_sound(str(_click()), time_offset=0.5)
        self.wait(1)


class TaggedByAnUpdater(Scene):
    """A moment tag an updater places half-way through a 1 s wait."""

    def construct(self):
        self.next_section("intro")
        circle = Circle()
        self.add(circle)
        placed = []

        def tag(mob, dt):
            if self.time >= Fraction(1, 2) and not placed:
                self.mark("halfway")
                placed.append(True)

        circle.add_updater(tag)
        self.wait(1)


class KeepsItsOwnRecord(Scene):
    """A scene whose code takes ``record`` and ``timeline`` for its own: a
    method that keeps entries, and a value. Two words overlap all along.
    """

    def record(self, entry):
        self.entries.append(entry)

    def construct(self):
        self.entries = []
        self.timeline = "the scene's own"
        top = Text("top", font_size=36)
        low = Text("low", font_size=36).move_to(top.get_center() + DOWN * 0.1)
        self.add(top, low)
        self.record("added")
        self.play(low.animate.shift(RIGHT * 0.1), run_time=0.5)
        self.wait(0.5)
