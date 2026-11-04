"""Scenes with one layout defect each, planted over a known span of scene
time, beside a twin that differs only in leaving the defect out.

The spans the tests hold findings to are in the docstrings. Only the
number line in :class:`NumbersPastTheEdge` asks for TeX.
"""

from __future__ import annotations

from arachne import (
    BLACK,
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    Arrow,
    ArrowVectorField,
    Circle,
    Create,
    FadeIn,
    Group,
    Line,
    NumberLine,
    Rectangle,
    Scene,
    Square,
    Text,
    ThreeDScene,
)

# A caption pushed past the frame's right edge -----------------------------


class CaptionPushedOut(Scene):
    """The caption slides half off the right edge during 2-3 s and is held
    there to the end at 3.5 s.
    """

    push = 9.0

    def construct(self):
        caption = Text("A caption", font_size=48).shift(LEFT * 2)
        self.play(FadeIn(caption))
        self.play(Create(Circle(radius=0.5).shift(LEFT * 4 + DOWN * 2)))
        self.play(caption.animate.shift(RIGHT * self.push))
        self.wait(0.5)


class CaptionKeptIn(CaptionPushedOut):
    push = 4.0


class CaptionGone(CaptionPushedOut):
    """The caption leaves the frame altogether (2-3.5 s)."""

    push = 14.0


class PlatedCaptionAtTheTop(Scene):
    """A caption on a wide plate of its own: the plate runs past the top
    edge, every glyph stays inside.
    """

    def construct(self):
        caption = Text("plated", font_size=36)
        caption.add_background_rectangle(opacity=1, buff=0.4)
        caption.move_to(UP * (4.0 - 0.3))
        self.add(caption)
        self.wait(0.5)


class NumbersPastTheEdge(Scene):
    """A number line wider than the frame: its end numerals run past both
    edges, as an axis's furniture may.
    """

    def construct(self):
        line = NumberLine(x_range=[-8, 8, 2], length=16, include_numbers=True)
        self.add(line)
        self.wait(0.5)


# Two texts --------------------------------------------------------------------


class WordsOnWords(Scene):
    """Two phrases fade in drawn over each other (planted 1.0-1.5 s)."""

    apart = 0.0

    def construct(self):
        first = Text("one phrase", font_size=40).shift(UP * self.apart)
        second = Text("another phrase", font_size=40).shift(
            RIGHT * 0.6 + DOWN * (0.1 + self.apart)
        )
        self.play(FadeIn(first), FadeIn(second))
        self.wait(0.5)


class WordsApart(WordsOnWords):
    apart = 1.5


class RowsTight(Scene):
    """Two rows a hair apart: closer than a declared text gap, not touching
    (planted 1.0-1.5 s).
    """

    buff = 0.08

    def construct(self):
        upper = Text("top row", font_size=40)
        lower = Text("bottom row", font_size=40).next_to(upper, DOWN, buff=self.buff)
        self.play(FadeIn(upper), FadeIn(lower))
        self.wait(0.5)


class RowsLoose(RowsTight):
    buff = 1.0


# A shape over a text ---------------------------------------------------------


class SquareOverWords(Scene):
    """An opaque square faded in after the words, on top of them (planted
    1.0-1.5 s).
    """

    away = 0.0

    def construct(self):
        words = Text("hidden words", font_size=40)
        self.add(words)
        plate = Square(side_length=1.0, fill_opacity=1.0).move_to(words)
        plate.shift(DOWN * self.away)
        self.play(FadeIn(plate))
        self.wait(0.5)


class SquareBelowWords(SquareOverWords):
    away = 2.5


# Two outlines ------------------------------------------------------------------


class RingsCross(Scene):
    """A circle's and a square's outlines cross (planted 1.0-1.5 s)."""

    offset = 0.6

    def construct(self):
        ring = Circle(radius=1.0).shift(LEFT * self.offset)
        box = Square(side_length=2.0).shift(RIGHT * self.offset)
        self.play(Create(ring), Create(box))
        self.wait(0.5)


class RingsApart(RingsCross):
    offset = 2.5


# A filled thing of several leaves ------------------------------------------------


class WordBetweenTiles(Scene):
    """Two opaque tiles held as one group (no vector group), a word in the
    gap between them: inside the pair's box, clear of each tile.
    """

    shift = 0.0

    def construct(self):
        tiles = Group(
            Square(side_length=0.6, fill_opacity=1).shift(LEFT * 1.5),
            Square(side_length=0.6, fill_opacity=1).shift(RIGHT * 1.5),
        )
        word = Text("gap", font_size=30).shift(LEFT * self.shift)
        self.add(tiles, word)
        self.wait(0.5)


class WordOnATile(WordBetweenTiles):
    shift = 1.5


# A stroke and a letter -------------------------------------------------------


class _Letter(Scene):
    """A tall letter T and a short line, the line drawn before the letter
    unless ``line_on_top``.
    """

    line_on_top = False

    def stroke(self, letter):
        raise NotImplementedError

    def construct(self):
        letter = Text("T", font_size=144)
        line = self.stroke(letter)
        self.add(*((letter, line) if self.line_on_top else (line, letter)))
        self.wait(0.5)


class LineThroughTheStem(_Letter):
    """The line crosses the stem: glyph ink."""

    def stroke(self, letter):
        foot = letter.get_bottom() + UP * 0.25
        return Line(foot + LEFT * 0.4, foot + RIGHT * 0.4)


class LineOverTheStem(LineThroughTheStem):
    line_on_top = True


class LineByTheCorner(_Letter):
    """The line cuts the letter box's empty lower-left corner, far from any
    ink.
    """

    def stroke(self, letter):
        corner = letter.get_corner(DOWN + LEFT)
        return Line(
            corner + LEFT * 0.12 + UP * 0.22, corner + RIGHT * 0.22 + DOWN * 0.12
        )


# A text on a plate, and a text with a halo ---------------------------------


class WordOnACard(Scene):
    """A word laid on an opaque card drawn under it, the two not grouped."""

    def construct(self):
        card = Rectangle(
            width=3, height=1, fill_opacity=1, fill_color=BLACK, stroke_width=0
        )
        word = Text("on a card", font_size=36)
        self.add(card, word)
        self.wait(0.5)


class LineUnderAWord(Scene):
    """A line drawn first, a word laid across it."""

    def construct(self):
        line = Line(LEFT * 2, RIGHT * 2)
        word = Text("crossed", font_size=40)
        self.add(line, word)
        self.wait(0.5)


class LineUnderACard(Scene):
    """A line drawn first, an opaque card over the middle of it, and a word
    on the card: the card hides the line wherever it meets the word.
    """

    opacity = 1.0

    def construct(self):
        line = Line(LEFT * 3, RIGHT * 3)
        card = Rectangle(
            width=3,
            height=1,
            fill_opacity=self.opacity,
            fill_color=BLACK,
            stroke_width=0,
        )
        word = Text("crossed", font_size=40)
        self.add(line, card, word)
        self.wait(0.5)


class LineUnderAFaintCard(LineUnderACard):
    opacity = 0.3


class LineUnderAPlatedWord(Scene):
    """A line drawn first, and a word on an opaque plate of its own over it,
    wider than the word.
    """

    def construct(self):
        line = Line(LEFT * 3, RIGHT * 3)
        word = Text("crossed", font_size=40)
        word.add_background_rectangle(opacity=1, buff=0.2)
        self.add(line, word)
        self.wait(0.5)


class LineUnderAHaloedWord(Scene):
    """A line drawn first, a word with a wide opaque background stroke (a
    halo) laid across it.
    """

    halo = 8

    def construct(self):
        line = Line(LEFT * 2, RIGHT * 2)
        word = Text("crossed", font_size=40)
        word.set_stroke(BLACK, width=self.halo, background=True)
        self.add(line, word)
        self.wait(0.5)


class LineUnderAThinHalo(LineUnderAHaloedWord):
    halo = 1


# A text over a field ------------------------------------------------------------


class WordOverAField(Scene):
    """A vector field drawn first, a word laid over its middle arrow."""

    def construct(self):
        field = ArrowVectorField(
            lambda p: RIGHT, x_range=[-2, 2, 1], y_range=[-1, 1, 1]
        )
        word = Text("field", font_size=40)
        self.add(field, word)
        self.wait(0.5)


class WordOverAnArrow(Scene):
    """An arrow drawn first, a word laid over it."""

    def construct(self):
        arrow = Arrow(LEFT * 1.5, RIGHT * 1.5, buff=0)
        word = Text("field", font_size=40)
        self.add(arrow, word)
        self.wait(0.5)


# Under a ThreeDCamera ------------------------------------------------------


class FarWordInDepth(ThreeDScene):
    """A word the camera's zoom puts past the right edge, though it sits
    inside the frame in scene units; and a caption fixed in the frame.
    """

    x = 4.0

    def construct(self):
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=2)
        word = Text("far", font_size=36).shift(RIGHT * self.x)
        caption = Text("fixed", font_size=36).to_edge(UP)
        self.add_fixed_in_frame_mobjects(caption)
        self.add(word)
        self.wait(0.5)


class NearWordInDepth(FarWordInDepth):
    x = 1.0


# Small text ------------------------------------------------------------------


class TinyWord(Scene):
    """A word set too small to read beside one at a readable size."""

    def construct(self):
        tiny = Text("tiny", font_size=8, color=WHITE).shift(UP)
        big = Text("readable", font_size=48).move_to(ORIGIN)
        self.add(tiny, big)
        self.wait(0.5)
