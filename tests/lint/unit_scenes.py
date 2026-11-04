"""Scenes whose drawn leaves belong to the things a sweep measures whole.

A comment ``# @name`` tags a line the tests look up by name.
"""

from __future__ import annotations

from arachne import (
    BLACK,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Circle,
    DashedLine,
    Dot,
    Group,
    Paragraph,
    Scene,
    Square,
    Text,
    VGroup,
    VMobject,
)


class Card(VMobject):
    """A container of the scene's own that draws nothing itself: a square
    holding a word.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add(Square(side_length=1.0), Text("card", font_size=24))


class Units(Scene):
    """Mobjects of every shape the sweep tells apart: a compound path, a
    plain group, a group that is no vector group, a container of the
    scene's own, a paragraph, a dashed line, a pair of axes with something
    mounted on it, a text on its own plate and a text with a halo.
    """

    def construct(self):
        arrow = Arrow(LEFT * 6, LEFT * 4)  # @arrow
        row = VGroup(  # @row
            Square(side_length=0.5).shift(UP * 3 + LEFT * 6),
            Circle(radius=0.3).shift(UP * 3 + LEFT * 5),
        )
        pile = Group(  # @pile
            Square(side_length=0.4, fill_opacity=1).shift(DOWN * 3 + LEFT * 6),
            Square(side_length=0.4, fill_opacity=1).shift(DOWN * 3 + LEFT * 5),
        )
        card = Card().shift(LEFT * 2 + UP * 2)  # @card
        lines = Paragraph("first", "second").shift(LEFT * 2 + DOWN * 2)  # @lines
        dashes = DashedLine(UP * 3, UP * 3 + RIGHT * 2)  # @dashes
        axes = Axes(  # @axes
            x_range=[0, 2, 1],
            y_range=[0, 2, 1],
            x_length=3,
            y_length=3,
            axis_config={"include_numbers": True},
        ).shift(RIGHT * 4 + DOWN)
        mounted = Dot(axes.c2p(1, 1))  # @mounted
        axes.add(mounted)
        plated = Text("plated", font_size=24).shift(UP * 1 + RIGHT * 1)  # @plated
        plated.add_background_rectangle(opacity=1)
        haloed = Text("halo", font_size=24).shift(DOWN * 0.5 + RIGHT * 1)  # @haloed
        haloed.set_stroke(BLACK, width=6, background=True)
        ringed = Square(0.6, stroke_width=8).shift(DOWN * 3 + RIGHT)  # @ringed
        faded = Square(0.6, fill_opacity=[0, 1]).shift(DOWN * 3 + RIGHT * 2)  # @faded
        self.add(arrow, row, pile, card, lines, dashes, axes, plated, haloed, ringed)
        self.add(faded)
        self.wait(0.2)
