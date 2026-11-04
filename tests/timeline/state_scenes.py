"""Scenes whose record keeps the states the lint's temporal rules read:
the scene as each play ended, and as the script removed from it.
"""

from __future__ import annotations

from arachne import (
    LEFT,
    RIGHT,
    UP,
    Dot,
    FadeOut,
    Line,
    Scene,
    Square,
    Text,
    VGroup,
)


class FadesAndRemoves(Scene):
    """Armed from the start: a word fades out (its cleanup takes it out),
    the script removes a square, then clears the scene.
    """

    armed = True

    def construct(self):
        if self.armed:
            self.timeline.arm_states()
        word = Text("word").shift(UP)
        square = Square(1).shift(LEFT * 2)
        dot = Dot(RIGHT * 2)
        self.add(word, square, dot)
        self.play(FadeOut(word))
        self.remove(square)  # @remove
        self.play(dot.animate.shift(UP))
        self.clear()  # @clear
        self.wait(0.5)


class FadesAndRemovesUnarmed(FadesAndRemoves):
    """The same, never armed."""

    armed = False


class ArmsStatesLate(Scene):
    """A removal and a play before the arming, one of each after."""

    def construct(self):
        a, b = Dot(LEFT), Dot(RIGHT)
        self.add(a, b)
        self.remove(a)
        self.wait(0.5)
        self.timeline.arm_states()
        self.remove(b)
        self.wait(0.5)


class FadesAGroupMadeForIt(Scene):
    """A word and a line faded out in a group made only for the fade."""

    def construct(self):
        rail = Line(LEFT, RIGHT)
        note = Text("note")
        self.add(rail, note)
        self.wait(0.5)
        self.play(FadeOut(VGroup(note, rail)))
