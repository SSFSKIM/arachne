"""Scenes for the temporal rules: what stays wrong on screen for a while,
and what departs from a frame the viewer had settled on.

Each defect has a twin that differs only in what makes it a defect. The
scenes use only what Manim CE also has, so the same file runs there.
"""

from __future__ import annotations

from arachne import (
    BLUE,
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Circle,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    MovingCameraScene,
    Rectangle,
    Scene,
    Square,
    Text,
    ThreeDScene,
    VGroup,
    always_redraw,
)


class CaptionParkedAtEdge(Scene):
    """A caption faded in across the right edge, held there for two
    seconds, then slid to the bottom: the scene ends clean.
    """

    x = 6.4

    def construct(self):
        caption = Text("Edge of reason", font_size=44).move_to(RIGHT * self.x)
        self.play(FadeIn(caption))
        self.wait(2)
        self.play(caption.animate.move_to(DOWN * 2))
        self.wait(0.5)


class CaptionParkedInside(CaptionParkedAtEdge):
    """The same caption, held well inside the frame."""

    x = 1.5


class TokenDrifts(Scene):
    """A word pushed right by an updater during a wait of ``drift``
    seconds, until it stops at ``stop``; then it is set back in the middle.
    """

    stop = 6.8
    drift = 4.5

    def construct(self):
        token = Text("token", font_size=32).move_to(LEFT * 2)
        self.add(token)
        self.wait(0.5)
        stop = self.stop
        token.add_updater(
            lambda m, dt: m.shift(RIGHT * 3 * dt) if m.get_center()[0] < stop else None
        )
        self.wait(self.drift)
        token.clear_updaters()
        token.move_to(UP)
        self.wait(0.5)


class TokenStopsShort(TokenDrifts):
    """The same drift, stopping well inside the frame."""

    stop = 3.5


class TokenDriftsBriefly(TokenDrifts):
    """The same drift in a shorter wait: at the edge for just over a second."""

    drift = 4.0


class TagOrphaned(Scene):
    """A one-letter tag set under a box, which then moves off without it."""

    def construct(self):
        box = Square(1.2).shift(LEFT * 2.5)
        tag = Text("B", font_size=32).next_to(box, DOWN, buff=0.12)
        self.play(FadeIn(box), FadeIn(tag))
        self.wait(1)
        self.play(box.animate.shift(RIGHT * 4.5))
        self.wait(2)


class TagFollows(Scene):
    """The same box and tag, moved together."""

    def construct(self):
        box = Square(1.2).shift(LEFT * 2.5)
        tag = Text("B", font_size=32).next_to(box, DOWN, buff=0.12)
        self.play(FadeIn(box), FadeIn(tag))
        self.wait(1)
        self.play(VGroup(box, tag).animate.shift(RIGHT * 4.5))
        self.wait(2)


class NoteRidesRail(Scene):
    """A word laid across a line for a second and a half, then lifted off."""

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("rail", font_size=36).move_to(rail.get_center())
        self.add(rail)
        self.play(FadeIn(note))
        self.wait(1.5)
        self.play(note.animate.shift(UP * 1.5))
        self.wait(0.5)


class NoteOnCard(Scene):
    """A word laid on a filled card for as long: a readout on its card."""

    def construct(self):
        card = Rectangle(width=2.4, height=0.9, fill_opacity=1, color=BLUE)
        note = Text("card", font_size=36).move_to(card.get_center())
        self.add(card)
        self.play(FadeIn(note))
        self.wait(1.5)
        self.play(note.animate.shift(UP * 1.5))
        self.wait(0.5)


class PuckSlidesOut(Scene):
    """A filled puck pushed past the right edge by an updater, held there,
    then set back.
    """

    def construct(self):
        puck = Circle(radius=0.4, fill_opacity=1).move_to(LEFT * 2)
        self.add(puck)
        self.wait(0.5)
        puck.add_updater(
            lambda m, dt: m.shift(RIGHT * 4 * dt) if m.get_center()[0] < 7.2 else None
        )
        self.wait(4)
        puck.clear_updaters()
        puck.move_to(ORIGIN)
        self.wait(0.5)


class PuckCarriedOff(Scene):
    """The same puck, carried wholly off the frame by an animation."""

    def construct(self):
        puck = Circle(radius=0.4, fill_opacity=1).move_to(LEFT * 2)
        self.add(puck)
        self.wait(0.5)
        self.play(puck.animate.shift(RIGHT * 10))
        self.wait(2)


class TagChasesDot(Scene):
    """A tag rebuilt every frame beside a dot that runs to the right edge
    and waits there, then comes back.
    """

    def construct(self):
        dot = Dot(LEFT * 2)
        tag = always_redraw(
            lambda: Text("dot", font_size=30).next_to(dot, RIGHT, buff=0.2)
        )
        self.add(dot, tag)
        self.play(dot.animate.move_to(RIGHT * 6.4))
        self.wait(1.5)
        self.play(dot.animate.move_to(ORIGIN))


class SettledNoteFades(Scene):
    """A word sat across a line for ``hold`` seconds, then faded out."""

    hold = 1.0

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36).move_to(rail.get_center())
        self.add(rail, note)
        self.wait(self.hold)
        self.play(FadeOut(note))  # @fade
        self.wait(0.5)


class NoteFadesAtHalf(SettledNoteFades):
    """The same, faded after exactly the half second that settles it."""

    hold = 0.5


class SettledNoteFadesInAGroup(Scene):
    """The settled word faded out inside a group made only for the fade."""

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36).move_to(rail.get_center())
        mark = Dot(UP * 2)
        self.add(rail, note, mark)
        self.wait(1)
        self.play(FadeOut(VGroup(note, mark)))  # @group-fade
        self.wait(0.5)


class NoteSweptAcross(Scene):
    """A word carried onto the line and faded the moment it lands."""

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36).move_to(UP * 2)
        self.add(rail, note)
        self.wait(1)
        self.play(note.animate.move_to(rail.get_center()))
        self.play(FadeOut(note))
        self.wait(0.5)


class SettledNoteRemoved(Scene):
    """A word sat across a line, then taken out by the script."""

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36).move_to(rail.get_center())
        self.add(rail, note)
        self.wait(1)
        self.remove(note)
        self.wait(0.5)


class FixedCaptionAtEdge(ThreeDScene):
    """Under a tilted camera, a caption fixed in the frame across its right
    edge, and a word of the scene by the same edge, held.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-90 * DEGREES)
        fixed = Text("fixed", font_size=40).move_to(RIGHT * 6.8 + UP * 3)
        self.add_fixed_in_frame_mobjects(fixed)
        loose = Text("loose", font_size=40).move_to(RIGHT * 6.8 + DOWN)
        self.add(loose)
        self.wait(2)


class ZoomedPastCaption(MovingCameraScene):
    """The camera closes in on the frame's left half: one caption is left
    outside the view, another is cut by its edge, and the view holds.
    """

    def construct(self):
        far = Text("far", font_size=36).move_to(RIGHT * 5)
        cut = Text("cut here", font_size=36).move_to(UP * 2)
        self.add(far, cut)
        self.play(self.camera.frame.animate.scale(0.5).move_to(LEFT * 3.5))
        self.wait(2)
