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
    Animation,
    Circle,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    Line,
    MovingCameraScene,
    Rectangle,
    Scene,
    Square,
    Text,
    ThreeDScene,
    UpdateFromAlphaFunc,
    VGroup,
    always_redraw,
    linear,
)


def at_the_end(alpha):
    """A rate that holds a play's start until its very end."""
    return 0 if alpha < 1 else 1


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


class NoteFadesAFrameLate(SettledNoteFades):
    """The same, faded one frame (at 15 fps) past settling."""

    hold = 17 / 30


class NoteFadesWellAfter(SettledNoteFades):
    """The same, faded a frame and a half past settling."""

    hold = 0.6


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
        self.remove(note)  # @remove
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


# The scene as a play leaves it -----------------------------------------------


class RimWordSnapsIn(Scene):
    """A word across the right edge for under a second, brought to the
    middle by a play that holds it still until its last instant: it was at
    the edge only until the play ended.
    """

    def construct(self):
        word = Text("edge", font_size=32).move_to(RIGHT * 6.95)
        self.add(word)
        self.wait(0.4)
        self.play(word.animate.move_to(ORIGIN), run_time=0.6, rate_func=at_the_end)


class BoxSnapsAway(Scene):
    """A box that jumps away from its tag as its play ends: the tag was
    never seen beside a still box.
    """

    def construct(self):
        box = Square(1).shift(LEFT * 3)
        tag = Text("B", font_size=32).next_to(box, DOWN, buff=0.12)
        self.add(box, tag)
        self.play(box.animate.shift(RIGHT * 4), run_time=0.6, rate_func=at_the_end)
        self.wait(1.4)


class PuckRunsOffThenBack(Scene):
    """A filled puck carried from near the edge wholly off the frame, then
    set back in the middle by the script.
    """

    def construct(self):
        puck = Circle(radius=0.4, fill_opacity=1).move_to(RIGHT * 6.5)
        self.add(puck)
        self.wait(0.6)
        self.play(puck.animate.move_to(RIGHT * 9), run_time=2, rate_func=linear)
        puck.move_to(ORIGIN)
        self.wait(0.2)


class RailSnapsUnderNote(Scene):
    """A line moved onto a settled word as its play ends, and the word then
    taken out by the script.
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4).shift(UP * 2)
        note = Text("note", font_size=36)
        self.add(rail, note)
        self.wait(0.6)
        self.play(
            UpdateFromAlphaFunc(
                rail, lambda m, a: m.move_to(UP * 2 if a < 1 else ORIGIN)
            ),
            run_time=0.6,
        )
        self.remove(note)  # @remove-note
        self.wait(0.2)


# Groups a play makes, and the script's removals ------------------------------


class PanelPulsesTwo(Scene):
    """One authored panel (a line, a word on it, a dot), two of whose
    members pulse in a group made for the pulse, then hold.
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36)
        mark = Dot(UP * 2)
        self.add(VGroup(rail, note, mark))
        self.wait(0.6)
        self.play(Indicate(VGroup(note, mark)), run_time=0.6)
        self.wait(2)


class DotTakenFromPanel(Scene):
    """A dot taken out of the panel that also holds a settled word lying
    on a line: only the dot departs.
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36)
        mark = Dot(UP * 2)
        self.add(rail, VGroup(note, mark))
        self.wait(0.6)
        self.remove(mark)
        self.wait(0.2)


class NoteRemovedAndReadded(Scene):
    """A settled word on a line, removed by the script and added straight
    back.
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36)
        self.add(rail, note)
        self.wait(0.6)
        self.remove(note)  # @readd
        self.add(note)
        self.wait(0.2)


# Sampling, and the clock's margin ----------------------------------------------


class WordSlipsToEdge(Scene):
    """One long play that holds a word in the middle for 2.6 s, then puts it
    across the right edge for its last 1.13 s.
    """

    def construct(self):
        word = Text("edge", font_size=32)
        self.add(word)
        run = 56 / 15
        self.play(
            word.animate.move_to(RIGHT * 6.95),
            run_time=run,
            rate_func=lambda a: 0 if a * run < 2.6 else 1,
        )


class CaptionHeldAtEdge(Scene):
    """A caption across the right edge for ``hold`` seconds, and the end."""

    hold = 1.0

    def construct(self):
        self.add(Text("margin call", font_size=32).move_to(RIGHT * 6.4))
        self.wait(self.hold)


class CaptionHeldAFrameMore(CaptionHeldAtEdge):
    """The same, held one frame (at 15 fps) longer."""

    hold = 16 / 15


class CaptionHeldLonger(CaptionHeldAtEdge):
    """The same, held a frame and a half longer."""

    hold = 1.1


# The scene as the script calls a play ----------------------------------------


class NoteFadesAtOnce(Scene):
    """A settled word on a line, faded by a play whose rate is already 1 at
    its first instant: no frame shows it, but the script called the fade on
    the settled frame.
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36)
        self.add(rail, note)
        self.wait(0.6)
        self.play(FadeOut(note, rate_func=lambda a: 1), run_time=0.4)  # @at-once
        self.wait(0.2)


class TakenOutAsItBegins(Animation):
    """Takes its word out of the panel holding it as the play begins, then
    removes it as it ends.
    """

    def __init__(self, word, panel, scene):
        self.panel, self.scene = panel, scene
        super().__init__(word, remover=True)

    def begin(self):
        self.panel.remove(self.mobject)
        self.scene.add(self.mobject)
        super().begin()

    def interpolate_mobject(self, alpha):
        pass


class WordLeavesItsPanel(Scene):
    """A settled word, member of a panel when the script calls the play
    that departs it; the play's beginning takes it out of the panel.
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36)
        panel = VGroup(note, Dot(UP * 2))
        self.add(rail, panel)
        self.wait(0.6)
        self.play(TakenOutAsItBegins(note, panel, self), run_time=0.4)  # @leaves
        self.wait(0.2)


class NoteFadesBetweenFrames(Scene):
    """A settled word faded by a play too short to show a frame."""

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36)
        self.add(rail, note)
        self.wait(0.61)
        self.play(FadeOut(note), run_time=0.01)  # @between
        self.wait(0.2)


class PuckJumpsOutAtOnce(Scene):
    """A filled puck inside the frame when the script calls the play that
    puts it past the right edge from its first instant on.
    """

    def construct(self):
        puck = Circle(radius=0.4, fill_opacity=1)
        self.add(puck)
        self.play(
            puck.animate.move_to(RIGHT * 7.1), run_time=0.6, rate_func=lambda a: 1
        )
        self.wait(1.4)


class StatesArmedLate(Scene):
    """A word across the right edge for 1.2 s, then set in the middle; the
    engine's record begins keeping states only then.
    """

    def construct(self):
        word = Text("edge", font_size=32).move_to(RIGHT * 6.95)
        self.add(word)
        self.wait(1.2)
        word.move_to(ORIGIN)
        timeline = getattr(self, "timeline", None)
        if timeline is not None:
            timeline.arm_states()
        self.wait(0.2)


class NoteReordered(Scene):
    """A settled word on a line, sent to the back (``bring_to_back`` removes
    it and adds it again) or, as a twin, to the front (which only adds it).
    """

    method = "bring_to_back"

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36)
        self.add(rail, note)
        self.wait(0.6)
        getattr(self, self.method)(note)  # @reorder
        self.wait(0.2)


class NoteBroughtToFront(NoteReordered):
    method = "bring_to_front"


class ForegroundNoteDropped(Scene):
    """A settled foreground word on a line, dropped from the foreground
    (which takes it out of no list ``remove`` reaches).
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36)
        self.add(rail)
        self.add_foreground_mobjects(note)
        self.wait(0.6)
        self.remove_foreground_mobjects(note)
        self.wait(0.2)
