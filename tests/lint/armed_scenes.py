"""Scenes that arm the temporal rules themselves, and state the terms they
are held to: through the bridge here, through the reference checks'
``guard_fades`` and ``assert_scene_clear`` on Manim CE, call for call.

Each arms first thing in ``construct``. The departures read the arming in
force at each departure, its exemptions as the value it was handed stands
then; the hold watch, the pad and text_gap in force at each sample, and
every exemption the scene stated, read at its end.
"""

from __future__ import annotations

from arachne import DOWN, LEFT, RIGHT, UP, FadeOut, Line, Scene, Text, VGroup
from arachne.lint.bridge import arm, declare, sweep
from tests.lint import run_counter


class ExemptedNoteFades(Scene):
    """A word settled across a line, faded out: the arming excuses the
    two.
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36).move_to(rail.get_center())
        arm(exempt=[(note, rail)])
        self.add(rail, note)
        self.wait(1)
        self.play(FadeOut(note))
        self.wait(0.5)


class ExemptionGrowsBetweenRemovals(Scene):
    """Two words settled across a line, each taken out by the script; the
    list the arming was handed names the second's contact only after the
    first is gone.
    """

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        first = Text("first", font_size=36).move_to(rail.get_center() + LEFT * 2)
        second = Text("second", font_size=36).move_to(rail.get_center() + RIGHT * 2)
        excused = []
        arm(exempt=excused)
        self.add(rail, first, second)
        self.wait(0.7)
        self.remove(first)  # @remove-first
        excused.append((second, rail))
        self.remove(second)
        self.wait(0.5)


class RearmedWithoutTextGap(Scene):
    """Two pairs of words 0.3 apart, each faded out after it settled; the
    first under a text_gap of 0.5, the second after the scene re-armed
    without one.
    """

    def construct(self):
        top = Text("top", font_size=30).shift(LEFT * 3)
        low = Text("low", font_size=30).next_to(top, direction=UP, buff=0.3)
        left = Text("left", font_size=30).shift(RIGHT * 3)
        right = Text("right", font_size=30).next_to(left, direction=UP, buff=0.3)
        arm(text_gap=0.5)
        self.add(top, low, left, right)
        self.wait(0.7)
        self.play(FadeOut(low))  # @fade-low
        arm()
        self.play(FadeOut(right))
        self.wait(0.5)


class HeldGapUntilRearmed(Scene):
    """Two words 0.3 apart under a text_gap of 0.5 for two seconds; the scene
    re-arms without one and holds them two seconds more.
    """

    def construct(self):
        top = Text("top", font_size=30)
        low = Text("low", font_size=30).next_to(top, direction=UP, buff=0.3)
        arm(text_gap=0.5)
        self.add(top, low)
        self.wait(2)
        arm()
        self.wait(2)


class CaptionExcusedAtClose(Scene):
    """A caption across the right edge for two seconds, then moved in; the
    closing sweep excuses it.
    """

    def construct(self):
        arm()
        caption = Text("Edge of reason", font_size=44).move_to(RIGHT * 6.4)
        self.add(caption)
        self.wait(2)
        self.play(caption.animate.move_to(UP * 2))
        sweep(exempt=[caption])


class CaptionExcusedThroughGroup(Scene):
    """A caption across the right edge for two seconds; the arming excuses
    a group the scene never draws, which holds the caption from the middle
    of the scene to its end.
    """

    keep = True

    def construct(self):
        excused = VGroup()
        arm(exempt=[excused])
        caption = Text("Edge of reason", font_size=44).move_to(RIGHT * 6.4)
        self.add(caption)
        self.wait(2)
        excused.add(caption)
        self.play(caption.animate.move_to(UP * 2))
        if not self.keep:
            excused.remove(caption)
        self.wait(0.5)


class CaptionLeavesTheGroup(CaptionExcusedThroughGroup):
    """...and leaves it before the end: nothing excuses it there."""

    keep = False


class ExemptionGrowsBeforeADeparture(Scene):
    """A caption across the right edge for two seconds, then moved in; the
    list the arming was handed names it only after that, and then the
    script removes a settled word, which the departure guard checks.
    """

    #: Whether the word is removed (``"after"``: once the list has grown
    #: and half a second has passed; ``"before"``: before the list grows).
    removal: str | None = "after"

    def construct(self):
        excused = []
        arm(exempt=excused)
        caption = Text("Edge of reason", font_size=44).move_to(RIGHT * 6.4)
        aside = Text("aside", font_size=30).move_to(DOWN * 2 + LEFT * 3)
        self.add(caption, aside)
        self.wait(2)
        self.play(caption.animate.move_to(UP * 2))
        if self.removal == "before":
            self.remove(aside)
            self.wait(0.5)
        excused.append(caption)
        self.wait(0.5)
        if self.removal == "after":
            self.remove(aside)
        self.wait(0.5)


class ExemptionGrowsWithNoDeparture(ExemptionGrowsBeforeADeparture):
    """...and nothing departs after the list grows."""

    removal = None


class ExemptionGrowsAfterTheDeparture(ExemptionGrowsBeforeADeparture):
    """...and the word departs before the list grows."""

    removal = "before"


class DeclaredNoteFades(Scene):
    """A word across a line for a second and a half, faded out; the scene
    arms nothing and declares the two excused (the bridge's own call, which
    the reference checks have no counterpart of).
    """

    declared = True

    def construct(self):
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36).move_to(rail.get_center())
        if self.declared:
            declare(exempt=[(note, rail)])
        self.add(rail, note)
        self.wait(1.5)
        self.play(FadeOut(note))
        self.wait(0.5)


class UndeclaredNoteFades(DeclaredNoteFades):
    """...declaring nothing."""

    declared = False


class CountedArmedScene(Scene):
    """An armed scene that counts its executions: a word settled across a
    line, faded out, and a caption held across the right edge.
    """

    def construct(self):
        run_counter.executions += 1
        arm()
        rail = Line(LEFT * 4, RIGHT * 4)
        note = Text("note", font_size=36).move_to(rail.get_center())
        caption = Text("Edge of reason", font_size=44).move_to(RIGHT * 6.4 + UP * 2)
        self.add(rail, note, caption)
        self.wait(1.5)
        self.play(FadeOut(note))
        self.remove(caption)
        self.wait(0.5)
