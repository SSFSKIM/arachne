"""Scenes whose record keeps the states the lint's temporal rules read:
the scene as the script called each play, as each play ended, and as the
script removed from it.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    YELLOW,
    Dot,
    FadeOut,
    Line,
    Mobject,
    Scene,
    Square,
    Text,
    VGroup,
    ZoomedScene,
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


class RemovesThroughAGroup(Scene):
    """Armed: the script removes a group it never added, which then lets
    go of what it held.
    """

    def construct(self):
        self.timeline.arm_states()
        word = Text("word")
        group = VGroup(word)
        self.add(word)
        self.wait(0.5)
        self.remove(group)
        group.remove(word)
        self.wait(0.5)


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
        self.timeline.arm_states()
        rail = Line(LEFT, RIGHT)
        note = Text("note")
        self.add(rail, note)
        self.wait(0.5)
        self.play(FadeOut(VGroup(note, rail)))


class DisplayLeftBetweenPlays(ZoomedScene):
    """A zoomed display the camera stops drawing between two plays, the
    script then calling ``remove``: the record refuses drawing from then on,
    whether or not it keeps states.
    """

    armed = False

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3,
            zoomed_display_height=3,
            zoomed_display_width=3,
            **kwargs,
        )

    def construct(self):
        if self.armed:
            self.timeline.arm_states()
        dot = Dot(radius=0.2, color=YELLOW)
        self.add(dot)
        self.zoomed_camera.frame.move_to(dot)
        self.activate_zooming()
        self.play(dot.animate.shift(UP * 0.1), run_time=0.4)
        self.renderer.camera.image_mobjects_from_cameras.remove(self.zoomed_display)
        self.remove(Mobject())
        self.play(dot.animate.shift(DOWN * 0.5), run_time=0.4)


class DisplayLeftBetweenPlaysArmed(DisplayLeftBetweenPlays):
    """The same, keeping states."""

    armed = True
