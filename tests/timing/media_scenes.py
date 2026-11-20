"""Scenes of media objects beside stepped timing: a clip run after and
before a scene declares timing; a sprite on a clock of its own, declared
with stepped timing at half its rate and at its rate, and moved while
declared; a clip declared; and a rig of stills posed while declared.
"""

from __future__ import annotations

from arachne import (
    RIGHT,
    UP,
    Pose,
    Rig,
    Scene,
    Sprite,
    Square,
    VideoMobject,
    set_timing,
)
from arachne.utils.rate_functions import linear

#: The sprite's own clock runs at its 12 drawings a second; the drawing
#: rate it is declared at in the unequal-rates scenes.
HALF = 6
#: How far the moved sprite goes.
MOVE = 1.5 * RIGHT + 0.5 * UP


class ClipRunAfterTiming(Scene):
    """A clip run after the scene declared stepped timing on something
    else: its driver is made in a scene that already has a clock.
    """

    def construct(self):
        box = Square()
        clip = VideoMobject("fixture:numbered-clip").scale(0.5)
        self.add(box, clip)
        set_timing(box, 12)
        self.wait(0.25)
        clip.run()
        self.wait(1)


class ClipRunBeforeTiming(Scene):
    """A clip run before the scene declares stepped timing: the clock, made
    later, takes its driver over.
    """

    def construct(self):
        box = Square()
        clip = VideoMobject("fixture:numbered-clip").scale(0.5)
        self.add(box, clip)
        clip.run()
        self.wait(0.25)
        set_timing(box, 12)
        self.wait(1)


class ClipEndingOffTickBesideTiming(Scene):
    """A clip run beside declared timing, in a scene whose one wait ends
    between two ticks, after the clip's frame 9 begins and before the next
    tick.
    """

    def construct(self):
        box = Square()
        clip = VideoMobject("fixture:numbered-clip").scale(0.5)
        self.add(box, clip)
        set_timing(box, 12)
        clip.run()
        self.wait(0.375)


class SpriteAtHalfItsRate(Scene):
    """A sprite looping at its 12 drawings a second, declared to be drawn 6
    times a second.
    """

    def construct(self):
        sprite = Sprite("fixture:stepper")
        self.add(sprite)
        set_timing(sprite, HALF)
        sprite.loop()
        self.wait(1.5)


class SpriteAtItsRate(Scene):
    """The same sprite declared at its own rate."""

    def construct(self):
        sprite = Sprite("fixture:stepper")
        self.add(sprite)
        set_timing(sprite, 12)
        sprite.loop()
        self.wait(1.5)


class MovedSteppedSprite(Scene):
    """A declared sprite on its clock, moved by an animation that ends off
    its grid, then left to wait.
    """

    def construct(self):
        sprite = Sprite("fixture:stepper")
        self.add(sprite)
        set_timing(sprite, HALF)
        sprite.loop()
        self.play(sprite.animate.shift(MOVE), rate_func=linear, run_time=0.9)
        self.wait(0.6)


class DeclaredClip(Scene):
    """A clip declared with stepped timing, then run."""

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        set_timing(clip, HALF)
        clip.run()
        self.wait(1)


class PosedSteppedFlap(Scene):
    """A rig of stills declared with stepped timing, posed by a play."""

    def construct(self):
        rig = Rig("fixture:hinged-flap")
        self.add(rig)
        set_timing(rig, HALF)
        self.play(Pose(rig, lift=60), rate_func=linear, run_time=1)
        self.wait(0.25)
