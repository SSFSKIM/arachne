"""Scenes of media objects on a clock, for the clock tests. A line a test
names carries a ``# @tag``.

The five programs each start at 0.37 s, an instant off the 60 Hz tick, and
run a clip, a sprite and a still beside it for 2 s: the clip fills the
frame, pixel for pixel, so that its stamp reads where the clip draws it, and
the sprite's program is the clip's at a third of its times, the sprite
being a third as long.
"""

from __future__ import annotations

from fractions import Fraction

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    ImageMobject,
    Rectangle,
    Scene,
    Sprite,
    Still,
    Transform,
    VideoMobject,
    config,
    linear,
)
from arachne.assets.reader import FIXTURES
from arachne.compose import LayerSpec, set_layer

#: The PNG file holding the test clip's first frame.
FIRST_FRAME = FIXTURES / "numbered-clip" / "frames" / "0000.png"

#: Where every program starts: an instant between two ticks.
START = 0.37
#: The sprite's times and rates, against the clip's.
THIRD = Fraction(1, 3)


def full(clip: VideoMobject) -> VideoMobject:
    """``clip`` over the whole frame, drawn by nearest resampling."""
    clip.set_resampling_algorithm(0)
    clip.stretch_to_fit_width(config.frame_width)
    clip.stretch_to_fit_height(config.frame_height)
    return clip


def beside(scene: Scene) -> tuple[VideoMobject, Sprite, Still]:
    """A clip filling the frame, and a sprite and a still small in its lower
    right corner, away from the clip's stamp, all added to ``scene``.
    """
    clip = full(VideoMobject("fixture:numbered-clip"))
    sprite = Sprite("fixture:stepper").scale(0.5).to_corner(DOWN + RIGHT, buff=0.2)
    still = Still("fixture:swatch").scale(0.3).next_to(sprite, LEFT)
    scene.add(clip, sprite, still)
    return clip, sprite, still


class ClockPlay(Scene):
    """Run."""

    def construct(self):
        clip, sprite, _ = beside(self)
        self.wait(START)
        clip.run()
        sprite.run(rate=THIRD)
        self.wait(2)


class ClockLoop(Scene):
    """Loop, from a second in, so that it wraps a second later."""

    def construct(self):
        clip, sprite, _ = beside(self)
        self.wait(START)
        clip.loop(at=1)
        sprite.loop(rate=THIRD, at=THIRD)
        self.wait(2)


class ClockHold(Scene):
    """Run for half a second, then hold where it is."""

    def construct(self):
        clip, sprite, _ = beside(self)
        self.wait(START)
        clip.run()
        sprite.run(rate=THIRD)
        self.wait(0.5)
        clip.hold()
        sprite.hold()
        self.wait(1.5)


class ClockScrub(Scene):
    """Stand at 1.75 s, and scrub back to 0.25 s over 1.5 s."""

    def construct(self):
        clip, sprite, _ = beside(self)
        self.wait(START)
        clip.hold(at=Fraction(7, 4))
        sprite.hold(at=Fraction(7, 12))
        clip.scrub(to=Fraction(1, 4), over=1.5)
        sprite.scrub(to=Fraction(1, 12), over=1.5)
        self.wait(2)


class ClockHalf(Scene):
    """Run at half rate."""

    def construct(self):
        clip, sprite, _ = beside(self)
        self.wait(START)
        clip.run(rate=Fraction(1, 2))
        sprite.run(rate=THIRD / 2)
        self.wait(2)


class ClipRuns(Scene):
    """The clip run from the scene's start for its duration."""

    def construct(self):
        clip = full(VideoMobject("fixture:numbered-clip"))  # @clip-runs
        self.add(clip)
        clip.run()
        self.wait(2)


class ClipRunsThenChanges(Scene):
    """:class:`ClipRuns`' first second, then the clip looped backward."""

    def construct(self):
        clip = full(VideoMobject("fixture:numbered-clip"))
        self.add(clip)
        clip.run()
        self.wait(1)
        clip.loop(rate=-1)
        self.wait(1)


class LoopsBackward(Scene):
    """The clip looped at rate -1 from its first frame."""

    def construct(self):
        clip = full(VideoMobject("fixture:numbered-clip"))
        self.add(clip)
        clip.loop(rate=-1)
        self.wait(1)


class ClipFadedInAndMoved(Scene):
    """A running clip faded in, then shifted: its clock runs through both."""

    def construct(self):
        clip = full(VideoMobject("fixture:numbered-clip"))
        clip.run()
        self.play(FadeIn(clip, shift=DOWN * 0.5), run_time=0.5, rate_func=linear)
        self.play(clip.animate.shift(RIGHT * 0.3), run_time=0.5, rate_func=linear)


class ClipInWindow(Scene):
    """A running clip the scene holds, shown through a window on the
    frame's left half.
    """

    def construct(self):
        clip = full(VideoMobject("fixture:numbered-clip"))
        window = Rectangle(width=config.frame_width / 2, height=config.frame_height)
        window.shift(LEFT * config.frame_width / 4)
        set_layer(window, LayerSpec(window=(clip,)))
        self.add(clip, window)
        clip.run()
        self.wait(1)


class ClipOnlyInWindow(Scene):
    """A running clip a window lists and the scene does not hold."""

    def construct(self):
        clip = full(VideoMobject("fixture:numbered-clip"))
        window = Rectangle(width=config.frame_width / 2, height=config.frame_height)
        set_layer(window, LayerSpec(window=(clip,)))
        self.add(window)
        clip.run()
        self.wait(1)  # @only-in-window


class StandingClipOnlyInWindow(Scene):
    """A clip that stands still, listed by a window only."""

    def construct(self):
        clip = full(VideoMobject("fixture:numbered-clip"))
        window = Rectangle(width=config.frame_width / 2, height=config.frame_height)
        set_layer(window, LayerSpec(window=(clip,)))
        self.add(window)
        clip.hold(frame=5)
        self.wait(0.5)


class RunsOneTickLong(Scene):
    """A clip run, and a wait one tick longer than the clip."""

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        clip.run()
        self.wait(Fraction(2) + Fraction(1, 60))  # @one-tick-long


class FadesPastTheEnd(Scene):
    """A clip run for its duration, then faded out over half a second."""

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        clip.run()
        self.wait(2)
        self.play(FadeOut(clip), run_time=0.5)  # @fade-past


class FadeEndsWithTheClip(Scene):
    """A fade that ends when the clip ends."""

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        clip.run()
        self.wait(1.5)
        self.play(FadeOut(clip), run_time=0.5)


class HeldBeforeTheFade(Scene):
    """A clip held on its last frame before it fades."""

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        clip.run()
        self.wait(2)
        clip.hold()
        self.play(FadeOut(clip), run_time=0.5)


class RunThenHold(Scene):
    """A clip run to stand on the frame it reaches, faded after its end."""

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        clip.run(then="hold")
        self.wait(2)
        self.play(FadeOut(clip), run_time=0.5)


class RanOutOffstage(Scene):
    """A clip whose clock runs out while it is out of the scene, then
    added back for a wait.
    """

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        clip.run()
        self.wait(1)
        self.remove(clip)
        self.wait(1.5)
        self.add(clip)
        self.wait(0.5)  # @back-on


class RanOutAddedLast(Scene):
    """A clip whose clock runs out while it is out of the scene, added back
    after the last play.
    """

    def construct(self):
        clip = full(VideoMobject("fixture:numbered-clip"))  # @added-last
        self.add(clip)
        clip.run()
        self.wait(1)
        self.remove(clip)
        self.wait(1.5)
        self.add(clip)


class StandingMedia(Scene):
    """Media that stand still: a clip held on a frame, a sprite held on a
    drawing, a still; moved and faded.
    """

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip").hold(frame=3).shift(UP)
        sprite = Sprite("fixture:stepper").hold(frame=2).shift(DOWN)
        still = Still("fixture:swatch").shift(RIGHT * 3)
        self.add(sprite, still)
        self.play(FadeIn(clip), run_time=0.5)
        self.play(clip.animate.shift(LEFT), still.animate.scale(0.5), run_time=0.5)
        self.wait(0.5)


class StandingImages(Scene):
    """:class:`StandingMedia` with images opened by path in its media's
    places.
    """

    def construct(self):
        clip = ImageMobject(str(FIRST_FRAME)).shift(UP)
        sprite = ImageMobject(str(FIRST_FRAME)).shift(DOWN)
        still = ImageMobject(str(FIRST_FRAME)).shift(RIGHT * 3)
        self.add(sprite, still)
        self.play(FadeIn(clip), run_time=0.5)
        self.play(clip.animate.shift(LEFT), still.animate.scale(0.5), run_time=0.5)
        self.wait(0.5)


class ReadsItsTime(Scene):
    """A clip whose media time the scene's code reads as it runs."""

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        self.read = [clip.media_time]
        clip.run(rate=Fraction(1, 2))
        self.wait(0.5)
        self.read.append(clip.media_time)
        clip.loop(at=Fraction(3, 2), rate=2)
        self.wait(0.5)
        self.read.append(clip.media_time)


class CopiesOfARunningClip(Scene):
    """A running clip copied half a second in, the copy added beside it; and
    the copy a transform of it makes as it begins, added too.
    """

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip").shift(UP * 2)
        self.add(clip)
        clip.run()
        self.wait(0.5)
        twin = clip.copy().shift(DOWN * 2)
        self.add(twin)
        move = Transform(clip, clip.copy().shift(LEFT))
        self.play(move, run_time=0.5)
        started = move.starting_mobject.shift(DOWN * 4)
        self.add(started)
        self.wait(0.5)


class RunsOnPastAnOffTickEnd(Scene):
    """A clip run for its duration and half a tick more: the play ends
    between two ticks.
    """

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        self.add(clip)
        clip.run()
        self.wait(Fraction(2) + Fraction(1, 120))
