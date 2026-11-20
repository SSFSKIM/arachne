"""Scenes of media objects at rest, for the media tests. A line a test names
carries a ``# @tag``.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    ImageMobject,
    Scene,
    Sprite,
    Square,
    Still,
    VideoMobject,
    config,
)
from arachne.assets.reader import FIXTURES

#: The four corners (CE's image order: upper left, upper right, lower left,
#: lower right) both scenes of the routing fixture set their picture to.
CORNERS = np.array([[-3.0, 1.5, 0], [2.0, 1.5, 0], [-3.0, -1.5, 0], [2.0, -1.5, 0]])
#: The resampling both of them draw with.
RESAMPLING = 2
#: The PNG file holding the test clip's first frame.
FIRST_FRAME = FIXTURES / "numbered-clip" / "frames" / "0000.png"


def placed(picture, corners=CORNERS):
    """``picture`` set to ``corners`` and :data:`RESAMPLING`."""
    picture.points = corners
    picture.set_resampling_algorithm(RESAMPLING)
    return picture


class FixtureImage(Scene):
    """The routing fixture's first scene: the clip's first frame, opened by
    path with CE's constructor.
    """

    def construct(self):
        self.add(Square().shift(3 * RIGHT))
        self.add(placed(ImageMobject(str(FIRST_FRAME))))
        self.wait(0.5)


class FixtureClip(Scene):
    """The routing fixture's second scene: the clip, on its first frame."""

    def construct(self):
        self.add(Square().shift(3 * RIGHT))
        self.add(placed(VideoMobject("fixture:numbered-clip")))  # @fixture-clip
        self.wait(0.5)


class FixtureStill(Scene):
    """The fixture's third twin: a still where the first has an image."""

    def construct(self):
        self.add(Square().shift(3 * RIGHT))
        self.add(placed(Still("fixture:swatch")))
        self.wait(0.5)


class StillMoved(Scene):
    """A still faded in, moved, scaled and faded: a CPU frame."""

    def construct(self):
        still = Still("fixture:swatch").shift(2 * LEFT)
        self.play(FadeIn(still), run_time=0.5)
        self.play(still.animate.shift(2 * RIGHT).scale(0.5), run_time=0.5)
        still.set_opacity(0.5)
        self.wait(0.5)


class ClipMoved(Scene):
    """A clip faded in, moved and scaled, beside a sprite."""

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip").shift(UP)  # @clip-moved
        sprite = Sprite("fixture:stepper").shift(2 * DOWN)
        self.add(sprite)
        self.play(FadeIn(clip, shift=RIGHT), run_time=0.5)  # @clip-fade
        self.play(clip.animate.shift(LEFT).scale(1.5), run_time=0.5)
        self.wait(0.5)


class SpriteAlone(Scene):
    """A sprite at rest."""

    def construct(self):
        self.add(Sprite("fixture:stepper"))
        self.wait(0.5)


class ClipOverBackdrop(Scene):
    """The clip over the frame exactly, pixel for pixel (the session's frame
    is the clip's size), drawn with nearest resampling over the camera's
    background.
    """

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip", resampling_algorithm=0)
        clip.stretch_to_fit_width(config.frame_width)
        clip.stretch_to_fit_height(config.frame_height)
        self.add(clip)
        self.wait(0.5)


class StillOverBackdrop(Scene):
    """The clip's first frame opened by path, over the frame as the clip is
    in :class:`ClipOverBackdrop`.
    """

    def construct(self):
        image = ImageMobject(str(FIRST_FRAME), resampling_algorithm=0)
        image.stretch_to_fit_width(config.frame_width)
        image.stretch_to_fit_height(config.frame_height)
        self.add(image)
        self.wait(0.5)


class TwoOfOneAsset(Scene):
    """Two clips of one asset."""

    def construct(self):
        self.add(VideoMobject("fixture:numbered-clip").shift(2 * LEFT))
        self.add(VideoMobject("fixture:numbered-clip").shift(2 * RIGHT))
        self.wait(0.5)
