"""Scenes whose objects hold GPU-only capabilities at chosen points of their
scripts, for the compose tests.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    BLUE,
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    WHITE,
    Circle,
    Create,
    Dot,
    FadeIn,
    Group,
    ImageMobject,
    Rectangle,
    Scene,
    Square,
    ValueTracker,
    VGroup,
)
from arachne.compose.layer import LayerSpec, set_focus, set_layer
from arachne.routing import declare
from tests.diagram.zoom_out_scenes import ZoomPastAPicture


class LastPlay(Scene):
    """Three plays; with ``switch`` set, the second is coloured otherwise and
    the circle shown in the last holds a blur. The first play is the same
    either way.
    """

    switch = False

    def construct(self):
        self.play(Create(Square()))
        dot = Dot(color=RED if self.switch else BLUE)
        self.play(FadeIn(dot))
        circle = Circle()
        if self.switch:
            declare(circle, "effect.blur")
        self.play(FadeIn(circle))


class AfterLastPlay(Scene):
    """A capability set on a shown object once its last play has ended."""

    def construct(self):
        square = Square()
        self.play(Create(square))
        self.play(square.animate.shift([1, 0, 0]))
        declare(square, "effect.soft_mask")


class NoPlay(Scene):
    """A declared object added, with no play at all."""

    def construct(self):
        square = Square()
        declare(square, "layer.depth")
        self.add(square)


class HashDot(Dot):
    """A dot whose class defines ``__hash__``: drawing it runs the scene's
    code, so the record draws nothing and its frames are drawn as it runs.
    """

    def __hash__(self):
        return object.__hash__(self)


class Live(Scene):
    """A scene drawn as it runs, whose dot holds a capability."""

    def construct(self):
        dot = HashDot()
        declare(dot, "effect.blur")
        self.play(FadeIn(dot))


class DeclaredUnseen(Scene):
    """A capability declared on a square never shown, and a few plays of a
    dot: nothing drawn holds a capability.
    """

    def construct(self):
        declare(Square(), "effect.blur")
        dot = Dot()
        self.add(dot)
        for _ in range(4):
            self.play(dot.animate.shift(RIGHT), run_time=0.5)


class Plain(Scene):
    """Nothing declared."""

    def construct(self):
        self.play(Create(Square()))
        self.play(FadeIn(Circle()))


# Scenes whose objects are layers of their own (the layer seam).


class EffectLast(Scene):
    """A square, then a circle blurred in the last play only; with
    ``effect`` unset the circle is not a layer, and the scene uses nothing
    new.
    """

    effect = True

    def construct(self):
        self.play(Create(Square()))  # @effect-last-first
        circle = Circle()
        if self.effect:
            set_layer(circle, LayerSpec(blur=0.2))
        self.play(FadeIn(circle))  # @effect-last-blur


class EffectLastRemoved(EffectLast):
    """:class:`EffectLast` with its effect removed."""

    effect = False


class BlurredEdge(Scene):
    """A white half-plane blurred by ``sigma`` scene units: its edge, at
    x = 0, is a step blurred to a declared width.
    """

    sigma = 0.25

    def construct(self):
        plane = Rectangle(
            width=20, height=20, fill_color=WHITE, fill_opacity=1, stroke_width=0
        ).shift(10 * LEFT)
        set_layer(plane, LayerSpec(blur=self.sigma))
        self.add(plane)
        self.wait(0.2)


#: A translucent colour, straight RGBA.
TINT = (200, 120, 40, 128)


def tinted_still(width: int, height: int) -> ImageMobject:
    """A still of one translucent colour covering the frame exactly, drawn
    with nearest resampling.
    """
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    pixels[:] = TINT
    still = ImageMobject(pixels, scale_to_resolution=height)
    still.set_resampling_algorithm(0)
    return still


class TintedStill(Scene):
    """The translucent still over the frame; with ``hidden_layer`` set, a
    blurred layer painting nothing puts the scene on the gpu route.
    """

    hidden_layer = True

    def construct(self):
        from arachne import config

        self.add(tinted_still(config.pixel_width, config.pixel_height))
        if self.hidden_layer:
            unseen = Dot(fill_opacity=0, stroke_width=0)
            set_layer(unseen, LayerSpec(blur=0.1))
            self.add(unseen)
        self.wait(0.2)


class TintedStillAlone(TintedStill):
    """:class:`TintedStill` with no layer: drawn by the camera alone."""

    hidden_layer = False


class MovingMask(Scene):
    """A white square cut out by a circle never added to the scene, which
    moves between plays; and a frame showing a dot, never added, which moves
    likewise.
    """

    def construct(self):
        square = Square(side_length=6, fill_color=WHITE, fill_opacity=1)
        mask = Circle(radius=1, fill_opacity=1).shift(2 * LEFT)
        set_layer(square, LayerSpec(mask=mask, mask_feather=0.1))
        frame = Rectangle(width=3, height=2, stroke_width=0).shift(3 * UP + 4 * RIGHT)
        dot = Dot(radius=0.8, color=RED).shift(2 * LEFT)
        set_layer(frame, LayerSpec(window=(dot,)))
        self.add(square, frame)
        self.wait(0.5)
        mask.shift(4 * RIGHT)
        dot.shift(4 * RIGHT)
        self.wait(0.5)


class LayerComesAndGoes(Scene):
    """A square, plain, then blurred, then blurred more, then plain again,
    each for one wait.
    """

    def construct(self):
        square = Square(side_length=3, fill_color=WHITE, fill_opacity=1)
        self.add(square)
        self.wait(0.5)
        set_layer(square, LayerSpec(blur=0.1))
        self.wait(0.5)
        set_layer(square, LayerSpec(blur=0.3))
        self.wait(0.5)
        set_layer(square, None)
        self.wait(0.5)


class Nested(Scene):
    """A layer within another's family, which only a window may hold."""

    def construct(self):
        inner = Circle()
        set_layer(inner, LayerSpec(blur=0.1))
        group = VGroup(Square(), inner)
        set_layer(group, LayerSpec(blur=0.2))
        self.add(group)
        self.wait(0.2)


class LiveLayer(Scene):
    """A scene drawn as it runs, holding a blurred layer."""

    def construct(self):
        dot = HashDot()
        set_layer(dot, LayerSpec(blur=0.1))
        self.play(FadeIn(dot))


class LiveLayerFound(Scene):
    """A scene drawn as it runs, holding a blurred layer, whose code also
    found what refuses drawing it.
    """

    def construct(self):
        from arachne.lint.raised import found

        found("annotation.anchor_unknown", "an anchor no rig has")  # @found
        dot = HashDot()
        set_layer(dot, LayerSpec(blur=0.1))
        self.play(FadeIn(dot))


SHIFT = 0.0


class ReboundLayer(Scene):
    """An updater reading a module global the scene rebinds between plays,
    so an instant of the first play is answered by a forward run; the
    circle shown in the last play is blurred.
    """

    def construct(self):
        global SHIFT
        SHIFT = 0.0
        tracker = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + SHIFT))
        self.add(dot)
        self.play(tracker.animate.set_value(2), run_time=1)
        SHIFT = 1.0
        self.play(tracker.animate.set_value(0), run_time=1)
        circle = Circle().shift(DOWN)
        set_layer(circle, LayerSpec(blur=0.2))
        self.play(FadeIn(circle))


class MixedMask(Scene):
    """A white square cut out by a mask of two kinds: a circle at the left
    and an opaque image at the right.
    """

    def construct(self):
        square = Square(side_length=7, fill_color=WHITE, fill_opacity=1)
        pixels = np.full((8, 8, 4), 255, dtype=np.uint8)
        still = ImageMobject(pixels, scale_to_resolution=32).shift(2 * RIGHT)
        still.set_resampling_algorithm(0)
        mask = Group(Circle(radius=1, fill_opacity=1).shift(2 * LEFT), still)
        set_layer(square, LayerSpec(mask=mask))
        self.add(square)
        self.wait(0.2)


class FocusInWindow(Scene):
    """A frame showing a white square at depth 0, through a window, under a
    focus that moves from it to depth 3 between two waits.
    """

    def construct(self):
        inner = Square(side_length=6, fill_color=WHITE, fill_opacity=1)
        set_layer(inner, LayerSpec(depth=0))
        frame = Rectangle(width=8, height=4.5, stroke_width=0)
        set_layer(frame, LayerSpec(window=(inner,)))
        self.add(frame)
        set_focus(self.camera, 0, 0.5)
        self.wait(0.5)
        set_focus(self.camera, 3, 0.5)
        self.wait(0.5)


class MaskShownToo(Scene):
    """:class:`MovingMask`'s square and mask, the mask also added to the
    scene: it is still painted only through the square.
    """

    shown = True

    def construct(self):
        square = Square(side_length=6, fill_color=WHITE, fill_opacity=1)
        mask = Circle(radius=1, color=RED, fill_opacity=1).shift(2 * LEFT)
        set_layer(square, LayerSpec(mask=mask))
        self.add(square)
        if self.shown:
            self.add(mask)
        self.wait(0.2)


class MaskNotShown(MaskShownToo):
    """:class:`MaskShownToo` without the mask added."""

    shown = False


class WindowMaskShownToo(Scene):
    """A frame whose window shows a white square cut out by a circle. With
    ``shown`` the circle is added to the scene too; with ``listed`` it is
    listed in the window beside the square. Either way it is painted only
    through the square.
    """

    shown = True
    listed = False

    def construct(self):
        inner = Square(side_length=6, fill_color=WHITE, fill_opacity=1)
        mask = Circle(radius=1, color=RED, fill_opacity=1).shift(2 * LEFT)
        set_layer(inner, LayerSpec(mask=mask))
        frame = Rectangle(width=8, height=4.5, stroke_width=0)
        window = (inner, mask) if self.listed else (inner,)
        set_layer(frame, LayerSpec(window=window))
        self.add(frame)
        if self.shown:
            self.add(mask)
        self.wait(0.2)


class WindowMaskListed(WindowMaskShownToo):
    """The circle listed in the window beside the square it cuts out."""

    shown = False
    listed = True


class WindowMaskNotShown(WindowMaskShownToo):
    """The circle neither added nor listed."""

    shown = False


class ZoomPastAPictureOnGpu(ZoomPastAPicture):
    """:class:`~tests.diagram.zoom_out_scenes.ZoomPastAPicture`, the zoom
    fading the picture it cuts and the return bringing it back, with a
    blurred layer painting nothing, which puts the scene on the gpu route.
    """

    def construct(self):
        unseen = Dot(fill_opacity=0, stroke_width=0)
        set_layer(unseen, LayerSpec(blur=0.1))
        self.add(unseen)
        super().construct()
