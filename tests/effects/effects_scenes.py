"""Scenes the effects' tests open: steps and planes whose edges an effect
spreads or cuts at a known place, and objects whose effects change.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    ChangeBlur,
    Circle,
    Dot,
    RackFocus,
    Rectangle,
    Scene,
    Square,
    VGroup,
    clear_effects,
    linear,
    set_blur,
    set_depth,
    set_focus,
    set_mask,
)
from arachne.compose.layer import LayerSpec, set_layer
from arachne.quantities import Literal, attach

# Registers the prefix "probe.", which a literal of another owner is named in.
from arachne.utils.testing import declared_probe  # noqa: F401

#: Where a mask's edge stands, in scene units right of the frame's centre.
MASK_EDGE = 0.37


def step(height: float = 20) -> Rectangle:
    """A white plane left of x = 0, ``height`` tall: a step at x = 0."""
    return Rectangle(
        width=20, height=height, fill_color=WHITE, fill_opacity=1, stroke_width=0
    ).shift(10 * LEFT)


class BlurredStep(Scene):
    """The step blurred by ``sigma``, or, with ``depth`` set, placed at that
    depth under a focus at 0 with ``aperture``.
    """

    sigma = 0.15
    depth: float | None = None
    aperture = 0.15

    def construct(self):
        plane = step()
        if self.depth is None:
            set_blur(plane, self.sigma)
        else:
            set_focus(self, 0, self.aperture)
            set_depth(plane, self.depth)
        self.add(plane)
        self.wait(0.2)


class BlurredStepMedium(BlurredStep):
    sigma = 0.3


class BlurredStepWide(BlurredStep):
    sigma = 0.6


class FocusedStep(BlurredStep):
    """The step at depth 2 under a focus at 0 with aperture 0.15: 0.3."""

    depth = 2.0


class SharpStep(Scene):
    """The step with no effect, drawn by Cairo."""

    def construct(self):
        self.add(step())
        self.wait(0.2)


class MaskedPlane(Scene):
    """A white plane over the frame cut out by a plane whose right edge is
    at :data:`MASK_EDGE`, feathered by ``feather``, inverted with ``invert``.
    """

    feather = 0.0
    invert = False

    def construct(self):
        plane = Rectangle(
            width=20, height=20, fill_color=WHITE, fill_opacity=1, stroke_width=0
        )
        mask = Rectangle(width=20, height=20, fill_opacity=1, stroke_width=0)
        mask.shift((MASK_EDGE - 10) * RIGHT)
        set_mask(plane, mask, feather=self.feather, invert=self.invert)
        self.add(plane)
        self.wait(0.2)


class MaskedPlaneInverted(MaskedPlane):
    invert = True


class MaskedPlaneSoft(MaskedPlane):
    feather = 0.25


class MaskedPlaneSoftInverted(MaskedPlaneSoft):
    invert = True


class MaskedPlaneSofter(MaskedPlane):
    feather = 0.6


class MaskedPlaneSofterInverted(MaskedPlaneSofter):
    invert = True


class ChangingBlur(Scene):
    """The step's blur moved from 0.2 to 0.6 over one second, linearly, then
    to 0 over half a second; a quarter second's wait after.
    """

    def construct(self):
        plane = step()
        set_blur(plane, 0.2)
        self.add(plane)
        self.play(ChangeBlur(plane, 0.6, run_time=1, rate_func=linear))
        self.play(ChangeBlur(plane, 0, run_time=0.5))
        self.wait(0.25)


class RackingFocus(Scene):
    """Two steps, the upper at depth 0 and the lower at depth 4, under a
    focus at 0 with aperture 0.1 that is moved to 4 over one second,
    linearly; then its aperture to 0.05 over half a second; a quarter
    second's wait after.
    """

    def construct(self):
        near = step(height=10).shift(5 * UP)
        far = step(height=10).shift(5 * DOWN)
        set_depth(near, 0)
        set_depth(far, 4)
        set_focus(self, 0, 0.1)
        self.add(near, far)
        self.play(RackFocus(self, 4, run_time=1, rate_func=linear))
        self.play(RackFocus(self, 4, aperture=0.05, run_time=0.5, rate_func=linear))
        self.wait(0.25)


class RackWithoutFocus(Scene):
    """A focus racked in a scene that has none."""

    def construct(self):
        square = Square()
        set_depth(square, 1)
        self.add(square)
        self.play(RackFocus(self, 2))  # @rack


class NumbersOverTime(Scene):
    """A square holding a literal of its own, given a blur twice, a mask
    and a depth for one wait, then cleared of them for another.
    """

    def construct(self):
        square = Square(fill_opacity=1)
        attach(square, Literal("probe.size", 2.0, "1"))
        set_blur(square, 0.1)
        set_blur(square, 0.3)
        set_mask(square, Circle(radius=0.8, fill_opacity=1), feather=0.2)
        set_depth(square, 1.5)
        self.add(square)
        self.wait(0.5)
        clear_effects(square)
        self.wait(0.5)


class NestedBlur(Scene):
    """A blurred circle inside a blurred group, which only a window may
    hold.
    """

    def construct(self):
        inner = Circle()
        set_blur(inner, 0.1)
        group = VGroup(Square(), inner)
        set_blur(group, 0.2)
        self.add(group)
        self.wait(0.2)


class Removal(Scene):
    """A white square at the left with ``effect`` (``"blur"``, ``"mask"`` or
    ``"depth"``; with none it is a layer of its own with an empty spec), and
    a blurred dot far right, under a focus at 0: every frame is stacked the
    same way, the square's effect the one difference.
    """

    effect: str | None = None

    def construct(self):
        set_focus(self, 0, 0.1)
        subject = Square(side_length=2, fill_color=WHITE, fill_opacity=1)
        subject.shift(3 * LEFT + 0.5 * UP)
        other = Dot(radius=0.5).shift(5 * RIGHT + DOWN)
        set_blur(other, 0.1)
        if self.effect == "blur":
            set_blur(subject, 0.2)
        elif self.effect == "mask":
            cut = Circle(radius=1, fill_opacity=1).move_to(
                subject.get_corner(UP + LEFT)
            )
            set_mask(subject, cut, feather=0.3)
        elif self.effect == "depth":
            set_depth(subject, 3)
        else:
            set_layer(subject, LayerSpec())
        self.add(subject, other)
        self.wait(0.25)


class RemovalBlur(Removal):
    effect = "blur"


class RemovalMask(Removal):
    effect = "mask"


class RemovalDepth(Removal):
    effect = "depth"
