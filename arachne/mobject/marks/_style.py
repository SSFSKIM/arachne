"""Style for the marks made of several parts, and opacities a fade keeps.

A mark that is a group draws nothing itself, so a style keyword handed to
it is handed on to the parts that draw it (:func:`take_style`,
:func:`apply_style`); each mark names those parts, leaving its texts, its
ports and any part with a style of its own as they are.

A fade reads a member's opacities and writes them scaled: a vector
member's every colour stop of each paint, and a picture's
(:class:`~arachne.mobject.types.image_mobject.ImageMobject`) pixels as it
shows them (:class:`PictureOpacity`), its alpha scaled from theirs, so a
picture written back at the share read shows those very pixels again. Any
other member is left as it is.
"""

from __future__ import annotations

__all__ = [
    "STYLE_KEYWORDS",
    "Opacities",
    "PictureOpacity",
    "apply_style",
    "opacities_of",
    "scaled",
    "set_opacities",
    "take_style",
]

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from arachne.mobject.mobject import Mobject
from arachne.mobject.types.image_mobject import ImageMobject
from arachne.mobject.types.vectorized_mobject import VMobject
from arachne.mobject.values import FrozenPixels, adopt

#: The style keywords a vector object takes, which a group hands on.
STYLE_KEYWORDS = (
    "fill_color",
    "fill_opacity",
    "stroke_color",
    "stroke_width",
    "stroke_opacity",
    "background_stroke_color",
    "background_stroke_width",
    "background_stroke_opacity",
    "sheen_factor",
    "sheen_direction",
)


@dataclass(frozen=True, eq=False)
class PictureOpacity:
    """How a picture shows: its ``pixels`` as read, its own ``opacity``
    then, and the ``share`` of both it is to show at. Scaled as an opacity
    is (``* share``).
    """

    pixels: Any
    opacity: float
    share: float = 1.0

    def __mul__(self, share: float) -> PictureOpacity:
        return replace(self, share=self.share * share)

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PictureOpacity)
            and self.share == other.share
            and np.array_equal(self.pixels, other.pixels)
        )

    __hash__ = None  # type: ignore[assignment]


#: A member's fill, stroke and background stroke opacity, one per colour
#: stop of each; a picture's pixels as it shows them as its fill, no stroke.
Opacities = tuple[
    tuple[float | PictureOpacity, ...], tuple[float, ...], tuple[float, ...]
]


def take_style(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The style keywords among ``kwargs``, taken out of it."""
    return {key: kwargs.pop(key) for key in STYLE_KEYWORDS if key in kwargs}


def apply_style(parts: Iterable[VMobject], style: dict[str, Any]) -> None:
    """Give each of ``parts``, itself alone, the keywords of ``style``."""
    if not style:
        return
    for part in parts:
        part.set_style(**style, family=False)


def opacities_of(member: Mobject) -> Opacities:
    """A vector member's opacities, every colour stop of each paint; a
    picture's pixels as it shows them; none for any other.
    """
    if isinstance(member, ImageMobject):
        shown = PictureOpacity(member.pixel_array, float(member.stroke_opacity))
        return ((shown,), (), ())
    if not isinstance(member, VMobject):
        return ((), (), ())
    return (
        tuple(float(o) for o in member.get_fill_opacities()),
        tuple(float(o) for o in member.get_stroke_opacities()),
        tuple(float(o) for o in member.get_stroke_opacities(background=True)),
    )


def scaled(opacities: Opacities, share: float) -> Opacities:
    """``opacities`` each times ``share``."""
    fill, stroke, background = opacities
    return (
        tuple(o * share for o in fill),
        tuple(o * share for o in stroke),
        tuple(o * share for o in background),
    )


def set_opacities(member: Mobject, opacities: Opacities) -> None:
    """Give a vector member, itself alone, ``opacities``, stop by stop; a
    picture the pixels it was read with, at their share.
    """
    if isinstance(member, ImageMobject):
        if opacities[0]:
            (shown,) = opacities[0]
            assert isinstance(shown, PictureOpacity)
            _show_picture(member, shown)
        return
    if not isinstance(member, VMobject):
        return
    # The setters spread a sequence of opacities over the colour stops, one
    # each, though they are typed for one opacity.
    fill, stroke, background = (cast(Any, stops) for stops in opacities)
    if fill:
        member.set_fill(opacity=fill, family=False)
    if stroke:
        member.set_stroke(opacity=stroke, family=False)
    if background:
        member.set_stroke(opacity=background, background=True, family=False)


def _show_picture(picture: ImageMobject, shown: PictureOpacity) -> None:
    """Show ``picture`` as ``shown`` says: its pixels as read, their alpha
    times its share (cast as :meth:`~.ImageMobject.set_opacity` casts it).
    The pixels it holds are kept when they already show so: at the whole
    share those read, and at any other share those last written at it.
    """
    opacity = shown.opacity * shown.share
    if shown.share == 1:
        picture.pixel_array = shown.pixels
    elif float(picture.stroke_opacity) != opacity:
        pixels = np.array(shown.pixels)
        pixels[:, :, 3] = shown.pixels[:, :, 3] * shown.share
        picture.pixel_array = adopt(pixels, FrozenPixels)
    picture.stroke_opacity = opacity
