"""Style for the marks made of several parts, and opacities a fade keeps.

A mark that is a group draws nothing itself, so a style keyword handed to
it is handed on to the parts that draw it (:func:`take_style`,
:func:`apply_style`); each mark names those parts, leaving its texts, its
ports and any part with a style of its own as they are.

A fade reads a member's opacities and writes them scaled: a vector
member's every colour stop of each paint, and a picture's
(:class:`~arachne.mobject.types.image_mobject.ImageMobject`) own opacity,
the one :meth:`~arachne.mobject.types.image_mobject.ImageMobject.set_opacity`
sets, so a picture written back at the opacity read shows its pixels as
before. A picture whose alpha is not that opacity's (one recoloured with an
alpha, or caught mid-transform) has none a fade could give back, and is left
as it is, as is any other member.
"""

from __future__ import annotations

__all__ = [
    "STYLE_KEYWORDS",
    "Opacities",
    "apply_style",
    "opacities_of",
    "scaled",
    "set_opacities",
    "take_style",
]

from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from arachne.mobject.mobject import Mobject
from arachne.mobject.types.image_mobject import ImageMobject
from arachne.mobject.types.vectorized_mobject import VMobject

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

#: A member's fill, stroke and background stroke opacity, one per colour
#: stop of each; a picture's one opacity as its fill, no stroke.
Opacities = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]


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
    picture's own opacity, when its alpha is that opacity's; none for any
    other.
    """
    if isinstance(member, ImageMobject):
        opacity = _picture_opacity(member)
        return ((), (), ()) if opacity is None else ((opacity,), (), ())
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
    picture its one opacity.
    """
    if isinstance(member, ImageMobject):
        if opacities[0]:
            member.set_opacity(opacities[0][0])
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


def _picture_opacity(picture: ImageMobject) -> float | None:
    """``picture``'s own opacity, when its pixels' alpha is the alpha
    :meth:`~.ImageMobject.set_opacity` gives at it; ``None`` otherwise.
    """
    opacity = float(picture.stroke_opacity)
    pixels = np.asarray(picture.pixel_array)
    alpha = np.empty_like(pixels[:, :, 3])
    # Cast as set_opacity's assignment casts it.
    alpha[...] = picture.orig_alpha_pixel_array * opacity
    return opacity if np.array_equal(alpha, pixels[:, :, 3]) else None
