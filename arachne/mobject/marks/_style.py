"""Style for the marks made of several parts, and opacities a fade keeps.

A mark that is a group draws nothing itself, so a style keyword handed to
it is handed on to the parts that draw it (:func:`take_style`,
:func:`apply_style`); each mark names those parts, leaving its texts, its
ports and any part with a style of its own as they are.
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

from arachne.mobject.mobject import Mobject
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
#: stop of each.
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
    """A vector member's opacities, every colour stop of each paint; none
    for any other.
    """
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
    """Give a vector member, itself alone, ``opacities``, stop by stop."""
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
