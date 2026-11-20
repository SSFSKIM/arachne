"""A window: a rectangle of the scene that shows a group of objects as a
frame of their own.

What a :class:`Window` shows is what a scene of its objects alone would
draw at the rectangle's size in pixels, effects and all, and nothing of it
reaches outside the rectangle. Its objects are drawn over the same view of
the scene as the frame holding it, scaled into the rectangle. A window
inside another's objects is a frame of its own again, at its own size.
"""

from __future__ import annotations

__all__ = ["Window"]

import math
from typing import Any

from arachne.compose import layer as seam
from arachne.compose.layer import LayerSpec
from arachne.mobject.geometry.polygram import Rectangle
from arachne.mobject.mobject import Mobject


class Window(Rectangle):
    """A rectangle ``width`` by ``height`` scene units, centred at the
    origin, showing ``content`` in place of a picture of its own.

    The window has no paint of its own: a border is any shape drawn over
    it. Its objects are listed in it and drawn only through it, never on
    their own, even when the scene holds one too, as it does when a play
    animates it (a play adds what it animates to the scene). Showing them
    makes the window a layer of its own, so a scene that shows one is drawn
    on a GPU (:mod:`arachne.compose`). Effects set on the window itself
    (:func:`~.layers.set_blur` and the others) act on the picture it shows.
    """

    def __init__(
        self, *content: Mobject, width: float, height: float, **kwargs: Any
    ) -> None:
        for each in content:
            if not isinstance(each, Mobject):
                raise TypeError(f"A window shows mobjects, not {type(each).__name__}.")
        for name, size in (("width", width), ("height", height)):
            if not (math.isfinite(size) and size > 0):
                raise ValueError(
                    f"A window's {name} is a positive length, not {size!r}."
                )
        super().__init__(
            width=width, height=height, stroke_width=0, fill_opacity=0, **kwargs
        )
        seam.set_layer(self, LayerSpec(window=content))
