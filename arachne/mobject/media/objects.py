"""Media objects: a still, a clip or a sprite, placed in a scene like any
image by naming its asset.

``Still(ref)``, ``VideoMobject(ref)`` and ``Sprite(ref)`` load the asset
``ref`` names (:func:`~arachne.assets.load_asset`: a directory, or
``fixture:<name>``) of the kind ``still``, ``clip`` or ``sprite``, and stand
at the size its manifest declares, on its first frame. Each is an image
object, so it is placed, scaled, ordered and faded as any image is.

What a media object holds, and the scene's record keeps at every capture,
is never pixels: the asset's digest, the index of the frame on display, an
opacity and its four corners, which decide the picture; and beside them the
asset loaded (an immutable value that knows its directory) and the scene's
line that made the object. Its pixels (:meth:`get_pixel_array`) are that
frame, decoded and checked by the process's frame cache
(:mod:`.frames`), with the opacity applied to their alpha. So a clip's
record stays small whatever the clip's length, and whatever reads the
record gets the pixels the live scene had.

An animation may change where a media object is, how large, in what order
and how opaque, and never which frame it shows. Two media objects of other
assets, or a media object and another kind of object, do not morph: such a
transform is a ``ValueError`` naming the fade that replaces it.

A clip or a sprite declares a GPU-only capability (``media.video``,
``media.sprite``), so a scene that shows one is drawn on the GPU. A still
declares none: a scene whose only new object is a still stays a CPU frame.
"""

from __future__ import annotations

__all__ = ["Sprite", "Still", "VideoMobject"]

import os
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from PIL.Image import Resampling

from ...assets.reader import Asset, load_asset
from ...constants import DOWN, LEFT, RIGHT, UP
from ...routing import declare
from ...timeline.origins import ORIGINS
from ...utils.bezier import interpolate
from ..mobject import morph_refusal
from ..types.image_mobject import AbstractImageMobject
from ..values import FrozenPixels, adopt
from . import frames

if TYPE_CHECKING:
    from typing import Self

    from ...typing import PixelArray
    from ..mobject import Mobject


@dataclass(frozen=True)
class _Made:
    """The scene's line that made a media object, kept beside what decides
    its picture: a refusal raised while one of its frames is drawn, with no
    scene code running, names it.
    """

    line: int | None


class MediaObject(AbstractImageMobject):
    """A media object: the frames of one media asset, standing on one."""

    #: The manifest kind this class shows.
    asset_kind: ClassVar[str]
    #: The GPU-only capability its objects declare, if any.
    capability: ClassVar[str | None] = None

    def __init__(
        self,
        asset: Asset | str | os.PathLike[str],
        resampling_algorithm: Resampling = Resampling.BICUBIC,
        **kwargs: Any,
    ) -> None:
        loaded = load_asset(asset)
        if loaded.kind != self.asset_kind:
            shown_by = _CLASSES.get(loaded.kind, "Rig")
            raise ValueError(
                f"the asset {loaded.name} is a {loaded.kind}, not a {self.asset_kind}:"
                f" show it with {shown_by}"
            )
        origin = ORIGINS.get(self)
        self._asset = loaded
        self._digest = loaded.digest
        self._frame = 0
        self._opacity = 1.0
        self._made = _Made(origin.scene_line if origin is not None else None)
        super().__init__(
            scale_to_resolution=0, resampling_algorithm=resampling_algorithm, **kwargs
        )
        if self.capability is not None:
            declare(self, self.capability)

    @property
    def asset(self) -> Asset:
        """The asset this object shows."""
        return self._asset

    @property
    def frame(self) -> int:
        """The index of the frame on display."""
        return self._frame

    @property
    def opacity(self) -> float:
        """How opaque the object is, from 0 to 1."""
        return self._opacity

    @property
    def pixel_array(self) -> FrozenPixels:
        """The frame on display, as :meth:`get_pixel_array` gives it."""
        return self.get_pixel_array()

    @pixel_array.setter
    def pixel_array(self, value: PixelArray) -> None:
        raise AttributeError(
            "a media object's pixels are its asset's frames: show another asset"
            " with an object of its own"
        )

    def get_pixel_array(self) -> FrozenPixels:
        """The frame on display, decoded and checked against its digest, its
        alpha multiplied by the object's opacity where that is below 1.
        """
        pixels = frames.cache.frame(self._asset, self._frame, self._made.line)
        if self._opacity >= 1:
            return pixels
        faded = np.array(pixels)
        faded[:, :, 3] = pixels[:, :, 3] * self._opacity
        return adopt(faded, FrozenPixels)

    def reset_points(self) -> Self:
        """The four corners of the asset's natural size, about the origin."""
        self.points = np.array([UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT])
        extent = self._asset.manifest.extent
        self.stretch_to_fit_height(extent.height)
        self.stretch_to_fit_width(extent.width)
        return self

    def set_opacity(self, alpha: float) -> Self:
        """Show the object ``alpha`` opaque, from 0 to 1."""
        self._opacity = min(max(float(alpha), 0.0), 1.0)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True) -> Self:
        """Show the object ``1 - darkness`` opaque."""
        self.set_opacity(1 - darkness)
        super().fade(darkness, family)
        return self

    def interpolate_color(
        self, mobject1: Mobject, mobject2: Mobject, alpha: float
    ) -> Self:
        """The opacity between ``mobject1``'s and ``mobject2``'s, media
        objects of this object's asset; the frame on display is left as it
        is.
        """
        start, end = self._same_picture(mobject1), self._same_picture(mobject2)
        self._opacity = float(interpolate(start._opacity, end._opacity, alpha))
        return self

    def _same_picture(self, other: Mobject) -> MediaObject:
        """``other``, a media object of this object's asset; anything else is
        refused as alignment refuses it.
        """
        if isinstance(other, MediaObject) and other._digest == self._digest:
            return other
        raise morph_refusal(self, other)

    def _morph_key(self) -> str:
        """The asset's digest: a media object aligns, and so is transformed,
        only with media objects of its own asset.
        """
        return self._digest

    def _morph_name(self) -> str:
        return f"{super()._morph_name()} of the asset {self._asset.name}"


class Still(MediaObject):
    """A still: one frame of an asset of the kind ``still``, standing at the
    size its manifest declares. It declares no GPU-only capability.
    """

    asset_kind = "still"


class _Clocked(MediaObject):
    """A media object with frames at a rate: a clip or a sprite."""

    @property
    def frame_count(self) -> int:
        """How many frames the asset has."""
        media = self._asset.manifest.media
        assert media is not None
        return len(media.frames)

    @property
    def frame_rate(self) -> Fraction:
        """The asset's frames a second."""
        media = self._asset.manifest.media
        assert media is not None
        assert media.rate is not None
        return media.rate

    @property
    def duration(self) -> Fraction:
        """The asset's length in seconds: its frame count over its rate."""
        return self.frame_count / self.frame_rate


class VideoMobject(_Clocked):
    """A clip: the frames of an asset of the kind ``clip``, standing at the
    size its manifest declares, on its first frame. A scene that shows one
    is drawn on the GPU (``media.video``).
    """

    asset_kind = "clip"
    capability = "media.video"


class Sprite(_Clocked):
    """A sprite: the drawings of an asset of the kind ``sprite``, standing
    at the size its manifest declares, on its first. A scene that shows one
    is drawn on the GPU (``media.sprite``).
    """

    asset_kind = "sprite"
    capability = "media.sprite"


#: The class that shows each media kind.
_CLASSES = {"still": "Still", "clip": "VideoMobject", "sprite": "Sprite"}
