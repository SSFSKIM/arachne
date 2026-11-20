"""Raster members of a rig: parts, or slots' drawings, that are stills.

A rig is a group of vector objects, so a raster member stands in it as a
:class:`RasterPart`: an unpainted four-cornered vector object whose rest
outline is the rectangle of the member's ``box`` on the canvas, its corners
first at the box's upper left and then clockwise as drawn. It is a part
like any other: it has its pivot, parent, order and drivers, and a pose, a
settle, the rig's carrier and its anchors treat it as they treat a part of
vector shapes, so the rig's centre, its box and a scale about it take the
path a vector rig's take. A hidden drawing's quad holds no points, as every
hidden drawing does.

Its **picture**, the member's still, is drawn at the quad's four corners
whenever a frame is drawn (:meth:`RasterPart.picture`), from the quad's
state there: its corners, and its fill opacity, which is the picture's
opacity. So a picture never lags its quad: an animation that moves or fades
the rig moves and fades it at every instant. The quad's colours and its
stroke are never drawn, so a colour given to the rig recolours no still.
The quad says so (``fill_is_picture``), and the lint reads it as it reads a
still: an image over its box, in no unit a sweep measures.

What the quad holds that decides the picture is plain: the asset's digest,
the member's file and its frame digest, as strings. The picture's pixels
are the member's file, read through the process's frame cache by the
asset's digest and the file's path, and checked against the frame digest
each time it is decoded (``media.inexact`` otherwise,
:mod:`arachne.mobject.media.frames`). Where this host keeps the asset (its
directory) is held beside them, and is no part of what a content key
covers.

A raster member is drawn by the GPU build only: each quad declares the
capability ``media.raster_part``, so a scene that shows one is on the
``gpu`` route.
"""

from __future__ import annotations

__all__ = ["RasterPart", "RasterPicture"]

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL.Image import Resampling

from ..mobject.types.vectorized_mobject import VMobject
from ..mobject.values import FrozenPixels, adopt
from ..routing import declare
from ..utils.color import WHITE

#: The capability a raster member declares.
CAPABILITY = "media.raster_part"


class RasterPart(VMobject):
    """The quad of a raster member, at rest on the canvas: the rectangle
    ``box`` (``x0, y0, x1, y1``, canvas units, y down) of the file ``file``
    (as the manifest names it) of the asset whose digest is
    ``asset_digest``, named ``asset`` and kept in ``directory``; ``digest``
    is the frame digest its manifest lists for the file. Its fill opacity
    is the picture's opacity.
    """

    #: The path's fill is a picture: it is never painted, and the picture
    #: (:meth:`picture`) is painted in its place, an image at the corners
    #: :meth:`corners` gives, as opaque as the path's fill. A reader of what
    #: is drawn (the lint) reads such a path as an image, as a still, and
    #: reads nothing of its paint but its fill opacity. Absent on every other
    #: mobject: ``getattr(mob, "fill_is_picture", False)``.
    fill_is_picture: ClassVar[bool] = True

    #: What a content key leaves out: where this host keeps the asset, and
    #: the asset's name, which a refusal reads.
    _content_key_omits: ClassVar[frozenset[str]] = frozenset(
        {"_directory", "_asset_name"}
    )

    def __init__(
        self,
        box: Sequence[float],
        *,
        asset_digest: str,
        file: str,
        digest: str,
        asset: str,
        directory: str | os.PathLike[str],
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("fill_color", WHITE)
        kwargs.setdefault("fill_opacity", 1.0)
        kwargs.setdefault("stroke_width", 0)
        super().__init__(**kwargs)
        x0, y0, x1, y1 = (float(v) for v in box)
        self.set_points_as_corners(
            [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0]]
        )
        self._asset_digest = asset_digest
        self._file = file
        self._digest = digest
        self._asset_name = asset
        self._directory = Path(directory)
        declare(self, CAPABILITY)

    @property
    def file(self) -> str:
        """The member's file, as the manifest names it."""
        return self._file

    def corners(self) -> np.ndarray | None:
        """Where the picture's corners are now, in CE's image order (upper
        left, upper right, lower left, lower right of the still as drawn):
        the quad's first, second, fourth and third corner; ``None`` for a
        quad that holds no corners (a hidden drawing's).
        """
        points = np.asarray(self.points)
        if len(points) < 16:
            return None
        return np.array([points[0], points[4], points[12], points[8]], dtype=float)

    def pixels(self, line: int | None = None) -> FrozenPixels:
        """The picture's pixels: the member's file, decoded and checked
        against its frame digest (refusals raised at ``line``), its alpha
        multiplied by the quad's fill opacity where that is below 1.
        """
        from ..mobject.media import frames

        found = frames.cache.member(
            self._asset_digest,
            self._directory,
            self._asset_name,
            self._file,
            self._digest,
            line,
        )
        opacity = float(self.get_fill_opacity())
        if opacity >= 1:
            return found
        faded = np.array(found)
        faded[:, :, 3] = found[:, :, 3] * max(opacity, 0.0)
        return adopt(faded, FrozenPixels)

    def picture(self, line: int | None = None) -> RasterPicture | None:
        """What a camera paints for the quad as it stands: its picture at
        its corners, read when painted (refusals raised at ``line``, the
        scene's line that made the rig); ``None`` for a quad with no
        corners, which paints nothing.
        """
        corners = self.corners()
        if corners is None:
            return None
        return RasterPicture(self, corners, line)


@dataclass(frozen=True, eq=False)
class RasterPicture:
    """A raster member's picture as a camera's image painter reads an
    image: its ``points``, the four corners in CE's image order, its pixels
    (:meth:`get_pixel_array`) and its resampling. It is no mobject: it is
    made for one paint and is in no scene.
    """

    part: RasterPart
    points: np.ndarray
    line: int | None = None
    #: How the still is resampled where it is drawn larger or smaller, as
    #: an image object's is by default.
    resampling_algorithm: ClassVar[Resampling] = Resampling.BICUBIC

    def get_pixel_array(self) -> FrozenPixels:
        """The picture's pixels (:meth:`RasterPart.pixels`)."""
        return self.part.pixels(self.line)
