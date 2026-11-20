"""What one frame stacks: layers in drawing order, bottom first, each with its effects; and what
the object map says of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass

import numpy as np


@dataclass(frozen=True)
class Picture:
    """A texture of the frame's size, of either kind, covering the frame."""

    key: bytes


@dataclass(frozen=True)
class Quad:
    """A texture placed at four corners in pixels (x right, y down), in CE's image order: upper
    left, upper right, lower left, lower right. ``resampling`` is PIL's value; this build draws
    nearest (0) and refuses any other with :class:`~arachne_gpu.Unsupported`.
    """

    key: bytes
    corners: Sequence[tuple[float, float]]
    resampling: int


@dataclass(frozen=True)
class Window:
    """Another stack, ``frame``, composed as a frame of its own of ``size`` (width, height) pixels,
    or of the size of the frame holding it when ``size`` is None, then placed at four corners as
    a :class:`Quad` is. Its pictures and masks are of its size, and its effects run at its
    size, so they stay inside it. Windows nest at most four deep, each with its own size. Inside
    a sized window, a picture or mask of another size raises
    :class:`~arachne_gpu.SizeMismatch`; with no size, every refusal is the one it was before
    windows had sizes.

    ``resampling`` is PIL's value: nearest (0) copies texels; a sized window at an axis-aligned
    rectangle, and only there, may be placed by bilinear (2), which weights the four texels round
    each pixel's centre in fixed point, times the area of the pixel's cell inside the rectangle,
    so the edge is antialiased, nothing lands in a pixel the rectangle does not touch, and a
    window moved by a fraction of a pixel moves by that fraction. Bilinear at other corners
    raises :class:`~arachne_gpu.Unsupported`. At whole-pixel corners spanning exactly ``size``, both copy the frame texel for
    texel.

    Sizing a window shown at an axis-aligned rectangle is one rule: per axis, ``size`` is the
    true rectangle's size rounded to the nearest whole pixel, ties half up, at least 1; the true
    corners pass unchanged; resampling is nearest when the rectangle is integral and 1:1 with
    ``size``, bilinear otherwise. Other quads keep the unsized window.
    """

    frame: Frame
    corners: Sequence[tuple[float, float]]
    resampling: int
    size: tuple[int, int] | None = None


@dataclass(frozen=True)
class Blur:
    """A Gaussian blur of standard deviation ``sigma_px``, in pixels.

    It runs on integers: fixed-point weights summing to exactly one, a pass along rows then one
    along columns, with 8 fractional bits kept between them. The layer is rounded back to 8 bits
    once, half up, after its mask has multiplied it.
    """

    sigma_px: float


@dataclass(frozen=True)
class SoftMask:
    """A picture (a texture of the frame's size, of either kind) whose alpha, and only its
    alpha, feathered by ``feather_px``, multiplies the layer; ``invert`` keeps what it would
    remove. The feather runs along rows then columns, so its cutoff is per axis: a pixel whose
    centre is farther than the feather from every pixel across the mask's edge, along both
    axes, is kept or removed whole. It runs on integers as a :class:`Blur` does; the product of
    the blurred layer and the feathered alpha is the one rounding back to 8 bits.
    """

    key: bytes
    feather_px: float
    invert: bool = False


@dataclass(frozen=True)
class Focus:
    """The depth in focus: a layer at depth ``d`` is blurred with
    ``sigma_px = aperture * |d - depth|``, combined with its own :class:`Blur` as the root of the
    sum of their squares into one Gaussian, which runs and rounds as a blur does. A layer at
    the focus depth is not defocused; its own blur still applies.
    """

    depth: float
    aperture: float


@dataclass(frozen=True)
class Layer:
    """One layer: its source, blurred (by ``blur`` and by ``depth`` under the frame's focus, as one
    Gaussian), multiplied by ``mask``, scaled by ``opacity`` (0 to 1) and stacked over what is
    below. ``object_id`` names the object it draws, for the object map.
    """

    source: Picture | Quad | Window
    _: KW_ONLY
    opacity: float = 1.0
    depth: float | None = None
    blur: Blur | None = None
    mask: SoftMask | None = None
    object_id: str | None = None


@dataclass(frozen=True)
class Frame:
    """One frame: ``background``, painted RGBA as 4 bytes, under every layer, then ``layers``
    bottom first; ``focus`` blurs every layer with a depth by its distance from it.
    """

    background: bytes
    layers: Sequence[Layer]
    focus: Focus | None = None


@dataclass(frozen=True)
class Footprint:
    """Where one object covers the frame, as the compositor draws it.

    ``box`` is ``(x0, y0, x1, y1)`` in pixels, half-open, y down, all zero when nothing is
    covered; ``mask`` is a bool array of the box's size, true where the layer drawn with no effect
    puts any alpha; ``grow_px`` is how far its effects spread it (three sigma of its blur, or its
    mask's feather, whichever is larger). ``layer`` is the index of the layer that draws it.
    """

    object_id: str
    layer: int
    box: tuple[int, int, int, int]
    mask: np.ndarray
    grow_px: float
