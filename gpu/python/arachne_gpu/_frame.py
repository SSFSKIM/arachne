"""What one frame stacks: layers in drawing order, bottom first."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass


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
class Layer:
    """One layer: its source, scaled by ``opacity`` (0 to 1) and stacked over what is below."""

    source: Picture | Quad
    _: KW_ONLY
    opacity: float = 1.0


@dataclass(frozen=True)
class Frame:
    """One frame: ``background``, painted RGBA as 4 bytes, under every layer, then ``layers``
    bottom first.
    """

    background: bytes
    layers: Sequence[Layer]
