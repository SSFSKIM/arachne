"""Fixture pictures and closed forms for the GPU build's tests, and the size they draw at.

The tests that draw frames of a frame's size draw at ``ARACHNE_GPU_TEST_SIZE``
(``<width>x<height>``, at least 480 x 270), or at 480 x 270 when it is unset: CI's size, and a GPU
host's second size, 1920x1080. Their closed-form bounds hold at any size. Test code is the only
reader of it, as of the launch mode.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping

import numpy as np

SIZE = "ARACHNE_GPU_TEST_SIZE"
DEFAULT_SIZE = (480, 270)


def drawing_size(environ: Mapping[str, str]) -> tuple[int, int]:
    """The size frame-sized tests draw at; raises on a malformed or too small one."""
    value = environ.get(SIZE)
    if value is None:
        return DEFAULT_SIZE
    found = re.fullmatch(r"(\d+)x(\d+)", value)
    if found is None:
        raise RuntimeError(f"{SIZE}={value}: not <width>x<height>")
    width, height = int(found[1]), int(found[2])
    if width < DEFAULT_SIZE[0] or height < DEFAULT_SIZE[1]:
        raise RuntimeError(f"{SIZE}={value}: the tests draw at 480x270 or larger")
    return width, height


W, H = drawing_size(os.environ)
CLEAR = bytes(4)


def painted_picture(width: int, height: int, seed: int) -> np.ndarray:
    """Valid painted RGBA: every colour value at most its alpha."""
    rng = np.random.default_rng(seed)
    alpha = rng.integers(0, 256, (height, width, 1), dtype=np.uint16)
    colour = rng.integers(0, 256, (height, width, 3), dtype=np.uint16)
    return np.concatenate([colour * alpha // 255, alpha], axis=2).astype(np.uint8)


def flat(width: int, height: int, rgba: tuple[int, int, int, int]) -> np.ndarray:
    return np.broadcast_to(np.array(rgba, np.uint8), (height, width, 4)).copy()


def xf_image_key(rgba: np.ndarray) -> bytes:
    """The Frame IR's image-shape key, as ``core/XF.md`` states it: BLAKE2b, 16-byte digest,
    over the kind code 3, one array, dtype code 3 (u8), three dimensions, each as u64, the data.
    """
    height, width, channels = rgba.shape
    canonical = bytes([3, 1, 3, 3])
    for dimension in (height, width, channels):
        canonical += dimension.to_bytes(8, "little")
    return hashlib.blake2b(
        canonical + np.ascontiguousarray(rgba).tobytes(), digest_size=16
    ).digest()


def over(top: np.ndarray, below: np.ndarray) -> np.ndarray:
    """Painted "over" in real arithmetic, rounded."""
    keep = 1.0 - top[..., 3:4].astype(np.float64) / 255.0
    return np.rint(top + below.astype(np.float64) * keep).astype(np.uint8)
