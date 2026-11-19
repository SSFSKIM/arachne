"""Fixture pictures and closed forms for the GPU build's tests."""

from __future__ import annotations

import hashlib

import numpy as np

W, H = 480, 270
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
