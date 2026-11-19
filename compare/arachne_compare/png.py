"""Reading and writing 8-bit PNG files with the standard library and numpy."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}  # colour type: grey, RGB, grey and alpha, RGBA


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def write_png(path: str | Path, image: np.ndarray) -> None:
    """Write a ``uint8`` image: height x width (grey, or a mask as black and white),
    or height x width x 3 or 4.
    """
    image = np.asarray(image)
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    if image.dtype != np.uint8:
        raise ValueError(f"a PNG is written from uint8 pixels, not {image.dtype}")
    if image.ndim == 2:
        image = image[..., None]
    height, width, channels = image.shape
    kind = {1: 0, 3: 2, 4: 6}.get(channels)
    if kind is None:
        raise ValueError(f"a PNG holds 1, 3 or 4 channels, not {channels}")
    rows = np.concatenate(
        [np.zeros((height, 1), np.uint8), image.reshape(height, width * channels)],
        axis=1,
    )
    header = struct.pack(">IIBBBBB", width, height, 8, kind, 0, 0, 0)
    Path(path).write_bytes(
        SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )


def _unfilter(method: int, line: bytes, previous: bytes, channels: int) -> bytes:
    """Undo one row's filter (PNG specification, section 9), byte by byte."""
    if method == 0:
        return line
    if method == 2:
        return bytes((a + b) & 0xFF for a, b in zip(line, previous, strict=True))
    row = bytearray(line)
    for x in range(len(row)):
        left = row[x - channels] if x >= channels else 0
        up = previous[x]
        if method == 1:
            row[x] = (row[x] + left) & 0xFF
        elif method == 3:
            row[x] = (row[x] + ((left + up) >> 1)) & 0xFF
        elif method == 4:
            up_left = previous[x - channels] if x >= channels else 0
            p = left + up - up_left
            pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
            guess = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
            row[x] = (row[x] + guess) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter {method}")
    return bytes(row)


def read_png(path: str | Path) -> np.ndarray:
    """Read an 8-bit, non-interlaced grey, grey and alpha, RGB or RGBA PNG.

    Grey images come back as RGB (with alpha where they have it): height x width x 3
    or 4, ``uint8``.
    """
    data = Path(path).read_bytes()
    if not data.startswith(SIGNATURE):
        raise ValueError(f"{path}: not a PNG file")
    at = len(SIGNATURE)
    header = None
    compressed = bytearray()
    while at < len(data):
        (length,) = struct.unpack(">I", data[at : at + 4])
        kind = data[at + 4 : at + 8]
        body = data[at + 8 : at + 8 + length]
        at += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            compressed += body
        elif kind == b"IEND":
            break
    if header is None:
        raise ValueError(f"{path}: no IHDR chunk")
    width, height, depth, kind_code, _, _, interlace = header
    if depth != 8 or kind_code not in CHANNELS or interlace:
        raise ValueError(
            f"{path}: only 8-bit, non-interlaced grey, RGB and RGBA PNGs are read "
            f"(depth {depth}, colour type {kind_code}, interlace {interlace})"
        )
    channels = CHANNELS[kind_code]
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    if len(raw) != height * (stride + 1):
        raise ValueError(
            f"{path}: the pixel data has {len(raw)} bytes, not {height * (stride + 1)}"
        )
    rows = []
    previous = bytes(stride)
    for y in range(height):
        start = y * (stride + 1)
        previous = _unfilter(
            raw[start], raw[start + 1 : start + 1 + stride], previous, channels
        )
        rows.append(previous)
    image = (
        np.frombuffer(b"".join(rows), np.uint8).reshape(height, width, channels).copy()
    )
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    elif channels == 2:
        image = np.concatenate(
            [np.repeat(image[..., :1], 3, axis=2), image[..., 1:]], axis=2
        )
    return image
