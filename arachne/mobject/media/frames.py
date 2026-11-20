"""Frames of media assets, read exactly: the forms the engine admits, the
frame digest, and the process's frame cache.

A frame is 8-bit RGBA pixels, straight: one picture of a clip or a sprite,
or the one picture of a still. It is stored as one file in an admitted form
(:data:`ADMITTED`): a PNG of 8 bits per channel, truecolour with or without
alpha. A PNG stores its samples without loss, so every conforming decoder
gives the same pixels; Pillow decodes it, applying no gamma and no colour
profile, with alpha 255 where the file has none. A file in any other form
(another bit depth, a palette, greyscale, a transparency chunk, an
animated PNG, no PNG at all) is ``media.inexact`` with the reason ``format``, found from its header
when its asset loads (:func:`refuse_unadmitted_form`). A later form is one
more row of :data:`ADMITTED`, with the header reading its container needs.

A frame's digest (:func:`frame_digest`) is ``sha256:`` and the hex SHA-256
of its width and its height, each as 4 bytes big-endian, then its RGBA
bytes row by row from the top. A media asset's manifest lists the digest of
every frame, and every time a process decodes a frame it checks the pixels'
size against the manifest's and then the pixels against the digest, before
anything keeps or draws them (:func:`checked_frame`): a mismatch of either
is ``media.inexact`` with the reason ``digest``, raised with the line of
the scene that made the object, so a host that decodes otherwise draws
nothing of that frame.

Decoded frames are kept in memory by the process's :data:`cache`, bounded
in bytes, the least recently used leaving first, keyed by the asset's
digest and the frame file's path inside the asset; nothing of it is written
to disk, and each process fills its own. Importing this module reads no
file and decodes nothing.
"""

from __future__ import annotations

__all__ = [
    "ADMITTED",
    "CACHE_BYTES",
    "EXPECTED_FORM",
    "Form",
    "FrameCache",
    "cache",
    "checked_frame",
    "decode",
    "form_of",
    "frame_digest",
    "refuse_unadmitted_form",
]

import hashlib
import io
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

import numpy as np
from PIL import Image

from ...assets.manifest import normalised
from ...lint.raised import RaisedFinding
from ..values import FrozenPixels, adopt

if TYPE_CHECKING:
    from ...assets.reader import Asset


@dataclass(frozen=True)
class Form:
    """One admitted way of storing a frame: a PNG whose samples are
    ``bit_depth`` bits each, of the PNG colour type ``colour_type``.
    """

    name: str
    bit_depth: int
    colour_type: int


#: The forms a frame file may be in: data, so that a later form is a row.
ADMITTED: tuple[Form, ...] = (
    Form("a PNG of 8 bits per channel, truecolour", 8, 2),
    Form("a PNG of 8 bits per channel, truecolour with alpha", 8, 6),
)
#: What a refusal for a frame file's form says the engine expects.
EXPECTED_FORM = "a PNG of 8 bits per channel, truecolour with or without alpha"

#: The first eight bytes of every PNG file.
_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOUR_TYPES = {
    0: "greyscale",
    2: "truecolour",
    3: "indexed colour (a palette)",
    4: "greyscale with alpha",
    6: "truecolour with alpha",
}


def form_of(data: bytes) -> Form | str:
    """The admitted form the file ``data`` is in, read from its header; or,
    when it is in none, what it is, in words.
    """
    if not data.startswith(_SIGNATURE):
        return "not a PNG file"
    header = data[8:33]
    if len(header) < 25 or header[:8] != b"\x00\x00\x00\x0dIHDR":
        return "a PNG file whose header cannot be read"
    depth, colour = header[16], header[17]
    found = (
        f"a PNG of {depth} bits per channel,"
        f" {_COLOUR_TYPES.get(colour, f'of colour type {colour}')}"
    )
    form = next(
        (f for f in ADMITTED if (f.bit_depth, f.colour_type) == (depth, colour)),
        None,
    )
    if form is None:
        return found
    if _chunk_before_data(data, b"tRNS"):
        # A transparency chunk makes one colour transparent in a file that
        # holds no alpha: alpha the frame's samples do not carry.
        return f"{found}, with a transparency chunk"
    if _chunk_before_data(data, b"acTL"):
        # An animated PNG (APNG): its first frame is only one of the
        # pictures the file holds, and decoders differ on which they show.
        return f"{found}, animated (an APNG)"
    return form


def _chunk_before_data(data: bytes, kind: bytes) -> bool:
    """Whether the PNG ``data`` holds a chunk of ``kind`` before its image
    data.
    """
    at = len(_SIGNATURE)
    while at + 8 <= len(data):
        length = int.from_bytes(data[at : at + 4], "big")
        found = data[at + 4 : at + 8]
        if found in (b"IDAT", b"IEND"):
            return False
        if found == kind:
            return True
        at += 12 + length
    return False


def refuse_unadmitted_form(data: bytes, asset: str, file: str) -> None:
    """Refuse the frame file ``file`` of the asset ``asset`` (as it was asked
    for), whose bytes are ``data``, unless it is in an admitted form:
    ``media.inexact`` with the reason ``format``.
    """
    found = form_of(data)
    if isinstance(found, str):
        _inexact(asset, file, "format", EXPECTED_FORM, found, None)


def decode(data: bytes) -> np.ndarray:
    """The pixels of the frame file ``data``, in an admitted form: height by
    width by 4, 8 bits each, straight RGBA, alpha 255 where the file has
    none. Raises :class:`ValueError`, saying why, for a file that does not
    decode so.
    """
    try:
        with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
            image.load()
            mode = image.mode
            pixels = np.array(image, dtype=np.uint8)
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"a PNG that does not decode ({error})") from None
    if mode == "RGB":
        opaque = np.full((*pixels.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate([pixels, opaque], axis=2)
    if mode != "RGBA":
        raise ValueError(f"a PNG that decodes as Pillow's mode {mode}")
    return pixels


def frame_digest(pixels: np.ndarray) -> str:
    """The frame digest of ``pixels``, height by width by 4, 8 bits each:
    ``sha256:`` and the hex SHA-256 of the width and the height, each as 4
    bytes big-endian, then the RGBA bytes row by row from the top.
    """
    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
        raise ValueError("a frame is height by width by 4 bytes of RGBA")
    height, width = pixels.shape[:2]
    sha = hashlib.sha256()
    sha.update(width.to_bytes(4, "big"))
    sha.update(height.to_bytes(4, "big"))
    sha.update(np.ascontiguousarray(pixels).tobytes())
    return f"sha256:{sha.hexdigest()}"


def checked_frame(asset: Asset, index: int, line: int | None = None) -> FrozenPixels:
    """The frame ``index`` of the media asset ``asset``, read from its file
    now, decoded, and checked against the size and the frame digest its
    manifest lists, read-only. Every refusal is raised with ``line``, the
    scene's line that made the object being drawn: the file gone, unreadable
    or now reached through a link, its form, its size and its pixels
    (``media.inexact``).
    """
    from ...assets.reader import read_file

    media = asset.manifest.media
    if media is None:
        raise ValueError(f"the asset {asset.name} is no media asset")
    spec = media.frames[index]
    file = normalised(spec.file)
    try:
        data = read_file(asset.directory, file, asset.name)
    except RaisedFinding as refused:
        raise RaisedFinding(
            refused.code, refused.what, detail=refused.detail, line=line
        ) from None
    found = form_of(data)
    if isinstance(found, str):
        _inexact(asset.name, spec.file, "format", EXPECTED_FORM, found, line)
    try:
        pixels = decode(data)
    except ValueError as error:
        _inexact(asset.name, spec.file, "format", EXPECTED_FORM, str(error), line)
    height, width = pixels.shape[:2]
    if (width, height) != media.pixels:
        # Checked at every decode, before a digest is computed or the frame
        # is kept: a frame of another size is drawn by no one.
        size = f"{width} x {height} pixels"
        declared = f"{media.pixels[0]} x {media.pixels[1]} pixels"
        _inexact(
            asset.name,
            spec.file,
            "digest",
            declared,
            size,
            line,
            what=(
                f"the frame {spec.file} of the asset {asset.name} decodes here to"
                f" {size}, not the {declared} its manifest declares; nothing of"
                " it is drawn"
            ),
        )
    digest = frame_digest(pixels)
    if digest != spec.digest:
        _inexact(
            asset.name,
            spec.file,
            "digest",
            spec.digest,
            digest,
            line,
            what=(
                f"the frame {spec.file} of the asset {asset.name} decodes here to"
                f" pixels whose digest is {digest}, not the {spec.digest} its"
                " manifest lists; nothing of it is drawn"
            ),
        )
    return adopt(pixels, FrozenPixels)


def _inexact(
    asset: str,
    file: str,
    reason: str,
    expected: str,
    found: str,
    line: int | None,
    *,
    what: str | None = None,
) -> NoReturn:
    raise RaisedFinding(
        "media.inexact",
        what
        or f"the frame file {file} of the asset {asset} is {found}; the engine"
        f" decodes exactly only {expected}",
        detail={
            "asset": asset,
            "file": file,
            "reason": reason,
            "expected": expected,
            "found": found,
        },
        line=line,
    )


#: How many bytes of decoded frames a process keeps, at most.
CACHE_BYTES = 256 * 2**20


class FrameCache:
    """Decoded, checked frames, kept in memory up to :attr:`bound` bytes,
    the least recently used leaving first; a frame larger than the bound is
    decoded and not kept. Keyed by the asset's digest and the frame file's
    path inside the asset, so two files of one asset are never one entry,
    and two objects of one asset share their frames.
    """

    def __init__(self, bound: int = CACHE_BYTES) -> None:
        #: The most bytes of frames kept.
        self.bound = bound
        self._held: OrderedDict[tuple[str, str], FrozenPixels] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def nbytes(self) -> int:
        """How many bytes of frames are kept now."""
        return self._bytes

    def __len__(self) -> int:
        return len(self._held)

    def frame(self, asset: Asset, index: int, line: int | None = None) -> FrozenPixels:
        """The frame ``index`` of ``asset``, read-only: kept, or decoded and
        checked now (:func:`checked_frame`, whose refusals carry ``line``).
        """
        media = asset.manifest.media
        if media is None:
            raise ValueError(f"the asset {asset.name} is no media asset")
        key = (asset.digest, normalised(media.frames[index].file))
        with self._lock:
            found = self._held.get(key)
            if found is not None:
                self._held.move_to_end(key)
                return found
        pixels = checked_frame(asset, index, line)
        with self._lock:
            if key not in self._held and pixels.nbytes <= self.bound:
                self._held[key] = pixels
                self._bytes += pixels.nbytes
                while self._bytes > self.bound:
                    _, gone = self._held.popitem(last=False)
                    self._bytes -= gone.nbytes
        return pixels

    def clear(self) -> None:
        """Let every frame go."""
        with self._lock:
            self._held.clear()
            self._bytes = 0


#: The process's frame cache.
cache = FrameCache()
