"""The frame reader: the admitted forms, read from a file's header as its
asset loads; frames decoded exactly, to the digests their manifests list, on
every host the tests run on; each decode checked; and the frame cache.
"""

from __future__ import annotations

import io
import os
import shutil
import struct

import numpy as np
import pytest
from PIL import Image

from arachne import Still, VideoMobject
from arachne.assets import load_asset
from arachne.assets.reader import read_asset
from arachne.lint.raised import RaisedFinding
from arachne.mobject.media import frames
from arachne.mobject.media.frames import (
    ADMITTED,
    FrameCache,
    decode,
    form_of,
    frame_digest,
)

from .conftest import copied, png

FIXTURE_MEDIA = ("numbered-clip", "stepper", "swatch")
#: An APNG's animation control chunk: one frame, played once.
ANIMATED = (b"acTL", struct.pack(">II", 1, 0))


@pytest.mark.parametrize("name", FIXTURE_MEDIA)
def test_every_fixture_frame_decodes_to_its_listed_digest(name):
    """The frames are PNG files, which every conforming decoder reads to the
    same pixels: here, on every platform these tests run on.
    """
    asset = load_asset(f"fixture:{name}")
    for index, frame in enumerate(asset.manifest.media.frames):
        pixels = frames.checked_frame(asset, index)
        assert frame_digest(pixels) == frame.digest
        assert pixels.shape == (*asset.manifest.media.pixels[::-1], 4)


def _rgb_rows(width: int, height: int) -> list[bytes]:
    return [
        bytes((x * 7 + y) % 256 for x in range(width) for _ in range(3))
        for y in range(height)
    ]


def test_the_admitted_forms_are_eight_bit_truecolour_with_or_without_alpha():
    assert [(form.bit_depth, form.colour_type) for form in ADMITTED] == [(8, 2), (8, 6)]
    rgb = png(_rgb_rows(4, 3), 4, 3, 8, 2)
    assert form_of(rgb) is ADMITTED[0]
    pixels = decode(rgb)
    assert pixels.shape == (3, 4, 4)
    # A file without alpha is read with alpha 255.
    assert (pixels[..., 3] == 255).all()
    assert pixels[2, 1, :3].tolist() == [(1 * 7 + 2) % 256] * 3


@pytest.mark.parametrize(
    ("data", "found"),
    [
        (
            png([bytes(16)] * 2, 2, 2, 16, 6),
            "a PNG of 16 bits per channel, truecolour with alpha",
        ),
        (
            png([bytes(2)] * 2, 2, 2, 8, 3),
            "a PNG of 8 bits per channel, indexed colour",
        ),
        (png([bytes(2)] * 2, 2, 2, 8, 0), "a PNG of 8 bits per channel, greyscale"),
        (
            png([bytes(6)] * 2, 2, 2, 8, 2, chunks=((b"tRNS", bytes(6)),)),
            "with a transparency chunk",
        ),
        (png([bytes(8)] * 2, 2, 2, 8, 6, chunks=(ANIMATED,)), "animated (an APNG)"),
        (b"GIF89a" + bytes(40), "not a PNG file"),
    ],
    ids=[
        "sixteen-bit",
        "palette",
        "greyscale",
        "transparency-chunk",
        "animated",
        "not-png",
    ],
)
def test_a_file_in_no_admitted_form_is_named_for_what_it_is(data, found):
    named = form_of(data)
    assert isinstance(named, str)
    assert found in named


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(out, format="JPEG")
    return out.getvalue()


@pytest.mark.parametrize(
    "data",
    [
        png([bytes(64 * 8)] * 64, 64, 64, 16, 6),
        png([bytes(64)] * 64, 64, 64, 8, 3, chunks=((b"PLTE", bytes(3)),)),
        _jpeg(),
        png([bytes(64 * 4)] * 64, 64, 64, 8, 6, chunks=(ANIMATED,)),
    ],
    ids=["sixteen-bit", "palette", "jpeg", "animated"],
)
def test_loading_refuses_a_frame_file_in_no_admitted_form(tmp_path, data):
    directory = copied(tmp_path, "swatch")
    (directory / "frames" / "0000.png").write_bytes(data)
    with pytest.raises(RaisedFinding) as raised:
        read_asset(directory)
    found = raised.value
    assert found.code == "media.inexact"
    assert found.detail["reason"] == "format"
    assert found.detail["file"] == "frames/0000.png"
    assert found.detail["expected"] == frames.EXPECTED_FORM


@pytest.fixture
def changed_decoder(monkeypatch):
    """A decoder that gives one pixel of every frame other than the file
    says.
    """
    decoded = frames.decode

    def changed(data: bytes) -> np.ndarray:
        pixels = decoded(data)
        pixels[0, 0, 0] ^= 1
        return pixels

    monkeypatch.setattr(frames, "decode", changed)
    monkeypatch.setattr(frames, "cache", FrameCache())
    return changed


def test_a_frame_that_decodes_otherwise_is_refused_with_the_line_that_made_it(
    changed_decoder,
):
    still = Still("fixture:swatch")
    still._made = type(still._made)(17)
    with pytest.raises(RaisedFinding) as raised:
        still.get_pixel_array()
    found = raised.value
    assert found.code == "media.inexact"
    assert found.line == 17
    listed = still.asset.manifest.media.frames[0].digest
    assert found.detail["reason"] == "digest"
    assert found.detail["expected"] == listed
    assert found.detail["found"] != listed


def test_the_cache_keeps_frames_by_asset_and_file_and_lets_the_oldest_go(monkeypatch):
    kept = FrameCache()
    monkeypatch.setattr(frames, "cache", kept)
    asset = load_asset("fixture:numbered-clip")
    first = kept.frame(asset, 0)
    assert kept.frame(asset, 0) is first
    assert len(kept) == 1
    # Two objects of one asset share their decoded frames.
    one, other = VideoMobject(asset), VideoMobject("fixture:numbered-clip")
    assert one.get_pixel_array() is other.get_pixel_array() is first
    # Bounded: three frames' bytes, the least recently used leaving first.
    kept.bound = 3 * first.nbytes
    second, third = kept.frame(asset, 1), kept.frame(asset, 2)
    # Frame 0, the oldest kept, is used again: frame 1 is now the least
    # recently used, and the one the next frame pushes out (first in, first
    # out would push out frame 0).
    assert kept.frame(asset, 0) is first
    kept.frame(asset, 3)
    assert len(kept) == 3
    assert kept.nbytes <= kept.bound
    assert kept.frame(asset, 0) is first
    assert kept.frame(asset, 2) is third
    assert kept.frame(asset, 1) is not second
    assert all(kept.frame(asset, i).flags.writeable is False for i in range(4))


def test_a_frame_larger_than_the_bound_is_decoded_and_not_kept():
    kept = FrameCache(bound=10)
    asset = load_asset("fixture:swatch")
    assert kept.frame(asset, 0).shape == (64, 64, 4)
    assert len(kept) == kept.nbytes == 0


def test_a_frame_decoded_again_after_the_cache_let_it_go_is_checked_again(
    monkeypatch,
):
    kept = FrameCache()
    monkeypatch.setattr(frames, "cache", kept)
    still = Still("fixture:swatch")
    still._made = type(still._made)(23)
    still.get_pixel_array()
    kept.clear()
    decoded = frames.decode

    def changed(data: bytes) -> np.ndarray:
        pixels = decoded(data)
        pixels[-1, -1, 3] ^= 1
        return pixels

    monkeypatch.setattr(frames, "decode", changed)
    with pytest.raises(RaisedFinding) as raised:
        still.get_pixel_array()
    assert (raised.value.code, raised.value.line) == ("media.inexact", 23)


def test_a_frame_of_another_size_is_refused_at_every_decode(monkeypatch):
    kept = FrameCache()
    monkeypatch.setattr(frames, "cache", kept)
    decoded = frames.decode
    monkeypatch.setattr(frames, "decode", lambda data: decoded(data)[1:])
    still = Still("fixture:swatch")
    still._made = type(still._made)(31)
    with pytest.raises(RaisedFinding) as raised:
        still.get_pixel_array()
    found = raised.value
    assert (found.code, found.line) == ("media.inexact", 31)
    assert found.detail["reason"] == "digest"
    assert found.detail["expected"] == "64 x 64 pixels"
    assert found.detail["found"] == "64 x 63 pixels"
    assert len(kept) == 0


@pytest.mark.parametrize("linked", ["frames/0000.png", "frames"])
def test_a_frame_read_after_loading_through_a_link_is_refused(
    tmp_path, monkeypatch, linked
):
    """A frame file, or the directory it is in, turned into a link after its
    asset loaded is refused when the frame is next read, as loading refuses
    it, with the line that made the object.
    """
    monkeypatch.setattr(frames, "cache", FrameCache())
    directory = copied(tmp_path, "swatch")
    still = Still(directory)
    still._made = type(still._made)(41)
    moved = tmp_path / "elsewhere"
    shutil.move(directory / linked, moved)
    os.symlink(moved, directory / linked)
    with pytest.raises(RaisedFinding) as raised:
        still.get_pixel_array()
    found = raised.value
    assert (found.code, found.line) == ("asset.remote", 41)
    assert found.detail["file"] == "frames/0000.png"
