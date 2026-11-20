"""The manifest of a media asset: the kinds ``still``, ``clip`` and
``sprite``, the member ``media``, the frame digest, and what the reader
refuses as ``asset.manifest_invalid``.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import pytest

from arachne import Rig
from arachne.assets import load_asset
from arachne.assets.manifest import Extent, FrameSpec, Size
from arachne.assets.reader import MEDIA_CHECKS, checks_for, read_asset
from arachne.lint.raised import RaisedFinding
from arachne.mobject.media.frames import decode, frame_digest

from .conftest import copied, edited


def _invalid(directory) -> dict:
    with pytest.raises(RaisedFinding) as raised:
        read_asset(directory)
    assert raised.value.code == "asset.manifest_invalid", str(raised.value)
    return dict(raised.value.detail)


def test_the_three_kinds_load_with_their_frames():
    clip = load_asset("fixture:numbered-clip")
    assert clip.kind == "clip"
    media = clip.manifest.media
    assert media.pixels == (96, 54)
    assert media.alpha is True
    assert media.rate == Fraction(24)
    assert len(media.frames) == 48
    assert media.frames[0] == FrameSpec("frames/0000.png", media.frames[0].digest)
    assert clip.manifest.extent == Extent(3.2, 1.8)
    sprite = load_asset("fixture:stepper").manifest
    assert (sprite.kind, sprite.media.rate, len(sprite.media.frames)) == (
        "sprite",
        12,
        8,
    )
    still = load_asset("fixture:swatch").manifest
    assert (still.kind, still.media.rate, len(still.media.frames)) == ("still", None, 1)
    # A media asset has no canvas and no parts.
    assert not isinstance(still.extent, Size)
    assert still.parts == still.slots == still.parameters == still.anchors == ()
    assert clip.manifest.files() == tuple(f"frames/{i:04d}.png" for i in range(48))


def test_a_media_asset_holds_the_frame_checks():
    for name in ("numbered-clip", "stepper", "swatch"):
        assert checks_for(load_asset(f"fixture:{name}").manifest) == MEDIA_CHECKS
    assert MEDIA_CHECKS == ("frames_exact", "frames_uniform")


def test_the_frame_digest_hashes_the_size_then_the_pixels_row_by_row():
    pixels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    expected = hashlib.sha256(
        (3).to_bytes(4, "big") + (2).to_bytes(4, "big") + pixels.tobytes()
    ).hexdigest()
    assert frame_digest(pixels) == f"sha256:{expected}"
    # Another shape of the same bytes is another frame.
    assert frame_digest(pixels.reshape(3, 2, 4)) != frame_digest(pixels)


def test_every_listed_digest_is_the_frame_digest_of_its_file():
    clip = load_asset("fixture:numbered-clip")
    for frame in clip.manifest.media.frames:
        data = (clip.directory / frame.file).read_bytes()
        assert frame_digest(decode(data)) == frame.digest


def test_a_canvas_on_a_media_asset_is_refused(tmp_path):
    directory = edited(
        copied(tmp_path, "swatch"), lambda d: d["size"].update(origin=[0, 0])
    )
    assert _invalid(directory)["member"] == "size.origin"


@pytest.mark.parametrize(
    ("name", "edit", "member"),
    [
        ("swatch", lambda d: d.pop("media"), "media"),
        ("swatch", lambda d: d["media"].update(pixels=[64]), "media.pixels"),
        ("swatch", lambda d: d["media"].update(pixels=[0, 64]), "media.pixels"),
        ("swatch", lambda d: d["media"].update(alpha=1), "media.alpha"),
        ("swatch", lambda d: d["size"].update(width=0), "size.width"),
        ("numbered-clip", lambda d: d["media"].pop("rate"), "media.rate"),
        ("numbered-clip", lambda d: d["media"].update(rate=[24, 0]), "media.rate"),
        ("numbered-clip", lambda d: d["media"].update(rate=[24]), "media.rate"),
        ("swatch", lambda d: d["media"].update(rate=[1, 1]), "media.rate"),
        # A sprite's rate divides 60, so each drawing changes on a tick.
        ("stepper", lambda d: d["media"].update(rate=[24, 1]), "media.rate"),
        ("stepper", lambda d: d["media"].update(rate=[25, 2]), "media.rate"),
        # A frame list of the wrong length.
        ("numbered-clip", lambda d: d["media"].update(frames=[]), "media.frames"),
        (
            "swatch",
            lambda d: d["media"]["frames"].append(d["media"]["frames"][0]),
            "media.frames",
        ),
        (
            "numbered-clip",
            lambda d: d["media"]["frames"][1].update(digest="sha256:AB"),
            "media.frames[1].digest",
        ),
        (
            "numbered-clip",
            lambda d: d["media"]["frames"][1].update(file="frames/0000.png"),
            "media.frames[1].digest",
        ),
    ],
    ids=[
        "no-media",
        "one-dimension",
        "no-width",
        "alpha-not-boolean",
        "no-width-in-scene",
        "clip-without-rate",
        "zero-denominator",
        "rate-one-number",
        "still-with-rate",
        "sprite-rate-not-dividing-60",
        "sprite-rate-not-whole",
        "clip-without-frames",
        "still-with-two-frames",
        "malformed-digest",
        "one-file-two-digests",
    ],
)
def test_a_media_manifest_that_contradicts_itself_is_refused(
    tmp_path, name, edit, member
):
    assert _invalid(edited(copied(tmp_path, name), edit))["member"] == member


def test_a_sprite_whose_rate_divides_60_in_another_spelling_loads(tmp_path):
    directory = edited(
        copied(tmp_path, "stepper"), lambda d: d["media"].update(rate=[24, 2])
    )
    assert read_asset(directory).manifest.media.rate == 12


def test_an_unknown_minor_loads_and_an_unknown_requirement_is_refused(tmp_path):
    later = edited(
        copied(tmp_path, "swatch", "later"),
        lambda d: d.update(manifest_version="1.7", future_member={"x": 1}),
    )
    assert read_asset(later).manifest.manifest_version == "1.7"
    asks = edited(
        copied(tmp_path, "swatch", "asks"), lambda d: d.update(requires=["stereo"])
    )
    assert _invalid(asks)["member"] == "requires"


def test_a_rig_of_a_media_asset_is_a_plain_error():
    with pytest.raises(ValueError, match="no canvas"):
        Rig("fixture:numbered-clip")
