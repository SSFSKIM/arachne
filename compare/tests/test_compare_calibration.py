"""The calibration pairs, and the PNG files they are written as."""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest
from arachne_compare import calibration
from arachne_compare.png import read_png, write_png
from compare_plants import busy, render


def gradient(height: int, width: int) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    frame = np.empty((height, width, 3), np.uint8)
    frame[..., 0] = xs * 255 // (width - 1)
    frame[..., 1] = ys * 255 // (height - 1)
    frame[..., 2] = 90
    return frame


@pytest.fixture(scope="module")
def bases():
    small, _ = render(busy())
    return gradient(1080, 1920), small


@pytest.fixture(scope="module")
def made(bases):
    return calibration.pairs(*bases)


EXPECTED = {
    "share-under": ("pass", "pass"),
    "share-over": ("block", "block"),
    "strength-under": ("pass", "pass"),
    "strength-over": ("block", "block"),
    "floor-under": ("block", "pass"),
    "floor-over": ("block", "block"),
    "edge-under": ("block", "pass"),
}


def test_each_threshold_has_a_pair_under_and_over_it_at_both_sizes(made):
    found = {p.name: (p.expected["same"], p.expected["cross"]) for p in made}
    for size in ("1920x1080", "480x270"):
        for name, verdicts in EXPECTED.items():
            assert found[f"{name}-{size}"] == verdicts
    assert found["edge-over-1920x1080"] == ("block", "block")


def test_the_three_small_cases_are_among_the_480x270_pairs(made):
    found = {p.name: p for p in made}
    seven = found["floor-under-7-480x270"]
    assert (seven.expected["same"], seven.expected["cross"]) == ("block", "pass")
    difference = np.abs(seven.candidate.astype(int) - seven.reference.astype(int)).max(
        axis=2
    )
    assert (difference > 0).sum() == 26
    assert sorted(set(difference[difference > 0].tolist())) == [7, 17]
    for name in ("edge-over-inside-480x270", "edge-over-outside-480x270"):
        assert found[name].expected == {"same": "block", "cross": "block"}
    outside = found["edge-over-outside-480x270"]
    changed = (
        np.abs(outside.candidate.astype(int) - outside.reference.astype(int)).max(
            axis=2
        )
        > 0
    )
    assert changed.sum() == 40
    assert not (changed & outside.edges()).any()


def test_the_pair_generator_asserts_each_verdict_with_the_comparator(made, monkeypatch):
    pair = made[0]
    wrong = calibration.Pair(
        **{**pair.__dict__, "expected": {"same": "block", "cross": "pass"}}
    )
    monkeypatch.setattr(calibration, "_pairs_at", lambda base, thresholds: [wrong])
    with pytest.raises(RuntimeError, match="the comparator says pass"):
        calibration.pairs(gradient(1080, 1920), gradient(270, 480))


def test_a_base_of_another_size_is_refused():
    with pytest.raises(ValueError, match="not 480 x 270"):
        calibration.pairs(gradient(270, 480), gradient(270, 480))


def test_the_command_writes_the_pairs_their_edge_masks_and_their_list(
    bases, tmp_path, capsys
):
    write_png(tmp_path / "base-1080.png", bases[0])
    write_png(tmp_path / "base-270.png", bases[1])
    out = tmp_path / "pairs"
    status = calibration.main(
        [str(tmp_path / "base-1080.png"), str(tmp_path / "base-270.png"), str(out)]
    )
    assert status == 0
    document = json.loads((out / "pairs.json").read_text())
    names = [p["name"] for p in document["pairs"]]
    assert len(names) == 18
    assert document["thresholds"]["cluster_share"] == 0.0002
    entry = next(
        p for p in document["pairs"] if p["name"] == "edge-over-outside-480x270"
    )
    assert entry["verdict"] == {"same": "block", "cross": "block"}
    assert (entry["width"], entry["height"]) == (480, 270)
    for kind in ("reference", "candidate", "edges"):
        assert read_png(out / entry["files"][kind]).shape == (270, 480, 3)
    assert "18 pairs written" in capsys.readouterr().out


def png_bytes(image: np.ndarray, filters: list[int]) -> bytes:
    """A PNG whose rows use the given filter methods, written by the specification."""
    height, width, channels = image.shape
    stride = width * channels
    flat = image.reshape(height, stride).astype(int)
    raw = bytearray()
    for y, method in enumerate(filters):
        row, up = flat[y], flat[y - 1] if y else np.zeros(stride, int)
        left = np.concatenate([np.zeros(channels, int), row[:-channels]])
        up_left = np.concatenate([np.zeros(channels, int), up[:-channels]])
        if method == 0:
            line = row
        elif method == 1:
            line = row - left
        elif method == 2:
            line = row - up
        elif method == 3:
            line = row - (left + up) // 2
        else:
            p = left + up - up_left
            pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
            guess = np.where(
                (pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, up_left)
            )
            line = row - guess
        raw += bytes([method]) + bytes((line % 256).astype(np.uint8))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    kind = {3: 2, 4: 6}[channels]
    header = struct.pack(">IIBBBBB", width, height, 8, kind, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("channels", [3, 4])
def test_every_row_filter_is_read(tmp_path, channels):
    image = (np.arange(5 * 7 * channels).reshape(5, 7, channels) * 37 % 256).astype(
        np.uint8
    )
    path = tmp_path / "filters.png"
    path.write_bytes(png_bytes(image, [0, 1, 2, 3, 4]))
    assert np.array_equal(read_png(path), image)


def test_a_written_png_reads_back(tmp_path):
    image = gradient(30, 40)
    write_png(tmp_path / "a.png", image)
    assert np.array_equal(read_png(tmp_path / "a.png"), image)
    mask = np.zeros((4, 5), bool)
    mask[1, 2] = True
    write_png(tmp_path / "m.png", mask)
    back = read_png(tmp_path / "m.png")
    assert back.shape == (4, 5, 3)
    assert back[1, 2].tolist() == [255, 255, 255]
    assert back.sum() == 765
