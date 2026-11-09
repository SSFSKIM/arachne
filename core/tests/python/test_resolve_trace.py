"""The diagnostic resolve trace through the binding (``Stream._resolve_trace``): each pixel's
trace, re-composited with the resolves' own arithmetic restated here, gives back the pixel the
core drew.
"""

from __future__ import annotations

import arachne_core as ac
import pytest

SAMPLES = 16


def mul(a: int, b: int) -> int:
    t = a * b + 0x80
    return ((t >> 8) + t) >> 8


def mul_7f(a: int, b: int) -> int:
    t = a * b + 0x7F
    return (t + (t >> 8)) >> 8


def blend(src, coverage: int, dst):
    """``composite::blend``: one premultiplied source over a pixel at an 8-bit coverage."""
    if src[3] == 255:
        if coverage == 255:
            return tuple(src)
        keep = 255 - coverage
        return tuple(
            min(mul_7f(s, coverage) + mul_7f(d, keep), 255)
            for s, d in zip(src, dst, strict=True)
        )
    if coverage != 255:
        src = tuple(mul(c, coverage) for c in src)
    keep = 255 - src[3]
    return tuple(min(s + mul(d, keep), 255) for s, d in zip(src, dst, strict=True))


def recomposite(p: dict):
    """The pixel a trace composites to (no emitted light)."""
    if p["resolve"] == "background":
        return tuple(p["background"])
    if p["resolve"] in ("fast", "coverage"):
        px = tuple(p["background"])
        for _members, colour, coverage in p["units"]:
            px = blend(colour, coverage, px)
        return px
    total = [0, 0, 0, 0]
    for k in range(SAMPLES):
        px = tuple(p["background"])
        for i in reversed(p["samples"][k]):
            entry = p["entries"][i]
            px = blend(entry[5], entry[6][k], px)
        total = [t + c for t, c in zip(total, px, strict=True)]
    return tuple((t + SAMPLES // 2) // SAMPLES for t in total)


@pytest.mark.parametrize("name", ["intersecting_surfaces", "stroke_crossing_surface"])
def test_the_trace_recomposites_to_the_frame_drawn_without_the_overlay(name):
    fixtures = {f[0]: f for f in ac.probe_fixtures()}
    _, width, height, deltas = fixtures[name]
    stream = ac.Stream()
    for delta in deltas:
        stream.apply(delta)
    rgba = stream.draw(width, height, overlay=False)
    traces = stream._resolve_trace(0, 0, width, height)
    assert len(traces) == width * height
    seen = set()
    for (x, y), p in traces.items():
        assert p["emission"] is None
        seen.add(p["resolve"])
        assert recomposite(p) == tuple(int(c) for c in rgba[y, x]), (x, y, p)
    assert {"coverage", "sample"} <= seen


def test_an_entry_carries_its_leaf_layer_order_colour_and_coverage():
    fixtures = {f[0]: f for f in ac.probe_fixtures()}
    _, width, height, deltas = fixtures["stroke_crossing_surface"]
    stream = ac.Stream()
    for delta in deltas:
        stream.apply(delta)
    traces = stream._resolve_trace(0, 0, width, height)
    layers = set()
    for p in traces.values():
        for obj, part, layer, order, sub, colour, coverage in p["entries"]:
            assert all(isinstance(v, int) for v in (obj, part, order, sub))
            assert len(colour) == 4
            assert len(coverage) == SAMPLES
            layers.add(layer)
        assert len(p["samples"]) == SAMPLES
    assert {"stroke", "surface"} <= layers


def test_asked_for_a_trace_carries_the_pairwise_orders_its_ranks_read():
    fixtures = {f[0]: f for f in ac.probe_fixtures()}
    _, width, height, deltas = fixtures["intersecting_surfaces"]
    stream = ac.Stream()
    for delta in deltas:
        stream.apply(delta)
    plain = stream._resolve_trace(0, 0, width, height)
    traces = stream._resolve_trace(0, 0, width, height, decisions=True)
    kinds = set()
    for xy, p in traces.items():
        assert "pairs" not in plain[xy]
        assert {k: v for k, v in p.items() if k in plain[xy]} == plain[xy]
        n = len(p["entries"])
        assert len(p["pairs"]) == n * (n - 1) // 2
        for a, b, front, kind in p["pairs"]:
            assert a < b
            assert front in (a, b, None)
            kinds.add(kind)
        if n > 1:
            assert len(p["sample_pairs"]) == SAMPLES
            assert isinstance(p["sample_ranked"], bool)
        for pairs in p["sample_pairs"]:
            for a, b, front, kind in pairs:
                assert front in (a, b)
                assert kind in ("depth", "tie", "paint")
    assert {"depth", "paint"} <= kinds
