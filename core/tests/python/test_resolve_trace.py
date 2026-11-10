"""The diagnostic resolve trace through the binding (``Stream._resolve_trace``): each pixel's
trace, re-composited with the resolves' own arithmetic restated here, gives back the pixel the
core drew. Asked for its reads (``reads=True``), it adds their keys and nothing else: every other
key, and every byte drawn, stays as it was.
"""

from __future__ import annotations

import arachne_core as ac
import numpy as np
import pytest

SAMPLES = 16
#: The keys of a trace pixel, and those ``decisions=True`` adds.
PLAIN = {"resolve", "background", "entries", "samples", "units", "emission"}
DECISIONS = {"pairs", "sample_ranked", "sample_pairs"}
#: The keys ``reads=True`` adds, and those it adds with ``decisions=True`` too.
READS = {"coverage8", "depths", "err", "samples_all"}
AGAINST = {"against", "flat", "relations"}


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


def fixture(name: str):
    """A fixture's stream after its last delta, and its size."""
    fixtures = {f[0]: f for f in ac.probe_fixtures()}
    _, width, height, deltas = fixtures[name]
    stream = ac.Stream()
    for delta in deltas:
        stream.apply(delta)
    return stream, width, height


@pytest.mark.parametrize("decisions", [False, True])
def test_reads_add_their_keys_and_change_nothing_else_the_trace_holds(decisions):
    stream, width, height = fixture("stroke_crossing_surface")
    plain = stream._resolve_trace(0, 0, width, height, decisions=decisions)
    read = stream._resolve_trace(0, 0, width, height, decisions=decisions, reads=True)
    assert plain.keys() == read.keys()
    keys = PLAIN | (DECISIONS if decisions else set())
    for xy, p in read.items():
        # Without reads, the trace's keys are today's; its entries keep their seven fields.
        assert set(plain[xy]) == keys, xy
        assert all(len(e) == 7 for e in p["entries"]), xy
        assert set(p) == set(plain[xy]) | READS | (AGAINST if decisions else set()), xy
        assert {k: v for k, v in p.items() if k in plain[xy]} == plain[xy], xy
        n = len(p["entries"])
        assert len(p["coverage8"]) == len(p["depths"]) == len(p["err"]) == n
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in p["coverage8"])
        assert all(len(row) == SAMPLES for row in p["depths"])
        assert len(p["samples_all"]) == SAMPLES
        for entry, other, depths, err in p.get("against", ()):
            assert entry != other
            assert {entry, other} <= set(range(n))
            assert len(depths) == SAMPLES
            assert isinstance(err, float)


def test_a_trace_with_its_reads_changes_no_byte_any_fixture_draws():
    # Every frame of every fixture, drawn by a stream never traced and by one traced with its
    # reads over the whole frame before each draw: the same bytes and resolve classes, with the
    # overlay and without.
    for name, width, height, deltas in ac.probe_fixtures():
        untraced, traced = ac.Stream(), ac.Stream()
        for i, delta in enumerate(deltas):
            untraced.apply(delta)
            traced.apply(delta)
            traced._resolve_trace(0, 0, width, height, decisions=True, reads=True)
            for got, want in zip(
                traced.draw_resolves(width, height),
                untraced.draw_resolves(width, height),
                strict=True,
            ):
                assert np.array_equal(got, want), (name, i)
            for overlay in (False, True):
                assert np.array_equal(
                    traced.draw(width, height, overlay=overlay),
                    untraced.draw(width, height, overlay=overlay),
                ), (name, i, overlay)


def test_coverage8_is_the_byte_the_coverage_resolve_blended_an_entry_alone_with():
    checked = 0
    for name in ("stroke_crossing_surface", "translucent_over_opaque", "paint_kinds"):
        stream, width, height = fixture(name)
        for xy, p in stream._resolve_trace(0, 0, width, height, reads=True).items():
            if p["resolve"] not in ("fast", "coverage"):
                continue
            for members, _colour, coverage in p["units"]:
                if len(members) == 1:
                    assert coverage == p["coverage8"][members[0]], (name, xy)
                    checked += 1
    assert checked > 100, checked


def test_samples_all_holds_every_entry_present_where_samples_drops_hidden_ones():
    hidden = 0
    for name in (
        "curve_behind_surface",
        "intersecting_surfaces",
        "stroke_crossing_surface",
    ):
        stream, width, height = fixture(name)
        for xy, p in stream._resolve_trace(0, 0, width, height, reads=True).items():
            for k in range(SAMPLES):
                present = [i for i, e in enumerate(p["entries"]) if e[6][k] != 0]
                assert sorted(p["samples_all"][k]) == present, (name, xy, k)
                assert set(p["samples"][k]) <= set(p["samples_all"][k]), (name, xy, k)
                # What ``samples`` lacks the opaque drop removed: behind an entry opaque and
                # covering the sub-sample whole.
                ranked = p["samples_all"][k]
                for place, i in enumerate(ranked):
                    if i in p["samples"][k]:
                        continue
                    hidden += 1
                    assert any(
                        p["entries"][j][5][3] == 255 and p["entries"][j][6][k] == 255
                        for j in ranked[:place]
                    ), (name, xy, k, i)
    assert hidden > 0


def test_a_reader_of_today_s_trace_reads_the_trace_with_its_reads_as_before():
    # A reader that unpacks each entry's seven fields and indexes the keys it knows, as the
    # trace's readers do, gets the same from a trace with its reads.
    def reader(trace):
        out = {}
        for xy, p in trace.items():
            entries = [
                (obj, part, layer, order, sub, tuple(colour), tuple(coverage))
                for obj, part, layer, order, sub, colour, coverage in p["entries"]
            ]
            out[xy] = (
                p["resolve"],
                entries,
                p["samples"],
                p["units"],
                p.get("pairs"),
                p.get("sample_ranked"),
                p.get("sample_pairs"),
            )
        return out

    stream, width, height = fixture("intersecting_surfaces")
    plain = stream._resolve_trace(0, 0, width, height, decisions=True)
    read = stream._resolve_trace(0, 0, width, height, decisions=True, reads=True)
    assert reader(read) == reader(plain)


def test_relations_are_each_pair_s_shared_orders_and_absent_on_a_flat_pixel():
    rows = flat = 0
    for name in ("intersecting_surfaces", "curve_behind_surface", "paint_kinds"):
        stream, width, height = fixture(name)
        for xy, p in stream._resolve_trace(
            0, 0, width, height, decisions=True, reads=True
        ).items():
            entries = p["entries"]

            def both(a, b, entries=entries):
                return sum(
                    1 << k
                    for k in range(SAMPLES)
                    if entries[a][6][k] and entries[b][6][k]
                )

            if p["flat"]:
                flat += 1
                assert p["relations"] == [], (name, xy)
                continue
            n = len(entries)
            wanted = [(a, b) for a in range(n) for b in range(a + 1, n) if both(a, b)]
            assert [tuple(r[:2]) for r in p["relations"]] == wanted, (name, xy)
            for a, b, covered, *masks in p["relations"]:
                assert covered == both(a, b), (name, xy, a, b)
                assert all(0 <= m < 1 << SAMPLES for m in masks)
                rows += 1
    assert rows > 100, rows
