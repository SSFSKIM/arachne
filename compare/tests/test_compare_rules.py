"""The visible-scale rule's edges, the edge rule, presence states, and what is refused."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from arachne_compare import (
    ArrayFrames,
    EdgeBudget,
    Footprint,
    ObjectMap,
    Thresholds,
    cheap,
    compare,
    structural,
)
from arachne_compare.pixels import object_edges
from compare_plants import handed, nothing_changed, single


def plain(height: int, width: int, value: int = 100) -> np.ndarray:
    return np.full((height, width, 3), value, np.uint8)


def with_blob(
    frame: np.ndarray, count: int, delta: int, *, first: int | None = None
) -> np.ndarray:
    """``count`` connected pixels changed by ``delta`` (the first by ``first``), row by row."""
    out = frame.copy()
    width = int(np.ceil(np.sqrt(count)))
    for k in range(count):
        y, x = 5 + k // width, 5 + k % width
        out[y, x] = frame[y, x] + (first if k == 0 and first is not None else delta)
    return out


def verdict_of(
    reference,
    candidate,
    mode,
    reference_map=None,
    candidate_map=None,
    thresholds=Thresholds(),
):
    return compare(
        single(reference),
        single(candidate),
        mode=mode,
        maps=handed(
            [reference_map] if reference_map else [],
            [candidate_map] if candidate_map else [],
        ),
        changed=nothing_changed,
        plays=[(0, 1)],
        fps=Fraction(30),
        seed=0,
        key="rule",
        thresholds=thresholds,
    )


@pytest.mark.parametrize(
    ("size", "under"),
    [((270, 480), 25), ((1080, 1920), 414)],
)
@pytest.mark.parametrize("mode", ["same", "cross"])
def test_a_cluster_covers_more_than_the_share_exactly(size, under, mode):
    base = plain(*size)
    assert verdict_of(base, with_blob(base, under, 40), mode).verdict == "pass"
    blocked = verdict_of(base, with_blob(base, under + 1, 40), mode)
    assert blocked.codes == {"gate.cluster"}
    assert blocked.findings[0].detail == {"pixels": under + 1, "peak": 40}


@pytest.mark.parametrize("mode", ["same", "cross"])
def test_a_cluster_holds_a_pixel_changed_by_more_than_16(mode):
    base = plain(270, 480)
    assert verdict_of(base, with_blob(base, 30, 16), mode).verdict == "pass"
    assert verdict_of(base, with_blob(base, 30, 16, first=17), mode).verdict == "block"


def test_in_same_mode_any_difference_makes_a_member():
    base = plain(270, 480)
    assert verdict_of(base, with_blob(base, 26, 1, first=17), "same").verdict == "block"


def test_in_cross_mode_a_member_changes_by_more_than_the_floor():
    base = plain(270, 480)
    assert verdict_of(base, with_blob(base, 26, 8, first=17), "cross").verdict == "pass"
    assert (
        verdict_of(base, with_blob(base, 26, 9, first=17), "cross").verdict == "block"
    )
    assert verdict_of(base, with_blob(base, 26, 7, first=17), "same").verdict == "block"


def test_clusters_are_eight_connected():
    base = plain(270, 480)
    diagonal = base.copy()
    for k in range(26):
        diagonal[10 + k, 10 + k] += 40
    assert verdict_of(base, diagonal, "same").codes == {"gate.cluster"}


def test_counts_hold_what_is_below_the_thresholds():
    base = plain(270, 480)
    candidate = with_blob(base, 20, 40)
    candidate[100, 100] += 3
    found = verdict_of(base, candidate, "same")
    assert found.verdict == "pass"
    assert found.counts["changed_pixels"] == 21
    assert found.counts["groups_below"] == 2
    assert found.counts["largest_below"] == 20


def full(height: int, width: int, box) -> Footprint:
    x0, y0, x1, y1 = box
    return Footprint(box, np.ones((y1 - y0, x1 - x0), bool))


def edge_mask(footprint: Footprint, size) -> np.ndarray:
    found = object_edges(footprint, size)
    out = np.zeros(size, bool)
    if found is not None:
        (x0, y0, x1, y1), mask = found
        out[y0:y1, x0:x1] = mask
    return out


def test_below_1080_an_edge_pixel_is_a_boundary_pixel_and_nothing_more():
    size = (270, 480)
    edges = edge_mask(full(*size, (100, 50, 140, 80)), size)
    ring = np.zeros(size, bool)
    ring[50:80, 100:140] = True
    ring[51:79, 101:139] = False
    assert np.array_equal(edges, ring)


def test_a_boundary_pixel_has_one_of_its_four_neighbours_outside():
    size = (270, 480)
    ys, xs = np.mgrid[0:10, 0:10]
    staircase = Footprint((100, 50, 110, 60), ys >= xs)
    edges = edge_mask(staircase, size)[50:60, 100:110]
    expected = (ys == xs) | ((ys >= xs) & ((xs == 0) | (ys == 9)))
    assert np.array_equal(edges, expected)
    # One step inside the diagonal only a corner touches the outside: no edge.
    assert not edges[5, 4]


@pytest.mark.parametrize(
    ("height", "width", "reach"), [(1080, 1920, 1), (2160, 3840, 2)]
)
def test_from_1080_an_edge_reaches_one_pixel_per_1080_of_height(height, width, reach):
    size = (height, width)
    edges = edge_mask(full(*size, (100, 50, 140, 80)), size)
    band = np.zeros(size, bool)
    band[50 - reach : 80 + reach, 100 - reach : 140 + reach] = True
    band[51 + reach : 79 - reach, 101 + reach : 139 - reach] = False
    assert np.array_equal(edges, band)


def test_a_clip_at_the_frames_border_is_no_edge():
    size = (270, 480)
    edges = edge_mask(full(*size, (0, 0, 20, 10)), size)
    assert edges[[0, 5, 9, 5], [5, 0, 5, 19]].tolist() == [False, False, True, True]


def test_a_colour_change_inside_one_footprint_is_no_edge():
    size = (270, 480)
    reference = plain(*size)
    candidate = reference.copy()
    candidate[60:70, 110:130] = 220  # inside a 40 x 30 object, away from its boundary
    footprint = full(*size, (100, 50, 140, 80))
    found = ObjectMap(size, {"panel": footprint}, complete=False)
    assert verdict_of(reference, candidate, "cross", found, found).codes == {
        "gate.cluster"
    }


def budget_case(differing: int):
    size = (270, 480)
    footprint = full(*size, (100, 50, 120, 60))  # 20 x 10: 56 boundary pixels
    reference = plain(*size)
    candidate = reference.copy()
    for k in range(differing):
        candidate[50, 100 + 2 * k] += 30  # apart, so they join no cluster
    found = ObjectMap(size, {"panel": footprint}, complete=False)
    thresholds = Thresholds(edge_budget=EdgeBudget(delta=16, share=0.1, floor_px=4))
    return verdict_of(reference, candidate, "cross", found, found, thresholds)


def test_an_object_over_its_edge_budget_is_flagged_and_one_under_it_is_not():
    # At most max(4, 0.1 * 56) = 5.6 of its edge pixels may differ by more than 16.
    under = budget_case(5)
    assert under.verdict == "pass"
    assert under.edges["panel"] == {"pixels": 56, "changed": 5, "strong": 5}
    over = budget_case(6)
    assert over.codes == {"gate.edge.over_budget"}
    assert over.findings[0].detail == {
        "edge_pixels": 56,
        "differing": 6,
        "allowed": pytest.approx(5.6),
    }


def test_without_a_budget_edge_pixels_are_counted_and_never_block():
    size = (270, 480)
    reference = plain(*size)
    candidate = reference.copy()
    candidate[50, 100:120] += 60
    found = ObjectMap(size, {"panel": full(*size, (100, 50, 120, 60))}, complete=False)
    result = verdict_of(reference, candidate, "cross", found, found)
    assert result.verdict == "pass"
    assert result.counts["edge_strong"] == 20


def maps_for(size, reference_objects, candidate_objects, *, complete=(False, False)):
    return (
        ObjectMap(size, reference_objects, complete=complete[0]),
        ObjectMap(size, candidate_objects, complete=complete[1]),
    )


def test_an_id_untested_on_one_side_is_never_read_as_missing():
    size = (270, 480)
    frame = plain(*size)
    ref_map, cand_map = maps_for(size, {"dot": full(*size, (10, 10, 12, 12))}, {})
    result = structural(frame, frame, ref_map, cand_map, mode="same")
    assert result.findings == ()
    assert result.counts["presence_untested"] == 1


def test_an_id_a_complete_map_does_not_list_is_empty():
    size = (270, 480)
    frame = plain(*size)
    ref_map, cand_map = maps_for(
        size, {"dot": full(*size, (10, 10, 12, 12))}, {}, complete=(False, True)
    )
    result = structural(frame, frame, ref_map, cand_map, mode="same")
    assert [f.code for f in result.findings] == ["gate.presence.missing"]


def test_a_footprint_known_only_by_its_box_is_untested_for_presence():
    size = (270, 480)
    frame = plain(*size)
    ref_map, cand_map = maps_for(
        size,
        {"dot": full(*size, (10, 10, 12, 12))},
        {"dot": Footprint((10, 10, 12, 12))},
    )
    result = structural(frame, frame, ref_map, cand_map, mode="same")
    assert result.findings == ()
    assert result.counts["presence_untested"] == 1


def test_a_part_gives_edges_and_is_not_tested_for_presence():
    size = (270, 480)
    frame = plain(*size)
    face = Footprint((10, 10, 20, 20), np.ones((10, 10), bool), parent="surface")
    ref_map, cand_map = maps_for(size, {"face-1": face}, {}, complete=(True, True))
    result = structural(frame, frame, ref_map, cand_map, mode="cross")
    assert result.findings == ()
    assert result.edge_counts["face-1"]["pixels"] == 36


def test_overlapping_footprints_and_a_map_that_is_not_complete_are_accepted():
    size = (270, 480)
    reference = plain(*size)
    objects = {
        "under": full(*size, (10, 10, 40, 40)),
        "over": full(*size, (20, 20, 50, 50)),
    }
    ref_map, cand_map = maps_for(size, objects, dict(objects))
    result = verdict_of(reference, reference.copy(), "cross", ref_map, cand_map)
    assert result.verdict == "pass"
    assert set(result.edges) == {"over", "under"}


def test_from_id_image_gives_a_complete_map():
    ids = np.full((6, 8), -1)
    ids[1:3, 2:5] = 0
    ids[4, 7] = 1
    found = ObjectMap.from_id_image(ids, ["a", "b"])
    assert found.complete
    assert found.objects["a"].box == (2, 1, 5, 3)
    assert found.objects["a"].mask.all()
    assert found.objects["b"].box == (7, 4, 8, 5)


def test_a_footprints_mask_must_fit_its_box():
    with pytest.raises(ValueError, match="mask"):
        Footprint((0, 0, 4, 4), np.ones((3, 4), bool))
    with pytest.raises(ValueError, match="outside"):
        ObjectMap((10, 10), {"a": Footprint((5, 5, 12, 6))}, complete=False)


def test_frames_of_different_sizes_are_refused():
    with pytest.raises(ValueError, match="sizes differ"):
        verdict_of(plain(270, 480), plain(271, 480), "same")


def test_frames_at_different_instants_are_refused():
    frames = [plain(27, 48)] * 3
    with pytest.raises(ValueError, match="instants"):
        cheap(
            ArrayFrames(frames, Fraction(30)),
            ArrayFrames(frames, Fraction(25)),
            mode="same",
        )


def test_sequences_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="lengths"):
        cheap(
            ArrayFrames([plain(27, 48)] * 3, 30),
            ArrayFrames([plain(27, 48)] * 2, 30),
            mode="same",
        )


def test_a_map_of_another_size_is_refused():
    frame = plain(270, 480)
    wrong = ObjectMap((270, 481), {}, complete=True)
    with pytest.raises(ValueError, match="another size"):
        verdict_of(frame, frame, "same", wrong, wrong)


def test_a_mode_other_than_the_two_is_refused():
    frame = plain(27, 48)
    with pytest.raises(ValueError, match="mode"):
        verdict_of(frame, frame, "loose")
