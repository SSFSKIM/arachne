"""Which frames the structural pass reads, and which objects a replay examines."""

from __future__ import annotations

from fractions import Fraction

import pytest
from arachne_compare import Group, choose_objects, sample


def test_the_seeded_share_is_pinned():
    # The same on every Python: sha256 of "<seed>:<key>:<i>", first 8 bytes, big-endian.
    assert sample(100, Fraction(30), [], [], seed=0, key="case") == (
        0,
        30,
        38,
        44,
        56,
        60,
        67,
        70,
        87,
        90,
    )
    assert sample(100, Fraction(30), [], [], seed=1, key="case") == (
        0,
        8,
        30,
        60,
        65,
        87,
        90,
        99,
    )


def test_the_first_and_last_frame_of_every_play_are_read():
    found = sample(
        100, Fraction(30), [(3, 17), (17, 18), (20, 20)], [], seed=0, key="case", rate=0
    )
    assert {3, 16, 17} <= set(found)
    assert {19, 20}.isdisjoint(found)  # an empty play has no frames


def test_one_frame_per_second_is_read():
    assert sample(100, Fraction(30), [], [], seed=0, key="case", rate=0) == (
        0,
        30,
        60,
        90,
    )
    assert sample(10, Fraction(5, 2), [], [], seed=0, key="case", rate=0) == (
        0,
        3,
        5,
        8,
    )


def test_every_flagged_frame_is_read():
    assert sample(50, Fraction(30), [], [7, 41], seed=0, key="case", rate=0) == (
        0,
        7,
        30,
        41,
    )


def test_frames_outside_the_sequence_are_refused():
    with pytest.raises(ValueError, match="outside"):
        sample(10, Fraction(30), [(0, 11)], [], seed=0, key="case")
    with pytest.raises(ValueError, match="outside"):
        sample(10, Fraction(30), [], [10], seed=0, key="case")


def group(box, pixels, cluster):
    return Group(box, pixels, 40, cluster)


BOXES = {f"o{k:02d}": (10 * k, 0, 10 * k + 8, 8) for k in range(20)}


def test_objects_come_one_sided_first_then_by_cluster_then_by_group_below():
    reference = dict(BOXES)
    candidate = {k: v for k, v in BOXES.items() if k != "o05"}
    candidate["new"] = (300, 300, 310, 310)
    groups = [
        group((101, 1, 103, 3), 4, False),  # meets o10
        group((150, 1, 160, 3), 30, True),  # meets o15
        group((60, 1, 72, 3), 9, False),  # meets o06 and o07
    ]
    chosen = choose_objects(
        reference, candidate, groups, cap=10, reserve=2, seed=0, key="k"
    )
    assert chosen[:6] == ["new", "o05", "o15", "o06", "o07", "o10"]
    assert len(chosen) == 10
    assert len(set(chosen)) == 10


def test_the_reserve_is_kept_for_the_seeded_sample():
    groups = [group((10 * k, 1, 10 * k + 2, 3), 5, False) for k in range(10)]
    chosen = choose_objects(BOXES, BOXES, groups, cap=6, reserve=2, seed=0, key="k")
    assert chosen[:4] == ["o00", "o01", "o02", "o03"]
    assert len(chosen) == 6
    assert set(chosen[4:]) <= {f"o{k}" for k in range(10, 20)}


def test_room_the_sample_leaves_goes_back_to_the_ranked_objects():
    few = {k: BOXES[k] for k in ("o00", "o01", "o02", "o03")}
    groups = [group((10 * k, 1, 10 * k + 2, 3), 5, False) for k in range(4)]
    assert choose_objects(few, few, groups, cap=4, reserve=2, seed=0, key="k") == [
        "o00",
        "o01",
        "o02",
        "o03",
    ]


def test_the_sample_of_the_others_is_seeded():
    first = choose_objects(BOXES, BOXES, [], cap=5, reserve=5, seed=0, key="k")
    assert first == choose_objects(BOXES, BOXES, [], cap=5, reserve=5, seed=0, key="k")
    assert first != choose_objects(BOXES, BOXES, [], cap=5, reserve=5, seed=1, key="k")
