"""The motion rule on groups ``become`` rebuilds between plays: a child that
took another's item, or was born at the cut, is no jump; a child that stands
where none stood, in a group that gained no child, still is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint.motion import motion_jump

SCENES = Path(__file__).parent / "rebuilt_scenes.py"


@pytest.fixture
def jumps_of(session_of):
    """The ``motion.jump`` findings of a scene of ``rebuilt_scenes.py``."""

    def find(name: str):
        session = session_of(name, SCENES, frame_rate=15)
        result = lint(session, rules=(motion_jump,))
        return [f for f in result.findings if f.code == "motion.jump"]

    return find


@pytest.mark.parametrize("name", ["StreamGrows", "StreamGrowsByItsOwnBecome"])
def test_a_stream_that_gains_a_particle_hands_its_items_on_without_a_jump(
    jumps_of, name
):
    """Each old child takes the item of the child after it, or the newcomer's
    at the source: the one that lands where no child stood is the born one,
    and the group's count rose by as many.
    """
    assert jumps_of(name) == []


def test_a_child_that_takes_an_unlike_sibling_s_item_is_no_jump(jumps_of):
    """Landing where a child of the rebuilt group stood passes whatever its
    size: the arrows here are no look-alikes of each other.
    """
    assert jumps_of("StreamOfUnlikeArrowsGrows") == []


def _newcomer_in_a_held_count(jumps_of, name):
    (jump,) = jumps_of(name)
    (arrow,) = jump.objects
    assert arrow.kind == "Arrow"
    # The child that held the youngest arrow now holds the newcomer, at the
    # source: no child stood there, and the count gave it no birth.
    assert jump.detail["to"][0] == pytest.approx(-4.8, abs=1e-6)
    assert jump.detail["distance"] == pytest.approx(3.69, abs=0.01)


def test_a_stream_that_turns_over_still_jumps(jumps_of):
    """One particle leaves as another spawns: the count holds, so the child
    standing at the source is no birth.
    """
    _newcomer_in_a_held_count(jumps_of, "StreamTurnsOver")


def test_a_vacated_slot_refilled_at_the_cut_still_jumps(jumps_of):
    """Known limit: a child left the list within the play, and ``become``
    kept its slot as padding, so the count the newcomer refills at the cut
    did not rise: its birth is read as a jump.
    """
    _newcomer_in_a_held_count(jumps_of, "StreamRefillsAVacatedSlot")


def test_a_rebuilt_group_that_teleports_whole_still_jumps(jumps_of):
    (jump,) = jumps_of("RebuiltGroupTeleports")
    assert jump.objects[0].kind == "VGroup"
    assert jump.detail["distance"] == pytest.approx(3.0, abs=1e-6)


def test_a_rig_that_snaps_back_still_jumps(jumps_of):
    """Every outer atom stands where no child stood, and the rig gained no
    child: each is the same atom set back.
    """
    found = jumps_of("RigSnapsBack")
    assert [f.objects[0].kind for f in found] == ["Dot"] * 3
    for jump in found:
        assert jump.detail["distance"] == pytest.approx(0.959, abs=1e-3)


def test_one_atom_snapped_back_while_the_rest_hold_still_jumps(jumps_of):
    (jump,) = jumps_of("OneAtomSnapsBack")
    assert jump.objects[0].kind == "Dot"
    assert jump.detail["distance"] == pytest.approx(1.6, abs=1e-6)


def test_a_group_that_gains_one_child_as_another_snaps_back(jumps_of):
    """Two children stand where none stood, and the group gained one: the
    newcomer, farthest from every place, is the birth, and the atom set back
    is the jump.
    """
    (jump,) = jumps_of("GrowsAsOneSnapsBack")
    assert jump.objects[0].kind == "Dot"
    assert jump.detail["from"] == pytest.approx([0.6, -0.8], abs=1e-6)
    assert jump.detail["to"] == pytest.approx([-1.0, -0.8], abs=1e-6)


def test_a_rig_turned_by_its_spacing_is_a_known_miss(jumps_of):
    """Known limit (a permutation coincidence): every atom set back lands
    where a sibling stood. A source-aware rule is the open fix.
    """
    assert jumps_of("RigTurnsByItsSpacing") == []


def test_a_tie_among_births_goes_to_the_child_first_seen(jumps_of):
    """The second and the padded child land on one place, as far from every
    place: the second, seen first, is the birth, though the padded child
    stands before it in the list, and the padded child is the jump, from the
    place it took at the first cut.
    """
    (jump,) = jumps_of("TieBrokenByFirstSight")
    assert jump.detail["boundary"] == {"from_play": 1, "to_play": 2}
    assert jump.detail["from"][0] == pytest.approx(1.2, abs=0.01)
    assert jump.detail["to"] == pytest.approx([0.2, 3.0], abs=0.01)


def test_a_group_nothing_rebuilds_is_judged_as_before(jumps_of):
    (jump,) = jumps_of("PlainGroupPutBack")
    assert jump.objects[0].kind == "Arrow"
    assert jump.detail["distance"] == pytest.approx(4.0, abs=1e-6)
