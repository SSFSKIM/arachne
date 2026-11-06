"""Findings as data: their severity, folding of look-alikes, the caller's
time range and exemption selectors.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from arachne.lint.exempt import Exemptions
from arachne.lint.findings import Finding, FindingObject, Holder, fold, in_range
from arachne.timeline.source import SourceLine


def _object(
    oid: str,
    *,
    line: int | None = None,
    name: str | None = None,
    text: str | None = None,
    kind: str = "Dot",
    within: tuple[Holder, ...] = (),
) -> FindingObject:
    return FindingObject(
        id=oid,
        source_line=line,
        kind=kind,
        label_or_text=text if text is not None else name,
        bbox=(0.0, 0.0, 1.0, 1.0),
        screen_bbox=None,
        name=name,
        within=within,
    )


def _finding(*objects: FindingObject, code="overlap.text", t=Fraction(1)) -> Finding:
    return Finding(code, t, t, objects, message="planted")


# Severity


def test_a_finding_takes_its_code_s_severity_class_and_fix():
    found = _finding(_object("a.py:3#0"))
    assert found.severity == "error"
    assert found.cls == "Clearance"
    assert found.fix


def test_a_finding_may_be_lowered_to_a_note_and_never_raised():
    assert _finding(code="replay.diverged").severity == "error"
    lowered = Finding("replay.diverged", None, None, severity="note")
    assert lowered.severity == "note"
    with pytest.raises(ValueError, match="note"):
        Finding("motion.jump", None, None, severity="error")


def test_a_finding_carries_only_its_code_s_detail_fields():
    Finding("motion.jump", None, None, detail={"side": "after"})
    with pytest.raises(ValueError, match="colour"):
        Finding("motion.jump", None, None, detail={"colour": "red"})


def test_a_reserved_or_unknown_code_is_no_finding():
    with pytest.raises(KeyError):
        Finding("image.credit", None, None)


# Folding


def _label(i: int) -> FindingObject:
    return _object(f"s.py:{10 + i}#0", kind="Text", text=f"label {i}")


def _planet(i: int) -> FindingObject:
    return _object(f"s.py:{20 + i}#0", kind="Circle")


def test_folding_keeps_the_first_of_more_than_three_look_alikes():
    """A label on each of five planets, overlapping it, none of them named:
    one finding shown, four folded into it and counted.
    """
    alike = [_finding(_label(i), _planet(i), t=Fraction(i)) for i in range(5)]
    (kept,) = fold(alike)
    assert kept is not alike[0]
    assert kept.objects == alike[0].objects
    assert kept.folded == tuple(alike[1:])


@pytest.mark.parametrize("line", [None, 3])
def test_a_code_that_never_folds_keeps_every_look_alike(line):
    """Each jump is its own event at its own cut: none folds into another,
    of one creating line or none.
    """
    alike = [
        _finding(
            _object("s.py:3#0", kind="Dot", line=line),
            code="motion.jump",
            t=Fraction(i),
        )
        for i in range(5)
    ]
    assert fold(alike) == alike


def test_three_look_alikes_are_all_kept():
    alike = [_finding(_label(i), _planet(i), t=Fraction(i)) for i in range(3)]
    assert fold(alike) == alike


def test_look_alikes_fold_across_the_findings_between_them():
    """A finding of another code among them does not keep them apart: the
    fold stands where the first of them stood.
    """
    alike = [_finding(_label(i), _planet(i), t=Fraction(i)) for i in range(4)]
    other = _finding(_label(9), code="gap.text", t=Fraction(2, 3))
    kept, rest = fold([*alike[:2], other, *alike[2:]])
    assert kept.objects == alike[0].objects
    assert kept.folded == tuple(alike[1:])
    assert rest is other


def test_objects_the_scene_names_apart_do_not_fold():
    named = [
        _finding(_object(f"s.py:{i}#0", name=f"dot{i}"), t=Fraction(i))
        for i in range(4)
    ]
    assert fold(named) == named
    same = [
        _finding(_object(f"s.py:{i}#0", name="dot"), t=Fraction(i)) for i in range(4)
    ]
    assert len(fold(same)) == 1


def test_parts_of_one_group_fold_and_parts_of_two_do_not():
    one = (Holder("s.py:5#0", "planets", 5),)
    two = (Holder("s.py:6#0", "moons", 6),)
    grouped = [
        _finding(_object(f"s.py:7#{i}", kind="Dot", within=one)) for i in range(4)
    ]
    assert len(fold(grouped)) == 1
    mixed = [
        _finding(_object(f"s.py:7#{i}", kind="Dot", within=one if i % 2 else two))
        for i in range(4)
    ]
    assert fold(mixed) == mixed


# Time range


def test_a_range_keeps_findings_that_overlap_it():
    early = Finding("overlap.text", Fraction(0), Fraction(1))
    late = Finding("overlap.text", Fraction(3), Fraction(4))
    timeless = Finding("source.changed", None, None)
    window = (1.0, 2.0)
    assert in_range(early, window)
    assert not in_range(late, window)
    assert in_range(timeless, window)
    assert in_range(late, None)


def test_a_range_s_ends_are_the_times_they_name():
    """0.2 as a float is a hair above 1/5; a finding at 1/5 is at 0.2."""
    at_a_fifth = Finding("overlap.text", Fraction(1, 5), Fraction(1, 5))
    assert in_range(at_a_fifth, (0.2, 0.3))
    assert in_range(at_a_fifth, (0.1, 0.2))


# Exemptions


LEFT_DOT = _object("scene.py:12#0", line=12, name="left")
RIGHT_DOT = _object("scene.py:13#0", line=13, name="right")
GROUP = Holder("scene.py:14#0", "held", 14)
HELD_DOT = _object("scene.py:14#1", line=14, name="held[0]", within=(GROUP,))


@pytest.mark.parametrize(
    "selector",
    ["scene.py:12#0", "line:12", "name:left"],
)
def test_a_bare_selector_exempts_every_finding_with_its_object(selector):
    exemptions = Exemptions.parse([selector])
    assert exemptions.exempts(_finding(LEFT_DOT))
    assert exemptions.exempts(_finding(LEFT_DOT, RIGHT_DOT))
    assert not exemptions.exempts(_finding(RIGHT_DOT))


def test_a_selector_of_a_group_exempts_its_parts():
    for selector in ("scene.py:14#0", "name:held", "line:14"):
        assert Exemptions.parse([selector]).exempts(_finding(HELD_DOT))


def test_a_pair_exempts_only_a_finding_between_the_two():
    exemptions = Exemptions.parse([["name:right", "line:12"]])
    assert exemptions.exempts(_finding(LEFT_DOT, RIGHT_DOT))
    assert exemptions.exempts(_finding(RIGHT_DOT, LEFT_DOT))
    assert not exemptions.exempts(_finding(LEFT_DOT))
    assert not exemptions.exempts(_finding(LEFT_DOT, HELD_DOT))


def test_an_objectless_finding_is_never_exempted_by_a_selector():
    exemptions = Exemptions.parse(["line:12", ["line:12", "line:13"]])
    assert not exemptions.exempts(Finding("replay.diverged", None, None))


@pytest.mark.parametrize(
    "bad", ["line:twelve", "name:", "", ["a"], ["a", "b", "c"], [["a", "b"]], 3]
)
def test_a_malformed_selector_is_refused(bad):
    with pytest.raises(ValueError):
        Exemptions.parse([bad])


def test_named_parts_of_one_group_do_not_fold():
    """Four labels of one group, each with a name of its own, are four."""
    group = (Holder("s.py:1#0", "labels", 1),)
    named = [
        _finding(_object(f"s.py:{i}#0", kind="Text", name=f"label{i}", within=group))
        for i in range(4)
    ]
    assert fold(named) == named


# Folding by the lines that made the objects, within an event


def _member(name: str, i: int, line: int) -> FindingObject:
    return _object(f"s.py:{line}#{i}", line=line, name=f"{name}[{i}]", kind="Square")


def test_members_made_by_one_line_fold_by_line_pair_whatever_their_names():
    """A row made by one line over a row made by another: each pair named
    apart (``upper[i]``, ``lower[i]``), found at one instant with another
    pair among them, folds into one per line pair, counted.
    """
    rows = [
        _finding(_member("upper", i, 7), _member("lower", i, 8), code="contact.held")
        for i in range(5)
    ]
    rings = _finding(
        _object("s.py:9#0", line=9, name="ring"),
        _object("s.py:10#0", line=10, name="other"),
        code="contact.held",
    )
    kept, other = fold([*rows[:2], rings, *rows[2:]], timeline=[Fraction(0)])
    assert kept.objects == rows[0].objects
    assert kept.folded == tuple(rows[1:])
    assert other is rings


def test_objects_made_by_lines_of_their_own_stay_apart():
    pairs = [
        _finding(
            _object(f"s.py:{i}#0", line=i, name="dot"),
            _object("s.py:20#0", line=20, name="ring"),
        )
        for i in range(4)
    ]
    assert fold(pairs, timeline=[Fraction(0)]) == pairs


def test_look_alikes_fold_only_within_one_event():
    """Two of them in each of two plays are two and two: none folds."""
    alike = [
        _finding(_member("upper", i, 7), _member("lower", i, 8), t=Fraction(i))
        for i in range(4)
    ]
    timeline = [Fraction(0), Fraction(2)]
    assert fold(alike, timeline=timeline) == alike
    assert len(fold(alike, timeline=[Fraction(0)])) == 1


def test_a_folded_finding_holds_over_its_members_times():
    """The kept finding spans the earliest start and the latest end of the
    look-alikes it folds; each folded one keeps its own.
    """
    alike = [
        Finding(
            "contact.held",
            Fraction(i),
            Fraction(i) + Fraction(1, 2),
            (_member("upper", i, 7), _member("lower", i, 8)),
            message="planted",
        )
        for i in range(1, 5)
    ]
    (kept,) = fold(alike, timeline=[Fraction(0)])
    assert (kept.t_start, kept.t_end) == (Fraction(1), Fraction(9, 2))
    assert kept.objects == alike[0].objects
    assert kept.folded == tuple(alike[1:])
    latest = alike[-1]
    assert kept.t_start <= latest.t_start <= latest.t_end <= kept.t_end


def test_an_objectless_finding_folds_by_the_line_that_raised_it():
    """A check of points, objectless, folds with those its own line raised
    in the event, never with another line's; one raised at no line of the
    scene never folds.
    """

    def raised(line: int | None, t: Fraction) -> Finding:
        at = SourceLine("s.py", line) if line is not None else None
        return Finding("axes.outside", t, t, (), message="planted", line=at)

    found = [raised(12 + i % 2, Fraction(i, 10)) for i in range(8)]
    first, second = fold(found, timeline=[Fraction(0)])
    assert first.line == SourceLine("s.py", 12)
    assert first.folded == tuple(found[2::2])
    assert second.line == SourceLine("s.py", 13)
    assert second.folded == tuple(found[3::2])
    nowhere = [raised(None, Fraction(i, 10)) for i in range(4)]
    assert fold(nowhere, timeline=[Fraction(0)]) == nowhere


def test_objects_made_by_no_known_line_fold_by_kind_group_and_name():
    """With no creating line to go by, an object is told apart as before:
    by its kind, its outermost group and its own name.
    """
    named = [_finding(_object(f"o{i}", name=f"dot{i}")) for i in range(4)]
    assert fold(named, timeline=[Fraction(0)]) == named
    same = [_finding(_object(f"o{i}", name="dot")) for i in range(4)]
    assert len(fold(same, timeline=[Fraction(0)])) == 1
