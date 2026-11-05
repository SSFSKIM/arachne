"""The temporal rules: what the hold watch keeps of what stayed wrong on
screen, and what a departure sweep finds in the frame a settled thing
leaves.

Each defect comes back with its code, ``held`` and its span; its twin is
silent. The spans are compared with the ones the reference checks report
for the same scenes on Manim CE 0.20.1, at a 15 fps clock: ``CE_SPANS``
(rounded there to a tenth of a second). The watch samples as CE's checks
do, phased from each play's start, but the two clocks differ by design: a
CE wait lasts a whole number of frames, so CE ends a half-second wait at
7/15 s. A span is held to CE's within a frame, that 1/30 s and CE's
rounding.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint.armed import ArmedRule
from arachne.lint.departures import departures
from arachne.lint.plays import played
from arachne.lint.serialise import report
from arachne.lint.watch import HOLD_PERSIST, hold_watch

from . import armed_scenes, run_counter, watch_scenes
from .conftest import line_of
from .report_shape import assert_report

SCENES = Path(watch_scenes.__file__)

#: How far a span may sit from CE's: a frame, CE's shorter half-second wait
#: and CE's rounding to a tenth.
NEAR = Fraction(1, 15) + Fraction(1, 30) + Fraction(1, 20)

#: What the reference checks report on CE for each scene (armed at the top
#: of construct): code, held, and the span, or the instant for a departure.
CE_SPANS = {
    "CaptionParkedAtEdge": ("clip.frame_edge", True, (1.0, 3.3)),
    "TokenDrifts": ("clip.frame_edge", True, (3.4, 5.0)),
    "TagOrphaned": ("label.left_behind", True, (2.5, 5.0)),
    "NoteRidesRail": ("overlap.text", True, (1.0, 2.7)),
    "PuckSlidesOut": ("clip.frame_edge", True, (2.9, 4.5)),
    "TagChasesDot": ("clip.frame_edge", True, (1.0, 2.5)),
    "FixedCaptionAtEdge": ("clip.frame_edge", True, (0.0, 2.0)),
    "ZoomedPastCaption": ("clip.frame_edge", True, (0.8, 3.0)),
    "WordSlipsToEdge": ("clip.frame_edge", True, (2.7, 3.7)),
    # Inside the frame as the script called the play; past the edge from its
    # first instant on.
    "PuckJumpsOutAtOnce": ("clip.frame_edge", True, (0.3, 2.0)),
    # The engine's record keeps states only from 1.2 s on: read from the
    # lint's forward run, which keeps them all.
    "StatesArmedLate": ("clip.frame_edge", True, (0.0, 1.2)),
    "SettledNoteFades": ("overlap.text", False, (1.0, 1.0)),
    "SettledNoteFadesInAGroup": ("overlap.text", False, (1.0, 1.0)),
    "SettledNoteRemoved": ("overlap.text", False, (1.0, 1.0)),
    "RailSnapsUnderNote": ("overlap.text", False, (1.2, 1.2)),
    "NoteRemovedAndReadded": ("overlap.text", False, (0.6, 0.6)),
    # Swept in the scene the script called the play on: a fade at rate 1
    # from its first instant, a play whose beginning takes the word out of
    # its panel, a fade too short to show a frame.
    "NoteFadesAtOnce": ("overlap.text", False, (0.6, 0.6)),
    "WordLeavesItsPanel": ("overlap.text", False, (0.6, 0.6)),
    "NoteFadesBetweenFrames": ("overlap.text", False, (0.6, 0.6)),
    # bring_to_back removes the word, then adds it again.
    "NoteReordered": ("overlap.text", False, (0.6, 0.6)),
}
#: Scenes the reference checks find nothing in.
CE_SILENT = [
    "CaptionParkedInside",
    "TokenStopsShort",
    "TagFollows",
    "NoteOnCard",
    "PuckCarriedOff",
    "NoteSweptAcross",
    # Judged on the scene as a play left it, finished and cleaned up.
    "RimWordSnapsIn",
    "BoxSnapsAway",
    "PuckRunsOffThenBack",
    # One authored group a play's own group took members of.
    "PanelPulsesTwo",
    # A removal departs only what it removes.
    "DotTakenFromPanel",
    # bring_to_front and remove_foreground_mobjects call no remove.
    "NoteBroughtToFront",
    "ForegroundNoteDropped",
]
#: The planted span each held defect must lie in.
PLANTED = {
    "CaptionParkedAtEdge": (0.0, 3.5),
    "TokenDrifts": (2.5, 5.0),
    "TagOrphaned": (2.0, 5.0),
    "NoteRidesRail": (0.0, 3.0),
    "PuckSlidesOut": (2.5, 4.5),
    "TagChasesDot": (0.5, 3.0),
    "FixedCaptionAtEdge": (0.0, 2.0),
    "ZoomedPastCaption": (0.5, 3.0),
    "WordSlipsToEdge": (2.6, 3.8),
    "PuckJumpsOutAtOnce": (0.0, 2.0),
    "StatesArmedLate": (0.0, 1.4),
}
HELD = [name for name, (_, held, _) in CE_SPANS.items() if held]
DEPARTED = [name for name, (_, held, _) in CE_SPANS.items() if not held]
CLEARANCE = {
    "clip.frame_edge",
    "overlap.text",
    "gap.text",
    "contact.held",
    "occlusion.glyph",
    "label.left_behind",
}


@pytest.fixture
def run(session_of):
    """Lint a watch scene with the temporal rules armed from its start."""

    def lint_scene(name, rules=None):
        session = session_of(name, SCENES, frame_rate=15)
        armed = (hold_watch.where(), departures.where()) if rules is None else rules
        return session, lint(session, rules=armed)

    return lint_scene


def _found(result):
    return [f for f in result.findings if f.code in CLEARANCE]


@pytest.mark.parametrize("name", HELD)
def test_a_defect_held_on_screen_is_reported_held_with_its_span(run, name):
    code, _, (start, end) = CE_SPANS[name]
    _, result = run(name, (hold_watch.where(),))
    [finding] = _found(result)
    assert finding.code == code
    assert finding.held is True
    assert finding.t_end - finding.t_start >= HOLD_PERSIST
    low, high = PLANTED[name]
    assert low <= finding.t_start
    assert finding.t_end <= high
    assert abs(finding.t_start - Fraction(start)) <= NEAR, float(finding.t_start)
    assert abs(finding.t_end - Fraction(end)) <= NEAR, float(finding.t_end)


def test_a_hold_just_over_a_second_is_held_where_ce_holds_it(run):
    # CE samples this drift at 3.40 s (in from the edge's margin since about
    # 3.32) and last at its wait's end, 4.467: held, 1.07 s. The engine's
    # samples are phased from each play's start as CE's are; its half-second
    # wait ends 1/30 s later than CE's 7 frames, so its samples fall 1/15 s
    # later (3.467, and the wait's end at 4.5).
    _, result = run("TokenDriftsBriefly", (hold_watch.where(),))
    [finding] = _found(result)
    assert (finding.code, finding.held) == ("clip.frame_edge", True)
    assert finding.t_start == Fraction(52, 15)
    assert finding.t_end == Fraction(9, 2)
    assert abs(finding.t_start - Fraction(34, 10)) <= Fraction(1, 15)
    assert abs(finding.t_end - Fraction(67, 15)) <= Fraction(1, 30)


def test_the_watch_reads_a_play_s_frames_at_the_rate_asked(run):
    # CE's pass at 8 of the 15 fps the clock keeps never samples the word at
    # the edge a second apart; reading every frame, it is held.
    _, every = run("WordSlipsToEdge", (hold_watch.where(),))
    assert [f.code for f in _found(every)] == ["clip.frame_edge"]
    _, sampled = run("WordSlipsToEdge", (hold_watch.where(sample_fps=8),))
    assert _found(sampled) == []


@pytest.mark.parametrize(
    ("name", "margin"),
    [
        ("CaptionHeldAtEdge", Fraction(0)),
        ("CaptionHeldAFrameMore", Fraction(1, 15)),
        ("CaptionHeldLonger", None),
    ],
)
def test_a_hold_decided_within_a_frame_says_so(run, name, margin):
    _, result = run(name, (hold_watch.where(),))
    [finding] = _found(result)
    assert finding.held is True
    if margin is None:
        assert "clock_margin" not in finding.detail
    else:
        assert finding.detail["clock_margin"] == pytest.approx(float(margin))


@pytest.mark.parametrize(
    ("name", "asked"),
    [("FixedCaptionAtEdge", "fixed"), ("ZoomedPastCaption", "cut here")],
)
def test_only_what_the_view_frames_is_asked(run, name, asked):
    # Under a ThreeDCamera only the fixed-in-frame layer is asked; a camera
    # away from home asks only about what its view's edge cuts.
    _, result = run(name, (hold_watch.where(),))
    assert [o.label_or_text for f in _found(result) for o in f.objects] == [asked]


@pytest.mark.parametrize("name", CE_SILENT)
def test_a_twin_is_silent(run, name):
    _, result = run(name)
    assert _found(result) == []


def test_the_held_findings_validate_as_a_report(run):
    _, result = run("CaptionParkedAtEdge")
    shape = report(result)
    assert_report(shape)
    [violation] = shape["violations"]
    assert violation["held"] is True
    assert violation["line"] is None


def test_the_watch_names_what_stayed_wrong(run):
    _, result = run("TagOrphaned", (hold_watch.where(),))
    [finding] = _found(result)
    tag, box = finding.objects
    assert (tag.kind, tag.label_or_text, box.kind) == ("Text", "B", "Square")
    assert finding.detail["referent"] == box.id
    assert finding.detail["distance"] >= 0.75


def test_a_rebuilt_tag_keeps_one_finding(run):
    session, result = run("TagChasesDot", (hold_watch.where(),))
    found = _found(result)
    assert [f.code for f in found] == ["clip.frame_edge"]
    # The tag was rebuilt at every frame it was seen at, and is one thing.
    assert len({o.id for f in found for o in f.objects}) == 1


def test_nothing_runs_unasked(run):
    _, result = run("CaptionParkedAtEdge", rules=())
    assert _found(result) == []


def test_an_exemption_mutes_a_held_finding(run, session_of):
    session = session_of("CaptionParkedAtEdge", SCENES, frame_rate=15)
    caption = next(
        f.objects[0].id for f in lint(session, rules=(hold_watch.where(),)).findings
    )
    muted = lint(session, rules=(hold_watch.where(exempt=[caption]),))
    assert _found(muted) == []


def test_the_watch_starts_where_it_was_armed(run, session_of):
    session = session_of("TagOrphaned", SCENES, frame_rate=15)
    late = lint(session, rules=(hold_watch.where(since=4.5),))
    assert _found(late) == []


def test_a_watch_armed_inside_a_play_watches_the_rest_of_it(session_of):
    # The caption stays clipped through the frozen two-second wait from 1 s
    # to 3 s: armed a quarter of the way in, the watch holds it from there.
    session = session_of("CaptionParkedAtEdge", SCENES, frame_rate=15)
    [whole] = _found(lint(session, rules=(hold_watch.where(),)))
    [late] = _found(lint(session, rules=(hold_watch.where(since=1.25),)))
    assert (late.code, late.held) == ("clip.frame_edge", True)
    assert late.t_start == Fraction(5, 4)
    assert late.t_end == whole.t_end
    assert [o.id for o in late.objects] == [o.id for o in whole.objects]


@pytest.mark.parametrize("rate", [0, -1, -0.5])
def test_a_sampling_rate_must_be_positive(rate):
    with pytest.raises(ValueError, match="sample_fps"):
        hold_watch.where(sample_fps=rate)


# Departures -----------------------------------------------------------------


@pytest.mark.parametrize("name", DEPARTED)
def test_a_settled_caption_that_departs_is_swept(run, name):
    code, _, (at, _) = CE_SPANS[name]
    _, result = run(name, (departures.where(),))
    [finding] = _found(result)
    assert (finding.code, finding.held) == (code, False)
    assert finding.t_start == finding.t_end
    assert abs(finding.t_start - Fraction(at)) <= NEAR
    assert {o.kind for o in finding.objects} == {"Line", "Text"}
    assert "clock_margin" not in finding.detail


@pytest.mark.parametrize(
    ("name", "tag"),
    [
        ("SettledNoteFades", "fade"),
        # A group made for the fade takes the word out of the scene's top
        # level as the play begins; the word was settled when it was called.
        ("SettledNoteFadesInAGroup", "group-fade"),
        ("SettledNoteRemoved", "remove"),
        # Swept in the scene the play left, before the script removed it.
        ("RailSnapsUnderNote", "remove-note"),
        # Put back at once, it was removed all the same.
        ("NoteRemovedAndReadded", "put-back"),
        ("NoteFadesAtOnce", "at-once"),
        ("WordLeavesItsPanel", "leaves"),
        ("NoteFadesBetweenFrames", "between"),
        ("NoteReordered", "reorder"),
    ],
)
def test_a_departure_names_the_line_that_made_it(run, name, tag):
    _, result = run(name, (departures.where(),))
    [finding] = _found(result)
    assert finding.line is not None
    assert finding.line.line == line_of(tag, SCENES)


def test_a_caption_swept_across_a_curve_in_transit_is_not(run):
    _, result = run("NoteSweptAcross", (departures.where(),))
    assert _found(result) == []


def test_a_departure_at_exactly_the_settling_wait_is_swept_and_tagged(run):
    # CE at a 15 fps clock ends this half-second wait at 7/15 s, short of
    # settling, and finds nothing; the engine's clock ends it at 1/2. The
    # finding says it was decided within a frame of the threshold.
    _, result = run("NoteFadesAtHalf", (departures.where(),))
    [finding] = _found(result)
    assert finding.code == "overlap.text"
    assert finding.detail["clock_margin"] == 0


@pytest.mark.parametrize(
    ("name", "margin"),
    [("NoteFadesAFrameLate", Fraction(1, 15)), ("NoteFadesWellAfter", None)],
)
def test_a_departure_a_frame_past_settling_still_says_so(run, name, margin):
    # CE ends these waits at 8 and 9 frames (0.533 and 0.6 s) and sweeps both.
    _, result = run(name, (departures.where(),))
    [finding] = _found(result)
    assert finding.code == "overlap.text"
    if margin is None:
        assert "clock_margin" not in finding.detail
    else:
        assert finding.detail["clock_margin"] == pytest.approx(float(margin))


def test_a_departure_s_exemption_excuses_it(run, session_of):
    session = session_of("SettledNoteFades", SCENES, frame_rate=15)
    [finding] = _found(lint(session, rules=(departures.where(),)))
    pair = [o.id for o in finding.objects]
    excused = lint(session, rules=(departures.where(exempt=[pair]),))
    assert _found(excused) == []


# The terms a scene arms ------------------------------------------------------

ARMED = Path(armed_scenes.__file__)

#: What the reference checks report on CE for each scene of
#: ``armed_scenes.py``, its bridge calls made as theirs (``arm`` as
#: ``guard_fades``, ``sweep`` as ``assert_scene_clear``): code, held, the
#: objects' kinds, and the span, or the instant for a departure.
ARMED_CE = {
    # The arming's pair excuses the departure.
    "ExemptedNoteFades": [],
    # The list the arming was handed names the second word's contact only
    # after the first was removed: the first is swept, the second excused.
    "ExemptionGrowsBetweenRemovals": [
        ("overlap.text", False, {"Line", "Text"}, (0.7, 0.7)),
    ],
    # A text_gap of 0.5 holds the first pair, swept as it departs, and the
    # second, on screen from the start; re-armed without one as the first
    # fade ends, the second departs unswept and its hold ends there: the
    # fade's end is held to the arming before it.
    "RearmedWithoutTextGap": [
        ("gap.text", True, {"Text"}, (0.0, 1.7)),
        ("gap.text", False, {"Text"}, (0.7, 0.7)),
    ],
    # Re-armed without a text_gap as the first wait ends.
    "HeldGapUntilRearmed": [("gap.text", True, {"Text"}, (0.0, 2.0))],
    # Excused by the closing sweep: every exemption is read at the end.
    "CaptionExcusedAtClose": [],
    # The group the arming excuses holds the caption at the end, not while
    # it was held at the edge.
    "CaptionExcusedThroughGroup": [],
    "CaptionLeavesTheGroup": [("clip.frame_edge", True, {"Text"}, (0.0, 2.3))],
    # A departure the guard checks adds the arming's list as it stands there
    # to what the end excuses: grown before a departure, the caption is
    # excused; grown with none after it, it is not.
    "ExemptionGrowsBeforeADeparture": [],
    "ExemptionGrowsWithNoDeparture": [
        ("clip.frame_edge", True, {"Text"}, (0.0, 2.3)),
    ],
    "ExemptionGrowsAfterTheDeparture": [
        ("clip.frame_edge", True, {"Text"}, (0.0, 2.3)),
    ],
    # A one-shot generator is read once, as the arming declares it: the
    # declaration the hold watch is held to keeps its entries, while the
    # value the arming was handed, read again at the departure, is spent.
    "GeneratorExempted": [],
    "GeneratorExemptedThenRemoved": [
        ("overlap.text", False, {"Line", "Text"}, (1.6, 1.6)),
    ],
    "ListExemptedThenRemoved": [],
}


@pytest.fixture
def run_armed(session_of):
    """Lint a scene that arms itself, the temporal rules asked from its start."""

    def lint_scene(name):
        session = session_of(name, ARMED, frame_rate=15)
        return session, lint(session, rules=(hold_watch.where(), departures.where()))

    return lint_scene


@pytest.mark.parametrize("name", list(ARMED_CE))
def test_the_rules_hold_a_scene_to_the_terms_it_armed(run_armed, name):
    _, result = run_armed(name)
    found = sorted(_found(result), key=lambda f: (not f.held, f.t_start, f.code))
    expected = sorted(ARMED_CE[name], key=lambda e: (not e[1], e[3][0], e[0]))
    assert len(found) == len(expected), [(f.code, f.held) for f in found]
    for finding, (code, held, kinds, (start, end)) in zip(found, expected, strict=True):
        assert (finding.code, finding.held) == (code, held)
        assert {o.kind for o in finding.objects} == kinds
        assert abs(finding.t_start - Fraction(start)) <= NEAR
        assert abs(finding.t_end - Fraction(end)) <= NEAR


def test_an_unarmed_scene_s_declaration_holds_the_rules_asked(run_armed):
    # Its twin, declaring nothing, is held and swept.
    _, plain = run_armed("UndeclaredNoteFades")
    assert sorted((f.code, f.held) for f in _found(plain)) == [
        ("overlap.text", False),
        ("overlap.text", True),
    ]
    _, result = run_armed("DeclaredNoteFades")
    assert _found(result) == []


def test_the_first_removal_is_swept_at_its_line(run_armed):
    _, result = run_armed("ExemptionGrowsBetweenRemovals")
    [finding] = _found(result)
    assert finding.line.line == line_of("remove-first", ARMED)
    assert "first" in {o.label_or_text for o in finding.objects}


def test_a_scene_s_arming_keeps_the_states_the_rules_read(session_of):
    # Every execution of the scene counts itself: the session's, and any
    # forward run a lint makes.
    run_counter.executions = 0
    session = session_of("CountedArmedScene", ARMED, frame_rate=15)
    assert run_counter.executions == 1
    assert session.record.states is not None
    result = lint(session, rules=(hold_watch.where(), departures.where()))
    assert "overlap.text" in {f.code for f in _found(result)}
    assert run_counter.executions == 1


# The states the rules read --------------------------------------------------


def _execution_seen(session, since=0):
    seen = []

    def watch(context, rule):
        seen.append(context.frames.execution)
        return []

    lint(session, rules=(ArmedRule(watch).where(since=since),))
    return seen[0]


def test_a_record_keeping_states_only_from_later_is_not_read_from_earlier(
    session_of,
):
    session = session_of("StatesArmedLate", SCENES, frame_rate=15)
    assert session.record.states.since == 1
    early = _execution_seen(session, since=0)
    assert early is not session.record
    assert early.states.since == 0
    # Armed at 1.2 s, the record keeps every state from then on.
    assert _execution_seen(session, since=1.2) is session.record


def test_a_lint_s_forward_run_keeps_every_play_s_state(session_of):
    session = session_of("SettledNoteRemoved", SCENES, frame_rate=15)
    assert session.record.states is None
    execution = _execution_seen(session)
    assert execution is not session.record
    assert set(execution.states.exits) == set(range(len(execution.plays)))
    [removal] = execution.states.removals
    assert removal.source_line.line == line_of("remove", SCENES)


def test_a_record_that_keeps_its_states_is_read_itself(session_of):
    armed = Path(__file__).parents[1] / "timeline" / "state_scenes.py"
    session = session_of("FadesAndRemoves", armed, frame_rate=15)
    assert _execution_seen(session) is session.record


# The play accessor ----------------------------------------------------------


def test_each_play_s_leaf_animations_say_what_they_do(session_of):
    session = session_of("SettledNoteFades", SCENES, frame_rate=15)
    record = session.record
    kinds = [
        [type(p.animation).__name__ for p in played(play)] for play in record.plays
    ]
    assert kinds == [["Wait"], ["FadeOut"], ["Wait"]]
    [fade] = played(record.plays[1])
    assert (fade.remover, fade.departs, fade.introducer) == (True, True, False)
    assert type(fade.mobject).__name__ == "Text"


def test_a_group_s_members_are_its_leaves(session_of):
    session = session_of("TagOrphaned", SCENES, frame_rate=15)
    first = played(session.record.plays[0])
    assert [type(p.animation).__name__ for p in first] == ["FadeIn", "FadeIn"]
    assert all(p.introducer for p in first)
    [move] = played(session.record.plays[2])
    assert not move.departs
    assert move.target is not None
