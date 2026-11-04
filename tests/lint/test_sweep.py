"""The instant rules: what one sweep finds in the frame shown at a lint
instant. Each planted defect comes back with its code, not held, at a time
inside the span it was planted in, and its twin is silent.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.lint import RULES, lint
from arachne.lint.rules import contact, frame_edge, occlusion, sweep, text_small
from arachne.lint.rules._geometry import box_gap, signed_to_box
from arachne.lint.rules._units import measured_units
from arachne.lint.serialise import notes, report

from . import defect_scenes
from .report_shape import assert_notes, assert_report

SCENES = Path(defect_scenes.__file__)
CLEARANCE = {
    "clip.frame_edge",
    "overlap.text",
    "gap.text",
    "contact.held",
    "occlusion.glyph",
}


@pytest.fixture
def run(session_of):
    """Lint a defect scene with one rule, by default as a sweep at the
    scene's end (the closing sweep); ``at=None`` sweeps every instant.
    """

    def lint_scene(name, rule=sweep, *, at="end", **terms):
        session = session_of(name, SCENES, frame_rate=15)
        when = [session.record.duration] if at == "end" else at
        return lint(session, rules=(rule.where(at=when, **terms),))

    return lint_scene


def _codes(result) -> list[str]:
    return [f.code for f in result.findings if f.code in CLEARANCE]


DEFECTS = {
    # scene: (code, planted span, terms)
    "CaptionPushedOut": ("clip.frame_edge", (2.0, 3.5), {}),
    "WordsOnWords": ("overlap.text", (1.0, 1.5), {}),
    "RowsTight": ("gap.text", (1.0, 1.5), {"text_gap": 0.3}),
    "SquareOverWords": ("occlusion.glyph", (1.0, 1.5), {}),
    "RingsCross": ("contact.held", (1.0, 1.5), {}),
}
TWINS = {
    "CaptionKeptIn": {},
    "WordsApart": {},
    "RowsLoose": {"text_gap": 0.3},
    "SquareBelowWords": {},
    "RingsApart": {},
}


@pytest.mark.parametrize("scene", sorted(DEFECTS))
def test_a_planted_defect_comes_back_in_its_span(run, scene):
    code, (low, high), terms = DEFECTS[scene]
    result = run(scene, **terms)
    found = [f for f in result.findings if f.code in CLEARANCE]
    assert [f.code for f in found] == [code]
    (finding,) = found
    assert finding.held is False
    assert finding.severity == "error"
    assert Fraction(low) <= finding.t_start <= finding.t_end <= Fraction(high)
    assert finding.objects
    assert not result.passed


@pytest.mark.parametrize("scene", sorted(TWINS))
def test_a_clean_twin_is_silent(run, scene):
    result = run(scene, **TWINS[scene])
    assert _codes(result) == []
    assert result.passed


def test_nothing_runs_unasked(session_of):
    assert RULES == ()
    session = session_of("WordsOnWords", SCENES, frame_rate=15)
    assert lint(session).findings == ()


def test_crowding_is_asked_only_when_a_text_gap_is_declared(run):
    assert _codes(run("RowsTight")) == []
    assert _codes(run("RowsTight", text_gap=0.3)) == ["gap.text"]


def test_a_rule_at_every_instant_reports_one_finding_over_its_span(run):
    result = run("WordsOnWords", contact, at=None)
    (finding,) = result.findings
    assert finding.code == "overlap.text"
    assert finding.t_end == Fraction(3, 2)
    assert finding.t_start < Fraction(1)


def test_each_rule_reports_its_own_codes(run):
    assert _codes(run("SquareOverWords", occlusion)) == ["occlusion.glyph"]
    assert _codes(run("SquareOverWords", contact)) == []
    assert _codes(run("CaptionPushedOut", contact)) == []
    assert _codes(run("CaptionPushedOut", frame_edge)) == ["clip.frame_edge"]
    assert _codes(run("RingsCross", frame_edge)) == []


# The frame edge -------------------------------------------------------------


def test_a_text_wholly_outside_is_reported_as_gone(run):
    (finding,) = run("CaptionGone").findings
    assert finding.code == "clip.frame_edge"
    assert finding.detail["edge"] == "right"
    assert "outside" in finding.message


def test_the_edge_is_asked_of_glyphs_not_their_plate(run):
    assert _codes(run("PlatedCaptionAtTheTop")) == []


def test_axis_numerals_may_run_past_the_edge(run):
    assert _codes(run("NumbersPastTheEdge", frame_edge)) == []


# Exemptions: a pair excuses a contact; a bare one mutes its object ------------


def test_a_pair_excuses_a_contact(run):
    pair = [["name:first", "name:second"]]
    assert _codes(run("WordsOnWords", exempt=pair)) == []


def test_a_pair_does_not_excuse_a_shape_painted_over_a_text(run):
    pair = [["name:plate", "name:words"]]
    assert _codes(run("SquareOverWords", exempt=pair)) == ["occlusion.glyph"]


def test_a_bare_exemption_mutes_its_object_occlusion_included(run):
    assert _codes(run("SquareOverWords", exempt=["name:plate"])) == []


def test_only_a_bare_exemption_excuses_the_frame_edge(run):
    pair = [["name:caption", "name:caption"]]
    assert _codes(run("CaptionPushedOut", exempt=pair)) == ["clip.frame_edge"]
    assert _codes(run("CaptionPushedOut", exempt=["name:caption"])) == []


# A filled thing of several leaves --------------------------------------------


def test_a_filled_group_is_held_to_its_leaves_not_their_box(run):
    assert _codes(run("WordBetweenTiles")) == []
    assert _codes(run("WordOnATile")) == ["overlap.text"]


# A stroke against glyph ink --------------------------------------------------


def test_a_stroke_through_glyph_ink_touches_the_text(run):
    assert _codes(run("LineThroughTheStem")) == ["overlap.text"]


def test_a_stroke_painted_through_glyph_ink_covers_it(run):
    (finding,) = run("LineOverTheStem").findings
    assert finding.code == "occlusion.glyph"
    assert finding.objects[0].label_or_text == "T"


def _units_at_end(session_of, scene):
    session = session_of(scene, SCENES, frame_rate=15)
    seen = []
    lint(
        session,
        rules=(lambda c: seen.extend(c.parts) or (),),
        at=[session.record.duration],
    )
    return measured_units(seen[-1])


def test_a_stroke_in_a_glyph_box_s_empty_corner_is_clear(run, session_of):
    assert _codes(run("LineByTheCorner")) == []
    # The line does cut the letter's box: only its ink is clear.
    line, letter = sorted(
        _units_at_end(session_of, "LineByTheCorner"), key=lambda u: u.text
    )
    assert float(signed_to_box(line.points, letter.outer).min()) < 0


def test_a_filled_group_s_word_sits_inside_the_group_s_box(session_of):
    tiles, word = _units_at_end(session_of, "WordBetweenTiles")
    assert tiles.unit.type == "Group"
    assert box_gap(tiles.outer, word.outer) < 0


# Plates and halos --------------------------------------------------------------


def test_a_text_on_a_card_touches_it_at_the_end_but_reads_clean_mid_scene(run):
    assert _codes(run("WordOnACard")) == ["overlap.text"]
    assert _codes(run("WordOnACard", mid_scene=True)) == []


def _between(result) -> set[tuple[str, frozenset[str]]]:
    """Each finding's code with the kinds of the two objects it names."""
    return {(f.code, frozenset(o.kind for o in f.objects)) for f in result.findings}


def test_a_stroke_hidden_under_the_card_a_text_sits_on_is_clear(run):
    crossed = ("overlap.text", frozenset({"Line", "Text"}))
    assert _between(run("LineUnderAWord")) == {crossed}
    # The card still touches both, as at any closing sweep; the line meets
    # the word only where the card hides it.
    assert _between(run("LineUnderACard")) == {
        ("contact.held", frozenset({"Line", "Rectangle"})),
        ("overlap.text", frozenset({"Text", "Rectangle"})),
    }
    assert crossed in _between(run("LineUnderAFaintCard"))


def test_a_stroke_hidden_under_a_text_s_own_plate_is_clear(run):
    assert _codes(run("LineUnderAPlatedWord")) == []


def test_a_text_over_its_own_halo_reads_clean_mid_scene_only(run):
    assert _codes(run("LineUnderAHaloedWord")) == ["overlap.text"]
    assert _codes(run("LineUnderAHaloedWord", mid_scene=True)) == []
    assert _codes(run("LineUnderAThinHalo", mid_scene=True)) == ["overlap.text"]


# Grids --------------------------------------------------------------------


def test_a_text_over_a_field_drawn_under_it_is_clear(run, session_of):
    assert _codes(run("WordOverAField")) == []
    assert _codes(run("WordOverAnArrow")) == ["overlap.text"]
    # The word does lie over one of the field's arrows.
    units = _units_at_end(session_of, "WordOverAField")
    (word,) = (u for u in units if u.text)
    assert any(u.grid and box_gap(u.outer, word.outer) < 0 for u in units)


# Under a ThreeDCamera ---------------------------------------------------------


def test_under_a_three_d_camera_the_whole_scene_is_linted_projected(run):
    (finding,) = run("FarWordInDepth").findings
    assert finding.code == "clip.frame_edge"
    assert finding.objects[0].label_or_text == "far"
    # Its scene box is inside the frame; the camera's zoom puts it out.
    assert finding.objects[0].bbox[2] < 7.0
    assert _codes(run("NearWordInDepth")) == []


# Small text ------------------------------------------------------------------


def test_small_text_is_a_note_kept_out_of_the_violations(run):
    result = run("TinyWord", text_small)
    (finding,) = result.findings
    assert finding.code == "text.small"
    assert finding.severity == "note"
    assert finding.objects[0].label_or_text == "tiny"
    assert finding.detail["height_px"] < 20
    assert result.passed
    assert report(result)["violations"] == []
    assert [n["code"] for n in notes(result)] == ["text.small"]
    assert_notes(notes(result))


@pytest.mark.parametrize("scene", sorted(DEFECTS))
def test_the_rules_findings_report_in_the_lint_s_shape(run, scene):
    code, _, terms = DEFECTS[scene]
    out = report(run(scene, **terms))
    assert_report(out)
    (violation,) = out["violations"]
    assert violation["code"] == code
    assert violation["class"] == "Clearance"
    assert all(o["source_line"] is not None for o in violation["objects"])
