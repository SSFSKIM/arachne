"""The lint over a session: rules read the parts at each lint instant, the
session's divergences and staleness become findings, the caller's range
and exemptions filter them, look-alikes fold, and the result is data,
serialised for the machine-facing report.
"""

from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path

import pytest

from arachne.lint import Finding, LintResult, lint
from arachne.lint.instants import lint_instants
from arachne.lint.serialise import TARGET_HEIGHT, notes, report
from arachne.timeline.execution import Divergence

from ..timeline import camera_scenes
from .conftest import line_of
from .report_shape import assert_notes, assert_report

CAMERA_SCENES = Path(camera_scenes.__file__)


def _named(name: str):
    """A rule reporting the part named ``name`` at every lint instant it is
    in, each instant its own finding of ``overlap.text``.
    """

    def rule(context):
        for parts in context.parts:
            for part in parts.parts:
                if part.name == name:
                    yield Finding(
                        "overlap.text",
                        parts.t,
                        parts.t,
                        (part.as_object(parts.view),),
                        message=f"{name} planted",
                        line=part.source_line,
                    )

    return rule


def test_findings_are_data_and_the_lint_returns(session_of):
    session = session_of("Pair", frame_rate=15)
    result = lint(session, rules=(_named("left"),))
    assert isinstance(result, LintResult)
    assert result.findings
    assert not result.passed
    assert {f.code for f in result.findings} == {"overlap.text"}


def test_a_rule_reads_every_lint_instant_once(session_of):
    session = session_of("Pair", frame_rate=15)
    seen = []

    def count(context):
        seen.append([p.t for p in context.parts])
        assert [i.t for i in context.instants] == seen[-1]
        return ()

    result = lint(session, rules=(count,))
    assert seen == [[i.t for i in lint_instants(session.record, 4)]]
    assert result.passed
    assert result.instants == lint_instants(session.record, 4)


def test_divergences_are_replay_diverged_findings(session_of):
    """A probe's divergence is a note (the engine corrected itself); an
    execution's is an error (the render may not show what was checked).
    Each spans its play, or the whole scene when it names none.
    """
    session = session_of("Pair", frame_rate=15)
    record = session.record
    play = record.plays[0]
    session._keep(Divergence("probe", 0, play.source_line, "restored apart"))
    session._keep(Divergence("execution", None, None, "the final state differs"))
    found = {f.severity: f for f in lint(session).findings}
    probe, execution = found["note"], found["error"]
    for finding in (probe, execution):
        assert finding.code == "replay.diverged"
        assert finding.cls == "ReplayDiverged"
        assert finding.objects == ()
    assert (probe.t_start, probe.t_end) == (play.start, play.end)
    assert probe.line == play.source_line
    assert probe.detail["kind"] == "probe"
    assert probe.detail["play"] == 0
    assert "restored apart" in probe.message
    assert (execution.t_start, execution.t_end) == (0, record.duration)
    assert execution.line is None
    assert execution.detail["kind"] == "execution"


def test_a_stale_session_is_a_source_changed_note(session_of):
    session = session_of("Pair", frame_rate=15)
    assert not any(f.code == "source.changed" for f in lint(session).findings)
    session.stale = True
    result = lint(session)
    (stale,) = [f for f in result.findings if f.code == "source.changed"]
    assert stale.severity == "note"
    assert result.stale
    assert result.passed


def test_a_refused_record_is_linted_through_one_forward_run(session_of):
    session = session_of("Mapped", CAMERA_SCENES, frame_rate=6)
    assert session.record.refusal is not None
    seen = []

    def keep(context):
        seen.extend(context.parts)
        return ()

    lint(session, rules=(keep,))
    assert len(session.forward_runs) == 1
    assert [p.t for p in seen] == [i.t for i in lint_instants(session.record, 4)]
    assert all(p.answer == "forward" for p in seen)
    assert all(len(p.parts) == 1 for p in seen)
    assert session.divergences == []


def test_the_range_keeps_findings_that_overlap_it(session_of):
    session = session_of("Pair", frame_rate=15)
    every = _unfolded(lint(session, rules=(_named("left"),)).findings)
    window = (0.2, 0.3)
    kept = _unfolded(lint(session, rules=(_named("left"),), t_range=window).findings)
    assert kept
    assert len(kept) < len(every)
    assert all(0.2 <= f.t_start <= 0.3 for f in kept)


def _unfolded(findings):
    return [g for f in findings for g in (f, *f.folded)]


def test_exemptions_filter_findings(session_of):
    session = session_of("Pair", frame_rate=15)
    rules = (_named("left"), _named("right"))
    found = lint(session, rules=rules).findings
    assert {f.objects[0].name for f in found} == {"left", "right"}
    left_out = lint(session, rules=rules, exempt=[f"line:{line_of('left')}"])
    assert {f.objects[0].name for f in left_out.findings} == {"right"}
    with pytest.raises(ValueError):
        lint(session, exempt=["line:left"])


def test_look_alikes_fold_in_the_result(session_of):
    """The same finding at each of many instants: a run of look-alikes,
    the first kept and the rest folded into it.
    """
    session = session_of("Pair", frame_rate=15)
    (found,) = lint(session, rules=(_named("right"),)).findings
    assert len(found.folded) == len(lint_instants(session.record, 4)) - 1


def test_the_session_lints_itself(session_of):
    session = session_of("Pair", frame_rate=15)
    result = session.lint(rules=(_named("left"),), fps=2)
    assert result.instants == lint_instants(session.record, 2)
    assert not result.passed


# The report


def _reported(session_of):
    session = session_of("Pair", frame_rate=15)
    session._keep(Divergence("probe", 0, session.record.plays[0].source_line, "x"))
    return session, lint(session, rules=(_named("left"),))


def test_the_report_has_its_fields(session_of):
    _, result = _reported(session_of)
    out = report(result)
    assert_report(out)
    assert out["pass"] is False
    overlap = out["violations"][0]
    assert overlap["code"] == "overlap.text"
    assert overlap["class"] == "Clearance"
    assert overlap["severity"] == "error"
    assert_notes(notes(result))


def test_the_report_carries_errors_and_warnings_and_notes_go_apart(session_of):
    _, result = _reported(session_of)
    out = report(result)
    assert {v["severity"] for v in out["violations"]} == {"error"}
    (note,) = notes(result)
    assert note["code"] == "replay.diverged"
    assert note["severity"] == "note"
    assert_notes([note])
    assert note not in report(result)["violations"]


def test_only_an_error_fails_the_report(session_of):
    session = session_of("Pair", frame_rate=15)
    session._keep(Divergence("probe", 0, session.record.plays[0].source_line, "x"))

    def warn(context):
        yield Finding("motion.jump", Fraction(0), Fraction(1), detail={"side": "after"})

    result = lint(session, rules=(warn,))
    out = report(result)
    assert_report(out)
    assert out["pass"] is True
    (warning,) = out["violations"]
    assert warning["severity"] == "warning"
    assert warning["detail"] == {"side": "after"}


def test_a_violation_names_its_objects_in_scene_units_and_target_pixels(session_of):
    _, result = _reported(session_of)
    out = report(result)
    first = out["violations"][0]
    assert first["line"] == line_of("left")
    (obj,) = first["objects"]
    assert obj["source_line"] == line_of("left")
    assert obj["name"] == "left"
    assert obj["kind"] == "Dot"
    x0, y0, x1, y1 = obj["bbox"]
    # The dot (radius 0.08) begins at (-1, 0), in scene units.
    assert (x0 + x1) / 2 == pytest.approx(-1)
    assert (y0 + y1) / 2 == pytest.approx(0)
    # In pixels of the declared target, from the top-left corner, y down.
    width = TARGET_HEIGHT * 16 / 9
    px0, py0, px1, py1 = obj["screen_bbox"]
    assert (px0 + px1) / 2 == pytest.approx(width / 2 - width / 14.222222222222221)
    assert (py0 + py1) / 2 == pytest.approx(TARGET_HEIGHT / 2)
    assert py0 < py1
    assert first["t_start"] == 0.0
    assert out["timeline"][0] == {
        "index": 0,
        "kind": "play",
        "t_start": 0.0,
        "t_end": 0.5,
    }


def test_the_shape_refuses_what_the_report_never_holds(session_of):
    """A note among the violations, a box of three numbers, a missing
    screen box, a pass beside an error, a violation of no severity: each
    fails the shape, and so does a report's violation read as a note.
    """
    _, result = _reported(session_of)
    good = report(result)
    (note,) = notes(result)

    def broken(change):
        out = copy.deepcopy(good)
        change(out)
        with pytest.raises(AssertionError):
            assert_report(out)

    broken(lambda out: out["violations"].append(note))
    broken(lambda out: out["violations"][0]["objects"][0].update(bbox=[0.0, 1.0, 2.0]))
    broken(lambda out: out["violations"][0]["objects"][0].pop("screen_bbox"))
    broken(lambda out: out.update({"pass": True}))
    broken(lambda out: out["violations"][0].pop("severity"))
    with pytest.raises(AssertionError):
        assert_notes([good["violations"][0]])
