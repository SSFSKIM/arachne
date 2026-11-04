"""The assertion bridge: a scene's own ``axes.outside`` checks, recorded as
it runs, returned to it as findings, and merged into the lint at their
line and time; its sweeps and pairwise checks, measured at the call;
declarations, which the lint's rules are held to; a watch stopped; the
motion rule armed; and the record's refusal of a scene that replaced a
camera method, which then lints through a forward run.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Axes, Camera, Dot
from arachne.lint import lint
from arachne.lint.bridge import (
    Watch,
    apart,
    arm,
    armed,
    clear,
    declare,
    declared,
    exemptions,
    record,
    sweep,
    watch,
)
from arachne.lint.exempt import Exemptions
from arachne.lint.instants import lint_instants
from arachne.lint.motion import motion_jump
from arachne.lint.rules import contact
from arachne.lint.rules import sweep as sweep_rule
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.clock import frames_in
from arachne.timeline.purity import Proof

from ..timeline.conftest import render
from . import bridge_scenes
from .conftest import SMALL, line_of

BRIDGE = Path(bridge_scenes.__file__)


def _line(tag: str) -> int:
    return line_of(tag, BRIDGE)


def _outside(result):
    return [f for f in result.findings if f.code == "axes.outside"]


def _id_of(session, kind: str) -> str:
    (found,) = {n.id for n in _nodes(session.state_at(0)) if n.type == kind}
    return found


def _nodes(state):
    return state.nodes


# record


def test_a_point_off_the_plot_is_found_at_the_call_s_line_and_time(session_of):
    session = session_of("PointOff", BRIDGE, frame_rate=15)
    (found,) = _outside(lint(session))
    assert found.line.line == _line("point-off")
    assert found.line.path == "bridge_scenes.py"
    assert found.t_start == found.t_end == Fraction(1)
    assert found.severity == "error"
    assert found.held is False
    assert found.objects == ()
    assert found.detail["axes"] == _id_of(session, "Axes")
    assert found.detail["point"] == pytest.approx([4.0, 0.0])


def test_record_returns_its_findings_in_the_report_s_shape(session_of):
    session = session_of("PointOff", BRIDGE, frame_rate=15)
    (returned,) = session.scene.found
    assert returned["code"] == "axes.outside"
    assert returned["class"] == "Clearance"
    assert returned["severity"] == "error"
    assert returned["line"] == _line("point-off")
    assert returned["t_start"] == returned["t_end"] == 1.0
    assert returned["objects"] == []
    assert returned["held"] is False
    assert returned["detail"]["point"] == pytest.approx([4.0, 0.0])
    assert returned["fix"]


def test_a_point_on_the_plot_is_silent(session_of):
    session = session_of("PointOn", BRIDGE, frame_rate=15)
    assert session.scene.found == []
    assert _outside(lint(session)) == []


def test_operands_that_never_enter_the_scene_are_checked(session_of):
    session = session_of("Unadded", BRIDGE, frame_rate=15)
    found = _outside(lint(session))
    assert [f.line.line for f in found] == [
        _line("unadded-dot"),
        _line("unadded-point"),
    ]
    assert all(f.t_start == Fraction(1, 2) for f in found)
    # The dot's own outline reaches past y = 2 (its centre is at 3.6).
    assert found[0].detail["point"][1] > 2
    assert found[1].detail["point"] == pytest.approx([-2.8, 0.0])
    # Neither the dot nor the axes was ever in the scene.
    types = {n.type for n in _nodes(session.state_at(Fraction(1, 2)))}
    assert not types & {"Dot", "Axes"}


def test_a_margin_holds_points_that_far_inside_the_plot(session_of):
    session = session_of("Margined", BRIDGE, frame_rate=15)
    (found,) = _outside(lint(session))
    assert found.line.line == _line("margined")


def test_calls_are_stored_on_the_record_per_play(session_of):
    session = session_of("PointOff", BRIDGE, frame_rate=15)
    bridge = session.record.bridge
    (call,) = bridge.calls
    # Between plays a call belongs to the play about to begin.
    assert bridge.of(1) == (call,)
    assert bridge.of(0) == ()
    assert (call.kind, call.code, call.t, call.play) == ("record", "axes.outside", 1, 1)
    assert call.line.line == _line("point-off")
    assert call.reference == _id_of(session, "Axes")
    (points,) = call.operands
    assert isinstance(points, np.ndarray)
    assert points == pytest.approx(np.array([[4.0, 0.0]]))


def test_a_mobject_operand_is_stored_by_its_stable_id(session_of):
    session = session_of("Unadded", BRIDGE, frame_rate=15)
    dot_call, point_call = session.record.bridge.calls
    (dot_id,) = dot_call.operands
    assert dot_id == f"bridge_scenes.py:{_line('unadded-dot')}#0"
    assert isinstance(point_call.operands[0], np.ndarray)


# watch


def _first_outside(session, x_max: float) -> Fraction:
    """The first render frame at which the session's one dot is past
    ``x_max``, by the record's own state.
    """
    fps = session.record.grid
    for k in range(int(session.record.duration * fps) + 1):
        t = Fraction(k) / fps
        (dot,) = [n for n in _nodes(session.state_at(t)) if n.type == "Dot"]
        x_min, _, x_hi, _ = dot.box
        if (x_min + x_hi) / 2 > x_max:
            return t
    raise AssertionError("the dot never left")


def test_a_watch_fires_on_frames_between_lint_instants(session_of):
    """CE's guard checks every frame: an excursion of two frames, neither a
    lint instant, is found.
    """
    session = session_of("Blip", BRIDGE, frame_rate=15)
    (found,) = _outside(lint(session))
    assert (found.t_start, found.t_end) == (Fraction(5, 15), Fraction(6, 15))
    assert found.line.line == _line("blip")
    instants = {i.t for i in lint_instants(session.record, 4)}
    assert not {found.t_start, found.t_end} & instants


def test_a_watched_pen_off_the_plot_is_found_where_it_left(session_of):
    session = session_of("PenOff", BRIDGE, frame_rate=15)
    (found,) = _outside(lint(session))
    assert found.t_start == _first_outside(session, 2.0)
    assert 1 < found.t_start < 2
    # It stays off to the end.
    assert found.t_end == session.record.duration
    # CE's rule: the scene line running when it left; the watch's own line
    # is in the message.
    assert found.line.line == _line("pen-play")
    assert f"line {_line('pen-watch')}" in found.message
    assert found.objects == ()


def test_a_watched_pen_that_stays_on_the_plot_is_silent(session_of):
    session = session_of("PenOn", BRIDGE, frame_rate=15)
    assert _outside(lint(session)) == []


def test_each_excursion_of_a_watched_pen_is_one_finding(session_of):
    session = session_of("PenBack", BRIDGE, frame_rate=15)
    out, down = _outside(lint(session))
    assert out.line.line == _line("back-out")
    assert down.line.line == _line("back-down")
    assert out.t_end < down.t_start
    assert down.t_end == session.record.duration


def test_a_watched_callable_is_evaluated_while_recording_never_when_linting(
    session_of,
):
    session = session_of("ProbeOff", BRIDGE, frame_rate=15)
    session.module.RECORDED[0] = True
    (found,) = _outside(lint(session))
    assert session.forward_runs == []
    assert found.line.line == _line("probe-play")
    assert 0 < found.t_start < 1
    (call,) = session.record.bridge.calls
    assert call.kind == "watch"
    assert call.operands == ()
    # A reading at every frame from the declaration on, the frozen wait's
    # included, and at the end.
    duration = session.record.duration
    read = {r.t for r in call.readings}
    assert read >= set(frames_in(Fraction(0), duration, Fraction(15)))
    assert call.readings[-1].t == duration


def _past_readings(session) -> list[bool]:
    return [
        r.values is not None and r.values["past"] > 0
        for call in session.record.bridge.calls
        for r in call.readings
    ]


def test_a_record_on_another_grid_is_checked_as_the_render_s_frames_show_it(
    session_of,
):
    """A record kept on a finer grid ran the scene's frame hook more often
    than the render does, and its own readings take the point off the plot;
    the lint reads the watch and the record where the render's frames do,
    from a forward run on the render's grid.
    """
    render = session_of("HookedIn", BRIDGE, frame_rate=15)
    finer = session_of("HookedIn", BRIDGE, grid=None, frame_rate=15)
    assert not any(_past_readings(render))
    assert any(_past_readings(finer))
    assert _outside(lint(render)) == []
    assert _outside(lint(finer)) == []
    assert len(finer.forward_runs) == 1


def test_a_watch_on_another_grid_is_found_where_the_render_s_frames_find_it(
    session_of,
):
    render = session_of("HookedOut", BRIDGE, frame_rate=15)
    finer = session_of("HookedOut", BRIDGE, grid=None, frame_rate=15)
    on_render = _outside(lint(render))
    on_finer = _outside(lint(finer))
    assert [f.line.line for f in on_render] == [
        _line("hooked-play"),
        _line("hooked-record"),
    ]
    assert [(f.t_start, f.t_end, f.line, f.message) for f in on_finer] == [
        (f.t_start, f.t_end, f.line, f.message) for f in on_render
    ]
    # The finer record's own readings would have found it earlier.
    (watched,) = finer.record.bridge.watches
    first = next(r.t for r in watched.readings if r.values["past"] > 0)
    assert first < on_render[0].t_start


# Re-recording


@pytest.fixture
def planted(monkeypatch):
    """The prover takes a :func:`.bridge_scenes.planted` setter as proven, so
    the probe finds it history-dependent and the session records again.
    """
    prove = admission.prove_setter_body

    def forged(function, receiver, scene, **kwargs):
        if getattr(function, "planted", None) is True:
            return Proof(writes=[receiver])
        return prove(function, receiver, scene, **kwargs)

    monkeypatch.setattr(admission, "prove_setter_body", forged)


def test_calls_survive_a_probe_re_recording(session_of, planted):
    config = {**SMALL, "frame_rate": 15}
    once = Session.open(BRIDGE, "DriftChecked", config=config, probe=False)
    again = Session.open(BRIDGE, "DriftChecked", config=config)
    try:
        assert again.probe.demoted == (0,)
        assert again.record is not once.record
        for session in (once, again):
            kinds = [(c.kind, c.line.line) for c in session.record.bridge.calls]
            assert kinds == [
                ("record", _line("drift-record")),
                ("watch", _line("drift-watch")),
            ]
        found = _outside(lint(again))
        assert [f.line.line for f in found] == [
            _line("drift-record"),
            _line("drift-up"),
        ]
        assert [(f.t_start, f.t_end) for f in _outside(lint(once))] == [
            (f.t_start, f.t_end) for f in found
        ]
    finally:
        once.close()
        again.close()


# declare


def test_declared_terms_are_stored_for_the_rules(session_of):
    session = session_of("Declared", BRIDGE, frame_rate=15)
    assert session.scene.found == []
    bridge = session.record.bridge
    first, second = bridge.calls
    assert (first.kind, first.t, first.line.line) == ("declare", 0, _line("declare"))
    a, b = session.scene.a, session.scene.b
    # Each exempted mobject is kept as its family's stable ids at the call.
    exempt = Exemptions.parse([a, (a, b)])
    assert {s.by for s in exempt.bare} == {"family"}
    assert len(next(iter(exempt.bare)).value) == len(a.get_family()) > 1
    assert dict(first.terms) == {"text_gap": 0.3, "exempt": exempt}
    assert first.play == 0
    assert (second.t, second.play) == (Fraction(1, 2), 1)
    assert dict(second.terms) == {"pad": 0.05}
    assert declared(session.record, Fraction(1, 4)) == {
        "text_gap": 0.3,
        "exempt": exempt,
    }
    assert declared(session.record, 1) == {
        "text_gap": 0.3,
        "exempt": exempt,
        "pad": 0.05,
    }
    # Nothing runs unasked: no finding comes of them alone.
    assert lint(session).findings == ()


def test_an_unknown_term_is_refused(session_of):
    with pytest.raises(TypeError, match="text_spacing"):
        session_of("BadTerm", BRIDGE, frame_rate=15)


# Outside a recording


@pytest.mark.parametrize(
    "call",
    [
        lambda ax: record("axes.outside", [(0, 0, 0)], reference=ax),
        lambda ax: watch("axes.outside", Dot(), reference=ax),
        lambda ax: declare(text_gap=0.3),
        lambda ax: sweep(),
        lambda ax: apart(Dot(), ax),
        lambda ax: arm(),
        lambda ax: clear(None),
    ],
    ids=["record", "watch", "declare", "sweep", "apart", "arm", "clear"],
)
def test_a_bridge_call_outside_a_recording_is_refused(call):
    with pytest.raises(RuntimeError, match="recording"):
        call(Axes())


@pytest.mark.parametrize(
    ("scene", "error"),
    [("UnknownCode", "overlap.text"), ("NoReference", "reference")],
)
def test_an_unknown_code_or_a_missing_reference_is_refused(session_of, scene, error):
    with pytest.raises(ValueError, match=error):
        session_of(scene, BRIDGE, frame_rate=15)


# The record refuses a replaced camera method


@pytest.mark.parametrize(
    ("scene", "member", "form"),
    [
        ("InstanceReplaced", "Camera.get_mobjects_to_display", "instance"),
        ("ClassReplaced", "Camera.get_mobjects_to_display", "class"),
        ("MemberReplaced", "Camera.get_coords_of_all_pixels", "instance"),
    ],
)
def test_a_replaced_camera_method_refuses_the_record(session_of, scene, member, form):
    original = vars(Camera)["get_mobjects_to_display"]
    try:
        session = session_of(scene, BRIDGE, frame_rate=15)
    finally:
        Camera.get_mobjects_to_display = original
    refusal = session.record.refusal
    assert refusal is not None
    assert refusal.startswith("camera.replaced:")
    assert member in refusal
    assert form in refusal
    assert all(play.route == "ticked" for play in session.record.plays)


def test_the_engine_s_camera_refuses_nothing(session_of):
    session = session_of("NothingReplaced", BRIDGE, frame_rate=15)
    assert session.record.refusal is None


def test_a_refused_scene_lints_through_one_forward_run(session_of):
    session = session_of("InstanceReplaced", BRIDGE, frame_rate=15)
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


def test_a_refused_scene_renders_the_frames_it_renders_unrefused():
    _, replaced = render(bridge_scenes.InstanceReplaced, 15)
    _, engine = render(bridge_scenes.NothingReplaced, 15)
    assert len(replaced) == len(engine)
    assert all(np.array_equal(a, b) for a, b in zip(replaced, engine, strict=True))


def test_a_camera_method_replaced_during_the_last_play_refuses_the_record(
    session_of,
):
    """Replaced mid-play, after the play began and before the script ended:
    the record refuses, and the lint calls no scene camera code.
    """
    session = session_of("LateCamera", BRIDGE, frame_rate=15)
    refusal = session.record.refusal
    assert refusal is not None
    assert "Camera.transform_points_pre_display (on the instance)" in refusal
    session.module.RECORDED[0] = True
    seen = []

    def keep(context):
        seen.extend(context.parts)
        return ()

    lint(session, rules=(keep,))
    assert seen
    assert all(p.answer == "forward" for p in seen)


def test_a_camera_method_replaced_mid_play_renders_the_same_frames():
    _, replaced = render(bridge_scenes.LateCamera, 15)
    _, engine = render(bridge_scenes.LateUnchanged, 15)
    assert len(replaced) == len(engine)
    assert all(np.array_equal(a, b) for a, b in zip(replaced, engine, strict=True))


def test_a_watch_is_read_at_every_frame_of_a_frozen_wait(session_of):
    """CE's guard runs on every frame of a wait: a frozen wait's frames are
    each read, with the clock at the frame's time.
    """
    session = session_of("ClockWatch", BRIDGE, frame_rate=15)
    assert [play.held for play in session.record.plays] == [True]
    (found,) = _outside(lint(session))
    assert (found.t_start, found.t_end) == (Fraction(5, 15), Fraction(6, 15))
    assert found.line.line == _line("clock-wait")


def test_a_watch_reports_the_frames_its_readings_show(session_of):
    """At 24 fps a ticked play's capture is of a tick, not of the frame that
    shows it: the finding is at the frame.
    """
    session = session_of("TickWatch", BRIDGE, frame_rate=24)
    (dot,) = [n for n in session.state_at(Fraction(1, 24)).nodes if n.type == "Dot"]
    assert dot.box[0] > 2
    (found,) = _outside(lint(session))
    assert (found.t_start, found.t_end) == (Fraction(1, 24), Fraction(1, 24))
    assert found.line.line == _line("tick-wait")


def test_an_execution_s_own_calls_are_read_from_its_forward_run(session_of):
    """A finer grid's record holds other calls than the render's execution
    makes: the bridge reads the forward run's.
    """
    finer = session_of("ConditionalWatches", BRIDGE, grid=None, frame_rate=15)
    assert len(finer.record.bridge.watches) == 1
    found = _outside(lint(finer))
    assert len(finer.forward_runs) == 1
    assert [(f.t_start, f.t_end, f.line.line) for f in found] == [
        (1, 2, _line("conditional-play")),
        (2, 2, _line("conditional-record")),
    ]


def test_a_lint_at_chosen_times_reads_every_bridge_call(session_of):
    """The geometry read at one instant still reads the bridge's calls over
    the whole execution.
    """
    render = session_of("ConditionalWatches", BRIDGE, frame_rate=15)
    finer = session_of("ConditionalWatches", BRIDGE, grid=None, frame_rate=15)

    def spans(session):
        found = _outside(lint(session, at=[0]))
        return [(f.t_start, f.t_end, f.line.line) for f in found]

    assert spans(render) == [
        (1, 2, _line("conditional-play")),
        (2, 2, _line("conditional-record")),
    ]
    assert spans(finer) == spans(render)


def test_each_point_is_checked_against_its_own_reference_and_margin(session_of):
    session = session_of("ReferenceRanges", BRIDGE, frame_rate=15)
    scene = session.scene
    assert len(scene.out) == len(scene.margin) == 1
    assert scene.inside_broad == scene.near == []
    found = _outside(lint(session))
    assert len(found) == 3
    assert len({f.detail["axes"] for f in found}) == 1
    watched = found[-1]
    assert (watched.t_start, watched.t_end) == (0, session.record.duration)
    assert all(n.type != "Axes" for n in session.state_at(0).nodes)


@pytest.mark.parametrize("width", [96, 98], ids=["render-longer", "render-shorter"])
def test_a_forward_run_that_ends_elsewhere_diverges_and_keeps_its_checks(
    session_of, width
):
    """The render's execution ends at another time than a finer grid's: the
    lint reports the divergence and the bridge's findings it read, and
    raises nothing.
    """
    finer = session_of(
        "DivergentDuration", BRIDGE, grid=None, frame_rate=15, pixel_width=width
    )
    result = lint(finer, at=[0])
    (run,) = finer.forward_runs
    assert run.divergence is not None
    assert any(f.code == "replay.diverged" for f in result.findings)
    watched, recorded = _outside(result)
    assert watched.t_start == 0
    assert recorded.t_start == recorded.t_end
    assert recorded.t_start != finer.record.duration


def test_checks_only_the_forward_run_makes_are_read(session_of):
    """A finer grid's record holds no bridge call; the render's execution,
    which the lint's forward run is, makes two.
    """
    render = session_of("ForwardOnlyBridge", BRIDGE, frame_rate=15)
    finer = session_of("ForwardOnlyBridge", BRIDGE, grid=None, frame_rate=15)
    assert finer.record.bridge.calls == []

    def spans(session):
        found = _outside(lint(session))
        return [(f.t_start, f.t_end, f.line.line) for f in found]

    expected = spans(render)
    assert len(expected) == 2
    assert spans(finer) == expected
    assert len(finer.forward_runs) == 1


def test_a_watch_reports_no_frame_after_the_scene_s_end(session_of):
    """The tick a stop condition ends the scene on holds until a tick that
    never comes: a finding ends at the scene's end.
    """
    session = session_of("StoppedWatch", BRIDGE, frame_rate=24)
    end = session.record.duration
    assert (end * 24).denominator != 1
    (found,) = _outside(lint(session))
    assert found.t_end == end


# sweep

CLEARANCE = {
    "clip.frame_edge",
    "overlap.text",
    "gap.text",
    "contact.held",
    "occlusion.glyph",
}


def _clearance(result):
    return [f for f in result.findings if f.code in CLEARANCE]


def _codes(violations):
    return [v["code"] for v in violations]


def _swept(session, t, **terms):
    """The clearance codes a sweep of the frame shown at ``t`` finds."""
    rules = (sweep_rule.where(at=[t], **terms),)
    return [f.code for f in _clearance(lint(session, rules=rules))]


def test_a_sweep_finds_at_the_call_s_line_and_time(session_of):
    session = session_of("SweptRows", BRIDGE, frame_rate=15)
    (returned,) = session.scene.found
    assert returned["code"] == "gap.text"
    assert returned["line"] == _line("sweep-rows")
    assert returned["t_start"] == returned["t_end"] == 1.5
    assert returned["held"] is False
    (found,) = _clearance(lint(session))
    assert (found.code, found.line.line) == ("gap.text", _line("sweep-rows"))
    assert found.t_start == found.t_end == Fraction(3, 2)
    assert [o.id for o in found.objects] == [o["id"] for o in returned["objects"]]
    assert len(found.objects) == 2


@pytest.mark.parametrize("scene", ["SweptRowsLoose", "SweptRowsNoGap"])
def test_a_sweep_s_twins_are_silent(session_of, scene):
    session = session_of(scene, BRIDGE, frame_rate=15)
    assert session.scene.found == []
    assert _clearance(lint(session)) == []


def test_a_sweep_measures_the_scene_as_it_stands_at_the_call(session_of):
    """Two phrases over each other for no frame: the sweep at the call sees
    them, a sweep of the frame at its time does not.
    """
    session = session_of("SweptAtTheCall", BRIDGE, frame_rate=15)
    assert _codes(session.scene.found) == ["overlap.text"]
    (found,) = _clearance(lint(session))
    assert (found.line.line, found.t_start) == (_line("swept-at-call"), Fraction(1, 2))
    rules = (sweep_rule.where(at=[Fraction(1, 2)]),)
    assert _clearance(lint(session, rules=rules)) == [found]


def test_a_sweep_is_held_to_its_own_terms_then_the_declared_ones(session_of):
    session = session_of("DeclaredThenSwept", BRIDGE, frame_rate=15)
    (returned,) = session.scene.found
    assert (returned["code"], returned["line"]) == ("gap.text", _line("declared-sweep"))
    assert session.scene.overridden == []


def test_a_sweep_s_exemptions_name_mobjects_and_their_families(session_of):
    session = session_of("ExemptedSweeps", BRIDGE, frame_rate=15)
    found = {form: _codes(v) for form, v in session.scene.found.items()}
    assert found["none"] == found["other"] == ["overlap.text"]
    for form in ("bare", "pair", "listed", "held"):
        assert found[form] == [], form


def test_every_exemption_the_scene_states_is_kept_for_the_hold_watch(session_of):
    """R's hold watch reads every exemption a sweep or declaration made,
    each once.
    """
    session = session_of("ExemptedSweeps", BRIDGE, frame_rate=15)
    kept = exemptions(session.record)
    assert (len(kept.bare), len(kept.pairs)) == (3, 1)


# declare


def test_a_declared_text_gap_holds_the_rules_from_its_call_on(session_of):
    session = session_of("DeclaredLate", BRIDGE, frame_rate=15)
    assert _swept(session, Fraction(5, 4)) == []
    assert _swept(session, Fraction(2)) == ["gap.text"]
    # The scene's declaration replaces the rule's own.
    assert _swept(session, Fraction(2), text_gap=0.01) == ["gap.text"]
    (found,) = _clearance(lint(session, rules=(contact,)))
    assert found.code == "gap.text"
    # From the first frame the declaration holds at.
    assert Fraction(3, 2) <= found.t_start == Fraction(23, 15)


@pytest.mark.parametrize(
    ("scene", "codes"),
    [("DeclaredExemptions", []), ("DeclaredLateExemption", ["overlap.text"])],
)
def test_declared_exemptions_hold_the_rules_as_their_families_stood(
    session_of, scene, codes
):
    """A group's family is read at the declaration: what it is given later
    is not exempted.
    """
    session = session_of(scene, BRIDGE, frame_rate=15)
    end = session.record.duration
    assert _swept(session, end) == codes
    # The rule's own exemptions hold beside the scene's.
    assert _swept(session, end, exempt=["name:first"]) == []


# clear


def test_a_cleared_watch_is_read_at_no_frame_after_it(session_of):
    uncleared = _outside(lint(session_of("UnclearedWatch", BRIDGE, frame_rate=15)))
    assert len(uncleared) == 2
    session = session_of("ClearedWatch", BRIDGE, frame_rate=15)
    (found,) = _outside(lint(session))
    assert (found.t_start, found.t_end) == (uncleared[0].t_start, uncleared[0].t_end)
    assert session.scene.found == session.scene.again == []
    handle = session.scene.handle
    assert isinstance(handle, Watch)
    assert handle.found == []
    stop, _ = [c for c in session.record.bridge.calls if c.kind == "clear"]
    assert (stop.t, stop.line.line) == (Fraction(1), _line("clear"))
    assert all(reading.t <= stop.t for reading in handle.call.readings)


def test_clear_takes_a_watch(session_of):
    with pytest.raises(TypeError, match="watch"):
        session_of("ClearsAPoint", BRIDGE, frame_rate=15)


# arm


def _jumps(result):
    return [f for f in result.findings if f.code == "motion.jump"]


def test_an_armed_scene_has_its_motion_measured_unasked(session_of):
    session = session_of("ArmedJump", BRIDGE, frame_rate=15)
    assert session.scene.found == []
    (jump,) = _jumps(lint(session))
    assert jump.detail["distance"] == pytest.approx(4.0)
    (call,) = [c for c in session.record.bridge.calls if c.kind == "arm"]
    assert call.line.line == _line("arm")
    assert session.record.armed_at == 0
    # Asked for as well, the rule runs once.
    assert len(_jumps(lint(session, rules=(motion_jump,)))) == 1


@pytest.mark.parametrize("scene", ["UnarmedJump", "ArmedTolerant", "ReArmedTolerant"])
def test_an_unarmed_or_tolerant_scene_finds_no_jump(session_of, scene):
    assert _jumps(lint(session_of(scene, BRIDGE, frame_rate=15))) == []


def test_the_arming_in_force_is_the_last_one(session_of):
    record = session_of("ReArmedTolerant", BRIDGE, frame_rate=15).record
    assert armed(record).terms["motion_jump"] == 5.0
    assert armed(record, 0) is armed(record)
    assert armed(session_of("UnarmedJump", BRIDGE, frame_rate=15).record) is None


def test_arming_declares_the_terms_it_is_given(session_of):
    session = session_of("ArmedWithTerms", BRIDGE, frame_rate=15)
    assert declared(session.record, 0) == {"text_gap": 0.3, "pad": 0.1}
    assert len(_jumps(lint(session))) == 1


# apart


def test_apart_measures_a_pair_at_the_call(session_of):
    session = session_of("Apart", BRIDGE, frame_rate=15)
    found = {pair: _codes(v) for pair, v in session.scene.found.items()}
    assert found == {
        "rows": ["gap.text"],
        "rows_min_gap": [],
        "far": ["gap.text"],
        "dot_on_line": [],
        "word_on_line": found["word_on_line"],
        "rings": ["contact.held"],
        "rings_apart": [],
        "padded": ["contact.held"],
    }
    assert set(found["word_on_line"]) == {"overlap.text"}
    (rows,) = session.scene.found["rows"]
    assert rows["line"] == _line("apart-rows")
    assert rows["t_start"] == rows["t_end"] == 0.5
    linted = [
        f for f in _clearance(lint(session)) if f.line.line == _line("apart-rows")
    ]
    assert [f.code for f in linted] == ["gap.text"]
