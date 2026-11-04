"""The assertion bridge: a scene's own ``axes.outside`` checks, recorded as
it runs, returned to it as findings, and merged into the lint at their
line and time; declarations stored for the rules; and the record's refusal
of a scene that replaced a camera method, which then lints through a
forward run.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Axes, Camera, Dot
from arachne.lint import lint
from arachne.lint.bridge import declare, record, watch
from arachne.lint.instants import lint_instants
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
    a, b = (session.record.origin(m).id for m in (session.scene.a, session.scene.b))
    assert dict(first.terms) == {"text_gap": 0.3, "exempt": (a, (a, b))}
    assert first.play == 0
    assert (second.t, second.play) == (Fraction(1, 2), 1)
    assert dict(second.terms) == {"pad": 0.05}
    assert bridge.terms_at(Fraction(1, 4)) == {"text_gap": 0.3, "exempt": (a, (a, b))}
    assert bridge.terms_at(1) == {
        "text_gap": 0.3,
        "exempt": (a, (a, b)),
        "pad": 0.05,
    }
    # They apply nothing yet: no finding comes of them.
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
    ],
    ids=["record", "watch", "declare"],
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
