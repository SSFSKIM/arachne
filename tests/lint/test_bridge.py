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
    """The first lint instant at which the session's one dot is past
    ``x_max``, by the record's own state.
    """
    for instant in lint_instants(session.record, 4):
        (dot,) = [n for n in _nodes(session.state_at(instant.t)) if n.type == "Dot"]
        x_min, _, x_hi, _ = dot.box
        if (x_min + x_hi) / 2 > x_max:
            return instant.t
    raise AssertionError("the dot never left")


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
    # A reading at every kept capture from the declaration on.
    kept = [c.instant for play in session.record.plays for c in play.samples]
    assert [r.t for r in call.readings][1:] == [*kept, session.record.duration]


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
