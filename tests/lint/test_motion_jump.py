"""The motion rule: a body seen elsewhere across the cut between two plays."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.lint import RULES, lint
from arachne.lint.motion import JUMP_TOL, MotionJump, motion_jump
from arachne.lint.serialise import report

from . import run_counter
from .conftest import line_of
from .report_shape import assert_report

MOTION = Path(__file__).parent / "motion_scenes.py"


@pytest.fixture
def jumps_of(session_of):
    """The ``motion.jump`` findings of a scene of ``motion_scenes.py``."""

    def find(name: str, *, rule=motion_jump, exempt=(), **settings):
        session = session_of(name, MOTION, frame_rate=15, **settings)
        result = lint(session, rules=(rule,), exempt=exempt)
        return [f for f in result.findings if f.code == "motion.jump"]

    return find


def test_a_teleport_is_a_warning_at_the_next_play_s_first_frame(jumps_of):
    (jump,) = jumps_of("Teleport")
    assert jump.severity == "warning"
    assert jump.t_start == jump.t_end == Fraction(1)
    assert jump.line is not None
    assert jump.line.line == line_of("after-teleport", MOTION)
    (puck,) = jump.objects
    assert puck.source_line == line_of("puck", MOTION)
    assert puck.name == "puck"
    assert puck.kind == "Dot"
    assert puck.bbox is None
    assert puck.screen_bbox is None
    detail = jump.detail
    assert detail["side"] == "before"
    assert detail["boundary"] == {"from_play": 0, "to_play": 1}
    # From where the slide left it, as the play ended, to where it is seen.
    assert detail["distance"] == pytest.approx(4.0, abs=1e-6)
    assert detail["from"] == pytest.approx([-1.0, 0.0], abs=1e-6)
    assert detail["to"] == pytest.approx([3.0, 0.0], abs=1e-6)
    assert "puck" in jump.message


def test_a_record_that_can_t_draw_finds_the_same_teleport(jumps_of, session_of):
    assert session_of("RefusedTeleport", MOTION, frame_rate=15).record.refusal
    (drawn,) = jumps_of("Teleport")
    (forward,) = jumps_of("RefusedTeleport")
    assert forward.t_start == drawn.t_start
    assert forward.objects[0].id == drawn.objects[0].id
    assert forward.detail["distance"] == pytest.approx(drawn.detail["distance"])


def test_a_teleport_under_a_three_d_camera_is_measured_in_the_scene(jumps_of):
    """Under a three-dimensional camera, distances and centres are the
    scene's x and y, z left out, as the other backend reads them.
    """
    (jump,) = jumps_of("TiltedTeleport")
    assert jump.detail["distance"] == pytest.approx(4.0, abs=1e-6)
    assert jump.detail["from"] == pytest.approx([-1.0, 0.0], abs=1e-6)
    assert jump.detail["to"] == pytest.approx([3.0, 0.0], abs=1e-6)


def test_a_step_within_the_tolerance_is_no_jump(jumps_of):
    assert JUMP_TOL == 0.25
    assert jumps_of("Nudge") == []
    (jump,) = jumps_of("Nudge", rule=MotionJump(jump_tol=0.1))
    assert jump.detail["distance"] == pytest.approx(0.2, abs=0.01)


def test_a_static_object_rearranged_between_plays_is_silent(jumps_of):
    assert jumps_of("StaticReset") == []


def test_a_body_set_down_and_then_thrown_jumps_on_the_after_side(jumps_of):
    (jump,) = jumps_of("PlacedThenThrown")
    assert jump.detail["side"] == "after"
    assert jump.t_start == Fraction(1)
    assert jump.line.line == line_of("throw", MOTION)
    assert jump.objects[0].source_line == line_of("stone", MOTION)
    assert jump.detail["distance"] == pytest.approx(3.0, abs=1e-6)


def test_a_camera_cut_moves_nothing(jumps_of):
    assert jumps_of("CameraCut") == []


def test_a_body_faded_out_moved_and_faded_in_is_silent(jumps_of):
    assert jumps_of("FadedAcross") == []


def test_a_group_redrawn_while_its_parts_stand_is_silent(jumps_of):
    assert jumps_of("RedrawnRig") == []


def test_a_shape_redrawn_at_another_size_is_silent(jumps_of):
    assert jumps_of("Restarted") == []


def test_a_set_back_within_the_body_s_own_steps_is_churn(jumps_of):
    assert jumps_of("FastReset") == []
    (jump,) = jumps_of("SlowReset")
    assert jump.detail["distance"] == pytest.approx(1.2, abs=1e-6)


def test_look_alikes_trading_places_are_silent(jumps_of):
    assert jumps_of("Conveyor") == []


def test_a_group_that_jumped_whole_is_one_finding(jumps_of):
    (jump,) = jumps_of("PairJumps")
    (pair,) = jump.objects
    assert pair.name == "pair"
    assert pair.kind == "VGroup"
    assert pair.source_line == line_of("pair", MOTION)
    assert jump.detail["distance"] == pytest.approx(3.0, abs=0.02)


def test_a_group_defers_to_the_member_that_jumped(jumps_of):
    (jump,) = jumps_of("MemberJumps")
    (square,) = jump.objects
    assert square.kind == "Square"
    assert square.name == "pair[1]"
    assert [holder.name for holder in square.within] == ["pair"]
    assert jumps_of("MemberJumps", exempt=["name:pair"]) == []


def test_a_body_named_after_it_jumped_keeps_its_object(jumps_of, session_of):
    (jump,) = jumps_of("NamedLate")
    (puck,) = jump.objects
    assert puck.source_line == line_of("unnamed", MOTION)
    assert puck.name == "puck"
    session = session_of("NamedLate", MOTION, frame_rate=15)
    (view,) = [
        node
        for node in session.state_at(0).nodes
        if node.source_line is not None
        and node.source_line.line == line_of("unnamed", MOTION)
    ]
    assert puck.id == view.id
    for selector in (puck.id, "name:puck", f"line:{puck.source_line}"):
        assert jumps_of("NamedLate", exempt=[selector]) == [], selector


def test_the_report_carries_a_jump_as_a_warning_that_passes(session_of):
    session = session_of("Teleport", MOTION, frame_rate=15)
    out = report(lint(session, rules=(motion_jump,)))
    assert_report(out)
    assert out["pass"] is True
    (violation,) = out["violations"]
    assert violation["code"] == "motion.jump"
    assert violation["severity"] == "warning"
    assert violation["objects"][0]["bbox"] is None
    assert set(violation["detail"]) == {"side", "distance", "from", "to", "boundary"}


def test_nothing_looks_for_jumps_unasked(session_of):
    assert motion_jump not in RULES
    session = session_of("Teleport", MOTION, frame_rate=15)
    assert lint(session).findings == ()


def test_a_zoom_cut_after_a_steady_slide_moves_nothing(jumps_of):
    assert jumps_of("ZoomCut") == []


def test_a_zoom_cut_after_the_camera_followed_the_body_moves_nothing(jumps_of):
    """The body's own steps count, in the scene, however the camera moved
    with it: the last frame's ordinary step is still its own motion.
    """
    assert jumps_of("FollowedZoomCut") == []


def test_a_set_back_behind_a_tenfold_zoom_jumps(jumps_of):
    (jump,) = jumps_of("ZoomedReset")
    # From where the slide ended (x = 0.5) to 0, at 10x.
    assert jump.detail["distance"] == pytest.approx(5.0, abs=1e-6)


def test_a_body_s_steps_under_a_pan_are_as_the_viewer_saw_them(jumps_of):
    """As the other backend finds them: the 0.4 set-back is churn (its steps
    on screen were 4/15, the pan's 3/15 among them) and the 0.7 one jumps,
    0.7 from where the slide ended.
    """
    assert jumps_of("PanReset") == []
    (jump,) = jumps_of("PanResetFar")
    assert jump.detail["distance"] == pytest.approx(0.7, abs=1e-6)


def test_a_teleport_behind_a_zoom_cut_still_jumps(jumps_of):
    (jump,) = jumps_of("ZoomedTeleport")
    # 2 units set down from where the slide ended, at 2x.
    assert jump.detail["distance"] == pytest.approx(4.0, abs=1e-6)


def test_every_repeated_jump_is_reported_on_its_own(session_of):
    session = session_of("Repeated", MOTION, frame_rate=15)
    out = report(lint(session, rules=(motion_jump,)))
    jumps = [v for v in out["violations"] if v["code"] == "motion.jump"]
    # Each at the first frame of the wait after a set-back: of 15 fps frames,
    # the first at or after 1, 2.5, 4, 5.5 and 7.
    assert [v["t_start"] for v in jumps] == pytest.approx([1, 38 / 15, 4, 83 / 15, 7])
    assert all("folded" not in v["detail"] for v in jumps)


def test_one_lint_reads_one_forward_run_for_every_rule(session_of):
    """Parts and the motion rule see the same execution of a scene whose
    every execution differs.
    """
    run_counter.executions = 0
    session = session_of("CountedRuns", MOTION, frame_rate=15)
    seen = []

    def observe(context):
        seen.extend(
            (p.scene_box[0] + p.scene_box[2]) / 2
            for parts in context.parts
            if parts.t == 1
            for p in parts.parts
            if p.type == "Dot"
        )
        return ()

    result = lint(session, rules=(motion_jump, observe))
    assert len(session.forward_runs) == 1
    (jump,) = [f for f in result.findings if f.code == "motion.jump"]
    (x,) = seen
    assert jump.detail["to"][0] == pytest.approx(x, abs=1e-6)


def test_a_body_that_joins_late_and_jumps_is_found(jumps_of):
    (jump,) = jumps_of("LateComer")
    assert jump.objects[0].source_line == line_of("late-comer", MOTION)
    assert jump.detail["boundary"] == {"from_play": 3, "to_play": 4}
    assert jump.detail["distance"] == pytest.approx(3.0, abs=1e-6)


def test_an_armed_session_is_measured_from_its_own_record(session_of):
    session = session_of("ArmedTeleport", MOTION, frame_rate=15)
    assert set(session.record.exits) == {0, 1}
    (found,) = lint(session, rules=(motion_jump,)).findings
    assert session.forward_runs == []
    assert found.detail["distance"] == pytest.approx(4.0, abs=1e-6)


@pytest.mark.parametrize("strip", ["unarmed", "all", "one"])
def test_a_record_without_its_exits_is_read_from_a_forward_run(session_of, strip):
    """A record that keeps no state of a play's end, or not of every play's
    (a scene that never armed the rule, or armed it late), is never measured
    another way: the lint's forward run keeps them.
    """
    armed = session_of("ArmedTeleport", MOTION, frame_rate=15)
    (expected,) = lint(armed, rules=(motion_jump,)).findings
    if strip == "unarmed":
        session = session_of("Teleport", MOTION, frame_rate=15)
        assert session.record.exits is None
    else:
        session = session_of("ArmedTeleport", MOTION, frame_rate=15)
        if strip == "all":
            session.record.exits = None
        else:
            del session.record.exits[0]
    (found,) = lint(session, rules=(motion_jump,)).findings
    assert len(session.forward_runs) == 1
    assert dict(found.detail) == dict(expected.detail)


def test_a_forward_run_keeps_every_play_s_exit(session_of):
    session = session_of("Repeated", MOTION, frame_rate=15)
    assert session.record.exits is None
    read = []

    def observe(context):
        read.append(context.frames.execution)
        return ()

    lint(session, rules=(motion_jump, observe))
    (execution,) = read
    assert execution is not session.record
    assert len(execution.plays) == len(session.record.plays) > 2
    assert set(execution.exits) == {play.index for play in execution.plays}


@pytest.mark.parametrize("forward", [False, True])
def test_only_cuts_from_the_arming_on_are_measured(session_of, forward):
    """As the other backend captures them: a teleport before the scene armed
    the rule is not seen, from the session's record or a forward run's.
    """
    session = session_of("ArmedAtTwo", MOTION, frame_rate=15)
    assert session.record.armed_at == 2
    assert set(session.record.exits) == {2, 3}
    if forward:
        session.record.exits = None
    read = []

    def observe(context):
        read.append(context.frames.execution)
        return ()

    (jump,) = lint(session, rules=(motion_jump, observe)).findings
    assert len(session.forward_runs) == int(forward)
    (execution,) = read
    assert execution.armed_at == 2
    assert jump.detail["boundary"] == {"from_play": 2, "to_play": 3}
    assert jump.line.line == line_of("after-armed-teleport", MOTION)
    assert jump.detail["distance"] == pytest.approx(4.0, abs=1e-6)


def test_a_scene_that_never_armed_has_every_cut_measured(session_of):
    """Asked for the rule, a scene that never armed it is read whole: both
    teleports are found.
    """
    session = session_of("TwoTeleports", MOTION, frame_rate=15)
    assert session.record.armed_at is None
    found = lint(session, rules=(motion_jump,)).findings
    assert [f.detail["boundary"] for f in found] == [
        {"from_play": 0, "to_play": 1},
        {"from_play": 2, "to_play": 3},
    ]


def test_a_live_camera_picture_that_turned_visible_jumps(session_of):
    """As the other backend finds it: 3 units, from play 1 to play 2, the
    display having turned visible while it held the same view of its camera.
    """
    session = session_of("LiveImageJump", MOTION, frame_rate=15)
    (jump,) = lint(session, rules=(motion_jump,)).findings
    assert session.forward_runs == []
    assert jump.detail["boundary"] == {"from_play": 1, "to_play": 2}
    assert jump.detail["distance"] == pytest.approx(3.0, abs=1e-6)
    assert jump.detail["from"] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert jump.detail["to"] == pytest.approx([3.0, 0.0], abs=1e-6)
