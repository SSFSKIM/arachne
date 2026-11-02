"""Route 2's admission: a callback is admitted by a proof that it is a
function of the instant (its shape, its setters, its closure) and by the
dependency order of the play's callbacks; anything else ticks.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from . import route_scenes
from .conftest import render

ROUTE_SCENES = Path(route_scenes.__file__)


def _line_of(tag: str) -> int:
    for number, line in enumerate(ROUTE_SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def _play(session_of, name: str, index: int = 0):
    return session_of(name, ROUTE_SCENES).record.plays[index]


def test_admitted_callbacks_make_a_state_function_play(session_of):
    session = session_of("Readouts", ROUTE_SCENES)
    routes = [play.route for play in session.record.plays]
    assert routes == ["state"] * 4
    kinds = {callback.kind for callback in session.record.plays[0].admission.callbacks}
    assert kinds == {"updater", "redraw"}
    assert {c.kind for c in session.record.plays[2].admission.callbacks} >= {
        "alpha_func"
    }
    # Blink's parts are UpdateFromFunc over the engine's own setters.
    assert {c.kind for c in session.record.plays[3].admission.callbacks} >= {"func"}
    for play in session.record.plays:
        for callback in play.admission.callbacks:
            assert callback.refusal is None
            assert callback.proof is not None


def test_a_callback_carries_its_source_line_relative_to_the_scene(session_of):
    play = _play(session_of, "Readouts")
    lines = {str(c.source_line) for c in play.admission.callbacks}
    assert f"route_scenes.py:{_line_of('readout')}" in lines


def test_a_proof_records_what_the_callback_reads_and_writes(session_of):
    play = _play(session_of, "EarlyReader")
    tag_updater, dot_updater = sorted(
        play.admission.callbacks, key=lambda c: c.source_line.line
    )
    assert play.route == "state"
    assert [type(m).__name__ for m in tag_updater.proof.writes] == ["Dot"]
    assert tag_updater.proof.reads == [dot_updater.owner]
    assert any(b.name == "t" for b in dot_updater.proof.bindings)


def test_a_conditional_that_can_leave_an_output_unassigned_is_refused(session_of):
    play = _play(session_of, "Conditional")
    assert play.route == "ticked"
    assert any("conditional" in reason for reason in play.admission.reasons)


@pytest.mark.parametrize("fps", [4, 15, 30])
def test_the_conditional_ends_the_same_whatever_the_frame_rate(fps):
    scene, _ = render(route_scenes.Conditional, fps)
    reference, _ = render(route_scenes.Conditional, 60)
    assert np.array_equal(scene.mobjects[-1].points, reference.mobjects[-1].points)


COLLAPSE_WARNING = "ignore:put_start_and_end_on has been called:UserWarning"


@pytest.mark.filterwarnings(COLLAPSE_WARNING)
def test_a_setter_that_depends_on_the_geometry_it_is_given_ticks(session_of):
    play = _play(session_of, "Collapse")
    assert play.route == "ticked"
    assert any(
        "put_start_and_end_on is not an admitted setter" in r
        for r in play.admission.reasons
    )


@pytest.mark.filterwarnings(COLLAPSE_WARNING)
def test_a_path_collapsed_at_alpha_three_tenths_is_the_same_at_4_and_30_fps():
    """The ends meet at 0.3 s, a tick: CE collapses the path there for good,
    and every frame rate sees it, since the play steps on the tick.
    """
    slow, slow_frames = render(route_scenes.Collapse, 4)
    fast, fast_frames = render(route_scenes.Collapse, 30)
    assert np.array_equal(slow.mobjects[0].points, fast.mobjects[0].points)
    # The collapse kept: every point at one place.
    assert np.ptp(fast.mobjects[0].points, axis=0).max() == 0
    shared = [k for k in range(len(slow_frames)) if k * 30 % 4 == 0]
    assert len(shared) >= 2
    for k in shared:
        assert np.array_equal(slow_frames[k], fast_frames[k * 30 // 4]), k


@pytest.mark.parametrize("name", list(route_scenes.IMPURE))
def test_a_sign_of_impurity_ticks_the_play(session_of, name):
    _, expected = route_scenes.IMPURE[name]
    play = _play(session_of, f"Impure{name}")
    assert play.route == "ticked"
    (reason,) = play.admission.reasons
    assert expected in reason, reason


def test_a_pure_readout_of_a_container_is_admitted_with_its_contents(session_of):
    assert _play(session_of, "ImpurePure").route == "state"
    play = _play(session_of, "ImpureGlobalList")
    assert play.route == "state"
    (callback,) = play.admission.callbacks
    (binding,) = [b for b in callback.proof.bindings if b.name == "_counter"]
    assert binding.where == "global"
    assert list(binding.contents) == [0]


def test_a_callback_reading_what_a_later_one_writes_ticks(session_of):
    late = _play(session_of, "LateReader")
    assert late.route == "ticked"
    (reason,) = late.admission.reasons
    assert f"route_scenes.py:{_line_of('late')}" in reason
    assert "previous sample's value" in reason
    assert _play(session_of, "EarlyReader").route == "state"


@pytest.mark.parametrize("fps", [4, 30])
def test_a_late_reader_reads_the_tick_before_whatever_the_frame_rate(fps):
    scene, _ = render(route_scenes.LateReader, fps)
    reference, _ = render(route_scenes.LateReader, 60)
    tag, expected = scene.mobjects[0], reference.mobjects[0]
    assert np.array_equal(tag.points, expected.points)


@pytest.mark.parametrize("name", ["AnimationAsUpdater", "IllusionRotation"])
def test_an_animation_or_camera_rotation_run_by_dt_ticks(session_of, name):
    play = _play(session_of, name)
    assert play.route == "ticked"
    assert any("takes dt" in reason for reason in play.admission.reasons)


def test_a_held_wait_answers_its_span_with_one_sample(session_of):
    session = session_of("TrustedWait", ROUTE_SCENES)
    play = session.record.plays[0]
    assert play.held
    assert len(play.samples) == 1
    assert (
        session.state_at(Fraction(1, 3)).nodes == session.state_at(Fraction(2, 3)).nodes
    )
