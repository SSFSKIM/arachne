"""Route 2's admission: a callback is admitted by a proof that it is a
function of the instant (its shape, its setters, its closure) and by the
dependency order of the play's callbacks; anything else ticks.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import LEFT, RIGHT, UP, VMobject
from arachne.timeline.clock import frames_in
from arachne.utils.bezier import interpolate

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


def _collapse_as_ce_draws_it(fps: int) -> np.ndarray:
    """``Collapse``'s path as CE leaves it at ``fps``: its updater called at
    each frame of the one-second play, then once more as the play ends.
    """
    path = VMobject().set_points_as_corners([LEFT, 0.5 * UP, RIGHT])

    def update(value: float) -> None:
        path.put_start_and_end_on(LEFT, LEFT + RIGHT * abs(value))

    for k in range(fps):
        update(interpolate(-0.3, 0.7, k / fps))
    update(0.7)
    return path.points


@pytest.mark.filterwarnings(COLLAPSE_WARNING)
@pytest.mark.parametrize(("fps", "collapsed"), [(4, False), (30, True)])
def test_a_path_collapsed_at_alpha_three_tenths_is_as_ce_draws_it_at_each_rate(
    fps, collapsed
):
    """The ends meet at 0.3 s. CE collapses the path there for good when a
    frame falls there (30 fps), and never at 4 fps. No stepping work reads
    the path, so its dt-free updater runs where CE runs it, at the frames.
    """
    scene, _ = render(route_scenes.Collapse, fps)
    points = scene.mobjects[0].points
    np.testing.assert_allclose(points, _collapse_as_ce_draws_it(fps), atol=1e-12)
    assert (np.ptp(points, axis=0).max() == 0) == collapsed


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
def test_a_late_reader_reads_the_frame_before_as_in_ce(session_of, fps):
    """``tag`` reads ``dot`` before ``dot``'s updater runs: CE shows it next
    to where ``dot`` was at the frame before. Nothing stepping reads either,
    so both run at the frames.
    """
    session = session_of("LateReader", ROUTE_SCENES, frame_rate=fps)
    play = session.record.plays[0]
    instants = frames_in(play.start, play.end, Fraction(fps))
    assert len(instants) == fps
    for before, t in itertools.pairwise(instants):
        x0, _, x1, _ = session.state_at(before).named("dot").box
        t0, _, t1, _ = session.state_at(t).named("tag").box
        assert (t0 + t1) / 2 == pytest.approx((x0 + x1) / 2), t


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
