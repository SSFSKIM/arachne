"""The probe: each state-function play restored at two instants its record
kept, from its start or its last event before them, against what the record
kept there. A play whose restore differs keeps history its proof did not
see: it is demoted to route 3 for good, the scene is recorded again, and
the session reports where.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.timeline import admission
from arachne.timeline.execution import Divergence
from arachne.timeline.purity import Proof

from . import camera_scenes, probe_scenes, restore_scenes, route_scenes
from .conftest import render

PROBE_SCENES = Path(probe_scenes.__file__)
RESTORE_SCENES = Path(restore_scenes.__file__)
ROUTE_SCENES = Path(route_scenes.__file__)
CAMERA_SCENES = Path(camera_scenes.__file__)


def _line(tag: str) -> int:
    for number, line in enumerate(PROBE_SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


@pytest.fixture
def planted(monkeypatch):
    """The prover takes a :func:`.probe_scenes.planted` setter as proven, one
    that writes its own mobject and reads nothing: a forged proof, outside
    what admission vouches for, which the probe is the backstop of.
    """
    prove = admission.prove_setter_body

    def forged(function, receiver, scene, **kwargs):
        if getattr(function, "planted", False):
            return Proof(writes=[receiver])
        return prove(function, receiver, scene, **kwargs)

    monkeypatch.setattr(admission, "prove_setter_body", forged)


def _outcomes(session) -> dict[int, str]:
    return {found.play: found.outcome for found in session.probe.plays}


def test_a_planted_history_dependent_setter_is_demoted_recorded_again_and_reported(
    session_of, planted
):
    unprobed = session_of("PlantedDrift", PROBE_SCENES, frame_rate=15, probe=False)
    first = unprobed.record.plays[0]
    # The patch admits it: without the probe it stays a state play.
    assert first.route == "state"

    session = session_of("PlantedDrift", PROBE_SCENES, frame_rate=15)
    play = session.record.plays[0]
    assert session.probe.demoted == (0,)
    assert _outcomes(session)[0] == "demoted"
    # Recorded again, with the play on route 3 for the reason the probe gave.
    assert play.route == "ticked"
    assert play.admission.reasons[0].startswith("probe.demoted")
    assert session.record.plays[1].route == "trusted"
    # Its instants are answered as route 3 answers them.
    assert session.state_at(Fraction(1, 2) + Fraction(1, 60)).answer == "recorded"
    # The restore from the play's start missed the furthest the dot reached.
    found_play = session.probe.plays[0]
    assert found_play.bases == (play.start, play.start)

    probes = [d for d in session.divergences if d.kind == "probe"]
    assert len(probes) == 1
    found = probes[0]
    assert isinstance(found, Divergence)
    assert (found.code, found.play) == ("replay.diverged", 0)
    assert found.source_line == play.source_line
    assert found.source_line.line == _line("drift")
    assert found.source_line.path == "probe_scenes.py"
    assert "_points" in found.reason or "points" in found.reason

    # For good: a later session of the source ticks it from its first run.
    later = session_of("PlantedDrift", PROBE_SCENES, frame_rate=15, probe=False)
    assert later.record.plays[0].route == "ticked"
    assert later.record.plays[0].admission.reasons[0].startswith("probe.demoted")


def test_a_frame_hook_writing_what_its_frames_show_is_demoted(session_of):
    """A restore from the play's start can't know what the hook wrote at
    the frames since.
    """
    session = session_of("FrameCounter", RESTORE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0,)
    assert session.record.plays[0].route == "ticked"
    assert [d.kind for d in session.divergences][:1] == ["probe"]


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (ROUTE_SCENES, "Readouts"),
        (RESTORE_SCENES, "ClosureCell"),
        (RESTORE_SCENES, "DetachedTracker"),
        (RESTORE_SCENES, "SuccessionReadout"),
        (CAMERA_SCENES, "PureCentre"),
    ],
)
def test_a_clean_state_play_stays_on_route_2(session_of, path, name):
    unprobed = session_of(name, path, frame_rate=15, probe=False)
    session = session_of(name, path, frame_rate=15)
    routes = [play.route for play in session.record.plays]
    assert routes == [play.route for play in unprobed.record.plays]
    assert "state" in routes
    assert session.probe.demoted == ()
    assert session.divergences == []
    outcomes = _outcomes(session)
    assert "agreed" in outcomes.values()
    assert set(outcomes.values()) <= {"agreed", "unprobed"}
    assert sorted(outcomes) == [
        play.index
        for play in session.record.plays
        if play.route == "state" and not play.held
    ]


def test_the_probe_restores_the_first_and_last_interior_frames(session_of):
    """Neither the play's start nor an event: the Succession's event at
    0.65 s falls between frames, and the last frame is restored from it.
    """
    session = session_of("SuccessionReadout", RESTORE_SCENES, frame_rate=15)
    (found,) = session.probe.plays
    play = session.record.plays[found.play]
    assert found.instants == (Fraction(1, 15), Fraction(19, 15))
    assert found.bases == (play.start, Fraction(13, 20))


def test_an_event_on_a_frame_is_never_a_probed_instant(session_of):
    """Restoring across an event is no restore: the instant just before it is
    the last probed, and the play agrees.
    """
    session = session_of("EventOnLastFrame", PROBE_SCENES, frame_rate=15)
    play = session.record.plays[0]
    assert Fraction(3, 5) in play.events
    assert play.sample_at(Fraction(3, 5)) is not None
    (found,) = session.probe.plays
    assert found.outcome == "agreed"
    assert found.instants == (Fraction(1, 15), Fraction(8, 15))


def test_a_play_its_restore_refuses_is_counted_apart(session_of):
    """A global rebound after the play: no restore of it is sound, so the
    probe can't compare it, and nothing demotes it.
    """
    session = session_of("ReboundGlobal", RESTORE_SCENES, frame_rate=15)
    found = {each.play: each for each in session.probe.plays}
    assert found[0].outcome == "unrestorable"
    assert found[0].reason == "restore.global_rebound"
    assert session.probe.demoted == ()
    assert session.record.plays[0].route == "state"


def test_plays_with_fewer_than_two_interior_frames_are_counted_unprobed(session_of):
    session = session_of("ShortPlays", PROBE_SCENES, frame_rate=15)
    routes = [play.route for play in session.record.plays]
    assert routes == ["state", "state", "trusted", "state", "trusted"]
    assert session.record.plays[2].held
    assert _outcomes(session) == {0: "unprobed", 1: "unprobed", 3: "agreed"}
    assert session.probe.unprobed == 2
    found = {each.play: each for each in session.probe.plays}
    assert found[3].instants == (Fraction(2, 5), Fraction(4, 3))


def test_geometry_agrees_within_1e_9_and_paint_by_its_exact_key(session_of, planted):
    """Both drift by 1e-12 an evaluation: points within the tolerance agree,
    a paint whose key differs at all does not.
    """
    moved = session_of("PlantedGeometryDrift", PROBE_SCENES, frame_rate=15)
    assert _outcomes(moved) == {0: "agreed"}
    assert moved.record.plays[0].route == "state"
    assert moved.divergences == []

    faded = session_of("PlantedPaintDrift", PROBE_SCENES, frame_rate=15)
    assert faded.probe.demoted == (0,)
    assert faded.record.plays[0].route == "ticked"
    (found,) = [d for d in faded.divergences if d.kind == "probe"]
    assert "fill" in found.reason


def test_a_render_never_reads_a_demotion(session_of, planted):
    session = session_of("PlantedDrift", PROBE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0,)
    scene, _ = render(probe_scenes.PlantedDrift, 15)
    assert scene.record.plays[0].route == "state"
