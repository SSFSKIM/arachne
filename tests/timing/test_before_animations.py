"""The clock puts back what is true before a play's animations write, through
the timeline's ``before_animations`` hook, which only a declaring scene
registers: a scene that declares nothing records and draws as it did
without the hook; a still wait under declared timing keeps one recorded
state for a member between two boundaries, though the clock installs what
is true and then the drawing on display at every tick; and a declaring
scene ticks every play, so no play of it is drawn by interpolating its
animations from an earlier capture, which would run no clock.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import pytest

from arachne import Square
from arachne.timeline import hooks, routes
from arachne.timeline.record import RegisteredState

from .reading import boundary
from .timing_scenes import SLOW

#: Scenes that declare timing, among them every kind of play the timing
#: tests make.
DECLARED = (
    "PosedBox",
    "FadedInBox",
    "BoxBesideText",
    "OffGridThenWait",
    "Boiling",
    "SuccessionOfMoves",
    "MorphedRig",
    "PosesInTurn",
    "LayeredBackBetweenDrawings",
    "MovedThenStill",
)


def _recorded(session) -> tuple[list[bytes], list[str]]:
    """Every kept capture's content key, and frames at a few instants."""
    record = session.record
    keys = [
        RegisteredState(record, capture).content_key()
        for play in record.plays
        for capture in play.samples
    ]
    frames = [
        hashlib.sha256(session.frame(t).tobytes()).hexdigest()
        for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    ]
    return keys, frames


def test_a_scene_that_declares_nothing_records_as_without_the_hook(opened, monkeypatch):
    with_hook = opened("Undeclared", 30)
    assert id(with_hook.scene) not in hooks.BEFORE_ANIMATIONS
    kept = _recorded(with_hook)
    # As before the hook existed: no call site does anything.
    monkeypatch.setattr(hooks, "before_animations", lambda scene: None)
    monkeypatch.setattr(routes, "before_animations", lambda scene: None)
    without = opened("Undeclared", 30)
    assert _recorded(without) == kept
    assert len(set(kept[1])) > 3


def test_a_still_wait_keeps_one_recorded_state_between_two_boundaries(opened):
    session = opened("MovedThenStill", 60)
    record = session.record
    square = next(m for m in record.mobjects() if type(m) is Square)
    waits = []
    for play in record.plays:
        states: dict[Fraction, set[int]] = {}
        for capture in play.samples:
            state = record.node_state(square, capture)
            states.setdefault(boundary(capture.instant, SLOW), set()).add(id(state))
        waits.append(states)
    assert len(waits) == 2
    # In each wait, one state for each drawing; in the second, two drawings
    # (the move between 1/3 and 2/3 is shown from 2/3), though what is
    # true and what is shown stand apart until 2/3.
    for states in waits:
        for at, held in states.items():
            assert len(held) == 1, at
    second = waits[1]
    assert len({next(iter(held)) for held in second.values()}) == 2
    assert len(second) >= 3


@pytest.mark.parametrize("scene", DECLARED)
def test_a_declaring_scene_ticks_every_play(opened, monkeypatch, scene):
    def refused(*args, **kwargs):
        raise AssertionError("a play of a declaring scene was interpolated")

    monkeypatch.setattr(routes, "_evaluator", refused)
    session = opened(scene, 30)
    assert id(session.scene) in hooks.BEFORE_ANIMATIONS
    assert {play.route for play in session.record.plays} == {"ticked"}
    for k in range(0, 13):
        session.state_at(Fraction(k, 10) + Fraction(1, 70))


def _centre_x(session, t: Fraction) -> float:
    record = session.record
    square = next(m for m in record.mobjects() if type(m) is Square)
    _, capture = record.capture_at(t)
    return float(
        np.asarray(record.node_state(square, capture).get("_points"))[:, 0].mean()
    )


def test_a_scene_without_a_hash_that_declares_nothing_plays(opened):
    # The hook looks a scene up by identity, never by its hash or equality.
    session = opened("EqualScenes", 30)
    assert type(session.scene).__hash__ is None
    assert id(session.scene) not in hooks.BEFORE_ANIMATIONS
    assert len(session.record.plays) == 1
    assert _centre_x(session, session.record.duration) == pytest.approx(1)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="pre-existing: the scene-keyed registry media/clock.py _DRIVERS"
    " hashes the scene",
)
def test_a_scene_without_a_hash_declares_timing(opened):
    session = opened("EqualScenesDeclared", 60)
    assert type(session.scene).__hash__ is None
    assert id(session.scene) in hooks.BEFORE_ANIMATIONS
    # Held to 3 drawings a second: the move is shown at its last boundary.
    for k in range(45):
        t = Fraction(k, 60)
        want = 2 * float(min(boundary(t, SLOW), Fraction(1, 2)))
        assert _centre_x(session, t) == pytest.approx(want), t
    assert _centre_x(session, session.record.duration) == pytest.approx(1)
