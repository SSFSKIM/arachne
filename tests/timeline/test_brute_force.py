"""The stepping tick held against itself computed by brute force
(:mod:`arachne.timeline.reference`): every scene the stepping, soundness and
plan-read fixtures hold, run stepping, keeps the captures a brute-force run
keeps, and the brute-force run never touches the plan; nor does a stepping
run touch the analysis kept for diagnostics.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path

import pytest

from arachne.scene.scene import Scene
from arachne.timeline import recorder, reference, routes, ticking
from arachne.timeline.reference import BRUTE_FORCE, integrates
from arachne.timeline.ticking import loads_dt

from . import plan_read_scenes, soundness_scenes, stepping_scenes


def _scenes(module) -> list[tuple[Path, str]]:
    return [
        (Path(module.__file__), name)
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, Scene)
        and value.__module__ == module.__name__
        and not name.startswith("_")
    ]


CASES = [
    *_scenes(stepping_scenes),
    *_scenes(soundness_scenes),
    *_scenes(plan_read_scenes),
]
IDS = [name for _, name in CASES]


def _outcome(session_of, path, name, fps, **kwargs):
    try:
        session = session_of(name, path, frame_rate=fps, **kwargs)
    except Exception as raised:  # noqa: BLE001 (the outcome is compared)
        return type(raised).__name__, None
    return None, session


@pytest.mark.parametrize("fps", [4, 15])
@pytest.mark.parametrize(("path", "name"), CASES, ids=IDS)
def test_stepping_keeps_what_brute_force_keeps(session_of, path, name, fps):
    raised, stepped = _outcome(session_of, path, name, fps)
    brute_raised, brute = _outcome(session_of, path, name, fps, stepping=BRUTE_FORCE)
    assert raised == brute_raised
    if stepped is None or brute is None:
        return
    for ours, theirs in zip(stepped.record.plays, brute.record.plays, strict=True):
        assert ours.sample_keys() == theirs.sample_keys(), (name, fps, ours.index)
        assert (ours.stepping_ticks, ours.full_ticks) == (
            theirs.stepping_ticks,
            theirs.full_ticks,
        ), (name, fps, ours.index)


def test_brute_force_is_not_ticking_everything(session_of):
    """A dt-free callback left out at the ticks no frame reads changes what
    a dt reader integrates: brute force and a run ticking everything differ.
    """
    path = Path(plan_read_scenes.__file__)
    brute = session_of("StoredLambdaRead", path, frame_rate=4, stepping=BRUTE_FORCE)
    full = session_of("StoredLambdaRead", path, frame_rate=4, stepping=False)
    assert any(p.stepping_ticks for p in brute.record.plays)
    assert any(
        a.sample_keys() != b.sample_keys()
        for a, b in zip(brute.record.plays, full.record.plays, strict=True)
    )


def test_a_recorder_takes_only_the_three_modes(session_of):
    with pytest.raises(ValueError, match="brute_force"):
        session_of("Unchanged", Path(stepping_scenes.__file__), stepping="brute")


# Independence


class _Touched(BaseException):
    """Raised by whatever a run must not enter; no handler of the engine's
    catches it.
    """


def _forbid(monkeypatch, owner, *names):
    for name in names:

        def refuse(*args, _name=name, **kwargs):
            raise _Touched(f"{owner.__name__}.{_name}")

        monkeypatch.setattr(owner, name, refuse)


@pytest.mark.parametrize(("path", "name"), CASES, ids=IDS)
def test_brute_force_never_enters_the_plan(session_of, monkeypatch, path, name):
    """Planning, the stepping path's dt judge, its stepping ticks and plans,
    and the plan itself all raise: a brute-force run of every scene meets
    none of them, and records no plan.
    """
    _forbid(monkeypatch, ticking, "plan_ticks", "loads_dt", "_code_loads", "skipping")
    _forbid(monkeypatch, recorder, "plan_ticks", "current_structure")
    _forbid(monkeypatch, routes, "skipping")
    _forbid(monkeypatch, ticking.TickPlan, "__init__")
    _forbid(monkeypatch, recorder.PlayRun, "stepping_ticks", "stepping_plan")
    _, session = _outcome(session_of, path, name, 4, stepping=BRUTE_FORCE)
    if session is not None:
        assert all(p.tick_plan is None for p in session.record.plays)


@pytest.mark.parametrize(("path", "name"), CASES, ids=IDS)
def test_stepping_never_enters_the_analysis(session_of, monkeypatch, path, name):
    """What the analysis computes for diagnostics raises: a stepping run of
    every scene meets none of it.
    """
    _forbid(
        monkeypatch,
        ticking,
        "_stepping_families",
        "reach",
        "_animation_node",
        "_updater_node",
    )
    _forbid(monkeypatch, ticking._Writes, "__init__", "effect", "plan_reads_hold")
    _forbid(monkeypatch, ticking.TickPlan, "stepping_families")
    _forbid(monkeypatch, reference, "integrates")
    _outcome(session_of, path, name, 4)


# The two dt judges


def _dt_free_by_signature(m, dt):
    m.shift(0)


def _reads_dt(m, dt):
    m.shift(dt)


def _reads_dt_by_frame(m, dt):
    return locals()["dt"]


def _reads_dt_in_an_inner_function(m, dt):
    return (lambda: dt)()


def _reads_an_attribute_named_dt(m, dt):
    return m.dt


class _Updater:
    def __call__(self, m, dt):
        return m


@pytest.mark.parametrize(
    ("updater", "loads"),
    [
        (_dt_free_by_signature, False),
        (_reads_dt, True),
        (_reads_dt_by_frame, True),
        (_reads_dt_in_an_inner_function, True),
        (_reads_an_attribute_named_dt, False),
        (lambda m: m, False),
        (functools.partial(_dt_free_by_signature), True),
        (_Updater(), True),
    ],
    ids=[
        "unread",
        "loaded",
        "frame",
        "inner",
        "attribute",
        "no-dt",
        "partial",
        "object",
    ],
)
def test_brute_force_and_stepping_judge_dt_alike(updater, loads):
    assert integrates(updater) is loads
    assert loads_dt(updater) is loads
