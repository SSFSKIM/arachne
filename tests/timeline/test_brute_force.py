"""The stepping tick held against itself computed by brute force
(:mod:`arachne.timeline.reference`): every scene the stepping, soundness and
plan-read fixtures hold, run stepping, keeps the captures a brute-force run
keeps, and the brute-force run never touches the plan; nor does a stepping
run touch the brute-force run's judges of dt.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path

import pytest

from arachne.scene.scene import Scene
from arachne.timeline import recorder, reference, routes, ticking
from arachne.timeline.reference import (
    BRUTE_FORCE,
    integrates,
    scene_updater_integrates,
)
from arachne.timeline.ticking import loads_dt, scene_updater_loads_dt

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
    _forbid(
        monkeypatch,
        ticking,
        "plan_ticks",
        "loads_dt",
        "scene_updater_loads_dt",
        "_handed_slot_loaded",
        "_slot_loaded",
        "skipping",
    )
    _forbid(monkeypatch, recorder, "plan_ticks", "current_structure")
    _forbid(monkeypatch, routes, "skipping")
    _forbid(monkeypatch, ticking.TickPlan, "__init__")
    _forbid(monkeypatch, recorder.PlayRun, "stepping_ticks", "stepping_plan")
    _, session = _outcome(session_of, path, name, 4, stepping=BRUTE_FORCE)
    if session is not None:
        assert all(p.tick_plan is None for p in session.record.plays)


@pytest.mark.parametrize(("path", "name"), CASES, ids=IDS)
def test_stepping_never_enters_the_brute_force_judges(
    session_of, monkeypatch, path, name
):
    """The brute-force run's judges of dt raise: a stepping run of every
    scene meets none of them.
    """
    _forbid(
        monkeypatch,
        reference,
        "integrates",
        "scene_updater_integrates",
        "_reads_argument",
        "_reads_parameter",
    )
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


def _stores_dt_second_of_a_pair(m, dt):
    unused, dt = 1, 0.0
    return m


def _stores_dt_first_of_a_pair(m, dt):
    dt, unused = 0.0, 1
    return m


def _loops_dt_in_a_pair(m, dt):
    for dt, unused in [(0.0, 1)]:  # noqa: B007 (the store is the case)
        m.shift(0)


def _forwarding(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def _dt_in_the_third_slot(m, step, dt=0):
    m.shift(step)


def _dt_keyword_only(m, step, *, dt=0):
    m.shift(step)


def _dt_into_star_args(m, *args, dt=0):
    m.shift(args[0])


def _dead_third_slot(m, step, dt=0):
    return m


def _signed_over(function, *names):
    """``function`` with a ``__signature__`` naming ``names``, whatever its
    code declares.
    """
    function.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in names
        ]
    )
    return function


def _overridden_with_a_declared_dt(m, step, dt=0):
    m.shift(step)


def _overridden_with_a_dead_second_slot(m, unused, dt=0):
    return m


class _Updater:
    def __call__(self, m, dt):
        return m


class _Proxy:
    """Exposes the wrapped function's attributes, its ``__code__`` among
    them, while its own ``__call__`` is what runs.
    """

    def __init__(self, function):
        self.__wrapped__ = function

    def __getattr__(self, name):
        return getattr(self.__wrapped__, name)

    def __call__(self, *args):
        return args


class _Mover:
    @_forwarding
    def wrapped(self, m, dt):
        m.shift(dt)

    def dead(self, m, dt):
        return m

    def third(self, m, step, dt=0):
        m.shift(step)


@pytest.mark.parametrize(
    ("updater", "loads"),
    [
        (_dt_free_by_signature, False),
        (_reads_dt, True),
        (_reads_dt_by_frame, True),
        (_reads_dt_in_an_inner_function, True),
        (_reads_an_attribute_named_dt, False),
        (_stores_dt_second_of_a_pair, False),
        (_stores_dt_first_of_a_pair, False),
        (_loops_dt_in_a_pair, False),
        (lambda m: m, False),
        (functools.partial(_dt_free_by_signature), True),
        (_Updater(), True),
        (_forwarding(_reads_dt), True),
        (_forwarding(_dt_free_by_signature), True),
        (_Mover().wrapped, True),
        (_Mover().dead, False),
        (_dt_in_the_third_slot, True),
        (_dt_keyword_only, True),
        (_Mover().third, True),
        (_dt_into_star_args, True),
        (_signed_over(_overridden_with_a_declared_dt, "m", "dt"), True),
        (_dead_third_slot, False),
        (_signed_over(_overridden_with_a_dead_second_slot, "m", "dt"), False),
        (_Proxy(_dt_free_by_signature), True),
    ],
    ids=[
        "unread",
        "loaded",
        "frame",
        "inner",
        "attribute",
        "pair-second",
        "pair-first",
        "for-pair",
        "no-dt",
        "partial",
        "object",
        "wraps",
        "wraps-dead",
        "wraps-method",
        "method-dead",
        "third",
        "keyword",
        "method-third",
        "star-args",
        "signature-override",
        "third-dead",
        "signature-override-dead",
        "proxy",
    ],
)
def test_brute_force_and_stepping_judge_dt_alike(updater, loads):
    """A mobject updater whose signature names dt is handed it as its second
    positional argument, whatever its code calls that parameter.
    """
    assert integrates(updater) is loads
    assert loads_dt(updater) is loads


def _scene_dead(dt):
    return 0


def _scene_reads(step):
    return step


class _SceneMover:
    def dead(self, dt):
        return 0

    def reads(self, dt):
        return dt

    @staticmethod
    def nothing_handed():
        return 0

    def receiver_only(self):
        return 0


@pytest.mark.parametrize(
    ("updater", "loads"),
    [
        (_scene_dead, False),
        (_scene_reads, True),
        (lambda *args: args, True),
        (_SceneMover().dead, False),
        (_SceneMover().reads, True),
        (functools.partial(_scene_dead), True),
        (_forwarding(_scene_dead), True),
        (_SceneMover.nothing_handed, True),
        (_SceneMover().receiver_only, True),
        (_Proxy(_scene_dead), True),
    ],
    ids=[
        "dead",
        "renamed",
        "star-args",
        "method-dead",
        "method",
        "partial",
        "wraps",
        "no-slot",
        "method-no-slot",
        "proxy",
    ],
)
def test_brute_force_and_stepping_judge_scene_updaters_alike(updater, loads):
    """A scene updater is handed dt as its first positional argument,
    whatever it names it.
    """
    assert scene_updater_integrates(updater) is loads
    assert scene_updater_loads_dt(updater) is loads
