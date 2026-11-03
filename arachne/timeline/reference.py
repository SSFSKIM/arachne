"""A verification mode, not a production path: the stepping tick
(:mod:`.ticking`) computed by brute force, apart from the stepping plan.

``Recorder(stepping=BRUTE_FORCE)`` (and ``Session.open(stepping=BRUTE_FORCE)``)
runs a ticked play tick by tick and decides each tick from the play alone:

- a tick runs in full (every callback) when a frame of the record's grid
  reads it, an event falls on it (the first tick at or after the event), it
  is the play's last, or the play has a stop condition; and from the first
  instant the play's updaters do not run through :meth:`.Mobject.update`
  (the scene's own ``update_mobjects`` or ``update_self``, a mobject's own
  ``update``, a camera drawing other cameras), checked at the play's start
  and at every other tick, on to the play's end;
- at every other tick, every mobject updater and every scene updater whose
  own code does not load the ``dt`` it is handed (:func:`integrates`,
  :func:`scene_updater_integrates`, judged as each is called) is left out,
  and everything else runs: the callbacks that load ``dt``, the camera's
  inputs and every interpolation.

It makes no :class:`~.ticking.TickPlan` and reads none (``PlayRecord.tick_plan``
stays ``None``), never asks :func:`~.ticking.plan_ticks`, and its judges of
``dt`` share no code or cache with :func:`~.ticking.loads_dt` and
:func:`~.ticking.scene_updater_loads_dt`. What it shares with a stepping run
is the run itself and the hooks :meth:`.Mobject.update` and
:meth:`.Scene.update_self` consult to leave an updater out
(``tech-debt-tracker.md`` lists the rest). A comparison runs a scene
stepping and brute force, and requires the same frames.
"""

from __future__ import annotations

__all__ = ["BRUTE_FORCE", "BruteForce", "integrates", "scene_updater_integrates"]

import contextlib
import dis
import inspect
import math
import types
from collections.abc import Iterator
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from .clock import TICK_RATE

if TYPE_CHECKING:
    from arachne.scene.scene import Scene

    from .recorder import PlayRun

#: The ``stepping`` value of a recorder that runs brute force.
BRUTE_FORCE = "brute_force"

#: Names through which code may read its own frame's locals without loading
#: one by name.
_FRAME_READERS = frozenset(
    {
        "locals",
        "vars",
        "eval",
        "exec",
        "_getframe",
        "currentframe",
        "stack",
        "getouterframes",
        "f_locals",
    }
)


def integrates(updater: Any) -> bool:
    """Whether a mobject's ``updater`` is handed ``dt`` and may read it. Its
    signature naming ``dt`` decides the hand-off; :meth:`.Mobject.update`
    then calls ``updater(mobject, dt)``, so what reads ``dt`` is whatever
    receives the second positional argument (:func:`_reads_argument`).
    """
    try:
        parameters = inspect.signature(updater).parameters
    except (TypeError, ValueError):
        return True
    return "dt" in parameters and _reads_argument(updater, 1)


def scene_updater_integrates(updater: Any) -> bool:
    """Whether a scene's ``updater`` may read the ``dt`` :meth:`.Scene.update_self`
    hands it as its first positional argument, whatever its code calls it.
    """
    return _reads_argument(updater, 0)


def _reads_argument(updater: Any, index: int) -> bool:
    """Whether the parameter receiving positional argument ``index`` of a
    call to ``updater`` is read. Unless its own code has a positional
    parameter there (past a bound method's receiver), the argument goes
    where nothing here follows (``*args``, a builtin, a callable object, a
    ``partial``), and counts as read.
    """
    code = _own_code(updater)
    if code is None:
        return True
    names = code.co_varnames[: code.co_argcount]
    if isinstance(updater, types.MethodType):
        names = names[1:]
    return len(names) <= index or _reads_parameter(code, names[index])


def _own_code(updater: Any) -> types.CodeType | None:
    code = getattr(getattr(updater, "__func__", updater), "__code__", None)
    return code if isinstance(code, types.CodeType) else None


def _reads_parameter(code: types.CodeType, name: str) -> bool:
    if name in code.co_cellvars or _FRAME_READERS & set(code.co_names):
        return True
    for instruction in dis.get_instructions(code):
        if _reads(instruction.opname, instruction.argval, name):
            return True
    return False


def _reads(opname: str, value: Any, name: str) -> bool:
    """Whether one instruction reads local ``name``."""
    if "FAST" not in opname and "DEREF" not in opname:
        return False
    if opname in ("STORE_FAST", "DELETE_FAST", "STORE_DEREF", "DELETE_DEREF"):
        return False
    if opname in ("STORE_FAST_STORE_FAST", "STORE_FAST_MAYBE_NULL"):
        return False
    if opname == "STORE_FAST_LOAD_FAST":
        return bool(value[1] == name)  # stores the first, loads the second
    if isinstance(value, tuple):
        return name in value  # LOAD_FAST_LOAD_FAST loads both; any other form
    return bool(value == name)


def _ungoverned(scene: Scene) -> bool:
    """Whether the scene's updaters may run other than through
    :meth:`.Mobject.update`, where nothing could leave one out.
    """
    from .admission import _update_replaced
    from .dispatch import user_overrides

    return (
        bool({"update_mobjects", "update_self"} & set(user_overrides(type(scene))))
        or _update_replaced(scene) is not None
        or bool(vars(scene.renderer.camera).get("image_mobjects_from_cameras"))
    )


class BruteForce:
    """Which ticks of one ticked play run in full, decided tick by tick.

    Made once the play has been evaluated at its start, with its ticks.
    """

    def __init__(self, run: PlayRun, ticks: list[Fraction]) -> None:
        self.run = run
        self.every = run.scene.stop_condition is not None
        self.marked = {
            Fraction(math.ceil(event * TICK_RATE), TICK_RATE)
            for event in run.play.events
        }
        if ticks:
            self.marked.add(ticks[-1])
        self.ungoverned = _ungoverned(run.scene)

    def full(self, tick: Fraction, following: Fraction) -> bool:
        """Whether ``tick``, whose state holds until ``following``, runs in
        full.
        """
        if (
            self.every
            or tick in self.marked
            or self.run.recorder.keeps(tick, following)
        ):
            return True
        self.ungoverned = self.ungoverned or _ungoverned(self.run.scene)
        return self.ungoverned

    @staticmethod
    @contextlib.contextmanager
    def dt_work_only() -> Iterator[None]:
        """Evaluate a tick leaving out every mobject updater
        :func:`integrates` rejects and every scene updater
        :func:`scene_updater_integrates` rejects.
        """
        from arachne.mobject.mobject import Mobject
        from arachne.scene.scene import Scene

        Mobject._tick_steps = integrates
        Scene._tick_steps = scene_updater_integrates
        try:
            yield
        finally:
            Mobject._tick_steps = None
            Scene._tick_steps = None
