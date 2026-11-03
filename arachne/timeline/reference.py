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
- at every other tick, every updater whose own code does not load ``dt``
  (:func:`integrates`, judged as it is called) is left out, and everything
  else runs: the dt updaters, the scene's updaters, the camera's inputs and
  every interpolation.

It makes no :class:`~.ticking.TickPlan` and reads none (``PlayRecord.tick_plan``
stays ``None``), never asks :func:`~.ticking.plan_ticks`, and its ``dt``
judge shares no code or cache with :func:`~.ticking.loads_dt`. What it shares
with a stepping run is the run itself and the hook :meth:`.Mobject.update`
consults to leave an updater out. A comparison runs a scene stepping and
brute force, and requires the same frames.
"""

from __future__ import annotations

__all__ = ["BRUTE_FORCE", "BruteForce", "integrates"]

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

#: Instructions on a function's own locals or cells that read none.
_NO_READ = frozenset(
    {
        "STORE_FAST",
        "STORE_FAST_STORE_FAST",
        "STORE_FAST_MAYBE_NULL",
        "DELETE_FAST",
        "STORE_DEREF",
        "DELETE_DEREF",
    }
)


def integrates(updater: Any) -> bool:
    """Whether ``updater`` has a ``dt`` parameter its own code may read: a
    load of the local, an inner function closing over it, a read of its own
    frame, or no Python code to look at.
    """
    try:
        parameters = inspect.signature(updater).parameters
    except (TypeError, ValueError):
        return True
    if "dt" not in parameters:
        return False
    code = getattr(getattr(updater, "__func__", updater), "__code__", None)
    if not isinstance(code, types.CodeType):
        return True
    if "dt" in code.co_cellvars or _FRAME_READERS & set(code.co_names):
        return True
    return any(_reads(instruction, "dt") for instruction in dis.get_instructions(code))


def _reads(instruction: dis.Instruction, name: str) -> bool:
    opname = instruction.opname
    if ("FAST" not in opname and "DEREF" not in opname) or opname in _NO_READ:
        return False
    value = instruction.argval
    if opname == "STORE_FAST_LOAD_FAST":
        # (stored, loaded): only the second is read.
        return bool(value[1] == name)
    return value == name or (isinstance(value, tuple) and name in value)


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
        """Evaluate a tick leaving out every updater :func:`integrates`
        rejects.
        """
        from arachne.mobject.mobject import Mobject

        Mobject._tick_steps = integrates
        try:
            yield
        finally:
            Mobject._tick_steps = None
