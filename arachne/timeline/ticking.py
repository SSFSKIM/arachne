"""Which work of a ticked play runs at a tick no frame reads: the stepping tick.

A ticked play (route 3) is evaluated on the 60 Hz tick, so that what
integrates over time does so the same way whatever the frame rate. Only that
work needs the tick (a dt updater receives exactly 1/60 at each), and
everything else is evaluated where CE evaluates it, at the instants a frame
reads. So a ticked play runs a **full tick** (every callback, in CE's phase
order) at each instant a frame of the render's grid reads, at its event
instants, at its last tick and throughout a play with a stop condition, and
every other tick only **steps**: between frames it runs

- every animation's interpolation, the functions of ``UpdateFromFunc`` and
  ``UpdateFromAlphaFunc`` included (they are interpolation);
- the camera's inputs, its fixed-orientation centre functions included (as
  every tick evaluates them);
- the callbacks whose own code loads the ``dt`` they are handed: a mobject's
  updater whose signature names ``dt`` (:func:`loads_dt`), which
  :meth:`.Mobject.update` then hands ``dt`` as its second positional
  argument, and a scene's updater, which :meth:`.Scene.update_self` always
  hands ``dt`` as its first (:func:`scene_updater_loads_dt`);

and nothing else. Whatever the parameter receiving that argument is named, a
callback counts as loading ``dt`` unless its own code has a positional
parameter there and never reads it. Only code the call is certain to run is
read (a plain function's, a bound method's whose function is plain): a
wrapper forwarding ``*args``, a ``partial``, a callable object, a proxy or a
builtin counts as loading.
One that never loads it is dt-free, frame-rate dependent in CE exactly like
one without the parameter, and runs at the full ticks only, whoever reads
what it writes: a dt callback reading a dt-free value between frames sees it
as of the last full tick, as CE shows it once per frame. Each callback is
judged as it is called (:func:`skipping`, consulted by
:meth:`.Mobject.update` and the body of :meth:`.Scene.update_self`), so one
that joins as the play runs is judged the same way. A play whose updaters do
not run there (a mobject's own ``update``, a camera drawing other cameras)
ticks everything, under the reason ``tick.ungoverned``. The scene's own
``update_to_time``, ``update_mobjects`` and ``update_self`` are frame hooks
(:mod:`.hooks`): they run at the full ticks a frame reads, as CE runs them
once per frame, and a stepping tick runs the engine's bodies.

:func:`plan_ticks` makes the play's :class:`TickPlan`: whether its stepping
ticks may leave anything out, and the structure (:func:`current_structure`)
it was made for. Before every stepping tick the play compares the structure
with its plan's and, when they differ, plans again. That finds a play whose
updaters no longer run where a tick could leave one out, which then ticks
everything from that tick on; otherwise the tick steps as it would have: a
structure change alone makes no tick full. Every plan a play stepped by is
kept, with the tick it governed from (``PlayRecord.tick_plans``).
"""

from __future__ import annotations

__all__ = [
    "UNGOVERNED",
    "TickPlan",
    "current_structure",
    "loads_dt",
    "plan_ticks",
    "scene_updater_loads_dt",
    "skipping",
]

import contextlib
import dis
import functools
import inspect
import types
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arachne.animation.animation import Animation
    from arachne.scene.scene import Scene

#: The reason family of a play that ticks everything, its updaters not
#: running where a stepping tick could leave one out.
UNGOVERNED = "tick.ungoverned"


@dataclass(frozen=True)
class TickPlan:
    """A ticked play's stepping plan: whether its stepping ticks may leave
    anything out, and the structure it was made for.
    """

    #: Why the play ticks everything, when it does (``tick.ungoverned: …``).
    unbounded: str | None = None
    #: The structure the plan was made for (:func:`current_structure`): a
    #: stepping tick that finds another plans again before it steps.
    structure: tuple[int, ...] = ()
    #: Whether the play stepped by it. A run that ticks everything
    #: (``Recorder(stepping=False)``) makes and remakes its plans where a
    #: stepping run would, and applies none: every tick stays full.
    applied: bool = True

    @property
    def steps_only(self) -> bool:
        """Whether a stepping tick does less than a full one (when the plan
        is :attr:`applied`).
        """
        return self.unbounded is None


@contextlib.contextmanager
def skipping(plan: TickPlan) -> Iterator[None]:
    """Run a stepping tick: :meth:`.Mobject.update` runs only the updaters
    whose code loads ``dt`` (:func:`loads_dt`), and :meth:`.Scene.update_self`
    only the scene's updaters that do (:func:`scene_updater_loads_dt`),
    each decided as it runs, so an updater that joined since the plan was
    made is judged the same way.
    """
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene

    Mobject._tick_steps = loads_dt
    Scene._tick_steps = scene_updater_loads_dt
    try:
        yield
    finally:
        Mobject._tick_steps = None
        Scene._tick_steps = None


def current_structure(scene: Scene, animations: Iterable[Animation]) -> tuple[int, ...]:
    """What a plan weighs that callbacks may change as a play runs: every
    updater the scene's tick may run (the scene's own and every mobject's),
    and every mobject's place, under the scene or in its parent (a member
    joining a family another callback writes changes what a reader of the
    family reads). Every stepping tick compares it with its plan's.
    """
    # One walk, in preorder: each mobject's id, its updaters (counted), how
    # many members it has, then each member's walk. Cheap enough for every
    # stepping tick: no family lists, no sets.
    found: list[int] = [len(scene.updaters), *map(id, scene.updaters)]
    pending = list(reversed((*_copies(animations), *scene.mobjects)))
    found.append(len(pending))
    while pending:
        mob = pending.pop()
        updaters, members = mob.updaters, mob.submobjects
        found.append(id(mob))
        found.append(len(updaters))
        found.extend(map(id, updaters))
        found.append(len(members))
        pending.extend(reversed(members))
    return tuple(found)


# Which updaters integrate


def takes_dt(updater: Any) -> bool:
    """Whether ``updater`` has a ``dt`` parameter, as :meth:`.Mobject.update`
    decides which form to call.
    """
    try:
        return "dt" in inspect.signature(updater).parameters
    except (TypeError, ValueError):
        return True


@functools.cache
def _slot_loaded(code: types.CodeType, slot: int) -> bool:
    """Whether ``code`` may read its positional parameter ``slot``: a load
    of it, an inner function closing over it, or a read of its own frame.
    """
    name = code.co_varnames[slot]
    if name in code.co_cellvars:
        return True  # an inner function reads it
    if _FRAME_ACCESS & set(code.co_names):
        return True  # it may read its own frame's locals by name
    for instruction in dis.get_instructions(code):
        opname = instruction.opname
        if "FAST" not in opname and "DEREF" not in opname:
            continue
        value = instruction.argval
        slots = value if isinstance(value, tuple) else (value,)
        # Which of the instruction's slots it loads: a form not listed here
        # may load any it names.
        loaded = _LOADED_SLOTS.get(opname, tuple(range(len(slots))))
        if any(slots[k] == name for k in loaded):
            return True
    return False


#: The slots each instruction on a function's locals or cells loads, by
#: position in its argument (the 3.13 superinstructions name two: a
#: ``STORE_FAST_LOAD_FAST`` stores its first and loads its second).
_LOADED_SLOTS: dict[str, tuple[int, ...]] = {
    "LOAD_FAST": (0,),
    "LOAD_FAST_CHECK": (0,),
    "LOAD_FAST_AND_CLEAR": (0,),
    "LOAD_FAST_LOAD_FAST": (0, 1),
    "STORE_FAST_LOAD_FAST": (1,),
    "LOAD_DEREF": (0,),
    "STORE_FAST": (),
    "STORE_FAST_STORE_FAST": (),
    "STORE_FAST_MAYBE_NULL": (),
    "DELETE_FAST": (),
    "STORE_DEREF": (),
    "DELETE_DEREF": (),
}

#: Names through which code may read its own frame's locals without loading
#: them: ``locals()``, ``vars()``, ``eval``/``exec`` (their default
#: namespace), and frame objects (``sys._getframe``, ``inspect`` helpers,
#: ``f_locals``).
_FRAME_ACCESS = frozenset(
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


def loads_dt(updater: Any) -> bool:
    """Whether a mobject's ``updater`` integrates: it is handed ``dt`` (its
    signature names it, as :meth:`.Mobject.update` decides, which then
    calls ``updater(mobject, dt)``) and its own code may read the slot the
    second positional argument lands in, whatever that slot is named
    (:func:`_handed_slot_loaded`).
    """
    return takes_dt(updater) and _handed_slot_loaded(updater, 1)


def scene_updater_loads_dt(updater: Any) -> bool:
    """Whether a scene's ``updater`` integrates. :meth:`.Scene.update_self`
    hands every one ``dt`` as its first positional argument, whatever it
    names it (:func:`_handed_slot_loaded`).
    """
    return _handed_slot_loaded(updater, 0)


def _handed_slot_loaded(updater: Any, position: int) -> bool:
    """Whether ``updater``'s own code may read the positional argument
    ``position`` of a call to it (a bound method's receiver comes before
    it). Only the code a call is known to run is read: a plain function's,
    or a bound method's whose function is plain, by exact type, so nothing
    can present another's code as its own. Anything else counts as reading
    (a builtin, a ``partial``, a callable object, a proxy exposing a
    function's ``__code__`` while its ``__call__`` runs), and so does an
    argument no positional parameter of the code receives (it lands in
    ``*args``, or the call has nowhere to put it).
    """
    slot = position
    function = updater
    if type(updater) is types.MethodType:
        function = updater.__func__
        slot += 1
    if type(function) is not types.FunctionType:
        return True
    code = function.__code__
    if code.co_argcount <= slot:
        return True
    return _slot_loaded(code, slot)


def _copies(animations: Iterable[Animation]) -> list[Any]:
    """The mobjects the animations update before interpolating (their
    copies' updaters run there, as in CE).
    """
    from arachne.animation.composition import AnimationGroup

    from . import trusted

    found: list[Any] = []
    for animation in trusted.iter_animations(list(animations)):
        if isinstance(animation, AnimationGroup):
            continue
        with contextlib.suppress(AttributeError):
            found.extend(animation.get_all_mobjects_to_update())
    return found


def plan_ticks(scene: Scene, animations: list[Animation]) -> TickPlan:
    """The stepping plan of the ticked play ``scene`` is about to evaluate
    (its animations begun).

    A stepping tick runs every updater whose code loads ``dt`` and leaves out
    every other, judged as each is called (:func:`skipping`): the plan
    decides only whether the play's stepping ticks may leave anything out,
    and records the structure a stepping tick compares. The play ticks
    everything when its updaters do not run through :meth:`.Mobject.update`
    and the body of :meth:`.Scene.update_self`, where nothing could leave one
    out. The scene's own frame hooks never run at a stepping tick
    (:mod:`.hooks`), so they leave the plan as it is.
    """
    from .admission import _update_replaced

    def unbounded(why: str) -> TickPlan:
        return TickPlan(unbounded=f"{UNGOVERNED}: {why}")

    why = _update_replaced(scene)
    if why is not None:
        return unbounded(why)
    if vars(scene.renderer.camera).get("image_mobjects_from_cameras"):
        return unbounded(
            "the camera draws other cameras, whose inputs it does not weigh"
        )
    return TickPlan(structure=current_structure(scene, animations))
