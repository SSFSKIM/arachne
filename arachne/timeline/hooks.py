"""The scene's hooks: CE's ``Scene`` methods, called where CE calls them.

CE's renderer calls ``scene.play_internal()`` once for each play that is
not a frozen frame, and, inside it, ``scene.update_to_time(t)`` once per
frame with the play-local frame time, which calls ``update_mobjects``,
``update_meshes`` and ``update_self`` with the time since the last frame;
after the last frame, ``update_mobjects(0)``. A play here calls them the
same way, so a scene's overrides and the wrappers set on its object run as
they run on CE, while the methods' own bodies do the engine's work:

- ``play_internal``'s body is the play's evaluation (its route), then the
  animations' finish and cleanup.
- The frame hooks run at the instants a frame of the record's grid reads
  (:meth:`.Recorder.reads`; every 60 Hz tick without a grid), with the clock
  at that instant. ``update_to_time``'s body evaluates the instant (:func:`animate`)
  and calls the other three; ``update_mobjects``' and ``update_self``'s
  bodies (:func:`update_mobjects`, :func:`update_self`) run the updaters with
  the play's own ``dt`` there, whatever ``dt`` a hook is shown. At every
  other instant (an event, a tick no frame reads) the play runs the bodies
  directly and no scene code: a ``dt`` updater integrates once per
  evaluated instant with that instant's ``dt``, hooks or none.

Deliberate break 8 (a Scene override Arachne cannot honour where CE calls it
raises SceneHookRefused), approved 2026-11-03: an override of the time
progression (CE's frame instants, which the exact clock replaces), a
``play_internal`` that never runs its body, runs it twice or asks to skip
rendering, and a frame hook that shows its body another time than the
frame's, never reaches its body or reaches it twice.
"""

from __future__ import annotations

__all__ = [
    "FRAME_HOOKS",
    "SCENE_HOOKS",
    "TIME_PROGRESSION",
    "FrameInstant",
    "SceneHookRefused",
    "animate",
    "update_mobjects",
    "update_self",
]

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arachne.scene.scene import Scene

    from .events import EventPlan

#: The hooks CE calls at every frame of a play: ``update_to_time``, and the
#: three it calls.
FRAME_HOOKS = ("update_to_time", "update_mobjects", "update_meshes", "update_self")
#: The methods CE uses to choose a play's frame instants.
TIME_PROGRESSION = ("get_time_progression", "_get_animation_time_progression")
#: Every ``Scene`` method a play dispatches as CE does, beyond those it always
#: called (``play``, ``add`` and the rest): the scene's code under these names
#: runs where CE runs it, or is refused.
SCENE_HOOKS = (*FRAME_HOOKS, "play_internal", "get_run_time", *TIME_PROGRESSION)
#: The frame hooks whose bodies must be reached once at every frame.
_REACHED = ("update_to_time", "update_mobjects", "update_self")


class SceneHookRefused(Exception):
    """A scene overrides one of its hooks in a way a play cannot honour
    where CE calls it. ``reason`` is the stable code (``hook.time_progression``,
    ``hook.play_internal.skipped``, ``hook.update_to_time.time``,
    ``hook.update_mobjects.replaced`` and the rest).

    Deliberate break 8 (a Scene override Arachne cannot honour where CE calls
    it raises SceneHookRefused), approved 2026-11-03.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class FrameInstant:
    """One frame-read instant of a play while its frame hooks run: what CE
    shows the hooks (the play-local time ``t`` and the time since the play's
    last frame) and what their bodies evaluate (the exact instant and the
    play's ``dt`` there), with how often each body was reached.
    """

    def __init__(
        self,
        plan: EventPlan,
        start: Fraction,
        instant: Fraction,
        dt: float,
        frame_dt: float,
    ) -> None:
        self.plan = plan
        self.start = start
        self.instant = instant
        #: The play's own dt at the instant: the tick's, or the time since its
        #: last evaluated instant.
        self.dt = dt
        #: The time since the play's last frame-read instant, 0 at its first:
        #: what CE hands the hooks.
        self.frame_dt = frame_dt
        #: The play-local time CE passes ``update_to_time``.
        self.t = float(instant - start)
        self.reached = dict.fromkeys(_REACHED, 0)

    def reach(self, name: str) -> None:
        """The body of the hook ``name`` runs at this frame."""
        if name not in self.reached:
            return
        self.reached[name] += 1
        if self.reached[name] > 1:
            raise SceneHookRefused(
                f"hook.{name}.twice",
                f"the scene's {name} reached Scene.{name} twice at one frame, where"
                " the play evaluates each instant once",
            )

    def reach_time(self, t: float) -> None:
        """``update_to_time``'s body runs at this frame, shown ``t``."""
        self.reach("update_to_time")
        if float(t) != self.t:
            raise SceneHookRefused(
                "hook.update_to_time.time",
                f"the scene's update_to_time passed Scene.update_to_time t={t}"
                f" at the frame t={self.t}: the play's instants are its clock's",
            )

    def check(self) -> None:
        """Refuse a frame whose hooks returned without reaching a body: the
        play runs the bodies at every other instant, so replacing one would
        hold only at the frames.
        """
        for name in _REACHED:
            if not self.reached[name]:
                raise SceneHookRefused(
                    f"hook.{name}.replaced",
                    f"the scene's {name} returned without calling Scene.{name},"
                    " which the play runs at every instant between frames",
                )


def animate(
    scene: Scene, plan: EventPlan, start: Fraction, instant: Fraction, dt: float
) -> None:
    """Evaluate the animations of a play that began at ``start`` at
    ``instant``: each after its own copies' updaters (handed ``dt``), as in
    CE, then the composition events.
    """
    plan.begin_instant(instant)
    local = instant - start
    for animation in scene.animations or []:
        animation.update_mobjects(dt)
        animation.interpolate(_alpha(local, animation.run_time))
    plan.run_events()


def _alpha(local: Fraction, run_time: float) -> float:
    """CE's ``t / run_time`` for one animation; 1 when it has no length."""
    if run_time == 0:
        return 1.0
    return float(local) / run_time


def update_mobjects(scene: Scene, dt: float) -> None:
    """The body of ``Scene.update_mobjects``: every mobject's updaters."""
    for mobject in scene.mobjects:
        mobject.update(dt)


def update_self(scene: Scene, dt: float) -> None:
    """The body of ``Scene.update_self``: the scene's own updaters, after
    every mobject's.
    """
    from arachne.scene.scene import Scene

    steps = Scene._tick_steps
    for func in scene.updaters:
        # A stepping tick runs only the updaters whose code loads dt.
        if steps is not None and not steps(func):
            continue
        func(dt)
