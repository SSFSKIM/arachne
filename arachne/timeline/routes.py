"""How a play is evaluated, and how its instants are answered: its route.

Every route evaluates a play forward, once, in the scene's own run, and
leaves the play's samples in the record; :func:`answer` then reads an
instant through the route of the play that holds it.

- ``"ticked"`` (route 3): the play steps on the global 60 Hz tick, its
  events taken at the first tick at or after their instants, and its ticks
  are captured. Sound by construction: a sample is the forward run's own
  state. Every play takes this route until the trusted (1) and
  state-function (2) routes admit plays to themselves.
"""

from __future__ import annotations

__all__ = ["ROUTES", "Route", "TickedRoute", "answer", "choose_route"]

from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING, Protocol

from .clock import TICK_DT, ticks_in
from .events import EventPlan, plan_events

if TYPE_CHECKING:
    from arachne.scene.scene import Scene

    from .record import Record
    from .recorder import PlayRun
    from .state import State


class Route(Protocol):
    """A way of evaluating a play."""

    #: The name the record carries: ``"trusted"``, ``"state"`` or ``"ticked"``.
    name: str

    def run(self, run: PlayRun) -> None:
        """Evaluate the play forward from its begun state to its end, hand
        each instant evaluated to ``run.sample``, and set ``run.end``.
        Finishing the animations is not the route's: the play does it after.
        """


class TickedRoute:
    """Route 3: step the play on the 60 Hz tick."""

    name = "ticked"

    def run(self, run: PlayRun) -> None:
        scene = run.scene
        start, end = run.start, run.requested_end
        if run.held:
            # A frozen frame: CE runs nothing during it.
            run.sample(start, until=end)
            run.end = end
            return
        plan = plan_events(scene.animations or [], start)
        stop = scene.stop_condition
        try:
            ticks = list(ticks_in(start, end))
            run.set_clock(start)
            evaluate_instant(scene, plan, start, start, 0.0)
            run.sample(start, until=ticks[0] if ticks else end)
            ended = start if stop is not None and stop() else None
            if ended is None:
                progress = run.progress(ticks)
                try:
                    for index, tick in enumerate(progress):
                        run.set_clock(tick)
                        evaluate_instant(scene, plan, start, tick, TICK_DT)
                        following = ticks[index + 1] if index + 1 < len(ticks) else end
                        run.sample(tick, until=following)
                        if stop is not None and stop():
                            ended = tick
                            break
                finally:
                    progress.close()
            if ended is not None:
                run.end = ended
                return
            run.end = end
            if plan.successions and (not ticks or ticks[-1] < end):
                # The play ends between ticks: the events there still run in
                # the phase order before the animations finish.
                run.set_clock(end)
                plan.begin_instant(end)
                local = end - start
                for animation in scene.animations or []:
                    animation.interpolate(_alpha(local, animation.run_time))
                plan.run_events()
        finally:
            plan.release()


def _alpha(local: Fraction, run_time: float) -> float:
    """CE's ``t / run_time`` for one animation; 1 when it has no length."""
    if run_time == 0:
        return 1.0
    return float(local) / run_time


def evaluate_instant(
    scene: Scene, plan: EventPlan, start: Fraction, instant: Fraction, dt: float
) -> None:
    """Evaluate a play that began at ``start`` at ``instant``, in the phase
    order: the animations (each after its own copies' updaters, as in CE),
    then the composition events, then the scene's mobject updaters and its
    own updaters, which receive ``dt``.
    """
    plan.begin_instant(instant)
    local = instant - start
    for animation in scene.animations or []:
        animation.update_mobjects(dt)
        animation.interpolate(_alpha(local, animation.run_time))
    plan.run_events()
    scene.update_mobjects(dt)
    scene.update_self(dt)


def choose_route(scene: Scene) -> Route:
    """The route the play about to run takes."""
    return TickedRoute()


def _answer_recorded(record: Record, t: Fraction) -> State:
    from .state import build_state

    play, capture = record.capture_at(t)
    return build_state(
        record,
        t,
        play,
        capture,
        route=play.route if play else None,
        answer="recorded",
    )


#: How each route answers an instant of one of its plays.
ROUTES: dict[str, Callable[[Record, Fraction], State]] = {
    "ticked": _answer_recorded,
}


def answer(record: Record, t: Fraction) -> State:
    """The scene's state at ``t``, from the route of the play holding it."""
    play = record.play_at(t)
    if play is None:
        return _answer_recorded(record, t)
    return ROUTES[play.route](record, t)
