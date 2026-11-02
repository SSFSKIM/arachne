"""How a play is evaluated, and how its instants are answered: its route.

Every route evaluates a play forward, once, in the scene's own run, and
leaves the play's samples in the record; :func:`answer` and :func:`draw`
then read an instant through the route of the play that holds it. Which
route a play takes is decided when it begins (:mod:`.admission`).

- ``"trusted"`` (route 1): the play is evaluated at its start, at its event
  instants and at the frames a render draws, and at any other instant it is
  evaluated exactly there: the recorded state at the last of those instants
  before it is installed and the play's animations are interpolated to the
  instant (answer ``"analytic"``).
- ``"state"`` (route 2): the play is sampled at the instants its consumer
  reads (a render's frame grid; a session's 60 Hz tick) and at its events;
  an unsampled instant is :class:`Unavailable` until the planned restore work lands.
- ``"ticked"`` (route 3): the play steps on the global 60 Hz tick, its events
  taken at the first tick at or after their instants, and an instant reads
  the last tick at or before it. Sound by construction: a sample is the
  forward run's own state. A tick that no frame of the record's grid reads,
  no event falls on and that does not end the play only steps: its dt-free
  callbacks run at the frames, as CE runs them, unless stepping work may
  read them (:mod:`.ticking`).

At an instant a frame of the record's grid reads (:meth:`.Recorder.reads`),
a route evaluates the play through the scene's ``update_to_time``, as CE
does once per frame (:mod:`.hooks`); at every other instant it runs the
engine's bodies directly (:func:`evaluate_instant`), and no scene code.

A frozen frame (a held wait) is one sample on any route, and that sample
answers its whole span.
"""

from __future__ import annotations

__all__ = [
    "ROUTE_CLASSES",
    "RecordRefused",
    "Route",
    "StateRoute",
    "TickedRoute",
    "TrustedRoute",
    "Unavailable",
    "answer",
    "draw",
    "evaluate_instant",
    "snapshot_animations",
]

from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from . import ce_clock
from .clock import TICK_DT, frames_in, ticks_in
from .events import EventPlan
from . import hooks
from .hooks import animate, update_mobjects, update_self
from .reference import BRUTE_FORCE, BruteForce
from .ticking import skipping

# DIAGNOSTIC: every animation's alpha is CE's t / run_time over its float
# progression (ce_clock.alpha), here and in the hooks' animate; the exact
# instant still schedules the composition events.
hooks._alpha = _alpha = ce_clock.alpha

if TYPE_CHECKING:
    from arachne.scene.scene import Scene
    from arachne.typing import PixelArray

    from .draw import Drawing
    from .record import Capture, PlayRecord, Record
    from .recorder import PlayRun
    from .state import State


class Unavailable(LookupError):
    """The record can't answer this instant: a state-function play (route 2)
    was not sampled there, and restoring it is not built yet (planned restore work).
    """


class RecordRefused(LookupError):
    """The record answers no instant, for the reason it carries
    (:attr:`.Record.refusal`): its frames were drawn while the scene ran.
    """


class Route(Protocol):
    """A way of evaluating a play."""

    #: The name the record carries: ``"trusted"``, ``"state"`` or ``"ticked"``.
    name: str

    def run(self, run: PlayRun, plan: EventPlan) -> None:
        """Evaluate the play forward from its begun state to its end, hand
        each instant evaluated to ``run.sample``, and set ``run.end``.
        Finishing the animations is not the route's: the play does it after.
        """


def _held(run: PlayRun) -> bool:
    if run.held:
        # A frozen frame: CE runs nothing during it.
        run.sample(run.start, until=run.requested_end)
        run.end = run.requested_end
        return True
    return False


class TickedRoute:
    """Route 3: step the play on the 60 Hz tick."""

    name = "ticked"

    def run(self, run: PlayRun, plan: EventPlan) -> None:
        if _held(run):
            return
        scene = run.scene
        start, end = run.start, run.requested_end
        stop = scene.stop_condition
        ticks = list(ticks_in(start, end))
        run.set_clock(start)
        _evaluate(run, plan, start, 0.0, ticks[0] if ticks else end)
        run.started()
        run.sample(start, until=ticks[0] if ticks else end)
        # DIAGNOSTIC: CE checks the stop condition after each frame it draws, the
        # first included, and ends the play at that frame's end.
        ended = ce_clock.frame_end(start, end) if stop is not None and stop() else None
        if ended is None:
            # A verification run decides each tick apart from the plan
            # (:mod:`.reference`).
            brute = (
                BruteForce(run, ticks) if run.recorder.stepping == BRUTE_FORCE else None
            )
            stepping = run.stepping_ticks(ticks) if brute is None else frozenset()
            progress = run.progress(ticks)
            try:
                for index, tick in enumerate(progress):
                    run.set_clock(tick)
                    following = ticks[index + 1] if index + 1 < len(ticks) else end
                    if brute is not None:
                        full = brute.full(tick, following)
                        if full:
                            _evaluate(run, plan, tick, TICK_DT, following)
                        else:
                            with brute.dt_work_only():
                                evaluate_instant(scene, plan, start, tick, TICK_DT)
                    else:
                        tick_plan = (
                            run.stepping_plan(tick)
                            if tick in stepping
                            and not run.recorder.keeps(tick, following)
                            else None
                        )
                        full = tick_plan is None
                        if tick_plan is not None:
                            with skipping(tick_plan):
                                evaluate_instant(scene, plan, start, tick, TICK_DT)
                        else:
                            _evaluate(run, plan, tick, TICK_DT, following)
                    run.stepped(tick, full=full)
                    run.sample(tick, until=following)
                    if (
                        stop is not None
                        and (frame_end := ce_clock.frame_end(tick, end)) is not None
                        and stop()
                    ):  # DIAGNOSTIC: after a frame only, ending at its end, as CE
                        ended = frame_end
                        break
            finally:
                progress.close()
        if ended is not None:
            run.end = ended
            return
        run.end = end
        if not ticks or ticks[-1] < end:
            _events_at_end(run, plan)


def _events_at_end(run: PlayRun, plan: EventPlan) -> None:
    """The play ends after its last evaluated instant: the events there still
    run in the phase order before the animations finish.
    """
    if not plan.successions:
        return
    end = run.requested_end
    run.set_clock(end)
    plan.begin_instant(end)
    local = end - run.start
    for animation in run.scene.animations or []:
        animation.interpolate(_alpha(local, animation.run_time))
    plan.run_events()


class _SampledRoute:
    """Evaluate a play at its start, its event instants and the instants its
    consumer reads, and keep every one.
    """

    name = ""

    def consumer_instants(self, run: PlayRun) -> list[Fraction]:
        grid = run.recorder.grid
        if grid is None:
            # Without a grid a frame reads every tick: a play evaluates them
            # only for the scene's frame hooks.
            if run.play.frame_hooks:
                return list(ticks_in(run.start, run.requested_end))
            return []
        return frames_in(run.start, run.requested_end, grid)

    def run(self, run: PlayRun, plan: EventPlan) -> None:
        if _held(run):
            return
        start, end = run.start, run.requested_end
        instants = sorted(
            {start, *plan.instants(start, end)}
            | {t for t in self.consumer_instants(run) if start <= t < end}
        )
        progress = run.progress(instants)
        previous = start
        try:
            for index, instant in enumerate(progress):
                run.set_clock(instant)
                following = instants[index + 1] if index + 1 < len(instants) else end
                _evaluate(run, plan, instant, float(instant - previous), following)
                run.sample(instant, until=following, keep=True)
                previous = instant
        finally:
            progress.close()
        run.end = end
        _events_at_end(run, plan)


class TrustedRoute(_SampledRoute):
    """Route 1: analytic at any instant."""

    name = "trusted"


class StateRoute(_SampledRoute):
    """Route 2: sampled where its consumer reads, on the tick in a session."""

    name = "state"

    def consumer_instants(self, run: PlayRun) -> list[Fraction]:
        if run.recorder.grid is None:
            return list(ticks_in(run.start, run.requested_end))
        return super().consumer_instants(run)


#: The route of each name.
ROUTE_CLASSES: dict[str, type[Route]] = {
    "trusted": TrustedRoute,
    "state": StateRoute,
    "ticked": TickedRoute,
}


def _evaluate(
    run: PlayRun, plan: EventPlan, instant: Fraction, dt: float, following: Fraction
) -> None:
    """Evaluate the play at ``instant``, whose state holds until
    ``following``: through the scene's frame hooks where a frame reads it,
    else by the engine's bodies alone.
    """
    if run.recorder.reads(instant, following):
        run.frame(plan, instant, dt)
    else:
        evaluate_instant(run.scene, plan, run.start, instant, dt)


def evaluate_instant(
    scene: Scene, plan: EventPlan, start: Fraction, instant: Fraction, dt: float
) -> None:
    """Evaluate a play that began at ``start`` at ``instant``, in the phase
    order: the animations (each after its own copies' updaters, as in CE),
    then the composition events, then the scene's mobject updaters and its
    own updaters, which receive ``dt``. The engine's bodies of those
    methods run, never the scene's code under their names (:mod:`.hooks`).
    """
    animate(scene, plan, start, instant, dt)
    update_mobjects(scene, dt)
    update_self(scene, dt)


# The trusted algebra, evaluated at an instant


def snapshot_animations(
    animations: tuple[Any, ...],
) -> tuple[tuple[Any, dict[str, Any]], ...]:
    """Every animation of a play with what its ``__dict__`` holds now: values
    by reference, every array and list copied (CE changes some in place, and
    a read-only view changes with its base), and its path function holding
    detached copies of the arrays it binds.
    """
    from .rates import frozen_path
    from .trusted import iter_animations

    snapshot = []
    for animation in iter_animations(animations):
        state = _copied(vars(animation))
        if state.get("_path_func") is not None:
            state["_path_func"] = frozen_path(state["_path_func"])
        snapshot.append((animation, state))
    return tuple(snapshot)


def _copied(state: dict[str, Any]) -> dict[str, Any]:
    copied = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            value = value.copy()
        elif isinstance(value, list):
            value = list(value)
        copied[key] = value
    return copied


def _evaluator(
    play: PlayRecord, base: Capture, t: Fraction
) -> Callable[[Drawing], None]:
    """What brings the shown base capture of a trusted play to ``t``: its
    animations, given the state they had at the base, interpolated to ``t``
    as the forward run does there (phase 1 only: no event lies between the
    base and ``t``, since every event instant is a base).
    """
    from arachne.animation.composition import Succession

    def evaluate(drawing: Drawing) -> None:
        drawing.show(
            base, [m for a in play._animations if (m := a.mobject) is not None]
        )
        states = play._animation_states[base.seq]
        saved = [(animation, dict(vars(animation))) for animation, _ in states]
        try:
            for animation, state in states:
                vars(animation).clear()
                vars(animation).update(_copied(state))
            successions = [a for a, _ in states if isinstance(a, Succession)]
            plan = EventPlan(successions=successions, now=t)
            for succession in successions:
                succession._event_plan = plan
            plan.begin_instant(t)
            local = t - play.start
            for animation in play._animations:
                animation.interpolate(_alpha(local, animation.run_time))
        finally:
            for animation, state in saved:
                vars(animation).clear()
                vars(animation).update(state)

    return evaluate


def _resolve(
    record: Record, t: Fraction
) -> tuple[PlayRecord | None, Capture, Callable[[Drawing], None] | None, str]:
    """The play holding ``t``, the capture to show, what brings it to ``t``
    (``None`` when the capture is the instant's own) and the answer's kind.
    """
    if record.refusal is not None:
        raise RecordRefused(record.refusal)
    play = record.play_at(t)
    if play is None:
        _, capture = record.capture_at(t)
        return None, capture, None, "recorded"
    if play.held or play.route == "ticked":
        return play, play.capture_at(t), None, "recorded"
    if play.route == "state":
        sample = play.sample_at(t)
        if sample is None:
            raise Unavailable(
                f"Play {play.index} is a state-function play (route 2) the record did"
                f" not sample at {t}; answering it there needs restore, which is"
                " planned work."
            )
        return play, sample, None, "recorded"
    base = play.base_at(t)
    if base.instant == t:
        return play, base, None, "recorded"
    return play, base, _evaluator(play, base, t), "analytic"


def answer(record: Record, t: Fraction) -> State:
    """The scene's state at ``t``, from the route of the play holding it."""
    from .state import build_state

    play, capture, evaluate, kind = _resolve(record, t)
    return build_state(
        record,
        t,
        play,
        capture,
        route=play.route if play else None,
        answer=kind,
        prepare=evaluate,
    )


def draw(
    drawing: Drawing,
    record: Record,
    t: Fraction,
    static: Callable[[PlayRecord], PixelArray | None],
) -> PixelArray:
    """The frame at ``t``, drawn from the record through the route of the play
    holding it; ``static(play)`` gives the play's static picture.
    """
    play, capture, evaluate, _ = _resolve(record, t)
    if play is None:
        return drawing.final()
    return drawing.sample(play, capture, static(play), prepare=evaluate)
