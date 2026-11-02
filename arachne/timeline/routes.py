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
  at any other instant it is restored from its latest kept capture before
  the instant and evaluated there (answer ``"restored"``, :mod:`.restore`).
- ``"ticked"`` (route 3): the play steps on the global 60 Hz tick, its events
  taken at the first tick at or after their instants, and an instant reads
  the last capture a frame of the record's grid read at or before it: the
  frame a render shows there, an earlier play's before the play's first
  frame, never a capture no frame drew (answer ``"recorded"``, the state
  naming the instant it reflects). Sound by construction: a sample is the
  forward run's own state. A tick that no frame of the record's grid reads, no event falls on
  and that does not end the play only steps: its dt-free callbacks run at
  the frames, as CE runs them, unless stepping work may read them
  (:mod:`.ticking`), and the record keeps none of them.

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

import math
from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import numpy as np

from . import ce_clock
from .clock import TICK_RATE, frames_in, ticks_in
from .events import EventPlan
from . import hooks
from .hooks import animate, update_mobjects, update_self
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
    was not sampled there and can't be restored there
    (:class:`.restore.Unrestorable`).
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
        ticks, dts = ce_clock.frame_steps(start, end)  # DIAGNOSTIC: CE's dt
        run.set_clock(start)
        _evaluate(run, plan, start, 0.0, ticks[0] if ticks else end)
        run.started()
        run.sample(start, until=ticks[0] if ticks else end)
        # DIAGNOSTIC: CE checks the stop condition after each frame it draws, the
        # first included, and ends the play at that frame's end.
        ended = ce_clock.frame_end(start, end) if stop is not None and stop() else None
        if ended is None:
            # DIAGNOSTIC: every step is a full one, as in CE (no plan, and no
            # brute-force verification either: CE has neither).
            stepping: frozenset = frozenset()
            progress = run.progress(ticks)
            try:
                for index, tick in enumerate(progress):
                    run.set_clock(tick)
                    following = ticks[index + 1] if index + 1 < len(ticks) else end
                    tick_plan = (
                        run.stepping_plan(tick)
                        if tick in stepping and not run.recorder.keeps(tick, following)
                        else None
                    )
                    if tick_plan is not None:
                        with skipping(tick_plan):
                            evaluate_instant(scene, plan, start, tick, dts[index])
                    else:
                        _evaluate(run, plan, tick, dts[index], following)
                    run.stepped(tick, full=tick_plan is None)
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
        run.set_clock(end)  # DIAGNOSTIC: CE finishes the play with its clock at the end


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
    """Evaluate a play at its start, its event instants, the instants its
    consumer reads and those the recorder was asked for, and keep every one.
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
        asked = run.recorder.instants
        instants = sorted(
            {start, *plan.instants(start, end)}
            | {t for t in self.consumer_instants(run) if start <= t < end}
            | {t for t in asked if start <= t < end}
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


class _Resolved(NamedTuple):
    """How the record answers an instant: the play holding it, the capture
    to show, what brings that capture to the instant (``None`` when the
    capture is the instant's own state), the answer's kind and the instant
    whose state the answer is.
    """

    play: PlayRecord | None
    capture: Capture
    prepare: Callable[[Drawing], None] | None
    kind: str
    as_of: Fraction


def _last_frame(record: Record, t: Fraction) -> Fraction:
    """The last frame of the record's grid at or before ``t``: every 60 Hz
    tick without a grid.
    """
    grid = record.grid if record.grid is not None else Fraction(TICK_RATE)
    return Fraction(math.floor(t * grid)) / grid


def _resolve(record: Record, t: Fraction) -> _Resolved:
    """How the record answers ``t``, by the route of the play holding it."""
    if record.refusal is not None:
        raise RecordRefused(record.refusal)
    return _route(record, t)


def _route(record: Record, t: Fraction) -> _Resolved:
    """How the route of the play holding ``t`` answers it from the record's
    captures, whether or not the record can draw them: a forward run draws
    what its record can't while the scene runs (:mod:`.fallback`).
    """
    play = record.play_at(t)
    if play is None:
        _, capture = record.capture_at(t)
        return _Resolved(None, capture, None, "recorded", capture.instant)
    if play.held:
        capture = play.capture_at(t)
        return _Resolved(play, capture, None, "recorded", capture.instant)
    if play.route == "ticked":
        # A ticked play's record keeps only the ticks a frame of its grid
        # reads (not its events, nor its last tick unless a frame reads
        # them). The answer is the state the last frame at or before t
        # shows, as a render shows it there.
        frame = _last_frame(record, t)
        if frame >= play.start:
            # A tick the record kept, since a frame reads it.
            capture = play.capture_at(frame)
            return _Resolved(play, capture, None, "recorded", capture.instant)
        # Before the play's first frame the frame is an earlier play's, and
        # that play's route answers it: a trusted or state play without a
        # frame hook evaluates the frames of a record without a grid only
        # when asked, so its capture there may be an earlier instant's.
        return _route(record, frame)
    if play.route == "state":
        sample = play.sample_at(t)
        if sample is not None:
            return _Resolved(play, sample, None, "recorded", t)
        from .restore import restorer

        base, prepare = restorer(record, play, t)
        return _Resolved(play, base, prepare, "restored", t)
    base = play.base_at(t)
    if base.instant == t:
        return _Resolved(play, base, None, "recorded", t)
    return _Resolved(play, base, _evaluator(play, base, t), "analytic", t)


def answer(record: Record, t: Fraction) -> State:
    """The scene's state at ``t``, from the route of the play holding it."""
    from .state import build_state

    resolved = _resolve(record, t)
    play = resolved.play
    return build_state(
        record,
        t,
        play,
        resolved.capture,
        route=play.route if play else None,
        answer=resolved.kind,
        as_of=resolved.as_of,
        prepare=resolved.prepare,
    )


def draw(
    drawing: Drawing,
    record: Record,
    t: Fraction,
    static: Callable[[PlayRecord], PixelArray | None],
) -> PixelArray:
    """The frame at ``t``, drawn from the record through the route of the play
    holding it; ``static(play)`` gives the play's static picture. A restored
    frame takes no static picture (:func:`_restored_frame`).
    """
    return _draw(drawing, _resolve(record, t), static)


def _draw(
    drawing: Drawing,
    resolved: _Resolved,
    static: Callable[[PlayRecord], PixelArray | None],
) -> PixelArray:
    """The frame of an instant the record answers as ``resolved`` says."""
    play = resolved.play
    if play is None:
        return drawing.final()
    if resolved.kind == "restored":
        return _restored_frame(drawing, resolved)
    return drawing.sample(
        play, resolved.capture, static(play), prepare=resolved.prepare
    )


def _restored_frame(drawing: Drawing, resolved: _Resolved) -> PixelArray:
    """A restored instant's frame, with no static picture: a picture kept
    from the record's captures, or a layer of whatever they show unchanged,
    knows nothing of the values a restore gives. The frame is drawn whole,
    every mobject at its state at the instant, in a render's painter order:
    the frame's static list, rebuilt as a frame drawn without the static
    layer rebuilds it, then the play's moving mobjects over it.
    """
    from .draw import static_list

    capture, prepare = resolved.capture, resolved.prepare
    assert prepare is not None
    camera = drawing.camera
    roots = capture.draw_list()
    targets = list(capture.moving) or roots
    drawing.show(capture, [*roots, *targets])
    prepare(drawing)
    camera.reset()
    camera.capture_mobjects(static_list(camera, roots, capture.moving))
    camera.capture_mobjects(targets)
    return camera.pixel_array.copy()
