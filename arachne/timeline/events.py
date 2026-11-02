"""Event instants of a play's compositions, and the phase order they run in.

``AnimationGroup`` and ``LaggedStart`` begin every part when the group
begins (CE's eager semantics); only ``Succession`` begins parts later, when
its group time reaches the end of the active part. While a play runs, those
events are taken out of each ``Succession``'s ``interpolate`` and run by the
play, so that at every evaluated instant the phases come in one order for
the whole play:

1. evaluate the running parts at the instant,
2. finish the parts whose window has ended,
3. begin the parts whose window has started, capturing state that already
   reflects 1 and 2,
4. run the callbacks (the scene's and the mobjects' updaters).

Ties resolve in composition order. CE instead handles a sample's animations
one after another, so a ``Succession`` part begun there missed what a later
animation of the same play did at that sample.

An event's instant is exact when the chain of time maps above it is linear
(:func:`plan_events` solves it); where CE's monotone nonlinear rate functions
take part, :func:`crossing_instant` finds it by bisection, deterministically.
A nonmonotone or custom timing has no instant (:attr:`EventPlan.determined`
is false, and the play ticks): its parts begin at the first tick at which the
group time has crossed. A ticked play takes every event at the first tick at
or after its instant.
"""

from __future__ import annotations

__all__ = [
    "MONOTONE_RATE_FUNCTIONS",
    "EventPlan",
    "crossing_instant",
    "plan_events",
]

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from arachne.utils import rate_functions as rf

from .clock import _simplest_in, exact
from .rates import MONOTONE_RATE_FUNCTIONS, is_monotone

if TYPE_CHECKING:
    from arachne.animation.animation import Animation
    from arachne.animation.composition import Succession

#: How close bisection brings an instant, in seconds.
BISECTION_TOLERANCE = Fraction(1, 10**9)


@dataclass(frozen=True)
class _Affine:
    """``value(t) = slope * t + offset``, exactly."""

    slope: Fraction
    offset: Fraction

    def solve(self, value: Fraction) -> Fraction | None:
        if self.slope == 0:
            return None
        return (value - self.offset) / self.slope

    def then(self, scale: Fraction, shift: Fraction) -> _Affine:
        """``scale * value(t) + shift``."""
        return _Affine(self.slope * scale, self.offset * scale + shift)


@dataclass
class EventPlan:
    """Every ``Succession`` of one play, in composition order, with the
    instant the play is evaluating now.
    """

    successions: list[Succession] = field(default_factory=list)
    now: Fraction = Fraction(0)
    #: Whether every event of the play has an instant: false when a
    #: Succession's time runs through a nonmonotone or custom timing.
    determined: bool = True

    def instants(self, start: Fraction, end: Fraction) -> list[Fraction]:
        """Every event instant in ``(start, end)``, in order."""
        found = {
            instant
            for succession in self.successions
            for instant in succession._event_instants
            if instant is not None and start < instant < end
        }
        return sorted(found)

    def begin_instant(self, now: Fraction) -> None:
        """Start the phases of the instant ``now``."""
        self.now = now
        for succession in self.successions:
            succession._pending_time = None

    def run_events(self) -> None:
        """Phases 2 and 3 at the current instant.

        Every due part finishes, then every waiting part begins, again until
        nothing is left to finish or begin; the parts begun are then
        evaluated at the instant (as CE evaluates a part in the sample that
        began it), which can make parts nested in them due in turn.
        """
        while True:
            begun: list[Succession] = []
            while True:
                finished = [s for s in self.successions if s._finish_due_part()]
                started = [s for s in self.successions if s._begin_next_part()]
                begun.extend(s for s in started if s not in begun)
                if not finished and not started:
                    break
            if not begun:
                return
            for succession in begun:
                succession._evaluate_begun_part()

    def release(self) -> None:
        """Give every ``Succession`` back CE's own event handling."""
        for succession in self.successions:
            succession._event_plan = None
            succession._pending_time = None
            succession._next_index = None
        self.successions.clear()


def plan_events(
    animations: Sequence[Animation], start: Fraction, end: Fraction | None = None
) -> EventPlan:
    """Plan the events of a play that begins at ``start`` (and ends at
    ``end``): give each ``Succession`` in it the instants of its part ends
    (``None`` where the play does not reach one, or where no instant exists),
    and hand its events to the returned plan.
    """
    plan = EventPlan(now=start)
    for animation in animations:
        run_time = exact(animation._exact_run_time())
        affine = (
            None
            if run_time == 0
            else _Affine(1 / run_time, -start / run_time)  # (t - start) / run_time
        )
        float_run_time = animation.run_time

        def alpha(t: Fraction, run_time: float = float_run_time) -> float:
            # As the play evaluates it: CE's ``t / run_time`` on the float.
            return 1.0 if run_time == 0 else float(t - start) / run_time

        last = end if end is not None else start + run_time
        _plan(animation, _TimeMap(affine, alpha, True), plan, start, last)
    return plan


@dataclass(frozen=True)
class _TimeMap:
    """How a part's alpha follows the scene instant: exactly (``affine``),
    and as the play computes it in floats (``of``), which never decreases
    when ``monotone``.
    """

    affine: _Affine | None
    of: Callable[[Fraction], float]
    monotone: bool


def _plan(
    animation: Animation,
    alpha: _TimeMap,
    plan: EventPlan,
    start: Fraction,
    end: Fraction,
) -> None:
    from arachne.animation.composition import AnimationGroup, Succession

    if not isinstance(animation, AnimationGroup):
        return
    starts, ends = animation._exact_timings()
    max_end = max(ends, default=Fraction(0))
    rate = animation.rate_func
    float_max_end = animation.max_end_time
    float_starts = [float(x) for x in animation.anims_with_timings["start"]]
    float_ends = [float(x) for x in animation.anims_with_timings["end"]]
    group_affine = (
        alpha.affine.then(max_end, Fraction(0))
        if alpha.affine is not None and rate is rf.linear
        else None
    )
    of = alpha.of

    def group_time(t: Fraction) -> float:
        return float(rate(of(t)) * float_max_end)

    monotone = alpha.monotone and is_monotone(rate)
    if isinstance(animation, Succession):
        animation._exact_ends = ends
        instants: list[Fraction | None] = []
        for index, part_end in enumerate(ends):
            if group_affine is not None:
                instants.append(group_affine.solve(part_end))
            elif monotone:
                instants.append(
                    crossing_instant(group_time, float_ends[index], start, end)
                )
            else:
                instants.append(None)
                plan.determined = False
        animation._event_instants = instants
        animation._event_plan = plan
        plan.successions.append(animation)
    for index, (part, part_start, part_end) in enumerate(
        zip(animation.animations, starts, ends, strict=True)
    ):
        length = part_end - part_start
        part_affine = (
            group_affine.then(1 / length, -part_start / length)
            if group_affine is not None and length != 0
            else None
        )
        _plan(
            part,
            _TimeMap(
                part_affine,
                _part_alpha(
                    group_time,
                    float_starts[index],
                    float_ends[index],
                    part.run_time,
                    succession=isinstance(animation, Succession),
                ),
                monotone,
            ),
            plan,
            start,
            end,
        )


def _part_alpha(
    group_time: Callable[[Fraction], float],
    part_start: float,
    part_end: float,
    part_run_time: float,
    *,
    succession: bool,
) -> Callable[[Fraction], float]:
    """A part's alpha as its group computes it: a Succession's active part
    takes ``(time - start) / run_time``, an AnimationGroup's part the same
    over its window, clipped to ``[0, 1]``.
    """
    if succession:

        def of(t: Fraction) -> float:
            if part_run_time == 0:
                return 1.0
            return (group_time(t) - part_start) / part_run_time

        return of
    window = part_end - part_start

    def clipped(t: Fraction) -> float:
        if window == 0:
            return 1.0
        return float(np.clip((group_time(t) - part_start) / window, 0, 1))

    return clipped


def crossing_instant(
    time_of: Callable[[Fraction], float],
    target: float,
    low: Fraction,
    high: Fraction,
) -> Fraction | None:
    """The first instant in ``[low, high]`` at which a never-decreasing group
    time ``time_of`` reaches ``target``, found by bisection to
    :data:`BISECTION_TOLERANCE`; ``None`` when it does not reach it by
    ``high``.

    The answer has reached ``target`` and an instant within the tolerance
    before it has not. Of the instants in that last interval, the simplest
    fraction is taken when it has reached the target too, so a crossing that
    falls on a frame or tick (``k/fps``, as a symmetric rate function's
    midpoint does) is that frame itself; the midpoints are exact fractions, so
    the result is the same on every run.
    """
    if time_of(low) >= target:
        return low
    if time_of(high) < target:
        return None
    first = low
    while high - low > BISECTION_TOLERANCE:
        middle = (low + high) / 2
        if time_of(middle) >= target:
            high = middle
        else:
            low = middle
    simplest = _simplest_in(low, high) if low >= 0 else high
    if simplest > first and time_of(simplest) >= target:
        return simplest
    return high
