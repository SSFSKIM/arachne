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
(:func:`plan_events` solves it); for CE's monotone nonlinear rate functions
:func:`crossing_instant` finds it by bisection. A ticked play takes every
event at the first tick at or after its instant, which for a timing without
an exact instant is the first tick at which the group time has crossed.
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

from arachne.utils import rate_functions as rf

from .clock import exact

if TYPE_CHECKING:
    from arachne.animation.animation import Animation
    from arachne.animation.composition import Succession

#: CE's rate functions that never decrease, by identity. A group whose time
#: runs through one of these crosses each part boundary once, at an instant
#: bisection can find.
MONOTONE_RATE_FUNCTIONS: frozenset[Callable[[float], float]] = frozenset(
    {
        rf.linear,
        rf.smooth,
        rf.smoothstep,
        rf.smootherstep,
        rf.smoothererstep,
        rf.rush_into,
        rf.rush_from,
        rf.slow_into,
        rf.double_smooth,
        rf.lingering,
        rf.exponential_decay,
        rf.ease_in_sine,
        rf.ease_out_sine,
        rf.ease_in_out_sine,
        rf.ease_in_quad,
        rf.ease_out_quad,
        rf.ease_in_out_quad,
        rf.ease_in_cubic,
        rf.ease_out_cubic,
        rf.ease_in_out_cubic,
        rf.ease_in_quart,
        rf.ease_out_quart,
        rf.ease_in_out_quart,
        rf.ease_in_quint,
        rf.ease_out_quint,
        rf.ease_in_out_quint,
        rf.ease_in_expo,
        rf.ease_out_expo,
        rf.ease_in_out_expo,
        rf.ease_in_circ,
        rf.ease_out_circ,
        rf.ease_in_out_circ,
    }
)

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


def plan_events(animations: Sequence[Animation], start: Fraction) -> EventPlan:
    """Plan the events of a play that begins at ``start``: give each
    ``Succession`` in it the instants of its part ends (``None`` where no
    exact one exists), and hand its events to the returned plan.
    """
    plan = EventPlan(now=start)
    for animation in animations:
        run_time = exact(animation.run_time)
        alpha = (
            None
            if run_time == 0
            else _Affine(1 / run_time, -start / run_time)  # (t - start) / run_time
        )
        _plan(animation, alpha, plan)
    return plan


def _plan(animation: Animation, alpha: _Affine | None, plan: EventPlan) -> None:
    from arachne.animation.composition import AnimationGroup, Succession

    if not isinstance(animation, AnimationGroup):
        return
    starts, ends = animation._exact_timings()
    max_end = max(ends, default=Fraction(0))
    group_time = (
        alpha.then(max_end, Fraction(0))
        if alpha is not None and animation.rate_func is rf.linear
        else None
    )
    if isinstance(animation, Succession):
        animation._exact_ends = ends
        animation._event_instants = [
            group_time.solve(end) if group_time is not None else None for end in ends
        ]
        animation._event_plan = plan
        plan.successions.append(animation)
    for part, part_start, part_end in zip(
        animation.animations, starts, ends, strict=True
    ):
        length = part_end - part_start
        part_alpha = (
            group_time.then(1 / length, -part_start / length)
            if group_time is not None and length != 0
            else None
        )
        _plan(part, part_alpha, plan)


def crossing_instant(
    time_of: Callable[[float], float],
    target: Fraction,
    low: Fraction,
    high: Fraction,
) -> Fraction:
    """The instant in ``(low, high]`` at which a never-decreasing group time
    ``time_of(t)`` reaches ``target``, by bisection to
    :data:`BISECTION_TOLERANCE`: the returned instant has reached it, and an
    instant within the tolerance before it has not.

    ``time_of(low)`` must be below ``target`` and ``time_of(high)`` at or
    above it. The midpoints are exact fractions, so the result is the same on
    every run.
    """
    while high - low > BISECTION_TOLERANCE:
        middle = (low + high) / 2
        if Fraction(time_of(float(middle))) >= target:
            high = middle
        else:
            low = middle
    return high
