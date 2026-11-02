"""DIAGNOSTIC BUILD, never merged: CE 0.20.1's clock put back.

This branch exists so the private gate can rerun a case with one deliberate
break undone and see whether that break alone explains the case's difference
from CE (C5.2, M4's break sort). It restores CE's way of sizing plays and of
counting time; nothing else changes.

- A play or wait lasts the frames CE gives it, ``len(arange(0, d, 1/fps))``,
  or for a frozen wait ``int(d * fps)``, at least one frame either way (CE's
  ``validate_run_time``); so every play starts and ends on the frame grid, and
  there is no end-window rule.
- ``renderer.time`` reads CE's float clock from the scene's first frame: one
  float add of ``1/fps`` per frame drawn, a frozen wait's frames added at once
  (CE's ``add_frame``), so a wait sized from the clock (``wait(t -
  renderer.time)``) comes out as in CE.
- A play with a stop condition checks it as CE does, after each frame it
  renders and only then, and ends at that frame's end: a stop seen on frame
  ``k`` ends the play ``k + 1`` frames after its start.
- Sizing a play asks the scene nothing: a frozen frame is known from the
  play's own held status, which the play reads once, as CE does.
- The times a play shows are CE's float progression, ``k * (1/fps)`` at its
  ``k``-th frame: the ``t`` and ``t - last_t`` its scene's frame hooks are
  shown, ``scene.last_t`` once it is over, and every animation's alpha
  (``t / run_time``). The exact instant still schedules the composition
  events.

The call sites are a line or two each (``Recorder.requested_end``,
``Recorder.play`` once the play is known held, ``Recorder._run_play``,
``TickedRoute.run``'s stop checks, ``CairoRenderer.time``) so the patch
rebases onto a moving head.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np

from .clock import exact, frames_in

__all__ = [
    "advance", "alpha", "float_time", "frame_end", "frame_steps", "frame_time", "last_t", "play_end",
]


def _fps() -> Fraction:
    from arachne._config import config

    return exact(config.frame_rate)


def play_end(start: Fraction, run_time: Any, held: bool) -> Fraction:
    """Where a play of ``run_time`` (its animations' longest, or what the
    scene's own ``get_run_time`` gave) beginning at ``start`` ends on CE's
    clock: its start plus the frames CE renders for it, at least one;
    ``int(d / dt)`` for a frozen frame (``held``), ``len(arange(0, d, dt))``
    otherwise. It asks the scene nothing.
    """
    fps = _fps()
    dt = 1 / float(fps)
    d = max(dt, float(run_time))
    n = int(d / dt) if held else len(np.arange(0, d, dt))
    return start + Fraction(n) / fps


def frame_end(instant: Fraction, end: Fraction) -> Fraction | None:
    """Where a play ending at ``end`` (its requested end) whose stop condition
    holds after the frame at ``instant`` ends, as CE ends it: that frame drawn,
    the next one's instant; ``None`` where CE draws no frame and so checks no
    stop condition: off the frame grid, or at or past the play's end, which
    CE's progression never reaches.
    """
    fps = _fps()
    if (instant * fps).denominator != 1 or instant >= end:
        return None
    return instant + 1 / fps


def frame_time(start: Fraction, instant: Fraction) -> float:
    """The play-local time CE's progression gives the frame at ``instant`` of a
    play that began at ``start``: ``np.arange(0, run_time, 1/fps)``'s value
    there, ``k * (1/fps)`` in float for the ``k``-th frame (NumPy fills an
    arange as ``start + i * step``); off the frame grid, the exact time.
    """
    fps = _fps()
    k = (instant - start) * fps
    if k.denominator != 1:
        return float(instant - start)
    return int(k) * (1 / float(fps))


def alpha(local: Fraction, run_time: float) -> float:
    """An animation's alpha as CE computes it, ``t / run_time`` with CE's
    float progression time ``t`` (:func:`frame_time`) for the play-local time
    ``local``; 1 when it has no length. The exact instant still schedules the
    composition events; only the interpolation reads this.
    """
    if run_time == 0:
        return 1.0
    return frame_time(Fraction(0), local) / run_time


def last_t(start: Fraction, end: Fraction, held: bool) -> float:
    """``scene.last_t`` once a play that began at ``start`` and ended at
    ``end`` is over, as CE leaves it: the last frame's progression time, its
    end never evaluated; 0 for a frozen frame, whose play evaluates none.
    """
    if held:
        return 0.0
    return frame_time(start, end - 1 / _fps())


def advance(renderer: Any, end: Fraction, held: bool) -> None:
    """Advance CE's float clock over a play that ended at ``end``: one add of
    ``1/fps`` per frame, or the frames' sum at once for a frozen wait.
    """
    fps = _fps()
    start, value = getattr(renderer, "_ce_float_clock", None) or (Fraction(0), 0.0)
    frames = int((end - start) * fps)
    dt = 1 / float(fps)
    if held:
        value += frames * dt
    else:
        for _ in range(frames):
            value += dt
    renderer._ce_float_clock = (end, value)


def float_time(renderer: Any) -> float:
    """``renderer.time`` as CE's float clock reads it at the exact clock's
    instant: the last play's end plus one add of ``1/fps`` per frame since;
    an instant off the frame grid reads the exact clock.
    """
    clock = renderer.clock
    # Before any play has ended, the scene's first frame is the basis: CE's
    # clock starts at 0.0 and every frame adds 1/fps.
    start, value = getattr(renderer, "_ce_float_clock", None) or (Fraction(0), 0.0)
    steps = (clock - start) * _fps()
    if steps.denominator != 1 or steps < 0:
        return float(clock)
    dt = 1 / float(_fps())
    for _ in range(int(steps)):
        value += dt
    return value


def frame_steps(start: Fraction, end: Fraction) -> tuple[list[Fraction], list[float]]:
    """The clock+dt build: the instants a ticked play steps at after its
    start, and the ``dt`` each passes, as CE's ``update_to_time`` steps a
    play: once per frame, after the play's first frame (its start, evaluated
    at ``dt = 0``), up to the last frame before its end, which is not
    stepped; each ``dt`` is ``t - last_t`` over CE's own float progression
    (:func:`frame_time`), not a constant ``1/fps``.
    """
    fps = _fps()
    ticks = [t for t in frames_in(start, end, fps) if start < t < end]
    dts = []
    last = 0.0
    for tick in ticks:
        t = frame_time(start, tick)
        dts.append(t - last)
        last = t
    return ticks, dts
