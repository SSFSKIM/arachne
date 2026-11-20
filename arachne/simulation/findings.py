"""What a world's solver found in a stay, as the stay's occurrence holds it.

The solver records beside its rows (:meth:`arachne_sim.World.findings`) a
body whose state went non-finite, a body whose speed or angular speed rose
above the world's limit, and two bodies that passed through each other in a
step. A stay's occurrence holds those of its rows as
:class:`~arachne.timeline.bake.BakeFinding` (the lint reports them,
:mod:`arachne.lint.bakes`):

- ``sim.unstable`` about one body, its ``signal`` ``non_finite``, ``speed``
  or ``angular_speed``, with the ``value`` found and the world's ``limit``
  (none for a non-finite state);
- ``sim.tunnelled`` about two bodies.

Each carries the scene's ``tick``. A stay that begins where an earlier one
left the world shares that row with it: the earlier stay holds what was
found there.
"""

from __future__ import annotations

__all__ = ["occurrence_findings"]

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from arachne.timeline.bake import BakeFinding

#: What each kind the solver finds is, as a code and a signal.
_KINDS = {
    1: ("sim.unstable", "non_finite"),
    2: ("sim.unstable", "speed"),
    3: ("sim.unstable", "angular_speed"),
    4: ("sim.tunnelled", None),
}


def occurrence_findings(
    crate: Any, start: int, stop: int, first_tick: int, ids: Sequence[str]
) -> tuple[BakeFinding, ...]:
    """What ``crate`` found at its rows ``start`` to ``stop`` (row ``start``
    only when it is the world's first), by the stay's rows, which begin at
    the scene's ``first_tick``; ``ids`` are the bodies' mobjects' stable ids.
    """
    found = crate.findings()
    lower = start + 1 if start > 0 else 0
    window = found[(found[:, 0] >= lower) & (found[:, 0] < stop)]
    if not len(window):
        return ()
    velocity = crate.velocity(start, stop)
    planar = crate.dimension == 2
    out = []
    for solver_row, kind, a, b in window.tolist():
        code, signal = _KINDS[kind]
        row = solver_row - start
        detail: dict[str, Any] = {"tick": first_tick + row}
        if code == "sim.unstable":
            value, limit = _measured(signal, velocity[row, a], planar, crate)
            detail.update(signal=signal, value=value, limit=limit)
            bodies: tuple[str, ...] = (ids[a],)
        else:
            bodies = (ids[a], ids[b])
        out.append(BakeFinding(code, row, bodies, detail))
    return tuple(out)


def _measured(
    signal: str | None, velocity: np.ndarray, planar: bool, crate: Any
) -> tuple[float | None, float | None]:
    """The speed found and the world's limit for ``signal``; none for a
    non-finite state.
    """
    if signal == "speed":
        linear = velocity[:2] if planar else velocity[:3]
        return _norm(linear), float(crate.speed_limit)
    if signal == "angular_speed":
        angular = velocity[2:3] if planar else velocity[3:6]
        return _norm(angular), float(crate.angular_speed_limit)
    return None, None


def _norm(values: np.ndarray) -> float:
    return math.sqrt(math.fsum(float(v) * float(v) for v in values))
