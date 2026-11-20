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

Each carries the scene's ``tick``. A speed the solver scaled down to its cap
during a step is found though the step's last velocity is under the limit,
its value then at least the cap (:meth:`arachne_sim.World.finding_values`).
A stay that begins where an earlier one left the world shares that row with
it: the earlier stay holds what was found there.
"""

from __future__ import annotations

__all__ = ["occurrence_findings"]

import math
from collections.abc import Sequence
from typing import Any

from arachne.timeline.bake import BakeFinding

#: What each kind the solver finds is, as a code and a signal.
_KINDS = {
    1: ("sim.unstable", "non_finite"),
    2: ("sim.unstable", "speed"),
    3: ("sim.unstable", "angular_speed"),
    4: ("sim.tunnelled", None),
}


def occurrence_findings(
    crate: Any,
    start: int,
    stop: int,
    first_tick: int,
    ids: Sequence[str],
    *,
    owns_first: bool,
) -> tuple[BakeFinding, ...]:
    """What ``crate`` found at its rows ``start`` to ``stop``, by the stay's
    rows, which begin at the scene's ``first_tick``; ``ids`` are the bodies'
    mobjects' stable ids. Row ``start`` is the stay's only when it
    ``owns_first``: when no earlier stay of the world holds it.
    """
    found = crate.findings()
    values = crate.finding_values()
    lower = start if owns_first else start + 1
    within = (found[:, 0] >= lower) & (found[:, 0] < stop)
    out = []
    for (solver_row, kind, a, b), value in zip(
        found[within].tolist(), values[within].tolist(), strict=True
    ):
        code, signal = _KINDS[kind]
        row = solver_row - start
        detail: dict[str, Any] = {"tick": first_tick + row}
        if code == "sim.unstable":
            limit = None
            if signal == "speed":
                limit = float(crate.speed_limit)
            elif signal == "angular_speed":
                limit = float(crate.angular_speed_limit)
            measured = None if limit is None or not math.isfinite(value) else value
            detail.update(signal=signal, value=measured, limit=limit)
            bodies: tuple[str, ...] = (ids[a],)
        else:
            bodies = (ids[a], ids[b])
        out.append(BakeFinding(code, row, bodies, detail))
    return tuple(out)
