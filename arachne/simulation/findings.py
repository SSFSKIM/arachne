"""What a world's solver found in a stay, as the stay's occurrence holds it.

The solver records beside its rows (:meth:`arachne_sim.World.findings`) a
body whose state went non-finite, a body whose speed or angular speed rose
above the world's limit, and two bodies that passed through each other in a
step. A stay's occurrence holds those of its rows as
:class:`~arachne.timeline.bake.BakeFinding` (the lint reports them,
:mod:`arachne.lint.bakes`):

- ``sim.unstable`` about one body, its ``signal`` ``non_finite``, ``speed``
  or ``angular_speed``, with the ``value`` found and the world's ``limit``
  (none for a non-finite state); for a speed, whether the solver ``capped``
  it (scaled it down to its ``cap`` in the step), when the value is a lower
  bound, the larger of the cap and the row's speed;
- ``sim.tunnelled`` about two bodies, with the sweep's ``evidence``:
  ``overlap`` (the shapes were seen penetrating each other), ``near`` (a
  certified distance within the allowed linear error, which could not be
  told from a crossing) or ``indeterminate`` (a crossing could not be ruled
  out).

Each carries the scene's ``tick``. A stay that begins where an earlier one
left the world shares that row with it: the earlier stay holds what was
found there.
"""

from __future__ import annotations

__all__ = ["evidence", "occurrence_findings"]

import math
from collections.abc import Sequence
from typing import Any, Literal

from arachne.timeline.bake import BakeFinding

#: What each kind the solver finds is, as a code and a signal.
_KINDS = {
    1: ("sim.unstable", "non_finite"),
    2: ("sim.unstable", "speed"),
    3: ("sim.unstable", "angular_speed"),
    4: ("sim.tunnelled", None),
    5: ("sim.unstable", "speed"),
    6: ("sim.unstable", "angular_speed"),
}
#: The kinds of a speed the solver scaled down to its cap.
_CAPPED = frozenset({5, 6})


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
            if limit is not None:
                detail["capped"] = kind in _CAPPED
                if kind in _CAPPED:
                    detail["cap"] = _cap(signal)
            bodies: tuple[str, ...] = (ids[a],)
        else:
            detail["evidence"] = evidence(value)
            bodies = (ids[a], ids[b])
        out.append(BakeFinding(code, row, bodies, detail))
    return tuple(out)


def evidence(closest: float) -> Literal["overlap", "near", "indeterminate"]:
    """What the solver's sweep saw of two bodies, from its value
    (:meth:`arachne_sim.World.finding_values`): a negative one is the depth
    they were seen penetrating each other, ``overlap``; one at or above zero
    a certified distance within the allowed linear error, ``near``; NaN, a
    crossing that could not be ruled out, ``indeterminate``.
    """
    if math.isnan(closest):
        return "indeterminate"
    return "overlap" if closest < 0 else "near"


def _cap(signal: str | None) -> float:
    """The solver's cap on a body's speed (u/s) or angular speed (rad/s)."""
    import arachne_sim

    return float(
        arachne_sim.SPEED_CAP if signal == "speed" else arachne_sim.ANGULAR_SPEED_CAP
    )
