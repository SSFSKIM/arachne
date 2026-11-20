"""Arachne's simulation crate: rigid-body worlds in 2D and 3D, stepped on a fixed clock of 60
ticks per second, whose results are the same bytes on every machine of one architecture.

A :class:`World` is given bodies, shapes, joints and force laws, every number already a float32
(``numpy.float32`` or a float32 array: the stored type, converted once by the caller), and
computes its rows in order: row ``i`` is its state ``i`` steps of exactly 1/60 s after its first.
:meth:`World.advance_to` computes the rows up to one, the first time it is asked, and is a lookup
afterwards; an open world advances one row at a time with :meth:`World.step`, given the forces of
its callbacks. Rows are handed out as read-only arrays: :meth:`World.pose` (``x, y, angle`` in 2D;
``x, y, z`` and a unit quaternion ``qx, qy, qz, qw`` in 3D), :meth:`World.velocity` (``vx, vy, w``;
``vx, vy, vz, wx, wy, wz``) and :meth:`World.contacts` (``row, a, b, kind``, kind 1 when bodies
``a < b`` began to touch and 0 when they parted).

:meth:`World.findings` is what a world found as it computed its rows (``row, kind, a, b``):
a body whose state went non-finite (:data:`NON_FINITE`), whose speed or angular speed rose above
the world's limit (:data:`SPEED`, :data:`ANGULAR_SPEED`; ``speed_limit`` and
``angular_speed_limit``, below the solver's caps :data:`SPEED_CAP` and
:data:`ANGULAR_SPEED_CAP`), and two bodies that passed through each other (:data:`TUNNELLED`).

:meth:`World.declaration_bytes` is everything the world was given, in order;
:meth:`World.trace_bytes` is the canonical bytes of the bake content its rows make (channels
``pose`` and ``velocity``, event table ``contact``, and :meth:`World.identity`).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from arachne_sim._sim import (
    ANGULAR_SPEED,
    ANGULAR_SPEED_CAP,
    ANGULAR_SPEED_LIMIT,
    CONTENT_LAYOUT,
    DECLARATION_LAYOUT,
    NON_FINITE,
    SPEED,
    SPEED_CAP,
    SPEED_LIMIT,
    TUNNELLED,
)
from arachne_sim._sim import World as _World

__all__ = [
    "ANGULAR_SPEED",
    "ANGULAR_SPEED_CAP",
    "ANGULAR_SPEED_LIMIT",
    "CONTENT_LAYOUT",
    "DECLARATION_LAYOUT",
    "NON_FINITE",
    "SPEED",
    "SPEED_CAP",
    "SPEED_LIMIT",
    "TUNNELLED",
    "World",
]


class World(_World):
    def pose(self, start: int = 0, stop: int | None = None) -> npt.NDArray[np.float32]:
        """Rows ``start`` to ``stop`` (by default every computed row) of every body's pose,
        ``[rows, bodies, 3]`` in 2D and ``[rows, bodies, 7]`` in 3D, read-only.
        """
        data, shape = self._pose(start, stop)
        return np.frombuffer(data, dtype="<f4").reshape(shape)

    def velocity(
        self, start: int = 0, stop: int | None = None
    ) -> npt.NDArray[np.float32]:
        """Rows ``start`` to ``stop`` of every body's velocity, ``[rows, bodies, 3]`` in 2D and
        ``[rows, bodies, 6]`` in 3D, read-only.
        """
        data, shape = self._velocity(start, stop)
        return np.frombuffer(data, dtype="<f4").reshape(shape)

    def contacts(self) -> npt.NDArray[np.uint32]:
        """Every contact event so far, ``[events, 4]`` (row, a, b, kind), sorted, read-only."""
        data, count = self._contacts()
        return np.frombuffer(data, dtype="<u4").reshape((count, 4))

    def findings(self) -> npt.NDArray[np.uint32]:
        """What was found so far, ``[findings, 4]`` (row, kind, a, b), sorted, read-only: a
        body's finding names it as ``a`` and ``b``, a pair's its two bodies, ``a < b``.
        """
        data, count = self._findings()
        return np.frombuffer(data, dtype="<u4").reshape((count, 4))
