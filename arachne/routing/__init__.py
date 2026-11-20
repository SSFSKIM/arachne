"""Which renderer draws a scene, decided once from its complete record.

:data:`GPU_ONLY` is the one table of capabilities only the GPU build draws.
An object declares the rows it holds (:func:`declare`, :func:`withdraw`),
the scene's record keeps them with the object's state, and
:func:`capabilities` reads them back from the record: a scene holding one is
on the ``gpu`` route (:func:`route`), any other on the ``cpu`` route, where
nothing imports the GPU build. :func:`pin_for` gives a ``gpu``-route scene
the device it draws on, or refuses it with ``render.gpu_required`` on a host
with no GPU; ``render.gpu_live`` refuses one whose frames must be drawn as
it runs.

This package imports nothing of the GPU build when imported.
"""

from __future__ import annotations

__all__ = [
    "GPU_ONLY",
    "Capability",
    "GpuLive",
    "GpuRequired",
    "Where",
    "cache_token",
    "capabilities",
    "declare",
    "ever_declared",
    "gpu",
    "held",
    "pin_for",
    "route",
    "withdraw",
]

from .refusals import GpuLive, GpuRequired
from .route import Where, cache_token, capabilities, gpu, pin_for, route
from .table import GPU_ONLY, Capability, declare, ever_declared, held, withdraw
