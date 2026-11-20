"""The GPU device a process draws on.

:func:`pin` names it without opening it; :func:`compositor` opens it, at
the first GPU-drawn frame and never at import, one at a time per process (a
forked process opens its own). The GPU build is imported only when asked,
and only for a scene on the ``gpu`` route.

A test replaces the way a device is found (:data:`door`) with one onto a
software adapter, which only the GPU build's ``testing`` module opens; no
setting, variable or flag reaches it, and nothing under ``arachne/``
imports that module.
"""

from __future__ import annotations

__all__ = ["Door", "compositor", "door", "pin"]

import os
from typing import Any, Protocol

from arachne import routing


class Door(Protocol):
    """How a process finds its device: the pin of the device it would draw
    on (``None`` when it has none) and, given a frame size, a compositor
    drawing on that device (``None`` when there is none to open).
    """

    def pin(self) -> dict[str, str] | None: ...

    def open(self, width: int, height: int) -> Any: ...


class _Ordinary:
    """The GPU this process has, through the GPU build's ordinary
    constructor, which opens no software adapter.
    """

    def pin(self) -> dict[str, str] | None:
        return routing.gpu()

    def open(self, width: int, height: int) -> Any:
        try:
            import arachne_gpu
        except ModuleNotFoundError as error:
            if error.name != "arachne_gpu":
                raise
            return None
        try:
            return arachne_gpu.Compositor(width, height)
        except arachne_gpu.NoGpu:
            # The adapter a pin was read from can still fail to open.
            return None


#: The door every device is found through. Only a test replaces it.
door: Door = _Ordinary()


def pin() -> dict[str, str] | None:
    """The pin of the device this process would draw on, or ``None`` when
    it has none (no GPU build, or a build offering no GPU).
    """
    return door.pin()


#: The compositor this process draws with, once one is open, and the door
#: it came through.
_held: tuple[Door, Any] | None = None


def compositor(width: int, height: int) -> Any:
    """The compositor this process draws ``width`` x ``height`` frames
    with, opened now if need be (a frame of another size lets the one held
    go first), or ``None`` when this host has no GPU to open.
    """
    global _held
    if _held is not None:
        through, held = _held
        if through is door and (held.width, held.height) == (width, height):
            return held
    _held = None
    opened = door.open(width, height)
    if opened is not None:
        _held = (door, opened)
    return opened


def _forget() -> None:
    """A forked process opens a device of its own."""
    global _held
    _held = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget)
