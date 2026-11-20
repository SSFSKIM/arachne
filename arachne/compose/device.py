"""The GPU device a process draws on.

:func:`pin` names it without opening it. The GPU build is imported only
when asked, and only for a scene on the ``gpu`` route.

A test replaces the way a device is found (:data:`door`) with one onto a
software adapter, which only the GPU build's ``testing`` module opens; no
setting, variable or flag reaches it, and nothing under ``arachne/``
imports that module.
"""

from __future__ import annotations

__all__ = ["Door", "door", "pin"]

from typing import Any, Protocol

from arachne import routing


class Door(Protocol):
    """How a process finds its device: the pin of the device it would draw
    on (``None`` when it has none) and, given a frame size, a compositor
    drawing on that device.
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
        import arachne_gpu

        return arachne_gpu.Compositor(width, height)


#: The door every device is found through. Only a test replaces it.
door: Door = _Ordinary()


def pin() -> dict[str, str] | None:
    """The pin of the device this process would draw on, or ``None`` when
    it has none (no GPU build, or a build offering no GPU).
    """
    return door.pin()
