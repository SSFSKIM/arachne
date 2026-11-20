"""The errors the GPU build raises."""

from __future__ import annotations

from typing import Any


class NoGpu(Exception):
    """No adapter this build may draw on: the ordinary constructor found no GPU, or the testing
    door no software adapter. ``seen`` lists every adapter found, as :func:`adapters` does.
    """

    def __init__(self, message: str, seen: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.seen = seen


class ForkedProcessError(RuntimeError):
    """This process was forked from one that had already used the GPU build: the Vulkan state it
    inherited is its parent's and is neither used nor released here. Start GPU work in a process
    made by spawn or exec instead. Nothing was opened.
    """


class Unsupported(Exception):
    """A frame refused whole: it asks for something this build does not draw
    (``render.unsupported``). Nothing is drawn.
    """

    code = "render.unsupported"

    def __init__(self, reason: str) -> None:
        super().__init__(f"{self.code}: {reason}")
        self.reason = reason


class SizeMismatch(ValueError):
    """A frame refused whole: one of its pictures or masks is not of the size of the frame it is
    in, the outermost frame's or a window's own. Nothing is drawn.
    """
