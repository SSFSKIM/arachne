"""The errors the core raises."""

from __future__ import annotations


class XfError(ValueError):
    """Bytes or a delta that are not valid XF for this build; a refused delta changes nothing."""


class Unsupported(Exception):
    """A frame refused whole: a drawn record (or the view) carries a kind or feature bit this
    build does not draw (``render.unsupported``). Nothing is drawn.
    """

    code = "render.unsupported"

    def __init__(
        self, kind: str, slot: int | None, obj: int | None, reason: str
    ) -> None:
        super().__init__(f"{self.code}: {kind} (slot {slot}, object {obj}): {reason}")
        self.kind = kind
        self.slot = slot
        self.object = obj
        self.reason = reason
