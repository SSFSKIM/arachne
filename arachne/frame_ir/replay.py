"""The overlay's Cairo calls, kept from one frame for the next.

Over the core's frame, Cairo draws the overlay with CE's own camera code
(:meth:`~.cairo_painter.CairoDrawer.draw_over`). Most of that time is Python:
proxies, the camera's point transforms, its subpath splitting and colours.
What reaches the frame is only the calls the camera made on its Cairo
context. Where a member of the overlay is the same as in the frame before
(the same record, the same view it reads, and, where it takes the context's
current path, after the same member), the calls it made then are made again
on the new frame's context, in order. The bytes are then the camera's draw
by construction: the same calls on the same kind of surface, over the same
pixels beneath.

Only members the camera draws through its context are kept. Images,
displays and background-image fills write into the array by other means and
are drawn again on every frame.
"""

from __future__ import annotations

__all__ = ["Recording", "Replays", "replay"]

from typing import Any

#: A Cairo call: the context method's name, its arguments and keywords.
Call = tuple[str, tuple[Any, ...], dict[str, Any]]


class Recording:
    """A Cairo context that keeps every call made on it, in order, and makes
    it on the context it stands for.
    """

    __slots__ = ("_context", "_methods", "calls")

    def __init__(self, context: Any) -> None:
        self._context = context
        self._methods: dict[str, Any] = {}
        self.calls: list[Call] = []

    def __getattr__(self, name: str) -> Any:
        methods = self._methods
        found = methods.get(name)
        if found is not None:
            return found
        target = getattr(self._context, name)
        if not callable(target):
            return target
        calls = self.calls

        def call(*args: Any, **kwargs: Any) -> Any:
            calls.append((name, args, kwargs))
            return target(*args, **kwargs)

        methods[name] = call
        return call

    def begins_path(self) -> bool:
        """Whether the calls begin a path of their own, so that nothing of
        the context's current path before them reaches what they draw.
        """
        return bool(self.calls) and self.calls[0][0] == "new_path"


def replay(context: Any, calls: list[Call]) -> None:
    """Make ``calls`` on ``context``, in order."""
    methods: dict[str, Any] = {}
    for name, args, kwargs in calls:
        method = methods.get(name)
        if method is None:
            method = methods[name] = getattr(context, name)
        method(*args, **kwargs)


class Replays:
    """The calls of each kept member of the frame before, by the member's
    key, and those of the frame being drawn
    (:meth:`~.cairo_painter.CairoDrawer.draw_over` makes the keys).
    """

    def __init__(self) -> None:
        self._before: dict[Any, list[Call]] = {}
        self._now: dict[Any, list[Call]] = {}
        #: What the keys of both frames name by identity, kept alive with them.
        self._held: list[Any] = []
        self._holding: list[Any] = []
        #: Members made again from the frame before, and members drawn and kept.
        self.replayed = 0
        self.recorded = 0

    def found(self, key: Any) -> list[Call] | None:
        """The calls the frame before made for the member ``key``, now kept
        for the next frame too, if any.
        """
        calls = self._before.get(key)
        if calls is not None:
            self._now[key] = calls
            self.replayed += 1
        return calls

    def keep(self, key: Any, calls: list[Call]) -> None:
        """The member ``key`` of this frame made ``calls``."""
        self._now[key] = calls
        self.recorded += 1

    def hold(self, objects: Any) -> None:
        """Keep ``objects`` alive while this frame's keys may name them."""
        self._holding.append(objects)

    def next_frame(self) -> None:
        """This frame is drawn: its members are the ones the next frame finds."""
        self._before, self._now = self._now, {}
        self._held, self._holding = self._holding, []
