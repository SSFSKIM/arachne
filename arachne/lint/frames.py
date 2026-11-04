"""Reading a session's frames as its render shows them.

:func:`read_frames` hands a reader the scene at each of a list of frames of
the render's grid (or at the scene's end): from the record where it kept
that frame's capture as the render evaluated it, and every other frame from
one forward run of the scene on the render's grid, whose live scene the
reader is shown as it runs. A record kept on another grid than the render's
is read wholly from the forward run: its frame hooks and dt updaters ran at
instants no frame of the render shows.
"""

from __future__ import annotations

__all__ = ["FrameReader", "read_frames"]

from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any, TypeVar

from arachne._config import tempconfig
from arachne.timeline.restore import Unrestorable
from arachne.timeline.routes import RecordRefused, _resolve

from .instants import render_grid

if TYPE_CHECKING:
    from arachne._config.utils import ManimConfig
    from arachne.session import Session
    from arachne.timeline.record import Capture, PlayRecord, Record

T = TypeVar("T")

#: A reader of one frame: ``read(record, t, play, capture, live)``. With
#: ``live`` false, ``capture`` is the session's record's and the reader shows
#: it itself; with ``live`` true, the record is a forward run's, its scene
#: holds that capture now, and its camera the map the frame is drawn with.
FrameReader = Callable[["Record", Fraction, "PlayRecord | None", "Capture", bool], T]


def read_frames(
    session: Session, times: Sequence[Fraction], read: FrameReader[T]
) -> list[T]:
    """What ``read`` makes of the frame at each of ``times``, in their order;
    raises :class:`RuntimeError` when a forward run showed none at one.
    """
    record = session.record
    found: dict[Fraction, T] = {}
    missing: list[Fraction] = []
    settings: ManimConfig | dict[str, Any] = (
        session._source.settings if session._source is not None else {}
    )
    # A record on the render's grid evaluated each frame of it as the render
    # does; one on another grid holds states no frame of the render shows.
    on_render_grid = record.grid == render_grid(record)
    with tempconfig(settings):
        for t in dict.fromkeys(times):
            if not on_render_grid:
                missing.append(t)
                continue
            try:
                resolved = _resolve(record, t)
            except (RecordRefused, Unrestorable):
                missing.append(t)
                continue
            if resolved.kind != "recorded":
                # The render evaluates a frame the record did not keep through
                # the scene's own code, which an evaluation here would not run.
                missing.append(t)
                continue
            found[t] = read(record, t, resolved.play, resolved.capture, False)
        if missing:
            from arachne.timeline.fallback import visit_frames

            def visit(
                t: Fraction,
                forward: Record,
                play: PlayRecord | None,
                capture: Capture,
            ) -> None:
                found[t] = read(forward, t, play, capture, True)

            visit_frames(session, missing, visit)
    unread = [t for t in missing if t not in found]
    if unread:
        raise RuntimeError(
            f"The forward run showed no scene at {', '.join(map(str, unread))}."
        )
    return [found[t] for t in times]
