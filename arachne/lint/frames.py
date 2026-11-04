"""Reading a session's frames as its render shows them, once per lint.

A :class:`Frames` is one lint's source of frames. Each reader asks it, before
anything is read, for the frames of the render's grid (or the scene's end)
it needs and how to read one (:meth:`Frames.request`); then :meth:`Frames.read`
answers every request together: from the record where it kept a frame's
capture as the render evaluated it, and every other frame of every request
from **one** forward run of the scene on the render's grid, whose live scene
each reader asking for that frame is shown as the run reaches it. A record
kept on another grid than the render's is read wholly from the forward run:
its frame hooks and dt updaters ran at instants no frame of the render shows.

A reader that needs of the execution what the session's record lacks asks
for every frame from the forward run (:meth:`Frames.require_forward`), and
every reader is then served from it. A reader whose frames matter only if
the forward run is the execution read asks for them from that run alone
(``forward_only``): they are read when the other requests need one, and
not otherwise (:meth:`Frames.answered`). One that reads the states a
record keeps between frames (``Record.states``: the scene as the script
called each play, as each ended and as the script removed from it) asks
that the execution keep them from a time on (:meth:`Frames.require_states`):
the forward run then keeps them, and runs when the session's record does
not keep every one of them from then on.

One run matters beyond its cost: two executions of a scene need not agree (a
helper module's state is not restored between them), so what two readers of
one lint see must come from the same one. A reader that needs more of that
execution than frames (what it recorded per play) reads :attr:`Frames.execution`
after :meth:`Frames.read`: the forward run's record when one ran, else the
session's.

:func:`read_frames` is one request read at once.
"""

from __future__ import annotations

__all__ = ["FrameReader", "Frames", "read_frames"]

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
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


@dataclass(eq=False)
class _Request:
    times: tuple[Fraction, ...]
    read: FrameReader[Any]
    found: dict[Fraction, Any]
    #: Read only from a forward run the other requests need.
    forward_only: bool = False
    answered: bool = False


class Frames:
    """One lint's frames of ``session``, every request read together."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._requests: dict[Hashable, _Request] = {}
        #: The owners that need a forward run, when one runs, to the end.
        self._complete: set[Hashable] = set()
        self._read = False
        self._forward_only = False
        self._states = False
        #: The forward run's record, once one has given frames.
        self.forward: Record | None = None

    def request(
        self,
        owner: Hashable,
        times: Sequence[Fraction],
        read: FrameReader[Any],
        *,
        forward_only: bool = False,
    ) -> None:
        """Ask for ``read``'s reading of the frame at each of ``times``, for
        ``owner``; with ``forward_only``, only from a forward run the other
        requests need, and not at all when they need none
        (:meth:`answered`). Raises :class:`RuntimeError` once the frames are
        read, and :class:`ValueError` for an owner that asked already.
        """
        if self._read:
            raise RuntimeError("The lint's frames are read: ask before they are.")
        if owner in self._requests:
            raise ValueError(f"{owner!r} asked for frames already.")
        self._requests[owner] = _Request(tuple(times), read, {}, forward_only)

    def require_forward(self) -> None:
        """Read every frame of every request from the forward run: for a
        reader that needs of the execution what the session's record lacks.
        Raises :class:`RuntimeError` once the frames are read.
        """
        if self._read:
            raise RuntimeError("The lint's frames are read: ask before they are.")
        self._forward_only = True

    def require_states(self) -> None:
        """Have the execution the frames come from (:attr:`execution`) keep
        the scene as each play ended and as the script removed from it
        (``Record.states``): a forward run, when one runs, keeps them, and
        one runs, every frame read from it, when the session's record keeps
        none. Raises :class:`RuntimeError` once the frames are read.
        """
        if self._read:
            raise RuntimeError("The lint's frames are read: ask before they are.")
        self._states = True

    def complete(self, owner: Hashable) -> None:
        """Ask, for ``owner``, that a forward run, when one runs, run to the
        scene's end, so that :attr:`execution` holds the whole of it: no
        frame more is read for it. Raises :class:`RuntimeError` once the
        frames are read.
        """
        if self._read:
            raise RuntimeError("The lint's frames are read: ask before they are.")
        self._complete.add(owner)

    def read(self) -> None:
        """Read every request: from the record what it kept, the rest from
        one forward run; raises :class:`RuntimeError` when that run showed no
        scene at a frame asked for.
        """
        if self._read:
            raise RuntimeError("The lint's frames are read already.")
        self._read = True
        session = self.session
        record = session.record
        asking: dict[Fraction, list[_Request]] = {}
        # What only a forward run answers, if the others need one.
        later: dict[Fraction, list[_Request]] = {}
        for request in self._requests.values():
            for t in dict.fromkeys(request.times):
                (later if request.forward_only else asking).setdefault(t, []).append(
                    request
                )
        settings: ManimConfig | dict[str, Any] = (
            session._source.settings if session._source is not None else {}
        )
        # A record on the render's grid evaluated each frame of it as the
        # render does; one on another grid holds states no frame shows.
        from_record = (
            not self._forward_only
            and not (self._states and record.states is None)
            and record.grid == render_grid(record)
        )
        missing: list[Fraction] = []
        with tempconfig(settings):
            for t in sorted(asking):
                if not from_record:
                    missing.append(t)
                    continue
                try:
                    resolved = _resolve(record, t)
                except (RecordRefused, Unrestorable):
                    missing.append(t)
                    continue
                if resolved.kind != "recorded":
                    # The render evaluates a frame the record did not keep
                    # through the scene's own code, which no evaluation here
                    # would run.
                    missing.append(t)
                    continue
                for request in asking[t]:
                    request.found[t] = request.read(
                        record, t, resolved.play, resolved.capture, False
                    )
            if missing:
                from arachne.timeline.fallback import visit_frames

                needed = set(missing)
                for requests in later.values():
                    for request in requests:
                        request.answered = True

                def visit(
                    t: Fraction,
                    forward: Record,
                    play: PlayRecord | None,
                    capture: Capture,
                ) -> None:
                    self.forward = forward
                    shown = [
                        *(asking.get(t, ()) if t in needed else ()),
                        *later.get(t, ()),
                    ]
                    for request in shown:
                        request.found[t] = request.read(forward, t, play, capture, True)

                visits = sorted(needed | set(later))
                visit_frames(
                    session,
                    visits,
                    visit,
                    to_end=bool(self._complete),
                    states=self._states,
                )
                missing = visits
        unread = [
            t
            for t in missing
            if any(
                t not in r.found
                for r in [*asking.get(t, ()), *later.get(t, ())]
                if not r.forward_only or r.answered
            )
        ]
        if unread:
            raise RuntimeError(
                f"The forward run showed no scene at {', '.join(map(str, unread))}."
            )

    @property
    def execution(self) -> Record:
        """The execution the frames come from: the forward run's record when
        one ran, else the session's.
        """
        return self.forward if self.forward is not None else self.session.record

    def answered(self, owner: Hashable) -> bool:
        """Whether ``owner``'s frames were read: always, once :meth:`read`
        has run, but for a forward-only request when no forward run was
        needed.
        """
        request = self._requests.get(owner)
        return (
            self._read
            and request is not None
            and (not request.forward_only or request.answered)
        )

    def result(self, owner: Hashable) -> list[Any]:
        """What ``owner``'s reader made of each frame it asked for, in the
        order it asked; raises :class:`LookupError` for an owner that never
        asked, or before :meth:`read`.
        """
        request = self._requests.get(owner)
        if request is None or not self._read:
            raise LookupError(
                f"{owner!r} has no frames: ask for them (request) before the"
                " lint reads its frames."
            )
        return [request.found[t] for t in request.times]


def read_frames(
    session: Session, times: Sequence[Fraction], read: FrameReader[T]
) -> list[T]:
    """What ``read`` makes of the frame at each of ``times``, in their order,
    from frames read for this one request.
    """
    frames = Frames(session)
    frames.request(read_frames, times, read)
    frames.read()
    return frames.result(read_frames)
