"""Frames drawn by the core, with Cairo's overlay over them.

On ``draw_path=core`` a frame goes through XF as on ``ir_cairo``: the delta's
bytes are parsed back by the core and applied to its ``Stream``. The core then
draws the frame without its overlay, and Cairo draws the overlay over the
core's pixels with CE's own camera code, in CE's order
(:meth:`~.cairo_painter.CairoDrawer.draw_over`). The overlay is what the core's
``Stream.overlay_entries`` names: mobjects fixed in frame, images (MultiCamera
displays included) and paths filled through a background image. Drawn onto
the core's frame as CE draws onto its own, the overlay rounds where CE's
rounds.
"""

from __future__ import annotations

__all__ = ["CorePainter", "frame_digest"]

import contextlib
import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from typing import Any, ClassVar

import numpy as np

from .cairo_painter import CairoDrawer
from .extract import FrameStream, Term, Unable
from .replay import Replays
from .xf import Delta, StreamState


class CorePainter:
    """Frames through XF, drawn by the core, the overlay by Cairo over them.

    The painter's interface is :class:`~.painter.IrCairoPainter`'s. ``keep``
    keeps every delta sent (``deltas``), for tests; by default the class's
    ``keep_all``. ``threads`` is the core's drawing threads per frame (the
    frame is the same for any count).
    """

    keep_all = False
    #: Frames drawn by the core in this process.
    drawn_total = 0
    #: The digests of the frames the core drew in this process, while
    #: :meth:`keeping_frames` keeps them.
    _kept: ClassVar[set[str] | None] = None
    #: Whether the overlay's runs the camera draws through its context are
    #: made again from the frame before's calls where they are the same
    #: (:mod:`.replay`).
    replay_overlay = True

    def __init__(
        self,
        *,
        keep: bool | None = None,
        stream: FrameStream | None = None,
        threads: int = 1,
    ) -> None:
        try:
            import arachne_core
        except ImportError as error:  # pragma: no cover - environment
            raise ImportError(
                "draw_path core needs arachne_core: install the 'core' extra"
            ) from error
        self.core: Any = arachne_core.Stream()
        self.stream = FrameStream() if stream is None else stream
        #: The stream as the core read it back, which the overlay is drawn from.
        self.state = StreamState()
        self.drawer = CairoDrawer()
        #: The overlay's kept runs, from frame to frame.
        self.replays = Replays() if self.replay_overlay else None
        self.threads = threads
        self.keep = self.keep_all if keep is None else keep
        self.deltas: list[Delta] = []
        self.last: Delta | None = None
        #: Frames this painter drew.
        self.frames = 0
        #: Frames whose overlay held at least one entry.
        self.overlaid = 0

    def paint(
        self,
        camera: Any,
        groups: Sequence[tuple[str, Sequence[Any]]],
        *,
        t: Fraction = Fraction(0),
        terms: Mapping[int, Term] | None = None,
    ) -> np.ndarray:
        """The frame the camera draws from ``groups`` (each a layer and the
        list one capture is handed), drawn by the core from its XF
        description, the overlay by Cairo.
        """
        return self.draw(self.stream.frame(camera, groups, t=t, terms=terms))

    def draw(self, delta: Delta) -> np.ndarray:
        """The frame ``delta`` brings the stream to."""
        import arachne_core

        read = arachne_core.FrameDelta.from_bytes(delta.to_core().to_bytes())
        self.core.apply(read)
        self.state.apply(Delta.from_contents(read.contents()))
        self.last = delta
        self.frames += 1
        CorePainter.drawn_total += 1
        if self.keep:
            self.deltas.append(delta)
        view = self.state.view
        assert view is not None
        try:
            pixels = self.core.draw(
                view["pixel_width"],
                view["pixel_height"],
                threads=self.threads,
                overlay=False,
            )
        except arachne_core.Unsupported as refused:
            # Every frame of a scene is drawn on one path: a frame the core
            # refuses is not drawn on another.
            raise Unable(
                f"render.unable: the core does not draw a {refused.kind} (slot"
                f" {refused.slot}): {refused.reason}"
            ) from refused
        frame = np.array(pixels)
        overlay = self.core.overlay_entries()
        if overlay:
            self.overlaid += 1
            self.drawer.draw_over(self.state, frame, overlay, self.replays)
        if CorePainter._kept is not None:
            CorePainter._kept.add(frame_digest(frame))
        return frame

    @classmethod
    @contextlib.contextmanager
    def keeping_frames(cls) -> Iterator[Callable[[np.ndarray], bool]]:
        """Within: every frame the core draws in this process is kept by its
        digest. Yields whether a frame's pixels are a frame the core drew: a
        delivered frame is the core's only so (the core may have drawn a
        description that a fallback then replaced with the camera's own draw).
        """
        held = cls._kept
        kept: set[str] = set()
        cls._kept = kept
        try:
            yield lambda frame: frame_digest(frame) in kept
        finally:
            cls._kept = held

    def settle(self) -> None:
        """Leave each display's camera of the last frame holding the picture
        CE's last capture leaves in it, as :meth:`IrCairoPainter.settle
        <.painter.IrCairoPainter.settle>` does: displays are Cairo's.
        """
        for display, (_, view, order) in self.stream.last_sub_frames:
            camera = display.camera
            camera.pixel_array = self.drawer.capture(
                self.state, view, order, camera.background
            )


def frame_digest(frame: Any) -> str:
    """The sha256 of a frame's pixels."""
    return hashlib.sha256(memoryview(np.ascontiguousarray(frame)).tobytes()).hexdigest()
