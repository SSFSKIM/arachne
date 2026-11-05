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

__all__ = ["CorePainter"]

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from .cairo_painter import CairoDrawer
from .canonical import Canonical, canonical_order, frame_grid
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
        #: The deltas as the core received them, their geometry snapped
        #: (:mod:`.canonical`).
        self.canonical = Canonical()
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
        description, the overlay by Cairo. The frame's grid is decided once
        (:func:`~.canonical.frame_grid`): the camera's depth sort is taken on
        keys snapped by it (:func:`~.canonical.canonical_order`), and the
        geometry is snapped to it.
        """
        grid = frame_grid(camera)
        with canonical_order(camera, grid):
            delta = self.stream.frame(camera, groups, t=t, terms=terms)
        self.canonical.frame_grid = grid
        try:
            return self.draw(delta)
        finally:
            self.canonical.frame_grid = None

    def draw(self, delta: Delta) -> np.ndarray:
        """The frame ``delta`` brings the stream to, its geometry snapped
        (:mod:`.canonical`): the core is handed nothing else.
        """
        import arachne_core

        sent = self.canonical(delta)
        read = arachne_core.FrameDelta.from_bytes(sent.to_core().to_bytes())
        self.core.apply(read)
        self.state.apply(Delta.from_contents(read.contents()))
        self.last = sent
        self.frames += 1
        CorePainter.drawn_total += 1
        if self.keep:
            self.deltas.append(sent)
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
        return frame

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
