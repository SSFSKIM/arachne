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

from . import contributors
from .cairo_painter import CairoDrawer
from .canonical import _PATH_KINDS, Canonical, canonical_order, frame_grid, shades
from .contributors import Box, LiveFrame, Outline, PixelContributors
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
    #: Whether it draws captures as the scene runs (:mod:`.live`), each
    #: over the picture its camera's array holds: it then reports the
    #: contributors of a capture only where that capture is :attr:`whole`.
    live = False
    #: Whether the last frame drawn is whole: drawn over its camera's own
    #: background with nothing of an earlier picture composited into it.
    #: A frame drawn from the record is. A live painter's frame is not,
    #: unless its live painter records that the capture began on the
    #: camera's background as its reset left it (:mod:`.live`); never
    #: decided from the pixels.
    whole = True
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

    def draw(self, delta: Delta, *, _unguarded: bool = False) -> np.ndarray:
        """The frame ``delta`` brings the stream to, its geometry snapped
        (:mod:`.canonical`): the core is handed nothing else. ``_unguarded``
        draws past :meth:`_refuse_undecided`, for a test's reference drawn
        from the values as made; nothing else passes it.
        """
        import arachne_core

        self.whole = not self.live
        sent = self.canonical(delta)
        read = arachne_core.FrameDelta.from_bytes(sent.to_core().to_bytes())
        self.core.apply(read)
        self.state.apply(Delta.from_contents(read.contents()))
        overlay = self.core.overlay_entries()
        if not _unguarded:
            self._refuse_undecided(overlay)
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
        if overlay:
            self.overlaid += 1
            self.drawer.draw_over(self.state, frame, overlay, self.replays)
        return frame

    def outlines(self) -> dict[tuple[int, int], tuple[Outline, ...]]:
        """What Cairo is handed for each leaf of the last frame drawn through
        the camera's context, by ``(slot, part)`` (:meth:`.cairo_painter.
        CairoDrawer.outlines`): the geometry its edges are measured against.
        Refused (``LiveFrame``) unless the last frame is :attr:`whole`.
        """
        if not self.whole:
            raise LiveFrame(
                "outlines: the last capture was drawn over an earlier picture;"
                " take them from the frame's state extracted whole"
            )
        return self.drawer.outlines(self.state, self.state.order)

    def contributors(self, box: Box) -> dict[tuple[int, int], PixelContributors]:
        """The leaves the last frame composites at each pixel of ``box``,
        bottom first (:mod:`.contributors`): the core's, in depth order per
        sub-sample, under the overlay's, which Cairo paints over them in CE's
        order (each overlay leaf drawn again alone). Nothing the painter
        draws next changes. Refused (``LiveFrame``) unless the last frame is
        :attr:`whole`.
        """
        if not self.whole:
            raise LiveFrame(
                "contributors: the last capture was drawn over an earlier"
                " picture; attribute the frame from its state extracted whole"
            )
        found = contributors.from_core(self.core.contributors(*box), self.state.objects)
        overlay = self.core.overlay_entries()
        if not overlay:
            return found
        return contributors.over(
            found, self.drawer.contributors(self.state, overlay, box)
        )

    def _refuse_undecided(self, overlay: Sequence[tuple[str, int, int]]) -> None:
        """Refuses (``render.unable``) a frame the core would draw without
        what the writer decides before the snap (:mod:`.canonical`): a view
        without its input grid, a record drawing paths without its planes, a
        shaded one without its normals. The core would decide them from
        snapped points, which is the writer's bug, never a frame to draw on
        another path. Only what the core draws is read: the records of its
        order it does not leave to Cairo's ``overlay``, which draws them from
        their own points.
        """
        view = self.state.view
        assert view is not None
        if not view.get("input_grid", 0) > 0:
            raise Unable(
                "render.unable: the frame reaches the core without an input grid"
            )
        shading = shades(view)
        cairo = {slot for _, slot, _ in overlay}
        for _, slot, _ in self.state.order:
            if slot in cairo:
                continue
            record = self.state.records[slot]
            fields = record.fields
            # A record that draws paths: its own, or instances of a path (an
            # instances record of a mesh, which the Python stream does not
            # hold, draws none).
            key = fields.get("shape")
            prototype = None if key is None else self.state.shapes.get(key)
            if (
                record.kind not in _PATH_KINDS
                or prototype is None
                or prototype.kind != "path"
            ):
                continue
            if "planes" not in fields:
                raise Unable(
                    f"render.unable: slot {slot} reaches the core without its planes"
                )
            if (
                shading
                and fields.get("shade_in_3d")
                and fields.get("pre_projected") is None
                and "normals" not in fields
            ):
                raise Unable(
                    f"render.unable: shaded slot {slot} reaches the core without its normals"
                )

    def settle(self) -> None:
        """Leave each display's camera of the last frame holding the picture
        CE's last capture leaves in it, as :meth:`IrCairoPainter.settle
        <.painter.IrCairoPainter.settle>` does: displays are Cairo's. The
        sub-frame's records are drawn as the core was handed them, snapped,
        so its view is too, as the display's record carries it.
        """
        for display, (_, view, order) in self.stream.last_sub_frames:
            camera = display.camera
            camera.pixel_array = self.drawer.capture(
                self.state,
                self.canonical.snapped_view(view),
                order,
                camera.background,
            )
