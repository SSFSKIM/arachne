"""Arachne's CPU core: draws frames described in XF, the Frame IR.

A :class:`Stream` holds what a drawer keeps across frames; each drawn frame is a
:class:`FrameDelta` against it. ``Stream.draw`` returns a uint8 array (height, width, 4) in the
layout of CE's pixel array, or raises :class:`Unsupported` (``render.unsupported``) when a drawn
record carries what this build cannot draw. ``read_xf`` and ``write_xf`` read and write XF's binary
form, used for fixtures.
"""

from __future__ import annotations

from arachne_core._core import (
    XF_VERSION,
    ApplyStats,
    FrameDelta,
    Stream,
    read_xf,
    write_xf,
)
from arachne_core._errors import Unsupported, XfError

__all__ = [
    "XF_VERSION",
    "ApplyStats",
    "FrameDelta",
    "Stream",
    "Unsupported",
    "XfError",
    "read_xf",
    "write_xf",
]
