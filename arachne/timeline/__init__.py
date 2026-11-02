"""The scene's timeline: its exact clock and the forward record of its run.

:func:`record` executes a scene and returns its :class:`Record`: every play
and wait with its exact span, kind, source line, route, held flag and
markers, the state the scene was in at its start, and its samples, from
which :meth:`Record.state_at` answers any instant.
"""

from __future__ import annotations

__all__ = [
    "Capture",
    "Identity",
    "Marker",
    "NodeView",
    "PlayRecord",
    "Record",
    "SourceLine",
    "State",
    "exact",
    "record",
]

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .clock import exact
from .record import Capture, Identity, Marker, PlayRecord, Record
from .source import SourceLine
from .state import NodeView, State

if TYPE_CHECKING:
    from arachne.typing import StrPath


def record(
    scene_file: StrPath,
    scene_class: str,
    *,
    grid: Fraction | float | None | Literal["render"] = "render",
) -> Record:
    """Execute ``scene_class`` from ``scene_file`` and return its record.

    ``grid`` is the frame rate whose instants the record keeps: by default
    the configured render's (:meth:`.Session.open`); ``None`` keeps every
    instant the scene was evaluated at, so the record answers any time.
    """
    from arachne.session import Session

    return Session.open(Path(scene_file), scene_class, grid=grid).record
