"""The scene whose script is running now, for marks that keep its time."""

from __future__ import annotations

__all__ = ["running_scene"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arachne.scene.scene import Scene


def running_scene() -> Scene | None:
    """The scene whose script is running now, if any."""
    from arachne.timeline.recorder import running

    recorder = running()
    return recorder.scene if recorder is not None else None
