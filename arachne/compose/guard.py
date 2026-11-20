"""A render of a scene holding a GPU-only capability is refused.

A render draws each play as soon as it has run, before the scene's record
is complete, so the route the complete record would choose is not known
when its first frames are drawn; this build draws such a scene in a
session, never in a render. Until renders draw from the complete record,
a render checks, at every capture it draws from and at the scene's end,
whether the record holds a capability, and refuses at the first that does:

- with ``render.gpu_live`` when the scene is drawn as it runs (its record
  can't draw its frames), on every host;
- with ``render.gpu_required`` on a host with no GPU;
- with ``render.gpu_live`` on a host with one, saying that renders of such
  scenes are drawn once their record is complete, which is still to come.

Frames drawn before the refusal were written into the render's staging,
which the refusal removes (:mod:`arachne.scene.staging`): nothing is
published.
"""

from __future__ import annotations

__all__ = ["RECORD_FIRST", "check_render"]

from typing import TYPE_CHECKING

from arachne import routing

from . import device

if TYPE_CHECKING:
    from arachne.timeline.record import Record

#: Why a render of a recordable scene holding a capability is refused on a
#: host with a GPU.
RECORD_FIRST = (
    "a render draws each play as soon as it has run, and a scene holding a"
    " GPU-only capability is drawn once its record is complete, which renders"
    " do not do yet: draw it in a session"
)


def check_render(record: Record, live: str | None = None) -> None:
    """Refuse a render whose ``record``, as far as it has run (its final
    capture too, once the script has ended), holds a capability; ``live``
    is why a play of the render was drawn as it ran, if one was.
    """
    if not routing.ever_declared():
        # No object of this process ever held a capability.
        return
    found = routing.capabilities(record)
    if not found:
        return
    cause = live if live is not None else record.refusal
    if cause is not None:
        raise routing.GpuLive(found, cause)
    if device.pin() is None:
        raise routing.GpuRequired(found)
    raise routing.GpuLive(found, RECORD_FIRST)
