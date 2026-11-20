"""A session's route, and how its frames are drawn on it.

A session's route is taken once from its complete record, which it holds
from the moment it opens (:func:`route_of`), and again only when it holds
another (a reload). It never asks the host first, and never per frame:
every frame the session gives, a forward run's too, whose record is a
prefix of the scene, is drawn on the session's route (:func:`drawing`).

On the ``cpu`` route nothing changes, and nothing imports the GPU build. On
the ``gpu`` route every frame is drawn as a stack on this process's device
(:mod:`.stack`, :mod:`.device`), and:

- on a host with no GPU, drawing refuses with ``render.gpu_required``;
- a scene whose record refuses drawing must be drawn as it runs, which only
  Cairo does, so every answer that needs that run (a frame, a state, a
  query) refuses with ``render.gpu_live`` on every host (:func:`refuse_live`),
  and its lint reports that one error;
- a render or a clip of it is refused before it runs (:func:`refuse_render`):
  renders draw from a complete record only later.
"""

from __future__ import annotations

__all__ = [
    "Route",
    "drawing",
    "live_refusal",
    "refuse_live",
    "refuse_render",
    "route_of",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arachne import routing
from arachne.timeline.draw import Drawing

from . import device
from .guard import RECORD_FIRST

if TYPE_CHECKING:
    from arachne.session import Session
    from arachne.timeline.draw import LeaveOut
    from arachne.timeline.record import Record


@dataclass(frozen=True)
class Route:
    """The route of a session's complete ``record``: ``name``, ``"cpu"`` or
    ``"gpu"``, and on ``gpu`` the capabilities it holds, with where.
    """

    record: Record
    name: str
    capabilities: dict[str, routing.Where]


def route_of(session: Session) -> Route:
    """The route of the record ``session`` holds, taken once per record."""
    held: Route | None = session._route
    record = session.record
    if held is None or held.record is not record:
        if record.final is None:
            raise ValueError("A session's record is complete: this one's is not.")
        # As routing.route reads it: a record holding a capability is on gpu.
        found = routing.capabilities(record) if routing.ever_declared() else {}
        held = session._route = Route(record, "gpu" if found else "cpu", found)
    return held


def live_refusal(session: Session) -> routing.GpuLive | None:
    """The ``render.gpu_live`` refusal of a ``gpu``-route session whose
    record refuses drawing, else ``None``.
    """
    route = route_of(session)
    refusal = session.record.refusal
    if route.name != "gpu" or refusal is None:
        return None
    return routing.GpuLive(route.capabilities, refusal)


def refuse_live(session: Session) -> None:
    """Raise :func:`live_refusal`'s refusal, if the session has one."""
    refused = live_refusal(session)
    if refused is not None:
        raise refused


def refuse_render(session: Session) -> None:
    """Refuse a render or a clip of a ``gpu``-route session's scene before
    anything of it runs: ``render.gpu_live`` for one drawn as it runs, else
    ``render.gpu_required`` on a host with no GPU, else ``render.gpu_live``
    until renders draw from a complete record.
    """
    route = route_of(session)
    if route.name != "gpu":
        return
    refuse_live(session)
    if device.pin() is None:
        raise routing.GpuRequired(route.capabilities)
    raise routing.GpuLive(route.capabilities, RECORD_FIRST)


def drawing(
    session: Session, record: Record, *, leave_out: LeaveOut | None = None
) -> Drawing:
    """The drawing ``session`` draws frames of ``record`` with (its own, or
    a forward run's), on the session's route; given ``leave_out``, a
    capture filter (:class:`~arachne.timeline.draw.Drawing`).
    """
    route = route_of(session)
    if route.name == "cpu":
        return Drawing(record, draw_path=session.draw_path, leave_out=leave_out)
    from .stack import Stack

    refuse_live(session)
    camera = record._scene.renderer.camera
    opened = device.compositor(camera.pixel_width, camera.pixel_height)
    if opened is None:
        raise routing.GpuRequired(route.capabilities)
    session._renderer = renderer(opened)
    return Drawing(
        record, draw_path="cairo", leave_out=leave_out, stack=Stack(opened, record)
    )


def renderer(opened: Any) -> dict[str, Any]:
    """The ``renderer`` block of an answer drawn by ``opened``: its name and
    the pin of the device that drew.
    """
    return {"name": "gpu", "pin": dict(opened.pin)}
