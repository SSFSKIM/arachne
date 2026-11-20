"""A session's route, and how its frames are drawn on it.

A session's route is taken once from its complete record when it opens,
and again when a reload executes it anew (:class:`Route`, :func:`route_of`). It never asks the host first, and never per frame:
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
    "Refusals",
    "Route",
    "drawing",
    "live_refusal",
    "refuse_live",
    "refuse_pictures",
    "refuse_render",
    "route_of",
]

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from arachne import routing
from arachne.timeline.draw import Drawing
from arachne.timeline.source import SourceLine

from . import device
from .guard import RECORD_FIRST

if TYPE_CHECKING:
    from arachne.lint.raised import RaisedFinding
    from arachne.server.store import Manifest
    from arachne.session import Session
    from arachne.timeline.draw import LeaveOut
    from arachne.timeline.record import Record


@dataclass(frozen=True)
class Route:
    """The route a session's complete record chose: ``name``, ``"cpu"`` or
    ``"gpu"``; on ``gpu`` the capabilities it holds, with where each first
    appears; and ``live``, why the record must be drawn as it runs, if it
    must. It holds no record: a session keeps it from open to close, and a
    store keeps its form (:meth:`form`), which refuses without executing.
    """

    name: str
    capabilities: dict[str, routing.Where]
    live: str | None = None

    @classmethod
    def of(cls, record: Record) -> Route:
        """The route of ``record``, which is complete."""
        if record.final is None:
            raise ValueError(
                "A route is taken from a complete record: this one is not."
            )
        # As routing.route reads it: a record holding a capability is on gpu.
        found = routing.capabilities(record) if routing.ever_declared() else {}
        if not found:
            return cls("cpu", {})
        return cls("gpu", found, record.refusal)

    def form(self) -> dict[str, Any] | None:
        """What a stored manifest keeps of a ``gpu`` route, ``None`` on
        ``cpu``: each capability's first appearance and the live cause.
        """
        if self.name != "gpu":
            return None
        return {
            "capabilities": {
                row: {
                    "play": where.play,
                    "t": str(where.t),
                    "line": None
                    if where.line is None
                    else [where.line.path, where.line.line],
                    "object": where.object,
                }
                for row, where in self.capabilities.items()
            },
            "live": self.live,
        }

    @classmethod
    def from_form(cls, name: str, form: dict[str, Any] | None) -> Route:
        """The route a manifest stored as ``name`` and :meth:`form`."""
        if name == "cpu":
            return cls("cpu", {})
        assert form is not None, "a gpu route is stored with its form"
        found = {
            row: routing.Where(
                where["play"],
                Fraction(where["t"]),
                None if where["line"] is None else SourceLine(*where["line"]),
                where["object"],
            )
            for row, where in form["capabilities"].items()
        }
        return cls("gpu", found, form["live"])

    def refuse_live(self) -> None:
        """Raise ``render.gpu_live`` for a ``gpu`` route drawn as it runs."""
        if self.name == "gpu" and self.live is not None:
            raise routing.GpuLive(self.capabilities, self.live)

    def refuse_here(self) -> None:
        """Refuse drawing a ``gpu`` route on this host before anything runs:
        ``render.gpu_live`` drawn as it runs, ``render.gpu_required`` with
        no GPU.
        """
        if self.name != "gpu":
            return
        self.refuse_live()
        if device.pin() is None:
            raise routing.GpuRequired(self.capabilities)

    def refuse_render(self) -> None:
        """Refuse a render or a clip of a ``gpu`` route before anything of it
        runs: as :meth:`refuse_here`, else ``render.gpu_live`` until renders
        draw from a complete record.
        """
        if self.name != "gpu":
            return
        self.refuse_here()
        raise routing.GpuLive(self.capabilities, RECORD_FIRST)


@dataclass(frozen=True)
class Refusals:
    """Why a scene's pictures, or a render of it, are refused before
    anything of it runs, in the order they are refused: ``found``, the
    finding its code recorded that refuses drawing it
    (:func:`~arachne.lint.raised.found`), an error on every host and the
    author's to fix; then its ``route``'s (:class:`Route`). A session's are
    read from its record and route (:meth:`of`), a stored session's from
    its manifest (:meth:`stored`): one decision either way, without
    executing a stored scene.
    """

    found: RaisedFinding | None
    route: Route

    @classmethod
    def of(cls, session: Session) -> Refusals:
        record = session.record
        if not record.bridge.calls:
            return cls(None, route_of(session))
        from arachne.lint.raised import drawing_refusal

        return cls(drawing_refusal(record), route_of(session))

    @classmethod
    def stored(cls, manifest: Manifest) -> Refusals:
        kept = manifest.refuses
        found = None
        if kept is not None:
            from arachne.lint.raised import RaisedFinding

            found = RaisedFinding(
                kept["code"], kept["message"], detail=kept["detail"], line=kept["line"]
            )
        return cls(found, Route.from_form(manifest.route, manifest.gpu))

    def refuse_here(self) -> None:
        """Refuse drawing a picture of the scene on this host."""
        if self.found is not None:
            raise self.found
        self.route.refuse_here()

    def refuse_render(self) -> None:
        """Refuse a render or a clip of the scene."""
        if self.found is not None:
            raise self.found
        self.route.refuse_render()


def refuse_pictures(session: Session, record: Record) -> None:
    """Refuse drawing pictures of ``record`` (the session's own, or a
    forward run's) before anything is drawn: for the finding its scene's
    code recorded that refuses drawing it, then for a ``gpu`` route drawn
    as it runs (:class:`Refusals`, in that order).
    """
    if record.bridge.calls:
        from arachne.lint.raised import refuse_drawing

        refuse_drawing(record)
    refuse_live(session)


def route_of(session: Session) -> Route:
    """The route ``session`` took from its record when it was opened or
    last reloaded.
    """
    found: Route | None = session._route
    if found is None:
        raise RuntimeError("A session's route is taken when it is opened.")
    return found


def live_refusal(session: Session) -> routing.GpuLive | None:
    """The ``render.gpu_live`` refusal of a ``gpu``-route session whose
    record refuses drawing, else ``None``.
    """
    route = route_of(session)
    if route.name != "gpu" or route.live is None:
        return None
    return routing.GpuLive(route.capabilities, route.live)


def refuse_live(session: Session) -> None:
    """Raise :func:`live_refusal`'s refusal, if the session has one."""
    route_of(session).refuse_live()


def refuse_render(session: Session) -> None:
    """Refuse a render or a clip of a ``gpu``-route session's scene before
    anything of it runs (:meth:`Route.refuse_render`).
    """
    route_of(session).refuse_render()


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
