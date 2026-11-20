"""A scene's route, read from its record, and the device it draws on.

A scene whose record holds a row of the table (:mod:`.table`) is drawn on
the GPU throughout; any other scene is drawn by Cairo and the core, on every
host. The choice depends on the record alone, never on the host and never on
which frame is asked for. Only the ``gpu`` route asks for a device, and only
then is the GPU build imported.
"""

from __future__ import annotations

__all__ = ["Where", "cache_token", "capabilities", "gpu", "pin_for", "route"]

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

from .table import ATTRIBUTE, GPU_ONLY

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, Record
    from arachne.timeline.source import SourceLine

Route = Literal["cpu", "gpu"]

#: The fields of a pin: the adapter's name, the backend, the driver version
#: the adapter reports and the GPU build's wgpu version.
PIN_FIELDS = ("adapter", "backend", "driver", "wgpu")

#: The adapters a compositor opens, by device type, most preferred first.
_DRAWING_TYPES = ("DiscreteGpu", "IntegratedGpu", "VirtualGpu")

#: Each row's place in the table, which orders the rows one object holds.
_ORDER = {row: place for place, row in enumerate(GPU_ONLY)}


@dataclass(frozen=True)
class Where:
    """Where a scene first holds a capability: the first frame that shows
    an object holding it.

    ``play`` is the index of the play whose frame it is, ``None`` for the
    scene as its script left it (after its last play, or with no play);
    ``t`` the instant of that frame; ``line`` the play's line of the scene,
    or past the last play the line that made the object; ``object`` the
    object's stable id.
    """

    play: int | None
    t: Fraction
    line: SourceLine | None
    object: str


def capabilities(record: Record) -> dict[str, Where]:
    """Every capability an object holds in a capture of ``record`` that a
    frame is drawn from, its final capture included, each with where it
    first appears, in that order. An object declared and never shown, or a
    capability withdrawn before its object is, counts for nothing.

    A pure read of the record: it walks each drawn capture from what a frame
    of it draws, so it answers for a record that refuses drawing too.
    The states the record keeps for the lint alone are left out, so that the
    answer does not depend on whether the record keeps them.
    """
    # Most records hold no capability: their versions say so at once.
    declared: set[str] = set()
    for node in record._nodes.values():
        for state in node.states:
            if ATTRIBUTE in state.keys:
                declared |= state.get(ATTRIBUTE)
    found: dict[str, Where] = {}
    if not declared:
        return found
    plays = _plays_by_capture(record)
    for capture in record._captures:
        for mob, state in record._walk(capture, _drawn(record, capture)):
            for row in sorted(state.get(ATTRIBUTE, ()), key=_ORDER.__getitem__):
                if row not in found:
                    found[row] = _where(record, plays, capture, mob)
        if len(found) == len(declared):
            break
    return found


def route(record: Record) -> Route:
    """``"gpu"`` exactly when ``record`` holds a capability, else ``"cpu"``.
    Asked of a complete record only: a record of part of a script (a clip
    that ended early, a forward run that stopped) takes the route of the
    scene's complete record and is never routed afresh.
    """
    if record.final is None:
        raise ValueError(
            "A route is asked of a complete record: this one's script has not ended."
        )
    return "gpu" if capabilities(record) else "cpu"


def gpu() -> dict[str, str] | None:
    """The pin of the device this process would draw on, or ``None`` when
    the GPU build is not installed or offers no GPU (one installed and
    broken raises its import error): the adapter a
    compositor opens (discrete, then integrated, then virtual; the first of
    a kind as listed), never a software adapter.
    """
    try:
        build: Any = importlib.import_module("arachne_gpu")
    except ModuleNotFoundError as error:
        # Only the build's own absence is no build: an installed build that
        # fails to import (a native part that cannot load) is raised.
        if error.name != "arachne_gpu":
            raise
        return None
    ranked = [
        (_DRAWING_TYPES.index(adapter["device_type"]), index, adapter)
        for index, adapter in enumerate(build.adapters())
        if adapter["device_type"] in _DRAWING_TYPES
    ]
    if not ranked:
        return None
    adapter = min(ranked, key=lambda entry: entry[:2])[2]
    return {
        "adapter": adapter["name"],
        "backend": adapter["backend"],
        "driver": adapter["driver_info"],
        "wgpu": build.__wgpu_version__,
    }


def pin_for(record: Record) -> dict[str, str] | None:
    """The pin ``record``'s scene draws under: ``None`` on the ``cpu`` route;
    on the ``gpu`` route this process's device, or :class:`.GpuRequired`
    where it has none.
    """
    if route(record) == "cpu":
        return None
    pin = gpu()
    if pin is None:
        from .refusals import GpuRequired

        raise GpuRequired(capabilities(record))
    return pin


def cache_token(route: str, pin: Mapping[str, str] | None) -> str:
    """What a cache key adds for a route: nothing for ``cpu``, so that every
    key of a scene drawn by Cairo is as it was; for ``gpu``, a digest of the
    pin, so that nothing drawn under one pin is reused under another.
    """
    if route == "cpu":
        if pin is not None:
            raise ValueError("The cpu route draws under no pin.")
        return ""
    if route != "gpu":
        raise ValueError(f"{route!r} is no route: 'cpu' or 'gpu'.")
    if pin is None:
        raise ValueError("The gpu route draws under a pin.")
    fields = json.dumps([pin[name] for name in PIN_FIELDS]).encode()
    return "gpu-" + hashlib.blake2b(fields, digest_size=8).hexdigest()


def _drawn(record: Record, capture: Capture) -> tuple[Mobject, ...]:
    """What a frame drawn from ``capture`` paints: the scene's last frame
    only what the scene holds (an object removed after the last play stays
    in the moving list, unpainted); a play's frame its moving objects over
    the scene.
    """
    if capture is record.final:
        return tuple(capture.draw_list())
    return (*capture.roots, *capture.foreground, *capture.moving)


def _plays_by_capture(record: Record) -> dict[int, int]:
    """The play each drawn capture of ``record`` belongs to, by seq."""
    found: dict[int, int] = {}
    for play in record.plays:
        found[play.static_seq] = play.index
        for sample in play.samples:
            found[sample.drawn_seq] = play.index
    return found


def _where(
    record: Record, plays: dict[int, int], capture: Capture, mob: Mobject
) -> Where:
    origin = record.origin(mob)
    index = plays.get(capture.drawn_seq)
    if index is None and capture.kind != "final":
        found = record.play_at(capture.instant)
        index = found.index if found is not None else None
    if index is None:
        return Where(None, capture.instant, origin.source_line, origin.id)
    play = record.plays[index]
    return Where(index, capture.instant, play.source_line, origin.id)
