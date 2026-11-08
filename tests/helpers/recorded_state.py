"""What a mobject or a record holds, as plain values two processes compare.

A test that claims two constructions, or two executions, are the same
compares these: every mobject's class, the keys of its ``__dict__`` in
order and each value, the shape of the graph (which mobject holds which,
shared ones included), and each mobject's origin. Identity fields
(``original_id``) count by their presence only, since their values name
objects of one process.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import types
from collections.abc import Iterable
from pathlib import PurePath
from typing import Any

import numpy as np

from arachne.mobject.mobject import Mobject
from arachne.timeline.origins import ORIGINS
from arachne.timeline.record import IDENTITY_FIELDS, Capture, RegisteredState

__all__ = ["mobject_state", "record_state", "frame_digests"]


class _Canon:
    """Turns values into plain ones, naming each mobject by ``name``."""

    def __init__(self, name) -> None:
        self.name = name
        self._seen: set[int] = set()

    def value(self, value: Any, depth: int = 0) -> Any:
        if isinstance(value, Mobject):
            return ("mob", self.name(value))
        if value is None or isinstance(value, (bool, str, bytes)):
            return (type(value).__name__, value)
        if isinstance(value, float):
            return (type(value).__name__, float(value).hex())
        if isinstance(value, int):
            return (type(value).__name__, int(value))
        if isinstance(value, complex):
            return ("complex", value.real.hex(), value.imag.hex())
        if isinstance(value, np.ndarray):
            return (
                type(value).__name__,
                value.dtype.str,
                value.shape,
                np.ascontiguousarray(value).tobytes(),
            )
        if isinstance(value, np.generic):
            return (type(value).__name__, self.value(value.item(), depth))
        key = getattr(type(value), "key", None)
        if key is not None and isinstance(key, property):
            # Paint and Placement: their own content key.
            return (type(value).__name__, value.key)
        if isinstance(value, PurePath):
            return (type(value).__name__, str(value))
        if isinstance(value, enum.Enum):
            return (type(value).__qualname__, value.name)
        if isinstance(value, (types.FunctionType, types.MethodType, type)):
            return ("code", getattr(value, "__module__", ""), value.__qualname__)
        if isinstance(value, (list, tuple)):
            return (
                type(value).__name__,
                tuple(self.value(item, depth + 1) for item in value),
            )
        if isinstance(value, dict):
            return (
                type(value).__name__,
                tuple(
                    (self.value(k, depth + 1), self.value(v, depth + 1))
                    for k, v in value.items()
                ),
            )
        if isinstance(value, (set, frozenset)):
            return (type(value).__name__, tuple(sorted(map(repr, value))))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return (
                type(value).__qualname__,
                tuple(
                    (f.name, self.value(getattr(value, f.name), depth + 1))
                    for f in dataclasses.fields(value)
                ),
            )
        internal = getattr(value, "_internal_value", None)
        if isinstance(internal, np.ndarray):
            # A colour.
            return (type(value).__qualname__, internal.tobytes())
        own = getattr(value, "__dict__", None)
        if own is None or depth > 6 or id(value) in self._seen:
            return (type(value).__qualname__,)
        self._seen.add(id(value))
        try:
            return (
                type(value).__qualname__,
                tuple((k, self.value(v, depth + 1)) for k, v in own.items()),
            )
        finally:
            self._seen.discard(id(value))

    def fields(self, keys: Iterable[str], values: Iterable[Any]) -> tuple:
        return tuple(
            (key, ("identity",) if key in IDENTITY_FIELDS else self.value(value))
            for key, value in zip(keys, values, strict=True)
        )


def _origin(mob: Mobject) -> tuple | None:
    origin = ORIGINS.get(mob)
    if origin is None:
        return None
    return (origin.id, str(origin.source_line), origin.scene_line)


def mobject_state(root: Mobject, *, origins: bool = False) -> list[tuple]:
    """Every mobject reachable from ``root`` through what it holds (its
    submobjects, and mobjects among its values), each once, in the order
    first met: its class, place, fields and (with ``origins``) its origin.
    Mobjects are named by their place, so sharing shows.
    """
    places: dict[int, int] = {}
    order: list[Mobject] = []

    def name(mob: Mobject) -> int:
        if id(mob) not in places:
            places[id(mob)] = len(order)
            order.append(mob)
        return places[id(mob)]

    canon = _Canon(name)
    name(root)
    found = []
    index = 0
    while index < len(order):
        mob = order[index]
        own = vars(mob)
        found.append(
            (
                index,
                type(mob).__qualname__,
                canon.fields(own.keys(), own.values()),
                _origin(mob) if origins else None,
            )
        )
        index += 1
    return found


def _capture_state(record: Any, capture: Capture | None) -> tuple | None:
    if capture is None:
        return None

    def name(mob: Mobject) -> str:
        return record.origin(mob).id

    canon = _Canon(name)
    roots = (*capture.roots, *capture.foreground, *capture.moving)
    nodes = tuple(
        (
            record.origin(mob).id,
            str(record.origin(mob).source_line),
            record.origin(mob).scene_line,
            type(mob).__qualname__,
            canon.fields(state.keys, state.values),
            tuple(name(child) for child in state.submobjects),
        )
        for mob, state in record._walk(capture, roots)
    )
    return (
        str(capture.seq),
        str(capture.instant),
        capture.kind,
        str(capture.until),
        tuple(name(m) for m in capture.roots),
        tuple(name(m) for m in capture.foreground),
        tuple(name(m) for m in capture.moving),
        RegisteredState(record, capture).content_key(),
        nodes,
    )


def record_state(record: Any) -> dict[str, Any]:
    """What ``record`` holds, but the number of the execution that made it
    and anything naming an object of this process.
    """
    states = record.states
    return {
        "identity": record.identity,
        "scene": (record.scene_class, str(record.grid), str(record.duration)),
        "refusal": record.refusal,
        "plays": tuple(
            (
                play.index,
                str(play.start),
                str(play.end),
                play.kind,
                play.route,
                str(play.source_line),
                play.held,
                play.skipped,
                str(play.requested_end),
                play.animations,
            )
            for play in record.plays
        ),
        "markers": tuple(
            (marker.kind, str(marker.t), str(marker.source_line))
            for marker in record.markers
        ),
        "captures": tuple(_capture_state(record, c) for c in record._captures),
        "final": _capture_state(record, record.final),
        "calls": None
        if states is None
        else tuple(
            (index, _capture_state(record, c))
            for index, c in sorted(states.calls.items())
        ),
        "exits": None
        if states is None
        else tuple(
            (index, _capture_state(record, c))
            for index, c in sorted(states.exits.items())
        ),
        "removals": None
        if states is None
        else tuple(_capture_state(record, r.capture) for r in states.removals),
    }


def frame_digests(session: Any, fps: int) -> list[str]:
    """The sha256 of the session's frame at every instant of the grid at
    ``fps``.
    """
    return [
        hashlib.sha256(np.asarray(session.frame(sample.t)).tobytes()).hexdigest()
        for sample in session.record.samples(fps)
    ]
