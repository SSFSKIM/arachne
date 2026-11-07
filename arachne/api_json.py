"""A session's answers as JSON: the forms a client outside the process reads.

Each form is a dict of JSON values, described by the schema of its name in
``arachne/schemas/`` (:func:`schema`):

- ``record``, :func:`inspect`: the record's timeline, the identity of the
  execution that made it, where each mobject was made, and the divergences
  found;
- ``state``, :func:`state`: :meth:`~arachne.session.Session.state_at`'s
  answer;
- ``query``, :func:`query`: the answers of a
  :class:`~arachne.lint.query.Query`;
- ``key_frames``, :func:`key_frames`: the key frames written;
- ``lint_result``, :func:`lint`: a lint's report and its notes
  (:mod:`arachne.lint.serialise`);
- ``frame``, :func:`frame`: a frame drawn and written as a PNG;
- ``scene_error``, :func:`scene_error`: what executing a scene raised,
  explained (:class:`~arachne.errors.SceneError`).

Every form carries ``schema_version``, ``"major.minor"``: a field added is a
minor version, a field changed or removed a major one. The record's form
holds no samples, values or content keys: what a mobject is at an instant is
the ``state`` form's answer.

Times are seconds, and beside each the exact instant as a rational string
``"n/d"`` (``t`` and ``t_exact``), which :class:`fractions.Fraction` reads
back. Boxes are ``[x_min, y_min, x_max, y_max]``: ``bbox`` in scene units, y
up; ``screen_bbox`` in pixels of ``target_resolution``, the session's, from
the top-left corner, y down; a query's own boxes (``box``, ``frame``) in
home-frame units, as :mod:`arachne.lint.query` measures them. A
``source_line`` is a line number: a mobject's is the line of the scene file
that made it (the call that led to a helper module, when one did), a play's
or a marker's the line of its ``source``, the ``path:line`` of the scene's
code that ran it.
"""

from __future__ import annotations

__all__ = [
    "FORMS",
    "QUERY_ANSWERS",
    "SCHEMA_VERSION",
    "frame",
    "inspect",
    "key_frames",
    "lint",
    "query",
    "scene_error",
    "schema",
    "state",
]

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from importlib import resources
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .lint.core import target_resolution
from .lint.serialise import notes, report, violation

if TYPE_CHECKING:
    from .camera.projection import Box
    from .errors import Frame, SceneError
    from .lint.core import LintResult
    from .lint.keyframes import KeyFrames
    from .lint.query import CameraReplaced, Query, Thing
    from .session import Session
    from .timeline.execution import Divergence
    from .timeline.fallback import Answer
    from .timeline.probe import Probe
    from .timeline.record import Marker, PlayRecord, Record
    from .timeline.source import SourceLine
    from .timeline.state import NodeView, State
    from .typing import StrPath

#: The version of every form.
SCHEMA_VERSION = "1.0"

#: The forms, by the name of their schema.
FORMS = (
    "record",
    "state",
    "query",
    "key_frames",
    "lint_result",
    "frame",
    "scene_error",
)

#: What :func:`query` can be asked for, as :class:`~arachne.lint.query.Query`
#: names its answers.
QUERY_ANSWERS = (
    "bbox",
    "overlaps",
    "occlusions",
    "offscreen",
    "text_heights",
    "markers",
)


def schema(form: str) -> dict[str, Any]:
    """The JSON Schema of ``form``, one of :data:`FORMS`."""
    if form not in FORMS:
        raise ValueError(f"No form {form!r}: one of {', '.join(FORMS)}.")
    text = (
        resources.files("arachne")
        .joinpath("schemas", f"{form}.schema.json")
        .read_text(encoding="utf-8")
    )
    found: dict[str, Any] = json.loads(text)
    return found


def inspect(session: Session) -> dict[str, Any]:
    """The record's form: ``session``'s record, its timeline (plays and
    markers), identity and provenance, and what the session found of it
    (staleness, divergences, the probe).
    """
    from . import __version__

    record = session.record
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": __version__,
        "scene_file": record.scene_file,
        "scene_class": record.scene_class,
        **_timed("grid", record.grid),
        "target_resolution": _resolution(session),
        **_timed("duration", record.duration),
        "identity": {
            "source_digest": record.identity.source_digest,
            "input_digests": [
                {"path": path, "digest": digest}
                for path, digest in record.identity.input_digests
            ],
            "seed": record.identity.seed,
        },
        "generation": session.generation,
        "stale": session.stale,
        "plays": [_play(play) for play in record.plays],
        "markers": [_marker(marker) for marker in record.markers],
        "mobjects": _provenance(record),
        "divergences": [_divergence(found) for found in session.divergences],
        "probe": _probe(session.probe),
    }


def state(found: State, session: Session) -> dict[str, Any]:
    """``state_at``'s answer ``found``, of ``session``: the instant, the play
    and route that answered, and each mobject on the picture.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        **_timed("t", found.t),
        **_timed("as_of", found.as_of),
        "play": found.play,
        "route": found.route,
        "answer": found.answer,
        "target_resolution": _resolution(session),
        "roots": list(found.roots),
        "nodes": [_node(node) for node in found.nodes],
    }


def query(
    found: Query, what: str | Sequence[str], *, ids: Sequence[str] = ()
) -> dict[str, Any]:
    """The answers of ``found`` that ``what`` names (one or more of
    :data:`QUERY_ANSWERS`), each under its name; ``bbox`` boxes the mobjects
    whose stable ids ``ids`` lists. A measure of the picture the scene's
    camera code stands in the way of is the
    :class:`~arachne.lint.query.CameraReplaced` in its place.
    """
    asked = (what,) if isinstance(what, str) else tuple(what)
    unknown = [name for name in asked if name not in QUERY_ANSWERS]
    if unknown:
        raise ValueError(
            f"A query answers no {', '.join(map(repr, unknown))}: it answers"
            f" {', '.join(QUERY_ANSWERS)}."
        )
    if "bbox" in asked and not ids:
        raise ValueError("A query's bbox needs the ids of the mobjects to box.")
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        **_timed("t", found.t),
        "play": found.play,
        "answer": found.answer,
        "frame": _box(found.frame),
    }
    for name in asked:
        out[name] = _ANSWERS[name](found, ids)
    return out


def key_frames(found: KeyFrames) -> dict[str, Any]:
    """The key frames ``found`` lists, the plays with none and why, and the
    findings of frames the painter refused.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "frames": [
            {
                "play": frame.play,
                "kind": frame.kind,
                **_timed("t", frame.t),
                **_source(frame.line),
                "answer": frame.answer,
                "path": None if frame.path is None else os.fspath(frame.path),
            }
            for frame in found.frames
        ],
        "unshown": [
            {
                "play": unshown.play,
                "kind": unshown.kind,
                **_source(unshown.line),
                "reason": unshown.reason,
            }
            for unshown in found.unshown
        ],
        "findings": [violation(finding) for finding in found.findings],
    }


def lint(result: LintResult) -> dict[str, Any]:
    """A lint's form: its report and its notes, as
    :mod:`arachne.lint.serialise` writes them.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "report": report(result),
        "notes": notes(result),
    }


def frame(found: Answer, path: StrPath) -> dict[str, Any]:
    """The frame of ``found``, an answer of
    :meth:`~arachne.session.Session.answers`, written as the PNG ``path``:
    the instant asked, the instant whose state it shows, the play holding
    it and how it was answered.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        **_timed("t", found.state.t),
        **_timed("as_of", found.state.as_of),
        "play": found.state.play,
        "answer": found.state.answer,
        "path": os.fspath(path),
    }


def scene_error(error: SceneError) -> dict[str, Any]:
    """What executing a scene raised, explained (``error``), with the report
    that says it all (:meth:`~arachne.errors.SceneError.report`).
    """
    raised_in = error.raised_in
    return {
        "schema_version": SCHEMA_VERSION,
        "cls": error.cls,
        "message": error.message,
        "path": error.path,
        "line": error.line,
        **_timed("t", None if error.t is None else Fraction(error.t)),
        "frames": [_trace_frame(each) for each in error.frames],
        "raised_in": None if raised_in is None else _trace_frame(raised_in),
        "hints": list(error.hints),
        "report": error.report(),
    }


# The record


def _play(play: PlayRecord) -> dict[str, Any]:
    return {
        "index": play.index,
        "kind": play.kind,
        "route": play.route,
        **_timed("start", play.start),
        **_timed("end", play.end),
        **_timed("requested_end", play.requested_end),
        "held": play.held,
        "skipped": play.skipped,
        **_source(play.source_line),
        "animations": list(play.animations),
        "markers": [_marker(marker) for marker in play.markers],
    }


def _marker(marker: Marker) -> dict[str, Any]:
    return {
        "kind": marker.kind,
        **_timed("t", marker.t),
        "data": _plain(marker.data),
        **_source(marker.source_line),
    }


def _provenance(record: Record) -> dict[str, dict[str, Any]]:
    """Where each mobject the record holds was made, by stable id, with the
    name the scene's code last gave it.
    """
    names: dict[int, str] = {}
    for play in record.plays:
        names.update(play.names)
    names.update(record.final_names)
    table: dict[str, dict[str, Any]] = {}
    for mob in record.mobjects():
        origin = record.origin(mob)
        if origin.id in table:
            continue
        table[origin.id] = {
            "type": type(mob).__name__,
            "name": names.get(id(mob)),
            "source_line": origin.scene_line,
            "source": _path_line(origin.source_line),
        }
    return table


def _divergence(found: Divergence) -> dict[str, Any]:
    return {
        "code": found.code,
        "kind": found.kind,
        "play": found.play,
        **_source(found.source_line),
        "reason": found.reason,
        "changed_inputs": list(found.changed_inputs),
    }


def _probe(found: Probe | None) -> dict[str, Any] | None:
    if found is None:
        return None
    return {
        "settled": found.settled,
        "plays": [
            {"play": play.play, "outcome": play.outcome, "reason": play.reason}
            for play in found.plays
        ],
    }


# The state


def _node(node: NodeView) -> dict[str, Any]:
    return {
        "id": node.id,
        "source_line": node.scene_line,
        "source": _path_line(node.source_line),
        "name": node.name,
        "kind": node.type,
        "label_or_text": node.text if node.text is not None else node.name,
        "bbox": _box(node.scene_box),
        "screen_bbox": _box(node.screen_box),
        "z": node.z,
        "color": node.color,
        "children": list(node.children),
    }


# The query


def _bbox(found: Query, ids: Sequence[str]) -> Any:
    # Each asked for, so an id the frame doesn't hold raises either way.
    boxes = [found.bbox(node_id) for node_id in ids]
    if found.replaced is not None:
        return _replaced(found.replaced)
    return {
        node_id: _box(cast("Box | None", box))
        for node_id, box in zip(ids, boxes, strict=True)
    }


def _overlaps(found: Query, ids: Sequence[str]) -> Any:
    answer = found.overlaps()
    if not isinstance(answer, tuple):
        return _replaced(answer)
    return [
        {
            "code": contact.code,
            "a": _thing(contact.a),
            "b": _thing(contact.b),
            "gap": float(contact.gap),
            "required": None if contact.required is None else float(contact.required),
            "finding": violation(contact.finding),
        }
        for contact in answer
    ]


def _occlusions(found: Query, ids: Sequence[str]) -> Any:
    answer = found.occlusions()
    if not isinstance(answer, tuple):
        return _replaced(answer)
    return [
        {
            "text": _thing(each.text),
            "over": _thing(each.over),
            "depth": float(each.depth),
            "finding": violation(each.finding),
        }
        for each in answer
    ]


def _offscreen(found: Query, ids: Sequence[str]) -> Any:
    answer = found.offscreen()
    if not isinstance(answer, tuple):
        return _replaced(answer)
    return [
        {
            "thing": _thing(each.thing),
            "edge": each.edge,
            "past": float(each.past),
            "finding": violation(each.finding),
        }
        for each in answer
    ]


def _text_heights(found: Query, ids: Sequence[str]) -> Any:
    answer = found.text_heights()
    if not isinstance(answer, tuple):
        return _replaced(answer)
    return [
        {
            "thing": _thing(each.thing),
            "units": float(each.units),
            "pixels": float(each.pixels),
        }
        for each in answer
    ]


def _markers(found: Query, ids: Sequence[str]) -> Any:
    return [_marker(marker) for marker in found.markers()]


_ANSWERS: dict[str, Callable[[Query, Sequence[str]], Any]] = {
    "bbox": _bbox,
    "overlaps": _overlaps,
    "occlusions": _occlusions,
    "offscreen": _offscreen,
    "text_heights": _text_heights,
    "markers": _markers,
}


def _thing(thing: Thing) -> dict[str, Any]:
    return {
        "id": thing.id,
        "type": thing.type,
        "name": thing.name,
        "source_line": thing.line,
        "text": thing.text,
        "box": _box(thing.box),
    }


def _replaced(found: CameraReplaced) -> dict[str, Any]:
    return {
        "code": found.code,
        "replaced": list(found.replaced),
        "message": found.message,
    }


# The scene's error


def _trace_frame(found: Frame) -> dict[str, Any]:
    return {
        "path": found.path,
        "line": found.line,
        "function": found.function,
        "code": found.code,
    }


# Values


def _resolution(session: Session) -> dict[str, int]:
    width, height = target_resolution(session)
    return {"width": width, "height": height}


def _timed(name: str, t: Fraction | None) -> dict[str, Any]:
    """``t`` as ``name``, in seconds, and as ``name_exact``, ``"n/d"``."""
    if t is None:
        return {name: None, f"{name}_exact": None}
    return {name: float(t), f"{name}_exact": f"{t.numerator}/{t.denominator}"}


def _source(line: SourceLine | None) -> dict[str, Any]:
    """``line`` as a line number and its ``path:line``."""
    return {
        "source_line": None if line is None else line.line,
        "source": _path_line(line),
    }


def _path_line(line: SourceLine | None) -> str | None:
    return None if line is None else str(line)


def _box(box: Sequence[float] | None) -> list[float] | None:
    return None if box is None else [float(v) for v in box]


def _plain(value: Any) -> Any:
    """``value`` (a marker's data, which the scene's code gives) as JSON
    values: numbers as numbers, collections as lists and objects, and
    anything else by its ``repr``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, np.generic):
        item = value.item()
        # An extended-precision scalar gives itself back: no Python number
        # holds it.
        return repr(value) if isinstance(item, np.generic) else _plain(item)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return repr(value)
