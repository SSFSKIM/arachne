"""Clearance inside a diagram: its labels against each other, against the
frame's edge and for their height.

The sweep's pair rules let two pieces of one top-level mobject touch
(they were built to), so two labels inside one
:class:`~arachne.mobject.diagram.Diagram` could cover each other and the
sweep stay silent. This rule reads only the texts a diagram holds, at
every instant it is asked, and reports them under the clearance codes:

- ``overlap.text`` and ``gap.text``: two labels of one top-level mobject
  closer than the terms' ``pad`` (or the declared ``text_gap``), measured
  box to box, each label's box its pill's when it sits on one (its plate:
  :meth:`~arachne.mobject.mobject.Mobject.add_background_rectangle`), its
  glyphs' otherwise. Labels under two top-level mobjects are the sweep's.
- ``clip.frame_edge``: a label past the frame's edge, a pill by any amount
  past rounding, bare glyphs by more than the frame-edge rule's slack.
- ``text.small``: a label shorter than the legible height, a note, as the
  text-height rule measures it.
- ``overlap.text`` for a label a flow set beside its connector's line
  (:class:`~arachne.mobject.diagram.Flow`): its pill on another piece of
  the diagram, another connector or another label, the message naming the
  connector's ``label_side=`` as the fix. What the diagram holds, and
  whether a connector has carried a flow (its ``flow_reach``, which stays
  once a flow is made, the flow taken out or not), are read at the
  instant measured. A piece is measured by its box; another connector by
  its drawn line, each curve sampled at 17 points, as wide as
  :data:`LINE_REACH` (0.03 units) each side of its centre, so a line that
  passes the pill without touching it is clear though its box holds it.

A diagram is an engine :class:`~arachne.mobject.diagram.Diagram` (or a
class made from it), known by the mobject the record holds, never by a
class's name: a scene's own class called ``Diagram`` is not one. A scene
with no diagram has no finding here, whatever else it draws.
"""

from __future__ import annotations

__all__ = [
    "Beside",
    "DiagramRule",
    "beside_labels",
    "diagram_clear",
    "diagram_findings",
    "diagram_ids",
]

import itertools
import math
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from ..findings import Finding, Holder
from ..parts import Part, Parts
from ._excuse import muted, paired
from ._geometry import box_gap
from ._rule import InstantRule, Terms
from ._units import Measured, measured_units
from .frame_edge import EDGE_SLACK
from .text_small import SMALL_PIXELS, small_findings, text_heights

if TYPE_CHECKING:
    from arachne.camera.projection import Box
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Record

    from ..core import LintContext

#: A shortfall under this is noise: under a pixel at any size.
NOISE = 0.005

#: The stable ids of the diagrams of the record a lint is reading, for the
#: length of the rule's call.
_DIAGRAMS: ContextVar[frozenset[str]] = ContextVar("diagrams", default=frozenset())
#: The record a lint is reading, for the length of the rule's call.
_RECORD: ContextVar[Record | None] = ContextVar("record", default=None)
#: How far either side of its centre a connector's drawn line reaches, in
#: units, for a label set beside another line: half a default connector's
#: stroke, with room.
LINE_REACH = 0.03
#: What a label set beside its line is told.
BESIDE_FIX = "set the connector's label_side= to a side with room"


def diagram_findings(parts: Parts, terms: Terms) -> list[Finding]:
    """The labels of the diagrams drawn at ``parts``' instant that crowd
    each other, run past the frame's edge or are too small to read: the
    diagrams of the record the rule is reading (:func:`diagram_ids`).
    """
    kinds = _DIAGRAMS.get()
    if not any(_held(part.within, kinds) for part in parts.texts):
        return []
    labels = [
        unit
        for unit in measured_units(parts)
        if unit.text
        and _held(unit.unit.within, kinds)
        and not muted(terms.exempt, unit)
    ]
    beside = _beside_at(parts.t)
    found = [*_edge_findings(parts, labels), *_beside_findings(parts, labels, beside)]
    on = [unit for unit in labels if unit.on(parts.frame)]
    for a, b in itertools.combinations(on, 2):
        finding = _pair(parts, terms, a, b, beside)
        if finding is not None:
            found.append(finding)
    small = {
        part.id
        for part, _, tall in text_heights(parts)
        if tall < SMALL_PIXELS and _held(part.within, kinds)
    }
    if small:
        found.extend(
            f for f in small_findings(parts, terms) if f.objects[0].id in small
        )
    return found


def _pair(
    parts: Parts,
    terms: Terms,
    a: Measured,
    b: Measured,
    beside: Mapping[str, Beside],
) -> Finding | None:
    if a.root != b.root:
        return None
    required = terms.pad
    if terms.text_gap is not None:
        required = max(required, terms.text_gap)
    gap = box_gap(a.outer, b.outer)
    if required - gap < NOISE or paired(terms.exempt, a, b):
        return None
    t = Fraction(parts.t)
    shown = f"labels {a.said!r} and {b.said!r}"
    fix = (
        f" {BESIDE_FIX}."
        if any(unit.id in beside for unit in (a, b)) and gap <= 0
        else ""
    )
    if gap > 0:
        return Finding(
            "gap.text",
            t,
            t,
            (a.object, b.object),
            message=(
                f"{shown} of one diagram sit {gap:.3f} units apart, closer than"
                f" the {required:.3f} labels keep."
            ),
            detail={"gap": round(gap, 4), "required": round(required, 4)},
        )
    return Finding(
        "overlap.text",
        t,
        t,
        (a.object, b.object),
        message=f"{shown} of one diagram overlap by {-gap:.3f} units.{fix}",
        detail={"overlap": round(-gap, 4)},
    )


def _beside_findings(
    parts: Parts, labels: list[Measured], beside: Mapping[str, Beside]
) -> list[Finding]:
    """Each label set beside its line whose pill lies on a drawn piece of
    its diagram, by the piece's box, or on another connector, by its drawn
    line (labels are :func:`_pair`'s): one finding a piece, at its deepest.
    """
    found = []
    for unit in labels:
        keep_off = beside.get(unit.id)
        if keep_off is None:
            continue
        deepest: dict[str, tuple[float, Part]] = {}
        for part in parts.parts:
            held = {holder.id for holder in part.within}
            if part.kind == "text" or unit.id in held or part.id == unit.id:
                continue
            ids = (part.id, *held)
            piece = next((i for i in ids if i in keep_off.connectors), None)
            if piece is not None:
                depth = LINE_REACH - _line_distance(unit.outer, part)
            else:
                piece = next((i for i in ids if i in keep_off.pieces), None)
                if piece is None:
                    continue
                depth = -box_gap(unit.outer, part.screen_box)
            if depth > NOISE and depth > deepest.get(piece, (0.0, part))[0]:
                deepest[piece] = (depth, part)
        for depth, part in deepest.values():
            t = Fraction(parts.t)
            what = part.name or part.type
            found.append(
                Finding(
                    "overlap.text",
                    t,
                    t,
                    (unit.object, part.as_object(parts)),
                    message=(
                        f"diagram label {unit.said!r}, set beside its flowing"
                        f" connector's line, lies {depth:.3f} units on the {what}"
                        f" there: {BESIDE_FIX}."
                    ),
                    detail={"overlap": round(depth, 4)},
                )
            )
    return found


def _line_distance(box: Box, part: Part) -> float:
    """How far the box stands from the line ``part`` draws (0 where the
    line runs through it): its curves sampled, as straight steps.
    """
    nearest = math.inf
    low, high = np.array(box[:2]), np.array(box[2:])
    corners = np.array([low, high, [low[0], high[1]], [high[0], low[1]]])
    for leaf in part.leaves:
        outline = leaf.outline()
        if outline is None:
            continue
        for curve in outline:
            for start, stop in zip(curve[:-1], curve[1:], strict=True):
                nearest = min(nearest, _step_distance(start, stop, low, high, corners))
    return nearest


def _step_distance(
    start: np.ndarray,
    stop: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    corners: np.ndarray,
) -> float:
    """How far the box ``low``-``high`` stands from the step ``start``-``stop``:
    0 where the step enters it.
    """
    step = stop - start
    enter, leave = 0.0, 1.0
    for axis in (0, 1):
        if abs(step[axis]) < 1e-15:
            if not low[axis] <= start[axis] <= high[axis]:
                enter, leave = 1.0, 0.0
            continue
        a = (low[axis] - start[axis]) / step[axis]
        b = (high[axis] - start[axis]) / step[axis]
        enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
    if enter <= leave:
        return 0.0
    ends = [
        float(np.linalg.norm(np.maximum(np.maximum(low - p, p - high), 0.0)))
        for p in (start, stop)
    ]
    length = float(step @ step)
    for corner in corners:
        t = (
            0.0
            if length == 0
            else min(max(float((corner - start) @ step) / length, 0.0), 1.0)
        )
        ends.append(float(np.linalg.norm(start + t * step - corner)))
    return min(ends)


def _edge_findings(parts: Parts, labels: list[Measured]) -> list[Finding]:
    left, bottom, right, top = parts.frame
    found = []
    for unit in labels:
        pill = len(unit.glyphs) < len(unit.leaves)
        x0, y0, x1, y1 = unit.outer
        beyond = {
            "left": left - x0,
            "right": x1 - right,
            "bottom": bottom - y0,
            "top": y1 - top,
        }
        slack = NOISE if pill else EDGE_SLACK
        crossed = {edge: past for edge, past in beyond.items() if past > slack}
        if not crossed:
            continue
        edge = max(crossed, key=lambda e: crossed[e])
        past = crossed[edge]
        what = "pill" if pill else "glyphs"
        found.append(
            Finding(
                "clip.frame_edge",
                Fraction(parts.t),
                Fraction(parts.t),
                (unit.object,),
                message=(
                    f"diagram label {unit.said!r} is cut by the frame's {edge} edge:"
                    f" its {what} run {past:.3f} units past it."
                ),
                detail={"edge": edge, "past": round(past, 4)},
            )
        )
    return found


def _held(within: tuple[Holder, ...], diagrams: frozenset[str]) -> bool:
    return any(holder.id in diagrams for holder in within)


@dataclass(frozen=True)
class Beside:
    """What a label set beside its line keeps off: the stable ids of its
    diagram's other pieces and of its other connectors.
    """

    pieces: frozenset[str]
    connectors: frozenset[str]


def _beside_at(t: Fraction) -> Mapping[str, Beside]:
    """:func:`beside_labels` of the record the rule is reading, at ``t``."""
    record = _RECORD.get()
    return beside_labels(record, t) if record is not None else {}


def beside_labels(record: Record, t: Fraction | None = None) -> dict[str, Beside]:
    """The labels of ``record``'s engine diagrams set beside their lines at
    ``t`` (as the record ends, when ``None``), by stable id, each with what
    it keeps off there: the diagram's other pieces and connectors, as the
    capture answering ``t`` holds them. A label is set beside its line once
    its connector has carried a flow (its ``flow_reach``).
    """
    from arachne.mobject.diagram import Diagram, Flow

    capture = None
    if t is not None:
        try:
            _, capture = record.capture_at(Fraction(t))
        except LookupError:
            capture = None

    def held(mob: Mobject, name: str, default: Any) -> Any:
        if capture is None:
            return getattr(mob, name, default)
        state = record.node_state(mob, capture)
        return default if state is None else state.get(name, default)

    def members(mob: Mobject) -> tuple[Mobject, ...]:
        if capture is None:
            return tuple(mob.submobjects)
        state = record.node_state(mob, capture)
        return () if state is None else state.submobjects

    found: dict[str, Beside] = {}
    for diagram in record.mobjects():
        if not isinstance(diagram, Diagram):
            continue
        inside = members(diagram)
        edges = [e for e in held(diagram, "edges", ()) if any(e is m for m in inside)]
        labels = {id(held(e, "label", None)) for e in edges}
        for edge in edges:
            label = held(edge, "label", None)
            if label is None or not float(held(edge, "flow_reach", 0.0)) > 0:
                continue
            others = [
                m
                for m in inside
                if m is not edge and id(m) not in labels and not isinstance(m, Flow)
            ]
            connectors = {id(e) for e in edges}
            found[record.origin(label).id] = Beside(
                pieces=frozenset(
                    record.origin(m).id for m in others if id(m) not in connectors
                ),
                connectors=frozenset(
                    record.origin(m).id for m in others if id(m) in connectors
                ),
            )
    return found


def diagram_ids(record: Record) -> frozenset[str]:
    """The stable ids of the engine diagrams ``record`` holds."""
    from arachne.mobject.diagram import Diagram

    return frozenset(
        record.origin(mob).id for mob in record.mobjects() if isinstance(mob, Diagram)
    )


@dataclass(frozen=True)
class DiagramRule(InstantRule):
    """An instant rule that knows, for its call, which mobjects of the
    record it reads are diagrams.
    """

    def __call__(self, context: LintContext) -> list[Finding]:
        held = _DIAGRAMS.set(diagram_ids(context.record))
        record = _RECORD.set(context.record)
        try:
            return super().__call__(context)
        finally:
            _RECORD.reset(record)
            _DIAGRAMS.reset(held)


#: The rule: the clearance of the labels inside every diagram drawn.
diagram_clear = DiagramRule(diagram_findings)
