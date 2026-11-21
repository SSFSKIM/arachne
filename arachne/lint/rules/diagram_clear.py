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
  once a flow is made, the flow taken out or not), are read from the
  record and capture the instant's parts were read from (a forward run's,
  when one answered the frame). Parts read from no capture (none the lint
  builds) report nothing of this kind: the rule never guesses from another
  instant's diagram, which could name a piece that is not there. A piece is measured by its box; another connector by
  its drawn line, each curve sampled at 17 points, as wide as
  :data:`LINE_REACH` (0.03 units) each side of its centre, so a line that
  passes the pill without touching it is clear though its box holds it.
- ``overlap.text`` for a label on its plate at its connector's middle (a
  connector that has carried no flow): its plate over its own line, leaving
  less than :data:`MIN_STUB` (0.08 screen units, a unit of the home frame)
  of the line showing on either side of it, so a viewer cannot see a line
  runs there. The plate hides the line only where it paints (some fill)
  and is drawn after it. The stubs run along the line as drawn (its
  corners, its ends where the head and the pieces cut it), each curve
  sampled at 65 points, from where the plate's box covers a length of it;
  each is measured in scene units and put on screen by the home frame's
  width over the frame's (a zoomed-in frame makes it longer), whatever the
  frame's height. The label is judged at the instants, and when shown, as
  its legible height is; the message gives the stubs in scene units at
  that frame and the fix, to widen the gap or drop the label.

A diagram is an engine :class:`~arachne.mobject.diagram.Diagram` (or a
class made from it), known by the mobject the record holds, never by a
class's name: a scene's own class called ``Diagram`` is not one. A scene
with no diagram has no finding here, whatever else it draws.
"""

from __future__ import annotations

__all__ = [
    "INLINE_FIX",
    "MIN_STUB",
    "Beside",
    "DiagramRule",
    "beside_labels",
    "diagram_clear",
    "diagram_findings",
    "diagram_ids",
    "inline_labels",
    "line_stubs",
]

import itertools
import math
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from ..findings import Finding, Holder
from ..parts import Part, Parts, source_of
from ._excuse import muted, paired
from ._geometry import box_gap, union
from ._rule import InstantRule, Terms
from ._units import Measured, measured_units
from .frame_edge import EDGE_SLACK
from .text_small import SMALL_PIXELS, small_findings, text_heights

if TYPE_CHECKING:
    from arachne.camera.projection import Box
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, Record

    from ..core import LintContext

#: A shortfall under this is noise: under a pixel at any size.
NOISE = 0.005

#: The stable ids of the diagrams of the record a lint is reading, for the
#: length of the rule's call.
_DIAGRAMS: ContextVar[frozenset[str]] = ContextVar("diagrams", default=frozenset())
#: How far either side of its centre a connector's drawn line reaches, in
#: units, for a label set beside another line: half a default connector's
#: stroke, with room.
LINE_REACH = 0.03
#: What a label set beside its line is told.
BESIDE_FIX = "set the connector's label_side= to a side with room"
#: The least stretch of its own line a label on its connector's middle
#: leaves showing on each side of its plate, in screen units: units of the
#: home frame (the configuration's), whatever the camera's frame is then;
#: about 5 pixels at -ql and 11 at 1080p. Under it a viewer cannot see a
#: line runs under the label. Measured on Linux's fonts: the shortest stub
#: of a scene that hid its line, 0.031; the shortest kept, 0.148 (a zoom's
#: inner wire at the frame it ends on) and 0.211 at the whole frame.
MIN_STUB = 0.08
#: What a label on its connector's middle that hides its line is told.
INLINE_FIX = "widen the gap or drop the label"
#: How far under :data:`MIN_STUB` a stub may fall as floating point's
#: error, not as a line shorter on screen.
STUB_TOLERANCE = 1e-9
#: The points each curve of a connector is sampled at for its stubs: 64
#: steps, as a path's arc length is read (``ArcLength``), so a curved stub
#: near the least is measured as the connector's own length is.
STUB_SAMPLES = 65


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
    source = source_of(parts)
    beside = beside_labels(*source) if source is not None else {}
    inline = inline_labels(*source) if source is not None else {}
    seen = text_heights(parts)
    shown = {part.id for part, _, _ in seen}
    found = [
        *_edge_findings(parts, labels),
        *_beside_findings(parts, labels, beside),
        *_inline_findings(parts, [unit for unit in labels if unit.id in shown], inline),
    ]
    on = [unit for unit in labels if unit.on(parts.frame)]
    for a, b in itertools.combinations(on, 2):
        finding = _pair(parts, terms, a, b, beside)
        if finding is not None:
            found.append(finding)
    small = {
        part.id
        for part, _, tall in seen
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


def _inline_findings(
    parts: Parts, labels: list[Measured], inline: Mapping[str, str]
) -> list[Finding]:
    """Each label on its connector's middle whose plate leaves less than
    :data:`MIN_STUB` of the connector's drawn line showing, on screen, on
    either side of it. A plate hides the line where it paints (some fill)
    and is drawn after it; a label with no such plate, a connector whose
    line is not drawn, or a plate off its line reports nothing.
    """
    found = []
    view = parts.view
    # Points are on screen, in home-frame units: back to scene units, and a
    # scene length on screen, by the home frame's width over the frame's.
    to_scene = (
        view.frame_width / view.home_width,
        view.frame_height / view.home_height,
    )
    scale = view.home_width / view.frame_width
    for unit in labels:
        connector = inline.get(unit.id)
        line = next((p for p in parts.parts if p.id == connector), None)
        if line is None:
            continue
        (drawn,) = line.leaves
        plates = [
            leaf.screen_box
            for leaf in unit.leaves
            if leaf.plate and leaf.fill_opacity > 0 and leaf.order > drawn.order
        ]
        curves = drawn.outline(STUB_SAMPLES)
        if not plates or curves is None:
            continue
        stubs = line_stubs(curves, union(plates), to_scene=to_scene)
        if stubs is None:
            continue
        before, after = stubs
        on_screen = (before * scale, after * scale)
        if not _shows_too_little(on_screen):
            continue
        t = Fraction(parts.t)
        found.append(
            Finding(
                "overlap.text",
                t,
                t,
                (unit.object, line.as_object(parts)),
                message=(
                    f"diagram label {unit.said!r} leaves too little of its own"
                    f" connector's line showing: {before:.3f} units before its"
                    f" plate and {after:.3f} after at this frame ({on_screen[0]:.3f}"
                    f" and {on_screen[1]:.3f} on screen, under the {MIN_STUB} a side"
                    f" needs to read as a line): {INLINE_FIX}"
                ),
                detail={
                    "before": round(before, 4),
                    "after": round(after, 4),
                    "before_screen": round(on_screen[0], 4),
                    "after_screen": round(on_screen[1], 4),
                },
                fix=INLINE_FIX,
            )
        )
    return found


def _shows_too_little(on_screen: tuple[float, float]) -> bool:
    """Whether either stub, on screen, falls short of :data:`MIN_STUB` by
    more than floating point's error.
    """
    return min(on_screen) < MIN_STUB - STUB_TOLERANCE


def line_stubs(
    curves: np.ndarray, box: Box, to_scene: tuple[float, float] = (1.0, 1.0)
) -> tuple[float, float] | None:
    """How far a line runs, along its path, before ``box`` first covers a
    length of it and after ``box`` last does: ``curves`` its curves as
    polylines, ``(curves, samples, 2)``, joined as straight steps, clipped
    against ``box`` where they lie; each length a step's run between its cut
    shares, with ``to_scene`` scaling its two axes (screen to scene units).
    A box that only touches the line covers none of it; a jump from one
    curve to the next, where they do not meet, runs no length. ``None``
    when the box covers none of it.
    """
    low, high = np.array(box[:2]), np.array(box[2:])
    axes = np.array(to_scene)
    run = 0.0
    first = last = None
    for curve in curves:
        for start, stop in zip(curve[:-1], curve[1:], strict=True):
            length = float(np.linalg.norm((stop - start) * axes))
            inside = _crossing(start, stop, low, high)
            if inside is not None and inside[1] > inside[0]:
                enter, leave = inside
                if first is None:
                    first = run + float(enter) * length
                last = run + float(leave) * length
            run += length
    if first is None or last is None:
        return None
    return first, run - last


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
    if _crossing(start, stop, low, high) is not None:
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


def _crossing(
    start: np.ndarray, stop: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[float, float] | None:
    """The shares of the step ``start``-``stop`` at which it enters and
    leaves the box ``low``-``high``; ``None`` where it misses it.
    """
    step = stop - start
    enter, leave = 0.0, 1.0
    for axis in (0, 1):
        if abs(step[axis]) < 1e-15:
            if not low[axis] <= start[axis] <= high[axis]:
                return None
            continue
        a = (low[axis] - start[axis]) / step[axis]
        b = (high[axis] - start[axis]) / step[axis]
        enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
    return (enter, leave) if enter <= leave else None


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


def beside_labels(record: Record, capture: Capture) -> dict[str, Beside]:
    """The labels of ``record``'s engine diagrams set beside their lines in
    ``capture``, by stable id, each with what it keeps off there: the
    diagram's other pieces and connectors, as ``capture`` holds them. A
    label is set beside its line once its connector has carried a flow (its
    ``flow_reach``). A diagram, connector or label the record holds no
    state of at ``capture`` adds nothing.
    """
    from arachne.mobject.diagram import Flow

    found: dict[str, Beside] = {}
    for inside, edges in _diagram_connectors(record, capture):
        labels = {id(label) for _, label, _ in edges}
        connectors = {id(edge) for edge, _, _ in edges}
        for edge, label, flowed in edges:
            if label is None or not flowed:
                continue
            others = [
                m
                for m in inside
                if m is not edge and id(m) not in labels and not isinstance(m, Flow)
            ]
            found[record.origin(label).id] = Beside(
                pieces=frozenset(
                    record.origin(m).id for m in others if id(m) not in connectors
                ),
                connectors=frozenset(
                    record.origin(m).id for m in others if id(m) in connectors
                ),
            )
    return found


def inline_labels(record: Record, capture: Capture) -> dict[str, str]:
    """The labels of ``record``'s engine diagrams on their plates at their
    connectors' middles in ``capture`` (each connector's that has carried
    no flow), by stable id, each with its connector's stable id.
    """
    return {
        record.origin(label).id: record.origin(edge).id
        for _, edges in _diagram_connectors(record, capture)
        for edge, label, flowed in edges
        if label is not None and not flowed
    }


def _diagram_connectors(
    record: Record, capture: Capture
) -> Iterator[tuple[tuple[Mobject, ...], list[tuple[Mobject, Any, bool]]]]:
    """Each engine diagram of ``record`` as ``capture`` holds it: its
    members, and each connector among them with its label (``None`` without
    one) and whether it has carried a flow (its ``flow_reach``).
    """
    from arachne.mobject.diagram import Diagram

    def held(mob: Mobject, name: str, default: Any) -> Any:
        state = record.node_state(mob, capture)
        return default if state is None else state.get(name, default)

    for diagram in record.mobjects():
        if not isinstance(diagram, Diagram):
            continue
        state = record.node_state(diagram, capture)
        inside = () if state is None else state.submobjects
        yield (
            inside,
            [
                (
                    edge,
                    held(edge, "label", None),
                    float(held(edge, "flow_reach", 0.0)) > 0,
                )
                for edge in held(diagram, "edges", ())
                if any(edge is m for m in inside)
            ],
        )


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
        try:
            return super().__call__(context)
        finally:
            _DIAGRAMS.reset(held)


#: The rule: the clearance of the labels inside every diagram drawn.
diagram_clear = DiagramRule(diagram_findings)
