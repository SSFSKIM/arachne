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

A scene with no diagram has no finding here, whatever else it draws.
"""

from __future__ import annotations

__all__ = ["diagram_clear", "diagram_findings"]

import itertools
from fractions import Fraction

from ..findings import Finding, Holder
from ..parts import Parts
from ._excuse import muted, paired
from ._geometry import box_gap
from ._rule import InstantRule, Terms
from ._units import Measured, measured_units
from .frame_edge import EDGE_SLACK
from .text_small import SMALL_PIXELS, small_findings, text_heights

#: A shortfall under this is noise: under a pixel at any size.
NOISE = 0.005


def diagram_findings(parts: Parts, terms: Terms) -> list[Finding]:
    """The labels of the diagrams drawn at ``parts``' instant that crowd
    each other, run past the frame's edge or are too small to read.
    """
    kinds = _diagram_types()
    if not any(_held(part.within, kinds) for part in parts.texts):
        return []
    labels = [
        unit
        for unit in measured_units(parts)
        if unit.text
        and _held(unit.unit.within, kinds)
        and not muted(terms.exempt, unit)
    ]
    found = [*_edge_findings(parts, labels)]
    on = [unit for unit in labels if unit.on(parts.frame)]
    for a, b in itertools.combinations(on, 2):
        finding = _pair(parts, terms, a, b)
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


def _pair(parts: Parts, terms: Terms, a: Measured, b: Measured) -> Finding | None:
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
        message=f"{shown} of one diagram overlap by {-gap:.3f} units.",
        detail={"overlap": round(-gap, 4)},
    )


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


def _held(within: tuple[Holder, ...], kinds: frozenset[str]) -> bool:
    return any(holder.type in kinds for holder in within)


def _diagram_types() -> frozenset[str]:
    """The names of the diagram class and every class made from it."""
    from arachne.mobject.diagram import Diagram

    names = set()
    stack: list[type] = [Diagram]
    while stack:
        cls = stack.pop()
        names.add(cls.__name__)
        stack.extend(cls.__subclasses__())
    return frozenset(names)


#: The rule: the clearance of the labels inside every diagram drawn.
diagram_clear = InstantRule(diagram_findings)
