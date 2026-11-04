"""``clip.frame_edge`` for text: glyphs past the picture's edge.

A text is held by its glyphs, without a plate of its own: its glyphs may
run :data:`EDGE_SLACK` past an edge (a descender on the last pixel row)
before it counts as cut, and a text wholly outside is the worse case, not
an exempt one. Axis furniture (tick numerals among it) may run past the
edge, as may every shape: only text is asked. The edge is the picture's,
in home-frame units, whatever the camera does. Only a bare exemption
excuses this; a pair names two objects, and the edge is neither.
"""

from __future__ import annotations

__all__ = ["EDGE_SLACK", "edge_findings", "frame_edge"]

from fractions import Fraction

from ..findings import Finding
from ..parts import Parts
from ._excuse import muted
from ._rule import InstantRule, Terms
from ._units import measured_units

#: How far a text's glyphs may run past an edge before they are cut.
EDGE_SLACK = 0.05
#: A shortfall under this is noise, not a defect: under a pixel at any size.
NOISE = 0.005


def edge_findings(
    parts: Parts, terms: Terms, *, margin: float = -EDGE_SLACK
) -> list[Finding]:
    """The texts at ``parts``' instant whose glyphs come within ``margin``
    of the picture's edge (with a negative margin, run past it by more than
    that), each one finding naming the edge it is furthest past.
    """
    left, bottom, right, top = parts.frame
    units_per_pixel = parts.view.home_height / parts.target[1]
    found = []
    for unit in measured_units(parts):
        if not unit.text or unit.furniture or muted(terms.exempt, unit):
            continue
        x0, y0, x1, y1 = unit.glyph_box
        beyond = {
            "left": left - x0,
            "right": x1 - right,
            "bottom": bottom - y0,
            "top": y1 - top,
        }
        crossed = {edge: past for edge, past in beyond.items() if past > -margin}
        if not crossed:
            continue
        edge = max(crossed, key=lambda e: crossed[e])
        past = crossed[edge]
        gone = x1 < left or x0 > right or y1 < bottom or y0 > top
        pixels = round(abs(past) / units_per_pixel)
        what = f"text {unit.object.label_or_text!r}"
        if past <= NOISE:
            message = (
                f"{what} sits {-past:.3f} units ({pixels} px) inside the {edge} edge,"
                f" nearer than the {margin:.2g} units text keeps clear of it."
            )
        elif gone:
            message = (
                f"{what} is outside the frame altogether, {past:.3f} units"
                f" ({pixels} px) past its {edge} edge: no viewer sees it."
            )
        else:
            message = (
                f"{what} is cut by the frame's {' and '.join(crossed)}"
                f" {'edge' if len(crossed) == 1 else 'edges'}: its glyphs run"
                f" {past:.3f} units ({pixels} px) past the {edge} edge."
            )
        found.append(
            Finding(
                "clip.frame_edge",
                Fraction(parts.t),
                Fraction(parts.t),
                (unit.object,),
                message=message,
                detail={"edge": edge, "past": round(past, 4)},
            )
        )
    return found


#: The rule: texts cut by the frame's edge, or outside it.
frame_edge = InstantRule(edge_findings)
