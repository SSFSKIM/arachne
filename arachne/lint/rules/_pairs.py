"""Every two units on screen at an instant, measured against each other.

What a pair is held to: :attr:`Terms.pad` (they must not touch), and two
texts the declared ``text_gap`` when there is one. A ring drawn around some
but not all of a text's glyphs is that text's own annotation, held only to
touching. A pair is measured as the two are drawn:

- two texts box against box (crowding is a whole-label question);
- a text against a filled shape, each glyph's box against each of the
  shape's leaves' boxes, so a ring between glyphs, or a word in the gap
  between two tiles of one group, reads clear;
- a text against a stroke, the stroke's points against each glyph's ink
  (its outline filled by nonzero winding, plus the stroke the glyph itself
  is drawn with), wherever a point comes within :data:`INK_NEAR` of the
  glyph's box; further off, the box decides;
- a stroke against a filled shape, its points against the shape's leaves'
  boxes; two strokes, point against point.

Pairs built to touch are let pass: two pieces of one top-level mobject;
an axis's line or ticks against anything but text; a point-sized marker
against a stroke, or another marker; a stroke touching only at its own
ends (unless the other is text or axis furniture); two strokes drawn along
each other, or a stroke along a filled shape's own outline.

A contact that overlaps (a stroke whose centre line enters a glyph's ink,
or anything else past touching) between a text and something drawn after
its glyphs is ``occlusion.glyph``: a scene's pair does not excuse it, a
bare exemption does. Any other contact is ``overlap.text`` when a text is
in it (``gap.text`` when two texts merely sit closer than the declared gap)
and ``contact.held`` when none is.

Where the frame shows the text clean, a contact under it is let pass, as
ManimGX's diagnostics do (credited in ``NOTICE.md``): a text over a grid's
member drawn under it, and a text over a stroke an opaque filled shape
hides wherever the two meet (a card under the text, or the text's own
plate). In a frame the scene moves on from (``Terms.mid_scene``) a text
laid on a filled shape, or over a stroke inside a halo of its own wide and
opaque enough to mask it, reads clean too.
"""

from __future__ import annotations

__all__ = ["pair_findings"]

import itertools
import math
from fractions import Fraction

import numpy as np

from ..findings import Finding
from ..parts import Box, Leaf, Parts
from ._excuse import muted, paired
from ._geometry import (
    SAMPLE_STEP,
    box_gap,
    closest,
    least_box_gap,
    nearest,
    outline_edges,
    signed_to_box,
    signed_to_outline,
    within,
)
from ._rule import Terms
from ._units import Measured, measured_units

#: A shortfall under this is noise: under a pixel at any size.
NOISE = 0.005
#: A stroke point this near a glyph's box is held to the glyph's ink.
INK_NEAR = 0.3
#: How far from a stroke's ends a contact may lie and be its landing.
END_REACH = 0.25
#: The share of one stroke on another that makes it an overlay.
ALONG = 0.6
#: Points each stroke is thinned to when two strokes are compared along.
_ALONG_POINTS = 400
#: A background stroke this wide (before the camera scales it) and this
#: opaque on every glyph masks a stroke under the text.
HALO_WIDTH = 3.0
HALO_OPACITY = 0.9
#: A filled shape this opaque hides a stroke drawn under it.
HIDING_OPACITY = 0.9


def pair_findings(parts: Parts, terms: Terms) -> list[Finding]:
    """The contacts and covered texts among the units on screen at
    ``parts``' instant.
    """
    units = [u for u in measured_units(parts) if u.on(parts.frame)]
    found: list[Finding] = []
    for a, b in itertools.combinations(units, 2):
        finding = _pair(parts, units, terms, a, b)
        if finding is not None:
            found.append(finding)
    return found


def _pair(
    parts: Parts, units: list[Measured], terms: Terms, a: Measured, b: Measured
) -> Finding | None:
    required = terms.pad
    if terms.text_gap is not None and a.text and b.text:
        required = max(required, terms.text_gap)
    if _annotates(a, b) or _annotates(b, a):
        required = NOISE
    if box_gap(a.outer, b.outer) >= required or _built_to_touch(a, b):
        return None
    bare = muted(terms.exempt, a) or muted(terms.exempt, b)
    excused = bare or paired(terms.exempt, a, b)
    if excused and (a.text == b.text or bare):
        return None
    measured = _measure(a, b)
    if required - measured < NOISE:
        return None
    if _lands(a, b, required) or _along(a, b, required):
        return None
    stroke = _inked_stroke(a, b)
    if stroke is None:
        depth = -measured
        overlaps = depth > NOISE
    else:
        # Held to glyph ink from its centre line, a stroke covers a glyph once
        # that line enters the ink; its paint then lies half its width deeper.
        depth = stroke.reach - measured
        overlaps = measured < 0
    covered = _covering(a, b) if overlaps else None
    if covered is not None:
        text, over = covered
        return _finding(
            parts,
            "occlusion.glyph",
            (text, over),
            f"text {text.object.label_or_text!r} is drawn under"
            f" {_called(over)}, which lies over its glyphs by {depth:.3f} units.",
            {"covered": round(depth, 4)},
        )
    if excused or _reads_clean(parts, units, terms, a, b, required):
        return None
    shown = f"{_called(a)} and {_called(b)}"
    if not (a.text or b.text):
        return _finding(
            parts,
            "contact.held",
            (a, b),
            f"{shown} touch: {measured:.3f} units apart, against {required:.3f}.",
            {"gap": round(measured, 4), "required": round(required, 4)},
        )
    if a.text and b.text and measured > 0:
        return _finding(
            parts,
            "gap.text",
            (a, b),
            f"{shown} sit {measured:.3f} units apart, closer than the"
            f" {required:.3f} declared for texts.",
            {"gap": round(measured, 4), "required": round(required, 4)},
        )
    return _finding(
        parts,
        "overlap.text",
        (a, b),
        f"{shown} touch: {measured:.3f} units apart, against {required:.3f}.",
        {"overlap": round(-measured, 4)},
    )


def _finding(
    parts: Parts,
    code: str,
    units: tuple[Measured, Measured],
    message: str,
    detail: dict[str, float],
) -> Finding:
    t = Fraction(parts.t)
    return Finding(
        code, t, t, tuple(u.object for u in units), message=message, detail=detail
    )


def _called(unit: Measured) -> str:
    said = unit.object.label_or_text
    return f"{unit.unit.type} {said!r}" if said else unit.unit.type


# Measuring ---------------------------------------------------------------


def _measure(a: Measured, b: Measured) -> float:
    """How far apart ``a`` and ``b`` are; at or below zero, touching."""
    if a.region and b.region:
        if a.text and b.text:
            return box_gap(a.outer, b.outer)
        return least_box_gap(a.boxes, b.boxes)
    for region, path in ((a, b), (b, a)):
        if region.region:
            if region.text:
                return _from_glyphs(path.points, region)
            return min(float(signed_to_box(path.points, g).min()) for g in region.boxes)
    return closest(a.points, b.points)


def _from_glyphs(points: np.ndarray, text: Measured) -> float:
    """How far a stroke's points are from a text's glyph ink.

    A glyph's box, less the stroke the glyph is drawn with, is never
    further than its ink, so glyphs are taken nearest box first, and only
    points within :data:`INK_NEAR` of a box are traced to its ink.
    """
    if not len(points):
        return math.inf
    leaves = text.leaves
    reach = [leaf.line_width / 2 for leaf in leaves]
    to_box = [
        signed_to_box(points, leaf.screen_box) - r
        for leaf, r in zip(leaves, reach, strict=True)
    ]
    best = math.inf
    for index in sorted(range(len(leaves)), key=lambda i: float(to_box[i].min())):
        here = to_box[index]
        if float(here.min()) >= best:
            break
        edges = outline_edges(leaves[index])
        near = here < INK_NEAR
        if edges is not None and near.any():
            here = here.copy()
            here[near] = signed_to_outline(points[near], edges) - reach[index]
        best = min(best, float(here.min()))
    return best


def _annotates(path: Measured, text: Measured) -> bool:
    """A ring drawn around some, not all, of a text's glyphs."""
    if path.region or not text.text or not path.ring:
        return False
    inside = sum(within(leaf.screen_box, path.outer) for leaf in text.leaves)
    return 0 < inside < len(text.leaves)


def _built_to_touch(a: Measured, b: Measured) -> bool:
    if a.id == b.id or a.root == b.root:
        return True
    if a.marker and b.marker:
        return True
    for x, y in ((a, b), (b, a)):
        if x.structural and not y.text:
            return True
        if x.marker and not y.region and not y.furniture:
            return True
    return False


def _lands(a: Measured, b: Measured, required: float) -> bool:
    """Every contact lies within :data:`END_REACH` of one stroke's ends."""
    for x, y in ((a, b), (b, a)):
        if x.region or y.text or y.furniture:
            continue
        gaps = (
            signed_to_box(x.points, y.outer)
            if y.region
            else nearest(x.points, y.points)
        )
        hits = x.points[gaps < required]
        if len(hits) and float(nearest(hits, x.ends).max()) <= END_REACH:
            return True
    return False


def _along(a: Measured, b: Measured, required: float) -> bool:
    """Two strokes drawn along each other, or a stroke along a filled
    shape's own outline: an overlay.
    """
    if a.furniture or b.furniture:
        return False
    if a.region or b.region:
        return _rides_outline(a, b, required)
    if not len(a.points) or not len(b.points):
        return False
    thin_a = a.points[:: max(1, len(a.points) // _ALONG_POINTS)]
    thin_b = b.points[:: max(1, len(b.points) // _ALONG_POINTS)]
    share = max(
        float((nearest(thin_a, b.points) < required).mean()),
        float((nearest(thin_b, a.points) < required).mean()),
    )
    return share >= ALONG


def _rides_outline(a: Measured, b: Measured, required: float) -> bool:
    for path, region in ((a, b), (b, a)):
        if path.region or not region.region or region.text:
            continue
        short = path.points[signed_to_box(path.points, region.outer) < required]
        if not len(short):
            continue
        # The outline is sampled a step apart: a point on it may be that far
        # from the nearest sample.
        if float(nearest(short, region.points).max()) <= required + SAMPLE_STEP:
            return True
    return False


# Paint order -------------------------------------------------------------


def _inked_stroke(a: Measured, b: Measured) -> Measured | None:
    """The stroke of a stroke-and-text pair, which is held to glyph ink."""
    for path, text in ((a, b), (b, a)):
        if not path.region and text.text:
            return path
    return None


def _covering(a: Measured, b: Measured) -> tuple[Measured, Measured] | None:
    """``(text, what covers it)`` when a non-text is drawn after the
    text's glyphs.
    """
    for text, other in ((a, b), (b, a)):
        if text.text and not other.text and other.top > text.glyph_top:
            return text, other
    return None


# Contacts the frame shows clean --------------------------------------------


def _reads_clean(
    parts: Parts,
    units: list[Measured],
    terms: Terms,
    a: Measured,
    b: Measured,
    required: float,
) -> bool:
    if a.text == b.text:
        return False
    text, other = (a, b) if a.text else (b, a)
    under = other.top < text.glyph_top
    if other.grid and under:
        return True
    if not other.region and under and _hidden(units, text, other, required):
        return True
    if not terms.mid_scene:
        return False
    if other.region:
        return True
    return under and _haloed(parts, text)


def _haloed(parts: Parts, text: Measured) -> bool:
    scale = (
        parts.view.cairo_line_width_multiple
        * parts.view.home_width
        / parts.view.frame_width
    )
    return all(
        leaf.halo_width / scale >= HALO_WIDTH - 1e-9
        and leaf.halo_opacity >= HALO_OPACITY
        for leaf in text.glyphs
    )


def _hidden(
    units: list[Measured], text: Measured, path: Measured, required: float
) -> bool:
    """Whether an opaque filled shape drawn between ``path`` and the text's
    glyphs hides every point where the path meets them.
    """
    glyphs = [leaf.screen_box for leaf in text.glyphs or text.leaves]
    near = np.min([signed_to_box(path.points, box) for box in glyphs], axis=0)
    hits = path.points[near < required]
    if not len(hits):
        return False
    plates: list[tuple[Leaf, ...]] = []
    own = tuple(
        leaf
        for leaf in text.plate
        if path.top < leaf.order < text.glyph_top
        and leaf.fill_opacity >= HIDING_OPACITY
    )
    if own:
        plates.append(own)
    for unit in units:
        if not unit.region or unit.text or not path.top < unit.top < text.glyph_top:
            continue
        if all(leaf.fill_opacity >= HIDING_OPACITY for leaf in unit.leaves):
            plates.append(unit.leaves)
    return any(_covers(plate, hits) for plate in plates)


def _covers(plate: tuple[Leaf, ...], hits: np.ndarray) -> bool:
    covered = np.zeros(len(hits), dtype=bool)
    for leaf in plate:
        todo = ~covered & _in_box(hits, leaf.screen_box)
        if not todo.any():
            continue
        edges = outline_edges(leaf)
        if edges is not None:
            covered[todo] = signed_to_outline(hits[todo], edges) < 0
    return bool(covered.all())


def _in_box(points: np.ndarray, box: Box) -> np.ndarray:
    return (
        (points[:, 0] >= box[0])
        & (points[:, 0] <= box[2])
        & (points[:, 1] >= box[1])
        & (points[:, 1] <= box[3])
    )
