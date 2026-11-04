"""``text.small``: a text shorter than :data:`SMALL_PIXELS` on a picture
:data:`~arachne.lint.codes.THRESHOLD_HEIGHT` pixels tall, a note.

It is reported and never fails a lint. A text is measured by its ink, on
the picture's shorter side, as ManimGX's diagnostics measure it (credited
in ``NOTICE.md``); a text wholly outside the frame is not seen, so not
measured.
"""

from __future__ import annotations

__all__ = ["SMALL_PIXELS", "small_findings", "text_heights", "text_small"]

from fractions import Fraction

from ..codes import THRESHOLD_HEIGHT
from ..exempt import Exemptions
from ..findings import Finding, FindingObject
from ..parts import Part, Parts
from ._geometry import meets
from ._rule import InstantRule, Terms

#: The least height, in pixels at the threshold height, a text reads at.
SMALL_PIXELS = 20


def text_heights(parts: Parts) -> list[tuple[Part, float, float]]:
    """Each text seen at ``parts``' instant (its ink meets the frame), with
    its ink's height in home-frame units and in pixels of a picture
    :data:`~arachne.lint.codes.THRESHOLD_HEIGHT` pixels on its shorter side.
    """
    width, height = parts.target
    pixels_per_unit = (
        THRESHOLD_HEIGHT * height / min(width, height) / parts.view.home_height
    )
    found = []
    for part in parts.texts:
        if not meets(part.ink_box, parts.frame):
            continue
        units = part.ink_box[3] - part.ink_box[1]
        found.append((part, units, units * pixels_per_unit))
    return found


def small_findings(parts: Parts, terms: Terms) -> list[Finding]:
    """The texts at ``parts``' instant shorter than :data:`SMALL_PIXELS`."""
    found = []
    for part, _, tall in text_heights(parts):
        if tall >= SMALL_PIXELS:
            continue
        obj = part.as_object(parts)
        if _muted(terms.exempt, obj):
            continue
        found.append(
            Finding(
                "text.small",
                Fraction(parts.t),
                Fraction(parts.t),
                (obj,),
                message=(
                    f"text {part.text!r} is {tall:.0f} px tall at {THRESHOLD_HEIGHT}"
                    f" px, under the {SMALL_PIXELS} px it reads at."
                ),
                detail={"height_px": round(tall, 1)},
            )
        )
    return found


def _muted(exempt: Exemptions, obj: FindingObject) -> bool:
    return any(selector.selects(obj) for selector in exempt.bare)


#: The rule: texts too small to read, as notes.
text_small = InstantRule(small_findings)
