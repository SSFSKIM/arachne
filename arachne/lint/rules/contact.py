"""The contact rule: two things on screen that touch, or two texts closer
than a declared gap.

``contact.held`` when neither is text, ``overlap.text`` when one is, and
``gap.text`` when two texts sit apart but closer than the scene's declared
``text_gap``. What is measured, and the contacts that are let pass, is in
:mod:`._pairs`; a text something is drawn over is the occlusion rule's.
Under a ThreeDCamera two things that are not text, one wholly in front of
the other in depth and painted after it, are layered, not in contact
(:mod:`._pairs` says what wholly in front means); a pair with a text in it
is measured on screen as in a flat picture.
"""

from __future__ import annotations

__all__ = ["CODES", "contact"]

from ..findings import Finding
from ..parts import Parts
from ._pairs import pair_findings
from ._rule import InstantRule, Terms

#: The codes this rule reports.
CODES = frozenset({"contact.held", "overlap.text", "gap.text"})


def contact_findings(parts: Parts, terms: Terms) -> list[Finding]:
    return [f for f in pair_findings(parts, terms) if f.code in CODES]


#: The rule: things that touch, and texts that crowd.
contact = InstantRule(contact_findings)
