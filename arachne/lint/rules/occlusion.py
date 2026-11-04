"""The occlusion rule: ``occlusion.glyph``, a text something else is drawn
over.

The order is the order the frame is drawn in. A thing drawn after a
text's glyphs (its own plate aside) and overlapping them covers them: a
stroke once its centre line enters a glyph's ink, anything else once it
is past touching. The text drawn over a shape is the harmless direction
and is a contact, if anything. A pair the scene declares does not excuse
an occlusion, because the order shows that nothing under the text keeps
it clear; a bare exemption, a layer meant to cover text, does.
"""

from __future__ import annotations

__all__ = ["occlusion"]

from ..findings import Finding
from ..parts import Parts
from ._pairs import pair_findings
from ._rule import InstantRule, Terms


def occlusion_findings(parts: Parts, terms: Terms) -> list[Finding]:
    return [f for f in pair_findings(parts, terms) if f.code == "occlusion.glyph"]


#: The rule: texts drawn over.
occlusion = InstantRule(occlusion_findings)
