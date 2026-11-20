"""The instant rules: what a sweep of one frame finds.

Each but :data:`anchored` is an :class:`InstantRule`, which
:func:`arachne.lint.lint` runs only when it is passed one, at the times it is asked for and under the terms a
scene declares::

    lint(session, rules=(sweep.where(at=[t], text_gap=0.3),))

- :data:`frame_edge`: ``clip.frame_edge``, text cut by the picture's edge;
- :data:`contact`: ``contact.held``, ``overlap.text`` and ``gap.text``;
- :data:`occlusion`: ``occlusion.glyph``, text something is drawn over;
- :data:`sweep`: the three at once, one sweep of the frame;
- :data:`text_small`: ``text.small``, a note;
- :data:`anchored`: ``annotation.detached``, an annotation pointing more
  than a reference pixel from its anchor, read at its own instants;
- :data:`diagram_clear`: ``overlap.text``, ``gap.text``,
  ``clip.frame_edge`` and ``text.small`` for the labels inside a diagram,
  whose pieces the pair rules let touch.

They read the parts the frame draws (:mod:`arachne.lint.parts`), in
home-frame units, the camera's projection applied, and a pair's two sides
as the units a sweep measures whole (:class:`~arachne.lint.parts.Unit`).
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_PAD",
    "InstantRule",
    "Terms",
    "anchored",
    "contact",
    "diagram_clear",
    "frame_edge",
    "occlusion",
    "sweep",
    "sweep_findings",
    "text_small",
]

from ..findings import Finding
from ..parts import Parts
from ._pairs import pair_findings
from ._rule import DEFAULT_PAD, InstantRule, Terms
from .anchored import anchored
from .contact import contact
from .diagram_clear import diagram_clear
from .frame_edge import edge_findings, frame_edge
from .occlusion import occlusion
from .text_small import text_small


def sweep_findings(parts: Parts, terms: Terms) -> list[Finding]:
    """One sweep of the frame at ``parts``' instant: texts at its edge, and
    every pair on it.
    """
    return [*edge_findings(parts, terms), *pair_findings(parts, terms)]


#: The rule: one sweep of the frame, edge and pairs.
sweep = InstantRule(sweep_findings)
