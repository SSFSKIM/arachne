"""The scene's exemptions, as a sweep reads them.

A selector names a unit when it answers to a mobject the unit is made from
(anything in its family; for axis furniture, its axes' whole family) or to
a group holding it; an exempted mobject names every unit its own family
shares a member with, so a group the scene never draws names what it
holds. A bare selector mutes the units it names against
everything, the frame edge and paint order included. A pair says one
contact between the two units it names is decided: it excuses their
geometry and nothing else, so a shape a pair names that paints over the
text it names is still reported, and no pair speaks for the frame edge,
which is no unit.
"""

from __future__ import annotations

from typing import Any

from ..exempt import Exemptions
from ..findings import Holder
from ._units import Measured


def _answers(selector: Any, holder: Holder) -> bool:
    if selector.by == "family":
        # An exempted mobject: its family as it was when it was declared.
        return holder.id in selector.value
    value = {"id": holder.id, "name": holder.name, "line": holder.line}[selector.by]
    return bool(value == selector.value)


def _names(selector: Any, unit: Measured) -> bool:
    return selector.selects(unit.object) or any(
        _answers(selector, member) for member in unit.unit.family
    )


def muted(exempt: Exemptions, unit: Measured) -> bool:
    """Whether a bare selector names ``unit``."""
    return any(_names(s, unit) for s in exempt.bare)


def paired(exempt: Exemptions, a: Measured, b: Measured) -> bool:
    """Whether a pair names ``a`` and ``b``, in either order."""
    return any(
        (_names(x, a) and _names(y, b)) or (_names(x, b) and _names(y, a))
        for x, y in exempt.pairs
    )
