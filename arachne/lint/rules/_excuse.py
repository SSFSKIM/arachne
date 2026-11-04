"""The scene's exemptions, as a sweep reads them.

A selector names a unit when it selects the unit's object (the unit, or a
group holding it, answers to it) or, by id, one of the unit's own leaves. A
bare selector mutes the units it names against everything, the frame edge
and paint order included. A pair says one contact between the two units it
names is decided: it excuses their geometry and nothing else, so a shape a
pair names that paints over the text it names is still reported, and no
pair speaks for the frame edge, which is no unit.
"""

from __future__ import annotations

from typing import Any

from ..exempt import Exemptions
from ._units import Measured


def _names(selector: Any, unit: Measured) -> bool:
    if selector.selects(unit.object):
        return True
    return bool(selector.by == "id" and selector.value in unit.leaf_ids)


def muted(exempt: Exemptions, unit: Measured) -> bool:
    """Whether a bare selector names ``unit``."""
    return any(_names(s, unit) for s in exempt.bare)


def paired(exempt: Exemptions, a: Measured, b: Measured) -> bool:
    """Whether a pair names ``a`` and ``b``, in either order."""
    return any(
        (_names(x, a) and _names(y, b)) or (_names(x, b) and _names(y, a))
        for x, y in exempt.pairs
    )
