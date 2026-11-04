"""The caller's exemptions: object selectors, and pairs of them.

A selector names objects by their stable id (``path:line#n``), by the line
of the scene file that made them (``line:N``) or by the scene's name for
them (``name:<variable>``). It selects an object when the object, or a
group holding it, answers to it. A mobject is a selector too: it stands
for the stable ids of its family as it is when the exemption is read, so
it names what it holds whether the scene draws it or not. A bare selector exempts every finding
about an object it selects; a pair (two selectors) exempts a finding about
exactly two objects, one selected by each. A finding about no object is
never exempted by a selector.
"""

from __future__ import annotations

__all__ = ["Exemptions"]

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .findings import Finding, FindingObject


@dataclass(frozen=True)
class _Selector:
    by: str  # "id", "line", "name", or "family": a mobject's family's ids
    value: Any

    @classmethod
    def parse(cls, text: Any) -> _Selector:
        from arachne.mobject.mobject import Mobject
        from arachne.timeline.origins import origin_of

        if isinstance(text, Mobject):
            # What the mobject holds when the exemption is read, drawn or not.
            family = frozenset(origin_of(m).id for m in text.get_family())
            return cls("family", family)
        if not isinstance(text, str) or not text:
            raise ValueError(
                "An exemption selector is a non-empty string or a mobject,"
                f" not {text!r}."
            )
        if text.startswith("line:"):
            number = text[len("line:") :]
            if not number.isdigit():
                raise ValueError(f"{text!r}: a line selector is line:<number>.")
            return cls("line", int(number))
        if text.startswith("name:"):
            name = text[len("name:") :]
            if not name:
                raise ValueError(f"{text!r}: a name selector is name:<variable>.")
            return cls("name", name)
        return cls("id", text)

    def selects(self, obj: FindingObject) -> bool:
        if self.by == "family":
            return obj.id in self.value or any(h.id in self.value for h in obj.within)
        holders = [(obj.id, obj.name, obj.source_line)] + [
            (holder.id, holder.name, holder.line) for holder in obj.within
        ]
        index = {"id": 0, "name": 1, "line": 2}[self.by]
        return any(entry[index] == self.value for entry in holders)


@dataclass(frozen=True)
class Exemptions:
    """Parsed exemptions: bare selectors and pairs."""

    bare: tuple[_Selector, ...] = ()
    pairs: tuple[tuple[_Selector, _Selector], ...] = ()

    @classmethod
    def parse(cls, exempt: Iterable[Any]) -> Exemptions:
        """Exemptions from ``exempt``: each a selector string or a list of
        two; raises :class:`ValueError` for anything else.
        """
        bare: list[_Selector] = []
        pairs: list[tuple[_Selector, _Selector]] = []
        for entry in exempt:
            if isinstance(entry, (list, tuple)):
                if len(entry) != 2:
                    raise ValueError(
                        f"An exempted pair is two selectors, not {entry!r}."
                    )
                first, second = (_Selector.parse(item) for item in entry)
                pairs.append((first, second))
            else:
                bare.append(_Selector.parse(entry))
        return cls(tuple(bare), tuple(pairs))

    def exempts(self, finding: Finding) -> bool:
        """Whether a selector or pair leaves ``finding`` out."""
        objects = finding.objects
        if any(s.selects(o) for s in self.bare for o in objects):
            return True
        if len(objects) != 2:
            return False
        a, b = objects
        return any(
            (x.selects(a) and y.selects(b)) or (x.selects(b) and y.selects(a))
            for x, y in self.pairs
        )
