"""Findings: what a lint rule reports, as data.

A :class:`Finding` has a code from the table (:mod:`.codes`), the time range
it holds over, the objects it is about, a message, the line of the scene
that was running when it was found, and the code's severity, class and fix.
A rule may lower a finding to a ``note`` (something likely meant, such as a
line running off the frame), never raise it. ``held`` keeps the meaning the
report gives it: found by the hold watch, on screen at least a second.

:func:`fold` folds look-alikes: more than three findings in one event of
the timeline (a play or wait) of one code and severity, about objects the
same lines of the scene made (a row of squares one line made over a row
another made, whatever each square is named), keep one, which carries the
rest and holds over all their times; the findings between them stay. A
finding about no object is told by the line that raised it; an object made
by no line the record knows, by its kind, its outermost group and its own
name. A code the table marks as never folding (each finding an event at its
own time) keeps every one. The idea is ManimGX's report, credited in
``NOTICE.md``.
"""

from __future__ import annotations

__all__ = ["Finding", "FindingObject", "Holder", "fold", "in_range"]

import bisect
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from arachne.timeline.clock import exact
from arachne.timeline.source import SourceLine

from .codes import CODES, code

#: ``(x_min, y_min, x_max, y_max)``.
Box = tuple[float, float, float, float]

#: How many look-alikes in one event are all kept; more fold.
SIMILAR = 3


@dataclass(frozen=True)
class Holder:
    """A group holding an object: its stable id, the scene's name for it,
    the scene line that made it and its type.
    """

    id: str
    name: str | None
    line: int | None
    type: str = ""


@dataclass(frozen=True)
class FindingObject:
    """An object a finding is about.

    ``source_line`` is the line of the scene file that made it; ``bbox`` its
    box in scene units; ``screen_bbox`` its box in pixels of the declared
    target, from the top-left corner with y down (``None`` where it can't be
    projected); ``within`` the groups holding it, outermost first.
    """

    id: str
    source_line: int | None
    kind: str
    label_or_text: str | None
    bbox: Box | None
    screen_bbox: Box | None
    name: str | None = None
    within: tuple[Holder, ...] = ()

    @property
    def group(self) -> tuple[str | None, str | None]:
        """What tells it from look-alikes: the outermost group holding it
        and the scene's name for it, either ``None`` when it has none.
        """
        return (self.within[0].id if self.within else None, self.name)


@dataclass(frozen=True)
class Finding:
    """One finding. ``severity`` and ``fix`` default to the code's; ``cls``
    is the code's class. ``folded`` holds the look-alikes folded into it.
    """

    code: str
    t_start: Fraction | None
    t_end: Fraction | None
    objects: tuple[FindingObject, ...] = ()
    message: str = ""
    line: SourceLine | None = None
    held: bool = False
    severity: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)
    fix: str = ""
    folded: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        entry = code(self.code)
        if not self.severity:
            object.__setattr__(self, "severity", entry.severity)
        elif self.severity not in (entry.severity, "note"):
            raise ValueError(
                f"A {self.code} finding is a {entry.severity}, or a note; not a"
                f" {self.severity}."
            )
        if not self.fix:
            object.__setattr__(self, "fix", entry.fix)
        unknown = set(self.detail) - set(entry.detail)
        if unknown:
            raise ValueError(
                f"A {self.code} finding carries no {', '.join(sorted(unknown))}"
                f" (its detail fields: {', '.join(entry.detail) or 'none'})."
            )
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))
        object.__setattr__(self, "objects", tuple(self.objects))

    @property
    def cls(self) -> str:
        """The class a report names the finding's kind by."""
        return CODES[self.code].cls


def in_range(finding: Finding, t_range: Sequence[float] | None) -> bool:
    """Whether ``finding`` holds at some time in ``t_range`` (inclusive);
    a finding of no time is in every range.
    """
    if t_range is None or finding.t_start is None:
        return True
    low, high = (exact(value) for value in t_range)
    end = finding.t_end if finding.t_end is not None else finding.t_start
    return finding.t_start <= high and end >= low


def fold(
    findings: Iterable[Finding],
    similar: int = SIMILAR,
    timeline: Sequence[Fraction] = (),
) -> list[Finding]:
    """``findings`` with each group of more than ``similar`` look-alikes in
    one event folded into its first, where it stood: the first's objects and
    line, the earliest start and latest end of them all, and the rest in
    ``folded``. ``timeline`` holds the starts of the events (the plays, in
    order); a finding is in the last that starts at or before it, a finding
    of no time in none of them; with no timeline every finding is in one.
    """
    findings = list(findings)
    groups: dict[Hashable, list[int]] = {}
    for index, finding in enumerate(findings):
        if CODES[finding.code].folds:
            groups.setdefault(_alike(finding, timeline), []).append(index)
    first: dict[int, list[Finding]] = {}
    folded: set[int] = set()
    for members in groups.values():
        if len(members) > similar:
            first[members[0]] = [findings[i] for i in members]
            folded.update(members[1:])
    kept: list[Finding] = []
    for index, finding in enumerate(findings):
        if index in first:
            kept.append(_folded(first[index]))
        elif index not in folded:
            kept.append(finding)
    return kept


def _folded(members: list[Finding]) -> Finding:
    """The first of ``members`` carrying the rest, over all their times."""
    head, rest = members[0], members[1:]
    spans = [
        (m.t_start, m.t_end if m.t_end is not None else m.t_start)
        for m in members
        if m.t_start is not None
    ]
    if not spans:
        return replace(head, folded=(*head.folded, *rest))
    start = min(begin for begin, _ in spans)
    end = max(finish for _, finish in spans)
    return replace(head, t_start=start, t_end=end, folded=(*head.folded, *rest))


def _alike(finding: Finding, timeline: Sequence[Fraction]) -> Hashable:
    """What a finding's look-alikes share: its code and severity, the event
    it is in, and what it is about: the lines of the scene that made its
    objects, sorted; an object made by no known line, by its kind, its
    outermost group and its own name; and for a finding about no object, the
    line that raised it (one raised at no line of the scene is its own).
    """
    event = (
        None
        if finding.t_start is None
        else bisect.bisect_right(timeline, finding.t_start)
    )
    if finding.objects:
        about: Hashable = tuple(
            sorted(
                (
                    ("line", str(o.source_line))
                    if o.source_line is not None
                    else ("made", o.kind, str(o.group))
                )
                for o in finding.objects
            )
        )
    elif finding.line is not None:
        about = ("raised", str(finding.line))
    else:
        about = ("alone", id(finding))
    return (finding.code, finding.severity, event, about)
