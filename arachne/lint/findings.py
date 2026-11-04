"""Findings: what a lint rule reports, as data.

A :class:`Finding` has a code from the table (:mod:`.codes`), the time range
it holds over, the objects it is about, a message, the line of the scene
that was running when it was found, and the code's severity, class and fix.
A rule may lower a finding to a ``note`` (something likely meant, such as a
line running off the frame), never raise it. ``held`` keeps the meaning the
report gives it: found by the hold watch, on screen at least a second.

:func:`fold` folds look-alikes: more than three findings in a row of one
code about the same kinds of objects in the same groups, or unnamed (a
label on each of five planets) keep the first, which carries the rest. The idea is
ManimGX's report, credited in ``NOTICE.md``.
"""

from __future__ import annotations

__all__ = ["Finding", "FindingObject", "Holder", "fold", "in_range"]

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from arachne.timeline.clock import exact
from arachne.timeline.source import SourceLine

from .codes import CODES, code

#: ``(x_min, y_min, x_max, y_max)``.
Box = tuple[float, float, float, float]

#: How many look-alikes in a row are all kept; a longer run folds.
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
    def group(self) -> tuple[str, str] | None:
        """What it belongs to, for telling look-alikes: the outermost group
        holding it, else the scene's name for it, else nothing.
        """
        if self.within:
            return ("group", self.within[0].id)
        if self.name is not None:
            return ("name", self.name)
        return None


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


def fold(findings: Iterable[Finding], similar: int = SIMILAR) -> list[Finding]:
    """``findings`` with each run of more than ``similar`` look-alikes in a
    row folded into its first, which carries the rest in ``folded``.
    """
    runs: list[list[Finding]] = []
    for finding in findings:
        if runs and _alike(runs[-1][-1], finding):
            runs[-1].append(finding)
        else:
            runs.append([finding])
    kept: list[Finding] = []
    for run in runs:
        if len(run) > similar:
            kept.append(replace(run[0], folded=(*run[0].folded, *run[1:])))
        else:
            kept.extend(run)
    return kept


def _alike(a: Finding, b: Finding) -> bool:
    """Findings of one code and severity about objects of the same kinds,
    each held by the same outermost group, or named alike, or by none and
    unnamed.
    """

    def shape(found: Finding) -> tuple[Any, ...]:
        return (
            found.code,
            found.severity,
            tuple((o.kind, o.group) for o in found.objects),
        )

    return shape(a) == shape(b)
