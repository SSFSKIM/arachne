"""Findings and the verdict, with the verdict's canonical JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction

from .model import Box, Mode

PRESENCE_MISSING = "gate.presence.missing"
PRESENCE_EXTRA = "gate.presence.extra"
CLUSTER = "gate.cluster"
FLICKER = "gate.flicker"
EDGE_OVER_BUDGET = "gate.edge.over_budget"

FIXES = {
    PRESENCE_MISSING: "Draw the object in the candidate: the reference shows it and the candidate draws nothing there.",
    PRESENCE_EXTRA: "Remove the object from the candidate: it draws where the reference shows nothing of it.",
    CLUSTER: "Find what draws differently inside the box: the difference is large and strong enough to see.",
    FLICKER: "Find what repaints the box between these two frames: no object drawn there changed.",
    EDGE_OVER_BUDGET: "Check how the object's edges are rasterized: more of them differ than the budget allows.",
}

COUNTS = (
    "changed_pixels",
    "groups_below",
    "largest_below",
    "edge_pixels",
    "edge_changed",
    "edge_strong",
    "presence_untested",
    "flicker_unjudged",
    "undecided_objects",
    "objects_unexamined",
)


@dataclass(frozen=True)
class Finding:
    """One visible-scale difference: a stable code, where and when, and how to fix it."""

    code: str
    frame: int
    t: Fraction
    box: Box
    object: str | None = None
    source_line: str | None = None
    detail: Mapping[str, int | float] = field(default_factory=dict)

    @property
    def fix(self) -> str:
        return FIXES[self.code]

    def sort_key(self) -> tuple:
        return (self.frame, self.code, self.box, self.object or "")

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "frame": self.frame,
            "t": str(Fraction(self.t)),
            "box": list(self.box),
            "object": self.object,
            "source_line": self.source_line,
            "fix": self.fix,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class Verdict:
    """One case's verdict: ``block`` when it has a finding, else ``pass``.

    ``counts`` hold everything below the thresholds, which never blocks; ``edges``
    hold each object's edge pixels over the sampled frames (two-renderer mode only).
    """

    mode: Mode
    thresholds: Mapping
    frames: int
    sampled: tuple[int, ...]
    findings: tuple[Finding, ...]
    counts: Mapping[str, int]
    edges: Mapping[str, Mapping[str, int]]

    @property
    def verdict(self) -> str:
        return "block" if self.findings else "pass"

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(f.code for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "mode": self.mode,
            "thresholds": dict(self.thresholds),
            "frames": self.frames,
            "sampled": list(self.sampled),
            "findings": [f.to_dict() for f in self.findings],
            "counts": {name: int(self.counts[name]) for name in COUNTS},
            "edges": {name: dict(value) for name, value in self.edges.items()},
        }

    def to_json(self) -> str:
        """Canonical: sorted keys, no whitespace, instants as exact fractions in strings."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
