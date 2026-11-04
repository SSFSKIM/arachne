"""A lint result as the machine-facing report gives it.

:func:`report` is the report: its violations are the errors and warnings,
and it passes unless one is an error. :func:`notes` are the notes, in the
same violation shape, kept apart from the report. A violation's ``bbox``
is in scene units, its ``screen_bbox`` in pixels of the declared target
(:data:`.codes.TARGET_HEIGHT` tall, from the top-left corner, y down), and
its ``line`` the scene line running when it was found.
"""

from __future__ import annotations

__all__ = ["TARGET_HEIGHT", "notes", "report"]

from typing import Any

from .codes import TARGET_HEIGHT
from .core import LintResult
from .findings import Finding, FindingObject


def report(result: LintResult) -> dict[str, Any]:
    """The report of ``result``: its errors and warnings, and its timeline."""
    return {
        "pass": result.passed,
        "duration_s": result.duration_s,
        "violations": [_violation(f) for f in result.findings if f.severity != "note"],
        "timeline": [
            {
                "index": span.index,
                "kind": span.kind,
                "t_start": float(span.t_start),
                "t_end": float(span.t_end),
            }
            for span in result.timeline
        ],
    }


def notes(result: LintResult) -> list[dict[str, Any]]:
    """The notes of ``result``, as violations."""
    return [_violation(f) for f in result.findings if f.severity == "note"]


def _violation(finding: Finding) -> dict[str, Any]:
    out: dict[str, Any] = {
        "code": finding.code,
        "class": finding.cls,
        "severity": finding.severity,
        "held": finding.held,
        "t_start": _time(finding.t_start),
        "t_end": _time(finding.t_end),
        "objects": [_object(o) for o in finding.objects],
        "message": finding.message,
        "fix": finding.fix,
        "line": finding.line.line if finding.line is not None else None,
    }
    if finding.detail:
        out["detail"] = dict(finding.detail)
    if finding.folded:
        out["message"] += f" (and {len(finding.folded)} more like it)"
    return out


def _object(obj: FindingObject) -> dict[str, Any]:
    return {
        "id": obj.id,
        "source_line": obj.source_line,
        "kind": obj.kind,
        "label_or_text": obj.label_or_text,
        "name": obj.name,
        "bbox": _numbers(obj.bbox),
        "screen_bbox": _numbers(obj.screen_bbox),
    }


def _time(t: Any) -> float | None:
    return None if t is None else float(t)


def _numbers(box: Any) -> list[float] | None:
    return None if box is None else [float(v) for v in box]
