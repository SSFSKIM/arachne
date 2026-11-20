"""Findings raised while a scene's code runs.

A :class:`RaisedFinding` is a finding of the lint's table (:mod:`.codes`)
raised as an exception, where going on would put a wrong picture or an empty
space on screen: the scene does not build and nothing is drawn. It carries
the code, what is wrong, the code's fix and the ``detail`` fields the code
declares; the session server answers it as a refusal whose data holds the
finding whole, with the line of the scene file it was raised through
(:mod:`arachne.server.dispatch`).
"""

from __future__ import annotations

__all__ = ["RaisedFinding", "drawing_refusal", "found", "refuse_drawing"]

import sys
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .codes import code as _code

if TYPE_CHECKING:
    from arachne.timeline.record import Record


class RaisedFinding(Exception):
    """The finding of ``code`` (a code of the lint's table), raised.

    ``what`` says what is wrong, in this case; it defaults to the code's
    meaning. ``detail`` holds fields the code declares, and only those.
    ``line`` is the scene file's line that asked for what is refused, when
    the raiser knows it; else the server finds it from the traceback.

    ``str(error)`` is ``"<code>: <what is wrong>. <the code's fix>"``.
    """

    def __init__(
        self,
        code: str,
        what: str | None = None,
        *,
        detail: Mapping[str, Any] | None = None,
        line: int | None = None,
    ) -> None:
        entry = _code(code)
        detail = dict(detail or {})
        unknown = set(detail) - set(entry.detail)
        if unknown:
            raise ValueError(
                f"A {code} finding carries no {', '.join(sorted(unknown))}"
                f" (its detail fields: {', '.join(entry.detail) or 'none'})."
            )
        self.code = code
        self.what = (what or entry.meaning).rstrip(".")
        self.detail: Mapping[str, Any] = MappingProxyType(detail)
        self.line = line
        self.severity = entry.severity
        self.cls = entry.cls
        self.fix = entry.fix
        super().__init__(f"{code}: {self.what}. {self.fix}")

    def finding(self, line: int | None = None) -> dict[str, Any]:
        """The finding as the server answers it: ``{code, class, severity,
        message, fix, line, detail}``, ``message`` what is wrong; ``line``
        the raiser's when it gave one, else ``line``.
        """
        return {
            "code": self.code,
            "class": self.cls,
            "severity": self.severity,
            "message": self.what,
            "fix": self.fix,
            "line": self.line if self.line is not None else line,
            "detail": dict(self.detail),
        }


def found(
    code: str,
    what: str | None = None,
    *,
    detail: Mapping[str, Any] | None = None,
    objects: Iterable[Any] = (),
) -> None:
    """Record the finding of ``code`` for the lint, at the scene's line and
    time now: about ``objects`` (mobjects), saying ``what`` is wrong, with
    ``detail`` as :class:`RaisedFinding` takes them. Drawing a frame of the
    scene then raises it (:func:`refuse_drawing`). Outside a recording,
    raises it now.
    """
    from arachne.timeline.recorder import Reading, running

    raised = RaisedFinding(code, what, detail=detail)
    recorder = running()
    if recorder is None:
        raise raised
    from arachne.timeline.origins import origin_of

    from .findings import Finding, FindingObject

    mobjects = tuple(objects)
    call = recorder.bridge_call(
        "found",
        sys._getframe(1),
        code=code,
        operands=tuple(origin_of(mob).id for mob in mobjects),
    )
    finding = Finding(
        code,
        call.t,
        call.t,
        objects=tuple(
            FindingObject(
                id=origin.id,
                source_line=origin.scene_line,
                kind=type(mob).__name__,
                label_or_text=None,
                bbox=None,
                screen_bbox=None,
            )
            for mob in mobjects
            for origin in (origin_of(mob),)
        ),
        message=raised.what,
        line=call.line,
        detail=raised.detail,
    )
    call.readings.append(
        Reading(call.t, call.line, {"findings": (finding,)}, seq=call.seq)
    )


def refuse_drawing(record: Record) -> None:
    """Raise, as a :class:`RaisedFinding` at its line, the first finding of
    error severity :func:`found` recorded in ``record``; nothing when there
    is none.
    """
    refusal = drawing_refusal(record)
    if refusal is not None:
        raise refusal


def drawing_refusal(record: Record) -> RaisedFinding | None:
    """The first finding of error severity :func:`found` recorded in
    ``record``, as the :class:`RaisedFinding` drawing raises; ``None`` when
    there is none.
    """
    for call in record.bridge.calls:
        if call.kind != "found":
            continue
        for reading in call.readings:
            for finding in (reading.values or {}).get("findings", ()):
                if finding.severity == "error":
                    return RaisedFinding(
                        finding.code,
                        finding.message,
                        detail=finding.detail,
                        line=finding.line.line if finding.line is not None else None,
                    )
    return None
