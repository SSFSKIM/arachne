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

__all__ = ["RaisedFinding"]

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .codes import code as _code


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
