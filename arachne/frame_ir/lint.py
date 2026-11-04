"""The drawing's lint codes, appended to the lint table when it lands.

Until the lint table exists in the engine, the codes the Frame IR's drawers
raise are kept here, in the table's form: a code, its severity and what it
means; and the findings raised in this process are kept here too
(:func:`raise_finding`, :data:`raised`), each logged once.
"""

from __future__ import annotations

__all__ = ["CODES", "Finding", "raise_finding", "raised"]

from dataclasses import dataclass

#: code -> (severity, meaning)
CODES: dict[str, tuple[str, str]] = {
    "render.unsupported": (
        "error",
        "a frame's description carries a record kind or feature bit the drawer"
        " does not know; the frame is refused, never drawn partly",
    ),
    "render.fallback": (
        "warning",
        "a scene was drawn with Cairo because the Frame IR's path refused"
        " something in it (a camera class of the scene's own; for a 3D scene,"
        " anything the core refuses); the finding names the cause and its"
        " source line",
    ),
}


@dataclass(frozen=True)
class Finding:
    """One finding: its code, what it names (a class, a mobject) and why."""

    code: str
    subject: str
    message: str


#: The findings raised in this process, each once, in the order raised.
raised: list[Finding] = []


def raise_finding(code: str, subject: str, message: str) -> Finding:
    """Raise a finding of ``code`` (one of :data:`CODES`) about ``subject``:
    kept in :data:`raised` and logged, once per process.
    """
    if code not in CODES:
        raise ValueError(f"{code!r} is not a drawing lint code")
    finding = Finding(code, subject, message)
    if finding not in raised:
        from arachne._config import logger

        raised.append(finding)
        logger.warning("%s: %s", code, message)
    return finding
