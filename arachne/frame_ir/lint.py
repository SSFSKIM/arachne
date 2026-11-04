"""The drawing's lint codes, appended to the lint table when it lands.

Until the lint table exists in the engine, the codes the Frame IR's drawers
raise are kept here, in the table's form: a code, its severity and what it
means.
"""

from __future__ import annotations

__all__ = ["CODES"]

#: code -> (severity, meaning)
CODES: dict[str, tuple[str, str]] = {
    "render.unsupported": (
        "error",
        "a frame's description carries a record kind or feature bit the drawer"
        " does not know; the frame is refused, never drawn partly",
    ),
    "render.fallback": (
        "warning",
        "a 3D scene was drawn with Cairo because the core refused something in"
        " it; the finding names the cause and its source line",
    ),
}
