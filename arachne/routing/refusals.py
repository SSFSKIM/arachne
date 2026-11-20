"""The two refusals of a scene holding a GPU-only capability.

Each is an error under a code of its own, carries the finding that names
the capabilities and where the scene first holds one, and is answered by
the session server as every engine refusal is. Neither is ``render.fallback``:
a scene holding a capability is never drawn without it.
"""

from __future__ import annotations

__all__ = ["GpuLive", "GpuRequired", "Refusal"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from arachne.lint.findings import Finding

if TYPE_CHECKING:
    from .route import Where


class Refusal(Exception):
    """A scene refused because it holds a GPU-only capability: nothing of
    it is drawn or written.
    """

    code: ClassVar[str]
    #: What the message says of the scene, after its capabilities.
    because: ClassVar[str]

    def __init__(
        self, capabilities: Mapping[str, Where], *, why: str = "", **detail: Any
    ) -> None:
        if not capabilities:
            raise ValueError(f"{self.code} is raised for a scene holding a capability.")
        self.capabilities = dict(capabilities)
        first = next(iter(self.capabilities.values()))
        rows = ", ".join(self.capabilities)
        held = "capability" if len(self.capabilities) == 1 else "capabilities"
        at = f"play {first.play}" if first.play is not None else "the scene's end"
        line = f", {first.line}" if first.line is not None else ""
        message = (
            f"the scene holds the GPU-only {held} {rows} (first at {at}{line},"
            f" on {first.object}) {self.because}{why}"
        )
        self.finding = Finding(
            self.code,
            first.t,
            first.t,
            message=message,
            line=first.line,
            detail={
                "capabilities": list(self.capabilities),
                "play": first.play,
                "object": first.object,
                **detail,
            },
        )
        super().__init__(f"{self.code}: {message}. {self.finding.fix}")


class GpuRequired(Refusal):
    """``render.gpu_required``: a scene holding a GPU-only capability asked to
    draw on a host with no GPU. Nothing is drawn or written.
    """

    code = "render.gpu_required"
    because = "and this host has no GPU; nothing was drawn or written"


class GpuLive(Refusal):
    """``render.gpu_live``: a scene holding a GPU-only capability whose frames
    are drawn before its record is complete, which only Cairo does. Nothing
    is written. ``cause`` is why it draws as it runs (``Record.refusal``).
    """

    code = "render.gpu_live"
    because = (
        "and its frames are drawn as it runs, which only Cairo does; nothing was"
        " written"
    )

    def __init__(self, capabilities: Mapping[str, Where], cause: str | None) -> None:
        why = f". It draws as it runs: {cause}" if cause else ""
        super().__init__(capabilities, why=why, cause=cause)
        self.cause = cause
