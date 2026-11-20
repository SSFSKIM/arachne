"""The one table of GPU-only capabilities, and how an object declares one.

A capability is something only the GPU build draws. The table is data: each
row an id, named for what is drawn, and one sentence; nothing in it runs.
Rows are appended, never renamed or removed. Capability ids are a namespace
of their own: they are not lint codes. A plain still image has no row, and
never will: it is drawn by Cairo as it always was.

An object holds the rows it declares (:func:`declare`) as a frozen set under
one attribute of its own, :data:`ATTRIBUTE`. That attribute is part of the
object's state, so the scene's record keeps it with every version of the
object, and a capability that comes and goes is seen where it was held.
"""

from __future__ import annotations

__all__ = [
    "ATTRIBUTE",
    "Capability",
    "GPU_ONLY",
    "declare",
    "ever_declared",
    "held",
    "withdraw",
]

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Capability:
    """One row: its id and one sentence saying what is drawn."""

    id: str
    summary: str


ROWS = (
    Capability("effect.blur", "A layer blurred by a declared Gaussian radius."),
    Capability(
        "effect.depth_of_field",
        "The scene's focus: each layer blurred by how far its depth lies from the focal depth.",
    ),
    Capability("effect.soft_mask", "A layer cut out by a mask with a feathered edge."),
    Capability(
        "effect.window",
        "A layer showing another stack of layers inside a frame of four corners.",
    ),
    Capability(
        "layer.depth", "A layer placed at a declared depth, for the focus to blur."
    ),
)

#: Every capability, by id, in the order the rows were added.
GPU_ONLY: Mapping[str, Capability] = MappingProxyType({row.id: row for row in ROWS})

#: The attribute of an object holding the rows it declared.
ATTRIBUTE = "_gpu_only"


def held(mobject: Any) -> frozenset[str]:
    """The rows ``mobject`` holds now."""
    found: frozenset[str] = vars(mobject).get(ATTRIBUTE, frozenset())
    return found


#: Whether any object of this process has ever been declared to hold a row.
_declared = False


def ever_declared() -> bool:
    """Whether any object of this process has ever held a row: when not, no
    record made in it holds one, and nothing need be read to know it.
    """
    return _declared


def declare(mobject: Any, *ids: str) -> None:
    """Make ``mobject`` hold the rows ``ids`` as well as those it holds; a
    ``ValueError`` naming any id that is no row, and nothing declared.
    """
    global _declared
    rows = _rows(ids)
    _declared = _declared or bool(rows)
    _hold(mobject, held(mobject) | rows)


def withdraw(mobject: Any, *ids: str) -> None:
    """Make ``mobject`` hold the rows it holds but ``ids``; holding none, it
    keeps no attribute, as an object that never declared one.
    """
    _hold(mobject, held(mobject) - _rows(ids))


def _rows(ids: tuple[str, ...]) -> frozenset[str]:
    unknown = [name for name in ids if name not in GPU_ONLY]
    if unknown:
        raise ValueError(
            f"{', '.join(map(repr, unknown))} is no GPU-only capability; the table"
            f" holds {', '.join(GPU_ONLY)}"
        )
    return frozenset(ids)


def _hold(mobject: Any, rows: frozenset[str]) -> None:
    own = vars(mobject)
    if rows == own.get(ATTRIBUTE, frozenset()):
        # The same value: the record keeps no new version for nothing.
        return
    if rows:
        own[ATTRIBUTE] = rows
    else:
        del own[ATTRIBUTE]
