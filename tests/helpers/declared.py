"""Reading declared quantities as a client does: a scene opened at the
session server, ``state_at`` asked there, and each object's quantities read
from the answer by the name the scene's code gave the object; and the
content keys a mobject's captured values make.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from arachne.mobject.mobject import Mobject
from arachne.server.dispatch import Dispatcher
from arachne.timeline.record import NodeState, content_fields


class DeclaredServer:
    """A session server with the scene ``scene`` of ``scenes`` open."""

    def __init__(self, scenes: Path, scene: str, config: dict[str, Any]) -> None:
        self.server = Dispatcher()
        self.server.handle(
            "open", {"file": str(scenes), "scene": scene, "config": config}
        )

    def state(self, t: Fraction | float) -> dict[str, Any]:
        """``state_at``'s answer at ``t`` (an exact instant goes as ``"n/d"``)."""
        asked = f"{t.numerator}/{t.denominator}" if isinstance(t, Fraction) else t
        return self.server.handle("state_at", {"t": asked})["state"]

    def quantities(self, t: Fraction | float, name: str) -> dict[str, dict[str, Any]]:
        """The quantities of the object the scene's code named ``name``, at
        ``t``, each entry by its quantity's name.
        """
        return quantities_of(self.state(t), name)

    def close(self) -> None:
        self.server.handle("close", {})


def quantities_of(state: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    """The quantities of the object named ``name`` in the answer ``state``."""
    (node,) = [node for node in state["nodes"] if node["name"] == name]
    return {each["name"]: each for each in state["quantities"].get(node["id"], [])}


def values(found: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each quantity's value, by name; none of them unreadable."""
    assert not [each for each in found.values() if "unreadable" in each], found
    return {name: each["value"] for name, each in found.items()}


def keyed(mobject: Mobject, attribute: str) -> tuple[bytes, bytes]:
    """What the content keys make of the field ``attribute`` of
    ``mobject``'s values now: its part as given, and rounded to 6 decimals.
    A play's registered start state and a footprint both digest each node's
    :func:`~arachne.timeline.record.content_fields`, so a field there is in
    both keys, and a value they cannot encode is left out of both.
    """
    state = NodeState.of(mobject)
    given, rounded = dict(content_fields(state)), dict(content_fields(state, 6))
    assert attribute in given, f"{attribute} is in no content key"
    return given[attribute], rounded[attribute]


def apart(before: tuple[bytes, bytes], after: tuple[bytes, bytes]) -> bool:
    """Whether two :func:`keyed` results differ in every part."""
    return all(a != b for a, b in zip(before, after, strict=True))
