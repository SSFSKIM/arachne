"""What a world puts on the mobjects a record captures is plain and keys as
content: no value of a type the content key does not know, and each keys
alike on two reads and in two executions. (The content key keys every value
once :mod:`arachne.timeline.keys` is part of the engine; until then this
file has nothing to check.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("arachne_sim")
keys = pytest.importorskip("arachne.timeline.keys")

from arachne import World  # noqa: E402
from arachne.session import Session  # noqa: E402
from arachne.timeline import admission  # noqa: E402

from . import world_scenes  # noqa: E402

SCENES = Path(world_scenes.__file__)
CONFIG = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}


def _world_values(session: Session) -> list[tuple[str, object]]:
    """Every attribute of every world the scene ended with, and of each
    member of its bodies' families, by where it is.
    """
    found = []
    roots = session.record.plays[-1].samples[-1].roots
    for w, world in enumerate(m for m in roots if isinstance(m, World)):
        for member in world.get_family():
            for name, value in sorted(vars(member).items()):
                found.append((f"{w} {type(member).__name__} {name}", value))
    return found


@pytest.mark.parametrize("name", ["ClosedThenOpen", "AwayAndBack", "Tumbling"])
def test_every_value_a_world_puts_on_a_captured_mobject_keys(name):
    admission.DEMOTED.clear()
    sessions = [Session.open(SCENES, name, config=CONFIG) for _ in range(2)]
    try:
        first, second = (_world_values(session) for session in sessions)
        assert [where for where, _ in first] == [where for where, _ in second]
        for (where, value), (_, other) in zip(first, second, strict=True):
            assert keys.unkeyable(value) is None, where
            key = keys.value_key(value)[0]
            assert key == keys.value_key(value)[0], where
            assert key == keys.value_key(other)[0], where
    finally:
        for session in sessions:
            session.close()
        admission.DEMOTED.clear()
