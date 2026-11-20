"""What a world puts on the mobjects a record captures is plain and keys as
content: no value of a type the content key does not know, and one state
keys alike on two reads and in two executions. (The content key is the
record's once it keys every value; until then this file has nothing to
check.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("arachne_sim")
keys = pytest.importorskip("arachne.timeline.keys")

from arachne.session import Session  # noqa: E402
from arachne.timeline import admission  # noqa: E402
from arachne.timeline.execution import play_keys  # noqa: E402
from arachne.timeline.record import covered_fields  # noqa: E402

from . import world_scenes  # noqa: E402

SCENES = Path(world_scenes.__file__)
CONFIG = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}


@pytest.mark.parametrize("name", ["ClosedThenOpen", "AwayAndBack", "Tumbling"])
def test_every_value_a_world_puts_on_a_captured_mobject_keys(name):
    admission.DEMOTED.clear()
    sessions = [Session.open(SCENES, name, config=CONFIG) for _ in range(2)]
    try:
        record = sessions[0].record
        for play in record.plays:
            for mob, node in play.start_state.nodes():
                for field, value in covered_fields(node):
                    assert keys.unkeyable(value) is None, (type(mob).__name__, field)
            assert play.start_state.content_key() == play.start_state.content_key()
        assert play_keys(record) == play_keys(sessions[1].record)
    finally:
        for session in sessions:
            session.close()
        admission.DEMOTED.clear()
