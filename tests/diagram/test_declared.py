"""A flow's declared speed: ``state_at`` answers it at the session server,
read from what the record captured, and 0 once the flow is stopped; and
it is in the content keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne import RIGHT, Flow, Line

from ..helpers.declared import DeclaredServer, apart, keyed, values
from . import declared_scenes
from .declared_scenes import FAST, SLOW

SCENES = Path(declared_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}


@pytest.fixture
def flows(tmp_path):
    server = DeclaredServer(
        SCENES, "FlowStopped", {**SMALL, "media_dir": str(tmp_path / "media")}
    )
    yield server
    server.close()


def test_a_flow_answers_its_speed_and_0_once_stopped(flows):
    found = flows.quantities(0.25, "fast")
    assert values(found) == {"diagram.flow_speed": FAST}
    assert found["diagram.flow_speed"]["unit"] == "u/s"
    assert values(flows.quantities(0.25, "slow")) == {"diagram.flow_speed": SLOW}
    # Stopped, its tokens stand still.
    assert values(flows.quantities(0.75, "fast")) == {"diagram.flow_speed": FAST}
    assert values(flows.quantities(0.75, "slow")) == {"diagram.flow_speed": 0.0}


@pytest.mark.parametrize(("attribute", "changed"), [("speed", 2.0), ("stopped", True)])
def test_a_flow_s_speed_is_in_the_content_keys(attribute, changed):
    flow = Flow(Line(-RIGHT, RIGHT), speed=1.0)
    before = keyed(flow, attribute)
    setattr(flow, attribute, changed)
    after = keyed(flow, attribute)
    assert apart(before, after)
