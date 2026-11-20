"""A readout's and a gauge's declared value: ``state_at`` answers it at
the session server, read from the value the record captured, through a
count and after the scene's code sets it; and it is in the content keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne import BarGauge, DialGauge, Readout

from ..helpers.declared import DeclaredServer, apart, keyed, values
from . import declared_scenes

SCENES = Path(declared_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}


@pytest.fixture
def counted(tmp_path):
    server = DeclaredServer(
        SCENES, "Counted", {**SMALL, "media_dir": str(tmp_path / "media")}
    )
    yield server
    server.close()


def test_a_readout_answers_its_count(counted):
    found = counted.quantities(0.25, "reading")
    assert values(found) == {"annotation.count": 0.0}
    assert found["annotation.count"]["unit"] == "1"
    assert "tolerance" not in found["annotation.count"]
    # Half way through the linear count.
    assert values(counted.quantities(1.0, "reading")) == {
        "annotation.count": pytest.approx(5.0)
    }
    # Done, the target exactly.
    assert values(counted.quantities(2.0, "reading")) == {"annotation.count": 10.0}


@pytest.mark.parametrize(
    ("name", "start", "middle", "end"),
    [("bar", 0.25, 0.5, 0.75), ("column", 2.0, 3.0, 4.0), ("dial", 30.0, 40.0, 60.0)],
)
def test_a_gauge_answers_its_value(counted, name, start, middle, end):
    found = counted.quantities(0.25, name)
    assert values(found) == {"annotation.gauge": start}
    assert found["annotation.gauge"]["unit"] == "1"
    assert values(counted.quantities(1.0, name)) == {
        "annotation.gauge": pytest.approx(middle)
    }
    # The dial was set by the scene's code after its count.
    assert values(counted.quantities(2.0, name)) == {"annotation.gauge": end}


@pytest.mark.parametrize(
    "made",
    [lambda: Readout(1.0), lambda: BarGauge(0.25), lambda: DialGauge(0.25)],
    ids=["readout", "bar", "dial"],
)
def test_a_declared_value_is_in_the_content_keys(made):
    mobject = made()
    before = keyed(mobject, "_value")
    mobject.set_value(0.5)
    after = keyed(mobject, "_value")
    assert apart(before, after)
