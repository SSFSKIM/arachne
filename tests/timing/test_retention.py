"""A declared scene is let go with its session: nothing of stepped timing
outside the scene keeps it, whatever its clock holds (a Succession that
refers to the scene, a scene driver that holds it).
"""

from __future__ import annotations

import gc
import weakref

import pytest

from arachne.assets import timing
from arachne.session import Session

from .reading import SCENES, SMALL


@pytest.mark.parametrize("scene", ["SuccessionOfMoves", "StubDrivers"])
def test_a_closed_declared_session_lets_its_scene_go(scene):
    session = Session.open(SCENES, scene, config={**SMALL, "frame_rate": 15})
    assert timing.clock_of(session.scene) is not None
    kept = weakref.ref(session.scene)
    session.close()
    del session
    gc.collect()
    assert kept() is None
