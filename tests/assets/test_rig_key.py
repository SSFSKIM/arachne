"""A rig's asset is in the content keys: two rigs whose assets differ in
nothing a rig's points or values show (here a drive's angle unit, at a pose
whose angle is zero in either) key apart at a play's start, by the asset's
digest; the same asset read from two places keys alike.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.assets import load_asset
from arachne.session import Session
from arachne.timeline import admission

from ..timing.reading import SMALL
from .helpers import admitted, copied, edited

SCENE = """
from arachne import Scene
from arachne.assets import Rig


class Dial(Scene):
    def construct(self):
        self.add(Rig({asset!r}))
        self.wait(0.25)
"""


def _in_radians(document: dict) -> None:
    (parameter,) = document["parameters"]
    (drive,) = parameter["drives"]
    assert drive["unit"] == "deg"
    drive["unit"] = "rad"


@pytest.fixture
def start_key(tmp_path):
    """The content key of the state the first play of a scene showing a
    rig of the asset at a directory began from.
    """
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def key_of(asset: Path) -> bytes:
        scene = tmp_path / f"{asset.name}_scene.py"
        scene.write_text(SCENE.format(asset=str(asset)))
        session = Session.open(scene, "Dial", config={**SMALL, "frame_rate": 15})
        sessions.append(session)
        return session.record.plays[0].start_state.content_key()

    yield key_of
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def test_a_rig_s_asset_is_in_the_play_s_start_key(tmp_path, start_key):
    degrees = copied(tmp_path, "dial", "dial-deg")
    again = copied(tmp_path, "dial", "dial-again")
    radians = admitted(edited(copied(tmp_path, "dial", "dial-rad"), _in_radians))
    # The two assets are two: the unit is the manifest's, so the digest.
    assert load_asset(radians).digest != load_asset(degrees).digest
    assert start_key(radians) != start_key(degrees)
    # The same asset, read from another directory, keys alike.
    assert start_key(again) == start_key(degrees)
