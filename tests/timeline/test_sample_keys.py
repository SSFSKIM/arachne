"""Sample keys are a function of content alone: two processes executing the
same scene key every kept capture alike, scenes that copy and ``become``
mobjects as they play included (a copy records its original's ``id``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from arachne.mobject.values import FrozenPoints, freeze
from arachne.timeline.record import SAMPLE_DECIMALS, _value_key

from . import stepping_scenes

SCENES = ["BecomeRoot", "GroupBecome", "Staircases", "StaircasesPerFrame"]

#: One process's keys: every play of every scene, as hex, in JSON.
KEYS = """
import json, sys, tempfile
from pathlib import Path
from arachne.session import Session

path, names = Path(sys.argv[1]), sys.argv[2:]
found = {}
with tempfile.TemporaryDirectory() as media:
    for name in names:
        for stepping in (True, False):
            session = Session.open(path, name, stepping=stepping, config={
                "frame_rate": 15, "pixel_width": 96, "pixel_height": 54,
                "progress_bar": "none", "verbosity": "ERROR", "media_dir": media})
            found[f"{name} {stepping}"] = [
                [[str(t), key.hex()] for t, key in play.sample_keys()]
                for play in session.record.plays
            ]
            session.close()
print(json.dumps(found))
"""


def _keys() -> dict:
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(
        [sys.executable, "-c", KEYS, stepping_scenes.__file__, *SCENES],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    return json.loads(done.stdout.splitlines()[-1])


def test_two_processes_key_every_capture_alike():
    first, second = _keys(), _keys()
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], name


def _point_key(x: float) -> bytes | None:
    return _value_key(freeze(np.array([[x, 0.0, 0.0]]), FrozenPoints), SAMPLE_DECIMALS)


def test_a_sample_key_rounds_numbers_so_nearness_does_not_decide_equality():
    """A known limit (``tech-debt-tracker.md``): a sample key rounds to
    :data:`SAMPLE_DECIMALS`, so two coordinates one ulp apart on either side
    of a rounding boundary key apart, and two 9e-10 apart inside one bin key
    alike.
    """
    assert _point_key(-0.9213813375) != _point_key(-0.9213813374999997)
    assert _point_key(0.09999999955) == _point_key(0.10000000045)
