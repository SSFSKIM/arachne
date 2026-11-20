"""Nothing on the ``cpu`` route imports the GPU build: importing Arachne,
reading a scene's route and drawing it leave no record in the import
tripwire the gate ``cpu-bytes`` installs, and asking for the device does.
"""

from __future__ import annotations

import json
from pathlib import Path

from .conftest import SCENES, SMALL, run_child

TRIPWIRE = Path(__file__).resolve().parents[2] / "gpu" / "ci" / "tripwire.py"

CHILD = """
import site, sys
site.addsitedir({site!r})
from fractions import Fraction
from arachne.session import Session
from arachne import routing
session = Session.open({scenes!r}, "Plain", config={config!r})
record = session.record
assert routing.route(record) == "cpu"
assert routing.pin_for(record) is None
assert routing.cache_token("cpu", None) == ""
session.frame(Fraction(1, 2))
session.close()
print("cpu route drawn", flush=True)
with open({log!r}) as log:
    print("records before asking for the device:", len(log.readlines()), flush=True)
routing.gpu()
"""


def test_a_cpu_route_scene_never_imports_the_gpu_build(tmp_path):
    site, log = tmp_path / "site", tmp_path / "tripwire.log"
    site.mkdir()
    namespace: dict = {}
    exec(compile(TRIPWIRE.read_text(), str(TRIPWIRE), "exec"), namespace)
    namespace["install"](site, log)
    printed = run_child(
        CHILD.format(site=str(site), scenes=str(SCENES), config=SMALL, log=str(log))
    )
    assert "cpu route drawn" in printed
    assert "records before asking for the device: 0" in printed
    # Asking for the device is the one import, and the tripwire saw it.
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
