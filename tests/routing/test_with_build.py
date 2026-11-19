"""With the GPU build installed on a machine whose only adapter is a
software one, a GPU-route scene is refused as on a machine with no build.
Collected only where the build is installed.
"""

from __future__ import annotations

import glob
import json

from .conftest import SCENES, SMALL, run_child

CHILD = """
import json
import arachne_gpu
from arachne import routing
from arachne.session import Session
seen = [a["device_type"] for a in arachne_gpu.adapters()]
session = Session.open({scenes!r}, "LastPlay", config={config!r})
try:
    routing.pin_for(session.record)
except routing.GpuRequired as error:
    out = {{"seen": seen, "gpu": routing.gpu(), "code": error.code,
           "severity": error.finding.severity,
           "capabilities": list(error.capabilities)}}
else:
    out = {{"seen": seen, "refused": False}}
session.close()
print(json.dumps(out))
"""


def _software_only() -> dict[str, str]:
    """The environment of a machine whose one Vulkan driver is the software
    adapter's, whatever cards this one has.
    """
    manifests = sorted(glob.glob("/usr/share/vulkan/icd.d/lvp_icd*.json"))
    assert manifests, (
        "the software Vulkan adapter (mesa-vulkan-drivers) is not installed"
    )
    files = ":".join(manifests)
    return {"VK_DRIVER_FILES": files, "VK_ICD_FILENAMES": files}


def test_with_the_build_and_no_gpu_a_gpu_route_scene_is_refused():
    printed = run_child(
        CHILD.format(scenes=str(SCENES), config=SMALL), env=_software_only()
    )
    out = json.loads(printed.strip().splitlines()[-1])
    assert out["seen"]
    assert set(out["seen"]) == {"Cpu"}
    assert out == {
        "seen": out["seen"],
        "gpu": None,
        "code": "render.gpu_required",
        "severity": "error",
        "capabilities": ["effect.blur"],
    }
