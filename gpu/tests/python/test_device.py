"""Which adapters draw, what the pin says, and who may read the launch mode or open the testing
door. A test about a machine with no GPU makes that machine for itself in a child process.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import arachne_gpu
import pytest
from conftest import MODE, launch_mode

GPU = Path(__file__).resolve().parents[2]
REPOSITORY = GPU.parent

PROBE = """
import json, arachne_gpu
from arachne_gpu.testing import software_compositor
report = {"adapters": arachne_gpu.adapters()}
try:
    arachne_gpu.Compositor(64, 64)
    report["ordinary"] = None
except arachne_gpu.NoGpu as refused:
    report["ordinary"] = {"message": str(refused), "seen": refused.seen}
try:
    report["software"] = software_compositor(64, 64).software
except arachne_gpu.NoGpu as refused:
    report["software"] = str(refused)
print(json.dumps(report))
"""


def probe(env: dict[str, str]) -> dict:
    import os

    done = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **env},
    )
    return json.loads(done.stdout.splitlines()[-1])


def software_icd() -> str:
    found = sorted(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
    assert found, "no lavapipe ICD manifest in /usr/share/vulkan/icd.d"
    return str(found[0])


def test_the_ordinary_door_refuses_a_machine_whose_only_adapter_is_software():
    icd = software_icd()
    report = probe({"VK_DRIVER_FILES": icd, "VK_ICD_FILENAMES": icd})
    assert report["adapters"]
    assert all(a["device_type"] == "Cpu" for a in report["adapters"])
    refused = report["ordinary"]
    assert refused is not None, "no GPU, yet a compositor"
    assert refused["seen"] == report["adapters"]
    for adapter in report["adapters"]:
        assert adapter["name"] in refused["message"]
    assert report["software"] is True


def test_the_ordinary_door_refuses_a_machine_with_no_vulkan_loader(tmp_path):
    # A libvulkan.so.1 that is no library, found first: the loader cannot be loaded.
    (tmp_path / "libvulkan.so.1").write_bytes(b"")
    report = probe({"LD_LIBRARY_PATH": str(tmp_path)})
    assert report["adapters"] == []
    assert report["ordinary"] is not None
    assert report["ordinary"]["seen"] == []
    assert "no Vulkan adapter was found" in report["ordinary"]["message"]
    assert isinstance(report["software"], str)


def test_adapters_lists_what_the_machine_offers():
    for adapter in arachne_gpu.adapters():
        assert set(adapter) == {
            "name",
            "backend",
            "device_type",
            "driver",
            "driver_info",
            "vendor",
            "device",
        }
        assert adapter["backend"] == "Vulkan"


def test_the_pin_names_the_device_and_the_locked_wgpu(make_compositor):
    compositor = make_compositor(32, 32)
    pin, adapter = compositor.pin, compositor.adapter
    assert set(pin) == {"adapter", "backend", "driver", "wgpu"}
    assert pin["adapter"] == adapter["name"]
    assert pin["backend"] == "Vulkan"
    assert pin["driver"] == adapter["driver_info"]
    lock = tomllib.loads((GPU / "Cargo.lock").read_text())
    locked = [p["version"] for p in lock["package"] if p["name"] == "wgpu"]
    assert locked == [arachne_gpu.__wgpu_version__] == [pin["wgpu"]]


def test_with_no_launch_mode_the_drawing_tests_fail():
    with pytest.raises(RuntimeError, match="not set"):
        launch_mode({})
    with pytest.raises(RuntimeError):
        launch_mode({MODE: "gpu"})
    with pytest.raises(RuntimeError):
        launch_mode({MODE: "hardware"})
    assert launch_mode({MODE: "software"}) == ("software", None)


def sources(*roots: Path) -> list[Path]:
    found = []
    for root in roots:
        found += [
            p for p in root.rglob("*") if p.suffix in {".py", ".rs", ".wgsl", ".sh"}
        ]
    assert found
    return found


def test_no_library_code_reads_the_launch_mode():
    library = sources(GPU / "src", GPU / "python", REPOSITORY / "arachne") + [
        GPU / "build.rs"
    ]
    readers = [str(p) for p in library if MODE in p.read_text(errors="replace")]
    assert readers == []


def test_nothing_in_the_engine_opens_the_testing_door():
    door = re.compile(
        r"arachne_gpu(\.testing|\._gpu)|from\s+arachne_gpu\s+import\s+.*testing"
    )
    openers = [
        str(p)
        for p in sources(REPOSITORY / "arachne")
        if door.search(p.read_text(errors="replace"))
        or "software_compositor" in p.read_text()
    ]
    assert openers == []
