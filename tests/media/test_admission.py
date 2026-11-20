"""Admission of a media asset by its frames: ``frames_exact`` and
``frames_uniform``, the tool's fixed lines and exit status, the planted
defects each failing their own check and no other, and strict loading at the
asset's digest.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import arachne
from arachne import Still, VideoMobject
from arachne.assets import admit, load_asset
from arachne.assets.reader import FIXTURES, read_asset
from arachne.lint.raised import RaisedFinding

from .conftest import BROKEN, copied, edited

ROOT = Path(arachne.__file__).resolve().parents[1]

PLANTED = {
    "wrong-digest": "frames_exact",
    "wrong-size": "frames_uniform",
    "translucent-opaque": "frames_uniform",
}


def _tool(directory: Path) -> subprocess.CompletedProcess[str]:
    """``python -m arachne.assets admit <directory>``, in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, "-m", "arachne.assets", "admit", str(directory)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ},
        timeout=300,
    )


def test_the_tool_admits_a_media_asset_by_its_two_checks():
    directory = FIXTURES / "numbered-clip"
    done = _tool(directory)
    assert done.returncode == 0, done.stderr
    digest = read_asset(directory).digest
    assert done.stdout.splitlines() == [
        f"asset: numbered-clip 1 (clip) {digest}",
        "PASS frames_exact",
        "PASS frames_uniform",
        "admission: 2/2 pass",
    ]


def test_the_tool_fails_a_planted_frame_by_its_check():
    done = _tool(BROKEN / "wrong-digest")
    assert done.returncode == 1, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("asset: wrong-digest 1 (clip) sha256:")
    assert lines[1].startswith("FAIL frames_exact: frames/0001.png decodes to")
    assert lines[2:] == ["PASS frames_uniform", "admission: 1/2 pass"]


@pytest.mark.parametrize(("name", "check"), sorted(PLANTED.items()))
def test_each_planted_asset_fails_its_own_check_and_no_other(name, check):
    report = admit(BROKEN / name)
    assert [r.name for r in report.results] == ["frames_exact", "frames_uniform"]
    assert [r.name for r in report.results if not r.passed] == [check]
    failed = report.result(check)
    assert failed.pose is not None
    assert failed.failure is not None
    # Its committed record says so, at its digest, and loading refuses it.
    with pytest.raises(RaisedFinding) as raised:
        load_asset(BROKEN / name)
    assert raised.value.code == "asset.unadmitted"
    assert raised.value.detail["reason"] == "failing"
    assert raised.value.detail["checks"] == [check]


def test_the_checks_name_what_is_wrong():
    assert "is 16 x 12 pixels, and the manifest says 16 x 16" in (
        admit(BROKEN / "wrong-size").result("frames_uniform").failure
    )
    assert "a pixel of alpha 128" in (
        admit(BROKEN / "translucent-opaque").result("frames_uniform").failure
    )


def test_one_changed_byte_of_a_frame_file_asks_for_admission_again(tmp_path):
    directory = copied(tmp_path, "numbered-clip")
    frame = directory / "frames" / "0007.png"
    data = bytearray(frame.read_bytes())
    data[-1] ^= 1
    frame.write_bytes(bytes(data))
    with pytest.raises(RaisedFinding) as raised:
        load_asset(directory)
    assert raised.value.code == "asset.unadmitted"
    assert raised.value.detail["reason"] == "digest"
    with pytest.raises(RaisedFinding):
        VideoMobject(directory)


def test_a_media_asset_with_no_record_is_unadmitted_by_every_class(tmp_path):
    clip = edited(copied(tmp_path, "numbered-clip"), lambda d: d.pop("checks"))
    still = edited(copied(tmp_path, "swatch"), lambda d: d.pop("checks"))
    for build, directory in ((VideoMobject, clip), (Still, still)):
        with pytest.raises(RaisedFinding) as raised:
            build(directory)
        assert raised.value.code == "asset.unadmitted"
        assert raised.value.detail["reason"] == "unchecked"


def test_a_rig_record_does_not_admit_a_media_asset(tmp_path):
    from arachne.assets.reader import CHECKS

    def as_a_rig(document: dict) -> None:
        document["checks"]["results"] = [{"name": n, "pass": True} for n in CHECKS]

    directory = edited(copied(tmp_path, "swatch"), as_a_rig)
    with pytest.raises(RaisedFinding) as raised:
        load_asset(directory)
    # The record is at the asset's digest (it attests nothing of its own
    # checks), and lacks the checks a media asset holds.
    assert raised.value.detail["reason"] == "missing"
    assert raised.value.detail["checks"] == ["frames_exact", "frames_uniform"]
