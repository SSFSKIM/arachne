"""Strict loading: an asset loads only when its manifest records every check
of the supported admission tool, passing, at the digest of the asset as it
is; anything else is ``asset.unadmitted``, whose reason says what is not so.
Loading also checks what needs the asset's geometry: its anchors and the
names a pose reads.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from arachne.assets import Rig, admit, load_asset
from arachne.assets.__main__ import main
from arachne.assets.reader import CHECKS, TOOL, read_asset
from arachne.lint.raised import RaisedFinding
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError

from .helpers import copied, edited, refused

DEFECTS = Path(__file__).parent / "defects"


def test_every_fixture_loads_admitted():
    for name in ("lidded-box", "lidded-crate", "dial", "gate-valve", "figure"):
        assert load_asset(f"fixture:{name}").name == name
    assert Rig("fixture:layered-section").asset.name == "layered-section"


def test_an_asset_never_admitted_is_refused(tmp_path):
    directory = edited(copied(tmp_path, "dial"), lambda d: d.pop("checks"))
    found = refused("asset.unadmitted", lambda: load_asset(directory))
    assert found.detail == {"asset": str(directory), "reason": "unchecked"}
    assert "python -m arachne.assets admit" in found.fix


def test_an_asset_admitted_by_another_tool_version_is_refused(tmp_path):
    directory = edited(
        copied(tmp_path, "dial"),
        lambda d: d["checks"].update(tool="arachne.assets 2"),
    )
    found = refused("asset.unadmitted", lambda: Rig(directory))
    assert found.detail["reason"] == "tool"
    assert found.detail["expected"] == TOOL
    assert found.detail["found"] == "arachne.assets 2"


def test_an_asset_changed_since_its_admission_is_refused(tmp_path):
    directory = copied(tmp_path, "lidded-box")
    recorded = load_asset(directory).digest
    svg = directory / "lidded-box.svg"
    svg.write_text(svg.read_text().replace("#c8a165", "#c8a166", 1))
    found = refused("asset.unadmitted", lambda: load_asset(directory))
    assert found.detail["reason"] == "digest"
    assert found.detail["found"] == recorded
    assert found.detail["expected"] == read_asset(directory).digest != recorded


def test_a_manifest_changed_since_its_admission_is_refused(tmp_path):
    directory = edited(copied(tmp_path, "lidded-box"), lambda d: d.update(version=2))
    found = refused("asset.unadmitted", lambda: load_asset(directory))
    assert found.detail["reason"] == "digest"


@pytest.mark.parametrize("check", CHECKS)
def test_a_record_missing_a_check_is_refused_naming_it(tmp_path, check):
    def drop(document: dict) -> None:
        results = document["checks"]["results"]
        document["checks"]["results"] = [r for r in results if r["name"] != check]

    directory = edited(copied(tmp_path, "dial"), drop)
    found = refused("asset.unadmitted", lambda: load_asset(directory))
    assert found.detail["reason"] == "missing"
    assert found.detail["checks"] == [check]


def test_a_record_that_is_not_a_list_of_results_misses_every_check(tmp_path):
    directory = edited(
        copied(tmp_path, "dial"),
        lambda d: d["checks"].update(results={"hole": True}),
    )
    found = refused("asset.unadmitted", lambda: load_asset(directory))
    assert found.detail["checks"] == list(CHECKS)


@pytest.mark.parametrize("failed", [False, None, "true", 1])
def test_a_check_recorded_as_anything_but_passing_is_failing(tmp_path, failed):
    directory = edited(
        copied(tmp_path, "dial"),
        lambda d: d["checks"]["results"][1].update({"pass": failed}),
    )
    found = refused("asset.unadmitted", lambda: load_asset(directory))
    assert found.detail["reason"] == "failing"
    assert found.detail["checks"] == ["detached_part"]


def test_a_check_recorded_twice_passes_only_if_both_pass(tmp_path):
    def twice(document: dict) -> None:
        document["checks"]["results"].append({"name": "hole", "pass": False})

    directory = edited(copied(tmp_path, "dial"), twice)
    found = refused("asset.unadmitted", lambda: load_asset(directory))
    assert found.detail["checks"] == ["hole"]


def test_a_check_of_a_later_tool_beside_these_is_ignored(tmp_path):
    def later(document: dict) -> None:
        document["checks"]["results"].append({"name": "style", "pass": False})

    directory = edited(copied(tmp_path, "dial"), later)
    assert load_asset(directory).name == "dial"


def test_a_planted_defect_is_refused_by_its_failing_check():
    for defect, check in (
        ("open-seam", "hole"),
        ("loose-part", "detached_part"),
        ("stray-anchor", "anchor_off_part"),
    ):
        found = refused("asset.unadmitted", lambda d=defect: load_asset(DEFECTS / d))
        assert found.detail["checks"] == [check]


def test_the_refusals_come_in_order_of_what_a_maker_must_mend_first(tmp_path):
    # A file missing is reported before the record that no longer attests.
    directory = edited(copied(tmp_path, "dial"), lambda d: d.pop("checks"))
    (directory / "slots" / "lamp-on.svg").unlink()
    refused("asset.file_missing", lambda: load_asset(directory))
    # An invalid manifest before its record.
    directory = edited(
        copied(tmp_path, "figure"),
        lambda d: d["anchors"][1].update(at=[150, 60]),
    )
    refused("asset.manifest_invalid", lambda: load_asset(directory))


def test_the_admission_tool_reads_what_loading_refuses_as_unadmitted(tmp_path):
    directory = edited(copied(tmp_path, "dial"), lambda d: d.pop("checks"))
    refused("asset.unadmitted", lambda: load_asset(directory))
    assert read_asset(directory).name == "dial"
    assert admit(directory).passed


# What loading checks of an asset's geometry, before any rig is made


def test_loading_refuses_an_outline_anchor_off_its_outline(tmp_path):
    directory = edited(
        copied(tmp_path, "lidded-box"),
        lambda d: d["anchors"][0].update(at=[200, 109]),
    )
    found = refused("asset.manifest_invalid", lambda: load_asset(directory))
    assert found.detail["member"] == "anchors[0].at"
    refused("asset.manifest_invalid", lambda: read_asset(directory))


def test_loading_refuses_an_outline_anchor_on_a_slot(tmp_path):
    directory = edited(
        copied(tmp_path, "figure"),
        lambda d: d["anchors"][0].update(attach="outline"),
    )
    found = refused("asset.manifest_invalid", lambda: load_asset(directory))
    assert found.detail["member"] == "anchors[0].attach"


def test_loading_refuses_a_name_a_pose_could_not_tell_apart(tmp_path):
    def reserved(document: dict) -> None:
        document["parameters"][0]["name"] = "run_time"

    def shared(document: dict) -> None:
        document["slots"][0]["name"] = "reading"

    for edit, member in ((reserved, "parameters[0].name"), (shared, "slots[0].name")):
        directory = edited(copied(tmp_path / edit.__name__, "dial"), edit)
        found = refused("asset.manifest_invalid", lambda d=directory: load_asset(d))
        assert found.detail["member"] == member


# At the server

SCENE = """\
from arachne import *


class Stage(Scene):
    def construct(self):
        rig = Rig({asset!r})  # @asset
        self.add(rig)
        self.wait(0.5)
"""


def test_one_changed_byte_after_admission_is_refused_at_the_server(tmp_path):
    directory = edited(copied(tmp_path, "gate-valve"), lambda d: d.pop("checks"))
    refused("asset.unadmitted", lambda: load_asset(directory))
    assert main(["admit", str(directory), "--write"]) == 0
    load_asset(directory)
    svg = directory / "slots" / "state-open.svg"
    data = bytearray(svg.read_bytes())
    data[data.index(b"#27ae60") + 1] = ord("3")
    svg.write_bytes(bytes(data))

    scene = tmp_path / "stage.py"
    scene.write_text(textwrap.dedent(SCENE.format(asset=str(directory))))
    line = next(
        number
        for number, text in enumerate(scene.read_text().splitlines(), start=1)
        if text.endswith("# @asset")
    )
    server = Dispatcher()
    config = {
        "pixel_width": 96,
        "pixel_height": 54,
        "frame_rate": 4,
        "media_dir": str(tmp_path / "media"),
    }
    with pytest.raises(RpcError) as answer:
        server.handle("open", {"file": str(scene), "scene": "Stage", "config": config})
    assert answer.value.code == REFUSED
    assert answer.value.data["reason"] == "asset.unadmitted"
    finding = answer.value.data["finding"]
    assert finding["line"] == line
    assert finding["detail"]["reason"] == "digest"
    assert server.handle("ping", {}) == {"generation": None}


def test_an_unadmitted_asset_raises_a_finding_not_a_plain_error(tmp_path):
    directory = shutil.copytree(DEFECTS / "loose-part", tmp_path / "loose-part")
    with pytest.raises(RaisedFinding) as raised:
        Rig(directory)
    assert str(raised.value).startswith("asset.unadmitted: ")
