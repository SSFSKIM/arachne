"""Admission by geometry: an asset drawn at the poses its maker tested is
admitted or refused by its shape alone, and the tool's output can be read by
a maker's script.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arachne import DL, UR
from arachne.assets import Rig, admit, load_asset
from arachne.assets.__main__ import main
from arachne.assets.admission import HOLE_ALLOWANCE, REACH, write_checks
from arachne.assets.reader import CHECKS, FIXTURES, TOOL, read_asset

from . import draw_fixture_pack
from .helpers import copied, edited

DEFECTS = Path(__file__).parent / "defects"
DIGEST = r"sha256:[0-9a-f]{64}"


@pytest.mark.parametrize("name", draw_fixture_pack.NAMES)
def test_every_fixture_passes_and_records_its_passing_checks(name):
    report = admit(f"fixture:{name}")
    assert report.passed, report.lines()
    assert [result.name for result in report.results] == list(CHECKS)
    committed = json.loads((FIXTURES / name / "manifest.json").read_text())
    assert committed["checks"] == report.checks()
    assert report.checks()["tool"] == TOOL
    assert report.checks()["digest"] == load_asset(f"fixture:{name}").digest


@pytest.mark.parametrize(("defect", "check"), draw_fixture_pack.DEFECTS.items())
def test_each_planted_defect_fails_its_own_check_and_no_other(defect, check):
    report = admit(DEFECTS / defect)
    failed = [result.name for result in report.results if not result.passed]
    assert failed == [check], report.lines()


def test_a_seam_opened_inside_the_figure_is_a_hole_at_the_pose_that_opens_it():
    hole = admit(DEFECTS / "open-seam").result("hole")
    assert hole.pose == "slide=1"
    assert hole.worst > 10 * HOLE_ALLOWANCE
    assert hole.line() == (
        f"FAIL hole: {hole.worst:g} px open inside the figure at slide=1"
        f" (allowed {HOLE_ALLOWANCE})"
    )


def test_a_hole_the_figure_is_drawn_with_is_not_one_it_opens(tmp_path):
    # The pane rests slid aside: the seam is open at the default pose, and
    # every other pose opens less of it.
    directory = edited(
        _defect(tmp_path, "open-seam"),
        lambda d: d["parameters"][0].update(default=1),
    )
    report = admit(directory)
    assert report.passed, report.lines()


#: A window frame's four bars, on a canvas 300 units square.
_BARS = [(50, 50, 200, 20), (50, 230, 200, 20), (50, 70, 20, 160), (230, 70, 20, 160)]


def _windowed(root: Path, name: str, plate: list, shift: list) -> Path:
    """A frame with a plate behind it that the parameter ``slide`` moves
    by ``shift``; the plate is the rectangles ``plate``.
    """
    rect, Shape = draw_fixture_pack.rect, draw_fixture_pack.Shape
    pack = draw_fixture_pack.Pack(root, name, (300, 300))
    frame = pack.file("frame.svg", {None: [Shape(rect(*b), "#6d4c41") for b in _BARS]})
    behind = pack.file("plate.svg", {None: [Shape(rect(*r), "#90a4ae") for r in plate]})
    pack.manifest(
        (150, 150),
        parts=[
            draw_fixture_pack.part("frame", frame, None, (150, 150), 1),
            draw_fixture_pack.part("plate", behind, "frame", (150, 150), 0),
        ],
        slots=[],
        parameters=[
            draw_fixture_pack.one_parameter(
                "slide", 0, 1, "plate", "translate", [[0, 0], shift]
            )
        ],
        anchors=[],
    )
    return root / name


#: A plate covering the window but for a square opening of its own, whose
#: edges fall partway into pixels: moved, it can gain a row and a column.
_HOLED_PLATE = [
    (60, 60, 180, 60.274),
    (60, 140.274, 180, 99.726),
    (60, 120, 70.274, 20.274),
    (150.274, 120, 89.726, 20.274),
]


@pytest.mark.parametrize(
    "shift", [[-9, -3.7], [-3.1, 4.1], [3.3, 2.2], [6.1, -3.7], [9, 4.1]]
)
def test_an_opening_that_only_moves_is_no_hole(tmp_path, shift):
    # Wherever its edges fall between pixels, a moved opening is no new one,
    # though it may enclose a row and a column more clear pixels.
    directory = _windowed(tmp_path, "moving", _HOLED_PLATE, shift)
    hole = admit(directory).result("hole")
    assert hole.passed, hole.failure
    assert hole.worst > HOLE_ALLOWANCE


def test_a_moved_opening_that_also_opens_a_seam_is_a_hole(tmp_path):
    # Slid past the frame's edge, the plate opens a seam beside its own.
    directory = _windowed(tmp_path, "slid", _HOLED_PLATE, [13.1, 0])
    hole = admit(directory).result("hole")
    assert not hole.passed
    assert hole.pose == "slide=1"


def test_a_seam_opening_while_another_opening_narrows_is_a_hole(tmp_path):
    # Two windows: the left pane slides to narrow the left window's opening
    # by more than the right pane's slide opens a seam in the right window.
    rect, Shape = draw_fixture_pack.rect, draw_fixture_pack.Shape
    pack = draw_fixture_pack.Pack(tmp_path, "two-windows", (300, 200))
    bars = [(20, 20, 260, 20), (20, 160, 260, 20), (20, 40, 20, 120)]
    bars += [(140, 40, 20, 120), (260, 40, 20, 120)]
    frame = pack.file(
        "frame.svg", {None: [Shape(rect(*b), "#6d4c41", "#3e2723") for b in bars]}
    )
    left = pack.file("left.svg", {None: [Shape(rect(70, 36, 89, 128), "#b3e5fc")]})
    right = pack.file("right.svg", {None: [Shape(rect(156, 36, 108, 128), "#b3e5fc")]})
    slide = draw_fixture_pack.one_parameter(
        "shift", 0, 1, "left", "translate", [[0, 0], [-18, 0]]
    )
    slide["drives"].append(
        {
            "part": "right",
            "kind": "translate",
            "keys": draw_fixture_pack.keys((0, [0, 0]), (1, [8, 0])),
        }
    )
    pack.manifest(
        (150, 100),
        parts=[
            draw_fixture_pack.part("frame", frame, None, (150, 100), 2),
            draw_fixture_pack.part("left", left, "frame", (150, 100), 0),
            draw_fixture_pack.part("right", right, "frame", (150, 100), 1),
        ],
        slots=[],
        parameters=[slide],
        anchors=[],
    )
    hole = admit(tmp_path / "two-windows").result("hole")
    # The enclosed area only shrinks, so the area alone admits it.
    assert hole.worst == 0
    assert not hole.passed
    assert hole.pose == "shift=1"
    assert hole.failure == (
        f"2 openings of more than {HOLE_ALLOWANCE} px inside the figure at"
        " shift=1, where the default pose has 1"
    )


#: A spike whose apex meets at 12 degrees, stroked 8 wide with miter joins:
#: the miter reaches 1 / sin(6 degrees), about 9.6 half-widths, past it.
_SPIKE = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">
  <polygon points="150,40 139.49,140 160.51,140" fill="#455a64" stroke="#263238"
    stroke-width="8" stroke-linejoin="miter" stroke-miterlimit="10"/>
</svg>
"""


def test_a_thick_acute_miter_at_the_drawings_edge_is_inside_every_mask(tmp_path):
    from arachne.assets import admission
    from arachne.assets.rig import unadmitted_rig

    directory = tmp_path / "spike"
    directory.mkdir()
    (directory / "spike.svg").write_text(_SPIKE)
    document = {
        "manifest_version": "1.0",
        "name": "spike",
        "version": 1,
        "kind": "parts",
        "size": {
            "scene_units_per_unit": 0.01,
            "origin": [150, 90],
            "width": 1,
            "height": 1,
        },
        "parts": [
            {
                "name": "spike",
                "file": "spike.svg",
                "parent": None,
                "pivot": [150, 140],
                "order": 0,
            }
        ],
        "parameters": [
            draw_fixture_pack.one_parameter(
                "tilt", 0, 90, "spike", "rotate", [0, 90], unit="deg"
            )
        ],
        "provenance": draw_fixture_pack.PROVENANCE,
    }
    (directory / "manifest.json").write_text(json.dumps(document))
    rig = unadmitted_rig(read_asset(directory))
    poses = list(admission._poses(rig))
    grid = admission._Grid(rig, poses)
    margin = admission.MASK_MARGIN
    for pose in poses:
        figure = grid.masks(rig, pose).figure
        assert figure.any()
        for band in (
            figure[:margin],
            figure[-margin:],
            figure[:, :margin],
            figure[:, -margin:],
        ):
            assert not band.any(), pose.label
    assert admit(directory).passed


def test_a_part_pulled_clear_of_its_parent_is_detached_at_each_pose_past_reach():
    detached = admit(DEFECTS / "loose-part").result("detached_part")
    assert detached.pose == "pull=1"
    assert detached.worst > 10 * REACH
    assert detached.failure.startswith(
        "handle no longer meets its parent body at pull=1"
    )


def test_a_part_drawn_apart_from_its_parent_at_rest_is_never_found_to_leave_it():
    # The gate valve's state flag floats beside the body at every pose.
    rig = Rig("fixture:gate-valve")
    assert rig.model.placed["state"].parent == "body"
    flag = rig.part("state").family_members_with_points()
    for shape in rig.part("body").submobjects:
        assert all(_apart(shape, other) for other in flag)
    assert admit("fixture:gate-valve").result("detached_part").passed


def _apart(a, b) -> bool:
    """Whether the boxes of ``a`` and ``b`` are apart."""
    return bool(
        np.any(a.get_corner(DL)[:2] > b.get_corner(UR)[:2])
        or np.any(b.get_corner(DL)[:2] > a.get_corner(UR)[:2])
    )


def test_an_anchor_named_on_the_wrong_part_is_off_it_at_the_default_pose():
    anchor = admit(DEFECTS / "stray-anchor").result("anchor_off_part")
    assert anchor.pose == "the default pose"
    assert anchor.failure.startswith("knob lies")
    assert anchor.failure.endswith(
        f"off its part body at the default pose (allowed {REACH})"
    )


def test_an_anchor_off_its_part_at_one_pose_alone_is_found_there(tmp_path):
    # The granule's keyed place at a full swell is put outside the layer.
    directory = edited(
        copied(tmp_path, "layered-section"),
        lambda d: d["anchors"][1]["keys"][2].update(point=[330, 150]),
    )
    anchor = admit(directory).result("anchor_off_part")
    assert not anchor.passed
    assert anchor.pose == "swell=1"


def test_an_anchor_off_its_part_at_rest_is_found_at_the_default_pose(tmp_path):
    directory = edited(
        copied(tmp_path, "figure"),
        lambda d: d["anchors"][2].update(at=[20, 20]),
    )
    anchor = admit(directory).result("anchor_off_part")
    assert anchor.failure.startswith("chest lies")
    assert anchor.pose == "the default pose"


def test_every_parameter_is_drawn_at_its_extremes_and_middle_and_every_drawing_shown(
    monkeypatch,
):
    from arachne.assets import admission

    drawn: list[str] = []
    masks = admission._Grid.masks

    def recorded(self, rig, pose):
        drawn.append(pose.label)
        return masks(self, rig, pose)

    monkeypatch.setattr(admission._Grid, "masks", recorded)
    admit("fixture:figure")
    assert drawn == [
        "the default pose",
        "reach=0",
        "reach=0.5",
        "reach=1",
        "head_turn=-25",
        "head_turn=0",
        "head_turn=25",
        "eyes=closed",
        "hand=point",
    ]


def test_the_figures_eyes_follow_its_head_as_it_turns():
    for turn in (-25, 25):
        figure = Rig("fixture:figure")
        eyes = figure.part("eyes").get_center()
        head = figure.part("head").get_center()
        figure.pose(head_turn=turn)
        moved_eyes = figure.part("eyes").get_center() - eyes
        moved_head = figure.part("head").get_center() - head
        assert moved_eyes[0] == pytest.approx(np.sign(turn) * 6.667 * 0.01)
        # The head's points at its middle move furthest; the eyes, off its
        # middle, move less, the same way.
        assert np.sign(moved_eyes[0]) == np.sign(moved_head[0])
        assert moved_eyes[1] == pytest.approx(0)


# The tool's output


def _run(capsys, *argv: str) -> tuple[int, list[str]]:
    status = main(["admit", *argv])
    return status, capsys.readouterr().out.splitlines()


def test_the_tool_prints_its_fixed_lines_and_fails_a_defect(capsys):
    status, lines = _run(capsys, str(DEFECTS / "open-seam"))
    assert status == 1
    assert re.fullmatch(rf"asset: open-seam 1 \(parts\) {DIGEST}", lines[0])
    assert lines[1:3] == ["PASS anchor_off_part", "PASS detached_part"]
    assert re.fullmatch(
        rf"FAIL hole: \d+ px open inside the figure at slide=1 \(allowed {HOLE_ALLOWANCE}\)",
        lines[3],
    )
    assert lines[4:] == ["admission: 2/3 pass"]


def test_the_tool_passes_an_admissible_asset(tmp_path, capsys):
    status, lines = _run(capsys, str(copied(tmp_path, "dial")))
    assert status == 0
    assert lines[1:] == [
        "PASS anchor_off_part",
        "PASS detached_part",
        "PASS hole",
        "admission: 3/3 pass",
    ]


def test_the_tool_refuses_an_unreadable_asset_with_its_finding(tmp_path, capsys):
    directory = copied(tmp_path, "dial")
    (directory / "manifest.json").unlink()
    assert main(["admit", str(directory)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("asset.manifest_missing: ")


def test_the_tool_runs_as_a_module():
    done = subprocess.run(
        [sys.executable, "-m", "arachne.assets", "admit", str(DEFECTS / "loose-part")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1, done.stderr
    assert done.stdout.splitlines()[-1] == "admission: 2/3 pass"


def test_written_checks_admit_an_asset_and_attest_its_digest(tmp_path, capsys):
    # An asset changed by its maker carries a stale record until admitted
    # again.
    directory = edited(copied(tmp_path, "lidded-box"), lambda d: d.update(version=2))
    status, lines = _run(capsys, str(directory), "--write")
    assert status == 0
    assert lines[-1] == f"written: {directory / 'manifest.json'}"
    asset = load_asset(directory)
    document = json.loads((directory / "manifest.json").read_text())
    assert document["checks"] == {
        "tool": TOOL,
        "digest": asset.digest,
        "results": [{"name": name, "pass": True} for name in CHECKS],
    }
    assert list(document)[-2:] == ["checks", "provenance"]
    assert asset.version == 2


def test_a_failing_asset_written_records_its_failure_and_does_not_load(
    tmp_path, capsys
):
    directory = _defect(tmp_path, "loose-part")
    status, _ = _run(capsys, str(directory), "--write")
    assert status == 1
    results = json.loads((directory / "manifest.json").read_text())["checks"]["results"]
    assert {r["name"]: r["pass"] for r in results} == {
        "hole": True,
        "detached_part": False,
        "anchor_off_part": True,
    }
    assert read_asset(directory).digest == admit(directory).digest


def test_a_written_manifest_replaces_the_old_only_once_written_whole(
    tmp_path, monkeypatch
):
    from arachne.assets import admission

    directory = edited(
        copied(tmp_path, "dial"),
        lambda d: d["provenance"].update(author="Zo\u00eb, \u753b\u5bb6"),
    )
    before = (directory / "manifest.json").read_bytes()
    report = admit(directory)

    def interrupted(source, target):
        raise OSError("the disk is full")

    monkeypatch.setattr(admission.os, "replace", interrupted)
    with pytest.raises(OSError):
        write_checks(directory, report)
    assert (directory / "manifest.json").read_bytes() == before
    assert sorted(p.name for p in directory.iterdir()) == [
        "manifest.json",
        "pieces",
        "slots",
    ]
    monkeypatch.undo()
    write_checks(directory, report)
    assert sorted(p.name for p in directory.iterdir()) == [
        "manifest.json",
        "pieces",
        "slots",
    ]
    written = (directory / "manifest.json").read_bytes().decode("utf-8")
    assert "Zo\u00eb, \u753b\u5bb6" in written
    assert (
        load_asset(directory).manifest.provenance["author"] == "Zo\u00eb, \u753b\u5bb6"
    )


def _defect(tmp_path: Path, name: str) -> Path:
    """The planted defect ``name``, copied to ``tmp_path``."""
    target = tmp_path / name
    shutil.copytree(DEFECTS / name, target)
    return target
