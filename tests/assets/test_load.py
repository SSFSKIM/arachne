"""Loading an asset through the asset API: every part of every fixture asset
is a vector object reached by name, at its canvas place, in drawing order, at
the size its manifest declares; the manifest is read as written and its
digest is the asset's.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import subprocess
import sys
import textwrap
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import arachne
from arachne import VMobject, load_asset, tempconfig
from arachne.assets import Asset, Rig
from arachne.assets.manifest import digest, thawed
from arachne.assets.pose import CARRIER_WIDTH
from arachne.assets.reader import FIXTURES
from arachne.timeline.frozen import engine_class

from . import draw_fixture_pack
from .helpers import copied, edited, refused

SVG = "{http://www.w3.org/2000/svg}"


def _expected_points(path: ET.Element) -> np.ndarray:
    """The engine points of a fixture path, read from its ``d`` by this
    test: each cubic as its start, two handles and end.
    """
    tokens = path.get("d", "").split()
    points: list[list[float]] = []
    pen: list[float] = []
    i = 0
    while i < len(tokens):
        command = tokens[i]
        if command == "M":
            pen = [float(tokens[i + 1]), float(tokens[i + 2])]
            i += 3
        elif command == "C":
            numbers = [float(t) for t in tokens[i + 1 : i + 7]]
            curve = [pen, numbers[0:2], numbers[2:4], numbers[4:6]]
            points += [[x, y, 0.0] for x, y in curve]
            pen = numbers[4:6]
            i += 7
        else:
            assert command == "Z"
            i += 1
    return np.array(points)


def _paths(asset: Path, file: str, element: str | None) -> list[ET.Element]:
    root = ET.parse(asset / file).getroot()
    if element is not None:
        root = next(g for g in root.iter(f"{SVG}g") if g.get("id") == element)
    return list(root.iter(f"{SVG}path"))


@pytest.mark.parametrize("name", draw_fixture_pack.NAMES)
def test_every_part_is_a_vector_object_at_its_canvas_place(name):
    rig = Rig(f"fixture:{name}")
    document = json.loads((FIXTURES / name / "manifest.json").read_text())
    scale = document["size"]["scene_units_per_unit"]
    ox, oy = document["size"]["origin"]
    entries = [
        (p["name"], p["file"], p.get("element"), p["order"]) for p in document["parts"]
    ]
    entries += [
        (
            s["name"],
            next(d["file"] for d in s["drawings"] if d["name"] == s["default"]),
            None,
            s["order"],
        )
        for s in document["slots"]
    ]
    drawn = [name for name, *_, order in sorted(entries, key=lambda e: e[3])]
    # The carrier comes first: where the rig is placed, not a part.
    assert [member.name for member in rig.submobjects] == ["carrier", *drawn]
    slots = {s["name"]: s["default"] for s in document["slots"]}
    for part_name, file, element, _ in entries:
        part = rig.part(part_name)
        assert isinstance(part, VMobject)
        if part_name in slots:
            # A slot holds a group per drawing; the default one is drawn.
            part = next(d for d in part.submobjects if d.name == slots[part_name])
        shapes = part.submobjects
        paths = _paths(FIXTURES / name, file, element)
        assert len(shapes) == len(paths)
        for shape, path in zip(shapes, paths, strict=True):
            assert isinstance(shape, VMobject)
            expected = _expected_points(path)
            expected[:, 0] = (expected[:, 0] - ox) * scale
            expected[:, 1] = (oy - expected[:, 1]) * scale
            np.testing.assert_allclose(shape.points, expected, atol=1e-9)


@pytest.mark.parametrize("name", draw_fixture_pack.NAMES)
def test_a_rig_at_rest_has_the_size_its_manifest_declares(name):
    rig = Rig(f"fixture:{name}")
    size = rig.asset.manifest.size
    assert rig.width == pytest.approx(size.width, abs=1e-3)
    assert rig.height == pytest.approx(size.height, abs=1e-3)


def test_a_part_is_painted_as_its_file_says():
    lid = Rig("fixture:lidded-box").part("lid")
    board, knob = lid.submobjects
    assert board.get_fill_color().to_hex().lower() == "#b48a50"
    assert board.get_fill_opacity() == 1
    assert board.get_stroke_color().to_hex().lower() == "#4a3426"
    assert board.get_stroke_width() == 3
    assert knob.get_stroke_width() == 0


def test_an_unknown_part_name_is_a_finding_with_the_nearest_names():
    rig = Rig("fixture:figure")
    found = refused("rig.name_unknown", lambda: rig.part("hed"))
    assert found.detail["nearest"] == ["head"]
    assert found.detail["kind"] == "part"


def test_rig_is_an_engine_class():
    assert engine_class(Rig)
    assert engine_class(Asset)


def test_importing_the_engine_reads_no_asset_and_builds_none():
    program = textwrap.dedent(
        """
        import gc, sys
        opened = []
        sys.addaudithook(
            lambda event, args: opened.append(str(args[0])) if event == "open" else None
        )
        import arachne
        from arachne.assets import Asset
        print(sum("fixtures" in path for path in opened))
        print(sum(isinstance(found, Asset) for found in gc.get_objects()))
        """
    )
    root = Path(arachne.__file__).parent.parent
    out = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    ).stdout.split()
    assert out == ["0", "0"]


def test_an_asset_is_named_by_its_directory_its_manifest_or_under_the_assets_dir(
    tmp_path,
):
    directory = copied(tmp_path, "dial")
    by_directory = load_asset(directory)
    assert load_asset(directory / "manifest.json").digest == by_directory.digest
    assert load_asset(str(directory)).digest == by_directory.digest
    with tempconfig({"assets_dir": str(tmp_path)}):
        assert load_asset("dial").digest == by_directory.digest
    assert load_asset("fixture:dial").digest == by_directory.digest
    # A loaded asset is taken as it is, and a copy of it is itself.
    assert load_asset(by_directory) is by_directory
    rig = Rig(by_directory)
    assert rig.copy().asset is by_directory


def test_the_reach_of_the_figure_binds_each_drivers_keys_to_its_part(tmp_path):
    draw_fixture_pack.draw(tmp_path)
    for asset in (load_asset(tmp_path / "figure"), load_asset("fixture:figure")):
        reach = asset.parameters["reach"]
        assert (reach.min, reach.max, reach.default, reach.unit) == (0, 1, 0, "1")
        arm, hand = reach.drives
        assert (arm.part, arm.kind, arm.unit) == ("upper_arm", "rotate", "deg")
        assert [(k.at, k.value) for k in arm.keys] == [(0, 0), (1, 70)]
        assert (hand.part, hand.kind) == ("hand", "translate")
        assert [(k.at, k.value) for k in hand.keys] == [(0, (0, 0)), (1, (12, -4))]
        assert "upper_arm" in asset.parts
        assert asset.slots["hand"].parent == "lower_arm"


def test_the_layered_sections_anchors_and_stored_outlines_are_read():
    asset = load_asset("fixture:layered-section")
    (swell,) = asset.parameters["swell"].drives
    assert swell.kind == "shape"
    assert [k.at for k in swell.keys] == [0, 0.5, 1]
    tip, granule = asset.anchors["tip"], asset.anchors["granule"]
    assert (tip.attach, tip.at, tip.keys) == ("outline", (290, 150), ())
    assert granule.attach == "interior"
    assert [k.at for k in granule.keys] == [0, 0.5, 1]
    for key in swell.keys:
        outline = asset.outline(key.value)
        assert [len(s.points) for s in outline] == [
            len(s.points) for s in asset.rest("inner_layer")
        ]


def _second_digest(directory: Path) -> str:
    """The asset's digest, computed by this test from the files as they are:
    every value of a ``file`` member anywhere in the manifest names a file.
    """
    document = json.loads((directory / "manifest.json").read_text())
    files: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "file":
                    files.add(posixpath.normpath(item))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(document)
    hash = hashlib.sha256()
    for name in sorted(files):
        data = (directory / name).read_bytes()
        hash.update(name.encode() + b"\0" + len(data).to_bytes(8, "big") + data)
    document.pop("checks", None)
    hash.update(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )
    return "sha256:" + hash.hexdigest()


@pytest.mark.parametrize("name", ["figure", "layered-section"])
def test_the_digest_is_a_second_implementations(name):
    assert load_asset(f"fixture:{name}").digest == _second_digest(FIXTURES / name)


def test_the_digest_attests_every_file_and_the_manifest_but_not_its_checks(tmp_path):
    directory = copied(tmp_path, "figure")
    before = load_asset(directory).digest
    edited(directory, lambda d: d.update(checks={"tool": "arachne.assets 1"}))
    assert load_asset(directory).digest == before
    edited(directory, lambda d: d.update(version=2))
    changed = load_asset(directory).digest
    assert changed != before
    head = directory / "pieces" / "head.svg"
    head.write_text(head.read_text().replace('stroke-width="2"', 'stroke-width="3"'))
    assert load_asset(directory).digest not in (before, changed)


def test_an_unknown_minor_version_with_nothing_required_loads(tmp_path):
    directory = edited(
        copied(tmp_path, "dial"),
        lambda d: d.update(manifest_version="1.7", newer_member={"kept": True}),
    )
    asset = load_asset(directory)
    assert asset.manifest.manifest_version == "1.7"
    assert asset.manifest.document["newer_member"] == {"kept": True}


@pytest.mark.parametrize(
    ("edit", "member"),
    [
        (lambda d: d.update(manifest_version="2.0"), "manifest_version"),
        (lambda d: d.update(manifest_version="one"), "manifest_version"),
        (lambda d: d.update(requires=["mouth_sync"]), "requires"),
        (
            lambda d: d.update(manifest_version="1.7", requires=["mouth_sync"]),
            "requires",
        ),
        (lambda d: d.update(kind="clip"), "kind"),
    ],
)
def test_a_version_feature_or_kind_the_reader_does_not_know_is_refused(
    tmp_path, edit, member
):
    directory = edited(copied(tmp_path, "dial"), edit)
    found = refused("asset.manifest_invalid", lambda: load_asset(directory))
    assert found.detail["member"] == member


#: A small asset whose manifest exercises path normalisation, strings that
#: are not ASCII, and numbers of every JSON form: a rod, one line long.
ROD_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 100">
  <path d="M 0 50 L 100 50" fill="none" stroke="#3b4a5c" stroke-width="3"/>
</svg>
"""
ROD_MANIFEST = """\
{
  "manifest_version": "1.0",
  "name": "rod",
  "version": 1,
  "kind": "parts",
  "size": {"scene_units_per_unit": 0.01, "origin": [50, 50], "width": 1.0, "height": 0},
  "parts": [
    {"name": "rod", "file": "./pieces//rod.svg", "parent": null, "pivot": [0, 50], "order": 0}
  ],
  "slots": [
    {"name": "end", "parent": "rod", "pivot": [100, 50], "order": 1, "default": "plain",
     "drawings": [{"name": "plain", "file": "pieces/rod.svg"}]}
  ],
  "parameters": [
    {"name": "slide", "min": 0, "max": 2.50, "default": 1e-7, "unit": "u/s",
     "drives": [{"part": "rod", "kind": "translate",
                 "keys": [{"at": 0, "value": [0, 0]}, {"at": 2.5, "value": [0.1, 12345678901]}]}]},
    {"name": "tilt", "min": -0.0, "max": 1, "default": 0,
     "drives": [{"part": "rod", "kind": "rotate", "unit": "rad",
                 "keys": [{"at": 0, "value": 0}, {"at": 1, "value": 3.14159}]}]}
  ],
  "provenance": {"source_url": null, "author": "Zo\u00eb, \u753b\u5bb6", "license": "none",
                 "credit_line": "\u00a9 \u2014 a test", "retrieved_at": null}
}
"""
#: Its digest, as the second implementation above computed it.
ROD_DIGEST = "sha256:7d98ab27d051165499a7d7e9b0e3037f97a70f6d1a17b57e6d3dbe4bcb87dfcd"


def _rod(tmp_path: Path) -> Path:
    directory = tmp_path / "rod"
    (directory / "pieces").mkdir(parents=True)
    (directory / "pieces" / "rod.svg").write_text(ROD_SVG)
    (directory / "manifest.json").write_text(ROD_MANIFEST)
    return directory


def test_the_digest_of_a_fixed_asset_is_pinned(tmp_path):
    directory = _rod(tmp_path)
    assert _second_digest(directory) == ROD_DIGEST
    assert load_asset(directory).digest == ROD_DIGEST


def test_an_asset_may_be_a_line_with_one_extent_zero(tmp_path):
    rig = Rig(_rod(tmp_path))
    assert rig.width == pytest.approx(1.0)
    # Its carrier, which says where it is placed, spans 10**-4 units.
    assert rig.height == pytest.approx(0, abs=CARRIER_WIDTH)


def test_a_parameter_with_no_unit_is_a_plain_number(tmp_path):
    asset = load_asset(_rod(tmp_path))
    assert asset.parameters["slide"].unit == "u/s"
    assert asset.parameters["tilt"].unit == "1"


def test_an_asset_shared_by_its_rigs_cannot_be_written(tmp_path):
    rig = Rig(_rod(tmp_path))
    asset = rig.copy().asset
    assert asset is rig.asset
    with pytest.raises(AttributeError):
        asset.digest = "sha256:0"
    with pytest.raises(AttributeError):
        asset.manifest.name = "other"
    document = asset.manifest.document
    with pytest.raises(TypeError):
        document["name"] = "other"
    with pytest.raises(TypeError):
        document["size"]["width"] = 2
    assert isinstance(document["parts"], tuple)
    with pytest.raises(TypeError):
        asset.manifest.provenance["license"] = "any"
    with pytest.raises(TypeError):
        asset.parameters["slide"] = None
    with pytest.raises(AttributeError):
        asset.parameters["slide"].drives[0].keys[1].value.append(0)
    asset.rest("rod")[0].shift([1, 0, 0])
    assert asset.rest("rod")[0].get_start()[0] == 0
    # A detached copy of the manifest is plain JSON, its digest unchanged.
    plain = thawed(document)
    plain["name"] = "other"
    assert asset.manifest.document["name"] == "rod"
    files = {"pieces/rod.svg": (asset.directory / "pieces" / "rod.svg").read_bytes()}
    assert digest(asset.manifest.document, files) == asset.digest
