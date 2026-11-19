"""Every way an asset asked for through the asset API can be unusable, short
of admission, is a finding of its own code, raised before anything is drawn:
never an empty space on screen.
"""

from __future__ import annotations

import os
import shutil

import pytest

from arachne import load_asset
from arachne.assets import Rig

from .helpers import BROKEN, copied, edited, refused


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    (directory / "manifest.json").unlink()
    found = refused("asset.manifest_missing", lambda: Rig(directory))
    assert found.detail["searched"] == [str(directory / "manifest.json")]


def test_a_part_file_alone_is_no_asset(tmp_path):
    directory = copied(tmp_path, "dial")
    refused("asset.manifest_missing", lambda: Rig(directory / "pieces" / "needle.svg"))


def test_a_path_that_is_nothing_is_refused_with_where_it_was_looked_for(tmp_path):
    found = refused("asset.manifest_missing", lambda: load_asset(tmp_path / "nowhere"))
    assert str(tmp_path / "nowhere") in found.detail["searched"]
    assert len(found.detail["searched"]) == 2


def test_an_unknown_fixture_is_refused_naming_those_there_are():
    found = refused("asset.manifest_missing", lambda: load_asset("fixture:teapot"))
    assert "lidded-box" in str(found)
    refused("asset.manifest_missing", lambda: load_asset("fixture:../dial"))


def test_a_file_the_manifest_names_that_is_not_there_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    (directory / "slots" / "lamp-on.svg").unlink()
    found = refused("asset.file_missing", lambda: Rig(directory))
    assert found.detail["file"] == "slots/lamp-on.svg"


@pytest.mark.parametrize(
    ("broken", "says"),
    [
        ("text.svg", "<text>"),
        ("image.svg", "<image>"),
        ("gradient.svg", "Gradient"),
        ("pattern-paint.svg", "gradient or pattern"),
        ("clip-path.svg", "clip-path"),
        ("mask.svg", "mask"),
        ("filter.svg", "filter"),
        ("other-file.svg", "other.svg#needle"),
        ("not-xml.svg", "XML"),
        ("draws-nothing.svg", "draws nothing"),
        ("dashed.svg", "stroke-dasharray"),
        ("dash-style.svg", "stroke-dashoffset"),
        ("round-join.svg", "stroke-linejoin"),
        ("even-odd.svg", "fill-rule"),
        ("group-opacity.svg", "opacity"),
        ("hidden.svg", "display"),
        ("truncated-path.svg", "does not parse"),
        ("unknown-command.svg", "not SVG path data"),
        ("infinite-path.svg", "not finite"),
        ("dangling-use.svg", "#missing"),
        ("percent-rect.svg", "percentage"),
        ("unit-length.svg", "plain number"),
        ("unknown-colour.svg", "notacolour"),
        ("bad-transform.svg", "transform"),
        ("unknown-attribute.svg", "pathLength"),
    ],
)
def test_a_file_the_reader_cannot_draw_exactly_is_refused(tmp_path, broken, says):
    directory = copied(tmp_path, "dial")
    shutil.copy(BROKEN / broken, directory / "pieces" / "needle.svg")
    found = refused("asset.unreadable", lambda: Rig(directory))
    assert found.detail["file"] == "pieces/needle.svg"
    assert says in found.detail["reason"]


def test_reading_a_file_writes_nothing_beside_it(tmp_path):
    directory = copied(tmp_path, "figure")
    before = sorted(p.relative_to(directory) for p in directory.rglob("*"))
    Rig(directory)
    assert sorted(p.relative_to(directory) for p in directory.rglob("*")) == before


def test_part_files_on_different_canvases_are_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    shutil.copy(BROKEN / "other-canvas.svg", directory / "pieces" / "needle.svg")
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert "canvas" in found.detail["reason"]


@pytest.mark.parametrize(
    "file",
    [
        "https://example.com/needle.svg",
        "file:///tmp/needle.svg",
        "/tmp/needle.svg",
        "../needle.svg",
        "pieces/../../needle.svg",
    ],
)
def test_a_file_named_by_an_address_or_outside_the_asset_is_refused(tmp_path, file):
    directory = edited(
        copied(tmp_path, "dial"), lambda d: d["parts"][1].update(file=file)
    )
    shutil.copy(directory / "pieces" / "needle.svg", tmp_path / "needle.svg")
    found = refused("asset.remote", lambda: Rig(directory))
    assert found.detail["file"] == file


def test_a_link_out_of_the_asset_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    outside = tmp_path / "needle.svg"
    shutil.move(directory / "pieces" / "needle.svg", outside)
    (directory / "pieces" / "needle.svg").symlink_to(outside)
    refused("asset.remote", lambda: Rig(directory))


def test_an_asset_named_by_an_address_is_refused():
    refused("asset.remote", lambda: Rig("https://example.com/assets/dial"))


def _part(document, name):
    return next(p for p in document["parts"] if p["name"] == name)


@pytest.mark.parametrize(
    ("asset", "edit", "member"),
    [
        ("dial", lambda d: d.pop("provenance"), "provenance"),
        ("dial", lambda d: d["provenance"].pop("license"), "provenance.license"),
        ("dial", lambda d: d.pop("size"), "size"),
        (
            "dial",
            lambda d: d["size"].update(scene_units_per_unit=0),
            "size.scene_units_per_unit",
        ),
        ("dial", lambda d: d["parts"][1].update(parent="dial_face"), "parts[1].parent"),
        ("dial", lambda d: d["parts"][0].update(parent="needle"), "parts[0].parent"),
        ("dial", lambda d: d["parts"][1].update(name="face"), "parts[1].name"),
        ("dial", lambda d: d["slots"][0].update(default="dim"), "slots[0].default"),
        ("dial", lambda d: d["parts"][1].update(element="tip"), "parts[1]"),
        (
            "dial",
            lambda d: d["parameters"][0].update(default=120),
            "parameters[0].default",
        ),
        (
            "dial",
            lambda d: d["parameters"][0]["drives"][0]["keys"].pop(),
            "parameters[0].drives[0].keys",
        ),
        (
            "dial",
            lambda d: d["parameters"][0]["drives"][0]["keys"][1].update(at=90),
            "parameters[0].drives[0].keys",
        ),
        (
            "dial",
            lambda d: d["parameters"][0]["drives"][0].update(unit="turns"),
            "parameters[0].drives[0].unit",
        ),
        (
            "dial",
            lambda d: d["parameters"][0]["drives"][0].update(kind="bend"),
            "parameters[0].drives[0].kind",
        ),
        (
            "dial",
            lambda d: d["parameters"][0]["drives"][0].update(part="lamp", kind="warp"),
            "parameters[0].drives[0].part",
        ),
        ("dial", lambda d: d["anchors"][0].update(part="rim"), "anchors[0].part"),
        ("dial", lambda d: d["anchors"][0].update(attach="glued"), "anchors[0].attach"),
        (
            "figure",
            lambda d: d["parameters"][1]["drives"][0]["keys"][0]["offsets"].pop(),
            "parameters[1].drives[0].keys[0].offsets",
        ),
        (
            "figure",
            lambda d: _part(d, "head").pop("lattice"),
            "parameters[1].drives[0].part",
        ),
        (
            "layered-section",
            lambda d: d["anchors"][1].pop("keys"),
            "anchors[1].keys",
        ),
        (
            "layered-section",
            lambda d: d["anchors"][1]["keys"].pop(1),
            "anchors[1].keys",
        ),
        (
            "layered-section",
            lambda d: d["anchors"][0].update(keys=[]),
            "anchors[0].keys",
        ),
        (
            "layered-section",
            lambda d: d["parameters"].append(
                {**d["parameters"][0], "name": "swell_again"}
            ),
            "parameters[1].drives[0]",
        ),
    ],
)
def test_a_manifest_missing_or_contradicting_a_member_is_refused_naming_it(
    tmp_path, asset, edit, member
):
    directory = edited(copied(tmp_path, asset), edit)
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == member, found.detail["reason"]


def test_a_manifest_that_is_not_json_or_repeats_a_member_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    manifest = directory / "manifest.json"
    text = manifest.read_text()
    manifest.write_text(text[:-3])
    refused("asset.manifest_invalid", lambda: Rig(directory))
    manifest.write_text(text.replace('"version": 1,', '"version": 1, "version": 2,'))
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert "twice" in found.detail["reason"]
    manifest.write_text(text.replace('"version": 1,', '"version": NaN,'))
    refused("asset.manifest_invalid", lambda: Rig(directory))


def test_a_stored_outline_unlike_the_rest_outline_is_refused(tmp_path):
    directory = copied(tmp_path, "layered-section")
    (directory / "shapes" / "inner_layer-1.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">\n'
        '  <path d="M 110 150 L 200 95 L 290 150 L 200 205 Z" fill="#d9a3a3"/>\n'
        "</svg>\n"
    )
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == "parameters[0].drives[0].keys[1].file"
    assert "structure" in found.detail["reason"]


@pytest.mark.parametrize(
    ("edit", "member", "name"),
    [
        (
            lambda d: d["parameters"][0].update(name="lid open"),
            "parameters[0].name",
            "lid open",
        ),
        (
            lambda d: d["parameters"][0].update(name="ångle"),
            "parameters[0].name",
            "ångle",
        ),
        (lambda d: d.update(name="lidded.box"), "name", "lidded.box"),
        (lambda d: d["parts"][1].update(name="lid/top"), "parts[1].name", "lid/top"),
        (lambda d: d["anchors"][0].update(name=""), "anchors[0].name", None),
    ],
)
def test_a_name_outside_ascii_letters_digits_underscore_and_dash_is_refused(
    tmp_path, edit, member, name
):
    directory = edited(copied(tmp_path, "lidded-box"), edit)
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == member
    assert found.detail.get("name") == name


def test_a_name_with_dashes_and_underscores_is_accepted(tmp_path):
    def rename(document):
        document["name"] = "Lidded-Box_2"
        document["parameters"][0]["name"] = "lid-open_deg"
        document["anchors"][0]["name"] = "knob-1"

    asset = load_asset(edited(copied(tmp_path, "lidded-box"), rename))
    assert asset.name == "Lidded-Box_2"
    assert list(asset.parameters) == ["lid-open_deg"]
    assert list(asset.anchors) == ["knob-1"]


ALLOWED = """\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="300mm" height="300mm" viewBox="0 0 300 300" version="1.1">
  <title>needle</title>
  <metadata><anything><text>not drawn</text></anything></metadata>
  <inkscape:grid/>
  <defs><path id="tip" d="M 150 78 L 156 160 L 144 160 Z"/></defs>
  <g id="needle" data-name="needle" inkscape:label="needle" transform="translate(0, 0) rotate(0 150 160)">
    <use href="#tip"/>
    <path d="M 150 160 L 156 160 L 150 172 Z" style="fill: rgb(192, 57, 43); stroke: black; stroke-width: 2px;
          stroke-linejoin: miter; stroke-linecap: butt; opacity: 1" shape-rendering="geometricPrecision"/>
    <circle cx="150px" cy="160" r="3" fill="#333" fill-opacity="0.5" stroke="none" fill-rule="nonzero"/>
  </g>
</svg>
"""


def test_what_the_reader_draws_exactly_is_accepted(tmp_path):
    directory = copied(tmp_path, "dial")
    (directory / "pieces" / "needle.svg").write_text(ALLOWED)
    shapes = Rig(directory).part("needle").submobjects
    assert len(shapes) == 3
    assert shapes[1].get_stroke_width() == 2


@pytest.mark.parametrize(
    "file", ["pieces/../pieces/needle.svg", "pieces/./../pieces/needle.svg"]
)
def test_a_path_with_a_parent_segment_is_refused_even_inside(tmp_path, file):
    directory = edited(
        copied(tmp_path, "dial"), lambda d: d["parts"][1].update(file=file)
    )
    found = refused("asset.remote", lambda: Rig(directory))
    assert found.detail["file"] == file


def test_a_path_through_a_link_inside_the_asset_is_refused(tmp_path):
    """A link inside the asset, then ``..``, names two files by one key; and
    a link to a file of the asset is a second name for it: both refused.
    """
    directory = copied(tmp_path, "dial")
    (directory / "pieces" / "deep").mkdir()
    (directory / "link").symlink_to(directory / "pieces" / "deep")
    edited(directory, lambda d: d["parts"][1].update(file="link/../pieces/needle.svg"))
    refused("asset.remote", lambda: Rig(directory))
    (directory / "pieces" / "alias.svg").symlink_to(directory / "pieces" / "needle.svg")
    edited(directory, lambda d: d["parts"][1].update(file="pieces/alias.svg"))
    found = refused("asset.remote", lambda: Rig(directory))
    assert found.detail["file"] == "pieces/alias.svg"


def test_a_file_that_is_a_directory_is_unreadable(tmp_path):
    directory = copied(tmp_path, "dial")
    (directory / "pieces" / "needle.svg").unlink()
    (directory / "pieces" / "needle.svg").mkdir()
    found = refused("asset.unreadable", lambda: Rig(directory))
    assert found.detail["file"] == "pieces/needle.svg"
    assert "cannot be read" in found.detail["reason"]


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="needs a reader that permissions bind",
)
@pytest.mark.parametrize("file", ["pieces/needle.svg", "manifest.json"])
def test_a_file_the_process_may_not_read_is_unreadable(tmp_path, file):
    directory = copied(tmp_path, "dial")
    (directory / file).chmod(0)
    try:
        found = refused("asset.unreadable", lambda: Rig(directory))
    finally:
        (directory / file).chmod(0o644)
    assert found.detail["file"] == file


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ('"newer": 1e400,', "finite"),
        ('"newer": [1, -1E999],', "finite"),
        ('"newer": 9007199254740993,', "integers"),
        ('"newer": "\\ud800",', "Unicode"),
        ('"newer": Infinity,', "JSON number"),
    ],
)
def test_a_manifest_outside_the_json_domain_is_refused(tmp_path, text, says):
    directory = copied(tmp_path, "dial")
    manifest = directory / "manifest.json"
    manifest.write_text(
        manifest.read_text().replace('"version": 1,', f'"version": 1, {text}')
    )
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == "manifest.json"
    assert says in found.detail["reason"]


def test_a_manifest_that_is_not_utf8_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    manifest = directory / "manifest.json"
    manifest.write_bytes(manifest.read_bytes().replace(b'"dial"', b'"di\xe5l"', 1))
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert "UTF-8" in found.detail["reason"]


#: Outlines with as many points as the rest ellipse (8 curves, 32 points)
#: and another topology: two closed squares; one open run of 8 curves.
_TWO_SQUARES = (
    "M 110 150 L 150 150 L 150 120 L 110 120 L 110 150 Z"
    " M 250 150 L 290 150 L 290 180 L 250 180 L 250 150 Z"
)
_OPEN_RUN = "M 110 150 " + " ".join(
    f"L {120 + 20 * i} {150 + (-1) ** i * 10}" for i in range(8)
)


@pytest.mark.parametrize("d", [_TWO_SQUARES, _OPEN_RUN])
def test_a_stored_outline_of_the_same_count_but_another_topology_is_refused(
    tmp_path, d
):
    directory = copied(tmp_path, "layered-section")
    (directory / "shapes" / "inner_layer-1.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">\n'
        f'  <path d="{d}" fill="#d9a3a3"/>\n'
        "</svg>\n"
    )
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == "parameters[0].drives[0].keys[1].file"


def test_a_size_with_both_extents_zero_is_refused(tmp_path):
    directory = edited(
        copied(tmp_path, "dial"), lambda d: d["size"].update(width=0, height=0)
    )
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == "size"


@pytest.mark.parametrize("unit", ["deg rees", "", "x" * 17, "\u00b0", 7])
def test_a_unit_outside_short_printable_ascii_is_refused(tmp_path, unit):
    directory = edited(
        copied(tmp_path, "dial"), lambda d: d["parameters"][0].update(unit=unit)
    )
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == "parameters[0].unit"
    assert found.detail["name"] == "reading"
