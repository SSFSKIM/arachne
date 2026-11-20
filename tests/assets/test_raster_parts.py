"""Rigs whose parts are stills, with no GPU: the manifest's raster members
and depths, each behind its feature name; a raster member's quad, posed
and moved as a vector part is; what admission measures of such a rig and
of the vector rigs, whose measures do not change; a still that does not
morph; and the quad's plain, keyed fields.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arachne import FadeIn, Rig, load_asset
from arachne.assets import RasterPart, admit
from arachne.assets.admission import write_checks
from arachne.assets.motion import MorphParts
from arachne.assets.reader import (
    CHECKS,
    FIXTURES,
    MEDIA_CHECKS,
    RASTER_CHECKS,
    checks_for,
    read_asset,
)
from arachne.compose.layer import layer_of
from arachne.mobject.media.frames import checked_member, decode, frame_digest
from arachne.routing import held
from arachne.timeline.record import NodeState, content_fields
from tests.media.raster_scenes import flap_pictures, layer_pictures

from .draw_fixture_pack import DEFECTS, NAMES, RASTER
from .helpers import admitted, copied, edited, refused

DEFECT_DIR = Path(__file__).parent / "defects"


def _quads(rig: Rig, name: str) -> list[RasterPart]:
    return [m for m in rig.part(name).get_family() if isinstance(m, RasterPart)]


def _shown(rig: Rig, name: str) -> RasterPart:
    """The one quad of ``name`` that holds points."""
    (quad,) = [q for q in _quads(rig, name) if len(q.points)]
    return quad


# The manifest


def _invalid(directory: Path) -> dict:
    return refused("asset.manifest_invalid", lambda: load_asset(directory)).detail


def test_a_raster_member_without_raster_in_requires_is_refused(tmp_path):
    directory = edited(copied(tmp_path, "hinged-flap"), lambda d: d.update(requires=[]))
    detail = _invalid(directory)
    assert detail["member"] == "parts[0].box"
    assert '"raster" in requires' in detail["reason"]


def test_a_depth_without_depth_in_requires_is_refused(tmp_path):
    directory = edited(
        copied(tmp_path, "layered-ground"), lambda d: d.update(requires=["raster"])
    )
    detail = _invalid(directory)
    assert detail["member"] == "parts[0].depth"
    assert '"depth" in requires' in detail["reason"]


def _flap_driver(document: dict, kind: str, **more) -> None:
    lift = document["parameters"][0]
    lift["drives"][0] |= {"kind": kind, **more}
    lift["drives"][0].pop("unit", None)


@pytest.mark.parametrize(
    ("edit", "member", "says"),
    [
        (lambda d: _flap_driver(d, "warp"), "parameters[0].drives[0].kind", "a warp"),
        (
            lambda d: _flap_driver(d, "shape", keys=[{"at": 0, "file": "x.svg"}]),
            "parameters[0].drives[0].kind",
            "a shape",
        ),
        (
            lambda d: d["parts"][1].update(
                lattice={"cols": 1, "rows": 1, "bounds": [0, 20, 160, 44]}
            ),
            "parts[1].lattice",
            "no lattice",
        ),
        (lambda d: d["parts"][0].pop("digest"), "parts[0].digest", "box alone"),
        (
            lambda d: d["parts"][0].update(element="body"),
            "parts[0].element",
            "no elements",
        ),
        (lambda d: d["parts"][0].update(box=[0, 40, 0, 160]), "parts[0].box", "x1"),
        (
            lambda d: d["slots"][0]["drawings"][1].update(digest="sha256:00"),
            "slots[0].drawings[1].digest",
            "64 lower-case hex",
        ),
    ],
    ids=["warp", "shape", "lattice", "box-alone", "element", "empty-box", "digest"],
)
def test_a_raster_member_that_would_bend_or_is_malformed_is_refused(
    tmp_path, edit, member, says
):
    detail = _invalid(edited(copied(tmp_path, "hinged-flap"), edit))
    assert detail["member"] == member
    assert says in detail["reason"]


def test_the_check_set_is_looked_up_by_kind_and_requires():
    assert (*CHECKS, "frames_exact") == RASTER_CHECKS
    for name in NAMES:
        assert checks_for(load_asset(f"fixture:{name}").manifest) == CHECKS
    for name in RASTER:
        assert checks_for(load_asset(f"fixture:{name}").manifest) == RASTER_CHECKS
    assert checks_for(load_asset("fixture:swatch").manifest) == MEDIA_CHECKS


# Loading


def test_a_raster_file_in_no_admitted_form_is_refused_at_load(tmp_path):
    directory = copied(tmp_path, "hinged-flap")
    Image.new("P", (40, 6)).save(directory / "pieces" / "flap.png", format="PNG")
    detail = refused("media.inexact", lambda: load_asset(directory)).detail
    assert detail["reason"] == "format"
    assert detail["file"] == "pieces/flap.png"


def test_a_changed_raster_file_is_an_asset_not_admitted(tmp_path):
    directory = copied(tmp_path, "hinged-flap")
    path = directory / "pieces" / "on.png"
    pixels = decode(path.read_bytes())
    pixels[5, 5, 0] ^= 1
    Image.fromarray(pixels).save(path, format="PNG")
    assert refused("asset.unadmitted", lambda: Rig(directory)).detail["reason"] == (
        "digest"
    )


def test_a_raster_member_is_a_quad_that_declares_its_row():
    rig = Rig("fixture:hinged-flap")
    for name in ("body", "flap", "badge"):
        for quad in _quads(rig, name):
            assert held(quad) == {"media.raster_part"}
            assert quad.get_stroke_width() == 0
    # The slot's hidden drawing holds no corners and paints nothing.
    hidden = [q for q in _quads(rig, "badge") if not len(q.points)]
    assert len(hidden) == 1
    assert hidden[0].picture() is None


def test_the_quad_s_picture_is_its_member_s_file_checked_against_its_digest():
    rig = Rig("fixture:hinged-flap")
    asset = load_asset("fixture:hinged-flap")
    for part in asset.manifest.parts:
        quad = _shown(rig, part.name)
        expected = checked_member(asset.directory, asset.name, part.file, part.digest)
        assert np.array_equal(quad.pixels(), expected)
        assert frame_digest(np.asarray(quad.pixels())) == part.digest


def _keyed(mob) -> dict:
    """Each field of ``mob``'s content key, by its name."""
    return {key: part for key, part, *_ in content_fields(NodeState.of(mob))}


def test_the_quad_s_fields_that_decide_its_picture_are_plain_and_keyed():
    quad = _shown(Rig("fixture:hinged-flap"), "body")
    keyed = _keyed(quad)
    for field in ("_asset_digest", "_file", "_digest"):
        assert isinstance(vars(quad)[field], str)
        before = keyed[field]
        vars(quad)[field] = vars(quad)[field] + "x"
        assert _keyed(quad)[field] != before
    # Where this host keeps the asset is no part of a key.
    assert "_directory" not in keyed
    assert RasterPart._content_key_omits == {"_directory", "_asset_name"}


def test_one_asset_in_two_directories_keys_its_quads_alike(tmp_path):
    one, two = (
        _shown(Rig(copied(tmp_path, "hinged-flap", name)), "flap")
        for name in ("one", "two")
    )
    assert _keyed(one) == _keyed(two)


# Geometry


def _corners_by_name(rig: Rig) -> dict[str, np.ndarray]:
    found = {}
    for name in ("body", "flap"):
        found[name] = _shown(rig, name).corners()
    badge = _shown(rig, "badge")
    found[rig.shown("badge")] = badge.corners()
    return found


def test_posed_quads_stand_where_the_manifest_and_the_pose_put_them():
    rig = Rig("fixture:hinged-flap").pose(lift=30, grow=1.2).show(badge="off")
    expected = flap_pictures(30, 1.2, "off")
    found = _corners_by_name(rig)
    assert found.keys() == expected.keys()
    for name, corners in expected.items():
        assert np.allclose(found[name], corners, atol=1e-9), name


def test_a_displaced_rig_s_centre_box_scale_and_rotation_are_its_art_s():
    rig = Rig("fixture:hinged-flap").pose(lift=30, grow=1.2).show(badge="off")
    move = np.array([1.25, -0.75, 0.0])
    rig.shift(move)
    pictures = {n: c + move for n, c in flap_pictures(30, 1.2, "off").items()}
    every = np.concatenate(list(pictures.values()))
    low, high = every.min(axis=0), every.max(axis=0)
    centre = (low + high) / 2
    assert np.allclose(rig.get_center(), centre, atol=1e-9)
    assert np.allclose(rig.get_corner([-1, -1, 0]), low, atol=1e-9)
    assert np.allclose(rig.get_corner([1, 1, 0]), high, atol=1e-9)
    assert np.isclose(rig.width, high[0] - low[0])
    rig.scale(1.5)
    found = _corners_by_name(rig)
    for name, corners in pictures.items():
        assert np.allclose(found[name], centre + 1.5 * (corners - centre), atol=1e-9)
    rig.rotate(math.pi / 6)
    c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    found = _corners_by_name(rig)
    for name, corners in pictures.items():
        scaled = centre + 1.5 * (corners - centre)
        assert np.allclose(found[name], (scaled - centre) @ turn.T + centre, atol=1e-9)


def test_a_hidden_raster_drawing_adds_nothing_to_the_box():
    on = Rig("fixture:hinged-flap").pose(grow=0.8)
    off = Rig("fixture:hinged-flap").pose(grow=0.8).show(badge="off")
    assert np.allclose(on.get_corner([-1, -1, 0]), off.get_corner([-1, -1, 0]))
    assert np.allclose(on.get_corner([1, 1, 0]), off.get_corner([1, 1, 0]))
    assert len(_quads(off, "badge")) == 2


def test_a_colour_or_stroke_given_to_the_rig_changes_no_picture():
    rig = Rig("fixture:hinged-flap")
    before = {n: np.array(_shown(rig, n).pixels()) for n in ("body", "flap", "badge")}
    rig.set_color("#ff0000")
    rig.set_stroke(width=6)
    for name, pixels in before.items():
        assert np.array_equal(_shown(rig, name).pixels(), pixels)


def test_the_rig_s_opacity_is_its_pictures_opacity():
    rig = Rig("fixture:hinged-flap")
    full = np.array(_shown(rig, "body").pixels())
    rig.set_opacity(0.5)
    half = np.array(_shown(rig, "body").pixels())
    assert np.array_equal(half[..., :3], full[..., :3])
    assert np.array_equal(half[..., 3], (full[..., 3] * 0.5).astype(np.uint8))


def test_a_fade_in_sets_the_pictures_opacity_through_the_play():
    rig = Rig("fixture:hinged-flap")
    fade = FadeIn(rig)
    fade.begin()
    for alpha in (0.0, 0.25, 0.5, 1.0):
        fade.interpolate(alpha)
        opacity = _shown(rig, "flap").get_fill_opacity()
        assert np.isclose(opacity, fade.rate_func(alpha))
        expected = (255 * min(opacity, 1.0)) // 1
        assert int(np.asarray(_shown(rig, "flap").pixels())[2, 2, 3]) == expected


def test_an_anchor_on_a_raster_member_follows_its_pose():
    rig = Rig("fixture:hinged-flap").pose(lift=50).shift([0.5, 0.25, 0])
    flap = next(
        p for p in load_asset("fixture:hinged-flap").manifest.parts if p.name == "flap"
    )
    from tests.media.raster_scenes import FLAP, canvas_to_scene, turned

    tip = canvas_to_scene(FLAP, turned(np.array([[150.0, 32.0]]), flap.pivot, 50))[0]
    assert np.allclose(rig.anchor("tip").point(), tip + [0.5, 0.25, 0], atol=1e-9)


def test_a_layered_set_s_members_are_layers_at_their_depths_standing_at_home():
    rig = Rig("fixture:layered-ground")
    for name, corners in layer_pictures().items():
        group = rig.part(name)
        depth = next(p.depth for p in rig.asset.manifest.parts if p.name == name)
        assert layer_of(group).depth == depth
        assert np.allclose(_shown(rig, name).corners(), corners, atol=1e-9)
    assert layer_of(rig) is None


# Admission


def _results(report) -> list[tuple]:
    return [(r.name, r.passed, r.worst, r.pose) for r in report.results]


def test_the_vector_rigs_check_sets_and_measures_are_unchanged():
    clean = [
        ("hole", True, 0, None),
        ("detached_part", True, 0.0, None),
        ("anchor_off_part", True, 0.0, None),
    ]
    for name in NAMES:
        report = admit(FIXTURES / name)
        assert _results(report) == clean, name
        recorded = json.loads((FIXTURES / name / "manifest.json").read_text())
        assert recorded["checks"] == report.checks(), name
    measured = {
        "open-seam": ("hole", 3042, "slide=1"),
        "loose-part": ("detached_part", 105.0, "pull=1"),
        "stray-anchor": ("anchor_off_part", 34.0, "the default pose"),
    }
    for name, check in DEFECTS.items():
        report = admit(DEFECT_DIR / name)
        assert [r.name for r in report.results] == list(CHECKS)
        failing = report.result(check)
        assert (failing.name, failing.worst, failing.pose) == measured[name]
        assert [r.name for r in report.results if not r.passed] == [check]


def test_the_raster_rigs_are_admitted_with_their_four_checks():
    for name in RASTER:
        report = admit(FIXTURES / name)
        assert [r.name for r in report.results] == list(RASTER_CHECKS)
        assert report.passed, report.lines()
        assert report.lines()[-1] == "admission: 4/4 pass"


def test_a_planted_detached_raster_part_fails_that_check_alone(tmp_path):
    def pulled(document: dict) -> None:
        document["parameters"].append(
            {
                "name": "pull",
                "min": 0,
                "max": 1,
                "default": 0,
                "drives": [
                    {
                        "part": "flap",
                        "kind": "translate",
                        "keys": [
                            {"at": 0, "value": [0, 0]},
                            {"at": 1, "value": [0, -80]},
                        ],
                    }
                ],
            }
        )

    report = admit(edited(copied(tmp_path, "hinged-flap"), pulled))
    assert [r.name for r in report.results if not r.passed] == ["detached_part"]
    assert report.result("detached_part").pose == "pull=1"


def test_a_raster_file_unlike_its_listed_digest_fails_frames_exact_alone(tmp_path):
    def swapped(document: dict) -> None:
        on, off = document["slots"][0]["drawings"]
        on["digest"], off["digest"] = off["digest"], on["digest"]

    report = admit(edited(copied(tmp_path, "hinged-flap"), swapped))
    assert [r.name for r in report.results if not r.passed] == ["frames_exact"]
    assert report.result("frames_exact").worst == 2


def test_a_raster_rig_admitted_again_loads(tmp_path):
    directory = copied(tmp_path, "hinged-flap")
    path = directory / "pieces" / "on.png"
    pixels = decode(path.read_bytes())
    pixels[5, 5, 0] ^= 1
    Image.fromarray(pixels).save(path, format="PNG")
    edited(
        directory,
        lambda d: d["slots"][0]["drawings"][0].update(digest=frame_digest(pixels)),
    )
    write_checks(directory, admit(directory))
    rig = Rig(directory)
    assert np.array_equal(_shown(rig, "badge").pixels(), pixels)
    assert read_asset(directory).digest == load_asset(directory).digest


# Morphing


def test_a_still_has_no_counterpart_when_rigs_morph(tmp_path):
    from arachne import Scene

    scene = Scene()
    source = Rig("fixture:hinged-flap")
    destination = Rig(copied(tmp_path, "hinged-flap", "other"))
    scene.add(source)
    morph = MorphParts(source, destination)
    morph._setup_scene(scene)
    assert morph._pairs == []
    assert len(morph._leaving) == 3
    assert len(morph._arriving) == 3
    morph.begin()
    assert {type(part).__name__ for part in morph._parts} == {"FadeOut", "FadeIn"}


def test_an_arriving_slot_s_still_fades_in_on_its_slot_s_layer(tmp_path):
    from arachne import Scene

    deep = edited(
        copied(tmp_path, "hinged-flap", "deep"),
        lambda d: (
            d.update(requires=["raster", "depth"]),
            d["slots"][0].update(depth=3),
        ),
    )
    admitted(deep)
    scene = Scene()
    source = Rig("fixture:hinged-flap")
    destination = Rig(deep)
    scene.add(source)
    morph = MorphParts(source, destination)
    morph._setup_scene(scene)
    layers = {layer_of(m) for m in morph._arriving}
    assert layers == {None, layer_of(destination.part("badge"))}
    (badge,) = [m for m in morph._arriving if layer_of(m) is not None]
    assert layer_of(badge).depth == 3


# The frame cache


def test_a_kept_picture_is_given_only_for_the_digest_it_was_checked_against():
    from arachne.mobject.media.frames import FrameCache

    asset = load_asset("fixture:hinged-flap")
    part = asset.manifest.parts[0]
    cache = FrameCache()
    found = cache.member(
        asset.digest, asset.directory, asset.name, part.file, part.digest, 7
    )
    assert len(cache) == 1
    assert (
        cache.member(
            asset.digest, asset.directory, asset.name, part.file, part.digest, 7
        )
        is found
    )
    other = "sha256:" + "0" * 64
    error = refused(
        "media.inexact",
        lambda: cache.member(
            asset.digest, asset.directory, asset.name, part.file, other, 9
        ),
    )
    assert error.line == 9
    assert error.detail["reason"] == "digest"
    assert error.detail["file"] == part.file
    assert error.detail["expected"] == other
    assert error.detail["found"] == part.digest


# Admission of stills that do not decode


def test_a_still_whose_pixels_do_not_decode_is_a_named_failure_and_no_geometry(
    tmp_path,
):
    asset = copied(tmp_path, "hinged-flap", "corrupt")
    flap = asset / "pieces" / "flap.png"
    data = bytearray(flap.read_bytes())
    start = data.index(b"IDAT") + 8
    data[start : start + 24] = bytes(b ^ 0x5A for b in data[start : start + 24])
    flap.write_bytes(bytes(data))
    report = admit(asset)
    found = {result.name: result for result in report.results}
    assert [result.name for result in report.results] == list(RASTER_CHECKS)
    for name in CHECKS:
        assert not found[name].passed
        assert math.isnan(found[name].worst)
        assert found[name].pose is None
        assert found[name].failure.startswith("not measured: pieces/flap.png is")
    assert not found["frames_exact"].passed
    assert found["frames_exact"].pose == "pieces/flap.png"


def test_admission_decodes_each_still_once_for_its_masks(monkeypatch):
    from arachne.mobject.media import frames

    decoded: list[tuple[int, int]] = []
    real = frames.decode

    def counted(data: bytes) -> np.ndarray:
        pixels = real(data)
        decoded.append(pixels.shape[:2])
        return pixels

    monkeypatch.setattr(frames, "decode", counted)
    report = admit("fixture:hinged-flap")
    assert report.passed
    files = {f"pieces/{name}.png" for name in ("body", "flap", "on", "off")}
    # Once for the masks of every pose, and twice for frames_exact.
    assert len(decoded) == 3 * len(files)


def test_a_frame_kept_by_a_media_object_is_given_only_for_its_own_digest():
    """An entry a media object's frame read kept, asked for as a member's
    picture: given for the digest it was checked against, refused for any
    other at the asking line.
    """
    from arachne.mobject.media.frames import FrameCache

    swatch = load_asset("fixture:swatch")
    (spec,) = swatch.manifest.media.frames
    cache = FrameCache()
    kept = cache.frame(swatch, 0)
    args = (swatch.digest, swatch.directory, swatch.name, spec.file)
    assert cache.member(*args, spec.digest, 4) is kept
    other = "sha256:" + "1" * 64
    error = refused("media.inexact", lambda: cache.member(*args, other, 5))
    assert error.line == 5
    assert (error.detail["expected"], error.detail["found"]) == (other, spec.digest)


def test_a_refused_request_leaves_the_kept_entry_as_it_was():
    from arachne.mobject.media.frames import FrameCache

    asset = load_asset("fixture:hinged-flap")
    part = asset.manifest.parts[0]
    args = (asset.digest, asset.directory, asset.name, part.file)
    cache = FrameCache()
    kept = cache.member(*args, part.digest)
    refused("media.inexact", lambda: cache.member(*args, "sha256:" + "2" * 64))
    assert cache.member(*args, part.digest) is kept
    assert len(cache) == 1
