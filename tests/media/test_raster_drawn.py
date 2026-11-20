"""Rigs whose parts are stills, drawn on a device, the launch mode's: each
picture stands where the rig's placement and its part's transform put it,
as its still opened by path stands there; a fade of the rig fades its
pictures at every tick; two stills of one size show their own pixels in
either load order and after a slot switch; a still that decodes otherwise
draws nothing; a layered set slides by its depths in the drawn frame and
its focus blurs each layer by its depth; and old scenes keep their bytes
around a scene that drew a rig of stills. Collected only where the GPU
build is installed.
"""

from __future__ import annotations

import json
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.mobject.media import frames
from arachne.mobject.media.frames import FrameCache, decode, frame_digest
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError
from arachne.timeline.footprint import object_map
from tests.assets.helpers import admitted
from tests.helpers.gpu_door import run_child

from .conftest import SMALL, copied, edited, line_of
from .raster_scenes import (
    FLAP,
    PAN,
    depth_of,
    files_of,
    flap_pictures,
    layer_pictures,
)

SCENES = Path(__file__).parent / "raster_scenes.py"
#: The frame height the scenes are configured at: the default viewing
#: distance; and the frame's width.
DISTANCE, WIDTH = 8.0, 8.0 * 16 / 9


def _pixels(session, t: Fraction | float) -> np.ndarray:
    return np.asarray(session.frame(t)).astype(int)


def _close(a: np.ndarray, b: np.ndarray, within: int = 1) -> bool:
    return a.shape == b.shape and int(np.abs(a - b).max()) <= within


def test_a_posed_rig_s_pictures_stand_where_its_placement_puts_them(session_of, door):
    rig = session_of("PosedFlap", SCENES)
    images = session_of("PosedFlapAsImages", SCENES)
    for t in (0, Fraction(1, 4)):
        drawn = _pixels(rig, t)
        assert _close(drawn, _pixels(images, t))
    # Every picture is drawn, the badge's switched drawing included.
    assert len(np.unique(drawn.reshape(-1, 4), axis=0)) > 8
    assert door.draws == 2


def test_a_fade_of_the_rig_fades_its_pictures_at_every_tick(session_of, door):
    session = session_of("FadingFlap", SCENES, frame_rate=60)
    body = np.array([150, 110, 70])
    # The body's middle: the scene point (0, -0.2) on a frame 96 x 54.
    column, row = 48, int(27 + 0.2 * 96 / WIDTH)
    for k in range(31):
        opacity = k / 30
        alpha = int(255 * opacity) / 255
        drawn = _pixels(session, Fraction(k, 60))[row, column, :3]
        assert np.abs(drawn - body * alpha).max() <= 2, k


TWO = """\
import numpy as np

from arachne import *


def image(file, corners):
    picture = ImageMobject(file)
    picture.points = np.array(corners)
    return picture


class Two(Scene):
    def construct(self):
        if {a_first}:
            a, b = Rig({a!r}), Rig({b!r})
        else:
            b, a = Rig({b!r}), Rig({a!r})
        self.add(a.shift(2.5 * LEFT), b.shift(2.5 * RIGHT))
        self.wait(0.5)
        a.show(badge="off")
        b.show(badge="off")
        self.wait(0.5)


class TwoAsImages(Scene):
    def construct(self):
        self.add(*{before})
        self.wait(0.5)
        self.clear()
        self.add(*{after})
        self.wait(0.5)


def pictures(rig):
    from arachne.assets import RasterPart

    return [
        image(str(m._directory / m.file), m.corners())
        for m in rig.family_members_with_points()
        if isinstance(m, RasterPart) and m.corners() is not None
    ]


# The pictures at the corners the rigs' own quads hold.
class TwoAsRigImages(Scene):
    def construct(self):
        a, b = Rig({a!r}), Rig({b!r})
        a.shift(2.5 * LEFT)
        b.shift(2.5 * RIGHT)
        self.add(*pictures(a), *pictures(b))
        self.wait(0.5)
        self.clear()
        a.show(badge="off")
        b.show(badge="off")
        self.add(*pictures(a), *pictures(b))
        self.wait(0.5)
"""


def _images(directory: Path, shown: str, shift: float) -> str:
    """Source of the flap rig's pictures from ``directory`` as images, at
    rest with ``shown`` on its badge, moved ``shift`` across: the items of a
    list.
    """
    files = files_of(directory)
    made = []
    for name, corners in flap_pictures(0, 1, shown).items():
        at = (corners + [shift, 0, 0]).tolist()
        made.append(f"image({str(directory / files[name])!r}, {at!r})")
    return ", ".join(made)


@pytest.mark.parametrize("a_first", [True, False])
def test_two_stills_of_one_size_show_their_own_pixels_in_either_load_order(
    tmp_path, session_of, door, a_first
):
    other = copied(tmp_path, "hinged-flap", "recoloured")
    on = other / "pieces" / "on.png"
    pixels = decode(on.read_bytes())
    pixels[..., :3] = 255 - pixels[..., :3]
    from PIL import Image

    Image.fromarray(pixels).save(on, format="PNG")
    document = json.loads((other / "manifest.json").read_text())
    document["slots"][0]["drawings"][0]["digest"] = frame_digest(pixels)
    (other / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    admitted(other)
    file = tmp_path / "two.py"
    file.write_text(
        textwrap.dedent(
            TWO.format(
                a=str(FLAP),
                b=str(other),
                a_first=a_first,
                before=f"[{_images(FLAP, 'on', -2.5)}, {_images(other, 'on', 2.5)}]",
                after=f"[{_images(FLAP, 'off', -2.5)}, {_images(other, 'off', 2.5)}]",
            )
        )
    )
    rigs, images = session_of("Two", file), session_of("TwoAsImages", file)
    own = session_of("TwoAsRigImages", file)
    before, after = Fraction(1, 4), Fraction(3, 4)
    for t in (before, after):
        drawn = _pixels(rigs, t)
        # At the corners the rigs' quads hold, the same pictures: the same
        # pixels, within 1, whichever drawing each badge shows.
        assert _close(drawn, _pixels(own, t)), (t, _where(drawn, _pixels(own, t)))
    drawn, expected = _pixels(rigs, before), _pixels(images, before)
    assert _close(drawn, expected), _where(drawn, expected)
    # At the corners written out here, which differ from the quads' in their
    # last bits, a resampled pixel on the edge of the switched-in badge may
    # move by 2; no other pixel by more than 1.
    drawn, expected = _pixels(rigs, after), _pixels(images, after)
    edges = np.zeros(drawn.shape[:2], dtype=bool)
    for shift in (-2.5, 2.5):
        corners = flap_pictures(0, 1, "off")["off"] + [shift, 0, 0]
        x0, y0, x1, y1 = _pixel_box(corners, np.zeros(3))
        ring = np.zeros_like(edges)
        ring[int(y0) - 1 : int(np.ceil(y1)) + 1, int(x0) - 1 : int(np.ceil(x1)) + 1] = (
            True
        )
        ring[int(y0) + 2 : int(np.ceil(y1)) - 2, int(x0) + 2 : int(np.ceil(x1)) - 2] = (
            False
        )
        edges |= ring
    off = np.abs(drawn - expected).max(axis=2)
    assert off[~edges].max() <= 1, _where(drawn, expected)
    assert off.max() <= 2, _where(drawn, expected)


def _where(drawn: np.ndarray, expected: np.ndarray) -> dict:
    """Where two frames differ by more than 1: how much, the box of those
    pixels, and a few of them with both values.
    """
    off = np.abs(drawn - expected).max(axis=2) > 1
    rows, columns = np.nonzero(off)
    if not len(rows):
        return {}
    picked = list(zip(rows[:4].tolist(), columns[:4].tolist(), strict=True))
    return {
        "most": int(np.abs(drawn - expected).max()),
        "box": (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        ),
        "count": int(off.sum()),
        "at": [
            (r, c, drawn[r, c].tolist(), expected[r, c].tolist()) for r, c in picked
        ],
    }


def test_a_still_that_decodes_otherwise_draws_nothing_and_names_its_file(
    tmp_path, monkeypatch, door
):
    monkeypatch.setattr(frames, "cache", FrameCache())
    decoded = frames.decode

    def changed(data: bytes) -> np.ndarray:
        pixels = decoded(data)
        if pixels.shape[:2] == (6, 40):  # the flap's still alone
            pixels[2, 2, 0] ^= 8
        return pixels

    monkeypatch.setattr(frames, "decode", changed)
    server = Dispatcher()
    picture = tmp_path / "frame.png"
    try:
        server.handle(
            "open", {"file": str(SCENES), "scene": "PosedFlap", "config": SMALL}
        )
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.25, "path": str(picture)})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    finding = refused.value.data["finding"]
    assert finding["code"] == "media.inexact"
    assert finding["detail"]["reason"] == "digest"
    assert finding["detail"]["file"] == "pieces/flap.png"
    assert finding["line"] == line_of("posed-flap", SCENES).line
    assert door.draws == 0
    assert not picture.exists()


def _pixel_box(corners: np.ndarray, centre: np.ndarray) -> tuple[float, ...]:
    """The pixel box (x right, y down) of ``corners`` on a frame 96 by 54
    pixels centred on ``centre``.
    """
    scale = 96 / WIDTH
    x = (corners[:, 0] - centre[0]) * scale + 48
    y = 27 - (corners[:, 1] - centre[1]) * scale
    return (x.min(), y.min(), x.max(), y.max())


def _drawn_slides(session, home: dict, depths: dict, *, whole: bool) -> None:
    """At every tick of the pan, from the first the fade shows the set at,
    each layer's footprint in the drawn frame stands where its depth slides
    it, within 1 px, and where ``state_at`` says: its columns only, or with
    ``whole`` (every layer opaque and apart) its whole box.
    """
    names = {name: f"rig[{i}]" for i, name in enumerate(home, start=1)}
    sides = range(4) if whole else (0, 2)
    for k in range(1, 61):
        t = Fraction(k, 60)
        state = session.state_at(t)
        by_id = {node.id: node for node in state.nodes}
        named = {node.name: node for node in state.nodes if node.name}
        # Each layer's quad, the one drawable its group holds.
        quads = {
            name: next(
                by_id[child]
                for child in named[names[name]].children
                if by_id[child].type == "RasterPart"
            )
            for name in home
        }
        drawn = object_map(session, t, [quad.id for quad in quads.values()])
        moved = PAN * k / 60
        for name, corners in home.items():
            slide = (1 - DISTANCE / (DISTANCE + depths[name])) * moved
            expected = _pixel_box(corners + slide, moved)
            box = drawn.objects[quads[name].id].box
            for side in sides:
                assert abs(box[side] - expected[side]) <= 1, (name, k, side)
                assert abs(quads[name].screen_box[side] - box[side]) <= 1, (name, k)


def test_a_layered_set_slides_by_its_depths_in_the_drawn_frame(session_of, door):
    """The fixture's layers are clear above their crests and cover one
    another: their columns are what their footprints show whole.
    """
    session = session_of("LayeredPan", SCENES, frame_rate=60)
    home = layer_pictures()
    _drawn_slides(session, home, {name: depth_of(name) for name in home}, whole=False)


#: The layered set made opaque, each layer apart from the others however
#: the pan slides them: its boxes and the pictures' sizes.
APART = {
    "far": [0, 0, 240, 40],
    "middle": [0, 60, 240, 100],
    "near": [0, 120, 240, 160],
}

PANNED = """\
from arachne import *


class Panned(MovingCameraScene):
    def construct(self):
        rig = Rig({asset!r})
        rig.view_through(self.camera)
        self.play(
            self.camera.frame.animate.shift({pan!r}),
            FadeIn(rig),
            rate_func=linear,
            run_time=1,
        )
        self.wait(0.25)
"""


def test_an_opaque_layered_set_slides_whole_in_the_drawn_frame(
    tmp_path, session_of, door
):
    from PIL import Image

    from .raster_scenes import box_corners, canvas_to_scene

    asset = copied(tmp_path, "layered-ground", "apart")
    digests = {}
    for i, name in enumerate(APART):
        pixels = np.empty((8, 48, 4), dtype=np.uint8)
        pixels[...] = (60 + 60 * i, 200 - 50 * i, 90, 255)
        Image.fromarray(pixels).save(asset / "layers" / f"{name}.png", format="PNG")
        digests[name] = frame_digest(pixels)

    def apart(document: dict) -> None:
        for part in document["parts"]:
            part.update(box=APART[part["name"]], digest=digests[part["name"]])

    edited(asset, apart)
    admitted(asset)
    file = tmp_path / "panned.py"
    file.write_text(PANNED.format(asset=str(asset), pan=PAN.tolist()))
    session = session_of("Panned", file, frame_rate=60)
    home = {
        name: canvas_to_scene(asset, box_corners(box)) for name, box in APART.items()
    }
    depths = {name: depth_of(name) for name in APART}
    _drawn_slides(session, home, depths, whole=True)


def test_the_focus_blurs_each_layer_by_its_depth(session_of, door):
    layered = session_of("LayeredFocus", SCENES)
    images = session_of("LayeredFocusAsImages", SCENES)
    assert images.route == layered.route == "gpu"
    assert _close(_pixels(layered, 0), _pixels(images, 0))
    assert door.draws == 2


#: Opens each old scene, then draws a scene of a rig of stills on the
#: launch mode's device, then each old scene again, in one fresh
#: interpreter.
PROGRAM = """\
import hashlib, json, sys

from arachne.compose import device
from arachne.session import Session
from tests.helpers.gpu_door import LaunchDoor

door = device.door = LaunchDoor()

directory, media, scenes = sys.argv[1:4]
config = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4, "media_dir": media}


def opened(file, scene):
    session = Session.open(file, scene, config=config)
    try:
        return {
            "route": session.route,
            "frames": [
                hashlib.sha256(session.frame(t).tobytes()).hexdigest()
                for t in (0, 0.25, 0.5, 0.7)
            ],
        }
    finally:
        session.close()


old = f"{directory}/old.py"
fresh = {scene: opened(old, scene) for scene in ("OldSvg", "OldImage")}
raster = opened(scenes, "LayeredPan")
drawn = door.draws
after = {scene: opened(old, scene) for scene in ("OldSvg", "OldImage")}
print(json.dumps({"fresh": fresh, "raster": raster, "drawn": drawn, "after": after}))
"""


def test_old_scenes_draw_the_same_bytes_after_a_scene_drew_a_rig_of_stills(tmp_path):
    from tests.assets.test_old_constructors import _scenes

    directory = _scenes(tmp_path / "scenes")
    printed = run_child(PROGRAM, str(directory), str(tmp_path / "media"), str(SCENES))
    found = json.loads(printed.strip().splitlines()[-1])
    assert found["raster"]["route"] == "gpu"
    assert len(set(found["raster"]["frames"])) > 1
    assert found["drawn"] == 4
    assert found["fresh"] == found["after"]
    assert all(scene["route"] == "cpu" for scene in found["fresh"].values())


MORPH = """\
from arachne import *
from arachne.compose import set_focus


class Morph(Scene):
    def construct(self):
        if {focus}:
            set_focus(self.camera, 0.0, 0.08)
        source = Rig("fixture:hinged-flap")
        self.add(source)
        self.play(MorphParts(source, Rig({deep!r})), rate_func=linear, run_time=1)
"""


def test_a_still_fading_into_a_slot_at_a_depth_is_blurred_by_the_focus(
    tmp_path, session_of, door
):
    """Mid-morph, before the morph cleans up: under a focus, the drawing
    fading into a slot at a depth is blurred, and nothing else is.
    """
    deep = edited(
        copied(tmp_path, "hinged-flap", "deep"),
        lambda d: (
            d.update(requires=["raster", "depth"]),
            d["slots"][0].update(depth=3),
        ),
    )
    admitted(deep)
    frames = {}
    for focus in (True, False):
        file = tmp_path / f"morph_{focus}.py"
        file.write_text(MORPH.format(focus=focus, deep=str(deep)))
        frames[focus] = _pixels(session_of("Morph", file), Fraction(1, 2))
    off = np.abs(frames[True] - frames[False]).max(axis=2) > 1
    assert off.any()
    rows, columns = np.nonzero(off)
    x0, y0, x1, y1 = _pixel_box(flap_pictures(0, 1, "on")["on"], np.zeros(3))
    # The blur spreads the badge by its sigma (0.24 scene units, under 2
    # pixels) a few times over, no farther.
    assert columns.min() >= x0 - 8
    assert columns.max() <= x1 + 8
    assert rows.min() >= y0 - 8
    assert rows.max() <= y1 + 8
