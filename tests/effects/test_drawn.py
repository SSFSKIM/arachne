"""Effects drawn on this run's device (:mod:`tests.helpers.gpu_door`): a
blur spreads an edge by its declared width, a mask cuts at its declared
place, removing an effect changes no pixel outside its reach, and the plays
that move a blur or the focus draw what their numbers say at each instant.
"""

from __future__ import annotations

import json
import math
import textwrap
from fractions import Fraction

import numpy as np
import pytest

from arachne.compose.layer import LayerSpec, layer_of
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError
from tests.assets.test_old_constructors import OLD, RESOURCES, WITH_EFFECT
from tests.helpers.gpu_door import run_child

from .conftest import SCENES, SMALL, pixels
from .effects_scenes import MASK_EDGE

#: The width of a Gaussian-blurred step between its 10 % and 90 % levels,
#: in standard deviations.
STEP_WIDTH = 2.563


#: The fixtures' frame height in scene units, CE's default; its width is
#: that times the render's aspect.
FRAME_HEIGHT = 8.0


def _frame_width(session) -> float:
    """The scene units across the session's frame, from the fixture's frame
    height and the render's pixel size.
    """
    camera = session.scene.renderer.camera
    assert camera.frame_height == FRAME_HEIGHT
    return FRAME_HEIGHT * camera.pixel_width / camera.pixel_height


def _px(session, length: float) -> float:
    """``length`` scene units in the render's pixels, computed here and not
    by the engine's conversion: the frame's pixel width over its width in
    scene units.
    """
    pixel_width = session.scene.renderer.camera.pixel_width
    return length * pixel_width / _frame_width(session)


def _edge_x(session, x: float) -> float:
    """The pixel coordinate (a pixel's left edge at its index) of the scene's
    vertical line at ``x``.
    """
    pixel_width = session.scene.renderer.camera.pixel_width
    return (x / _frame_width(session) + 0.5) * pixel_width


def _crossing(levels: np.ndarray, at: float, near: float, span: float) -> float:
    """Where ``levels`` (0 to 1, along a row) cross ``at`` within ``span``
    pixels of ``near``, in pixel coordinates, read by linear interpolation
    between pixel centres (pixel ``i``'s at ``i + 0.5``).
    """
    lo = max(0, math.floor(near - span))
    hi = min(len(levels), math.ceil(near + span) + 1)
    window = levels[lo:hi]
    side = window >= at
    changed = np.flatnonzero(side != side[0])
    assert changed.size, f"no crossing of {at} near {near}: {window}"
    i = int(changed[0])
    a, b = float(window[i - 1]), float(window[i])
    return lo + i - 0.5 + (a - at) / (a - b)


def _step_width(row: np.ndarray, near: float, span: float) -> float:
    """From where a step is at 90 % to where it is at 10 %, in pixels."""
    levels = row / 255.0
    return abs(_crossing(levels, 0.1, near, span) - _crossing(levels, 0.9, near, span))


def _span(sigma_px: float) -> float:
    """How far from its place a blurred edge's levels can move: its blur's
    reach and a few pixels more.
    """
    return math.ceil(3 * sigma_px) + 3


STEPS = [
    ("BlurredStep", 0.15),
    ("BlurredStepMedium", 0.3),
    ("BlurredStepWide", 0.6),
    # At depth 2 under a focus at 0 with aperture 0.15.
    ("FocusedStep", 0.3),
]


@pytest.mark.parametrize("size", [(96, 54), (384, 216)])
@pytest.mark.parametrize(("scene", "sigma"), STEPS)
def test_a_blurred_step_has_the_declared_width(session_of, door, size, scene, sigma):
    width, height = size
    session = session_of(scene, pixel_width=width, pixel_height=height)
    sigma_px = _px(session, sigma)
    row = pixels(session, Fraction(1, 10))[height // 2, :, 0]
    edge = _edge_x(session, 0)
    found = _step_width(row, edge, _span(sigma_px))
    assert found == pytest.approx(STEP_WIDTH * sigma_px, abs=1)
    assert door.draws == 1


MASKS = [
    ("MaskedPlane", False),
    ("MaskedPlaneInverted", True),
    ("MaskedPlaneSoft", False),
    ("MaskedPlaneSoftInverted", True),
    ("MaskedPlaneSofter", False),
    ("MaskedPlaneSofterInverted", True),
]


#: The feather of each masked plane, in scene units.
FEATHERS = {
    "MaskedPlane": 0.0,
    "MaskedPlaneInverted": 0.0,
    "MaskedPlaneSoft": 0.25,
    "MaskedPlaneSoftInverted": 0.25,
    "MaskedPlaneSofter": 0.6,
    "MaskedPlaneSofterInverted": 0.6,
}


@pytest.mark.parametrize(("scene", "inverted"), MASKS)
def test_a_mask_s_half_way_line_lies_on_its_shape_s_edge(
    session_of, door, scene, inverted
):
    session = session_of(scene, pixel_width=192, pixel_height=108)
    row = pixels(session, Fraction(1, 10))[54, :, 0] / 255.0
    edge = _edge_x(session, MASK_EDGE)
    span = _span(_px(session, 0.6))
    assert _crossing(row, 0.5, edge, span) == pytest.approx(edge, abs=1)
    feather = FEATHERS[scene]
    if feather:
        # A feather is a Gaussian of sigma feather / 3 across the edge (the
        # GPU build's own test of it): its 10-90 % width is 2.563 sigma.
        width = abs(_crossing(row, 0.1, edge, span) - _crossing(row, 0.9, edge, span))
        assert width == pytest.approx(STEP_WIDTH * _px(session, feather) / 3, abs=1)
    # The kept side is the mask's, or with the mask inverted the other.
    left, right = row[int(edge - span)], row[int(edge + span)]
    assert (left, right) == ((0.0, 1.0) if inverted else (1.0, 0.0))


def _covered(footprint, shape: tuple[int, int]) -> np.ndarray:
    """A footprint of the compositor's object map, over the whole frame."""
    found = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = footprint.box
    found[y0:y1, x0:x1] = footprint.mask
    return found


def _grown(covered: np.ndarray, reach: int) -> np.ndarray:
    """``covered`` grown by ``reach`` pixels along each axis in turn, as a
    separable kernel of that reach spreads it.
    """
    found = covered.copy()
    for axis in (0, 1):
        source = found.copy()
        for k in range(1, min(reach, source.shape[axis] - 1) + 1):
            ahead = [slice(None)] * 2
            behind = [slice(None)] * 2
            ahead[axis], behind[axis] = slice(k, None), slice(None, -k)
            found[tuple(ahead)] |= source[tuple(behind)]
            found[tuple(behind)] |= source[tuple(ahead)]
    return found


def _reach(sigma_px: float) -> int:
    """How far a blur of ``sigma_px`` pixels reaches: three sigma, rounded
    up, as the GPU build's kernel is cut.
    """
    return math.ceil(3 * sigma_px)


@pytest.mark.parametrize("effect", ["blur", "mask", "depth"])
def test_removing_an_effect_changes_no_pixel_outside_its_reach(
    session_of, door, effect
):
    size = {"pixel_width": 192, "pixel_height": 108}
    t = Fraction(1, 10)
    door.keep_maps = True
    twin = session_of("Removal", **size)
    plain = pixels(twin, t)
    # The square is the one layer with an empty spec; the stack names each
    # layer by its object's id in the record.
    record = twin.record
    (subject,) = {
        record.origin(mob).id
        for mob in record.mobjects()
        if layer_of(mob) == LayerSpec()
    }
    footprints = {each.object_id: each for each in door.maps[-1]}
    covered = _covered(footprints[subject], plain.shape[:2])
    assert covered.any()
    if effect == "blur":
        support = _grown(covered, _reach(_px(twin, 0.2)))
    elif effect == "depth":
        # At depth 3 under a focus at 0 with aperture 0.1.
        support = _grown(covered, _reach(_px(twin, 0.1 * 3)))
    else:
        # A mask multiplies the layer and spreads nothing.
        support = covered
    with_effect = pixels(session_of(f"Removal{effect.title()}", **size), t)
    changed = (with_effect != plain).any(axis=-1)
    assert changed.any()
    assert not (changed & ~support).any(), np.argwhere(changed & ~support)[:8]


def test_a_changing_blur_shows_its_width_at_every_tick(session_of, door):
    session = session_of("ChangingBlur")
    edge = _edge_x(session, 0)
    for tick in range(61):
        t = Fraction(tick, 60)
        sigma_px = _px(session, 0.2 * (1 - t) + 0.6 * t)
        row = pixels(session, t)[27, :, 0]
        found = _step_width(row, edge, _span(sigma_px))
        assert found == pytest.approx(STEP_WIDTH * sigma_px, abs=1), tick
    # Brought to 0, it leaves the step as Cairo draws it.
    ended = pixels(session, Fraction(8, 5))
    sharp = pixels(session_of("SharpStep"), Fraction(1, 10))
    assert int(np.abs(ended - sharp).max()) <= 1


def test_a_racked_focus_gives_each_depth_its_width(session_of, door):
    size = {"pixel_width": 192, "pixel_height": 108}
    session = session_of("RackingFocus", **size)
    edge = _edge_x(session, 0)
    aperture = 0.1
    # The focus moves from depth 0 to 4 over the first second.
    for t in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
        focus = 4 * t
        frame = pixels(session, t)
        for row, depth in ((27, 0), (81, 4)):
            sigma_px = _px(session, aperture * abs(depth - focus))
            found = _step_width(frame[row, :, 0], edge, _span(sigma_px))
            assert found == pytest.approx(STEP_WIDTH * sigma_px, abs=1), (t, depth)
    # Then the aperture closes to 0.05 at depth 4.
    frame = pixels(session, Fraction(8, 5))
    for row, depth in ((27, 0), (81, 4)):
        sigma_px = _px(session, 0.05 * abs(depth - 4))
        found = _step_width(frame[row, :, 0], edge, _span(sigma_px))
        assert found == pytest.approx(STEP_WIDTH * sigma_px, abs=1), depth


def test_a_short_change_beside_a_longer_animation_holds_its_target(session_of, door):
    session = session_of("MixedDurations", pixel_width=192, pixel_height=108)
    edge = _edge_x(session, 0)
    # Past the changes' half second, before the play's end: the upper step
    # at its blur's target, the lower one, at the focus, sharp.
    frame = pixels(session, Fraction(3, 4))
    for row, sigma in ((27, 0.6), (81, 0.0)):
        sigma_px = _px(session, sigma)
        found = _step_width(frame[row, :, 0], edge, _span(sigma_px))
        assert found == pytest.approx(STEP_WIDTH * sigma_px, abs=1), sigma


def test_the_server_refuses_a_blurred_layer_inside_a_blurred_group(tmp_path, door):
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    try:
        server.handle(
            "open", {"file": str(SCENES), "scene": "NestedBlur", "config": config}
        )
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.1, "path": str(tmp_path / "f.png")})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    assert refused.value.data["reason"] == "render.unsupported"
    assert door.draws == 0


#: Opens each old scene, then the scene with an effect drawn on this run's
#: device, then each old scene again, in one fresh process; prints each
#: opening's routes and frames.
PROGRAM = """\
import hashlib, json, sys
from arachne.compose import device
from arachne.session import Session
from tests.helpers.gpu_door import LaunchDoor

directory, media = sys.argv[1], sys.argv[2]
config = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4, "media_dir": media}
door = device.door = LaunchDoor()


def opened(file, scene):
    session = Session.open(f"{directory}/{file}", scene, config=config)
    try:
        return {
            "route": session.route,
            "routes": [play.route for play in session.record.plays],
            "frames": [
                hashlib.sha256(session.frame(t).tobytes()).hexdigest()
                for t in (0, 0.25, 0.5, 0.7)
            ],
        }
    finally:
        session.close()


fresh = {scene: opened("old.py", scene) for scene in ("OldSvg", "OldImage")}
effect = opened("with_effect.py", "WithEffect")
drawn = door.draws
after = {scene: opened("old.py", scene) for scene in ("OldSvg", "OldImage")}
print(json.dumps({"fresh": fresh, "effect": effect, "drawn": drawn, "after": after}))
"""


def test_old_scenes_draw_the_same_after_a_scene_with_an_effect_drew(tmp_path):
    import shutil

    shutil.copy(RESOURCES / "heart.svg", tmp_path / "heart.svg")
    shutil.copy(RESOURCES / "tree_img_640x351.png", tmp_path / "tree.png")
    (tmp_path / "old.py").write_text(textwrap.dedent(OLD))
    (tmp_path / "with_effect.py").write_text(textwrap.dedent(WITH_EFFECT))
    printed = run_child(PROGRAM, str(tmp_path), str(tmp_path / "media"))
    found = json.loads(printed.strip().splitlines()[-1])
    assert found["effect"]["route"] == "gpu"
    assert found["drawn"] > 0
    assert found["fresh"] == found["after"]
    for opening in found["fresh"].values():
        assert opening["route"] == "cpu"
        assert len(set(opening["frames"])) > 1
