"""Scenes of rigs whose parts are stills, for the raster tests, and the
geometry their pictures are expected at, computed here from the fixture's
manifest alone. A line a test names carries a ``# @tag``.
"""

from __future__ import annotations

import json
import math

import numpy as np

from arachne import (
    Dot,
    FadeIn,
    ImageMobject,
    MovingCameraScene,
    Rig,
    Scene,
    Square,
)
from arachne.assets.reader import FIXTURES
from arachne.compose import set_focus
from arachne.effects import set_blur
from arachne.utils.rate_functions import linear

FLAP = FIXTURES / "hinged-flap"
GROUND = FIXTURES / "layered-ground"

#: How the posed scenes pose, show and move the flap rig.
LIFT, GROW, SHOWN = 30.0, 1.2, "off"
SHIFT = np.array([-1.5, 0.5, 0.0])
#: How the layered scenes move the camera, and the focus they set.
PAN = np.array([2.0, 0.0, 0.0])
FOCUS, APERTURE = 1.0, 0.04


def manifest(directory) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def canvas_to_scene(directory, points: np.ndarray) -> np.ndarray:
    """Canvas points (y down) where the manifest's ``size`` places them."""
    size = manifest(directory)["size"]
    scale, (ox, oy) = size["scene_units_per_unit"], size["origin"]
    points = np.asarray(points, dtype=float)
    return np.column_stack(
        [
            scale * (points[:, 0] - ox),
            -scale * (points[:, 1] - oy),
            np.zeros(len(points)),
        ]
    )


def turned(points: np.ndarray, pivot, degrees: float) -> np.ndarray:
    """``points`` on the canvas turned counterclockwise as drawn (the
    canvas's y points down) about ``pivot``.
    """
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    x, y = points[:, 0] - pivot[0], points[:, 1] - pivot[1]
    return np.column_stack([pivot[0] + c * x + s * y, pivot[1] - s * x + c * y])


def box_corners(box) -> np.ndarray:
    """A box's corners in CE's image order: upper left, upper right, lower
    left, lower right (the canvas's y points down).
    """
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=float)


def flap_pictures(lift: float, grow: float, shown: str) -> dict[str, np.ndarray]:
    """Each shown member of the flap rig at its natural place, posed: its
    picture's corners in the scene, in CE's image order.
    """
    found = manifest(FLAP)
    parts = {part["name"]: part for part in found["parts"]}
    (badge,) = found["slots"]
    drawing = next(d for d in badge["drawings"] if d["name"] == shown)
    flap = turned(box_corners(parts["flap"]["box"]), parts["flap"]["pivot"], lift)
    pivot = np.array(badge["pivot"], dtype=float)
    grown = pivot + grow * (box_corners(drawing["box"]) - pivot)
    return {
        "body": canvas_to_scene(FLAP, box_corners(parts["body"]["box"])),
        "flap": canvas_to_scene(FLAP, flap),
        shown: canvas_to_scene(FLAP, grown),
    }


def files_of(directory) -> dict[str, str]:
    """Each member's file, by the part's or drawing's name."""
    found = manifest(directory)
    files = {part["name"]: part["file"] for part in found["parts"]}
    for slot in found.get("slots", []):
        files |= {d["name"]: d["file"] for d in slot["drawings"]}
    return files


def depth_of(name: str) -> float:
    """A layer's depth in the layered set."""
    return next(p["depth"] for p in manifest(GROUND)["parts"] if p["name"] == name)


def layer_pictures() -> dict[str, np.ndarray]:
    """Each layer of the layered set at its natural place."""
    return {
        part["name"]: canvas_to_scene(GROUND, box_corners(part["box"]))
        for part in manifest(GROUND)["parts"]
    }


def images_at(directory, corners: dict[str, np.ndarray]) -> dict[str, ImageMobject]:
    """The members' stills opened by path, each at its ``corners``, by the
    member's name, in the order given.
    """
    files = files_of(directory)
    found = {}
    for name, at in corners.items():
        image = ImageMobject(str(directory / files[name]))
        image.points = at
        found[name] = image
    return found


class PosedFlap(Scene):
    """The flap rig posed, its slot switched, moved off the origin."""

    def construct(self):
        self.add(Square(side_length=6, fill_opacity=1, color="#203040"))
        rig = Rig("fixture:hinged-flap")  # @posed-flap
        rig.pose(lift=LIFT, grow=GROW)
        rig.show(badge=SHOWN)
        rig.shift(SHIFT)
        self.add(rig)
        self.wait(0.5)


class PosedFlapAsImages(Scene):
    """:class:`PosedFlap`'s pictures, opened by path and placed where the
    manifest says.
    """

    def construct(self):
        self.add(Square(side_length=6, fill_opacity=1, color="#203040"))
        for image in images_at(FLAP, flap_pictures(LIFT, GROW, SHOWN)).values():
            self.add(image.shift(SHIFT))
        self.wait(0.5)


class FadingFlap(Scene):
    """The flap rig faded in while it lifts, its own updater suspended."""

    def construct(self):
        rig = Rig("fixture:hinged-flap")
        self.play(FadeIn(rig), rate_func=linear, run_time=0.5)
        self.wait(0.25)


class LayeredPan(MovingCameraScene):
    """The layered set viewed through the camera, which pans while the set
    fades in, a dot kept on the middle layer's anchor by an updater.
    """

    def construct(self):
        rig = Rig("fixture:layered-ground")
        rig.view_through(self.camera)
        peak = rig.anchor("peak")
        dot = Dot(radius=0.05)
        dot.add_updater(lambda m: m.move_to(peak.point()))
        self.add(dot)
        self.play(
            self.camera.frame.animate.shift(PAN),
            FadeIn(rig),
            rate_func=linear,
            run_time=1,
        )
        self.wait(0.25)


class LayeredFocus(Scene):
    """The layered set under a focus."""

    def construct(self):
        self.add(Square(side_length=7, fill_opacity=1, color="#f0e0c0"))
        set_focus(self.camera, FOCUS, APERTURE)
        self.add(Rig("fixture:layered-ground"))
        self.wait(0.25)


class LayeredFocusAsImages(Scene):
    """:class:`LayeredFocus`'s layers opened by path, each blurred by what
    its depth declares under the focus.
    """

    def construct(self):
        self.add(Square(side_length=7, fill_opacity=1, color="#f0e0c0"))
        for name, image in images_at(GROUND, layer_pictures()).items():
            set_blur(image, APERTURE * abs(depth_of(name) - FOCUS))
            self.add(image)
        self.wait(0.25)


class VectorRig(Scene):
    """A vector fixture rig: a CPU frame."""

    def construct(self):
        self.add(Rig("fixture:lidded-box"))
        self.wait(0.25)
