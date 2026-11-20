"""Scenes the window tests open.

Each case is a group of objects with an opaque backdrop that fills the
view, and a small square that moves over two seconds. ``Alone<Case>`` is a
scene of those objects alone; ``Shown<Case><Size>`` shows them in a
:class:`~arachne.Window` over a contrasting parent, the window placed at
whole pixels at half or a third of the frame's size; a ``Without<Name>``
suffix leaves those objects out of the window. Every scene's frame is
:data:`FRAME`: 8 pixels to a scene unit, so every place below falls on
numbers a float holds exactly.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    RIGHT,
    Circle,
    Group,
    ImageMobject,
    Rectangle,
    Scene,
    Window,
    linear,
    set_blur,
    set_depth,
    set_focus,
    set_mask,
)
from arachne.compose.layer import LayerSpec, set_layer

#: The frame every scene here is drawn in.
FRAME = {
    "pixel_width": 96,
    "pixel_height": 54,
    "frame_width": 12.0,
    "frame_height": 6.75,
    "frame_rate": 15,
}
#: Pixels to a scene unit in :data:`FRAME`.
SCALE = 8
#: Where each size of window stands in :data:`FRAME`: its upper left
#: corner and its size, in pixels.
SIZES = {"Half": ((24, 13), (48, 27)), "Third": ((8, 4), (32, 18))}

PARENT = "#FF00FF"
#: The parent's colour as a frame holds it.
PARENT_RGBA = (255, 0, 255, 255)
BACKDROP = "#203060"
#: Close enough to the backdrop that a dot of it, missing, is no cluster.
FAINT = "#283868"

#: The depth in focus and the aperture of the focused case.
FOCUS = (0.0, 0.25)

#: The objects of each case, in the order the window lists them.
NAMES = {
    "Plain": ("backdrop", "square", "circle", "dot", "mover"),
    "Blurred": ("backdrop", "edge", "mover"),
    "Masked": ("backdrop", "plane", "mover"),
    "Focused": ("backdrop", "edge", "mover"),
    "Nested": ("backdrop", "inner", "mover"),
    "MixedMask": ("backdrop", "plane", "mover"),
}


def _plane(width: float, height: float, color: str) -> Rectangle:
    return Rectangle(
        width=width, height=height, fill_color=color, fill_opacity=1, stroke_width=0
    )


def _edge() -> Rectangle:
    """A white square across the view's left edge."""
    return _plane(3, 3, "#FFFFFF").move_to([-6, 0.5, 0])


def content(case: str, without: tuple[str, ...] = ()) -> dict:
    """The objects of ``case``, by name, but those ``without`` names."""
    made = {"backdrop": _plane(12, 6.75, BACKDROP)}
    if case == "Plain":
        made["square"] = _plane(2, 2, "#F0D020").move_to([-3, 1.5, 0])
        made["circle"] = Circle(
            radius=1, fill_color="#30C060", fill_opacity=1, stroke_width=0
        ).move_to([3, -1, 0])
        made["dot"] = Circle(
            radius=0.3, fill_color=FAINT, fill_opacity=1, stroke_width=0
        ).move_to([1.5, 1.5, 0])
    elif case == "Blurred":
        made["edge"] = _edge()
        set_blur(made["edge"], 0.5)
    elif case == "Focused":
        made["edge"] = _edge()
        set_depth(made["edge"], 2)
    elif case in ("Masked", "MixedMask"):
        made["plane"] = _plane(6, 6.75, "#FFFFFF").move_to([-3, 0, 0])
        if case == "Masked":
            mask = Circle(radius=2, fill_opacity=1).move_to([-6, 0, 0])
            set_mask(made["plane"], mask, feather=0.5)
        else:
            # A mask of two kinds: a vector circle and an image.
            image = ImageMobject(np.full((8, 8, 4), 255, dtype=np.uint8))
            image.scale_to_fit_height(2).move_to([-1.5, -1.5, 0])
            mask = Group(Circle(radius=2, fill_opacity=1).move_to([-6, 0, 0]), image)
            set_mask(made["plane"], mask)
    elif case == "Nested":
        inner = Window(
            _plane(12, 6.75, "#E08020"),
            Circle(
                radius=1.5, fill_color="#4080FF", fill_opacity=1, stroke_width=0
            ).move_to([2, 0, 0]),
            width=4,
            height=2.25,
        )
        made["inner"] = inner.move_to([-3, 1.75, 0])
    made["mover"] = _plane(1, 1, "#FFFFFF").move_to([-1, -2, 0])
    assert tuple(made) == NAMES[case], (case, tuple(made))
    return {name: mob for name, mob in made.items() if name not in without}


def _run(scene: Scene, case: str, objects: dict) -> None:
    """The case's focus, if it has one, then its two seconds: the mover
    moving, or a wait where it is left out.
    """
    if case == "Focused":
        set_focus(scene, *FOCUS)
    mover = objects.get("mover")
    if mover is None:
        scene.wait(2)
    else:
        scene.play(mover.animate.shift(2 * RIGHT), run_time=2, rate_func=linear)


class _Alone(Scene):
    """A case's objects alone; ``content`` holds them by name."""

    case = "Plain"

    def construct(self):
        self.content = content(self.case)
        self.add(*self.content.values())
        _run(self, self.case, self.content)


class _Shown(Scene):
    """A case's objects, but ``without``, in a window of ``size`` over the
    parent.
    """

    case = "Plain"
    size = "Half"
    without: tuple[str, ...] = ()

    def construct(self):
        objects = content(self.case, self.without)
        (x0, y0), (width, height) = SIZES[self.size]
        left, top = x0 / SCALE - 6, 3.375 - y0 / SCALE
        window = Window(
            *objects.values(), width=width / SCALE, height=height / SCALE
        ).move_to([left + width / (2 * SCALE), top - height / (2 * SCALE), 0])
        self.add(_plane(12, 6.75, PARENT), window)
        _run(self, self.case, objects)


def shown_name(case: str, size: str, without: tuple[str, ...] = ()) -> str:
    """The name of the scene showing ``case`` in a window of ``size``,
    ``without`` those objects.
    """
    return f"Shown{case}{size}" + "".join(f"Without{n.title()}" for n in without)


def _variants() -> dict[str, type[Scene]]:
    found: dict[str, type[Scene]] = {}
    sizes = {"Plain": ("Half", "Third")}
    for case, names in NAMES.items():
        found[f"Alone{case}"] = type(f"Alone{case}", (_Alone,), {"case": case})
        for size in sizes.get(case, ("Half",)):
            omitted = [(), *((name,) for name in names)]
            if case == "Plain" and size == "Half":
                # The planted window, which lacks the dot, and it lacking
                # each other object too.
                omitted += [("dot", name) for name in names if name != "dot"]
            for without in omitted:
                name = shown_name(case, size, without)
                found[name] = type(
                    name, (_Shown,), {"case": case, "size": size, "without": without}
                )
    return found


globals().update(_variants())


class WindowTooLargeForMixedMask(Scene):
    """The mixed mask's objects in a window larger than the frame."""

    def construct(self):
        window = Window(*content("MixedMask").values(), width=16, height=9)
        self.add(window)
        self.wait(0.2)


class BlurInAndOut(Scene):
    """One blur, of half a scene unit, on an object of the frame and on an
    object in a window of half the frame's size; a mask's feather and the
    focus's aperture in the same window.
    """

    def construct(self):
        outside = _plane(1, 1, "#FFFFFF").move_to([5, -2.5, 0])
        set_blur(outside, 0.5)
        blurred = _plane(2, 2, "#FFFFFF").move_to([-3, 0, 0])
        set_blur(blurred, 0.5)
        masked = _plane(2, 2, "#FFFFFF").move_to([3, 0, 0])
        set_mask(masked, Circle(radius=1).move_to([3, 0, 0]), feather=0.5)
        window = Window(blurred, masked, width=6, height=3.375).move_to([0, 0.0625, 0])
        self.add(outside, window)
        set_focus(self, *FOCUS)
        self.wait(0.2)


class OffGrid(Scene):
    """A window whose rectangle is no whole number of pixels, and less than
    one pixel tall.
    """

    def construct(self):
        window = Window(_plane(12, 6.75, BACKDROP), width=4.3, height=0.05)
        self.add(window.move_to([0.1, 0.3, 0]))
        self.wait(0.2)


#: The corners, in scene units and CE's image order, of an image carrier
#: placed level, mirrored left to right, turned a quarter, and tilted.
CARRIER_CORNERS = {
    "level": [(-2, 1), (2, 1), (-2, -1), (2, -1)],
    "mirrored": [(2, 1), (-2, 1), (2, -1), (-2, -1)],
    "quarter": [(-1, -2), (-1, 2), (1, -2), (1, 2)],
}


class ImageCarrier(Scene):
    """A window shown in an image's four corners, placed as ``placed`` says
    (:data:`CARRIER_CORNERS`, or ``"tilted"``: level, then turned 30
    degrees).
    """

    placed = "level"

    def construct(self):
        carrier = ImageMobject(np.zeros((4, 4, 4), dtype=np.uint8))
        if self.placed == "tilted":
            carrier.points = np.array(
                [[x, y, 0.0] for x, y in CARRIER_CORNERS["level"]]
            )
            carrier.rotate(np.pi / 6)
        else:
            corners = CARRIER_CORNERS[self.placed]
            carrier.points = np.array([[x, y, 0.0] for x, y in corners])
        objects = content("Plain")
        set_layer(carrier, LayerSpec(window=tuple(objects.values())))
        self.add(carrier)
        self.wait(0.2)


class ImageCarrierMirrored(ImageCarrier):
    placed = "mirrored"


class ImageCarrierQuarter(ImageCarrier):
    placed = "quarter"


class ImageCarrierTilted(ImageCarrier):
    placed = "tilted"


class BlurTooWideForWindow(Scene):
    """A blur of 6 scene units in a window of 16 x 9 pixels: 8 pixels of
    sigma there, whose reach of 24 pixels is past the window's longer side.
    """

    def construct(self):
        blurred = _plane(2, 2, "#FFFFFF")
        set_blur(blurred, 6)
        window = Window(_plane(12, 6.75, BACKDROP), blurred, width=2, height=1.125)
        self.add(window)
        self.wait(0.2)
