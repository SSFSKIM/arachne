"""The eight class probes as engine scenes.

The core's class probes (``core/parity/probes.rs``) are XF streams in which a
painter's order draws pixels wrong. These are the same scenes written with the
engine's API, the same views, shapes, colours and draw orders, so that the
core draws them from what the engine extracts, and so that CE draws them
too: the file renders under Arachne, and under CE 0.20.1 with
``PROBES_ENGINE=manim`` (``ce_check.py``).

Each probe's checks are the Rust probe's pixels (the scenes' geometry puts
the same world points there), or, where that pixel lies on a face seam or off
the sheet the check is about in CE's surfaces, the nearest one inside it. A
check either names what must not show at its
pixel (the core draws there what the scene draws without it: ``hide``) or
says the core draws what CE draws there (``hide`` empty). ``ce`` is CE's
pixel there, as Arachne's Cairo path draws it and as real CE 0.20.1 renders
it (``ce_check.py``); "(CE right)" marks a check where CE draws what the core
does.

Unlike the Rust probes, the sheets here are CE's surfaces: lit by CE's light
and drawn by CE face by face, each face sorted by its centre's depth. Where
that sort orders crossing faces as depth does, CE draws the pixel right.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

if os.environ.get("PROBES_ENGINE") == "manim":
    from manim import (
        DEGREES,
        ManimColor,
        Sphere,
        Surface,
        ThreeDScene,
        VMobject,
    )
else:
    from arachne import (
        DEGREES,
        ManimColor,
        Sphere,
        Surface,
        ThreeDScene,
        VMobject,
    )

#: The probes' frame, as the Rust probes': CE's frame width at 16:9.
WIDTH, HEIGHT = 320, 180
#: One frame per wait.
FPS = 15

RED = (0.85, 0.2, 0.15, 1.0)
BLUE = (0.15, 0.3, 0.85, 1.0)
GREEN = (0.2, 0.6, 0.3, 1.0)
GREY = (0.5, 0.5, 0.5, 1.0)
YELLOW = (1.0, 0.85, 0.2, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
ORANGE = (0.95, 0.5, 0.1, 1.0)


@dataclass(frozen=True)
class Check:
    """A pixel of a probe's frame: what must not show there (``hide``, the
    probe's attribute names; empty when the core draws CE's pixel), CE's
    bytes there, and what it is.
    """

    frame: int
    x: int
    y: int
    hide: tuple[str, ...]
    ce: tuple[int, int, int, int]
    what: str


def _colour(rgba: tuple[float, ...]) -> Any:
    return ManimColor(rgba[:3])


def _sheet(
    corners: list[list[float]], nu: int, nv: int, rgba: tuple[float, ...]
) -> Any:
    """A quad's bilinear surface of ``nu x nv`` faces in one colour, no stroke."""
    c = np.array(corners, dtype=float)

    def place(u: float, v: float) -> np.ndarray:
        return (c[0] + (c[1] - c[0]) * u) * (1 - v) + (c[3] + (c[2] - c[3]) * u) * v

    surface = Surface(
        place,
        u_range=(0, 1),
        v_range=(0, 1),
        resolution=(nu, nv),
        checkerboard_colors=False,
        fill_color=_colour(rgba),
        fill_opacity=rgba[3],
        stroke_width=0,
    )
    return surface


def _level(z: float, hx: float, hy: float) -> list[list[float]]:
    return [[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z]]


def _polyline(points: list[list[float]], rgba: tuple[float, ...], width: float) -> Any:
    """A stroked curve through ``points``, one straight cubic per step."""
    curve = VMobject(
        stroke_color=_colour(rgba), stroke_width=width, stroke_opacity=rgba[3]
    )
    curve.set_points_as_corners(np.array(points, dtype=float))
    return curve


def _closed(n: int, f: Any, rgba: tuple[float, ...], width: float) -> Any:
    return _polyline([list(f((i % n) / n)) for i in range(n + 1)], rgba, width)


def _square(cx: float, cy: float, cz: float, h: float) -> list[list[float]]:
    return [
        [cx - h, cy - h, cz],
        [cx + h, cy - h, cz],
        [cx + h, cy + h, cz],
        [cx - h, cy + h, cz],
        [cx - h, cy - h, cz],
    ]


def _filled(points: list[list[float]], rgba: tuple[float, ...]) -> Any:
    shape = VMobject(fill_color=_colour(rgba), fill_opacity=rgba[3], stroke_width=0)
    shape.set_points_as_corners(np.array(points, dtype=float))
    return shape


class Probe(ThreeDScene):
    """A probe: ``build`` makes its objects as attributes, in draw order;
    ``hide`` names those left out (the twin a check compares with).
    """

    phi = 0.0
    theta = 0.0
    hide: tuple[str, ...] = ()
    CHECKS: tuple[Check, ...] = ()

    def construct(self) -> None:
        self.set_camera_orientation(phi=self.phi * DEGREES, theta=self.theta * DEGREES)
        self.build()
        self.wait(1 / FPS)

    def build(self) -> None:
        raise NotImplementedError

    def put(self, name: str, mobject: Any, how: str = "add") -> None:
        """Add ``mobject`` as ``name`` unless the twin leaves it out."""
        setattr(self, name, mobject)
        if name in self.hide:
            return
        if how == "fixed_in_frame":
            self.add_fixed_in_frame_mobjects(mobject)
        elif how == "fixed_orientation":
            self.add_fixed_orientation_mobjects(mobject)
        else:
            self.add(mobject)


class IntersectingSurfaces(Probe):
    """Two opaque sheets crossing: a level one, and one tilted through it."""

    phi, theta = 62.0, -50.0
    CHECKS = (
        Check(
            0,
            151,
            91,
            ("tilt",),
            (255, 90, 77, 255),
            "the level sheet in front where the tilted one dips below it (CE right)",
        ),
        Check(
            0,
            199,
            79,
            ("level",),
            (95, 133, 255, 255),
            "the tilted sheet in front where it rises above (CE right)",
        ),
    )

    def build(self) -> None:
        self.put("level", _sheet(_level(0.0, 3.0, 2.0), 6, 4, RED))
        tilt = [
            [-3.0, -2.0, -1.5],
            [3.0, -2.0, 1.5],
            [3.0, 2.0, 1.5],
            [-3.0, 2.0, -1.5],
        ]
        self.put("tilt", _sheet(tilt, 6, 4, BLUE))


class CurveBehindSurface(Probe):
    """A curve in a vertical plane, its lower arc under an opaque level sheet."""

    phi, theta = 60.0, -50.0
    CHECKS = (
        Check(
            0,
            160,
            112,
            ("curve",),
            (255, 217, 51, 255),
            "the curve's lower arc, under the sheet",
        ),
        Check(
            0,
            160,
            65,
            (),
            (255, 217, 51, 255),
            "the curve's upper arc, above the sheet (CE right)",
        ),
    )

    def build(self) -> None:
        self.put("floor", _sheet(_level(0.0, 4.0, 3.0), 8, 6, GREEN))
        tau = 2 * np.pi
        self.put(
            "curve",
            _closed(
                96,
                lambda t: [2.2 * np.cos(tau * t), 0.0, 1.2 * np.sin(tau * t)],
                YELLOW,
                24.0,
            ),
        )


class TranslucentOverOpaque(Probe):
    """Translucent sheets over an opaque one: one crossing it, one above both."""

    phi, theta = 58.0, -40.0
    CHECKS = (
        Check(
            0,
            118,
            94,
            ("tilt",),
            (176, 176, 176, 255),
            "the opaque sheet where the translucent one dips below it (CE right)",
        ),
        Check(
            0,
            173,
            105,
            (),
            (208, 130, 124, 255),
            "the translucent sheet over the opaque one where it rises (CE right)",
        ),
        Check(
            0,
            168,
            59,
            (),
            (116, 137, 198, 255),
            "the high sheet over both, in depth order (CE right)",
        ),
    )

    def build(self) -> None:
        self.put("ground", _sheet(_level(0.0, 3.5, 2.5), 7, 5, GREY))
        tilt = [
            [-3.0, -2.0, -1.2],
            [3.0, -2.0, 1.2],
            [3.0, 2.0, 1.2],
            [-3.0, 2.0, -1.2],
        ]
        self.put("tilt", _sheet(tilt, 6, 4, (0.9, 0.15, 0.1, 0.5)))
        self.put("high", _sheet(_level(1.6, 1.4, 1.0), 4, 3, (0.1, 0.3, 0.95, 0.4)))


class FixedLabelOverSurface(Probe):
    """A label fixed in frame over a surface, and a fixed-orientation label
    below the surface.
    """

    phi, theta = 55.0, -60.0
    CHECKS = (
        Check(
            0,
            160,
            101,
            (),
            (255, 255, 255, 255),
            "the fixed label on top, though drawn before the surface (CE right)",
        ),
        Check(
            0,
            181,
            104,
            ("tag",),
            (243, 128, 25, 255),
            "the fixed-orientation label, below the surface",
        ),
    )

    def build(self) -> None:
        label = _filled(_square(0.0, -0.5, 0.0, 0.6), WHITE)
        label.set_stroke(_colour((0.1, 0.1, 0.1, 1.0)), width=8.0)
        self.put("label", label, "fixed_in_frame")
        self.put("floor", _sheet(_level(0.0, 4.0, 3.0), 8, 6, GREEN))
        self.put(
            "tag", _filled(_square(0.8, 0.6, -0.9, 0.5), ORANGE), "fixed_orientation"
        )


class DepthRuleBothWays(Probe):
    """Ordinary objects on both sides of a wall: one behind it, drawn after
    it with a higher z-index; one in front, drawn before it with a lower one.
    """

    phi, theta = 65.0, -70.0
    CHECKS = (
        Check(
            0,
            170,
            71,
            ("behind",),
            (25, 217, 230, 255),
            "a line behind the surface, drawn after it",
        ),
        Check(
            0,
            147,
            84,
            (),
            (255, 217, 51, 255),
            "a line in front of the surface, drawn before it (CE right)",
        ),
    )

    def build(self) -> None:
        front = _polyline([[-2.0, -1.5, 0.9], [2.0, -1.5, 0.9]], YELLOW, 30.0)
        front.set_z_index(-1)
        self.put("front", front)
        wall = [[-3.0, 0.0, -1.5], [3.0, 0.0, -1.5], [3.0, 0.0, 2.0], [-3.0, 0.0, 2.0]]
        self.put("wall", _sheet(wall, 6, 4, (0.5, 0.2, 0.7, 1.0)))
        behind = _polyline(
            [[-2.0, 1.5, 0.3], [2.0, 1.5, 0.3]], (0.1, 0.85, 0.9, 1.0), 30.0
        )
        behind.set_z_index(2)
        self.put("behind", behind)


class StrokeCrossingSurface(Probe):
    """A thick straight stroke rising through an opaque level sheet."""

    phi, theta = 60.0, -35.0
    # The Rust probe's first pixel (144, 82), under the stroke at x = -2, is
    # on the seam between two of the surface's faces here (the Rust sheet is
    # one mesh), where the stroke behind shows through the antialiased edge.
    # This one is under the stroke at x = -2.5, inside a face.
    CHECKS = (
        Check(0, 139, 81, ("line",), (243, 128, 25, 255), "the stroke below the sheet"),
        Check(
            0, 197, 92, (), (243, 128, 25, 255), "the stroke above the sheet (CE right)"
        ),
    )

    def build(self) -> None:
        self.put("floor", _sheet(_level(0.0, 4.0, 3.0), 8, 6, (0.1, 0.15, 0.45, 1.0)))
        self.put("line", _polyline([[-3.0, 0.5, -1.0], [3.0, 0.5, 1.0]], ORANGE, 40.0))


class EdgeOnSurface(Probe):
    """A sheet seen exactly edge-on, a ring tilted a degree out of its
    plane, and a wall in front of part of the ring.
    """

    phi, theta = 90.0, -90.0
    CHECKS = (
        Check(
            0,
            216,
            90,
            ("ring",),
            (255, 217, 51, 255),
            "the ring where its near arc is behind the wall",
        ),
        Check(
            0,
            160,
            89,
            (),
            (255, 217, 51, 255),
            "the ring's near arc in front of the wall, on the far arc's row (CE right)",
        ),
        Check(
            0,
            136,
            71,
            (),
            (50, 152, 152, 255),
            "the wall above the edge-on sheet, which covers nothing (CE right)",
        ),
    )

    def build(self) -> None:
        self.put("floor", _sheet(_level(0.0, 3.0, 3.0), 6, 6, RED))
        wall = [
            [-3.5, -1.0, -2.0],
            [3.5, -1.0, -2.0],
            [3.5, -1.0, 2.0],
            [-3.5, -1.0, 2.0],
        ]
        self.put("wall", _sheet(wall, 7, 4, (0.1, 0.5, 0.5, 1.0)))
        tau, tilt = 2 * np.pi, DEGREES

        def ring(t: float) -> list[float]:
            s = 2.5 * np.sin(tau * t)
            return [2.5 * np.cos(tau * t), s * np.cos(tilt), s * np.sin(tilt)]

        self.put("ring", _closed(120, ring, YELLOW, 30.0))


class Orbit(Probe):
    """A lit sphere ringed by a curve and a title fixed in frame, the camera
    orbiting once over eight frames: in every frame the ring's far side is
    behind the sphere.
    """

    phi, theta = 65.0, -90.0
    FRAMES = 8
    CHECKS = tuple(
        Check(
            f,
            160,
            63,
            ("ring",),
            (255, 255, 255, 255),
            "the ring's far side, behind the sphere",
        )
        for f in range(8)
    )

    def construct(self) -> None:
        self.set_camera_orientation(phi=self.phi * DEGREES, theta=self.theta * DEGREES)
        self.build()
        for f in range(self.FRAMES):
            theta = self.theta + 360.0 * f / self.FRAMES
            self.set_camera_orientation(theta=theta * DEGREES)
            self.wait(1 / FPS)

    def build(self) -> None:
        ball = Sphere(radius=1.5, resolution=(30, 16))
        ball.set_fill(_colour(ORANGE), opacity=1.0)
        ball.set_stroke(width=0)
        self.put("ball", ball)
        tau = 2 * np.pi
        self.put(
            "ring",
            _closed(
                128,
                lambda t: [2.4 * np.cos(tau * t), 2.4 * np.sin(tau * t), 0.3],
                WHITE,
                24.0,
            ),
        )
        title = [
            [-6.5, 3.3, 0.0],
            [-3.5, 3.3, 0.0],
            [-3.5, 3.7, 0.0],
            [-6.5, 3.7, 0.0],
            [-6.5, 3.3, 0.0],
        ]
        self.put("title", _filled(title, WHITE), "fixed_in_frame")


PROBES = (
    IntersectingSurfaces,
    CurveBehindSurface,
    TranslucentOverOpaque,
    FixedLabelOverSurface,
    DepthRuleBothWays,
    StrokeCrossingSurface,
    EdgeOnSurface,
    Orbit,
)


def twin(probe: type, hide: tuple[str, ...]) -> type:
    """``probe`` without the objects ``hide`` names."""
    return type(
        f"{probe.__name__}Without{''.join(h.title() for h in hide)}",
        (probe,),
        {"hide": hide},
    )
