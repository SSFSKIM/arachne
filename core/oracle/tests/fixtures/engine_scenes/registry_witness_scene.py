"""The registry's invisible offer, drawn by the engine.

Two half-transparent lines 60 wide on z = 0, red then blue, then a polygon on z = 0 that
paints nothing (zero fill and zero stroke opacity), under a 3D camera at phi 0.8 over a
160 x 90 frame. A record that paints nothing is offered to the registry, so its flat identity
may attach a line and move the pair from paint order to depth.
"""

from __future__ import annotations

import numpy as np

from arachne import BLUE, GREEN, RED, Line, Polygon, ThreeDScene

FPS = 15


class InvisibleRegistryOffer(ThreeDScene):
    """The lines at y = -0.1 and 0.1 from x = -3 to 3, the polygon over [-4, 4] x [-1, 0]."""

    def construct(self):
        self.set_camera_orientation(phi=0.8, theta=-np.pi / 2, gamma=0.0)
        self.renderer.camera.should_apply_shading = False
        red = Line([-3.0, -0.1, 0.0], [3.0, -0.1, 0.0], stroke_width=60.0)
        red.set_stroke(RED, opacity=0.5)
        blue = Line([-3.0, 0.1, 0.0], [3.0, 0.1, 0.0], stroke_width=60.0)
        blue.set_stroke(BLUE, opacity=0.5)
        hidden = Polygon(
            [-4.0, -1.0, 0.0],
            [4.0, -1.0, 0.0],
            [4.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            fill_color=GREEN,
            fill_opacity=0.0,
            stroke_opacity=0.0,
        )
        self.add(red, blue, hidden)
        self.wait(1 / FPS)
