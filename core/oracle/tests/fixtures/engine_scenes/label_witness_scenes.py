"""Two label witnesses drawn by the engine: a ring glyph kept facing the camera whose hole holds
its centre over a surface's fill, and a label drawn through a camera of the scene's own, whose
records are pre-projected. The 3D camera's defaults (phi 0, theta -90 degrees, focal distance 20)
over a 16 x 16 unit frame of 64 x 64 pixels.
"""

from __future__ import annotations

import numpy as np

from arachne import BLUE, WHITE, Polygon, Surface, ThreeDCamera, ThreeDScene, VMobject

FPS = 15


def _square(x0, x1, y0, y1, z, **style):
    return Polygon([x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], **style)


class RingCentreInHole(ThreeDScene):
    """A square ring at z = -4, its hole wound the other way, its centre (0, 0, -4) in the hole;
    two fills of one surface at z = 0: S1 in the hole, holding the centre and none of the ring's
    order points, and S2 inside the ring, nearer, missing the centre.
    """

    def construct(self):
        holder = Surface(
            lambda u, v: np.array([u, v, 0.0]),
            u_range=[-6.0, -5.9],
            v_range=[-6.0, -5.9],
            resolution=(1, 1),
            fill_opacity=0.0,
            stroke_width=0.0,
        )
        style = {"fill_color": BLUE, "fill_opacity": 1.0, "stroke_width": 0.0}
        holder.add(
            _square(-0.25, 0.25, -0.25, 0.25, 0.0, **style),
            _square(1.0, 1.25, -0.25, 0.25, 0.0, **style),
        )
        ring = VMobject(fill_color=WHITE, fill_opacity=1.0, stroke_width=0.0)
        z = -4.0
        ring.set_points_as_corners(
            [
                [-1.5, -1.5, z],
                [1.5, -1.5, z],
                [1.5, 1.5, z],
                [-1.5, 1.5, z],
                [-1.5, -1.5, z],
            ]
        )
        ring.start_new_path(np.array([-0.5, -0.5, z]))
        ring.add_points_as_corners(
            [[-0.5, 0.5, z], [0.5, 0.5, z], [0.5, -0.5, z], [-0.5, -0.5, z]]
        )
        self.add_fixed_orientation_mobjects(ring)
        self.add(holder, ring)
        self.wait(1 / FPS)


class SameCamera(ThreeDCamera):
    """A camera class of the scene's own whose projection is the engine's: its records are
    pre-projected.
    """

    def project_points(self, points):
        return super().project_points(points)


class LabelPreProjected(ThreeDScene):
    """W5's scene through :class:`SameCamera`: a polygon G over [-3, 3]^2 at z = 2, then a label,
    a square [-1.5, 1.5]^2 kept facing the camera about the origin, every record pre-projected.
    """

    def __init__(self, **kwargs):
        super().__init__(camera_class=SameCamera, **kwargs)

    def construct(self):
        g = _square(
            -3.0,
            3.0,
            -3.0,
            3.0,
            2.0,
            fill_color=BLUE,
            fill_opacity=1.0,
            stroke_width=0.0,
        )
        label = _square(
            -1.5,
            1.5,
            -1.5,
            1.5,
            0.0,
            fill_color=WHITE,
            fill_opacity=1.0,
            stroke_width=0.0,
        )
        self.add_fixed_orientation_mobjects(label)
        self.add(g, label)
        self.wait(1 / FPS)
