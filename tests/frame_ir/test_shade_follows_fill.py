"""A shaded face whose fill changes while its points hold is drawn by the core in each frame's colour: no
lowering, paint or lighting kept from an earlier frame stands in for the colour the frame describes.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import BLUE, DEGREES, RED, Polygon, ThreeDScene
from arachne.frame_ir.core_painter import CorePainter

from ..timeline.conftest import render

pytest.importorskip("arachne_core")

FPS = 15


class FillTurning(ThreeDScene):
    """A flat shaded face, its points fixed, its fill turning from blue to red over four frames."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        corners = [(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)]
        face = Polygon(*[np.array([x, y, 0.0]) for x, y in corners])
        face.set_fill(BLUE, 1.0).set_stroke(width=0).set_shade_in_3d(True)
        self.add(face)
        self.wait(1 / FPS)
        self.play(face.animate.set_fill(RED), run_time=4 / FPS)


def test_a_face_whose_fill_turns_while_its_points_hold_is_drawn_in_each_frames_colour():
    _, cairo = render(FillTurning, FPS, draw_path="cairo")
    drawn = CorePainter.drawn_total
    _, core = render(FillTurning, FPS, draw_path="core")
    assert len(core) == len(cairo) >= 5
    assert CorePainter.drawn_total - drawn >= len(core), (
        "the frames were not the core's"
    )
    # The face's middle (the origin, the frame's centre): inside the face, far from its edges.
    y, x = core[0].shape[0] // 2, core[0].shape[1] // 2
    middle = [frame[y, x, :3].astype(int) for frame in core]
    assert np.abs(middle[-1] - middle[0]).max() > 50, "the fill did not turn"
    for i, (mine, ce) in enumerate(zip(core, cairo, strict=True)):
        d = np.abs(mine[y, x, :3].astype(int) - ce[y, x, :3].astype(int)).max()
        assert d <= 1, f"frame {i}: core {mine[y, x]} against Cairo {ce[y, x]}"
    # Every frame after the first moves the colour: none repeats the frame before's.
    assert all(
        np.abs(b - a).max() > 0 for a, b in zip(middle[1:-1], middle[2:], strict=False)
    )
