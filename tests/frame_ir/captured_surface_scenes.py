"""Scenes whose frames are described from the camera's own captures (its
depth order is the scene's code, set on the camera, doing what the engine's
does), holding a still surface beside a turning one: the live description
walks the scene's roots to find a drawn face's surface.
"""

from __future__ import annotations

import functools

import numpy as np

from arachne import (
    BLUE,
    GREEN,
    OUT,
    PI,
    RIGHT,
    Rotate,
    Surface,
    ThreeDCamera,
    ThreeDScene,
)


def _wave(resolution=(8, 8)):
    return Surface(
        lambda u, v: np.array([u, v, 0.4 * np.sin(u) * np.cos(v)]),
        u_range=(-2, 2),
        v_range=(-2, 2),
        resolution=resolution,
        checkerboard_colors=[BLUE, GREEN],
    )


class StillSurfaceCaptured(ThreeDScene):
    """A still 8 x 8 surface (``still``) and a turning one, over two plays;
    ``cut``: the still surface loses eight faces between them.
    """

    cut = False

    def construct(self):
        self.camera.get_mobjects_to_display = functools.partial(
            ThreeDCamera.get_mobjects_to_display, self.camera
        )
        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)
        self.still = _wave()
        turning = _wave((4, 4)).scale(0.4).shift(RIGHT * 3)
        self.add(self.still, turning)
        self.play(Rotate(turning, PI / 3, axis=OUT), run_time=0.4)
        if self.cut:
            self.still.remove(*self.still.submobjects[:8])
        self.play(Rotate(turning, PI / 3, axis=OUT), run_time=0.4)


class StillSurfaceCapturedCut(StillSurfaceCaptured):
    cut = True


class StillSurfaceCutMidPlay(ThreeDScene):
    """:class:`StillSurfaceCaptured` whose still surface loses eight faces
    halfway through the first play, by the turning surface's updater.
    """

    def construct(self):
        self.camera.get_mobjects_to_display = functools.partial(
            ThreeDCamera.get_mobjects_to_display, self.camera
        )
        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)
        self.still = _wave()
        turning = _wave((4, 4)).scale(0.4).shift(RIGHT * 3)
        still = self.still
        elapsed = [0.0]

        def cut(mob, dt):
            elapsed[0] += dt
            if elapsed[0] >= 0.2 and len(still.submobjects) == 64:
                still.remove(*still.submobjects[:8])

        self.add(self.still, turning)
        turning.add_updater(cut)
        self.play(Rotate(turning, PI / 3, axis=OUT), run_time=0.4)
        turning.clear_updaters()
        self.play(Rotate(turning, PI / 3, axis=OUT), run_time=0.4)
