"""Scratch, never merged: the scenes of tests/frame_ir/camera_scenes.py named below, written
against Manim CE, rendered by Manim CE at 192x108, 15 fps; every frame it delivers is kept.

    python ce_frames.py <work> <SceneClass>
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from manim import (
    BLUE,
    DEGREES,
    GREEN,
    OUT,
    RED,
    UL,
    YELLOW,
    Dot,
    Rotate,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDCamera,
    ThreeDScene,
    tempconfig,
)

work = Path(sys.argv[1])
name = sys.argv[2]
NOLABEL = len(sys.argv) > 3 and sys.argv[3] == "nolabel"
random.seed(0)
np.random.seed(0)


class ShiftedCamera(ThreeDCamera):
    def project_points(self, points):
        return super().project_points(points) + np.array([-1.0, 0.3, 0.0])


class Solids(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.patch(self.renderer.camera)
        surface = Surface(
            lambda u, v: np.array([u, v, 0.4 * np.sin(u) * np.cos(v)]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(6, 6),
            fill_opacity=0.8,
        )
        axes = ThreeDAxes(
            x_range=[-2, 2], y_range=[-2, 2], z_range=[-1, 1], x_length=4, y_length=4
        )
        corner = Text("A", font_size=28, color=YELLOW).to_corner(UL)
        tag = Text("p", font_size=28, color=GREEN).move_to([1.5, 1.5, 0.6])
        self.add_fixed_in_frame_mobjects(corner)
        if NOLABEL:
            self.add(axes, surface, Dot([-1.0, -1.0, 0.0], color=RED))
        else:
            self.add_fixed_orientation_mobjects(tag)
            self.add(axes, surface, tag, Dot([-1.0, -1.0, 0.0], color=RED))
        self.play(Rotate(surface, 30 * DEGREES, axis=OUT), run_time=0.4)
        self.move_camera(theta=-20 * DEGREES, run_time=0.4)
        self.wait(0.2)

    def patch(self, camera) -> None:
        pass


class ShiftedSolids(Solids):
    def __init__(self, **kwargs):
        super().__init__(camera_class=ShiftedCamera, **kwargs)


class PatchedSolids(Solids):
    def patch(self, camera) -> None:
        project = camera.project_points

        def shifted(points):
            out = project(points)
            out[:, 0] += 1.0
            out[:, 1] -= 0.4
            return out

        camera.project_points = shifted


class ProjectingStateSolids(Solids):
    def patch(self, camera) -> None:
        project = camera.project_points

        def projected(points):
            out = np.array(project(points))
            if len(out) and self.rebinds(camera, points):
                camera.cairo_line_width_multiple = 0.03 * (
                    1 + abs(float(np.mean(out[:, 2])))
                )
            return out

        camera.project_points = projected

    def rebinds(self, camera, points) -> bool:
        return True


class SteadyProjectingSolids(ProjectingStateSolids):
    def rebinds(self, camera, points) -> bool:
        return False


class MidPlayStateSolids(ProjectingStateSolids):
    def rebinds(self, camera, points) -> bool:
        return -40 * DEGREES < camera.get_theta() < -25 * DEGREES


frames: list[np.ndarray] = []
with tempconfig(
    {
        "pixel_width": 192,
        "pixel_height": 108,
        "frame_rate": 15,
        "dry_run": True,
        "disable_caching": True,
    }
):
    scene = globals()[name]()

    def write_frame(frame_or_renderer, num_frames=1, **_):
        frame = (
            frame_or_renderer.get_frame()
            if hasattr(frame_or_renderer, "get_frame")
            else frame_or_renderer
        )
        frames.extend([np.array(frame)] * num_frames)

    scene.renderer.file_writer.write_frame = write_frame
    scene.render()
kind = "cenl" if NOLABEL else "ce"
np.savez_compressed(work / f"{name}-{kind}.npz", scene=np.stack(frames))
print(name, "ce frames", len(frames))
