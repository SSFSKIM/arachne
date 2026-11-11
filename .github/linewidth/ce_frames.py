"""Scratch, never merged: runs the source of tests/frame_ir/camera_scenes.py unchanged but for
its import line (``from arachne import`` read as ``from manim import``) under Manim CE, renders
one of its scenes at 192x108, 15 fps, and keeps every delivered frame, the line width multiple
the camera holds as each frame is delivered, and each call of the scene's ``rebinds``.

    python ce_frames.py <camera_scenes.py> <out.npz> <SceneClass>
"""

from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

import numpy as np
from manim import tempconfig

source_path = Path(sys.argv[1])
out = Path(sys.argv[2])
name = sys.argv[3]
random.seed(0)
np.random.seed(0)

source = source_path.read_text()
assert source.count("from arachne import (") == 1
module = types.ModuleType("camera_scenes_ce")
exec(compile(source.replace("from arachne import (", "from manim import ("), "camera_scenes_ce", "exec"), module.__dict__)
cls = getattr(module, name)

calls: list[list] = []
if hasattr(cls, "rebinds"):
    original_rebinds = cls.rebinds

    def rebinds(self, camera, points):
        result = original_rebinds(self, camera, points)
        entry = [round(float(camera.get_theta()) * 180 / np.pi, 3), bool(result),
                 float(camera.cairo_line_width_multiple)]
        if not calls or calls[-1][:2] != entry[:2]:
            calls.append(entry + [1])
        else:
            calls[-1][3] += 1
        return result

    cls.rebinds = rebinds

frames: list[np.ndarray] = []
multiples: list[float] = []
with tempconfig(
    {
        "pixel_width": 192,
        "pixel_height": 108,
        "frame_rate": 15,
        "dry_run": True,
        "disable_caching": True,
    }
):
    scene = cls()

    def write_frame(frame_or_renderer, num_frames=1, **_):
        frame = (
            frame_or_renderer.get_frame()
            if hasattr(frame_or_renderer, "get_frame")
            else frame_or_renderer
        )
        frames.extend([np.array(frame)] * num_frames)
        multiples.extend([float(scene.renderer.camera.cairo_line_width_multiple)] * num_frames)

    scene.renderer.file_writer.write_frame = write_frame
    scene.render()
np.savez_compressed(out, frames=np.stack(frames))
out.with_suffix(".json").write_text(
    json.dumps({"rebinds": calls, "frame_multiples": multiples, "frames": len(frames)}, indent=1)
)
print(name, "ce frames", len(frames))
print(" rebinds (theta deg, result, multiple before, calls):", calls[:60])
print(" camera multiple per delivered frame:", multiples)
