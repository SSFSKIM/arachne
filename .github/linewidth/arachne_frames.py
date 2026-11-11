"""Scratch, never merged: renders a scene of tests/frame_ir/camera_scenes.py at 192x108, 15 fps,
on one draw path of the checkout it runs from, and keeps every delivered frame, the line width
multiple the scene's camera holds as each frame is delivered, the one each core draw's view
carries, and each call of the scene's ``rebinds``.

    python arachne_frames.py <out.npz> <SceneClass> <draw_path>
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
name = sys.argv[2]
draw_path = sys.argv[3]
random.seed(0)
np.random.seed(0)

from arachne.frame_ir import core_painter  # noqa: E402

from tests.frame_ir import camera_scenes  # noqa: E402
from tests.timeline.conftest import render  # noqa: E402

cls = getattr(camera_scenes, name)
calls: list[list] = []
views: list[float] = []
cameras: list = []

if hasattr(cls, "rebinds"):
    original_rebinds = cls.rebinds

    def rebinds(self, camera, points):
        result = original_rebinds(self, camera, points)
        if not cameras or cameras[-1] is not camera:
            cameras.append(camera)
        entry = [round(float(camera.get_theta()) * 180 / np.pi, 3), bool(result),
                 float(camera.cairo_line_width_multiple)]
        if not calls or calls[-1][:2] != entry[:2]:
            calls.append(entry + [1])
        else:
            calls[-1][3] += 1
        return result

    cls.rebinds = rebinds

draw = core_painter.CorePainter.draw


def drawn(self, delta, **kwargs):
    frame = draw(self, delta, **kwargs)
    views.append(float(self.state.view.get("cairo_line_width_multiple", float("nan"))))
    return frame


core_painter.CorePainter.draw = drawn

_, frames = render(cls, 15, draw_path=draw_path, pixel_width=192, pixel_height=108)
camera_multiple = [float(c.cairo_line_width_multiple) for c in cameras]
np.savez_compressed(out, frames=np.stack([np.asarray(f) for f in frames]))
out.with_suffix(".json").write_text(
    json.dumps(
        {"rebinds": calls, "core_views": views, "camera_end": camera_multiple,
         "frames": len(frames)},
        indent=1,
    )
)
print(name, draw_path, "frames", len(frames), "core draws", len(views))
print(" rebinds (theta deg, result, multiple before, calls):", calls[:60])
print(" core view multiples:", views)
print(" camera multiple at end:", camera_multiple)
