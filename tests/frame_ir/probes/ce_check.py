"""The probes' CE bytes against real CE 0.20.1.

Run with CE 0.20.1's interpreter, from anywhere:

    PROBES_ENGINE=manim python tests/frame_ir/probes/ce_check.py

Renders each probe with CE at the probes' frame size and prints, for each
check, CE's pixel and the bytes the probe records (``Check.ce``), as Arachne's
Cairo path draws them. Exits 1 if any differs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


def _scenes():
    spec = importlib.util.spec_from_file_location(
        "probe_scenes", Path(__file__).with_name("scenes.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_scenes"] = module
    spec.loader.exec_module(module)
    return module


def frames(scene_class, scenes) -> list[np.ndarray]:
    """Every frame CE draws of ``scene_class``, written nowhere."""
    from manim import tempconfig

    drawn: list[np.ndarray] = []
    with tempconfig(
        {
            "pixel_width": scenes.WIDTH,
            "pixel_height": scenes.HEIGHT,
            "frame_rate": scenes.FPS,
            "dry_run": True,
            "disable_caching": True,
            "verbosity": "ERROR",
            "progress_bar": "none",
        }
    ):
        scene = scene_class()

        def write_frame(frame, num_frames=1):
            drawn.extend([np.array(frame)] * num_frames)

        scene.renderer.file_writer.write_frame = write_frame
        scene.render()
    return drawn


def main() -> int:
    if os.environ.get("PROBES_ENGINE") != "manim":
        print("set PROBES_ENGINE=manim and run with CE's interpreter")
        return 2
    import manim

    scenes = _scenes()
    print(f"CE {manim.__version__}")
    differ = 0
    for probe in scenes.PROBES:
        drawn = frames(probe, scenes)
        for check in probe.CHECKS:
            pixel = tuple(int(v) for v in drawn[check.frame][check.y, check.x])
            same = pixel == check.ce
            differ += not same
            print(
                f"{'ok  ' if same else 'DIFF'} {probe.__name__} frame {check.frame}"
                f" ({check.x}, {check.y}): CE {pixel}, recorded {check.ce}: {check.what}"
            )
    print(f"{differ} differ")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
