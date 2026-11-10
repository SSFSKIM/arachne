"""The C5.3 measurement corpus: every engine-authored scene class of the scene
modules below, rendered on the core and on ir_cairo, each frame it delivers
recorded by the c53_probe plugin. A scene that fails fails the same way on
every revision measured; its error is recorded, not raised.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path

import pytest

from arachne import Scene

from ..timeline.conftest import render

MODULES = [
    "tests.frame_ir.scenes",
    "tests.frame_ir.core_scenes",
    "tests.frame_ir.camera_scenes",
    "tests.frame_ir.stress_scenes",
    "tests.frame_ir.probes.scenes",
    "tests.timeline.camera_scenes",
    "tests.timeline.stale_scenes",
    "tests.timeline.draw_path_scenes",
    "tests.lint.projection_scenes",
]


def _cases():
    found = []
    for name in MODULES:
        module = importlib.import_module(name)
        for attr, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Scene) and cls.__module__ == name:
                found.append(f"{name}:{attr}")
    return sorted(found)


@pytest.mark.parametrize("draw_path", ["core", "ir_cairo"])
@pytest.mark.parametrize("spec", _cases())
def test_scene(spec, draw_path):
    import c53_probe

    name, _, attr = spec.partition(":")
    module = importlib.import_module(name)
    fps = getattr(module, "FPS", 15)
    try:
        _, frames = render(
            getattr(module, attr),
            fps,
            draw_path=draw_path,
            pixel_width=192,
            pixel_height=108,
        )
    except Exception as error:  # the same failure at both revisions
        record = {"error": f"{type(error).__name__}: {str(error)[:200]}"}
        out = Path(os.environ["C53_OUT"]) / f"errors-{os.getpid()}.jsonl"
        with open(out, "a") as f:
            f.write(json.dumps({"case": f"{spec}[{draw_path}]", **record}) + "\n")
        return
    c53_probe.record_frames(frames)
