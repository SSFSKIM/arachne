"""Scratch, never merged: renders one scene of tests/frame_ir/camera_scenes.py on the core at
192x108, 15 fps, and keeps every frame it delivers and, per core draw, its frame, its delta's seq,
the core's label_plan and a summary of the records the stream holds.

    python core_frames.py <work> <SceneClass> <on|off>

``off`` turns label placement off before every core draw.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

work = Path(sys.argv[1])
name = sys.argv[2]
mode = sys.argv[3]
work.mkdir(parents=True, exist_ok=True)
random.seed(0)
np.random.seed(0)

from arachne.frame_ir import core_painter  # noqa: E402

from tests.frame_ir import camera_scenes  # noqa: E402
from tests.timeline.conftest import render  # noqa: E402

WATCH = {605, 670, 795, 860}
draws: list[dict] = []
painters: dict[int, int] = {}
original = core_painter.CorePainter.draw


def _summary(records) -> dict:
    kinds: dict[str, list[int]] = {}
    watched = {}
    for slot, rec in records.items():
        pre = rec.fields.get("pre_projected") is not None
        key = f"{rec.kind}{'/pre' if pre else ''}"
        counts = kinds.setdefault(key, [0, 0])
        counts[0] += 1
        if rec.fields.get("facing_group"):
            counts[1] += 1
        if slot in WATCH or rec.fields.get("facing_group"):
            watched[str(slot)] = {
                "kind": rec.kind,
                "pre_projected": pre,
                "facing_group": rec.fields.get("facing_group"),
                "fields": sorted(rec.fields),
                "source": repr(rec.fields.get("source"))[:400],
            }
    return {"kinds": kinds, "watched": watched}


def drawn(self, delta, **kwargs):
    if mode == "off":
        self.core.set_label_placement(False)
    frame = original(self, delta, **kwargs)
    index = painters.setdefault(id(self), len(painters))
    view = self.state.view
    w, h = view["pixel_width"], view["pixel_height"]
    facing = any(r.fields.get("facing_group") for r in self.state.records.values())
    plan = [list(p) for p in self.core.label_plan(w, h)] if facing else []
    draws.append(
        {
            "painter": index,
            "seq": int(self.last.seq),
            "facing": facing,
            "plan": plan,
            "records": _summary(self.state.records) if facing else None,
            "frame": np.array(frame),
        }
    )
    return frame


core_painter.CorePainter.draw = drawn

_, frames = render(
    getattr(camera_scenes, name), 15, draw_path="core", pixel_width=192, pixel_height=108
)
scene = [np.asarray(f) for f in frames]
for d in draws:
    d["scene_index"] = [i for i, f in enumerate(scene) if np.array_equal(f, d["frame"])]
np.savez_compressed(
    work / f"{name}-{mode}.npz",
    scene=np.stack(scene),
    draws=np.stack([d.pop("frame") for d in draws]),
)
(work / f"{name}-{mode}.json").write_text(json.dumps(draws, indent=1))
print(name, mode, "frames", len(scene), "draws", len(draws), "painters", len(painters))
