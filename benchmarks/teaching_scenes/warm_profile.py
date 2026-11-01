"""Profile the ten teaching scenes in one warm process.

Each scene renders once untimed (typesetting caches fill, imports settle), then
``--reps`` times under the clock. Two modes:

- ``draw``: a dry run at 480p15: every frame is drawn, none is encoded;
- ``exec``: the same run with the camera's capture stubbed out, so it times
  the scene's Python (construction, animations, updaters) without drawing.

``--count`` instead runs every scene once and counts copies: mobject deep
copies, copies of value arrays, and deep copies of plain arrays, through
wrappers that work on any revision. It prints JSON either way.

Usage::

    python benchmarks/teaching_scenes/warm_profile.py --mode draw --reps 3
    python benchmarks/teaching_scenes/warm_profile.py --count
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenes import SCENES  # noqa: E402

import arachne  # noqa: E402
from arachne import tempconfig  # noqa: E402
from arachne.camera.camera import Camera  # noqa: E402
from arachne.mobject import values  # noqa: E402
from arachne.mobject.mobject import Mobject  # noqa: E402


def settings(media_dir: str) -> dict:
    return {
        "dry_run": True,
        "disable_caching": True,
        "pixel_width": 854,
        "pixel_height": 480,
        "frame_rate": 15,
        "media_dir": media_dir,
        "verbosity": "WARNING",
        "progress_bar": "none",
    }


def render(scene_class) -> float:
    start = time.perf_counter()
    scene_class().render()
    return time.perf_counter() - start


def timed(mode: str, reps: int, media_dir: str) -> dict:
    if mode == "exec":
        Camera.capture_mobjects = lambda self, mobjects, **kwargs: None
    results = {}
    with tempconfig(settings(media_dir)):
        for scene_class in SCENES:
            render(scene_class)  # warm
            times = [render(scene_class) for _ in range(reps)]
            results[scene_class.__name__] = {
                "median_s": statistics.median(times),
                "times_s": times,
            }
    return results


class Counts:
    def __init__(self) -> None:
        self.mobject_deep_copies = 0
        self.value_array_copies = 0
        self.plain_array_deep_copies = 0


def counted(media_dir: str) -> dict:
    counts = Counts()
    mobject_deepcopy = Mobject.__deepcopy__
    frozen_copy = values._frozen_copy
    deepcopy = copy.deepcopy

    def counting_mobject_deepcopy(self, memo):
        counts.mobject_deep_copies += 1
        return mobject_deepcopy(self, memo)

    def counting_frozen_copy(value):
        counts.value_array_copies += 1
        return frozen_copy(value)

    def counting_deepcopy(x, memo=None, _nil=[]):  # noqa: B006
        if type(x) is np.ndarray:
            counts.plain_array_deep_copies += 1
        return deepcopy(x, memo, _nil)

    Mobject.__deepcopy__ = counting_mobject_deepcopy
    values._frozen_copy = counting_frozen_copy
    copy.deepcopy = counting_deepcopy
    results = {}
    with tempconfig(settings(media_dir)):
        for scene_class in SCENES:
            render(scene_class)  # warm, as the timed runs are
            counts.__init__()
            render(scene_class)
            results[scene_class.__name__] = dict(vars(counts))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("draw", "exec"), default="draw")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as media_dir:
        if args.count:
            out = {"kind": "count", "results": counted(media_dir)}
        else:
            out = {
                "kind": args.mode,
                "reps": args.reps,
                "results": timed(args.mode, args.reps, media_dir),
            }
    out["arachne"] = str(Path(arachne.__file__).resolve().parent)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
