"""The scale benchmark: per-frame draw time of one shape at many placements.

Usage::

    python benchmarks/scale/run.py [--scenes spheres,arrows,points]
        [--frames 100] [--threads 2] [--cpus 0,1] [--size 1920x1080]
        [--light ce|soft] [--json FILE] [--save DIR | --xf DIR]
    python benchmarks/scale/run.py --smoke

Each scene (``scenes.py``: 10,000 shaded spheres orbiting, 10,000 arrows,
100,000 points) is built as XF before anything is timed. The process is then
pinned to ``--cpus`` (Linux; elsewhere it says it is not pinned), each frame
is applied to one ``arachne_core.Stream`` and drawn with its tiles spread
over ``--threads`` threads, and "per frame" is the wall time of that draw
call alone (``time.perf_counter`` around ``Stream.draw``), one frame after
another. ``--light`` is the spheres' light model (the arrows and points are
not shaded). One warm-up frame, the first, is drawn and not counted.

It prints, per scene, one line::

    <scene> n=<count> records=<r> instances=<i> factored=<f>
        draw median=<ms> p10=<ms> p90=<ms> apply median=<ms> (<frames> frames)

``records``, ``instances`` and ``factored`` are what the first frame sets:
its records, how many of them are ``instances`` records, and how many leaves
had their shape factored. ``apply`` is the time to read one frame's delta
into the stream, outside the draw. ``--json`` writes the same, with every
frame's time, to FILE. ``--save DIR`` also writes each scene's frames as an XF
file, ``DIR/<scene>.xf``, with its counts beside it (``<scene>.json``);
``--xf DIR`` times those files instead of building the scenes.

``--smoke`` builds tiny versions of the scenes without drawing them, and
needs no compiled core: it checks the arrows batch into one ``instances``
record, every arrow factored, and the points into one ``points`` record.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scenes  # noqa: E402


def _pin(cpus: list[int]) -> str:
    if not hasattr(os, "sched_setaffinity"):
        return "not pinned (no sched_setaffinity here)"
    os.sched_setaffinity(0, set(cpus))
    return "pinned to cpus " + ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))


def _ms(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "median": statistics.median(ordered) * 1e3,
        "p10": ordered[len(ordered) // 10] * 1e3,
        "p90": ordered[(len(ordered) * 9) // 10] * 1e3,
    }


def measure(built: scenes.Built, size: tuple[int, int], threads: int) -> dict[str, Any]:
    import arachne_core as ac

    stream = ac.Stream()
    draws: list[float] = []
    applies: list[float] = []
    for index, data in enumerate(built.frames):
        delta = ac.FrameDelta.from_bytes(data)
        start = time.perf_counter()
        stream.apply(delta)
        applied = time.perf_counter()
        stream.draw(size[0], size[1], threads=threads, overlay=False)
        drawn = time.perf_counter()
        if index > 0:
            applies.append(applied - start)
            draws.append(drawn - applied)
    return {
        "scene": built.name,
        "n": built.count,
        "records": built.records,
        "instances": built.instances,
        "factored": built.factored,
        "frames": len(draws),
        "draw_ms": _ms(draws),
        "apply_ms": _ms(applies)["median"],
        "draw_s": draws,
    }


def _line(row: dict[str, Any]) -> str:
    d = row["draw_ms"]
    return (
        f"{row['scene']} n={row['n']} records={row['records']}"
        f" instances={row['instances']} factored={row['factored']}"
        f" draw median={d['median']:.1f}ms p10={d['p10']:.1f}ms p90={d['p90']:.1f}ms"
        f" apply median={row['apply_ms']:.1f}ms ({row['frames']} frames)"
    )


def _store(directory: Path, built: scenes.Built) -> None:
    import arachne_core as ac

    deltas = [ac.FrameDelta.from_bytes(data) for data in built.frames]
    ac.write_xf(directory / f"{built.name}.xf", deltas)
    counts = {
        k: getattr(built, k) for k in ("count", "records", "instances", "factored")
    }
    (directory / f"{built.name}.json").write_text(json.dumps(counts))


def _load(directory: Path, name: str) -> scenes.Built:
    import arachne_core as ac

    counts = json.loads((directory / f"{name}.json").read_text())
    built = scenes.Built(name, counts.pop("count"), **counts)
    built.frames = [d.to_bytes() for d in ac.read_xf(directory / f"{name}.xf")]
    return built


def smoke() -> int:
    size = (192, 108)
    arrows = scenes.arrows(16, 2, size=size, light="ce", encode=False)
    assert arrows.instances == 1, arrows
    assert arrows.records == 1, arrows
    assert arrows.factored == 2 * arrows.count, arrows
    cloud = scenes.points(500, 2, size=size, light="ce", encode=False)
    (record,) = (r for _, r in cloud.frames[0].set)
    assert record.kind == "points", record.kind
    orbit = scenes.spheres(50, 2, size=size, light="ce", encode=False)
    assert orbit.frames[0].shape == (50, 3, 4)
    assert (orbit.frames[0] != orbit.frames[1]).any()
    print(
        f"smoke: arrows {arrows.count} -> {arrows.instances} instances record,"
        f" {arrows.factored} leaves factored; points {cloud.count} -> 1 points"
        " record; spheres move between frames"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", default="spheres,arrows,points")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cpus", default="0,1")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--light", choices=sorted(scenes.LIGHTS), default="ce")
    parser.add_argument(
        "--count", type=int, help="every scene's count instead of its own"
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--xf", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        return smoke()
    size = tuple(int(v) for v in args.size.split("x"))
    assert len(size) == 2
    names = args.scenes.split(",")
    built = []
    for name in names if args.xf is None else ():
        count = args.count or scenes.COUNTS[name]
        start = time.perf_counter()
        # The first frame warms the stream and is not counted.
        built.append(
            scenes.SCENES[name](count, args.frames + 1, size=size, light=args.light)
        )
        print(f"built {name} in {time.perf_counter() - start:.1f}s", file=sys.stderr)
    if args.xf is not None:
        built = [_load(args.xf, name) for name in names]
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for scene in built:
            _store(args.save, scene)
    pinned = _pin([int(c) for c in args.cpus.split(",")])
    import arachne_core

    print(
        f"scale benchmark: {size[0]}x{size[1]}, {args.threads} threads, {pinned},"
        f" light {args.light}, {platform.machine()} {platform.processor() or ''}".rstrip()
        + f", arachne_core XF {'.'.join(map(str, arachne_core.XF_VERSION))}"
    )
    rows = []
    for scene in built:
        row = measure(scene, size, args.threads)
        print(_line(row))
        rows.append(row)
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "size": size,
                    "threads": args.threads,
                    "pinned": pinned,
                    "light": args.light,
                    "scenes": rows,
                },
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
