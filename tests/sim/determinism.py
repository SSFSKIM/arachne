"""The scenes of :mod:`.determinism_scenes` baked, and each world's two
digests set against the committed ones.

For each scene and world, two digests, compared apart: the declaration
digest (what the world was given, its open world's pushes at every tick
included, as its occurrence holds it) and the trace digest (the content key:
the digest of the trace bytes, the canonical bytes of the bake content). The
rule they are read by:

- a world whose declaration digest is the committed one and whose trace
  digest is not is the solver breaking its promise: equal declaration bytes
  computed different trace bytes;
- a world of :class:`.ComputedWorlds` whose declaration digest is not the
  committed one is an input mismatch: the mathematical functions gave other
  values on this machine, and the trace, computed from other inputs, is not
  compared;
- a world of :class:`.LiteralWorlds` whose declaration digest is not the
  committed one declared other bytes from the same literals.

There is one committed pair per world, for every machine of the
architecture the constants were taken on; a machine whose bytes differ is a
finding, never a second pair.

Run as a script, it bakes one scene once and writes what it found as JSON,
for a driver that compares machines::

    python tests/sim/determinism.py LiteralWorlds --out literal.json

The file holds the scene, each world's declaration and trace digests, the
committed pair, the engine's identity string, and, to find the first
difference between two machines, each world's declared fields, a digest of
each of its rows, its contact events and the open world's pushes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import arachne
from arachne.session import Session
from arachne.simulation import World

SCENES_FILE = Path(__file__).with_name("determinism_scenes.py")
SCENES = ("LiteralWorlds", "ComputedWorlds")
#: The scene whose inputs are computed: a declaration that differs there is
#: an input mismatch.
COMPUTED = "ComputedWorlds"
#: The record keeps few, small frames: the bake is the same at every rate.
CONFIG = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}
#: Every stay spans the whole scene: ticks 0 to 20 s at 60 a second.
LAST_TICK = 1200

#: The committed digests, (declaration, trace) by (scene, world), taken on
#: x86-64 Linux.
COMMITTED: dict[tuple[str, str], tuple[str, str]] = {
    ("LiteralWorlds", "plane"): (
        "c8f26a54855a19c024b3ff8e4cb58a8f",
        "fc2774983df8dbdcfdd658f426933e62",
    ),
    ("LiteralWorlds", "space"): (
        "20960bad076d8f98243341485c3bd4fb",
        "82d7cec0edbaf5de24d300af8e429fcd",
    ),
    ("ComputedWorlds", "plane"): (
        "61f7cc579a5b19330cc546212dad9d97",
        "28265c3ec684e8a773a7e1f70dbe9c11",
    ),
    ("ComputedWorlds", "space"): (
        "7279e1addb592e06e80951c367d98896",
        "b0598ba2aba483e92de55524a8e73414",
    ),
    ("ComputedWorlds", "open"): (
        "8c42de2c4654f4dd40a9d0cbee5f11ef",
        "d0e668b2645d2c811dec37de994dbd58",
    ),
}


def _name(world: World) -> str:
    if world.is_open:
        return "open"
    return "plane" if world.dimension == 2 else "space"


def _digest(*parts: bytes) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _fields(value: Any, path: str = "") -> list[list[Any]]:
    """A declaration's fields, ``[path, value]`` in a fixed order: every
    number the solver is given, as stored. A body's drawing at rest is not
    given to the solver and is left out.
    """
    if dataclasses.is_dataclass(value):
        found = []
        for field in dataclasses.fields(value):
            if field.name != "rest":
                name = f"{path}.{field.name}" if path else field.name
                found.extend(_fields(getattr(value, field.name), name))
        return found
    if isinstance(value, tuple):
        found = [[f"{path}.length", len(value)]]
        for index, item in enumerate(value):
            found.extend(_fields(item, f"{path}[{index}]"))
        return found
    return [[path, value]]


def _world(world: World, session: Session, scene: str) -> dict[str, Any]:
    bakes = session.record.bakes
    occurrence = bakes.occurrence(world._bake.ordinal)
    if (occurrence.first_tick, occurrence.last_tick) != (0, LAST_TICK):
        raise AssertionError(
            f"{scene}'s {_name(world)} world stayed from tick"
            f" {occurrence.first_tick} to {occurrence.last_tick}, not 0 to {LAST_TICK}"
        )
    content = bakes.contents[occurrence.content]
    identity = content.identity
    pose = content.channels["pose"].values
    velocity = content.channels["velocity"].values
    contacts = content.events["contact"].rows
    name = _name(world)
    committed = COMMITTED.get((scene, name))
    found = {
        "world": name,
        "declaration": occurrence.declaration.hex(),
        "trace": occurrence.content.hex(),
        "identity": "; ".join(
            (
                f"{identity.producer} {identity.producer_version}",
                f"{identity.solver} {identity.solver_version}",
                identity.integrator,
                identity.build,
                ",".join(sorted(identity.features)),
            )
        ),
        "committed": None
        if committed is None
        else {"declaration": committed[0], "trace": committed[1]},
        "parts": {
            "identity": _digest(repr(dataclasses.astuple(identity)).encode()),
            "pose": _digest(pose.tobytes()),
            "velocity": _digest(velocity.tobytes()),
            "contact": _digest(contacts.tobytes()),
        },
        "findings": len(occurrence.findings),
        "detail": {
            "fields": _fields(world._declaration),
            "rows": [
                hashlib.blake2b(
                    pose[row].tobytes() + velocity[row].tobytes(), digest_size=4
                ).hexdigest()
                for row in range(content.ticks)
            ],
            "contacts": contacts.tolist(),
            "inputs": [list(row) for row in session.module.PUSHED]
            if world.is_open
            else None,
        },
    }
    return found


def bake(scene: str) -> dict[str, Any]:
    """Bake ``scene`` once, in a session of its own: each world's digests,
    identity and detail.
    """
    if scene not in SCENES:
        raise ValueError(f"No scene {scene!r}: one of {', '.join(SCENES)}.")
    with (
        tempfile.TemporaryDirectory() as media,
        arachne.tempconfig({"media_dir": media}),
    ):
        started = time.perf_counter()
        session = Session.open(SCENES_FILE, scene, config=CONFIG, probe=False)
        try:
            seconds = time.perf_counter() - started
            worlds = [m for m in session.scene.mobjects if isinstance(m, World)]
            found = sorted(
                (_world(world, session, scene) for world in worlds),
                key=lambda w: ("plane", "space", "open").index(w["world"]),
            )
        finally:
            session.close()
    return {
        "scene": scene,
        "engine": arachne.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
        "seconds": round(seconds, 3),
        "worlds": found,
    }


def problems(case: dict[str, Any]) -> list[str]:
    """What in a baked scene breaks the rule against the committed digests,
    one line each; none when every world gives the committed pair.
    """
    scene = case["scene"]
    expected = sorted(world for s, world in COMMITTED if s == scene)
    worlds = {w["world"]: w for w in case["worlds"]}
    found = []
    if sorted(worlds) != expected:
        found.append(f"{scene}: worlds {sorted(worlds)}, not {expected}")
    for name in expected:
        world = worlds.get(name)
        if world is None:
            continue
        declaration, trace = COMMITTED[(scene, name)]
        if world["declaration"] == declaration:
            if world["trace"] != trace:
                found.append(
                    f"{scene} {name}: SOLVER: the declaration bytes are the committed"
                    f" ones and the trace bytes are not (trace {world['trace']},"
                    f" committed {trace}): equal declarations computed different"
                    " traces"
                )
        elif scene == COMPUTED:
            found.append(
                f"{scene} {name}: INPUT MISMATCH: the declaration bytes are not the"
                f" committed ones (declaration {world['declaration']}, committed"
                f" {declaration}): the computed inputs differ, and the trace is not"
                " compared"
            )
        else:
            found.append(
                f"{scene} {name}: DECLARATION: the declaration bytes are not the"
                f" committed ones although every input is a literal (declaration"
                f" {world['declaration']}, committed {declaration})"
            )
    return found


def report(case: dict[str, Any], found: list[str]) -> str:
    """The problems, then what each world gave, and its committed pair as it
    would be written in :data:`COMMITTED`.
    """
    lines = [
        *found,
        f"{case['scene']}, {case['machine']}, baked in {case['seconds']} s:",
    ]
    for world in case["worlds"]:
        parts = ", ".join(f"{k} {v}" for k, v in world["parts"].items())
        lines += [
            f"  {world['world']}: identity {world['identity']}",
            f"    parts: {parts}; findings {world['findings']}",
            f'    ("{case["scene"]}", "{world["world"]}"): ("{world["declaration"]}",'
            f' "{world["trace"]}"),',
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bake one determinism scene once and write its digests as JSON."
    )
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    case = bake(args.scene)
    case["problems"] = problems(case)
    written = args.out.with_name(f".{args.out.name}.{os.getpid()}")
    written.write_text(json.dumps(case, sort_keys=True) + "\n")
    os.replace(written, args.out)
    for world in case["worlds"]:
        print(
            f"{case['scene']} {world['world']} declaration={world['declaration']}"
            f" trace={world['trace']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
