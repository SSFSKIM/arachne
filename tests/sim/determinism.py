"""The scenes of :mod:`.determinism_scenes` baked, and each world's
digests set against the committed ones.

For each scene and world, three digests, compared apart: the declaration
digest (what the world was given, its open world's pushes at every tick
included, as its occurrence holds it); the motion digest (the motion bytes:
the pose and velocity channels and the contact events, without the build's
identity); and the trace digest (the content key: the digest of the trace
bytes, the canonical bytes of the bake content, which carry the identity of
the build that computed them). The rule they are read by:

- a world whose declaration digest is the committed one and whose motion
  digest is not is the solver breaking its promise: equal declaration bytes
  computed different motion, on any machine;
- where the build's identity is the committed one (:data:`BUILD`), the
  trace digest is compared too: equal motion with another trace digest
  there is the identity alone (another producer, solver or feature); on
  another build the trace bytes carry that build's identity, and only the
  motion is compared;
- a world of :class:`.ComputedWorlds` whose declaration digest is not the
  committed one is an input mismatch: the mathematical functions gave other
  values on this machine, and the motion and trace, computed from other
  inputs, are not compared;
- a world of :class:`.LiteralWorlds` whose declaration digest is not the
  committed one declared other bytes from the same literals.

There is one committed value per world for each digest, for every machine;
a machine whose bytes differ is a finding, never a second value.

Run as a script, it bakes one scene once and writes what it found as JSON,
for a driver that compares machines::

    python tests/sim/determinism.py LiteralWorlds --out literal.json

The file holds the scene, each world's three digests, the committed ones,
the engine's identity string and the build, and, to find the first
difference between two machines, each world's declared fields, a digest of
each of its rows, its contact events and the open world's stored pushes.
With ``--ready`` and ``--go`` it starts its bake behind a barrier, and it
records the host's monotonic clock as the bake starts and ends, so that
runs meant to overlap can be shown to.
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
#: How long the script waits for its --go file, in seconds.
GO_TIMEOUT_S = 600

#: The build the trace digests were committed on (x86-64 Linux): where a
#: bake's build is another, its trace bytes carry that build's identity and
#: its trace digest is not compared.
BUILD = "rustc 1.99.1 (58cf8f9c3 2026-10-13); x86_64-unknown-linux-gnu; fxsr,sse,sse2"

#: The committed declaration and trace digests, by (scene, world), taken on
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

#: The committed motion digests, by (scene, world): the same on every machine.
MOTION: dict[tuple[str, str], str] = {
    ("LiteralWorlds", "plane"): "0" * 32,
    ("LiteralWorlds", "space"): "0" * 32,
    ("ComputedWorlds", "plane"): "0" * 32,
    ("ComputedWorlds", "space"): "0" * 32,
    ("ComputedWorlds", "open"): "0" * 32,
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


def motion_digest(pose: np.ndarray, velocity: np.ndarray, contacts: np.ndarray) -> str:
    """The motion bytes' digest: for the pose and velocity channels and the
    contact events in turn, its name, its shape as unsigned 32-bit integers
    and its values, little-endian in row-major order. The build's identity is
    not in it.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, values, dtype in (
        ("pose", pose, "<f4"),
        ("velocity", velocity, "<f4"),
        ("contact", contacts, "<u4"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype="<u4").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _declaration_digest(declared: bytes, rows: int) -> bytes:
    """The digest an occurrence holds of a stay from row 0 through ``rows``."""
    digest = hashlib.blake2b(digest_size=16)
    for part in (declared, f"window:0:{rows}".encode()):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def _applied(world: World, declared: bytes, steps: int) -> list[list[Any]]:
    """An open world's stored pushes, read from the end of its declaration
    bytes: for each step in order, ``[row, bits]``, where ``bits`` holds the
    force then the torque of each body in declaration order, each number's
    single-precision bits as eight hex digits (a negative zero is
    ``80000000``).
    """
    bodies = len(world._declaration.bodies)
    numbers = bodies * (world.dimension + (1 if world.dimension == 2 else 3))
    size = 1 + 4 + 4 * numbers
    tail = declared[len(declared) - steps * size :]
    if len(tail) != steps * size:
        raise AssertionError("The declaration bytes hold fewer pushes than steps.")
    found = []
    for step in range(steps):
        record = tail[step * size : (step + 1) * size]
        row = int.from_bytes(record[1:5], "little")
        words = np.frombuffer(record[5:], dtype="<u4")
        found.append([row, [f"{int(word):08x}" for word in words]])
    kinds = {tail[step * size] for step in range(steps)}
    rows = [row for row, _ in found]
    if len(kinds) != 1 or rows != list(range(rows[0], rows[0] + steps)):
        raise AssertionError("The declaration bytes do not end in one push per step.")
    return found


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
    declared = session.module.DECLARED[_name(world)]
    if _declaration_digest(declared, content.ticks) != occurrence.declaration:
        raise AssertionError(
            f"{scene}'s {_name(world)} world: the declaration bytes the scene kept"
            " are not the ones its occurrence's digest covers"
        )
    identity = content.identity
    pose = content.channels["pose"].values
    velocity = content.channels["velocity"].values
    contacts = content.events["contact"].rows
    name = _name(world)
    committed = COMMITTED.get((scene, name))
    motion = MOTION.get((scene, name))
    found = {
        "world": name,
        "declaration": occurrence.declaration.hex(),
        "motion": motion_digest(pose, velocity, contacts),
        "trace": occurrence.content.hex(),
        "build": identity.build,
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
        else {
            "declaration": committed[0],
            "motion": motion,
            "trace": committed[1],
            "build": BUILD,
        },
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
            "inputs": _applied(world, declared, content.ticks - 1)
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
    one line each; none when every world gives the committed ones.
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
        motion = MOTION[(scene, name)]
        where = f"{scene} {name}"
        if world["declaration"] != declaration:
            if scene == COMPUTED:
                found.append(
                    f"{where}: INPUT MISMATCH: the declaration bytes are not the"
                    f" committed ones (declaration {world['declaration']}, committed"
                    f" {declaration}): the computed inputs differ, and the motion"
                    " and trace are not compared"
                )
            else:
                found.append(
                    f"{where}: DECLARATION: the declaration bytes are not the"
                    " committed ones although every input is a literal"
                    f" (declaration {world['declaration']}, committed {declaration})"
                )
        elif world["motion"] != motion:
            found.append(
                f"{where}: MOTION: the declaration bytes are the committed ones and"
                f" the motion bytes are not (motion {world['motion']}, committed"
                f" {motion}): equal declarations computed different motion"
            )
        elif world["build"] == BUILD and world["trace"] != trace:
            found.append(
                f"{where}: IDENTITY: the declaration and motion bytes are the"
                " committed ones, on the committed build, and the trace bytes are"
                f" not (trace {world['trace']}, committed {trace}): the identity"
                f" alone differs ({world['identity']})"
            )
    return found


def report(case: dict[str, Any], found: list[str]) -> str:
    """The problems, then what each world gave, and its digests as they
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
            f"    build {world['build']}"
            f"{'' if world['build'] == BUILD else ' (not the committed build: trace not compared)'}",
            f'    ("{case["scene"]}", "{world["world"]}"): ("{world["declaration"]}",'
            f' "{world["trace"]}"), motion "{world["motion"]}"',
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bake one determinism scene once and write its digests as JSON."
    )
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ready",
        type=Path,
        help="a file to make once the engine is imported, before waiting for --go",
    )
    parser.add_argument(
        "--go", type=Path, help="a file whose existence starts the bake (a barrier)"
    )
    args = parser.parse_args(argv)
    if args.ready is not None:
        args.ready.touch()
    if args.go is not None:
        waited = time.monotonic()
        while not args.go.exists():
            if time.monotonic() - waited > GO_TIMEOUT_S:
                print(f"no {args.go} within {GO_TIMEOUT_S} s", file=sys.stderr)
                return 2
            time.sleep(0.005)
    started = time.monotonic_ns()
    case = bake(args.scene)
    case["window_ns"] = [started, time.monotonic_ns()]
    case["clock"] = "monotonic"
    case["problems"] = problems(case)
    written = args.out.with_name(f".{args.out.name}.{os.getpid()}")
    written.write_text(json.dumps(case, sort_keys=True) + "\n")
    os.replace(written, args.out)
    for world in case["worlds"]:
        print(
            f"{case['scene']} {world['world']} declaration={world['declaration']}"
            f" motion={world['motion']} trace={world['trace']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
