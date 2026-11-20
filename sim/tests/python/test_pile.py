"""The 200-box pile (W200's literals: a walled tray, and 8 layers of 5 x 5 spinning cubes
dropped into it, 3,600 ticks) through the binding's closed world: it finds nothing, its content
key is the one the measurement sitting recorded on the shipped build, and the closed-world path
stays under the sitting's bound.

The path timed is the crate's part of a closed world's stay, from the numbers to the sealed
content: the world built item by item, advanced to its last row in one call, and its channels,
contact table, findings, declaration digest and content key read out. The authoring API's own
declaration (arachne's ``World.body``), which the sitting's interval also holds, needs the engine,
which this gate does not install.

Collected only when this crate's tests run on their own (sim/conftest.py), never inside the
engine's parallel run, where other tests share the machine.
"""

from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace

import numpy as np
import pytest
from arachne_sim import World

LAST_TICK = 3600
GRAVITY = (0.0, 0.0, -9.8)
BOX_SIDE = 0.4
GRID = {"x": 5, "y": 5, "layers": 8}
SPACING = 0.6
FIRST_LAYER_Z = 1.0
ODD_LAYER_SHIFT = 0.3
SPINS = (
    (1.5, 0.0, 0.0),
    (0.0, 1.5, 0.0),
    (0.0, 0.0, 1.5),
    (-1.0, 1.0, 0.0),
    (0.0, -1.0, 1.0),
)
MATERIAL = {"density": 1.0, "friction": 0.5, "restitution": 0.1}
# (size x, y, z), (centre x, y, z): the floor, then the four walls.
TRAY = (
    ((8.0, 8.0, 0.2), (0.0, 0.0, -0.1)),
    ((0.2, 4.4, 2.5), (-2.1, 0.0, 1.25)),
    ((0.2, 4.4, 2.5), (2.1, 0.0, 1.25)),
    ((4.0, 0.2, 2.5), (0.0, -2.1, 1.25)),
    ((4.0, 0.2, 2.5), (0.0, 2.1, 1.25)),
)

#: The shipped wheel's build, on which the sitting ran: elsewhere the bytes differ by design.
PINNED_BUILD = (
    "rustc 1.99.1 (58cf8f9c3 2026-10-13); x86_64-unknown-linux-gnu; fxsr,sse,sse2"
)
#: What the sitting recorded on that build: the declaration's digest over the stay's window and
#: the content key (the digest of the trace bytes). The late-contact rule reads the motion; it
#: must not move it.
DECLARATION_DIGEST = "13b0b70a72953f06ff877e592e3ba8ca"
CONTENT_DIGEST = "67e552f526893930051eae6cecd7a046"
#: The bound, seconds: its authority is the sitting's result file (the pile on the production
#: CPU class, confirmed at 10 s), not this runner.
BOUND_S = 10.0


def f32(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def solid(w: World, size, centre, spin, fixed: bool) -> None:
    body = w.add_body(
        position=f32(*centre),
        rotation=f32(0.0, 0.0, 0.0, 1.0),
        velocity=f32(0.0, 0.0, 0.0),
        angular_velocity=f32(*spin),
        mass=np.float32(0.0),
        gravity_scale=np.float32(1.0),
        fixed=fixed,
    )
    w.add_shape(
        body,
        "cuboid",
        f32(*(extent / 2 for extent in size)),
        position=f32(0.0, 0.0, 0.0),
        rotation=f32(0.0, 0.0, 0.0, 1.0),
        density=np.float32(MATERIAL["density"]),
        friction=np.float32(MATERIAL["friction"]),
        restitution=np.float32(MATERIAL["restitution"]),
    )


def declare() -> World:
    w = World(3)
    w.add_gravity(f32(*GRAVITY))
    for size, centre in TRAY:
        solid(w, size, centre, (0.0, 0.0, 0.0), fixed=True)
    boxes = 0
    for layer in range(GRID["layers"]):
        shift = ODD_LAYER_SHIFT if layer % 2 else 0.0
        z = FIRST_LAYER_Z + layer * SPACING
        for j in range(GRID["y"]):
            for i in range(GRID["x"]):
                x = (i - (GRID["x"] - 1) / 2) * SPACING + shift
                y = (j - (GRID["y"] - 1) / 2) * SPACING + shift
                side = (BOX_SIDE,) * 3
                solid(w, side, (x, y, z), SPINS[boxes % len(SPINS)], fixed=False)
                boxes += 1
    return w


def declaration_digest(declaration: bytes, rows: int) -> str:
    """The stay's declaration digest, as a sealed occurrence holds it."""
    digest = hashlib.blake2b(digest_size=16)
    for part in (declaration, f"window:0:{rows}".encode()):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def pile() -> SimpleNamespace:
    started = time.perf_counter_ns()
    w = declare()
    w.advance_to(LAST_TICK)
    read = (w.pose(), w.velocity(), w.contacts(), w.findings(), w.finding_values())
    declared = declaration_digest(w.declaration_bytes(), w.rows)
    key = hashlib.blake2b(w.trace_bytes(), digest_size=16).hexdigest()
    seconds = (time.perf_counter_ns() - started) / 1e9
    return SimpleNamespace(
        world=w, read=read, declared=declared, key=key, seconds=seconds
    )


def test_the_pile_finds_nothing(pile):
    findings = pile.read[3]
    assert findings.shape == (0, 4), findings


def test_the_pile_s_content_key_is_the_sitting_s(pile):
    build = pile.world.identity()[5]
    print(f"pile: build {build!r}, declaration {pile.declared}, content key {pile.key}")
    if build != PINNED_BUILD:
        pytest.skip(f"not the pinned build ({build}): its bytes are its own")
    assert pile.declared == DECLARATION_DIGEST
    assert pile.key == CONTENT_DIGEST, "the pile's motion moved"


def test_the_closed_world_path_stays_under_the_sitting_s_bound(pile):
    """A tripwire on a shared runner: a pass shows nothing about the production CPU class, and
    a failure is a regression to measure there, against the sitting's result file, before it is
    called one.
    """
    print(f"pile: the closed-world path took {pile.seconds:.3f} s (bound {BOUND_S} s)")
    assert pile.seconds <= BOUND_S, (
        f"tripwire: the pile's closed-world path took {pile.seconds:.3f} s on this runner,"
        f" over the {BOUND_S} s the sitting's result file bounds it by"
    )
