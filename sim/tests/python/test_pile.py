"""The 200-box pile (the measured workload's literals: a walled tray, and 8 layers of 5 x 5
spinning cubes dropped into it, 3,600 ticks) through the binding's closed world: it finds
nothing; on the baseline lane its declaration digest and content key are the ones the
measurement sitting recorded on the shipped build; and a coarse alarm times the crate's part of
the closed-world path.

The alarm is not the sitting's measurement and not a pin of it. It times the crate's part of a
closed world's stay, from the numbers to the sealed content: the world built item by item,
advanced to its last row in one call, and its channels, contact table, findings, declaration
digest and content key read out. The authoring step (arachne's ``World.body`` calls, about 2.2 s
at the sitting), which the sitting's interval holds, needs the engine, which this gate does not
install, and is outside it. The acceptance number is the sitting's measurement on the
production CPU class; this runner is shared.

Collected only when this crate's tests run on their own (sim/conftest.py), never inside the
engine's parallel run, where other tests share the machine.
"""

from __future__ import annotations

import hashlib
import platform
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
#: The alarm's bound, seconds: the sitting's (the whole path on the production CPU class,
#: confirmed at 10 s), held over a part of the path on a shared runner.
ALARM_S = 10.0
#: The baseline lane: x86-64 Linux, the shipped wheel's target. There the build identity must
#: be the pinned one, so that a drift fails rather than skips.
BASELINE = platform.system() == "Linux" and platform.machine() == "x86_64"


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
    if not BASELINE:
        pytest.skip(f"not the baseline lane ({build}): its bytes are its build's own")
    assert build == PINNED_BUILD, "the baseline lane's build identity moved"
    assert pile.declared == DECLARATION_DIGEST
    assert pile.key == CONTENT_DIGEST, "the pile's motion moved"


def test_alarm_the_crate_s_part_of_the_closed_world_path_stays_coarsely_fast(pile):
    """A coarse alarm on a shared runner, over the crate's part of the closed-world path only
    (the authoring step, about 2.2 s at the sitting, is outside it). A pass shows nothing about
    the production CPU class; the acceptance number is the sitting's measurement, not this test.
    A failure is a regression to measure there before it is called one.
    """
    print(
        f"pile: alarm: the crate's part of the closed-world path took {pile.seconds:.3f} s"
        f" on this shared runner (alarm at {ALARM_S} s; the authoring step is outside it;"
        " the acceptance number is the sitting's measurement)"
    )
    assert pile.seconds <= ALARM_S, (
        f"alarm: the crate's part of the pile's closed-world path took {pile.seconds:.3f} s on"
        f" this shared runner, over {ALARM_S} s. This coarse alarm excludes the authoring step"
        " (about 2.2 s at the sitting) and is not the acceptance measurement, which is the"
        " sitting's on the production CPU class: measure there before calling it a regression."
    )
