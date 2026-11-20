"""The simulation extension's trace bytes are the bake content's canonical bytes: a content the
record makes from a world's rows and identity is the one the crate encoded.
"""

from __future__ import annotations

import pytest

pytest.importorskip("arachne_sim")

import numpy as np  # noqa: E402
from arachne_sim import World  # noqa: E402

from arachne.timeline.bake import (  # noqa: E402
    BakeContent,
    BakeIdentity,
    Channel,
    EventTable,
)


def _content(world: World) -> BakeContent:
    *strings, features = world.identity()
    pose_unit, velocity_unit = world.units()
    return BakeContent(
        BakeIdentity(*strings, features=tuple(features)),
        world.rows,
        {
            "pose": Channel("f32", pose_unit, world.pose()),
            "velocity": Channel("f32", velocity_unit, world.velocity()),
        },
        {"contact": EventTable(world.contacts())},
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_a_worlds_trace_bytes_are_its_bake_contents(dimension):
    world = World(dimension)
    zero = [0.0] * (dimension - 2)
    world.add_gravity(np.array([0.0, -9.8, *zero], dtype=np.float32))
    floor = world.add_body(
        fixed=True, position=np.array([0.0, -0.5, *zero], dtype=np.float32)
    )
    world.add_shape(
        floor, "cuboid", np.array([5.0, 0.5, *[5.0] * (dimension - 2)], np.float32)
    )
    ball = world.add_body(position=np.array([0.0, 1.0, *zero], dtype=np.float32))
    world.add_shape(ball, "ball", np.float32(0.25), restitution=np.float32(0.6))
    world.advance_to(90)
    assert len(world.contacts()) >= 1
    content = _content(world)
    assert content.canonical_bytes() == world.trace_bytes()
