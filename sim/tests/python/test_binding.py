"""The binding: a world takes its numbers in the stored type only, hands its rows out read-only,
gives the same bytes however it is advanced, and lets other threads run while it advances.
"""

from __future__ import annotations

import importlib.metadata
import sys
import threading
import tomllib
from pathlib import Path

import arachne_sim
import numpy as np
import pytest
from arachne_sim import World

CRATE = Path(__file__).resolve().parents[2]


def vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def dropped(dimension: int = 2) -> World:
    """Two balls dropped on a floor, one spinning."""
    w = World(dimension)
    zero = [0.0] * dimension
    w.add_gravity(vec(*zero[:1], -9.8, *zero[2:]))
    floor = w.add_body(fixed=True, position=vec(0.0, -0.5, *zero[2:]))
    w.add_shape(floor, "cuboid", vec(5.0, 0.5, *([5.0] if dimension == 3 else [])))
    for x in (-1.0, 1.0):
        b = w.add_body(
            position=vec(x, 2.0, *zero[2:]),
            angular_velocity=vec(1.0) if dimension == 2 else vec(0.0, 0.0, 1.0),
        )
        w.add_shape(b, "ball", np.float32(0.3), restitution=np.float32(0.5))
    return w


def test_a_python_float_is_refused_for_the_stored_type():
    w = World(2)
    with pytest.raises(TypeError, match=r"float32 .*not list"):
        w.add_body(position=[0.0, 0.0], mass=np.float32(1))
    with pytest.raises(TypeError, match="not float"):
        w.add_body(mass=1.0)
    with pytest.raises(TypeError, match="float32"):
        w.add_body(position=np.array([0.0, 0.0]), mass=np.float32(1))
    assert w.bodies == 0
    assert w.add_body(position=vec(0.0, 0.0), mass=np.float32(1)) == 0
    assert w.mass(0) == 1.0


def test_what_a_world_refuses_is_a_value_error_that_says_why():
    w = World(2)
    with pytest.raises(ValueError, match="finite"):
        w.add_body(position=vec(np.nan, 0.0), mass=np.float32(1))
    with pytest.raises(ValueError, match="takes 2 numbers, not 3"):
        w.add_body(position=vec(0.0, 0.0, 0.0), mass=np.float32(1))
    w.add_body(mass=np.float32(1))
    with pytest.raises(ValueError, match='not "disc"'):
        w.add_shape(0, "disc", np.float32(1))
    w.add_body(mass=np.float32(1))
    with pytest.raises(ValueError, match="3D only"):
        w.add_joint("ball", 0, 1)
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        World(4)
    with pytest.raises(ValueError, match="point_masses"):
        World(2, integrator="verlet")


def test_rows_are_read_only_arrays_of_their_shapes():
    for dimension, (pose, velocity) in ((2, (3, 3)), (3, (7, 6))):
        w = dropped(dimension)
        w.advance_to(60)
        assert w.rows == 61
        assert w.pose().shape == (61, 3, pose)
        assert w.velocity(10, 20).shape == (10, 3, velocity)
        assert w.pose().dtype == np.float32
        contacts = w.contacts()
        assert contacts.dtype == np.uint32
        assert contacts.shape[1] == 4
        assert {int(b) for b in contacts[:, 2][contacts[:, 3] == 1]} == {1, 2}
        for rows in (w.pose(), w.velocity(), contacts):
            assert not rows.flags.writeable
            with pytest.raises(ValueError):
                rows.flat[0] = 0
        with pytest.raises(ValueError, match="within the 61 computed"):
            w.pose(0, 62)


def test_the_bytes_do_not_depend_on_how_the_world_was_advanced():
    whole, pieces = dropped(3), dropped(3)
    whole.advance_to(120)
    for row in (1, 50, 49, 120):
        pieces.advance_to(row)
    assert pieces.rows == 121
    assert whole.declaration_bytes() == pieces.declaration_bytes()
    assert whole.trace_bytes() == pieces.trace_bytes()
    assert np.array_equal(whole.pose(), pieces.pose())


def test_an_open_world_steps_under_the_forces_it_is_given():
    w = World(2, open=True)
    w.add_body(mass=np.float32(2))
    forces = np.array([[4.0, 0.0]], dtype=np.float32)
    torques = np.zeros((1, 1), dtype=np.float32)
    assert [w.step(forces, torques) for _ in range(3)] == [1, 2, 3]
    assert w.velocity()[3, 0, 0] == pytest.approx(3 * 2.0 / 60, rel=1e-5)
    with pytest.raises(ValueError, match=r"step\(\)"):
        w.advance_to(10)
    with pytest.raises(TypeError, match="float32"):
        w.step(forces.astype(np.float64), torques)


def test_identity_units_and_versions():
    w = dropped(2)
    producer, version, solver, solver_version, integrator, build, features = (
        w.identity()
    )
    assert (producer, solver) == ("arachne-sim", "rapier2d")
    assert "b303bc13" in solver_version
    assert integrator.startswith("rapier/")
    assert build.startswith("rustc ")
    assert features == sorted(features)
    assert "enhanced-determinism" in features
    assert w.units() == ("u u rad", "u/s u/s rad/s")
    cargo = tomllib.loads((CRATE / "Cargo.toml").read_text())["package"]["version"]
    project = tomllib.loads((CRATE / "pyproject.toml").read_text())["project"]
    assert version == cargo == project["version"]
    assert importlib.metadata.version("arachne-sim") == cargo
    assert (arachne_sim.DECLARATION_LAYOUT, arachne_sim.CONTENT_LAYOUT) == (2, 1)


def test_a_world_s_limits_and_what_it_found():
    """A world's speed limits are below the solver's caps and default to the crate's; what it
    finds is read as rows of ``row, kind, a, b``.
    """
    assert arachne_sim.SPEED_LIMIT < arachne_sim.SPEED_CAP == 400.0
    assert arachne_sim.ANGULAR_SPEED_LIMIT < arachne_sim.ANGULAR_SPEED_CAP
    w = World(2)
    assert (w.speed_limit, w.angular_speed_limit) == (
        arachne_sim.SPEED_LIMIT,
        arachne_sim.ANGULAR_SPEED_LIMIT,
    )
    assert w.continuous_collision
    with pytest.raises(TypeError, match="float32"):
        World(2, speed_limit=5.0)
    with pytest.raises(ValueError, match="below the solver's own cap"):
        World(2, speed_limit=np.float32(400))
    w = World(
        2,
        integrator="point_masses",
        speed_limit=np.float32(1),
        continuous_collision=False,
    )
    assert (w.speed_limit, w.continuous_collision) == (1.0, False)
    w.add_body(velocity=vec(3.0, 0.0), mass=np.float32(1))
    w.advance_to(5)
    found = w.findings()
    assert found.dtype == np.uint32
    assert not found.flags.writeable
    assert found.tolist() == [[0, arachne_sim.SPEED, 0, 0]]
    values = w.finding_values()
    assert values.dtype == np.float32
    assert not values.flags.writeable
    assert values.tolist() == [3.0]
    assert World(2).findings().shape == (0, 4)
    assert World(2).finding_values().shape == (0,)


def test_other_threads_run_while_a_world_advances():
    """With the interpreter switching threads only when one blocks or releases its lock, the
    main thread runs between the worker entering ``advance_to`` and the worker's next line only if
    ``advance_to`` released the lock. That holds as long as the worker is still advancing when
    the main thread wakes, which the advance's length (a tenth of a second or so) provides.
    """
    w = World(3, integrator="point_masses")
    for i in range(100):
        w.add_body(position=vec(float(i % 10), float(i // 10), 0.0), mass=np.float32(1))
    w.add_attraction(list(range(100)), np.float32(0.001))
    entering = threading.Event()
    returned = threading.Event()
    failed: list[BaseException] = []

    def advance() -> None:
        entering.set()
        try:
            w.advance_to(600)
        except BaseException as error:
            failed.append(error)
        returned.set()

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1000.0)
    try:
        worker = threading.Thread(target=advance)
        worker.start()
        # Blocks, releasing the lock; it takes the lock back only once the worker releases it,
        # which the worker does first inside advance_to if at all.
        assert entering.wait(timeout=60), "the worker never started"
        ran_during = not returned.is_set()
        assert returned.wait(timeout=120), (
            "advance_to did not return within two minutes"
        )
        worker.join(timeout=60)
    finally:
        sys.setswitchinterval(interval)
    if failed:
        raise failed[0]
    assert ran_during, "the main thread ran only after advance_to returned"
    assert w.rows == 601
