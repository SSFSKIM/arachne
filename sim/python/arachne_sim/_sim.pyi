"""The extension module (src/python.rs); the package's World extends its World."""

from collections.abc import Sequence
from typing import Any

DECLARATION_LAYOUT: int
CONTENT_LAYOUT: int

# A float32 number or numbers: a numpy.float32, or any object exposing a float32 buffer.
Floats = Any

class World:
    def __init__(
        self, dimension: int, *, integrator: str = "solver", open: bool = False
    ) -> None: ...
    @property
    def dimension(self) -> int: ...
    @property
    def integrator(self) -> str: ...
    @property
    def open(self) -> bool: ...
    @property
    def bodies(self) -> int: ...
    @property
    def rows(self) -> int: ...
    def add_body(
        self,
        *,
        position: Floats | None = None,
        rotation: Floats | None = None,
        velocity: Floats | None = None,
        angular_velocity: Floats | None = None,
        mass: Floats | None = None,
        gravity_scale: Floats | None = None,
        fixed: bool = False,
    ) -> int: ...
    def add_shape(
        self,
        body: int,
        kind: str,
        size: Floats,
        *,
        position: Floats | None = None,
        rotation: Floats | None = None,
        density: Floats | None = None,
        friction: Floats | None = None,
        restitution: Floats | None = None,
    ) -> None: ...
    def add_joint(
        self,
        kind: str,
        body1: int,
        body2: int,
        *,
        anchor1: Floats | None = None,
        anchor2: Floats | None = None,
        axis: Floats | None = None,
        limits: Floats | None = None,
        length: Floats | None = None,
        rest: Floats | None = None,
        stiffness: Floats | None = None,
        damping: Floats | None = None,
        contacts: bool = False,
    ) -> int: ...
    def add_gravity(self, acceleration: Floats) -> None: ...
    def add_attraction(self, bodies: Sequence[int], strength: Floats) -> None: ...
    def add_spring(
        self,
        body_a: int,
        body_b: int,
        *,
        stiffness: Floats,
        rest: Floats,
        damping: Floats | None = None,
        anchor_a: Floats | None = None,
        anchor_b: Floats | None = None,
    ) -> None: ...
    def add_linear_drag(self, body: int, coefficient: Floats) -> None: ...
    def add_quadratic_drag(self, body: int, coefficient: Floats) -> None: ...
    def add_constant(
        self, body: int, *, force: Floats | None = None, torque: Floats | None = None
    ) -> None: ...
    def add_motor(
        self,
        joint: int,
        *,
        position: Floats | None = None,
        velocity: Floats | None = None,
        stiffness: Floats | None = None,
        damping: Floats | None = None,
    ) -> None: ...
    def mass(self, body: int) -> float: ...
    def advance_to(self, row: int) -> None: ...
    def step(self, forces: Floats, torques: Floats) -> int: ...
    def _pose(
        self, start: int = 0, stop: int | None = None
    ) -> tuple[bytes, tuple[int, int, int]]: ...
    def _velocity(
        self, start: int = 0, stop: int | None = None
    ) -> tuple[bytes, tuple[int, int, int]]: ...
    def _contacts(self) -> tuple[bytes, int]: ...
    def units(self) -> tuple[str, str]: ...
    def declaration_bytes(self) -> bytes: ...
    def trace_bytes(self) -> bytes: ...
    def identity(self) -> tuple[str, str, str, str, str, str, list[str]]: ...
