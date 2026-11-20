"""What a world is given, as the plain, immutable values its mobject holds,
and the objects a scene declares them with.

A scene declares a world's bodies with :meth:`.World.body`, giving each its
shape (:class:`BallShape`, :class:`BoxShape`, :class:`CapsuleShape`,
:class:`HullShape`, or one taken from the body's mobject), and adds joints
(:class:`PinJoint`, :class:`SlideJoint`, :class:`WeldJoint`,
:class:`BallJoint`, :class:`RopeJoint`, :class:`SpringJoint`) and force laws
(:class:`Gravity`, :class:`Attraction`, :class:`Spring`, :class:`LinearDrag`,
:class:`QuadraticDrag`, :class:`ConstantForce`; a joint's :class:`Motor`).
The world keeps what it was given as a :class:`Declaration` of items that
hold numbers, strings, tuples and frozen points only.

**Where a Python number becomes the stored value.** Every number a world is
given passes through :func:`stored` once, and through nothing else: it takes
a Python number, refuses one that is not finite (or that single precision
cannot hold), and rounds it to the simulation's stored type, single
precision, by IEEE round-to-nearest-even. The authoring layer does its own
arithmetic before that, in Python floats (a body's centre, a shape's points
in its body's frame, a density from a mass), and none after it: the items
hold the stored numbers exactly, as Python floats, so the solver receives
them unchanged, and two hosts that compute the same floats declare the same
bytes.
"""

from __future__ import annotations

__all__ = [
    "Attraction",
    "BallJoint",
    "BallShape",
    "BoxShape",
    "CapsuleShape",
    "ConstantForce",
    "Gravity",
    "HullShape",
    "LinearDrag",
    "Motor",
    "PinJoint",
    "Push",
    "QuadraticDrag",
    "RopeJoint",
    "SlideJoint",
    "Spring",
    "SpringJoint",
    "WeldJoint",
]

import dataclasses
import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from arachne.mobject.values import FrozenPoints

    from .world import Body

#: A vector a scene gives: two or three numbers.
VectorLike: TypeAlias = Sequence[float] | np.ndarray


def stored(value: Any, what: str) -> float:
    """``value``, a Python number, as the simulation stores it: rounded to
    single precision (IEEE round-to-nearest-even, ``numpy.float32``) and held
    exactly in a Python float. A value that is not a finite number, or that
    single precision cannot hold, is refused, naming ``what`` it is.

    The one conversion of the authoring layer: no other function makes a
    number a world holds.
    """
    if isinstance(value, bool) or not isinstance(value, (Real, np.floating)):
        raise TypeError(f"{what} is a number, not {value!r}.")
    number = float(value)
    with np.errstate(over="ignore"):
        rounded = float(np.float32(number))
    if not (math.isfinite(number) and math.isfinite(rounded)):
        raise ValueError(
            f"{what} is a finite number that single precision holds, not {number!r}."
        )
    return rounded


def stored_vector(values: Any, dimension: int, what: str) -> tuple[float, ...]:
    """A vector a scene gives, as ``dimension`` stored numbers: a 2D world
    takes two numbers, or three whose last is zero (a scene's vectors are
    three-dimensional).
    """
    numbers = list(np.asarray(values, dtype=object).ravel())
    if dimension == 2 and len(numbers) == 3:
        if stored(numbers[2], what) != 0.0:
            raise ValueError(
                f"{what} lies in the plane of a 2D world: its third number is"
                f" zero, not {numbers[2]!r}."
            )
        numbers = numbers[:2]
    if len(numbers) != dimension:
        raise ValueError(
            f"{what} takes {dimension} numbers in a {dimension}D world, not"
            f" {len(numbers)}."
        )
    return tuple(stored(number, what) for number in numbers)


def given_number(value: Any, what: str) -> float:
    """``value``, a finite real number a scene gives (not a bool, not a
    string), as a Python float: what the authoring layer computes with before
    the one conversion.
    """
    if isinstance(value, bool) or not isinstance(value, (Real, np.floating)):
        raise TypeError(f"{what} is a number, not {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is a finite number, not {number!r}.")
    return number


def given_vector(values: Any, dimension: int, what: str) -> tuple[float, ...]:
    """A vector a scene gives, its numbers checked before any arithmetic, as
    ``dimension`` Python floats: a 2D world takes two numbers, or three whose
    last is zero (a scene's vectors are three-dimensional).
    """
    numbers = [
        given_number(number, what)
        for number in np.asarray(values, dtype=object).ravel()
    ]
    if dimension == 2 and len(numbers) == 3:
        if numbers[2] != 0.0:
            raise ValueError(
                f"{what} lies in the plane of a 2D world: its third number is"
                f" zero, not {numbers[2]!r}."
            )
        numbers = numbers[:2]
    if len(numbers) != dimension:
        raise ValueError(
            f"{what} takes {dimension} numbers in a {dimension}D world, not"
            f" {len(numbers)}."
        )
    return tuple(numbers)


def given_angular(value: Any, dimension: int, what: str) -> tuple[float, ...]:
    """An angular quantity a scene gives, checked: one number in 2D, a
    vector of three in 3D.
    """
    if dimension == 2:
        return (given_number(value, what),)
    return given_vector(value, 3, what)


def stored_angular(value: Any, dimension: int, what: str) -> tuple[float, ...]:
    """An angular quantity: one number in 2D (about the axis out of the
    plane), a vector of three in 3D.
    """
    if dimension == 2:
        return (stored(value, what),)
    return stored_vector(value, 3, what)


# What a scene declares with


@dataclass(frozen=True)
class BallShape:
    """A disc in a 2D world, a sphere in a 3D one, of ``radius``, centred
    ``at`` an offset from its body's centre.
    """

    radius: float
    at: VectorLike = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class BoxShape:
    """A rectangle (2D) or a box (3D) of ``width`` along x, ``height`` along
    y and ``depth`` along z (3D only), centred ``at`` an offset from its
    body's centre and, in 2D, turned by ``angle`` radians.
    """

    width: float
    height: float
    depth: float | None = None
    at: VectorLike = (0.0, 0.0, 0.0)
    angle: float = 0.0


@dataclass(frozen=True)
class CapsuleShape:
    """A segment of ``length`` along y, rounded by ``radius`` all round, centred
    ``at`` an offset from its body's centre and, in 2D, turned by ``angle``.
    """

    length: float
    radius: float
    at: VectorLike = (0.0, 0.0, 0.0)
    angle: float = 0.0


@dataclass(frozen=True)
class HullShape:
    """The convex hull of ``points``, given as offsets from its body's
    centre (scene units).
    """

    points: Any


@dataclass(frozen=True)
class Motor:
    """A pin's or a slide's motor: it drives the joint's angle (or offset)
    toward ``position`` and its rate toward ``velocity``, with a force (a
    torque for a pin) of ``stiffness`` per unit of offset and ``damping`` per
    unit of rate. The angle or offset is zero as the bodies are declared.
    """

    position: float = 0.0
    velocity: float = 0.0
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass(frozen=True)
class PinJoint:
    """``a`` and ``b`` turn about the scene point ``at`` (in 3D about
    ``axis``, as ``a`` is declared), each keeping its distance from it;
    ``limits`` bound the angle between them, zero as declared.
    """

    a: Body
    b: Body
    at: VectorLike
    axis: VectorLike | None = None
    limits: tuple[float, float] | None = None
    motor: Motor | None = None
    collide: bool = False


@dataclass(frozen=True)
class SlideJoint:
    """``b`` slides along ``axis`` (as ``a`` is declared) through the scene
    point ``at`` (``b``'s centre by default), turning with ``a``; ``limits``
    bound the offset, zero as declared.
    """

    a: Body
    b: Body
    axis: VectorLike
    at: VectorLike | None = None
    limits: tuple[float, float] | None = None
    motor: Motor | None = None
    collide: bool = False


@dataclass(frozen=True)
class WeldJoint:
    """``a`` and ``b`` move as one, as they are declared."""

    a: Body
    b: Body
    at: VectorLike | None = None
    collide: bool = False


@dataclass(frozen=True)
class BallJoint:
    """In a 3D world, ``a`` and ``b`` turn freely about the scene point
    ``at``.
    """

    a: Body
    b: Body
    at: VectorLike
    collide: bool = False


@dataclass(frozen=True)
class RopeJoint:
    """The scene points ``anchor_a`` on ``a`` and ``anchor_b`` on ``b`` (their
    centres by default) stay at most ``length`` apart.
    """

    a: Body
    b: Body
    length: float
    anchor_a: VectorLike | None = None
    anchor_b: VectorLike | None = None
    collide: bool = True


@dataclass(frozen=True)
class SpringJoint:
    """The solver's spring between the scene points ``anchor_a`` on ``a`` and
    ``anchor_b`` on ``b`` (their centres by default): ``rest`` long, of
    ``stiffness`` and ``damping``. Its force is applied inside the solver's
    substeps, where :class:`Spring`'s is held through a tick.
    """

    a: Body
    b: Body
    rest: float
    stiffness: float
    damping: float = 0.0
    anchor_a: VectorLike | None = None
    anchor_b: VectorLike | None = None
    collide: bool = True


@dataclass(frozen=True)
class Gravity:
    """A uniform ``acceleration`` on every body, times its gravity scale."""

    acceleration: VectorLike


@dataclass(frozen=True, init=False)
class Attraction:
    """Attraction by the inverse square among ``bodies``:
    ``strength * m_a * m_b / r**2`` between each pair.
    """

    bodies: tuple[Body, ...]
    strength: float

    def __init__(self, *bodies: Body, strength: float) -> None:
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(self, "strength", strength)


@dataclass(frozen=True)
class Spring:
    """A damped spring between the scene points ``anchor_a`` on ``a`` and
    ``anchor_b`` on ``b`` (their centres by default), acting along the line
    between them: ``stiffness * (length - rest)`` plus ``damping`` times the
    rate the length changes.
    """

    a: Body
    b: Body
    stiffness: float
    rest: float = 0.0
    damping: float = 0.0
    anchor_a: VectorLike | None = None
    anchor_b: VectorLike | None = None


@dataclass(frozen=True)
class LinearDrag:
    """A drag ``-coefficient * v`` on ``body``."""

    body: Body
    coefficient: float


@dataclass(frozen=True)
class QuadraticDrag:
    """A drag ``-coefficient * |v| * v`` on ``body``."""

    body: Body
    coefficient: float


@dataclass(frozen=True)
class ConstantForce:
    """A constant ``force`` at ``body``'s centre of mass and a constant
    ``torque``.
    """

    body: Body
    force: VectorLike | None = None
    torque: Any = None


@dataclass(frozen=True)
class Push:
    """What a callback force applies to one body during a step: a ``force``
    at its centre of mass and a ``torque``. A callback may give a force
    vector alone instead.
    """

    force: VectorLike | None = None
    torque: Any = None


# What a world holds


@dataclass(frozen=True, slots=True)
class ShapeItem:
    """A declared shape: its kind as the solver names it (``ball``,
    ``cuboid``, ``capsule``, ``convex_hull``), its size, its place in its
    body's frame and its material; stored numbers.
    """

    kind: str
    size: tuple[float, ...]
    position: tuple[float, ...]
    rotation: tuple[float, ...]
    density: float
    friction: float
    restitution: float


@dataclass(frozen=True, slots=True)
class BodyItem:
    """A declared body: its state as declared (stored numbers), its shapes
    (none for a point), and the drawing it moves: the points of each member
    of its mobject's family that has points, as declared (``rest``), which
    every row places by the body's motion since.
    """

    fixed: bool
    position: tuple[float, ...]
    rotation: tuple[float, ...]
    velocity: tuple[float, ...]
    angular_velocity: tuple[float, ...]
    mass: float
    gravity_scale: float
    shapes: tuple[ShapeItem, ...]
    rest: tuple[FrozenPoints, ...]


@dataclass(frozen=True, slots=True)
class MotorItem:
    position: float
    velocity: float
    stiffness: float
    damping: float


@dataclass(frozen=True, slots=True)
class JointItem:
    """A declared joint, as the solver takes it: anchors and axis in its
    bodies' frames; stored numbers.
    """

    kind: str
    body1: int
    body2: int
    anchor1: tuple[float, ...]
    anchor2: tuple[float, ...]
    axis: tuple[float, ...] | None
    limits: tuple[float, float] | None
    numbers: tuple[tuple[str, float], ...]
    contacts: bool
    motor: MotorItem | None


@dataclass(frozen=True, slots=True)
class ForceItem:
    """A declared force law: its kind (the solver's ``add_<kind>``), the
    bodies it names and its stored numbers, by keyword.
    """

    kind: str
    bodies: tuple[int, ...]
    numbers: tuple[tuple[str, tuple[float, ...]], ...]


@dataclass(frozen=True, slots=True)
class CallbackItem:
    """A callback force, by its function's name: the function itself, called
    once for each new tick with the world's state at the tick before, is held
    by the world apart from what a capture keeps of it.
    """

    name: str


@dataclass(frozen=True, slots=True)
class Declaration:
    """Everything a world was given. ``integrator`` is the one it was
    declared with, or once its first stay began the one it took. ``order``
    is the kind of each body, joint and force law (``"body"``, ``"joint"``,
    ``"force"``) in the order they were declared, across kinds: the order the
    solver is given them in.
    """

    dimension: int
    integrator: str | None = None
    bodies: tuple[BodyItem, ...] = ()
    joints: tuple[JointItem, ...] = ()
    forces: tuple[ForceItem, ...] = ()
    callbacks: tuple[CallbackItem, ...] = ()
    order: tuple[str, ...] = ()

    @property
    def open(self) -> bool:
        """Whether scene code runs in the world's step: a callback force."""
        return bool(self.callbacks)

    def digest(self) -> str:
        """A digest of everything the world was given, in a canonical form
        two processes make alike (each number by its exact bits): the
        content a world's mobject keys as (:attr:`.World._declared`).
        """
        digest = hashlib.blake2b(digest_size=16)
        for part in _canonical(self):
            digest.update(len(part).to_bytes(4, "big"))
            digest.update(part)
        return digest.hexdigest()

    def routed(self) -> str:
        """The integrator the world takes: the declared one, else
        ``point_masses`` for a world of points with no joint (springs,
        orbits, N bodies: the solver holds an applied force fixed through a
        tick, which a force law between moving bodies cannot afford), else
        ``solver``.
        """
        if self.integrator is not None:
            return self.integrator
        points = all(not body.shapes for body in self.bodies)
        return "point_masses" if points and not self.joints else "solver"


def _canonical(value: Any) -> Iterator[bytes]:
    """``value``, a declaration or a part of one, as tagged parts."""
    if value is None or isinstance(value, bool):
        yield f"{type(value).__name__}:{value}".encode()
    elif isinstance(value, int):
        yield b"int:" + str(value).encode()
    elif isinstance(value, float):
        yield b"float:" + value.hex().encode()
    elif isinstance(value, str):
        yield b"str:" + value.encode()
    elif isinstance(value, tuple):
        yield b"tuple:" + str(len(value)).encode()
        for item in value:
            yield from _canonical(item)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        yield f"array:{array.dtype.str}:{array.shape}".encode()
        yield array.tobytes()
    elif dataclasses.is_dataclass(value):
        yield b"item:" + type(value).__name__.encode()
        for field in dataclasses.fields(value):
            yield field.name.encode()
            yield from _canonical(getattr(value, field.name))
    else:
        raise TypeError(f"A declaration holds plain values, not {value!r}.")


def measure(shape: ShapeItem, dimension: int) -> float:
    """``shape``'s area (2D) or volume (3D), in Python floats, from its stored
    size: what a density is computed from.
    """
    size = shape.size
    if shape.kind == "ball":
        (radius,) = size
        return math.pi * radius**2 if dimension == 2 else 4 / 3 * math.pi * radius**3
    if shape.kind == "cuboid":
        return math.prod(2 * half for half in size)
    if shape.kind == "capsule":
        half_height, radius = size
        if dimension == 2:
            return 4 * half_height * radius + math.pi * radius**2
        return math.pi * radius**2 * 2 * half_height + 4 / 3 * math.pi * radius**3
    return _hull_measure(np.array(size, dtype=float).reshape(-1, dimension))


def _hull_measure(points: np.ndarray) -> float:
    """The area (2D) or volume (3D) of ``points``' convex hull: the hull's
    faces from Qhull, which are combinatorial, and the sums in Python floats.
    """
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise ValueError(
            "A convex hull's points span an area (2D) or a volume (3D)."
        ) from error
    if points.shape[1] == 2:
        corners = [tuple(map(float, points[i])) for i in hull.vertices]
        return abs(
            math.fsum(
                x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(
                    corners, corners[1:] + corners[:1], strict=True
                )
            )
            / 2
        )
    centre = [math.fsum(map(float, points[:, k])) / len(points) for k in range(3)]
    volumes = []
    for face in hull.simplices:
        a, b, c = ([float(points[i][k]) - centre[k] for k in range(3)] for i in face)
        volumes.append(
            abs(
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            / 6
        )
    return math.fsum(volumes)
