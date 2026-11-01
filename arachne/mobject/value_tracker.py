"""Simple mobjects that can be used for storing (and updating) a value."""

from __future__ import annotations

__all__ = ["ValueTracker", "ComplexValueTracker"]

from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.mobject.mobject import Mobject
from arachne.mobject.values import FrozenTrackerPoints, adopt
from arachne.utils.paths import straight_path

if TYPE_CHECKING:
    from typing import Self

    from arachne.typing import PathFuncType, Point3DLike_Array


def _as_float(value: Any) -> np.float64:
    """``value`` cast as storing it in a float array casts it."""
    cell = np.empty(1)
    cell[0] = value
    return cell[0]


class ValueTracker(Mobject):
    """A mobject that can be used for tracking (real-valued) parameters.
    Useful for animating parameter changes.

    Not meant to be displayed.  Instead the position encodes some
    number, often one which another animation or continual_animation
    uses for its update function, and by treating it as a mobject it can
    still be animated and manipulated just like anything else.

    This value changes continuously when animated using the :attr:`animate` syntax.

    The value is held as a number. :attr:`points` is a read-only view of it,
    ``[[value, 0, 0]]``, kept so that code treating the tracker as a point keeps
    working: assigning points, or moving the tracker as a mobject, sets the value
    from the first coordinate.

    Examples
    --------
    .. arachne:: ValueTrackerExample

        class ValueTrackerExample(Scene):
            def construct(self):
                number_line = NumberLine()
                pointer = Vector(DOWN)
                label = MathTex("x").add_updater(lambda m: m.next_to(pointer, UP))

                tracker = ValueTracker(0)
                pointer.add_updater(
                    lambda m: m.next_to(
                                number_line.n2p(tracker.get_value()),
                                UP
                            )
                )
                self.add(number_line, pointer,label)
                tracker += 1.5
                self.wait(1)
                tracker -= 4
                self.wait(0.5)
                self.play(tracker.animate.set_value(5))
                self.wait(0.5)
                self.play(tracker.animate.set_value(3))
                self.play(tracker.animate.increment_value(-2))
                self.wait(0.5)

    .. note::

        You can also link ValueTrackers to updaters. In this case, you have to make sure that the
        ValueTracker is added to the scene by ``add``

    .. arachne:: ValueTrackerExample

        class ValueTrackerExample(Scene):
            def construct(self):
                tracker = ValueTracker(0)
                label = Dot(radius=3).add_updater(lambda x : x.set_x(tracker.get_value()))
                self.add(label)
                self.add(tracker)
                tracker.add_updater(lambda mobject, dt: mobject.increment_value(dt))
                self.wait(2)

    """

    _value: Any

    def __init__(self, value: float = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_value(value)

    # The value is the storage behind the points property: Mobject's setter and
    # its in-house writers assign _points, which lands here.
    @property
    def _points(self) -> FrozenTrackerPoints:
        return adopt(self._value_as_point()[np.newaxis], FrozenTrackerPoints)

    @_points.setter
    def _points(self, points: Point3DLike_Array) -> None:
        points = np.asarray(points)
        if points.shape[:1] == (0,):
            raise ValueError(
                f"A {type(self).__name__} holds one value, so its points can't be "
                "empty: use set_value(value)"
            )
        self._set_value_from_point(points[0])

    def _value_as_point(self) -> np.ndarray:
        return np.array([self._value, 0.0, 0.0])

    def _set_value_from_point(self, point: np.ndarray) -> None:
        self._value = _as_float(point[0])

    def reset_points(self) -> Self:
        self.set_value(0)
        return self

    def get_value(self) -> float:
        """Get the current value of this ValueTracker."""
        value: float = self._value
        return value

    def set_value(self, value: float) -> Self:
        """Sets a new scalar value to the ValueTracker."""
        self._value = _as_float(value)
        return self

    def increment_value(self, d_value: float) -> Self:
        """Increments (adds) a scalar value to the ValueTracker."""
        self.set_value(self.get_value() + d_value)
        return self

    def __bool__(self) -> bool:
        """Return whether the value of this ValueTracker evaluates as true."""
        return bool(self.get_value())

    def __add__(self, d_value: float | Mobject) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value plus
        ``d_value``.
        """
        if isinstance(d_value, Mobject):
            raise ValueError(
                "Cannot increment ValueTracker by a Mobject. Please provide a scalar value."
            )
        return ValueTracker(self.get_value() + d_value)

    def __iadd__(self, d_value: float | Mobject) -> Self:
        """adds ``+=`` syntax to increment the value of the ValueTracker."""
        if isinstance(d_value, Mobject):
            raise ValueError(
                "Cannot increment ValueTracker by a Mobject. Please provide a scalar value."
            )
        self.increment_value(d_value)
        return self

    def __floordiv__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the floor division of the current
        tracker's value by ``d_value``.
        """
        return ValueTracker(self.get_value() // d_value)

    def __ifloordiv__(self, d_value: float) -> Self:
        """Set the value of this ValueTracker to the floor division of the current value by ``d_value``."""
        self.set_value(self.get_value() // d_value)
        return self

    def __mod__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value
        modulo ``d_value``.
        """
        return ValueTracker(self.get_value() % d_value)

    def __imod__(self, d_value: float) -> Self:
        """Set the value of this ValueTracker to the current value modulo ``d_value``."""
        self.set_value(self.get_value() % d_value)
        return self

    def __mul__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value multiplied by
        ``d_value``.
        """
        return ValueTracker(self.get_value() * d_value)

    def __imul__(self, d_value: float) -> Self:
        """Set the value of this ValueTracker to the product of the current value and ``d_value``."""
        self.set_value(self.get_value() * d_value)
        return self

    def __pow__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value raised to the
        power of ``d_value``.
        """
        return ValueTracker(self.get_value() ** d_value)

    def __ipow__(self, d_value: float) -> Self:
        """Set the value of this ValueTracker to the current value raised to the power of ``d_value``."""
        self.set_value(self.get_value() ** d_value)
        return self

    def __sub__(self, d_value: float | Mobject) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value minus
        ``d_value``.
        """
        if isinstance(d_value, Mobject):
            raise ValueError(
                "Cannot decrement ValueTracker by a Mobject. Please provide a scalar value."
            )
        return ValueTracker(self.get_value() - d_value)

    def __isub__(self, d_value: float | Mobject) -> Self:
        """Adds ``-=`` syntax to decrement the value of the ValueTracker."""
        if isinstance(d_value, Mobject):
            raise ValueError(
                "Cannot decrement ValueTracker by a Mobject. Please provide a scalar value."
            )
        self.increment_value(-d_value)
        return self

    def __truediv__(self, d_value: float) -> ValueTracker:
        """Return a new :class:`ValueTracker` whose value is the current tracker's value
        divided by ``d_value``.
        """
        return ValueTracker(self.get_value() / d_value)

    def __itruediv__(self, d_value: float) -> Self:
        """Sets the value of this ValueTracker to the current value divided by ``d_value``."""
        self.set_value(self.get_value() / d_value)
        return self

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: PathFuncType = straight_path(),
    ) -> Self:
        """Turns ``self`` into an interpolation between ``mobject1`` and ``mobject2``."""
        self.set(points=path_func(mobject1.points, mobject2.points, alpha))
        return self


class ComplexValueTracker(ValueTracker):
    """Tracks a complex-valued parameter.

    The value is held as a complex number, and :attr:`points` shows it as the point
    ``[a, b, 0]``, which can be read directly to represent the value geometrically, see the
    usage example.
    When the value is set through :attr:`animate`, the value will take a straight path from the
    source point to the destination point.

    Examples
    --------
    .. arachne:: ComplexValueTrackerExample

        class ComplexValueTrackerExample(Scene):
            def construct(self):
                tracker = ComplexValueTracker(-2+1j)
                dot = Dot().add_updater(
                    lambda x: x.move_to(tracker.points)
                )

                self.add(NumberPlane(), dot)

                self.play(tracker.animate.set_value(3+2j))
                self.play(tracker.animate.set_value(tracker.get_value() * 1j))
                self.play(tracker.animate.set_value(tracker.get_value() - 2j))
                self.play(tracker.animate.set_value(tracker.get_value() / (-2 + 3j)))
    """

    def _value_as_point(self) -> np.ndarray:
        return np.array([self._value.real, self._value.imag, 0.0])

    def _set_value_from_point(self, point: np.ndarray) -> None:
        self._value = complex(_as_float(point[0]), _as_float(point[1]))

    def get_value(self) -> complex:  # type: ignore [override]
        """Get the current value of this ComplexValueTracker as a complex number."""
        value: complex = self._value
        return value

    def set_value(self, value: complex | float) -> Self:
        """Sets a new complex value to the ComplexValueTracker."""
        z = complex(value)
        self._value = complex(_as_float(z.real), _as_float(z.imag))
        return self
