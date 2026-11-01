"""Immutable values behind the mobject property API.

A mobject's points, colour arrays and pixels are read-only arrays, and a
:class:`~.VMobject`'s style is one :class:`Paint`. Setters build new values
instead of writing into the old ones, so a value handed out (to a copy, a
recorded state or a caller) never changes behind its holder's back. A write
into one raises :class:`ValueError` naming the call that replaces it.
"""

from __future__ import annotations

__all__ = [
    "FrozenArray",
    "FrozenColors",
    "FrozenPixels",
    "FrozenPointColors",
    "FrozenPoints",
    "FrozenTrackerPoints",
    "FrozenVector",
    "EMPTY_PAINT",
    "Paint",
    "adopt",
    "freeze",
]

import copy
import hashlib
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_Kind = TypeVar("_Kind", bound="FrozenArray")


class FrozenArray(np.ndarray):
    """A read-only array that holds part of a mobject's value.

    Reading it, slicing it and computing with it work as on any array, and
    arithmetic returns plain arrays. Writing into it raises
    :class:`ValueError`, whose message names the call that replaces the
    value. ``copy()`` gives a plain, writable array. Subclasses say which
    value they hold and how to change it.
    """

    value_name = "This array"
    replacement = "build a new array and assign it"

    def _refuse(self) -> NoReturn:
        raise ValueError(
            f"{self.value_name} is read-only in Arachne and can't be changed "
            f"in place: {self.replacement}."
        )

    def _writable(self) -> bool:
        return bool(self.flags.writeable)

    # Results of computations are plain arrays, not values.
    def __array_wrap__(
        self, array: np.ndarray, context: Any = None, return_scalar: bool = False
    ) -> Any:
        if return_scalar:
            return array[()]
        if type(array) is np.ndarray:
            return array
        return array.view(np.ndarray)

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        destination = _WRITING_FUNCTIONS.get(func)
        if destination is not None:
            target = args[0] if args else kwargs.get(destination)
            if isinstance(target, FrozenArray) and not target._writable():
                target._refuse()
        return super().__array_function__(func, types, args, kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        if not self._writable():
            self._refuse()
        super().__setitem__(key, value)

    # The overrides carry their own docstrings: an inherited numpy docstring
    # would bring numpy's doctests into the reference pages.
    def copy(self, order: Any = "C") -> np.ndarray:
        """A plain, writable copy of the array."""
        return np.array(self, order=order, copy=True)

    # A value prints as the array it holds, as CE's arrays did.
    def __repr__(self) -> str:
        return repr(self.view(np.ndarray))

    def __copy__(self) -> Any:
        return _frozen_copy(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        return _frozen_copy(self)

    def __reduce__(self) -> Any:
        return (_restore, (type(self), np.array(self)))

    def setflags(self, write: Any = None, align: Any = None, uic: Any = None) -> None:
        """As :meth:`numpy.ndarray.setflags`, except that a read-only value can't be
        made writable.
        """
        if write and not self._writable():
            self._refuse()
        super().setflags(write=write, align=align, uic=uic)


def _refusing(name: str) -> Callable[..., Any]:
    method = getattr(np.ndarray, name)

    def refusing(self: FrozenArray, *args: Any, **kwargs: Any) -> Any:
        if not self._writable():
            self._refuse()
        return method(self, *args, **kwargs)

    refusing.__name__ = name
    refusing.__doc__ = (
        f"As ``numpy.ndarray.{name}``, refused with :class:`ValueError` on a "
        "read-only value."
    )
    return refusing


for _name in (
    "__iadd__",
    "__isub__",
    "__imul__",
    "__imatmul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__irshift__",
    "__iand__",
    "__ixor__",
    "__ior__",
    "fill",
    "sort",
    "partition",
    "put",
    "resize",
):
    setattr(FrozenArray, _name, _refusing(_name))
del _name


def _byteswap(self: FrozenArray, inplace: bool = False) -> Any:
    """As :meth:`numpy.ndarray.byteswap`; swapping a read-only value in place is
    refused with :class:`ValueError`.
    """
    if inplace and not self._writable():
        self._refuse()
    return np.ndarray.byteswap(self, inplace)


FrozenArray.byteswap = _byteswap

# The numpy functions that write into their first argument, by its name.
_WRITING_FUNCTIONS: dict[Callable[..., Any], str] = {
    np.copyto: "dst",
    np.put: "a",
    np.place: "arr",
    np.putmask: "a",
    np.put_along_axis: "arr",
    np.fill_diagonal: "a",
}


class FrozenPoints(FrozenArray):
    value_name = "A mobject's points array"
    replacement = (
        "assign new points with mob.set_points(points) or mob.points = points, "
        "or map them with mob.apply_points_function_about_point(func)"
    )


class FrozenColors(FrozenArray):
    value_name = "A VMobject's colour array"
    replacement = (
        "change colours with mob.set_fill(...), mob.set_stroke(...), "
        "mob.set_color(...) or mob.set_opacity(...)"
    )


class FrozenVector(FrozenArray):
    value_name = "A VMobject's sheen direction"
    replacement = "use mob.set_sheen_direction(direction)"


class FrozenPointColors(FrozenArray):
    value_name = "A point cloud's rgbas array"
    replacement = "use mob.set_color(...) or assign a new array with mob.rgbas = rgbas"


class FrozenPixels(FrozenArray):
    value_name = "An image's pixel array"
    replacement = (
        "use mob.set_color(...) or mob.set_opacity(...), or assign a new array "
        "with mob.pixel_array = pixels"
    )


class FrozenTrackerPoints(FrozenArray):
    value_name = "A ValueTracker's points"
    replacement = (
        "the value is a number: use tracker.set_value(value) or "
        "tracker.increment_value(delta)"
    )


def freeze(value: Any, kind: type[_Kind]) -> _Kind:
    """Return ``value`` as a read-only array of ``kind``.

    A value that is already read-only is shared, not copied. Anything else is
    copied first, so the caller keeps its own array and can't change the
    value through it.
    """
    if isinstance(value, FrozenArray) and not value._writable():
        if type(value) is kind:
            return cast(_Kind, value)
        return cast(_Kind, value.view(kind))
    array = np.array(value)
    array.flags.writeable = False
    return cast(_Kind, array.view(kind))


def adopt(array: np.ndarray, kind: type[_Kind]) -> _Kind:
    """Return a freshly computed ``array`` as a read-only array of ``kind``.

    Unlike :func:`freeze` it doesn't copy, so it is only for arrays that the
    calling code has just built and holds no other reference to.
    """
    array.flags.writeable = False
    return cast(_Kind, array.view(kind))


def _frozen_copy(value: FrozenArray) -> FrozenArray:
    return adopt(np.array(value), type(value))


def _restore(kind: type[_Kind], array: np.ndarray) -> _Kind:
    return adopt(array, kind)


def _feed(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"N")
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"A{array.dtype.str}{array.shape}".encode())
        digest.update(array.tobytes())
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        # 4, 4.0 and np.float64(4) are one width.
        digest.update(f"F{float(value)!r}".encode())
    else:
        digest.update(f"S{type(value).__name__}:{value!r}".encode())


class Paint:
    """A :class:`~.VMobject`'s style as one immutable value.

    It carries the seven style fields that CE moves between mobjects in
    ``interpolate_color`` and ``become``: the fill, stroke and background
    stroke rgbas arrays (one row per gradient stop), the stroke and background
    stroke widths, the sheen direction and the sheen factor. A field that was
    never set is absent: reading it raises :class:`AttributeError`, as reading
    the attribute did in CE. Two paints are equal when their content is, which
    their :attr:`key` decides.

    Build one with keyword arguments, ``Paint(stroke_width=4)``, and change one
    with :meth:`replace`.
    """

    # Each field, with the kind its array is frozen as (None for a scalar).
    FIELD_KINDS: dict[str, type[FrozenArray] | None] = {
        "fill_rgbas": FrozenColors,
        "stroke_rgbas": FrozenColors,
        "background_stroke_rgbas": FrozenColors,
        "stroke_width": None,
        "background_stroke_width": None,
        "sheen_direction": FrozenVector,
        "sheen_factor": None,
    }
    FIELDS = tuple(FIELD_KINDS)

    fill_rgbas: FrozenColors
    stroke_rgbas: FrozenColors
    background_stroke_rgbas: FrozenColors
    stroke_width: Any
    background_stroke_width: Any
    sheen_direction: FrozenVector | None
    sheen_factor: Any

    def __init__(self, **fields: Any) -> None:
        self.__dict__["_key"] = None
        self._assign(fields)

    # The fields live in the instance dict, which only these methods write.
    def _assign(self, fields: dict[str, Any]) -> None:
        own = self.__dict__
        for name, value in fields.items():
            try:
                kind = self.FIELD_KINDS[name]
            except KeyError:
                raise TypeError(f"Paint has no field {name}") from None
            if kind is not None and value is not None:
                value = freeze(value, kind)
            own[name] = value

    def replace(self, **changes: Any) -> Paint:
        """A paint with ``changes`` applied to this one's fields.

        The fields it doesn't change are shared with this paint.
        """
        new = object.__new__(Paint)
        new.__dict__.update(self.__dict__)
        new.__dict__["_key"] = None
        new._assign(changes)
        return new

    def fields(self) -> dict[str, Any]:
        """The fields that are set, by name."""
        return {name: value for name, value in self.__dict__.items() if name != "_key"}

    @property
    def key(self) -> bytes:
        """The content key: a digest of every field, computed once."""
        own = self.__dict__
        key: bytes | None = own["_key"]
        if key is None:
            digest = hashlib.blake2b(digest_size=16)
            for name in self.FIELDS:
                if name in own:
                    digest.update(name.encode())
                    _feed(digest, own[name])
            key = digest.digest()
            own["_key"] = key
        return key

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError(
            "A Paint is immutable: build a new one with paint.replace(...)"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Paint):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __copy__(self) -> Paint:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Paint:
        # Copies its arrays, as CE's deep copies of a mobject do; numbers are
        # immutable already.
        new = object.__new__(Paint)
        own = new.__dict__
        own.update(self.__dict__)
        for name, value in own.items():
            if isinstance(value, FrozenArray):
                own[name] = _frozen_copy(value)
            elif name != "_key" and not isinstance(value, (int, float)):
                own[name] = copy.deepcopy(value, memo)
        return new

    def __reduce__(self) -> Any:
        return (_paint_from_fields, (self.fields(),))

    def __repr__(self) -> str:
        return f"Paint(key={self.key.hex()})"


def _paint_from_fields(fields: dict[str, Any]) -> Paint:
    return Paint(**fields)


#: The paint of a VMobject whose style was never set: CE's class default for
#: the sheen factor, and nothing else.
EMPTY_PAINT = Paint(sheen_factor=0.0)
