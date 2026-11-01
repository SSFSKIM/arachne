"""Utilities for working with mobjects."""

from __future__ import annotations

__all__ = [
    "get_mobject_class",
    "get_point_mobject_class",
    "get_vectorized_mobject_class",
]

from .mobject import Mobject
from .types.point_cloud_mobject import PMobject
from .types.vectorized_mobject import VMobject


def get_mobject_class() -> type:
    """Gets the base mobject class.

    Kept for plugins written against both of Manim's renderers; Cairo is the
    only renderer, so this is always :class:`.Mobject`.

    Examples
    --------
    ::

        >>> from arachne.mobject.utils import get_mobject_class
        >>> get_mobject_class().__name__
        'Mobject'
    """
    return Mobject


def get_vectorized_mobject_class() -> type:
    """Gets the vectorized mobject class.

    Kept for plugins written against both of Manim's renderers; Cairo is the
    only renderer, so this is always :class:`.VMobject`.

    Examples
    --------
    ::

        >>> from arachne.mobject.utils import get_vectorized_mobject_class
        >>> get_vectorized_mobject_class().__name__
        'VMobject'
    """
    return VMobject


def get_point_mobject_class() -> type:
    """Gets the point cloud mobject class.

    Kept for plugins written against both of Manim's renderers; Cairo is the
    only renderer, so this is always :class:`.PMobject`.

    Examples
    --------
    ::

        >>> from arachne.mobject.utils import get_point_mobject_class
        >>> get_point_mobject_class().__name__
        'PMobject'
    """
    return PMobject
