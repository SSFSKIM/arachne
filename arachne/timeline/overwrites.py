"""What an animation's interpolation overwrites in its target from what it
holds alone.

Some trusted animations set fields of each member of their mobject (the
target) from the copies they made as they began and from alpha, whatever
those fields held before the call: ``Create`` sets each member's points to a
part of its starting copy's, a ``Transform`` (``FadeIn`` and ``FadeOut``
among them) sets each member's points and paint between its starting copy's
and its target copy's. :func:`overwritten_fields` says which fields, for one
animation as it stands.

A declaration is keyed on the function objects the interpolation runs, not
on a class: it applies only while every method on the path is the declaring
class's own (a subclass or an instance that replaces one inherits nothing),
the rate, bounds and path functions are the engine's, and every member's
class runs the engine's own point and colour interpolation.

Metadata only: the stepping plan (:mod:`.ticking`) does not read it.
"""

from __future__ import annotations

__all__ = ["overwritten_fields"]

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .frozen import engine_code

if TYPE_CHECKING:
    from arachne.animation.animation import Animation


@dataclass(frozen=True)
class _Declaration:
    #: The instance fields of each target member the interpolation overwrites.
    fields: frozenset[str]
    #: Each method of the animation the interpolation runs, by name, with the
    #: function object it must be.
    path: tuple[tuple[str, Any], ...]
    #: Callables the animation holds or defines that the path calls with
    #: alpha (or with the copies' points): each must be the engine's code.
    engine: tuple[str, ...]
    #: Each method of a target member's class the path calls, with the
    #: function (or property) object it must be.
    member: tuple[tuple[str, Any], ...]
    #: Whether each member of the starting copy with points must hold a whole
    #: curve: ``pointwise_become_partial`` from one with none leaves the
    #: target's points as they were.
    whole_curves: bool = False


def _declarations() -> tuple[_Declaration, ...]:
    from arachne.animation.animation import Animation
    from arachne.animation.creation import ShowPartial
    from arachne.animation.transform import Transform
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.types.vectorized_mobject import VMobject

    common = (
        ("interpolate", Animation.interpolate),
        ("interpolate_mobject", Animation.interpolate_mobject),
        ("get_sub_alpha", Animation.get_sub_alpha),
    )
    points = ("points", inspect.getattr_static(Mobject, "points"))
    return (
        # Each member's points: a part of its starting copy's.
        _Declaration(
            frozenset({"_points"}),
            (
                *common,
                ("get_all_mobjects", Animation.get_all_mobjects),
                ("get_all_families_zipped", Animation.get_all_families_zipped),
                ("interpolate_submobject", ShowPartial.interpolate_submobject),
            ),
            ("rate_func", "_get_bounds"),
            (
                points,
                ("pointwise_become_partial", VMobject.pointwise_become_partial),
                ("set_points", Mobject.set_points),
            ),
            whole_curves=True,
        ),
        # Each member's points and paint: between its starting copy's and its
        # target copy's.
        _Declaration(
            frozenset({"_points", "_paint"}),
            (
                *common,
                ("get_all_families_zipped", Transform.get_all_families_zipped),
                ("interpolate_submobject", Transform.interpolate_submobject),
            ),
            ("rate_func", "path_func"),
            (
                points,
                ("interpolate", Mobject.interpolate),
                ("interpolate_color", VMobject.interpolate_color),
            ),
        ),
    )


_DECLARATIONS: tuple[_Declaration, ...] | None = None


def overwritten_fields(animation: Animation) -> frozenset[str]:
    """The instance fields of each member of ``animation``'s mobject that
    have points which a call of ``animation.interpolate(alpha)`` sets from
    the copies it holds and ``alpha`` alone: after the call, they are the
    same whatever they held before it. Empty when no declaration applies.

    It holds for a target whose members with points pair with the held
    copies' members as they did when the animation began (``begin`` pairs
    them, and the interpolation keeps them so): a member that lost all its
    points, or gained some its copy lacks, pairs the families otherwise (a
    ``Transform`` then raises; ``Create`` pairs them silently), so a reader
    relying on it checks that the target's members with points are as many
    as the held copy's.
    """
    global _DECLARATIONS
    if _DECLARATIONS is None:
        _DECLARATIONS = _declarations()
    cls = type(animation)
    own = vars(animation)
    for declaration in _DECLARATIONS:
        if any(
            name in own or inspect.getattr_static(cls, name, None) is not function
            for name, function in declaration.path
        ):
            continue
        if not all(
            engine_code(_function(getattr(animation, name, None)))
            for name in declaration.engine
        ):
            continue
        if declaration.whole_curves and not _whole_curves(animation):
            continue
        if all(
            _member_runs(member, declaration.member)
            for member in animation.mobject.family_members_with_points()
        ):
            return declaration.fields
    return frozenset()


def _whole_curves(animation: Animation) -> bool:
    """Whether every member with points of the copy the animation holds
    holds a whole curve: of its mobject, which the copy is made from, until
    it began (its copy has no points until then).
    """
    held = animation.starting_mobject
    if not held.family_members_with_points():
        held = animation.mobject
    from arachne.mobject.types.vectorized_mobject import VMobject

    return all(
        isinstance(member, VMobject) and member.get_num_curves() >= 1
        for member in held.family_members_with_points()
    )


def _function(value: Any) -> Any:
    return getattr(value, "__func__", value)


def _member_runs(member: Any, methods: tuple[tuple[str, Any], ...]) -> bool:
    cls = type(member)
    own = vars(member)
    return all(
        name not in own and inspect.getattr_static(cls, name, None) is function
        for name, function in methods
    )
