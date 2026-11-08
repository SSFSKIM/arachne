"""The trusted algebra: the animations a trusted play (route 1) is made of.

A play is trusted when it is analytic at any instant: built from animation
types whose interpolation is a known function of alpha and of inputs they
captured when they began (:data:`LEAVES`), under CE's own rate and path
functions (:mod:`.rates`), composed by CE's groups whose timing is exact and
whose time never runs back (:data:`GROUPS`), with no two parts that run at
once writing the same mobject. The table is code; anything not on it is
unproven, whatever it looks like: a user's subclass or rate function,
``MoveAlongPath`` (it reads its path live), ``Wiggle`` (it scales about its
mobject's live centre), ``ChangeSpeed`` (its own clock).

:func:`refusals` says why a play's animations are outside the algebra, and
:func:`writers` lists what each of them writes, for the dependency order of
route 2 (:mod:`.admission`).
"""

from __future__ import annotations

__all__ = [
    "CALLBACK_ANIMATIONS",
    "GROUPS",
    "LEAVES",
    "NEVER_ADMITTED",
    "iter_animations",
    "refusals",
    "user_overrides",
    "writes",
]

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

from arachne.animation import (
    composition,
    creation,
    fading,
    growing,
    indication,
    rotation,
    specialized,
    transform,
    transform_matching_parts,
)
from arachne.animation.animation import Add, Animation, Wait
from arachne.animation.updaters.update import UpdateFromAlphaFunc, UpdateFromFunc

from .dispatch import family_refusal, instance_shadows, user_overrides
from .frozen import engine_class
from .identity import ByIdentity, one_of
from .rates import admitted_path, admitted_rate, is_monotone

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject

#: Leaf animations whose interpolation is a function of alpha and of the
#: copies they made when they began; each entry says what it checks beyond
#: its rate function.
_TRANSFORMS: tuple[type[Animation], ...] = (
    transform.Transform,
    transform.ReplacementTransform,
    transform.TransformFromCopy,
    transform.ClockwiseTransform,
    transform.CounterclockwiseTransform,
    transform.MoveToTarget,
    transform._MethodAnimation,
    transform.ApplyMethod,
    transform.ApplyPointwiseFunction,
    transform.FadeToColor,
    transform.ScaleInPlace,
    transform.ShrinkToCenter,
    transform.Restore,
    transform.ApplyFunction,
    transform.ApplyMatrix,
    transform.ApplyComplexFunction,
    transform.CyclicReplace,
    transform.Swap,
    fading.FadeIn,
    fading.FadeOut,
    growing.GrowFromPoint,
    growing.GrowFromCenter,
    growing.GrowFromEdge,
    growing.GrowArrow,
    growing.SpinInFromNothing,
    rotation.Rotate,
    indication.FocusOn,
    indication.Indicate,
)
_REVEALS: tuple[type[Animation], ...] = (
    creation.Create,
    creation.Uncreate,
    creation.DrawBorderThenFill,
    creation.Write,
    creation.Unwrite,
    indication.ShowPassingFlash,
)


def _path(animation: Animation) -> str | None:
    if not admitted_path(animation.path_func):  # type: ignore[attr-defined]
        return f"its path function {animation.path_func!r} is not CE's"  # type: ignore[attr-defined]
    return None


def _nothing(animation: Animation) -> str | None:
    return None


#: The leaf entries: type -> a check returning why an instance is outside the
#: algebra, or ``None``. Matched by exact type, by identity: a subclass may
#: interpolate its own way, whatever its metaclass says it equals.
LEAVES: ByIdentity[type[Animation], Callable[[Animation], str | None]] = ByIdentity(
    {
        Animation: _nothing,
        Wait: _nothing,
        Add: _nothing,
        rotation.Rotating: _nothing,
        **dict.fromkeys(_TRANSFORMS, _path),
        **dict.fromkeys(_REVEALS, _nothing),
    }
)

#: Compositions whose interpolation is CE's group or succession timing over
#: their parts, by identity.
GROUPS: ByIdentity[type[Animation], None] = ByIdentity.of(
    (
        composition.AnimationGroup,
        composition.LaggedStart,
        composition.LaggedStartMap,
        composition.Succession,
        specialized.Broadcast,
        indication.Flash,
        indication.Circumscribe,
        indication.Blink,
        indication.ShowPassingFlashWithThinningStrokeWidth,
        transform_matching_parts.TransformMatchingShapes,
    )
)

#: Built-ins that keep state outside everything a record registers, named so
#: their refusal says why.
NEVER_ADMITTED: dict[str, str] = {
    "ChangeSpeed": "ChangeSpeed keeps its own clock and replaces its inner animations' rate functions",
    "Wiggle": "Wiggle scales and rotates about its mobject's live centre, so it is not a function of alpha",
    "MoveAlongPath": "MoveAlongPath reads its path live at every evaluation",
    "MaintainPositionRelativeTo": "MaintainPositionRelativeTo reads another mobject live",
    "PhaseFlow": "PhaseFlow integrates over its evaluations",
}


def iter_animations(animations: Sequence[Animation]) -> Iterator[Animation]:
    """Every animation of a play, the groups' parts included, depth first."""
    for animation in animations:
        yield animation
        parts = getattr(animation, "animations", None)
        if isinstance(animation, composition.AnimationGroup) and parts:
            yield from iter_animations(parts)


def writes(
    animation: Animation, callback_writes: Mapping[int, set[int]] | None = None
) -> set[int]:
    """The ids of the mobjects ``animation``'s interpolation writes: its
    mobject's family for a leaf, the union of its parts' for a group, and for
    a callback animation what its function's proof writes
    (``callback_writes``, by the animation's id).
    """
    if isinstance(animation, composition.AnimationGroup):
        found: set[int] = set()
        for part in animation.animations:
            found |= writes(part, callback_writes)
        return found
    if one_of(type(animation), (Wait, Add, Animation)):
        return set()
    mobject: Mobject | None = animation.mobject
    if callback_writes is not None and id(animation) in callback_writes:
        own = {id(member) for member in mobject.get_family()} if mobject else set()
        return own | callback_writes[id(animation)]
    return (
        {id(member) for member in mobject.get_family()}
        if mobject is not None
        else set()
    )


def refusals(
    animations: Sequence[Animation],
    callback_writes: Mapping[int, set[int]] | None = None,
) -> list[str]:
    """Why the play of ``animations`` is outside the algebra routes 1 and 2
    evaluate: empty when every animation is admitted and no two that run at
    once write the same mobject, a callback animation's writes
    (``callback_writes``) included.
    """
    found: list[str] = []
    for animation in animations:
        _refuse(animation, found, callback_writes)
    if not found:
        _conflicts(animations, "the play", found, callback_writes)
    return found


#: Animations that run a script's function: route 2 proves the function or
#: the play ticks (:mod:`.admission`); as parts of a group they are timed by
#: it. By exact type, by identity.
CALLBACK_ANIMATIONS: ByIdentity[type[Animation], None] = ByIdentity.of(
    (UpdateFromFunc, UpdateFromAlphaFunc)
)


def _refuse(
    animation: Animation,
    found: list[str],
    callback_writes: Mapping[int, set[int]] | None,
) -> None:
    kind = type(animation)
    name = kind.__name__
    shadows = instance_shadows(animation)
    if shadows:
        found.append(f"{name}'s {', '.join(shadows)} is replaced on the instance")
        return
    if kind in CALLBACK_ANIMATIONS:
        # Its function is route 2's to prove; its timing is the algebra's.
        if not admitted_rate(animation.rate_func):
            found.append(
                f"{name}'s rate function {animation.rate_func!r} is not one of CE's"
            )
        return
    if name in NEVER_ADMITTED and engine_class(kind):
        found.append(NEVER_ADMITTED[name])
        return
    if not admitted_rate(animation.rate_func):
        found.append(
            f"{name}'s rate function {animation.rate_func!r} is not one of CE's"
        )
        return
    if kind in GROUPS:
        if not is_monotone(animation.rate_func):
            found.append(
                f"{name}'s rate function runs its parts' time back"
                f" ({animation.rate_func.__name__})"
            )
            return
        for part in animation.animations:  # type: ignore[attr-defined]
            _refuse(part, found, callback_writes)
        if not isinstance(animation, composition.Succession):
            _conflicts(animation.animations, name, found, callback_writes)  # type: ignore[attr-defined]
        return
    check = LEAVES.get(kind)
    if check is None:
        found.append(f"{name} is not in the trusted algebra")
        return
    reason = check(animation)
    if reason is not None:
        found.append(f"{name}: {reason}")
        return
    dispatch = family_refusal(_mobjects_of(animation))
    if dispatch is not None:
        found.append(f"{name} animates {dispatch}")


def _mobjects_of(animation: Animation) -> list[Mobject]:
    """The mobjects a leaf animation interpolates and copies: its mobject,
    its target and the copies it made when it began.
    """
    from arachne.mobject.mobject import Mobject

    found = []
    for key in ("mobject", "target_mobject", "starting_mobject", "target_copy"):
        value = vars(animation).get(key)
        if isinstance(value, Mobject):
            found.append(value)
    return found


def _conflicts(
    parts: Sequence[Animation],
    where: str,
    found: list[str],
    callback_writes: Mapping[int, set[int]] | None = None,
) -> None:
    """Record a conflict when two parts that run at once write one mobject."""
    seen: set[int] = set()
    for part in parts:
        written = writes(part, callback_writes)
        if seen & written:
            found.append(f"two parts of {where} write the same mobject")
            return
        seen |= written
