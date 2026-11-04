"""What each play of a record animated: its leaf animations, as the
temporal rules read them.

A play's animations are opened down to their leaves: a group (an
``AnimationGroup``, ``LaggedStart`` or ``Succession``) stands for its
members. Each :class:`Played` names the mobject it animates, the target
a transform carries it to, and what kind of animation it is:

- ``remover``: its cleanup takes its mobject out of the scene (``FadeOut``);
- ``introducer``: it brings its mobject into the scene (``FadeIn``);
- ``emphasis``: a there-and-back pulse that leaves its mobject as it found
  it (``Indicate`` and the rest of :mod:`arachne.animation.indication`);
- ``departs``: what the viewer was reading goes, as the scene's call to
  ``play`` sees it: a removal, or a transform onto another mobject than
  its own. A transform that makes its target only when it begins (a fade,
  a growth) holds a stand-in target at the call, so it counts too. An
  in-place morph (``.animate``, ``MoveToTarget``, ``ApplyMethod``,
  ``ApplyFunction``) and a pulse do not.

A mobject's family at a capture comes from the record's states there
(:func:`family_at`), not from the mobject as the run left it.
"""

from __future__ import annotations

__all__ = ["Played", "family_at", "played"]

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, PlayRecord, Record

_EMPHASIS_MODULE = "arachne.animation.indication"


@dataclass(frozen=True, eq=False)
class Played:
    """One leaf animation of a play."""

    animation: Any
    mobject: Mobject | None
    target: Mobject | None
    remover: bool
    introducer: bool
    emphasis: bool
    departs: bool


def played(play: PlayRecord) -> tuple[Played, ...]:
    """The leaf animations ``play`` ran, in the order it was given them."""
    return tuple(_described(animation) for animation in _leaves(play._animations))


def family_at(record: Record, mob: Mobject, capture: Capture) -> list[Mobject]:
    """``mob`` and everything it held at ``capture``, depth first, each once.
    A mobject the record kept no state of there counts as holding nothing.
    """
    found: list[Mobject] = []
    seen: set[int] = set()
    stack = [mob]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        found.append(current)
        state = record.node_state(current, capture)
        if state is not None:
            stack.extend(reversed(state.submobjects))
    return found


def _leaves(animations: Iterable[Any]) -> Iterator[Any]:
    from arachne.animation.animation import Animation

    for animation in animations:
        parts = getattr(animation, "animations", None)
        if isinstance(animation, Animation) and parts:
            yield from _leaves(parts)
        else:
            yield animation


def _described(animation: Any) -> Played:
    from arachne.mobject.mobject import Mobject

    mobject = getattr(animation, "mobject", None)
    target = getattr(animation, "target_mobject", None)
    return Played(
        animation=animation,
        mobject=mobject if isinstance(mobject, Mobject) else None,
        target=target if isinstance(target, Mobject) else None,
        remover=_flag(animation, "is_remover"),
        introducer=_flag(animation, "is_introducer"),
        emphasis=type(animation).__module__ == _EMPHASIS_MODULE,
        departs=_departs(animation),
    )


def _flag(animation: Any, method: str) -> bool:
    asked = getattr(animation, method, None)
    return bool(asked()) if callable(asked) else False


def _departs(animation: Any) -> bool:
    from arachne.animation.creation import Unwrite
    from arachne.animation.transform import (
        ApplyFunction,
        ApplyMethod,
        FadeTransform,
        MoveToTarget,
        ShrinkToCenter,
        Transform,
    )

    if _flag(animation, "is_remover"):
        return True
    if isinstance(animation, (Unwrite, ShrinkToCenter)):
        return True
    if isinstance(animation, (MoveToTarget, ApplyMethod, ApplyFunction)):
        return False
    if type(animation).__module__ == _EMPHASIS_MODULE:
        return False
    if isinstance(animation, FadeTransform):
        return True
    if isinstance(animation, Transform):
        if type(animation).create_target is not Transform.create_target:
            # Its target is made only when it begins: at the scene's call it
            # holds a stand-in, another mobject than its own.
            return True
        target = getattr(animation, "target_mobject", None)
        return target is not None and target is not animation.mobject
    return False
