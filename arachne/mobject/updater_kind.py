"""Whether an updater takes ``dt``: the kind :meth:`.Mobject.update` reads
from its signature to decide how to call it, ``updater(mobject, dt)`` or
``updater(mobject)``.

``inspect.signature`` builds a whole signature to answer that one question,
and the engine asks it of every updater at every frame and tick. While the
speed levers are on (``speed_levers``; the kind is exact and has nothing to
tune, so it has no lever of its own), the answer for a function or a bound
method is kept beside what ``inspect.signature`` reads it from: the
function's code and the ``__signature__`` it carries. Each call compares
those by identity and reads the signature again when either was replaced,
so the kept answer is the one a read at that call gives. A function's
defaults and annotations name no parameter, so neither is compared.

Anything else is read at every call, as are functions that send
``inspect.signature`` elsewhere (``__wrapped__``, a ``partialmethod``, a
text signature): a callable object's answer follows its class, a
``functools.partial``'s the function it holds, and neither is compared
cheaply. A signature that can't be read raises at every call, as it does
when nothing is kept.
"""

from __future__ import annotations

__all__ = ["takes_dt"]

import inspect
import types
import weakref
from typing import Any

from .. import config

#: Attributes through which ``inspect.signature`` reads a function's
#: parameters from something other than its code and ``__signature__``.
_ELSEWHERE = (
    "__wrapped__",
    "__partialmethod__",
    "_partialmethod",
    "__text_signature__",
)

#: Per function or bound method: the code and ``__signature__`` its kind was
#: read from, and the kind. A bound method keys by its receiver's identity
#: and its function, so the one an updater list holds finds its entry.
_KINDS: weakref.WeakKeyDictionary[
    Any, tuple[types.CodeType, inspect.Signature | None, bool]
] = weakref.WeakKeyDictionary()


def takes_dt(updater: Any) -> bool:
    """Whether ``updater``'s signature names a ``dt`` parameter:
    ``"dt" in inspect.signature(updater).parameters``, raising what that
    raises.
    """
    if not config.speed_levers:
        return _read(updater)
    function = updater.__func__ if type(updater) is types.MethodType else updater
    if type(function) is not types.FunctionType:
        return _read(updater)
    attributes = function.__dict__
    signature = attributes.get("__signature__")
    if attributes and (
        any(name in attributes for name in _ELSEWHERE)
        or not (signature is None or type(signature) is inspect.Signature)
    ):
        return _read(updater)
    code = function.__code__
    kept = _KINDS.get(updater)
    if kept is not None and kept[0] is code and kept[1] is signature:
        return kept[2]
    kind = _read(updater)
    _KINDS[updater] = (code, signature, kind)
    return kind


def _read(updater: Any) -> bool:
    return "dt" in inspect.signature(updater).parameters
