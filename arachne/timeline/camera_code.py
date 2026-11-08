"""The camera methods and display functions a scene's code replaced.

A scene can change how an engine camera draws without defining a camera
class of its own: it puts its own callable on the camera instance in place
of a method (``camera.project_points = shifted``), or it replaces a method
on one of the engine's camera classes (``ThreeDCamera.project_points =
shifted``). Either way the camera then runs the scene's code while it draws.
:func:`replaced_methods` finds both, from the camera alone, on the default
path (it imports no ``arachne_core``).

A class patch is told from what the class held when the engine was imported
(:mod:`~arachne.timeline.frozen`), so a patch made before the scene module
runs, or in another process, is found the same. Every method of the engine's
camera classes counts, not only those a draw is known to reach: a replaced
method a draw never calls costs a fallback, never a wrong frame.
"""

from __future__ import annotations

__all__ = [
    "Replaced",
    "display_funcs_rewired",
    "drawn_cameras",
    "engine_display",
    "engine_own",
    "replaced_methods",
]

import functools
import types
import weakref
from dataclasses import dataclass
from typing import Any, Literal

#: The attribute that marks a callable the engine itself puts on a camera
#: instance while it draws (live drawing's capture wrappers).
_ENGINE_OWN = "__arachne_engine_own__"


#: The functions the engine puts in a camera's ``display_funcs`` while it
#: draws (:func:`engine_display`), each with the entry it stands over. Found
#: by the function object itself, never by anything the function says about
#: itself: a scene's function can carry any attribute.
_DISPLAY_WRAPPERS: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def engine_display(wrapper: Any, wrapped: Any) -> Any:
    """``wrapper``, a function the engine puts in a camera's
    ``display_funcs`` in place of ``wrapped``, recorded as the engine's so
    :func:`_replaced_displays` reads it as ``wrapped``.
    """
    _DISPLAY_WRAPPERS[wrapper] = wrapped
    return wrapper


def engine_own(function: Any) -> Any:
    """``function``, marked as one the engine puts on a camera instance, so
    :func:`replaced_methods` does not count it as the scene's.
    """
    setattr(function, _ENGINE_OWN, True)
    return function


@dataclass(frozen=True)
class Replaced:
    """One camera method, or display function, the scene's code replaced."""

    #: The method's name.
    name: str
    #: ``"instance"``: the camera instance holds a callable of its own under
    #: the name. ``"class"``: an engine camera class holds something other
    #: than what it held when the engine was imported. ``"display_funcs"``:
    #: an entry of the camera's table of display functions, named
    #: ``display_funcs[Kind]``, is not the engine's own (or the table is
    #: not, named ``display_funcs``).
    form: Literal["instance", "class", "display_funcs"]
    #: The class whose method is replaced: the engine class patched, or the
    #: camera's own class for an instance callable or a display entry.
    owner: type
    #: What stands in the method's place now (``None`` where a class patch
    #: deleted it).
    method: Any

    @property
    def qualname(self) -> str:
        """``Owner.name``."""
        return f"{self.owner.__name__}.{self.name}"


def drawn_cameras(camera: Any) -> tuple[Any, ...]:
    """``camera`` and every camera a frame it draws is drawn with: a
    MultiCamera's display cameras (``image_mobjects_from_cameras``), and
    theirs in turn, each once. A display camera the scene's code changed
    changes the frame as much as the scene's camera would.
    """
    found: list[Any] = []
    stack = [camera]
    while stack:
        each = stack.pop(0)
        if each is None or any(each is seen for seen in found):
            continue
        found.append(each)
        displays = getattr(each, "__dict__", {}).get("image_mobjects_from_cameras", ())
        stack.extend(getattr(display, "camera", None) for display in displays)
    return tuple(found)


def replaced_methods(camera: Any) -> tuple[Replaced, ...]:
    """Every method of ``camera`` the scene's code replaced: class patches
    first (by class, then name), then instance callables (by name), then
    entries of its table of display functions (:func:`_replaced_displays`).
    Empty for a camera as the engine made it, and for anything that is no
    engine camera.
    """
    from arachne.camera.camera import Camera

    from .frozen import class_members, engine_class

    if not isinstance(camera, Camera):
        return ()
    kind = type(camera)
    found: list[Replaced] = []
    for klass in kind.__mro__:
        if not engine_class(klass):
            continue
        taken = class_members(klass)
        now = vars(klass)
        for name in sorted(set(now) | set(taken)):
            was, held = taken.get(name), now.get(name)
            if held is not was and (callable(was) or callable(held)):
                found.append(Replaced(name, "class", klass, held))
    found.extend(
        Replaced(name, "instance", kind, value)
        for name, value in sorted(vars(camera).items())
        if callable(value)
        and not getattr(value, _ENGINE_OWN, False)
        and callable(getattr(kind, name, None))
    )
    found.extend(_replaced_displays(camera))
    return tuple(found)


#: The camera's display function for each kind of mobject, as
#: ``Camera.__init__`` makes them: the engine's method of that name, bound to
#: the camera; the do-nothing function for any other mobject (``None``).
_DISPLAY_METHODS = {
    "VMobject": "display_multiple_vectorized_mobjects",
    "PMobject": "display_multiple_point_cloud_mobjects",
    "AbstractImageMobject": "display_multiple_image_mobjects",
    "Mobject": None,
}


@functools.cache
def _nothing_code() -> types.CodeType:
    """The code of the do-nothing display function ``Camera.__init__``
    makes: the one function it defines.
    """
    from arachne.camera.camera import Camera

    (code,) = (
        const
        for const in Camera.__init__.__code__.co_consts
        if isinstance(const, types.CodeType)
    )
    return code


def _engine_method(kind: type, name: str) -> Any:
    """The function the engine defined as ``name`` for the camera class
    ``kind``: the first engine class in its order that defines it, as it was
    when the engine was imported.
    """
    from .frozen import class_members, engine_class

    return next(
        (
            class_members(klass)[name]
            for klass in kind.__mro__
            if engine_class(klass) and name in class_members(klass)
        ),
        None,
    )


def display_funcs_rewired(camera: Any) -> bool:
    """Whether ``camera.display_funcs`` holds anything other than what
    ``Camera.__init__`` puts there (:func:`_replaced_displays`), that is,
    any entry the scene's code replaced.
    """
    return bool(_replaced_displays(camera))


def _replaced_displays(camera: Any) -> list[Replaced]:
    """The entries of ``camera.display_funcs`` that are not the engine's own,
    named by kind (``display_funcs`` itself, whole, when its kinds are not
    the engine's); none when every one is. A live painter's own wrappers
    (:func:`engine_display`), installed while it describes the camera's
    captures, stand for the entries they wrap; while it is installed the
    camera is not read at all (``_xf_live``).
    """
    from arachne.mobject.mobject import Mobject
    from arachne.mobject.types.image_mobject import AbstractImageMobject
    from arachne.mobject.types.point_cloud_mobject import PMobject
    from arachne.mobject.types.vectorized_mobject import VMobject

    own = vars(camera)
    if "_xf_live" in own:
        return []
    funcs = own.get("display_funcs")
    kinds = {
        kind.__name__: kind
        for kind in (VMobject, PMobject, AbstractImageMobject, Mobject)
    }
    kind = type(camera)
    if not isinstance(funcs, dict) or set(funcs) != set(kinds.values()):
        return [Replaced("display_funcs", "display_funcs", kind, funcs)]
    changed = []
    for name, method in _DISPLAY_METHODS.items():
        func = funcs[kinds[name]]
        while type(func) is types.FunctionType and func in _DISPLAY_WRAPPERS:
            func = _DISPLAY_WRAPPERS[func]
        # A plain function or bound method first, by its exact type: any
        # other object can answer the reads below as the engine's does and
        # still run its own code when called.
        if method is None:
            # The do-nothing lambda Camera.__init__ makes, by its code: a
            # wrapper can copy its name and module, not its code.
            engine = (
                type(func) is types.FunctionType and func.__code__ is _nothing_code()
            )
        else:
            # Bound to this camera, as the engine defined it: one bound to
            # another camera (a shallow copy shares the original's
            # dictionary) draws with that camera's settings, and one of a
            # class patched or subclassed runs code the engine did not make.
            engine = (
                type(func) is types.MethodType
                and func.__self__ is camera
                and func.__func__ is _engine_method(kind, method)
            )
        if not engine:
            changed.append(
                Replaced(f"display_funcs[{name}]", "display_funcs", kind, func)
            )
    return changed
