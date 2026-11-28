"""The camera methods a scene's code replaced.

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
    "engine_own",
    "replaced_methods",
]

from dataclasses import dataclass
from typing import Any, Literal

#: The attribute that marks a callable the engine itself puts on a camera
#: instance while it draws (live drawing's capture wrappers).
_ENGINE_OWN = "__arachne_engine_own__"


def engine_own(function: Any) -> Any:
    """``function``, marked as one the engine puts on a camera instance, so
    :func:`replaced_methods` does not count it as the scene's.
    """
    setattr(function, _ENGINE_OWN, True)
    return function


@dataclass(frozen=True)
class Replaced:
    """One camera method the scene's code replaced."""

    #: The method's name.
    name: str
    #: ``"instance"``: the camera instance holds a callable of its own under
    #: the name. ``"class"``: an engine camera class holds something other
    #: than what it held when the engine was imported.
    form: Literal["instance", "class"]
    #: The class whose method is replaced: the engine class patched, or the
    #: camera's own class for an instance callable.
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
    first (by class, then name), then instance callables (by name). Empty for
    a camera as the engine made it, and for anything that is no engine
    camera.
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
    return tuple(found)


#: The engine's display method for each kind ``Camera.__init__`` puts in
#: ``display_funcs`` (the ``Mobject`` entry is its do-nothing).
_DISPLAY_METHODS = {
    "VMobject": "display_multiple_vectorized_mobjects",
    "PMobject": "display_multiple_point_cloud_mobjects",
    "AbstractImageMobject": "display_multiple_image_mobjects",
}


def display_funcs_rewired(camera: Any) -> bool:
    """Whether ``camera.display_funcs`` holds anything other than what
    ``Camera.__init__`` puts there: for each kind it draws, the camera's own
    display method as the engine defined it, and for ``Mobject`` the
    engine's do-nothing, with no kind added or removed. A wrapper the engine
    puts there while it draws (:func:`engine_own`, the original under
    ``__wrapped__``) stands for what it wraps.
    """
    from arachne.camera.camera import Camera

    from .frozen import class_members, engine_class, engine_code

    if not isinstance(camera, Camera):
        return False
    funcs = vars(camera).get("display_funcs")
    if not isinstance(funcs, dict):
        return funcs is not None
    names = {kind.__name__ for kind in funcs}
    if len(names) != len(funcs) or names != {*_DISPLAY_METHODS, "Mobject"}:
        return True
    for kind, entry in funcs.items():
        while getattr(entry, _ENGINE_OWN, False) and hasattr(entry, "__wrapped__"):
            entry = entry.__wrapped__
        if kind.__name__ == "Mobject":
            if not engine_code(entry):
                return True
            continue
        name = _DISPLAY_METHODS[kind.__name__]
        engine = next(
            (
                class_members(klass)[name]
                for klass in type(camera).__mro__
                if engine_class(klass) and name in class_members(klass)
            ),
            None,
        )
        bound_to = getattr(entry, "__self__", None)
        if bound_to is not camera or getattr(entry, "__func__", None) is not engine:
            return True
    return False
