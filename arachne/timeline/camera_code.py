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

__all__ = ["Replaced", "drawn_cameras", "engine_own", "replaced_methods"]

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
