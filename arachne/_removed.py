"""Names a scene written for Manim may use that Arachne doesn't have.

:data:`REMOVED` lists what Manim CE 0.20.1 has that Arachne deliberately does
not, with the deliberate break that removed it and what to use instead;
``tests/test_ce_surface.py`` checks each entry against CE's surface, so the
two can't drift. :data:`RENAMED` lists names from earlier Manim (CE before
its renames, and 3Blue1Brown's manim) with what Arachne has instead; the
list follows ManimGX's, credited in ``NOTICE.md``. :mod:`arachne.errors`
reads both to say what a scene should write.

Removed configuration options are not here: setting one raises with its
replacement already (``arachne._config.utils.REMOVED_OPTIONS``).

:func:`no_setter` is the message of a call CE accepted and Arachne refuses:
a ``set_<x>`` that names no method, for an ``x`` the mobject lacks.
"""

from __future__ import annotations

__all__ = [
    "OPENGL",
    "OPENGL_PREFIX",
    "REMOVED",
    "RENAMED",
    "Break",
    "Removal",
    "Renamed",
    "no_setter",
]

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Break:
    """A deliberate break from Manim CE: its number, the commit that made it,
    and what a hint says of a name it removed.
    """

    number: int
    commit: str
    removed_with: str


@dataclass(frozen=True)
class Removal:
    """A CE name the break ``by`` removed, and what to use instead, if
    anything does the same.
    """

    by: Break
    instead: str | None = None


@dataclass(frozen=True)
class Renamed:
    """What replaced an old name, as a hint says it, and the Arachne names
    (attribute paths from ``arachne``) that text uses.
    """

    text: str
    uses: tuple[str, ...]


#: Deliberate break 1: the OpenGL renderer and its mobject hierarchy removed.
OPENGL = Break(
    1, "7de40353", "was removed with the OpenGL renderer (Cairo is the renderer)"
)

#: Every name of CE's OpenGL hierarchy that starts with this was removed with
#: it; the Cairo class without the prefix, where Arachne has one, replaces it.
OPENGL_PREFIX = "OpenGL"

#: What CE 0.20.1 has that Arachne deliberately does not: a module (with
#: every module under it), ``module:name``, or ``*:name`` for a name
#: ``from manim import *`` binds. Paths are CE's (``manim.<path>``); the same
#: path under ``arachne`` is what a scene asks for.
REMOVED: dict[str, Removal] = {
    "manim.mobject.opengl": Removal(
        OPENGL, "the mobjects `from arachne import *` binds"
    ),
    "manim.opengl": Removal(OPENGL, "the mobjects `from arachne import *` binds"),
    "manim.renderer.opengl_renderer": Removal(OPENGL),
    "manim.renderer.opengl_renderer_window": Removal(OPENGL),
    "manim.renderer.shader": Removal(OPENGL),
    "manim.renderer.shader_wrapper": Removal(OPENGL),
    "manim.renderer.vectorized_mobject_rendering": Removal(OPENGL),
    "manim.utils.opengl": Removal(OPENGL),
    # The renderer option's values.
    "manim.constants:RendererType": Removal(OPENGL),
    # The OpenGL window's rerun on a changed file.
    "manim.utils.exceptions:RerunSceneException": Removal(OPENGL),
    # It wraps the OpenGL renderer's plays alone.
    "manim.utils.caching:handle_caching_play": Removal(OPENGL),
    # A disc of dots with a radius and a density.
    "*:DotCloud": Removal(OPENGL, "PointCloudDot"),
    # One round dot at a point.
    "*:TrueDot": Removal(OPENGL, "Dot"),
    "*:OpenGLPGroup": Removal(OPENGL, "PGroup"),
    # A point cloud of one point at a location.
    "*:OpenGLPMPoint": Removal(OPENGL, "Point"),
    "*:OpenGLPMobject": Removal(OPENGL, "PMobject"),
    "*:RendererType": Removal(OPENGL),
    "*:opengl": Removal(OPENGL),
}

#: Names from earlier Manim, with what Arachne has instead.
RENAMED: dict[str, Renamed] = {
    "ShowCreation": Renamed("Create", ("Create",)),
    "ShowCreationThenDestruction": Renamed("ShowPassingFlash", ("ShowPassingFlash",)),
    "ShowCreationThenFadeOut": Renamed("ShowPassingFlash", ("ShowPassingFlash",)),
    "CircleIndicate": Renamed("Circumscribe", ("Circumscribe",)),
    "WiggleOutThenIn": Renamed("Wiggle", ("Wiggle",)),
    "TextMobject": Renamed("Text, or Tex for text with LaTeX", ("Text", "Tex")),
    "TexText": Renamed("Tex", ("Tex",)),
    "OldTexText": Renamed("Tex", ("Tex",)),
    "TexMobject": Renamed("MathTex", ("MathTex",)),
    "OldTex": Renamed("MathTex", ("MathTex",)),
    "FadeInFrom": Renamed("FadeIn(mobject, shift=direction)", ("FadeIn",)),
    "FadeInFromDown": Renamed("FadeIn(mobject, shift=UP)", ("FadeIn", "UP")),
    "FadeOutAndShift": Renamed("FadeOut(mobject, shift=direction)", ("FadeOut",)),
    "FadeInFromLarge": Renamed("FadeIn(mobject, scale=factor)", ("FadeIn",)),
    "GraphScene": Renamed(
        "a Scene that draws Axes and plots on them: axes.plot(function)",
        ("Scene", "Axes", "Axes.plot"),
    ),
    "ParametricSurface": Renamed("Surface", ("Surface",)),
    "ContinualAnimation": Renamed(
        "an updater: mobject.add_updater(function)", ("Mobject.add_updater",)
    ),
}


def no_setter(owner: str, method: str, nearest: Sequence[str]) -> str:
    """The error for ``method``, a ``set_<x>`` that a mobject of class
    ``owner`` has no method of and no attribute ``x`` for; ``nearest`` are
    the setters nearest it that work.

    Deliberate break 9 (a misspelt synthesised setter raises), approved
    2026-11-07: CE's compatibility setter set a new attribute ``x``, so a
    typo changed nothing on screen and raised nothing.
    """
    name = method[len("set_") :]
    message = (
        f"{owner} has no method {method!r}, and no attribute {name!r} for it to set"
    )
    message += f": did you mean {' or '.join(nearest)}?" if nearest else "."
    # No closing stop: Python's traceback appends its own suggestion.
    return (
        f"{message} To give the mobject an attribute of its own, assign it: "
        f"mob.{name} = value"
    )
