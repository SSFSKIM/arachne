"""Every public name of Manim CE 0.20.1 is where a scene written for it looks:
``manim.<path>.<name>`` at ``arachne.<path>.<name>``, and every name
``from manim import *`` binds in ``from arachne import *``, but for the
deliberate breaks listed here, each of which must really be gone.

CE is never imported: its names are the committed list
``tests/data/ce_0_20_1_surface.json``, which ``scripts/ce_surface.py`` writes
from PyPI's ``manim`` 0.20.1.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

SURFACE = json.loads(
    (Path(__file__).parent / "data" / "ce_0_20_1_surface.json").read_text()
)
MODULES: dict[str, list[str]] = SURFACE["modules"]
STAR: list[str] = SURFACE["star"]

#: Deliberate break 1: the OpenGL renderer and its mobject hierarchy removed
#: (7de40353).
OPENGL = (1, "the OpenGL renderer and its mobject hierarchy removed, 7de40353")

#: What CE 0.20.1 has that Arachne deliberately does not, with the break
#: that removed it: a module (with every module under it), ``module:name``,
#: or ``*:name`` for a name ``from manim import *`` binds.
APPROVED_BREAKS: dict[str, tuple[int, str]] = {
    "manim.mobject.opengl": OPENGL,
    "manim.opengl": OPENGL,
    "manim.renderer.opengl_renderer": OPENGL,
    "manim.renderer.opengl_renderer_window": OPENGL,
    "manim.renderer.shader": OPENGL,
    "manim.renderer.shader_wrapper": OPENGL,
    "manim.renderer.vectorized_mobject_rendering": OPENGL,
    "manim.utils.opengl": OPENGL,
    # The renderer option's values.
    "manim.constants:RendererType": OPENGL,
    # The OpenGL window's rerun on a changed file.
    "manim.utils.exceptions:RerunSceneException": OPENGL,
    # It wraps the OpenGL renderer's plays alone.
    "manim.utils.caching:handle_caching_play": OPENGL,
    "*:DotCloud": OPENGL,
    "*:TrueDot": OPENGL,
    "*:OpenGLPGroup": OPENGL,
    "*:OpenGLPMPoint": OPENGL,
    "*:OpenGLPMobject": OPENGL,
    "*:RendererType": OPENGL,
    "*:opengl": OPENGL,
}


def _arachne(path: str) -> str:
    return "arachne" + path[len("manim") :]


def _module_broken(path: str) -> bool:
    return any(
        path == broken or path.startswith(broken + ".")
        for broken in APPROVED_BREAKS
        if ":" not in broken
    )


@pytest.mark.parametrize(
    "path",
    [path for path in MODULES if not _module_broken(path)],
    ids=_arachne,
)
def test_every_ce_name_is_at_its_path(path):
    module = importlib.import_module(_arachne(path))
    missing = [
        f"{_arachne(path)}.{name}"
        for name in MODULES[path]
        if f"{path}:{name}" not in APPROVED_BREAKS and not hasattr(module, name)
    ]
    assert missing == []


def test_every_ce_star_name_is_in_arachne_s_star():
    star: dict[str, object] = {}
    exec("from arachne import *", star)
    missing = [
        name for name in STAR if f"*:{name}" not in APPROVED_BREAKS and name not in star
    ]
    assert missing == []


@pytest.mark.parametrize("broken", APPROVED_BREAKS)
def test_every_approved_break_is_of_a_ce_name_arachne_lacks(broken):
    where, _, name = broken.rpartition(":")
    if not where:
        assert broken in MODULES
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(_arachne(broken))
    elif where == "*":
        assert name in STAR
        star: dict[str, object] = {}
        exec("from arachne import *", star)
        assert name not in star
    else:
        assert name in MODULES[where]
        assert not hasattr(importlib.import_module(_arachne(where)), name)
