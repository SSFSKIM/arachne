"""What ``from arachne import *`` binds: the engine's public names, and of its
modules only those it bound before the engine grew subpackages of its own.

A scene star-imports the engine, defines a function, then star-imports a
helper of its own that star-imports the engine too and has no ``__all__``
(a style module, say). Every module the engine's star binds comes back over
the scene's names with that second import, so a subpackage named like a
scene's function (``quantities``) would turn a call of it into a call of a
module. The subpackages are reached as ``arachne.<name>`` or
``from arachne import <name>`` instead.
"""

from __future__ import annotations

import sys
import types

import pytest

import arachne
import arachne.simulation

#: The module objects ``from arachne import *`` bound before the engine had
#: subpackages of its own: what it binds in a fresh interpreter at the commit
#: before them, and ``cli``, which it bound wherever the command line had been
#: imported first (a render) and which Manim CE's star binds. Scenes written
#: for CE may name them, so they stay. A new subpackage stays out of this list:
#: a scene reaches it by its path.
STAR_MODULES = [
    "AS2700",
    "BS381",
    "DVIPSNAMES",
    "SVGNAMES",
    "X11",
    "XKCD",
    "animation",
    "camera",
    "cli",
    "color",
    "constants",
    "core",
    "data_structures",
    "frame_ir",
    "manager",
    "manim_colors",
    "mobject",
    "np",
    "plugins",
    "rate_functions",
    "renderer",
    "scene",
    "timeline",
    "typing",
    "unit",
    "utils",
]


def _star() -> dict[str, object]:
    bound: dict[str, object] = {}
    exec("from arachne import *", bound)
    return bound


#: The modules the engine's star bound once its subpackages came: a scene of
#: the engine may well name a function of its own like one of them.
HIDDEN = [
    "animations",
    "assets",
    "compose",
    "effects",
    "layers",
    "lint",
    "quantities",
    "routing",
    "simulation",
    "window",
]


@pytest.mark.parametrize("name", HIDDEN)
def test_a_helper_s_star_import_leaves_a_scene_s_own_function(
    name, tmp_path, monkeypatch
):
    """A scene's helper star-imports the engine and has no ``__all__``: star-
    importing it after the scene defines a function keeps the function.
    """
    (tmp_path / "star_helper_style.py").write_text("from arachne import *\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    scene: dict[str, object] = {}
    try:
        exec(
            "from arachne import *\n"
            f"def {name}(scene):\n"
            "    return 'the scene s own'\n"
            "from star_helper_style import *\n"
            f"answer = {name}(None)\n",
            scene,
        )
    finally:
        sys.modules.pop("star_helper_style", None)
    assert scene["answer"] == "the scene s own"


def test_the_star_binds_only_the_modules_it_bound_before_the_subpackages():
    star = _star()
    modules = sorted(
        name for name, value in star.items() if isinstance(value, types.ModuleType)
    )
    assert modules == sorted(STAR_MODULES)


def test_the_star_binds_every_public_name_but_the_new_modules():
    star = _star()
    hidden = {
        name
        for name, value in vars(arachne).items()
        if not name.startswith("_")
        and isinstance(value, types.ModuleType)
        and name not in STAR_MODULES
    }
    assert set(HIDDEN) <= hidden
    public = {name for name in vars(arachne) if not name.startswith("_")}
    assert set(star) - {"__builtins__"} == public - hidden


def test_the_subpackages_are_reached_by_their_paths():
    from arachne import assets, compose, effects, quantities, routing, simulation

    assert arachne.simulation is simulation
    for module in (assets, compose, effects, quantities, routing, simulation):
        assert isinstance(module, types.ModuleType)
        assert sys.modules[module.__name__] is module
    assert callable(arachne.quantities.register)


@pytest.mark.parametrize("module", ["assets", "effects", "simulation"])
def test_every_name_a_star_imported_subpackage_exports_is_in_the_star(module):
    """The package star-imports these three: their classes and functions stay."""
    star = _star()
    package = getattr(arachne, module)
    exported = getattr(package, "__all__", None) or [
        name for name in vars(package) if not name.startswith("_")
    ]
    missing = [
        name
        for name in exported
        if not isinstance(getattr(package, name), types.ModuleType) and name not in star
    ]
    assert missing == []


def test_the_simulation_s_classes_are_in_the_star():
    star = _star()
    for name in ("World", "WorldState", "Body", "BodyReadout", "BallShape", "Gravity"):
        assert star[name] is getattr(arachne.simulation, name)
    assert callable(star["World"].readout)
