"""Every public name of Manim CE 0.20.1 is where a scene written for it looks:
``manim.<path>.<name>`` at ``arachne.<path>.<name>``, and every name
``from manim import *`` binds in ``from arachne import *``, but for the
deliberate breaks ``arachne._removed`` lists, each of which must really be
gone.

CE is never imported: its names are the committed list
``tests/data/ce_0_20_1_surface.json``, which ``scripts/ce_surface.py`` writes
from PyPI's ``manim`` 0.20.1.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import arachne
from arachne._removed import OPENGL_PREFIX, REMOVED

SURFACE = json.loads(
    (Path(__file__).parent / "data" / "ce_0_20_1_surface.json").read_text()
)
MODULES: dict[str, list[str]] = SURFACE["modules"]
STAR: list[str] = SURFACE["star"]


def _arachne(path: str) -> str:
    return "arachne" + path[len("manim") :]


def _module_broken(path: str) -> bool:
    return any(
        path == broken or path.startswith(broken + ".")
        for broken in REMOVED
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
        if f"{path}:{name}" not in REMOVED and not hasattr(module, name)
    ]
    assert missing == []


def test_every_ce_star_name_is_in_arachne_s_star():
    star: dict[str, object] = {}
    exec("from arachne import *", star)
    missing = [name for name in STAR if f"*:{name}" not in REMOVED and name not in star]
    assert missing == []


@pytest.mark.parametrize("broken", REMOVED)
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


def test_every_replacement_a_removal_names_is_arachne_s():
    for removal in REMOVED.values():
        if removal.instead is not None and "`" not in removal.instead:
            assert hasattr(arachne, removal.instead)


def test_every_ce_name_of_the_opengl_prefix_was_removed_with_its_hierarchy():
    """The errors read a name with the prefix as removed with the OpenGL
    renderer: every CE name that has it is in a removed module or is a
    removed name, and Arachne has none.
    """
    for path, names in MODULES.items():
        for name in names:
            if name.startswith(OPENGL_PREFIX):
                assert _module_broken(path) or f"*:{name}" in REMOVED, (path, name)
    assert [name for name in dir(arachne) if name.startswith(OPENGL_PREFIX)] == []
