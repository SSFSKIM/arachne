"""Write Manim CE's public names, by module, as JSON: the list
``tests/test_ce_surface.py`` holds Arachne to.

Run it with PyPI's ``manim`` of the version the list is for installed and
Arachne not on the path::

    python scripts/ce_surface.py 0.20.1 > tests/data/ce_0_20_1_surface.json

A module's public names are its ``__all__`` where it defines one; else every
name it binds that does not start with an underscore and is not a module,
a type variable or ``TYPE_CHECKING``, leaving out classes, functions and
typing constructs it imports from elsewhere. ``star`` is what
``from manim import *`` binds. A module that fails to import (one needing
an extra CE does not install) is listed under ``unimportable`` with its
error, and has no names.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import sys
import types
import typing
import warnings


def public_names(module: types.ModuleType) -> list[str]:
    if hasattr(module, "__all__"):
        return sorted(set(module.__all__))
    names = []
    for name, value in vars(module).items():
        if name.startswith("_") or name in ("annotations", "TYPE_CHECKING"):
            continue
        if isinstance(value, (types.ModuleType, typing.TypeVar)):
            continue
        defined_here = getattr(value, "__module__", None) == module.__name__
        imported = (
            inspect.isclass(value)
            or inspect.isfunction(value)
            or inspect.isbuiltin(value)
            or getattr(value, "__module__", None) in ("typing", "typing_extensions")
        )
        if defined_here or not imported:
            names.append(name)
    return sorted(names)


def surface(version: str) -> dict[str, object]:
    warnings.simplefilter("ignore")
    package = importlib.import_module("manim")
    if package.__version__ != version:
        raise SystemExit(f"manim {package.__version__} is installed, not {version}")
    paths = ["manim"] + [
        found.name for found in pkgutil.walk_packages(package.__path__, "manim.")
    ]
    modules: dict[str, list[str]] = {}
    unimportable: dict[str, str] = {}
    for path in paths:
        if path.split(".")[-1] == "__main__":
            continue
        try:
            module = importlib.import_module(path)
        except Exception as error:  # noqa: BLE001
            unimportable[path] = f"{type(error).__name__}: {error}"
            continue
        modules[path] = public_names(module)
    star: dict[str, object] = {}
    exec("from manim import *", star)
    return {
        "version": version,
        "star": sorted(name for name in star if name != "__builtins__"),
        "modules": dict(sorted(modules.items())),
        "unimportable": dict(sorted(unimportable.items())),
    }


if __name__ == "__main__":
    json.dump(surface(sys.argv[1]), sys.stdout, indent=1, sort_keys=False)
    sys.stdout.write("\n")
