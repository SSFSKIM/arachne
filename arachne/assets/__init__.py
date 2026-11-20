"""Assets made outside the engine, hosted by it.

An asset is a directory holding ``manifest.json`` (:mod:`.manifest`) and the
files it names. :func:`load_asset` reads one whole, or refuses it with a
finding whose code says why (:mod:`.reader`); :class:`Rig` puts its parts
in a scene as named vector objects (:mod:`.rig`). The engine's own fixture
assets, drawn by its own code for its tests, are named ``fixture:<name>``.
Importing this package reads no asset and builds no object.
"""

from __future__ import annotations

__all__ = ["Asset", "Rig", "load_asset"]

from .reader import Asset, load_asset
from .rig import Rig
