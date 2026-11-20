"""Assets made outside the engine, hosted by it.

An asset is a directory holding ``manifest.json`` (:mod:`.manifest`) and the
files it names. :func:`load_asset` reads one whole, or refuses it with a
finding whose code says why (:mod:`.reader`); :class:`Rig` puts its parts
in a scene as named vector objects (:mod:`.rig`; a part that is a still,
as the quad its picture is drawn at, :mod:`.raster`), moved through the
parameters its maker tested (:mod:`.pose`, :mod:`.motion`), with anchors
that follow points of its parts (:mod:`.anchor`). An asset loads only once
admitted by its geometry (:func:`admit`, :mod:`.admission`, and from a
shell ``python -m arachne.assets admit <directory> --write``). An
illustration can be drawn a declared number of times a second, with line
boil, the same at every frame rate (:func:`set_timing`, :mod:`.timing`). The engine's
own fixture assets, drawn by its own code for its tests, are named
``fixture:<name>``. Importing this package reads no asset and builds no
object.
"""

from __future__ import annotations

__all__ = [
    "AdmissionReport",
    "Anchor",
    "Asset",
    "Boil",
    "MorphParts",
    "Pose",
    "RasterPart",
    "Rig",
    "Timing",
    "admit",
    "load_asset",
    "set_timing",
    "timing_of",
]

from .admission import AdmissionReport, admit
from .anchor import Anchor
from .motion import MorphParts, Pose
from .raster import RasterPart
from .reader import Asset, load_asset
from .rig import Rig
from .timing import Boil, Timing, set_timing, timing_of
