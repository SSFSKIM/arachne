"""Media made outside the engine, hosted by it: stills, clips and sprites.

A media asset is a directory holding ``manifest.json`` (of the kind
``still``, ``clip`` or ``sprite``) and its frames, one PNG file each, every
frame's digest listed (:mod:`arachne.assets.manifest`). :class:`Still`,
:class:`VideoMobject` and :class:`Sprite` place one in a scene by naming it
(:mod:`.objects`); its frames are read exactly, and checked each time one
is decoded (:mod:`.frames`). Importing this package reads no asset and
builds no object.
"""

from __future__ import annotations

__all__ = ["Sprite", "Still", "VideoMobject"]

from .objects import Sprite, Still, VideoMobject
