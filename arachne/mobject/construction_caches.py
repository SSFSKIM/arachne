"""The module-level caches that hand out mobjects or the parts they are
built from, kept as one unit.

An image from the image cache is the image parsing builds
(:meth:`.SVGMobject.init_svg_mobject`). The glyph cache still keeps history:
it is keyed by a number's string alone, so a glyph is built as the first
number in the process to need it asked, whatever class or style a later one
asks for (Manim Community Edition's behaviour). The caches also fill each
other: a number fills the glyph cache and the image cache at once, and on a
later glyph hit it fills neither. So an execution builds what an earlier one
built only when it starts from every cache as that one found it. A session's
execution starts from them empty, as a fresh process does; a render executed
again starts from them as its first execution found them.

A cache the engine adds that hands out mobjects or their parts is registered
in :data:`CACHES`, and is then forgotten and restored with the rest.
"""

from __future__ import annotations

from typing import Any

from .svg.svg_mobject import SVG_HASH_TO_MOB_MAP
from .text.numbers import string_to_mob_map

__all__ = [
    "CACHES",
    "construction_caches",
    "forget_construction_caches",
    "restore_construction_caches",
]

#: Every construction cache: parsed images by their file and style, and a
#: number's glyphs by their string. Their entries are never changed (a hit
#: takes a copy), so a shallow copy keeps a cache's state.
CACHES: tuple[dict[Any, Any], ...] = (SVG_HASH_TO_MOB_MAP, string_to_mob_map)


def construction_caches() -> tuple[dict[Any, Any], ...]:
    """Every construction cache as it is now, to restore with
    :func:`restore_construction_caches`.
    """
    return tuple(dict(cache) for cache in CACHES)


def restore_construction_caches(kept: tuple[dict[Any, Any], ...]) -> None:
    """Make every construction cache what :func:`construction_caches`
    returned.
    """
    for cache, entries in zip(CACHES, kept, strict=True):
        cache.clear()
        cache.update(entries)


def forget_construction_caches() -> None:
    """Empty every construction cache, so that an execution builds its
    mobjects as a fresh process does.
    """
    for cache in CACHES:
        cache.clear()
