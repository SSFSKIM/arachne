"""A rig: an asset's parts as named vector objects in a scene.

``Rig(ref)`` loads the asset (:func:`~.reader.load_asset`) and holds one
group per part and per slot, in drawing order (the manifest's ``order``,
then its order of entries), each the part's shapes as an ordinary
:class:`~arachne.mobject.types.vectorized_mobject.VMobject`: crisp at any
zoom, styled and transformed as any other. A slot's group holds the drawing
it shows, its default at rest. The canvas is placed by the manifest's
``size``: ``scene_units_per_unit`` scene units to a canvas unit, the canvas
point ``origin`` at the scene's origin, y up. A rig is not centred or
rescaled as an :class:`~arachne.mobject.svg.svg_mobject.SVGMobject` is.
"""

from __future__ import annotations

__all__ = ["Rig"]

import difflib
import os
from typing import Any

import numpy as np

from ..mobject.types.vectorized_mobject import VGroup, VMobject
from .manifest import Size, refuse
from .reader import Asset, load_asset


class Rig(VGroup):
    """The asset ``asset`` (an :class:`~.reader.Asset`, or a reference
    :func:`~.reader.load_asset` takes) at rest: its parts, and its slots'
    default drawings, as named groups in drawing order.
    ``vmobject_kwargs`` go to the group, as a ``VGroup``'s do.
    """

    def __init__(
        self, asset: Asset | str | os.PathLike[str], **vmobject_kwargs: Any
    ) -> None:
        asset = load_asset(asset)
        super().__init__(**vmobject_kwargs)
        self.asset = asset
        manifest = asset.manifest
        entries: list[tuple[int, int, str, tuple[VMobject, ...]]] = [
            (part.order, i, part.name, asset.rest(part.name))
            for i, part in enumerate(manifest.parts)
        ]
        entries += [
            (
                slot.order,
                len(manifest.parts) + i,
                slot.name,
                asset.drawing(slot.name, slot.default),
            )
            for i, slot in enumerate(manifest.slots)
        ]
        members: list[tuple[str, VGroup]] = []
        for _, _, name, shapes in sorted(entries, key=lambda entry: entry[:2]):
            for shape in shapes:
                _place(shape, manifest.size)
            members.append((name, VGroup(*shapes, name=name)))
        self._members = tuple(members)
        self.add(*(group for _, group in members))

    def part(self, name: str) -> VMobject:
        """The group of the part or slot ``name``; ``rig.name_unknown`` for
        a name the asset's manifest does not have.
        """
        for found, group in self._members:
            if found == name:
                return group
        names = [found for found, _ in self._members]
        nearest = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
        hint = f", and the nearest is {' or '.join(nearest)}" if nearest else ""
        refuse(
            "rig.name_unknown",
            f"the asset {self.asset.name} has no part or slot {name!r}: it has"
            f" {', '.join(names)}{hint}",
            asset=self.asset.name,
            kind="part",
            name=name,
            nearest=nearest,
        )


def _place(shape: VMobject, size: Size) -> None:
    """Move ``shape``, in canvas coordinates, to the scene, as ``size``
    places the canvas.
    """
    scale = size.scene_units_per_unit
    origin_x, origin_y = size.origin
    for member in shape.family_members_with_points():
        points = np.array(member.points)
        points[:, 0] = (points[:, 0] - origin_x) * scale
        points[:, 1] = (origin_y - points[:, 1]) * scale
        member.points = points
