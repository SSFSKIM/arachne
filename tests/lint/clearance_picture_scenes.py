"""Scenes that ask ``apart`` about pictures as the sweep's parts read them:
each kind of picture (a still, a clip, a sprite, a rig's raster member)
set to the same corners, faded until no pixel shows, and in a group with
an unfilled leaf.
"""

from __future__ import annotations

import numpy as np

from arachne import Circle, Dot, Group, Scene, Sprite, Square, Still, VideoMobject
from arachne.assets import RasterPart, load_asset
from arachne.lint.bridge import apart

#: The corners every picture is set to (CE's image order).
CORNERS = np.array([[-3.0, 1.0, 0], [-1.0, 1.0, 0], [-3.0, -1.0, 0], [-1.0, -1.0, 0]])
#: How far apart a pair is held.
PAD = 0.1


def _member() -> RasterPart:
    """A raster member of the flap rig, its quad at :data:`CORNERS`."""
    asset = load_asset("fixture:hinged-flap")
    part = asset.manifest.parts[0]
    quad = RasterPart(
        part.box,
        asset_digest=asset.digest,
        file=part.file,
        digest=part.digest,
        asset=asset.name,
        directory=asset.directory,
    )
    upper_left, upper_right, lower_left, lower_right = CORNERS
    quad.set_points_as_corners(
        [upper_left, upper_right, lower_right, lower_left, upper_left]
    )
    return quad


def pictures() -> dict:
    made = {
        "still": Still("fixture:swatch"),
        "clip": VideoMobject("fixture:numbered-clip"),
        "sprite": Sprite("fixture:stepper"),
    }
    for picture in made.values():
        picture.points = CORNERS.copy()
    made["member"] = _member()
    return made


def _faded(picture):
    if isinstance(picture, RasterPart):
        return picture.set_fill(opacity=0)
    return picture.set_opacity(0)


class ApartFromPicturesAsDrawn(Scene):
    """Each picture faded out with a dot inside its box, as subject and as
    target; and each in a group with an unfilled circle, against a square
    across its right edge and one about its middle, where a curve through
    its corners would pass.
    """

    def construct(self):
        self.found = {}
        for name, picture in pictures().items():
            dot = Dot(radius=0.05).move_to([-2.0, 0.0, 0])
            edge = Square(side_length=0.4, fill_opacity=1).move_to([-1.0, 0.0, 0])
            middle = Square(side_length=0.2, fill_opacity=1).move_to([-2.0, 0.0, 0])
            shown = {"shows": apart(picture, dot, pad=PAD)}
            group = Group(picture, Circle(radius=0.3).move_to([3.0, 2.0, 0]))
            shown["edge"] = apart(group, edge, pad=PAD)
            shown["middle"] = apart(group, middle, pad=PAD)
            _faded(picture)
            shown["faded"] = apart(picture, dot, pad=PAD)
            shown["faded_target"] = apart(dot, picture, pad=PAD)
            self.found[name] = shown
        self.wait(0.1)
