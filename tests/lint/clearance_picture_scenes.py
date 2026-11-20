"""Scenes that ask ``apart`` about pictures as the sweep's parts read them:
each kind of picture (a still, a clip, a sprite, a rig's raster member)
set to the same corners, faded until no pixel shows, and in a group with
an unfilled leaf.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    Circle,
    Dot,
    Group,
    Scene,
    Sprite,
    Square,
    Still,
    Text,
    VideoMobject,
)
from arachne.assets import RasterPart, load_asset
from arachne.lint.bridge import apart

#: The corners every picture is set to (CE's image order).
CORNERS = np.array([[-3.0, 1.0, 0], [-1.0, 1.0, 0], [-3.0, -1.0, 0], [-1.0, -1.0, 0]])
#: How far apart a pair is held; and a word from a picture's edge.
PAD = 0.1
WORD_PAD = 0.3


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


class RefusedStill(Still):
    """A still whose pixels the engine refuses to read, as it refuses a
    frame that decodes otherwise.
    """

    def get_pixel_array(self):
        from arachne.lint.raised import RaisedFinding

        raise RaisedFinding("media.inexact", "the frame decodes otherwise")


class ApartFromARefusedStill(Scene):
    """A dot inside a still whose pixels are refused: present, so in
    contact, and asked again with nothing raised.
    """

    def construct(self):
        still = RefusedStill("fixture:swatch")
        still.points = CORNERS.copy()
        dot = Dot(radius=0.05).move_to([-2.0, 0.0, 0])
        self.found = {
            "subject": apart(still, dot, pad=PAD),
            "target": apart(dot, still, pad=PAD),
        }
        self.wait(0.1)


def _word() -> Text:
    """A word whose first letter stands inside the pictures' box and whose
    last stands outside it, its right edge between them, each letter
    between :data:`~arachne.lint.clearance.TOUCH` and :data:`WORD_PAD` from
    it.
    """
    word = Text("A    B", font_size=20)
    first, last = word[0], word[-1]
    middle = (first.get_right()[0] + last.get_left()[0]) / 2
    return word.shift([-1.0 - middle, -word.get_center()[1], 0])


class ApartFromAWordAcrossAPicture(Scene):
    """A still and a raster member at one place, each in a group with a
    closed outline inside it, asked apart from a word whose letters stand
    either side of the picture's edge.
    """

    def construct(self):
        self.found = {}
        for name in ("still", "member"):
            picture = pictures()[name]
            ring = Circle(radius=0.2).move_to([-2.0, 0.5, 0])
            word = _word()
            self.found[name] = {
                "gap": float(word[-1].get_left()[0] + 1.0),
                "found": apart(Group(picture, ring), word, pad=WORD_PAD),
            }
        self.wait(0.1)


class SweptPictureFill(Scene):
    """An outline that only declares its fill a picture, with a word over
    it: swept as an image at its box.
    """

    def construct(self):
        marker = Square(side_length=2).move_to([-2.0, 0.0, 0]).rotate(0.3)
        marker.fill_is_picture = True
        self.add(marker, Text("over", font_size=20).move_to([-2.0, 0.0, 0]))
        self.wait(0.25)
