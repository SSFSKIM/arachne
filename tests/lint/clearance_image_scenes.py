"""Scenes that ask ``apart`` about pictures: each kind of image object set
to the same corners, against a square it touches and one it stands clear of.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    Dot,
    ImageMobject,
    Scene,
    Sprite,
    Square,
    Still,
    VideoMobject,
)
from arachne.lint.bridge import apart

#: The corners every picture is set to (CE's image order).
CORNERS = np.array([[-3.0, 1.0, 0], [-1.0, 1.0, 0], [-3.0, -1.0, 0], [-1.0, -1.0, 0]])


def _pictures():
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    made = {
        "still": Still("fixture:swatch"),
        "clip": VideoMobject("fixture:numbered-clip"),
        "sprite": Sprite("fixture:stepper"),
        "image": ImageMobject(pixels),
    }
    for picture in made.values():
        picture.points = CORNERS.copy()
    return made


class ApartFromPictures(Scene):
    """Each picture asked apart from a square on its edge and one away from
    it, as subject and as target.
    """

    def construct(self):
        touching = Square(side_length=1).move_to([-1.0, 0.0, 0])
        away = Square(side_length=1).move_to([2.0, 0.0, 0])
        self.found = {}
        for name, picture in _pictures().items():
            self.found[name] = {
                "touching": apart(picture, touching),
                "away": apart(picture, away),
                "touching_target": apart(touching, picture),
                "away_target": apart(away, picture),
            }
        self.wait(0.1)


class ApartFromAPictureFill(Scene):
    """A dot in the hollow of an outline: apart from it as a path, then as an
    outline that declares its fill a picture.
    """

    def construct(self):
        dot = Dot()
        plain = Square(side_length=3, fill_opacity=0)
        declared = Square(side_length=3, fill_opacity=0)
        declared.fill_is_picture = True
        self.found = {
            "plain": apart(dot, plain),
            "declared": apart(dot, declared),
        }
        self.wait(0.1)
