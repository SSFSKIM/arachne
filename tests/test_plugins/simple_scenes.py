from __future__ import annotations

from arachne import *


class SquareToCircle(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Transform(square, circle))
