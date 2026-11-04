"""A helper module a projection scene makes a mobject through: the mobject's
innermost scene-code frame is here, its scene file's line is the call's.
"""

from __future__ import annotations

from arachne import Square


def framed_square() -> Square:
    return Square(side_length=1.2)
