"""Scenes whose lint the read-count tests count: many leaves under one camera,
held or moving, so that work done per leaf and work done per instant differ.
"""

from __future__ import annotations

from arachne import DEGREES, RIGHT, Square, ThreeDScene, VGroup

#: How many leaves each scene holds.
LEAVES = 8


def squares() -> VGroup:
    return VGroup(
        *(
            Square(side_length=0.4).shift(RIGHT * 0.5 * (i - LEAVES / 2))
            for i in range(LEAVES)
        )
    )


class HeldSquares3D(ThreeDScene):
    """Squares held under a tilted 3D camera."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(squares())
        self.wait(1)


class TurningSquares3D(ThreeDScene):
    """Squares as the 3D camera turns about them."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(squares())
        self.move_camera(theta=45 * DEGREES, run_time=1)
