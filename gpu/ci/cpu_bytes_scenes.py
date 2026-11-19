"""The scenes ``cpu_bytes.sh`` draws: each uses nothing of the GPU build, and each exercises one
part of the CPU drawing path. The still image is the public tier's own case, ``test_ImageMobject``.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Circle,
    Create,
    Cube,
    Dot,
    FadeIn,
    MathTex,
    Rotate,
    Scene,
    Square,
    Text,
    ThreeDScene,
    Write,
    ZoomedScene,
)
from tests.test_graphical_units.test_img_and_svg import test_ImageMobject


class ShapesAndText(Scene):
    def construct(self):
        circle = Circle(color="#58C4DD").shift(2 * LEFT)
        square = Square(color="#FC6255", fill_opacity=0.6).shift(2 * RIGHT)
        self.play(Create(circle), FadeIn(square), run_time=1)
        self.play(FadeIn(Text("bytes").shift(2 * DOWN)), run_time=0.5)


class TypesetMathematics(Scene):
    def construct(self):
        self.play(Write(MathTex(r"e^{i\pi} + 1 = 0")), run_time=1.5)


class ThreeDCamera(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * PI / 180, theta=-45 * PI / 180)
        cube = Cube(side_length=2, fill_opacity=0.8)
        self.add(cube)
        self.play(Rotate(cube, PI / 3, axis=UP), run_time=1.5)


class SixtyHertzClock(Scene):
    def construct(self):
        dot = Dot(2 * LEFT)
        dot.add_updater(lambda mobject, dt: mobject.shift(2 * dt * RIGHT))
        self.add(dot)
        self.wait(1.5)


class MultiCameraDisplay(ZoomedScene):
    def construct(self):
        dot = Dot().shift(UP)
        self.add(Square(), dot)
        self.activate_zooming(animate=False)
        self.play(dot.animate.shift(2 * DOWN), run_time=1.5)


class StillImage(Scene):
    def construct(self):
        test_ImageMobject.__wrapped__(self)


SCENES = (
    ShapesAndText,
    TypesetMathematics,
    ThreeDCamera,
    SixtyHertzClock,
    MultiCameraDisplay,
    StillImage,
)

#: The instants each scene's session draws: every scene runs at least this long.
INSTANTS = (0, 0.5, 1)
