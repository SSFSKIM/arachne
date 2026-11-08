"""Scenes whose states the content key tests key: one of every mobject class
the engine exports, as a scene makes it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arachne import *

SVG = Path(__file__).parents[1] / "test_graphical_units/img_svg_resources/penrose.svg"


def _dots() -> VGroup:
    return VGroup(Dot(LEFT), Dot(RIGHT), Dot(UP))


def _two_squares() -> tuple[Square, Square]:
    return Square(), Square().shift(RIGHT * 0.5)


def _camera_display() -> ImageMobjectFromCamera:
    camera = MovingCamera(pixel_width=32, pixel_height=18)
    return ImageMobjectFromCamera(camera)


def _wave(t: float) -> np.ndarray:
    return np.array([t, np.sin(t), 0.0])


def _field(point: np.ndarray) -> np.ndarray:
    return np.array([-point[1], point[0], 0.0])


#: A factory for every mobject class the engine exports, by class name.
#: The classes taking no argument are made with their defaults; the others
#: with the smallest arguments that make one.
MADE_WITH_ARGUMENTS = {
    "Angle": lambda: Angle(Line(ORIGIN, RIGHT), Line(ORIGIN, UP)),
    "AnimatedBoundary": lambda: AnimatedBoundary(Square()),
    "ArcBetweenPoints": lambda: ArcBetweenPoints(LEFT, RIGHT),
    "ArrowTip": lambda: Arrow().tip,
    "ArrowVectorField": lambda: ArrowVectorField(
        _field, x_range=[-1, 1, 1], y_range=[-1, 1, 1]
    ),
    "BarChart": lambda: BarChart([1, 2, 3]),
    "Brace": lambda: Brace(Square()),
    "BraceBetweenPoints": lambda: BraceBetweenPoints(LEFT, RIGHT),
    "BraceLabel": lambda: BraceLabel(Square(), "a"),
    "BraceText": lambda: BraceText(Square(), "a"),
    "Code": lambda: Code(code_string="x = 1\n", language="python"),
    "ConvexHull": lambda: ConvexHull(LEFT, RIGHT, UP),
    "ConvexHull3D": lambda: ConvexHull3D(LEFT, RIGHT, UP, OUT),
    "CubicBezier": lambda: CubicBezier(LEFT, UP, UP + RIGHT, RIGHT),
    "CurvedArrow": lambda: CurvedArrow(LEFT, RIGHT),
    "CurvedDoubleArrow": lambda: CurvedDoubleArrow(LEFT, RIGHT),
    "CurvesAsSubmobjects": lambda: CurvesAsSubmobjects(Circle()),
    "Cutout": lambda: Cutout(Square(side_length=2), Square(side_length=1)),
    "DashedVMobject": lambda: DashedVMobject(Circle()),
    "DecimalMatrix": lambda: DecimalMatrix([[1.5, 2.5]]),
    "DecimalTable": lambda: DecimalTable([[1.5, 2.5]]),
    # A circular layout: a spring layout's is drawn from its own generator.
    "DiGraph": lambda: DiGraph([1, 2], [(1, 2)], layout="circular"),
    "Difference": lambda: Difference(*_two_squares()),
    "Exclusion": lambda: Exclusion(*_two_squares()),
    "FunctionGraph": lambda: FunctionGraph(np.sin, x_range=[-1, 1]),
    "Graph": lambda: Graph([1, 2, 3], [(1, 2), (2, 3)], layout="circular"),
    "ImageMobject": lambda: ImageMobject(np.zeros((4, 4, 3), dtype=np.uint8)),
    "ImageMobjectFromCamera": _camera_display,
    "ImplicitFunction": lambda: ImplicitFunction(
        lambda x, y: x * x + y * y - 1, x_range=[-2, 2], y_range=[-2, 2]
    ),
    "IntegerMatrix": lambda: IntegerMatrix([[1, 2]]),
    "IntegerTable": lambda: IntegerTable([[1, 2]]),
    "Intersection": lambda: Intersection(*_two_squares()),
    "Label": lambda: Label("a"),
    "LabeledArrow": lambda: LabeledArrow("a"),
    "LabeledDot": lambda: LabeledDot("a"),
    "LabeledLine": lambda: LabeledLine("a"),
    "LabeledPolygram": lambda: LabeledPolygram(
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], label="a"
    ),
    "MarkupText": lambda: MarkupText("<b>a</b>"),
    "MathTable": lambda: MathTable([["x", "y"]]),
    "MathTypst": lambda: MathTypst("x^2"),
    "Matrix": lambda: Matrix([[1, 2]]),
    "MobjectMatrix": lambda: MobjectMatrix([[Dot(), Square()]]),
    "MobjectTable": lambda: MobjectTable([[Dot(), Square()]]),
    "ParametricFunction": lambda: ParametricFunction(_wave, t_range=[0, 1]),
    "Polygon": lambda: Polygon(LEFT, RIGHT, UP),
    "Polyhedron": lambda: Polyhedron(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    ),
    "RegularPolygram": lambda: RegularPolygram(5),
    "RightAngle": lambda: RightAngle(Line(ORIGIN, RIGHT), Line(ORIGIN, UP)),
    "SVGMobject": lambda: SVGMobject(SVG),
    "SingleStringMathTex": lambda: SingleStringMathTex("x"),
    "StreamLines": lambda: StreamLines(
        _field, x_range=[-1, 1, 1], y_range=[-1, 1, 1], virtual_time=0.5
    ),
    "Surface": lambda: Surface(lambda u, v: np.array([u, v, u * v]), resolution=(2, 2)),
    "Table": lambda: Table([["a", "b"]]),
    "TangentLine": lambda: TangentLine(Circle(), 0.25),
    "TangentialArc": lambda: TangentialArc(Line(LEFT, ORIGIN), Line(ORIGIN, UP), 0.5),
    "Text": lambda: Text("a"),
    "TracedPath": lambda: TracedPath(lambda: ORIGIN),
    "Typst": lambda: Typst("a"),
    "Underline": lambda: Underline(Square()),
    "Union": lambda: Union(*_two_squares()),
    "VMobjectFromSVGPath": lambda: SVGMobject(SVG).family_members_with_points()[0],
    "Variable": lambda: Variable(1.5, "x"),
    "VectorField": lambda: VectorField(_field),
}


PICTURE = (
    Path(__file__).parents[1]
    / "test_graphical_units/img_svg_resources/tree_img_640x351.png"
)


def _painted_with_a_picture() -> Square:
    from PIL import Image

    return Square().color_using_background_image(Image.open(PICTURE))


#: Mobjects made with options that give them fields their defaults do not:
#: numbers and tips on axes, a graph's labels, a picture from a file, a
#: picture painted onto a shape, a template of one's own.
MADE_WITH_OPTIONS = {
    "axes with numbers and tips": lambda: Axes(
        x_range=[-2, 2, 1],
        y_range=[-2, 2, 1],
        tips=True,
        axis_config={"include_numbers": True},
    ),
    "a graph with labels": lambda: Graph(
        [1, 2], [(1, 2)], labels=True, layout="circular"
    ),
    "a picture from a file": lambda: ImageMobject(PICTURE),
    "a shape painted with a picture": _painted_with_a_picture,
    "tex with a template of its own": lambda: MathTex(
        r"\mathbb{R}", tex_template=_own_template()
    ),
}


def _own_template() -> TexTemplate:
    template = TexTemplate(description="its own")
    template.add_to_preamble(r"\usepackage{bm}")
    return template


def public_mobject_classes() -> list[type[Mobject]]:
    """Every mobject class the engine exports, by name."""
    import arachne

    return sorted(
        (
            value
            for value in vars(arachne).values()
            if isinstance(value, type) and issubclass(value, Mobject)
        ),
        key=lambda cls: cls.__qualname__,
    )


def make(cls: type[Mobject]) -> Mobject:
    """One mobject of ``cls``, as a scene makes it."""
    factory = MADE_WITH_ARGUMENTS.get(cls.__qualname__, cls)
    return factory()


#: The classes whose points differ from one process to the next however the
#: scene is seeded: their hull is found by iterating sets of points hashed
#: by their coordinates' bytes, an order that follows each process's string
#: hash seed (``tech-debt-tracker.md``).
UNSTABLE_ACROSS_PROCESSES = frozenset({"ConvexHull", "ConvexHull3D"})


class EveryMobject(Scene):
    """One mobject of every class the engine exports, in the scene, and the
    mobjects made with options (:data:`MADE_WITH_OPTIONS`).
    """

    left_out: frozenset[str] = frozenset()

    def construct(self):
        self.made = [
            make(cls)
            for cls in public_mobject_classes()
            if cls.__qualname__ not in self.left_out
        ]
        self.made.extend(factory() for factory in MADE_WITH_OPTIONS.values())
        self.add(*self.made)
        self.wait(0.25)


class EveryMobjectMadeAlike(EveryMobject):
    """One mobject of every class the engine exports whose points are the
    same in every process.
    """

    left_out = UNSTABLE_ACROSS_PROCESSES
