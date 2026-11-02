"""What a state-function callback (route 2) may call.

Every entry is a method resolved to its function object on the built-in class
that defines it, so a call is admitted by what it runs, never by its name: a
subclass that overrides the method, or any type defined outside the engine,
resolves to another function and is not admitted.

- :data:`SETTERS`: the setters a callback may apply to a mobject it updates.
  Each (class, method) pair is listed with a property test
  (``tests/timeline/test_setters.py``): applying it after any sequence of
  earlier calls, from the play's start state, gives the state that applying
  it once to that start state gives. Pairs that fail it, or whose result
  depends on the current geometry (``put_start_and_end_on``, scale to fit),
  are not here, and a callback that calls them makes its play tick.
- Reads (:func:`read_kind`): methods that only read the mobject they are
  called on, returning a value, one of its own members, or a mobject they
  build.
- Calls a callback may make on a mobject it built in the same call
  (:func:`fresh_method`), and the constructors it may call
  (:func:`constructor_kind`); the library functions are :mod:`.library`'s.
"""

from __future__ import annotations

__all__ = [
    "SETTERS",
    "Setter",
    "builtin_class",
    "constructor_kind",
    "engine_classes",
    "fresh_method",
    "method_function",
    "pure_module",
    "read_kind",
    "returned_class",
    "setter_for",
]

import inspect
import types
from dataclasses import dataclass
from functools import cache
from typing import Any


@dataclass(frozen=True)
class Setter:
    """An admitted setter: the class that defines it, its name, and the
    arguments it is admitted with.

    ``params`` is the argument domain its property test covers: the leading
    parameters a call may pass, positionally or by name, every one a value
    (never ``None``, which leaves what it sets as it was). A call passing
    anything else (a mask, ``family=False``, an aligned edge) is outside the
    proof. ``fresh_argument`` says the setter takes a mobject argument that
    must be one the callback built in the same call (``become`` aligns its
    argument's family to its own, so given a scene mobject it would change it).
    """

    owner: str
    name: str
    params: tuple[str, ...]
    fresh_argument: bool = False


_MOBJECT = "arachne.mobject.mobject.Mobject"
_VMOBJECT = "arachne.mobject.types.vectorized_mobject.VMobject"

#: The admitted setters, by the defining class's dotted path and method name.
SETTERS: tuple[Setter, ...] = (
    Setter(_MOBJECT, "move_to", ("point_or_mobject",)),
    Setter(_MOBJECT, "set_x", ("x",)),
    Setter(_MOBJECT, "set_y", ("y",)),
    Setter(_MOBJECT, "set_z", ("z",)),
    Setter(_MOBJECT, "next_to", ("mobject_or_point", "direction", "buff")),
    Setter(_MOBJECT, "set_color", ("color",)),
    Setter(_MOBJECT, "become", ("mobject",), fresh_argument=True),
    Setter(_VMOBJECT, "set_color", ("color",)),
    Setter(_VMOBJECT, "set_fill", ("color", "opacity")),
    Setter(_VMOBJECT, "set_stroke", ("color", "width", "opacity")),
    Setter(_VMOBJECT, "set_opacity", ("opacity",)),
    Setter(_VMOBJECT, "set_points_as_corners", ("points",)),
    Setter("arachne.mobject.value_tracker.ValueTracker", "set_value", ("value",)),
    Setter("arachne.mobject.text.numbers.DecimalNumber", "set_value", ("number",)),
)


def _resolve(dotted: str) -> type:
    module_name, _, name = dotted.rpartition(".")
    module = __import__(module_name, fromlist=[name])
    owner: type = getattr(module, name)
    return owner


@cache
def _setter_functions() -> dict[Any, Setter]:
    found: dict[Any, Setter] = {}
    for setter in SETTERS:
        owner = _resolve(setter.owner)
        found[owner.__dict__[setter.name]] = setter
    return found


def builtin_class(cls: type) -> bool:
    """Whether ``cls`` is defined in the engine."""
    return cls.__module__ == "arachne" or cls.__module__.startswith("arachne.")


def _class_of(receiver: Any) -> type:
    return receiver if isinstance(receiver, type) else type(receiver)


def _method_function(receiver: Any, name: str) -> Any:
    """The function ``receiver.name`` resolves to on its class (``receiver``
    may be the class itself), or ``None``.
    """
    for cls in _class_of(receiver).__mro__:
        if name in cls.__dict__:
            return cls.__dict__[name]
    return None


def setter_for(receiver: Any, name: str) -> Setter | None:
    """The admitted setter ``receiver.name`` runs, or ``None``: the receiver's
    type must be the engine's own, and the method it resolves to an entry.
    """
    if not builtin_class(_class_of(receiver)):
        return None
    return _setter_functions().get(_method_function(receiver, name))


#: Methods that read the mobject they are called on and return a value.
_VALUE_READS = (
    "get_center",
    "get_x",
    "get_y",
    "get_z",
    "get_top",
    "get_bottom",
    "get_left",
    "get_right",
    "get_corner",
    "get_edge_center",
    "get_critical_point",
    "get_boundary_point",
    "get_coord",
    "get_width",
    "get_height",
    "get_depth",
    "length_over_dim",
    "get_start",
    "get_end",
    "get_start_and_end",
    "get_vector",
    "get_unit_vector",
    "get_angle",
    "get_slope",
    "get_length",
    "get_projection",
    "get_arc_center",
    "get_value",
    "get_color",
    "get_fill_color",
    "get_stroke_color",
    "get_fill_opacity",
    "get_stroke_opacity",
    "get_stroke_width",
    "get_num_points",
    "get_anchors",
    "get_points_defining_boundary",
    "get_all_points",
    "point_from_proportion",
    "proportion_from_point",
    "coords_to_point",
    "c2p",
    "point_to_coords",
    "p2c",
    "number_to_point",
    "n2p",
    "point_to_number",
    "p2n",
    "input_to_graph_point",
    "input_to_graph_coords",
    "i2gp",
    "angle_of_tangent",
    "slope_of_tangent",
    "get_origin",
    "get_x_unit_size",
    "get_y_unit_size",
    "get_unit_size",
    "point_to_polar",
    "polar_to_point",
    "pr2pt",
    "pt2pr",
    "get_phi",
    "get_theta",
    "get_gamma",
    "get_focal_distance",
    "get_zoom",
)
#: Methods that return one of the mobject's own members.
_MEMBER_READS = (
    "get_x_axis",
    "get_y_axis",
    "get_z_axis",
    "get_axis",
    "get_axes",
    "get_family",
    "get_tip",
)
#: Methods that build and return a new mobject, leaving the receiver as it is.
_BUILDS = (
    "copy",
    "plot",
    "plot_parametric_curve",
    "plot_polar_graph",
    "plot_derivative_graph",
    "plot_antiderivative_graph",
    "get_graph_label",
    "get_area",
    "get_riemann_rectangles",
    "get_vertical_line",
    "get_horizontal_line",
    "get_lines_to_point",
    "get_line_from_axis_to_point",
    "get_T_label",
    "get_secant_slope_group",
    "get_vertical_lines_to_graph",
)
#: Methods of a freshly built mobject that may take a scene mobject as an
#: argument: they read it and change only the receiver.
_READS_ITS_ARGUMENTS = (
    "next_to",
    "move_to",
    "align_to",
    "match_x",
    "match_y",
    "match_z",
    "match_coord",
    "match_height",
    "match_width",
    "match_depth",
    "match_dim_size",
    "match_style",
    "match_color",
    "replace",
    "surround",
    "put_start_and_end_on",
    "set_x",
    "set_y",
    "set_z",
    "set_color",
)


@cache
def engine_classes() -> tuple[type, ...]:
    """Every engine class a mobject or camera can be."""
    import arachne

    found: dict[type, None] = {}
    stack: list[type] = [
        value
        for value in vars(arachne).values()
        if isinstance(value, type) and builtin_class(value)
    ]
    while stack:
        cls = stack.pop()
        if cls in found:
            continue
        found[cls] = None
        stack.extend(c for c in cls.__subclasses__() if builtin_class(c))
    return tuple(found)


@cache
def _functions_named(names: tuple[str, ...]) -> frozenset[Any]:
    functions: set[Any] = set()
    for cls in engine_classes():
        for name in names:
            function = cls.__dict__.get(name)
            if function is not None:
                functions.add(function)
    return frozenset(functions)


def read_kind(receiver: Any, name: str) -> str | None:
    """How ``receiver.name()`` reads ``receiver``: ``"value"`` (it returns a
    value), ``"member"`` (one of its own members), ``"build"`` (a new
    mobject) or ``None`` when it is not a read the engine vouches for.
    """
    if not builtin_class(_class_of(receiver)):
        return None
    function = _method_function(receiver, name)
    if function is None:
        return None
    if function in _functions_named(_VALUE_READS):
        return "value"
    if function in _functions_named(_MEMBER_READS):
        return "member"
    if function in _functions_named(_BUILDS):
        return "build"
    return None


def fresh_method(receiver: Any, name: str) -> str | None:
    """Whether a callback may call ``receiver.name`` on a mobject it built in
    the same call, and what it returns: ``"mobject"`` (the receiver, changed,
    or a mobject it builds), ``"value"``, or ``None`` when the call is not
    admitted. ``"+reads"`` is appended (as ``"mobject+reads"``) when the
    method may take a scene mobject as an argument without changing it.

    Admitted by identity: the engine's own function under a name of
    :data:`_FRESH_CHANGES`, :data:`_BUILDS` or :data:`_VALUE_READS`. Each
    changes the receiver alone, from its arguments, and draws, reads and
    writes nothing else (no ``shuffle``, ``save_image``, ``add_updater``).
    """
    if not builtin_class(_class_of(receiver)):
        return None
    function = _method_function(receiver, name)
    if function is None:
        return None
    if function in _functions_named(_VALUE_READS):
        return "value"
    if function in _functions_named(_FRESH_CHANGES) or function in _functions_named(
        _BUILDS
    ):
        if function in _functions_named(_READS_ITS_ARGUMENTS):
            return "mobject+reads"
        return "mobject"
    return None


#: Methods a callback may call on a mobject it built: each changes that
#: mobject (its family) from its arguments alone.
_FRESH_CHANGES = (
    "shift",
    "scale",
    "rotate",
    "flip",
    "stretch",
    "center",
    "move_to",
    "next_to",
    "align_to",
    "to_edge",
    "to_corner",
    "set_x",
    "set_y",
    "set_z",
    "set_color",
    "set_fill",
    "set_stroke",
    "set_opacity",
    "set_z_index",
    "scale_to_fit_width",
    "scale_to_fit_height",
    "stretch_to_fit_width",
    "stretch_to_fit_height",
    "match_x",
    "match_y",
    "match_z",
    "match_coord",
    "match_height",
    "match_width",
    "match_depth",
    "match_dim_size",
    "match_style",
    "match_color",
    "replace",
    "surround",
    "put_start_and_end_on",
    "arrange",
    "arrange_in_grid",
    "add",
    "set_value",
    "set_points_as_corners",
)


def returned_class(function: Any) -> type | None:
    """The engine mobject class ``function``'s return annotation names, or
    ``None``.
    """
    try:
        annotation = inspect.signature(function).return_annotation
    except (TypeError, ValueError):
        return None
    if not isinstance(annotation, str):
        return None
    import arachne
    from arachne.mobject.mobject import Mobject

    value = getattr(arachne, annotation, None)
    if isinstance(value, type) and issubclass(value, Mobject):
        return value
    return None


def method_function(receiver: Any, name: str) -> Any:
    """The function ``receiver.name`` runs (``receiver`` may be a class)."""
    return _method_function(receiver, name)


#: Constructors that may take scene mobjects as arguments: they read them and
#: hold none of them.
_READER_CONSTRUCTORS = (
    "Line",
    "DashedLine",
    "Arrow",
    "DoubleArrow",
    "Vector",
    "TangentLine",
    "Elbow",
    "Angle",
    "RightAngle",
    "Brace",
    "BraceBetweenPoints",
    "SurroundingRectangle",
    "BackgroundRectangle",
    "Cross",
    "Underline",
    "ArcBetweenPoints",
    "CurvedArrow",
    "CurvedDoubleArrow",
    "Dot",
    "DecimalNumber",
    "Integer",
)
#: Constructors that take values (and mobjects the callback built) only.
#: Each builds a new mobject from its arguments alone: none reads a file
#: (``ImageMobject``, ``SVGMobject``, ``Code``), the camera
#: (``ImageMobjectFromCamera``) or a random generator (``Graph``'s layouts),
#: and none attaches an updater (``Variable``, ``TracedPath``).
_VALUE_CONSTRUCTORS = (
    "AnnotationDot",
    "AnnularSector",
    "Annulus",
    "Arc",
    "ArcBrace",
    "ArcPolygon",
    "Arrow3D",
    "Axes",
    "BraceLabel",
    "BraceText",
    "BulletedList",
    "Circle",
    "ComplexPlane",
    "Cone",
    "Cube",
    "CubicBezier",
    "Cylinder",
    "DashedVMobject",
    "DecimalMatrix",
    "DecimalTable",
    "Difference",
    "Dot3D",
    "Ellipse",
    "Exclusion",
    "FunctionGraph",
    "Group",
    "ImplicitFunction",
    "IntegerMatrix",
    "IntegerTable",
    "Intersection",
    "LabeledDot",
    "Line3D",
    "MarkupText",
    "MathTable",
    "MathTex",
    "Matrix",
    "Mobject",
    "MobjectMatrix",
    "MobjectTable",
    "NumberLine",
    "NumberPlane",
    "ParametricFunction",
    "Paragraph",
    "PolarPlane",
    "Polygon",
    "Polygram",
    "Prism",
    "Rectangle",
    "RegularPolygon",
    "RegularPolygram",
    "RoundedRectangle",
    "Sector",
    "SingleStringMathTex",
    "Sphere",
    "Square",
    "Star",
    "Surface",
    "Table",
    "Tex",
    "Text",
    "ThreeDAxes",
    "Title",
    "Torus",
    "Triangle",
    "Union",
    "UnitInterval",
    "VDict",
    "VGroup",
    "VMobject",
    "ValueTracker",
    "VectorizedPoint",
)


@cache
def _constructors() -> dict[type, str]:
    import arachne

    found: dict[type, str] = {}
    for names, kind in (
        (_VALUE_CONSTRUCTORS, "values"),
        (_READER_CONSTRUCTORS, "reads"),
    ):
        for name in names:
            found[getattr(arachne, name)] = kind
    return found


def constructor_kind(cls: Any) -> str | None:
    """How a callback may build an instance of ``cls``: ``"reads"`` (it may
    take scene mobjects as arguments), ``"values"`` (it may take only values
    and mobjects the callback built), or ``None`` (not an admitted
    constructor). Admitted by identity: the engine's own class.
    """
    if not isinstance(cls, type):
        return None
    return _constructors().get(cls)


#: Engine modules a callback may read names off (constants and the
#: functions the library table admits; a function it calls must be an
#: entry, :mod:`.library`).
_READABLE_MODULES = (
    "arachne.utils.space_ops",
    "arachne.utils.bezier",
    "arachne.utils.simple_functions",
    "arachne.utils.rate_functions",
    "arachne.utils.color",
)


def pure_module(module: types.ModuleType) -> bool:
    """Whether a callback may read names off ``module``."""
    name = module.__name__
    if name in ("math", "numpy", "numpy.linalg", "arachne"):
        return True
    return name.startswith(_READABLE_MODULES) or name.startswith("arachne.constants")
