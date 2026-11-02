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
  (:func:`fresh_method`), and the constructors and functions it may call.
"""

from __future__ import annotations

__all__ = [
    "PURE_BUILTINS",
    "SETTERS",
    "Setter",
    "builtin_class",
    "constructor_kind",
    "fresh_method",
    "pure_function",
    "method_function",
    "read_kind",
    "returned_class",
    "setter_for",
]

import inspect
import types
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Setter:
    """An admitted setter: the class that defines it and its name.

    ``fresh_argument`` says the setter takes a mobject argument that must be
    one the callback built in the same call (``become`` aligns its argument's
    family to its own, so given a scene mobject it would change it).
    """

    owner: str
    name: str
    fresh_argument: bool = False


#: The admitted setters, by the defining class's dotted path and method name.
SETTERS: tuple[Setter, ...] = (
    Setter("arachne.mobject.mobject.Mobject", "move_to"),
    Setter("arachne.mobject.mobject.Mobject", "set_x"),
    Setter("arachne.mobject.mobject.Mobject", "set_y"),
    Setter("arachne.mobject.mobject.Mobject", "set_z"),
    Setter("arachne.mobject.mobject.Mobject", "next_to"),
    Setter("arachne.mobject.mobject.Mobject", "set_color"),
    Setter("arachne.mobject.mobject.Mobject", "become", fresh_argument=True),
    Setter("arachne.mobject.types.vectorized_mobject.VMobject", "set_color"),
    Setter("arachne.mobject.types.vectorized_mobject.VMobject", "set_fill"),
    Setter("arachne.mobject.types.vectorized_mobject.VMobject", "set_stroke"),
    Setter("arachne.mobject.types.vectorized_mobject.VMobject", "set_opacity"),
    Setter(
        "arachne.mobject.types.vectorized_mobject.VMobject", "set_points_as_corners"
    ),
    Setter("arachne.mobject.value_tracker.ValueTracker", "set_value"),
    Setter("arachne.mobject.text.numbers.DecimalNumber", "set_value"),
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
def _engine_classes() -> tuple[type, ...]:
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
    for cls in _engine_classes():
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
    the same call, and what it returns: ``"mobject"``, ``"value"``, or
    ``None`` when the call is not admitted. ``"reads"`` is appended (as
    ``"mobject+reads"``) when the method may take a scene mobject as an
    argument without changing it.
    """
    if not builtin_class(_class_of(receiver)):
        return None
    function = _method_function(receiver, name)
    if not callable(function) or name.startswith("_") or name in _NEVER_ON_FRESH:
        return None
    if isinstance(function, (staticmethod, classmethod, property)):
        return None
    kind = "value"
    try:
        annotation = inspect.signature(function).return_annotation
    except (TypeError, ValueError):
        annotation = None
    if annotation == "Self" or returned_class(function) is not None:
        kind = "mobject"
    if function in _functions_named(_READS_ITS_ARGUMENTS):
        kind += "+reads"
    return kind


#: Methods of a fresh mobject that reach beyond it: callbacks, animations,
#: and the scene.
_NEVER_ON_FRESH = frozenset(
    {
        "add_updater",
        "remove_updater",
        "clear_updaters",
        "update",
        "suspend_updating",
        "resume_updating",
        "animate",
        "generate_target",
        "save_state",
        "restore",
        "add_to_back",
        "become",
        "match_points",
        "align_data",
        "align_submobjects",
        "interpolate",
        "add_background_rectangle",
    }
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


def constructor_kind(cls: Any) -> str | None:
    """How a callback may build an instance of ``cls``: ``"reads"`` (it may
    take scene mobjects as arguments), ``"values"`` (it may take only values
    and mobjects the callback built), or ``None`` (not a built-in mobject).
    """
    from arachne.mobject.mobject import Mobject

    if not (isinstance(cls, type) and issubclass(cls, Mobject) and builtin_class(cls)):
        return None
    if cls.__name__ in _READER_CONSTRUCTORS:
        return "reads"
    return "values"


#: Builtins with no effect beyond their result.
PURE_BUILTINS: frozenset[Any] = frozenset(
    {
        abs,
        min,
        max,
        round,
        int,
        float,
        complex,
        bool,
        len,
        range,
        sum,
        zip,
        enumerate,
        reversed,
        sorted,
        list,
        tuple,
        dict,
        set,
        frozenset,
        pow,
        divmod,
        any,
        all,
        isinstance,
        str,
        map,
        filter,
        slice,
    }
)

#: numpy functions a callback may not call: draws, I/O, and writes into an
#: array given to them.
_NUMPY_REFUSED = frozenset(
    {
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "load",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "memmap",
        "copyto",
        "put",
        "place",
        "putmask",
        "fill_diagonal",
        "seterr",
        "seterrcall",
        "setbufsize",
        "set_printoptions",
        "random",
    }
)
#: Engine modules whose functions compute values and change nothing.
_PURE_MODULES = (
    "arachne.utils.space_ops",
    "arachne.utils.bezier",
    "arachne.utils.simple_functions",
    "arachne.utils.rate_functions",
    "arachne.utils.color",
)


def pure_function(function: Any) -> bool:
    """Whether ``function`` is a library function whose only effect is its
    result: a pure builtin, a ``math`` function, a numpy function that draws,
    writes and reads nothing else, or a function of the engine's value
    modules.
    """
    if function in PURE_BUILTINS:
        return True
    if (
        isinstance(function, types.BuiltinFunctionType)
        and getattr(function, "__module__", None) == "math"
    ):
        return True
    if isinstance(function, np.ufunc):
        return True
    module = getattr(function, "__module__", None) or ""
    name = getattr(function, "__name__", "")
    if module.startswith("numpy"):
        if module.startswith("numpy.random") or name in _NUMPY_REFUSED:
            return False
        return any(getattr(space, name, None) is function for space in (np, np.linalg))
    if isinstance(function, types.FunctionType) and module.startswith(_PURE_MODULES):
        return True
    from arachne.utils.color.core import ManimColor

    return function is ManimColor


def pure_module(module: types.ModuleType) -> bool:
    """Whether a callback may read names off ``module``."""
    name = module.__name__
    if name in ("math", "numpy", "numpy.linalg", "arachne"):
        return True
    return name.startswith(_PURE_MODULES) or name.startswith("arachne.constants")
