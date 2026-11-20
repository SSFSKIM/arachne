#!/usr/bin/env python
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Use installed distribution version if available; otherwise fall back to a
# sensible default so that importing from a source checkout works without an
# editable install (pip install -e .).
try:
    __version__ = version("arachne-engine")
except PackageNotFoundError:
    # Package is not installed; provide a fallback version string.
    __version__ = "0.0.0+unknown"


# isort: off

# Importing the config module should be the first thing we do, since other
# modules depend on the global config dict for initialization.
from ._config import *

# many scripts depend on this -> has to be loaded first
from .utils.commands import *

# isort: on
import numpy as np

from .animation.animation import *
from .animation.changing import *
from .animation.composition import *
from .animation.creation import *
from .animation.fading import *
from .animation.growing import *
from .animation.indication import *
from .animation.movement import *
from .animation.numbers import *
from .animation.rotation import *
from .animation.specialized import *
from .animation.speedmodifier import *
from .animation.transform import *
from .animation.transform_matching_parts import *
from .animation.updaters.mobject_update_utils import *
from .animation.updaters.update import *
from .assets import *
from .camera.camera import *
from .camera.mapping_camera import *
from .camera.moving_camera import *
from .camera.multi_camera import *
from .camera.three_d_camera import *
from .constants import *
from .effects import *
from .manager import *
from .mobject.annotation import *
from .mobject.diagram import *
from .mobject.frame import *
from .mobject.geometry.arc import *
from .mobject.geometry.boolean_ops import *
from .mobject.geometry.labeled import *
from .mobject.geometry.line import *
from .mobject.geometry.polygram import *
from .mobject.geometry.shape_matchers import *
from .mobject.geometry.tips import *
from .mobject.graph import *
from .mobject.graphing.coordinate_systems import *
from .mobject.graphing.functions import *
from .mobject.graphing.number_line import *
from .mobject.graphing.probability import *
from .mobject.graphing.scale import *
from .mobject.logo import *
from .mobject.marks import *
from .mobject.matrix import *
from .mobject.media import *
from .mobject.mobject import *
from .mobject.svg.brace import *
from .mobject.svg.svg_mobject import *
from .mobject.table import *
from .mobject.text.code_mobject import *
from .mobject.text.numbers import *
from .mobject.text.tex_mobject import *
from .mobject.text.text_mobject import *
from .mobject.text.typst_mobject import *
from .mobject.three_d.polyhedra import *
from .mobject.three_d.three_d_utils import *
from .mobject.three_d.three_dimensions import *
from .mobject.types.image_mobject import *
from .mobject.types.point_cloud_mobject import *
from .mobject.types.vectorized_mobject import *
from .mobject.value_tracker import *
from .mobject.vector_field import *
from .renderer.cairo_renderer import *
from .scene.moving_camera_scene import *
from .scene.scene import *
from .scene.scene_file_writer import *
from .scene.section import *
from .scene.three_d_scene import *
from .scene.vector_space_scene import *
from .scene.zoomed_scene import *
from .simulation import *
from .utils import color, rate_functions, unit
from .utils.bezier import *
from .utils.color import *
from .utils.config_ops import *
from .utils.debug import *
from .utils.file_ops import *
from .utils.images import *
from .utils.iterables import *
from .utils.paths import *
from .utils.rate_functions import *
from .utils.simple_functions import *
from .utils.sounds import *
from .utils.space_ops import *
from .utils.tex import *
from .utils.tex_templates import *

try:
    from IPython import get_ipython

    from .utils.ipython_magic import ManimMagic
except ImportError:
    pass
else:
    ipy = get_ipython()
    if ipy is not None:
        ipy.register_magics(ManimMagic)

from .plugins import *
from .timeline.frozen import freeze as _freeze

_freeze()
del _freeze

# What `from arachne import *` binds: every public name, but of the modules only
# those it bound before the engine had subpackages of its own, which scenes
# written for Manim CE may name (`cli` among them: CE's star binds it, and so did
# this one wherever the command line was imported first, as in a render). A
# subpackage is reached by its path, `arachne.<name>` or `from arachne import
# <name>`: a star that bound it would bind it again over a scene's own name with
# every helper of the scene's that star-imports the engine without an `__all__`.
# A new subpackage stays out of this list (tests/test_star_import.py).
from types import ModuleType as _ModuleType  # noqa: E402

from . import cli  # noqa: E402

_STAR_MODULES = frozenset(
    {
        "AS2700",
        "BS381",
        "DVIPSNAMES",
        "SVGNAMES",
        "X11",
        "XKCD",
        "animation",
        "camera",
        "cli",
        "color",
        "constants",
        "core",
        "data_structures",
        "frame_ir",
        "manager",
        "manim_colors",
        "mobject",
        "np",
        "plugins",
        "rate_functions",
        "renderer",
        "scene",
        "timeline",
        "typing",
        "unit",
        "utils",
    }
)
__all__ = [
    _name
    for _name, _value in list(globals().items())
    if not _name.startswith("_")
    and (not isinstance(_value, _ModuleType) or _name in _STAR_MODULES)
]
del _ModuleType, _STAR_MODULES
