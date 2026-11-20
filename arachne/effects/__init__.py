"""Effects for a scene's objects: a blur, a mask, a depth under the scene's
focus, and the plays that move a blur or the focus.

Every effect makes its object a layer of its own, drawn by the GPU build
(:mod:`arachne.compose`): a scene that sets one is drawn on a GPU, or
refused with ``render.gpu_required`` on a host with none, and never drawn a
lesser way.
"""

from __future__ import annotations

from .animations import *
from .layers import *
