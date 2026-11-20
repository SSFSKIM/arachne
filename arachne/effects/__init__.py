"""Effects for a scene's objects: a blur, a mask, a depth under the scene's
focus, the plays that move a blur or the focus, and a window that shows a
group of objects as a frame of their own.

Every effect makes its object a layer of its own, drawn by the GPU build
(:mod:`arachne.compose`): a scene that sets one is drawn on a GPU, or
refused with ``render.gpu_required`` on a host with none, and never drawn a
lesser way.
"""

from __future__ import annotations

from .animations import *
from .layers import *
from .window import *
