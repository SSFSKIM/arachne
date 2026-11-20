"""Drawing a scene that holds a GPU-only capability.

The seam a scene uses: :class:`LayerSpec` and :func:`set_layer` make an
object a layer of its own with effects, :func:`set_focus` gives the camera
a depth in focus (:mod:`.layer`). A session draws such a scene as a stack
(:mod:`.session`, :mod:`.stack`) on the one GPU device a process draws on
(:mod:`.device`); :mod:`.guard` refuses a render of it, which this build
does not draw yet.

Nothing here imports the GPU build when imported: only a scene on the
``gpu`` route asks for it.
"""

from __future__ import annotations

__all__ = ["LayerSpec", "set_focus", "set_layer"]

from .layer import LayerSpec, set_focus, set_layer
