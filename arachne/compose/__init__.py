"""Drawing a scene that holds a GPU-only capability.

:mod:`.device` is the one GPU device a process draws on; :mod:`.guard`
refuses a render of such a scene, which this build does not draw yet.

Nothing here imports the GPU build when imported: only a scene on the
``gpu`` route asks for it.
"""

from __future__ import annotations
