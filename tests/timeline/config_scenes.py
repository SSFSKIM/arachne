"""A scene whose file changes the configuration as it is imported: opened
in a session only, which imports it under the session's configuration (no
test imports it, which would change the process's).
"""

from __future__ import annotations

import numpy as np

from arachne import YELLOW, PMobject, Scene, config

config.pixel_width *= 2


class PixelShapeChangedOnImport(Scene):
    """A point cloud for a wait, the configured pixel width doubled as the
    file was imported.
    """

    def construct(self):
        cloud = PMobject(stroke_width=4)
        cloud.add_points(np.array([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]), color=YELLOW)
        self.add(cloud)
        self.wait(0.2)
