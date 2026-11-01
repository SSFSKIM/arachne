from __future__ import annotations

from arachne import *
from arachne.utils.testing.frames_comparison import frames_comparison

__module_test__ = "specialized"


@frames_comparison(last_frame=False)
def test_Broadcast(scene):
    circle = Circle()
    scene.play(Broadcast(circle))
