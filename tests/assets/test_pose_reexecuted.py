"""A scene executed again on the same object (the manager renews it in place
when a play must be drawn as it runs) settles its posed rigs in the new
execution as in the first: the pose settle is the execution's, so a timing
declared later in the first execution leaves the renewed one's settle
registering itself until the renewed scene declares.
"""

from __future__ import annotations

import numpy as np

from arachne import Scene, Square, set_timing, tempconfig
from arachne.assets import Pose, Rig
from arachne.assets.motion import pose_settle
from arachne.manager import Manager
from arachne.timeline.execution import Reexecute
from arachne.utils.rate_functions import linear

SMALL = {"pixel_width": 96, "pixel_height": 54}


class PoseThenDeclaration(Scene):
    """A lidded box opened over half a second, then a square declared to
    stepped timing, then a wait. With ``again`` set, the first execution
    asks, before the wait, to be executed again with the wait drawn as it
    runs.
    """

    again = False

    def construct(self) -> None:
        box = Rig("fixture:lidded-box", name="box")
        self.add(box)
        self.play(Pose(box, lid_open=110, rate_func=linear), run_time=0.5)
        square = Square()
        self.add(square)
        set_timing(square, 3)
        if type(self).again:
            type(self).again = False
            raise Reexecute(1, "the test asks for the wait to be drawn as it runs")
        self.wait(0.25)


def _render() -> tuple[Scene, list[list[np.ndarray]]]:
    """Render :class:`PoseThenDeclaration` executed twice on one scene
    object, nothing written: the scene, and the frames each execution drew.
    """
    executions: list[list[np.ndarray]] = []
    PoseThenDeclaration.again = True
    with tempconfig(
        {**SMALL, "frame_rate": 30, "dry_run": True, "disable_caching": True}
    ):
        scene = PoseThenDeclaration()
        manager = Manager(scene)

        def collect() -> None:
            frames: list[np.ndarray] = []
            executions.append(frames)

            def write_frame(pixels, *, repeat=1):
                frames.extend([pixels] * repeat)

            scene.renderer.file_writer.write_frame = write_frame

        renew = manager._renew_scene

        def renewed() -> None:
            renew()
            collect()

        manager._renew_scene = renewed
        collect()
        scene.render()
    return scene, executions


def test_a_renewed_execution_settles_a_pose_made_before_its_declaration():
    scene, (first, renewed) = _render()
    # The first execution stopped after the pose: its frames are the pose's,
    # the lid opening across them.
    pose = len(first)
    assert pose > 2
    assert not np.array_equal(first[0], first[-1])
    # The renewed execution draws the lid opening as the first did.
    differing = [
        k
        for k, (a, b) in enumerate(zip(first, renewed[:pose], strict=True))
        if not np.array_equal(a, b)
    ]
    assert differing == []
    # The renewed scene's settle is its own: made registering, before its
    # declaration, and handed to the clock at it.
    settle = pose_settle(scene)
    assert settle is not None
    assert settle.registers is False
    assert settle.updater not in scene.updaters
