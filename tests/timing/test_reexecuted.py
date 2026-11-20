"""A scene executed again on the same object (the manager renews it in place
when a play must be drawn as it runs) declares its timing anew: its clock
is the new execution's, holding the new execution's mobjects, and it holds
and draws as a scene executed once does.
"""

from __future__ import annotations

import numpy as np

from arachne import RIGHT, Scene, Square, set_timing, tempconfig
from arachne.assets import timing
from arachne.manager import Manager
from arachne.timeline.execution import Reexecute
from arachne.utils.rate_functions import linear

SMALL = {"pixel_width": 96, "pixel_height": 54}


class SteppedMove(Scene):
    """A declared square moved over half a second at 3 drawings a second,
    then a wait. With ``again`` set, its first execution asks, before the
    wait, to be executed again with the wait drawn as it runs.
    """

    again = False

    def construct(self) -> None:
        square = Square()
        self.add(square)
        set_timing(square, 3)
        self.play(square.animate(rate_func=linear).shift(RIGHT), run_time=0.5)
        if type(self).again:
            type(self).again = False
            raise Reexecute(1, "the test asks for the wait to be drawn as it runs")
        self.wait(0.25)


def _render(again: bool) -> tuple[Scene, list[np.ndarray], int]:
    """Render :class:`SteppedMove` at 30 fps, nothing written: the scene,
    the frames of the execution that ended the render, and how many
    executions there were.
    """
    executions: list[list[np.ndarray]] = []
    SteppedMove.again = again
    with tempconfig(
        {**SMALL, "frame_rate": 30, "dry_run": True, "disable_caching": True}
    ):
        scene = SteppedMove()
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
    return scene, executions[-1], len(executions)


def test_a_scene_executed_again_holds_as_a_scene_executed_once():
    fresh, once, count = _render(again=False)
    assert count == 1
    renewed, twice, count = _render(again=True)
    assert count == 2
    clock = timing.clock_of(renewed)
    assert clock is not None
    # The clock is the second execution's: one illustration, of a square the
    # scene holds now, and the scene's last updater.
    assert [i.root for i in clock.illustrations] == [
        m for m in renewed.mobjects if isinstance(m, Square)
    ]
    assert renewed.updaters[-1] is clock.updater
    assert len(twice) == len(once)
    for k, (a, b) in enumerate(zip(once, twice, strict=True)):
        assert np.array_equal(a, b), k
    # It steps: frames between two boundaries repeat.
    assert any(np.array_equal(a, b) for a, b in zip(once, once[1:], strict=False))
    assert fresh is not renewed
