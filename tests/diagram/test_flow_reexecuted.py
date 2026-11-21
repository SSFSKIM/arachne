"""A scene executed again on the same object (the manager renews it in place
when a play must be drawn as it runs) moves its flows in the new execution
as in the first: the flow driver is the execution's, so the renewed scene
has its first flow register a fresh one.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    LEFT,
    RIGHT,
    UP,
    Flow,
    Line,
    MoveAlongPath,
    Scene,
    VGroup,
    tempconfig,
)
from arachne.manager import Manager
from arachne.mobject.diagram import flow_driver
from arachne.timeline.execution import Reexecute
from arachne.utils.rate_functions import linear

SMALL = {"pixel_width": 96, "pixel_height": 54}


class FlowInMovedGroup(Scene):
    """A flow on a line, both in a group moved over 0.4 s by a play that
    suspends the group's updaters, the flow's own among them, so only the
    scene's flow driver runs the token along the line while it plays; then
    a wait. With ``again`` set, the
    first execution asks, before the wait, to be executed again with the
    wait drawn as it runs.
    """

    again = False

    def construct(self) -> None:
        line = Line(3 * LEFT, 3 * RIGHT)
        group = VGroup(line, Flow(line, speed=6.0, tokens=1, radius=0.4))
        self.add(group)
        home = group.get_center()
        self.play(
            MoveAlongPath(
                group,
                Line(home, home + UP),
                rate_func=linear,
                suspend_mobject_updating=True,
            ),
            run_time=0.4,
        )
        if type(self).again:
            type(self).again = False
            raise Reexecute(1, "the test asks for the wait to be drawn as it runs")
        self.wait(0.2)


def _render() -> tuple[Scene, list[list[np.ndarray]]]:
    """Render :class:`FlowInMovedGroup` executed twice on one scene object,
    nothing written: the scene, and the frames each execution drew.
    """
    executions: list[list[np.ndarray]] = []
    FlowInMovedGroup.again = True
    with tempconfig(
        {**SMALL, "frame_rate": 30, "dry_run": True, "disable_caching": True}
    ):
        scene = FlowInMovedGroup()
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


def test_a_renewed_execution_runs_its_flow_through_a_play_as_the_first_did():
    scene, (first, renewed) = _render()
    # The first execution stopped after the move: its frames are the move's.
    move = len(first)
    assert move > 2
    # The renewed execution draws the move as the first did, the token
    # running along the line the play carries.
    differing = [
        k
        for k, (a, b) in enumerate(zip(first, renewed[:move], strict=True))
        if not np.array_equal(a, b)
    ]
    assert differing == []
    # The renewed scene's driver holds the renewed execution's flow.
    flow = next(m for m in scene.mobjects[0].submobjects if isinstance(m, Flow))
    driver = flow_driver(scene)
    assert driver is not None
    assert driver.holds(flow)
    assert driver.updater in scene.updaters
