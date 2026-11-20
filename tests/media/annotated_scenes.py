"""Scenes of callouts on media: on a point of a running clip that moves and
grows, on the anchor of a rig of stills while it is posed, and on an
anchor of a layered set's depth member while the camera pans and the set
fades in; each while the camera moves.
"""

from __future__ import annotations

import numpy as np

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UR,
    Callout,
    FadeIn,
    MovingCameraScene,
    Pose,
    Rig,
    VideoMobject,
    anchor,
)
from arachne.utils.rate_functions import linear

#: How the camera's frame moves while the annotated thing moves.
CAMERA_MOVE = 0.6 * RIGHT + 0.3 * DOWN
#: How much the camera's frame narrows: it zooms in.
CAMERA_ZOOM = 0.8
#: How the layered scene pans the camera.
PAN = np.array([2.0, 0.75, 0.0])


class _CameraMoves(MovingCameraScene):
    def camera_moves(self):
        return self.camera.frame.animate.shift(CAMERA_MOVE).scale(CAMERA_ZOOM)


class CalloutOnRunningClip(_CameraMoves):
    """A callout on a running clip's upper right corner, the clip moved and
    grown by an animation while the camera moves.
    """

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip").scale(0.6)
        label = Callout("clip", at=anchor(clip, at=UR), offset=UR, name="callout")
        self.add(clip, label)
        clip.run()
        self.play(
            clip.animate.shift(1.5 * LEFT + 0.5 * DOWN).scale(1.3),
            self.camera_moves(),
            run_time=1.5,
        )
        self.wait(0.25)


class CalloutOnPosedFlap(_CameraMoves):
    """A callout on the anchor of a rig of stills, on its flap, which the
    rig's pose lifts while the camera moves.
    """

    def construct(self):
        rig = Rig("fixture:hinged-flap", name="flap")
        label = Callout("tip", at=rig.anchor("tip"), offset=UL, name="callout")
        self.add(rig, label)
        self.play(Pose(rig, lift=60), self.camera_moves(), run_time=1)
        self.wait(0.25)


class CalloutOnViewedLayer(MovingCameraScene):
    """A callout on the anchor of a layered set's middle layer: the camera
    pans while the set, viewed through it, fades in, in one play.
    """

    def construct(self):
        rig = Rig("fixture:layered-ground")
        rig.view_through(self.camera)
        label = Callout("peak", at=rig.anchor("peak"), offset=UR, name="callout")
        self.add(label)
        self.play(
            self.camera.frame.animate.shift(PAN),
            FadeIn(rig),
            rate_func=linear,
            run_time=1,
        )
        self.wait(0.25)


class CalloutBeforeMovingClip(MovingCameraScene):
    """A callout added before the clip it names, which its own updater
    moves: the callout reads the clip where it stood a tick before.
    """

    def construct(self):
        clip = VideoMobject("fixture:numbered-clip").scale(0.6)
        label = Callout("clip", at=anchor(clip, at=UR), offset=UR, name="callout")
        clip.add_updater(lambda m, dt: m.shift(2.4 * RIGHT * dt))
        self.add(label, clip)
        clip.run()
        self.wait(1)
