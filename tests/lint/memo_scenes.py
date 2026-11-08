"""Scenes the leaf memo's tests read. Each holds a :class:`Probe`, the leaf a
test counts the reads of. A scene of one input holds the probe, changes that
input alone between two frames, and holds it again: everything a read
depends on is the same within each hold and different across the change.
"""

from __future__ import annotations

from arachne import (
    DEGREES,
    LEFT,
    RIGHT,
    UP,
    Dot,
    MovingCameraScene,
    Scene,
    Square,
    ThreeDScene,
    VGroup,
    VMobject,
    always_redraw,
)

#: How long each hold lasts, in seconds.
HOLD = 0.4


class Probe(Square):
    """The leaf a test counts."""


def probe() -> Probe:
    return Probe(side_length=1.0).set_fill(opacity=0.2).set_stroke(width=4)


class _Held3D(ThreeDScene):
    """A probe under a tilted 3D camera; ``change`` runs between two holds."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.probe = probe()
        self.add(self.probe)
        self.wait(HOLD)
        self.change()
        self.wait(HOLD)

    def change(self):
        pass


class HeldProbe3D(_Held3D):
    """Nothing changes."""


class HeldProbe2D(Scene):
    def construct(self):
        self.add(probe())
        self.wait(2 * HOLD)


class TurningProbe3D(ThreeDScene):
    """The camera turns at every frame; the probe holds."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(probe())
        self.move_camera(theta=45 * DEGREES, run_time=2 * HOLD)


class TurnsBetweenHolds3D(ThreeDScene):
    """The camera holds, turns, and holds where it turned to."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(probe())
        self.wait(HOLD)
        self.move_camera(theta=0, run_time=HOLD)
        self.wait(HOLD)


# ── one of the probe's own inputs ──────────────────────────────────────────


class ProbeShifts(_Held3D):
    def change(self):
        self.probe.shift(RIGHT * 0.5)


class ProbeFillChanges(_Held3D):
    def change(self):
        self.probe.set_fill(opacity=0.6)


class ProbeStrokeWidens(_Held3D):
    def change(self):
        self.probe.set_stroke(width=8)


class ProbeStrokeFades(_Held3D):
    def change(self):
        self.probe.set_stroke(opacity=0.5)


# ── one of the camera's inputs ─────────────────────────────────────────────


class CameraPhi(_Held3D):
    def change(self):
        self.set_camera_orientation(phi=40 * DEGREES)


class CameraTheta(_Held3D):
    def change(self):
        self.set_camera_orientation(theta=-20 * DEGREES)


class CameraGamma(_Held3D):
    def change(self):
        self.set_camera_orientation(gamma=10 * DEGREES)


class CameraFocalDistance(_Held3D):
    def change(self):
        self.set_camera_orientation(focal_distance=8)


class CameraZoom(_Held3D):
    def change(self):
        self.set_camera_orientation(zoom=1.3)


class CameraFrameCentre(_Held3D):
    def change(self):
        self.set_camera_orientation(frame_center=(0.5, 0.2, 0.0))


class ProbeFixedInFrame(_Held3D):
    def change(self):
        self.add_fixed_in_frame_mobjects(self.probe)


class FixedOrientationCentreMoves(ThreeDScene):
    """The probe keeps its orientation about an anchor's centre; the anchor
    moves and the probe holds its own state.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        anchor = Dot()
        self.add(anchor)
        label = probe()
        self.add_fixed_orientation_mobjects(label, center_func=anchor.get_center)
        self.wait(HOLD)
        anchor.shift(UP * 0.5)
        self.wait(HOLD)


class MovingFrameShifts(MovingCameraScene):
    def construct(self):
        self.add(probe())
        self.wait(HOLD)
        self.camera.frame.shift(LEFT * 0.5)
        self.wait(HOLD)


class MovingFrameWidens(MovingCameraScene):
    def construct(self):
        self.add(probe())
        self.wait(HOLD)
        self.camera.frame.scale(1.25)
        self.wait(HOLD)


# ── what the memo never keeps ──────────────────────────────────────────────


class ProbeWithUpdater(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(probe().add_updater(lambda mob: None))
        self.wait(2 * HOLD)


class ProbeInUpdatedGroup(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        group = VGroup(probe())
        group.add_updater(lambda mob: None)
        self.add(group)
        self.wait(2 * HOLD)


class RedrawnProbe(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(always_redraw(probe))
        self.wait(2 * HOLD)


class SceneUpdaterBeside(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add_updater(lambda dt: None)
        self.add(probe())
        self.wait(2 * HOLD)


class ReplacedProjection(ThreeDScene):
    """The scene's code puts its own projection on the camera instance."""

    def construct(self):
        camera = self.renderer.camera
        engine = camera.project_points

        def projected(points):
            return engine(points)

        camera.project_points = projected
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(probe())
        self.wait(2 * HOLD)


class ReplacedPathTrace(ThreeDScene):
    """The scene's code puts its own path trace on the camera instance."""

    def construct(self):
        camera = self.renderer.camera
        engine = camera.set_cairo_context_path

        def traced(ctx, vmobject):
            return engine(ctx, vmobject)

        camera.set_cairo_context_path = traced
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(probe())
        self.wait(2 * HOLD)


class ReplacedDisplay(Scene):
    """The scene's code puts its own function in the camera's table of display
    functions, in place of the engine's for vectorized mobjects.
    """

    def construct(self):
        display = self.renderer.camera.display_funcs
        engine = display[VMobject]
        display[VMobject] = lambda mobjects, pixel_array: engine(mobjects, pixel_array)
        self.add(probe())
        self.wait(2 * HOLD)


# ── a camera that moves, with leaves fixed in its frame ────────────────────


class _TurningFixed3D(ThreeDScene):
    """A probe fixed in the frame and a dot that is not, the camera turning
    across every hold; ``change`` runs between two turns.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.probe = probe()
        self.dot = Dot(RIGHT)
        self.add(self.dot)
        self.add_fixed_in_frame_mobjects(self.probe)
        self.move_camera(theta=-15 * DEGREES, run_time=HOLD)
        self.change()
        self.move_camera(theta=15 * DEGREES, run_time=HOLD)

    def change(self):
        pass


class TurningFixedProbe3D(_TurningFixed3D):
    """Nothing but the camera changes."""


class FixedProbeChangesWhileTurning(_TurningFixed3D):
    """While the camera turns, the fixed probe and the dot each change their
    points, their style, and are rebuilt.
    """

    def change(self):
        self.probe.shift(LEFT * 0.5)
        self.dot.shift(UP * 0.5)
        self.move_camera(theta=0, run_time=HOLD)
        self.probe.set_fill(opacity=0.6).set_stroke(width=8)
        self.dot.set_color("#ff0000")
        self.move_camera(theta=5 * DEGREES, run_time=HOLD)
        self.probe.become(probe().scale(0.5).shift(UP))
        self.dot.become(Dot(LEFT))


class FixedProbeUnderFrameCentreAndZoom(_TurningFixed3D):
    """The camera moves its frame's centre and zooms as it turns."""

    def change(self):
        self.move_camera(theta=0, frame_center=(0.5, 0.2, 0.0), zoom=1.3, run_time=HOLD)


class ProbeFixedMidTurn(ThreeDScene):
    """The probe is drawn unfixed while the camera turns, then fixed in the
    frame while it turns on.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.probe = probe()
        self.add(self.probe)
        self.move_camera(theta=-15 * DEGREES, run_time=HOLD)
        self.add_fixed_in_frame_mobjects(self.probe)
        self.move_camera(theta=15 * DEGREES, run_time=HOLD)


class ProbeUnfixedMidTurn(_TurningFixed3D):
    """The fixed probe is let go of the frame while the camera turns on."""

    def change(self):
        self.remove_fixed_in_frame_mobjects(self.probe)


class FixedGroupWhileTurning(ThreeDScene):
    """A group fixed in the frame: its children, the probe among them, are
    fixed with it.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        group = VGroup(probe(), Dot(UP))
        self.add_fixed_in_frame_mobjects(group)
        self.move_camera(theta=15 * DEGREES, run_time=2 * HOLD)


class FixedOrientationWhileTurning(ThreeDScene):
    """A probe off the origin that keeps its orientation, not fixed in the
    frame, as the camera turns: its centre is projected, so it moves.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        label = probe().shift(RIGHT * 2 + UP * 0.5)
        self.add_fixed_orientation_mobjects(label)
        self.move_camera(theta=15 * DEGREES, run_time=2 * HOLD)
