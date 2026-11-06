"""What CE's lighting derives from nearby points is decided before the snap.

A shaded path is lit at two corners by CE's normal there, the cross product of
the chords to the points three on and three back. On a finely sampled curve
those chords are nearly opposite, and the snap's step would turn the normal by
tens of degrees; at a sphere's pole they are a float's noise long or nothing,
and the normal is the noise's direction or CE's fixed fallback. The engine
decides the normals on the values as made (``Stream.normals``) and carries them
with the snapped record (``frame_ir.canonical``); at a corner less than a grid
step from its neighbour, coincident included, the core lights the face by its
own normal instead.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from arachne import (
    DEGREES,
    IN,
    PI,
    RED,
    RIGHT,
    TAU,
    ParametricFunction,
    Square,
    Surface,
    ThreeDAxes,
    ThreeDScene,
)
from arachne.frame_ir.canonical import RATIO_GRID, Canonical
from arachne.frame_ir.core_painter import CorePainter
from arachne.frame_ir.extract import Unable
from arachne.mobject.three_d.three_d_utils import (
    get_3d_vmob_end_corner_unit_normal,
    get_3d_vmob_start_corner_unit_normal,
)

from ..timeline.conftest import render
from .core_scenes import BackgroundFilledOverlay

pytest.importorskip("arachne_core")

FPS = 15
FULL = {"pixel_width": 854, "pixel_height": 480}


def spring() -> ParametricFunction:
    return ParametricFunction(
        lambda u: (1.2 * np.cos(u), 1.2 * np.sin(u), u * 0.05),
        color=RED,
        t_range=(-3 * TAU, 5 * TAU, 0.01),
    ).set_shade_in_3d(True)


class Spring(ThreeDScene):
    """The docs' parametric spring: a shaded curve sampled 0.01 apart."""

    def construct(self):
        self.add(ThreeDAxes(), spring())
        self.set_camera_orientation(phi=80 * DEGREES, theta=-60 * DEGREES)
        self.wait(1 / FPS)


class SpringAlone(ThreeDScene):
    """The spring, alone: the one shaded record."""

    def construct(self):
        self.add(spring())
        self.set_camera_orientation(phi=80 * DEGREES, theta=-60 * DEGREES)
        self.wait(1 / FPS)


def test_a_shaded_curve_carries_the_normals_of_its_points_as_made(monkeypatch):
    """The spring's record carries CE's normals at its lit corners, found from
    its points as made, to within the ratio grid it is snapped to.
    """
    call = Canonical.__call__
    carried: list[list[tuple[float, ...]]] = []

    def capturing(self, delta):
        found = call(self, delta)
        carried.extend(
            r.fields["normals"] for _, r in found.set if "normals" in r.fields
        )
        return found

    monkeypatch.setattr(Canonical, "__call__", capturing)
    render(SpringAlone, FPS, draw_path="core", **FULL)
    curve = spring()
    want = [
        get_3d_vmob_start_corner_unit_normal(curve),
        get_3d_vmob_end_corner_unit_normal(curve),
    ]
    assert carried, "no shaded record carried normals"
    for normals in carried:
        assert len(normals) == 2
        np.testing.assert_allclose(normals, want, rtol=0, atol=RATIO_GRID)


def _off_cairo(frame: np.ndarray, cairo: np.ndarray) -> int:
    """Pixels more than 8 levels from Cairo's in any channel."""
    d = np.abs(frame[..., :3].astype(int) - cairo[..., :3].astype(int)).max(-1)
    return int((d > 8).sum())


def test_a_shaded_curve_is_as_far_from_cairo_snapped_as_not(monkeypatch):
    """Lit by the normals decided before the snap, the spring is no further
    from Cairo's frame than the core draws it from its points unsnapped.
    """
    _, cairo = render(Spring, FPS, draw_path="ir_cairo", **FULL)
    _, snapped = render(Spring, FPS, draw_path="core", **FULL)
    # The core handed the deltas as made: the reference, past the guard.
    monkeypatch.setattr(Canonical, "__call__", lambda self, delta: delta)
    draw = CorePainter.draw
    monkeypatch.setattr(
        CorePainter, "draw", lambda self, delta: draw(self, delta, _unguarded=True)
    )
    _, unsnapped = render(Spring, FPS, draw_path="core", **FULL)
    assert _off_cairo(snapped[-1], cairo[-1]) <= _off_cairo(unsnapped[-1], cairo[-1])


class LitSphere(ThreeDScene):
    """The docs' sphere lit from below, one colour, no strokes: its pole's faces
    have two corners ``cos(pi / 2)`` apart.
    """

    #: Whether each pole's points coincide exactly instead.
    coincident_poles = False

    def construct(self):
        coincident = self.coincident_poles

        def point(u, v):
            ring = 0.0 if coincident and abs(u) == PI / 2 else np.cos(u)
            return np.array(
                [1.5 * ring * np.cos(v), 1.5 * ring * np.sin(v), 1.5 * np.sin(u)]
            )

        sphere = Surface(
            point,
            v_range=[0, TAU],
            u_range=[-PI / 2, PI / 2],
            checkerboard_colors=False,
            fill_color=RED,
            stroke_width=0,
            resolution=(15, 32),
        )
        self.renderer.camera.light_source.move_to(3 * IN)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(sphere)
        self.wait(1 / FPS)


def _largest_step(frame: np.ndarray, x: int, y: int) -> int:
    """The largest step between neighbouring pixels on the sphere just below
    its pole at ``(x, y)``, where the pole's faces meet (the silhouette runs
    through the pole; its antialiased edge is left out).
    """
    window = frame[y + 2 : y + 14, x - 24 : x + 24, :3].astype(int)
    inside = window[..., 0] > 200
    assert inside.sum() > 400, inside.sum()
    across = np.abs(window[:, 1:] - window[:, :-1]).max(-1)[
        inside[:, 1:] & inside[:, :-1]
    ]
    down = np.abs(window[1:] - window[:-1]).max(-1)[inside[1:] & inside[:-1]]
    return max(int(across.max()), int(down.max()))


class CoincidentPoleSphere(LitSphere):
    """The lit sphere with each pole's points exactly coincident."""

    coincident_poles = True


def _top_pole(scene) -> tuple[int, int]:
    """The pixel the sphere's top pole projects to."""
    camera = scene.renderer.camera
    px, py, _ = camera.project_point(np.array([0.0, 0.0, 1.5]))
    x = int((px / camera.frame_width + 0.5) * camera.pixel_width)
    y = int((0.5 - py / camera.frame_height) * camera.pixel_height)
    return x, y


def test_a_spheres_pole_lights_as_its_neighbours_do():
    """Around the sphere's top pole the core's light steps from pixel to pixel
    no more than Cairo's does (at its faces' seams): no blotch where the
    pole's faces meet.
    """
    scene, core = render(LitSphere, FPS, draw_path="core", **FULL)
    _, cairo = render(LitSphere, FPS, draw_path="ir_cairo", **FULL)
    x, y = _top_pole(scene)
    steps = (_largest_step(core[-1], x, y), _largest_step(cairo[-1], x, y))
    assert steps[0] <= steps[1] + 4, steps


def test_a_pole_of_coincident_points_lights_as_one_a_floats_noise_apart():
    """A pole whose points coincide exactly is lit as the pole whose points
    ``cos(pi / 2)`` keeps apart, and as its neighbours: which side of zero a
    machine's rounding leaves a pole's points on changes nothing drawn.
    """
    scene, noise = render(LitSphere, FPS, draw_path="core", **FULL)
    _, coincident = render(CoincidentPoleSphere, FPS, draw_path="core", **FULL)
    _, cairo = render(LitSphere, FPS, draw_path="ir_cairo", **FULL)
    x, y = _top_pole(scene)
    differ = np.abs(coincident[-1].astype(int) - noise[-1].astype(int)).max(-1)
    assert not differ.any(), int((differ > 0).sum())
    steps = (_largest_step(coincident[-1], x, y), _largest_step(cairo[-1], x, y))
    assert steps[0] <= steps[1] + 4, steps


class SpringLitLater(ThreeDScene):
    """The spring, alone, its camera shading only from its second frame:
    ``shading_from`` is the frame the camera starts shading at.
    """

    shading_from = 1

    def construct(self):
        camera = self.renderer.camera
        camera.should_apply_shading = self.shading_from == 0
        self.add(spring())
        self.set_camera_orientation(phi=80 * DEGREES, theta=-60 * DEGREES)
        self.wait(1 / FPS)
        camera.should_apply_shading = True
        self.wait(1 / FPS)


class SpringLitThroughout(SpringLitLater):
    """:class:`SpringLitLater`, shading from its first frame."""

    shading_from = 0


def test_a_curve_lit_once_the_camera_starts_shading_carries_its_normals():
    """The spring is uploaded while the camera does not shade, and nothing of
    it changes when the camera starts to: it is lit there by the normals of
    its points as made, as a spring shaded from the start is.
    """
    _, later = render(SpringLitLater, FPS, draw_path="core", **FULL)
    _, throughout = render(SpringLitThroughout, FPS, draw_path="core", **FULL)
    differ = np.abs(later[-1].astype(int) - throughout[-1].astype(int)).max(-1)
    assert not differ.any(), int((differ > 0).sum())


def _without_normals(monkeypatch):
    monkeypatch.setattr(Canonical, "_normals", lambda self, setting, grid: {})


def _without_planes(monkeypatch):
    decide = Canonical._decide

    def undecided(self, delta):
        decide(self, delta)
        self._planes.clear()

    monkeypatch.setattr(Canonical, "_decide", undecided)


def _without_grid(monkeypatch):
    view_of = Canonical._view_of

    def gridless(self, view):
        found = view_of(self, view)
        del found["input_grid"]
        return found

    monkeypatch.setattr(Canonical, "_view_of", gridless)


@pytest.mark.parametrize(
    ("losing", "named"),
    [
        (_without_normals, "without its normals"),
        (_without_planes, "without its planes"),
        (_without_grid, "without an input grid"),
    ],
    ids=["normals", "planes", "input-grid"],
)
def test_the_core_is_never_handed_a_frame_without_what_is_decided_before_the_snap(
    losing, named, monkeypatch
):
    """A frame that reaches the core without what the writer decides before
    the snap is refused, ``render.unable``, and drawn on no other path among
    the core's frames: the render draws the whole scene again on Cairo.
    """
    from arachne.frame_ir import lint

    _, cairo = render(SpringAlone, FPS, **FULL)
    monkeypatch.setattr(lint, "raised", [])
    losing(monkeypatch)
    _, core = render(SpringAlone, FPS, draw_path="core", **FULL)
    assert len(core) == len(cairo) > 0
    for index, (a, b) in enumerate(zip(cairo, core, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
    [finding] = [f for f in lint.raised if f.code == "render.fallback"]
    assert re.search(f"render.unable: .*{named}", finding.message), finding.message


class SpringFixedInFrame(ThreeDScene):
    """The spring fixed in frame, tipped towards the viewer: a shaded record
    Cairo's overlay draws. (Untipped, its two lit corners would be one point
    of the frame, and its gradient's direction a float's noise.)
    """

    shaded = True

    def construct(self):
        self.set_camera_orientation(phi=80 * DEGREES, theta=-60 * DEGREES)
        tipped = spring().set_shade_in_3d(self.shaded).rotate(-80 * DEGREES, RIGHT)
        self.add_fixed_in_frame_mobjects(tipped)
        self.wait(1 / FPS)


class UnshadedSpringFixedInFrame(SpringFixedInFrame):
    """:class:`SpringFixedInFrame` unshaded: where its snap moves its pixels."""

    shaded = False


def test_what_the_core_routes_to_cairo_s_overlay_is_outside_the_guard(monkeypatch):
    """The guard reads only the records the core draws: the spring fixed in
    frame, which the core leaves to Cairo's overlay to draw from its own
    points, is drawn without the planes the core would have needed.
    """
    _without_planes(monkeypatch)
    _, frames = render(SpringFixedInFrame, FPS, draw_path="core", **FULL)
    assert frames


def test_a_path_filled_from_a_background_image_is_drawn_past_the_guard():
    """A path filled from a background image, which the core refuses to
    draw, is Cairo's overlay's, and the guard lets its frames through.
    """
    _, frames = render(BackgroundFilledOverlay, FPS, draw_path="core", **FULL)
    assert frames


def _snapped_from_made(scene, monkeypatch) -> int:
    """Pixels of ``scene``'s last frame that differ from the frame the core
    draws from its values as made, past the guard.
    """
    _, snapped = render(scene, FPS, draw_path="core", **FULL)
    with monkeypatch.context() as patched:
        patched.setattr(Canonical, "__call__", lambda self, delta: delta)
        draw = CorePainter.draw
        patched.setattr(
            CorePainter, "draw", lambda self, delta: draw(self, delta, _unguarded=True)
        )
        _, made = render(scene, FPS, draw_path="core", **FULL)
    differ = np.abs(snapped[-1].astype(int) - made[-1].astype(int)).max(-1)
    return int((differ > 0).sum())


def test_a_curve_cairo_s_overlay_draws_is_drawn_from_its_values_as_made(
    monkeypatch,
):
    """Cairo's overlay draws the spring fixed in frame from its points as
    made, shaded (lit as CE lights them) or not: the snap moves none of its
    pixels, where drawn from its snapped points, the unshaded spring's edges
    moved and the shaded one, lit by normals found again from them, moved
    thousands more.
    """
    assert _snapped_from_made(SpringFixedInFrame, monkeypatch) == 0
    assert _snapped_from_made(UnshadedSpringFixedInFrame, monkeypatch) == 0


def _guarding(record_fields: dict, *, prototype_held: bool = True) -> None:
    """The guard on a frame whose one record is an ``instances`` record over
    a square's path shape (``prototype_held``; else over a shape the Python
    stream does not hold, as a mesh is), under a shading 3D view.
    """
    from arachne.frame_ir.xf import Delta, Record, Shape

    square = Shape("path", Square().points)
    view = {"camera": "three_d", "should_apply_shading": True, "input_grid": 2**-14}
    record = Record(
        "instances",
        {
            "shape": square.key,
            "placements": np.stack([np.eye(3, 4)] * 2),
            "objects": [1, 2],
            "shade_in_3d": True,
            **record_fields,
        },
    )
    painter = CorePainter()
    painter.state.apply(
        Delta(
            view=view,
            upload=[square] if prototype_held else [],
            set=[(1, record)],
            order=[("moving", 1, 0)],
        )
    )
    painter._refuse_undecided([])


#: What the writer decides for a two-member instances record of one path.
_DECIDED = {"planes": (0, 0), "normals": [(0.0, 0.0, 1.0)] * 4}


@pytest.mark.parametrize(
    ("dropped", "named"),
    [("planes", "without its planes"), ("normals", "without its normals")],
)
def test_instances_of_a_path_reach_the_core_only_with_what_is_decided_before_the_snap(
    dropped, named
):
    """The core draws an ``instances`` record whose prototype is a path as
    it draws a path: such a record without its planes, or shaded without its
    normals, is refused as a path is.
    """
    _guarding(_DECIDED)
    with pytest.raises(Unable, match=f"render.unable: .*{named}"):
        _guarding({k: v for k, v in _DECIDED.items() if k != dropped})


def test_instances_of_a_prototype_that_is_no_path_are_outside_the_path_checks():
    """An ``instances`` record over a prototype that is no path (a mesh,
    which the Python stream does not hold) carries no planes or normals.
    """
    _guarding({}, prototype_held=False)
