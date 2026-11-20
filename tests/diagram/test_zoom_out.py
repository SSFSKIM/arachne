"""A zoom comes back: :class:`ZoomOut` takes a played :class:`ZoomInto`'s
own path backwards, plays its fade windows backwards, and ends with the
frame where the move-in began and every piece showing as it did before it;
nested levels come back in order; a return the zoom cannot have left is
refused by name.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from arachne import (
    RIGHT,
    UP,
    Block,
    ScreenRectangle,
    ZoomInto,
    ZoomOut,
    config,
    linear,
    smooth,
    there_and_back,
)
from arachne.mobject.diagram import _ramp
from arachne.session import Session
from arachne.timeline import admission, hooks
from arachne.timeline.draw import Drawing

from . import zoom_out_scenes
from .zoom_out_scenes import DRAWINGS, PARTIAL, shows

SCENES = Path(zoom_out_scenes.__file__)
SMALL = {"pixel_width": 480, "pixel_height": 270}
#: The instants of a second, at the record's 60 Hz.
TICKS = 60


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str) -> Session:
        session = Session.open(SCENES, name, config={**SMALL, "frame_rate": TICKS})
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _shows(mob) -> float:
    """The most opaque any colour of ``mob``'s family is."""
    return max(
        float(np.max(opacities, initial=0.0))
        for m in mob.get_family()
        if len(m.points)
        for opacities in (m.get_fill_opacities(), m.get_stroke_opacities())
    )


def _drawn(session: Session, instants, read):
    """``read(scene)`` with the drawing on display at each of ``instants``."""
    record = session.record
    seen = []
    with Drawing(record) as drawing:
        for t in instants:
            _, capture = record.capture_at(t)
            drawing.show(capture, capture.draw_list())
            seen.append(read(record._scene))
    return seen


def _played(block: Block, **kwargs) -> tuple[ZoomInto, ScreenRectangle]:
    """A zoom onto ``block`` with a frame of its own, played outside a scene."""
    frame = ScreenRectangle(height=config.frame_height)
    zoom = ZoomInto(block, frame=frame, **kwargs)
    zoom.begin()
    zoom.finish()
    return zoom, frame


def _where(frame) -> tuple[float, np.ndarray]:
    return float(frame.height), np.array(frame.get_center())


#: Shares of a move at which the frame is read, in twentieths.
SHARES = (0, 2, 5, 8, 10, 12, 15, 18, 20)


def test_the_return_takes_the_move_in_s_path_backwards():
    block = Block("b", width=2.2, height=1.4).shift(RIGHT * 2 + UP)
    frame = ScreenRectangle(height=config.frame_height)
    zoom = ZoomInto(block, frame=frame, rate_func=linear)
    zoom.begin()
    going = {}
    for share in SHARES:
        zoom.interpolate(share / 20)
        going[share] = _where(frame)
    zoom.finish()
    back = ZoomOut(zoom)
    back.begin()
    for p in SHARES:
        back.interpolate(p / 20)
        height, centre = _where(frame)
        want_height, want_centre = going[20 - p]
        assert height == pytest.approx(want_height, rel=1e-12)
        assert np.abs(centre - want_centre).max() <= 1e-12


def test_the_return_eases_by_the_move_in_s_rate_func_unless_given_its_own():
    block = Block("b", width=2.2, height=1.4).shift(RIGHT * 2)
    zoom, frame = _played(block, run_time=3, rate_func=smooth)
    back = ZoomOut(zoom)
    assert (back.run_time, back.rate_func) == (3, smooth)
    # Read when the return is played, so a play's own run_time counts.
    zoom.run_time = 2.5
    assert back.run_time == 2.5
    own = ZoomOut(zoom, run_time=1, rate_func=linear)
    assert (own.run_time, own.rate_func) == (1, linear)
    for back, rate in ((ZoomOut(zoom), smooth), (own, linear)):
        zoom.finish()
        back.begin()
        back.interpolate(0.3)
        share = 1 - rate(0.3)
        assert frame.height == pytest.approx(zoom._height(share), rel=1e-12)
        assert np.abs(frame.get_center() - zoom._center(share)).max() <= 1e-12


def test_the_return_plays_each_fade_window_backwards(opened):
    session = opened("ZoomAndBack")
    scene = session.scene
    zoom = scene.zoom
    (neighbour_window,) = [
        window for family, _, window in zoom._fading if family[0] is scene.neighbour
    ]
    inside = scene.middle.inside
    shown = max(
        max(stops[0] + stops[1], default=0) for stops in scene.middle.inside_shown
    )
    steps = range(0, 2 * TICKS + 1)
    seen = _drawn(
        session,
        [Fraction(3) + Fraction(j, TICKS) for j in steps],
        lambda s: (_shows(inside) / shown, _shows(s.neighbour) / PARTIAL),
    )
    places = {"before": set(), "in": set(), "after": set()}
    for j, (inside_share, neighbour_share) in zip(steps, seen, strict=True):
        share = 1 - smooth(j / (2 * TICKS))
        assert inside_share == pytest.approx(_ramp(share, zoom._fade_in), abs=1e-9)
        assert neighbour_share == pytest.approx(
            1 - _ramp(share, neighbour_window), abs=1e-9
        )
        for name, window in (
            ("inside", zoom._fade_in),
            ("neighbour", neighbour_window),
        ):
            low, high = window
            place = "before" if share >= high else "after" if share <= low else "in"
            places[place].add(name)
    # Each fade read before, through and after its window.
    assert all(found == {"inside", "neighbour"} for found in places.values())


def test_the_return_restores_what_showed_before_the_move_in(opened):
    session = opened("ZoomAndBack")
    scene = session.scene
    (points, shown), (points_after, shown_after) = scene.before, scene.after
    # The frame where it began, the inside hidden, every piece as it was,
    # the neighbour's partial opacity included.
    assert np.array_equal(points_after, points)
    assert shown_after == shown
    assert _shows(scene.middle.inside) == 0
    assert _shows(scene.neighbour) == PARTIAL


def test_a_return_refuses_a_zoom_never_played():
    frame = ScreenRectangle(height=config.frame_height)
    back = ZoomOut(ZoomInto(Block("b"), frame=frame))
    with pytest.raises(ValueError, match="never played"):
        back.begin()


def test_a_return_refuses_a_frame_moved_from_where_the_zoom_left_it():
    zoom, frame = _played(Block("b", width=2.2, height=1.4).shift(RIGHT * 2))
    height = frame.height
    # Rounding is not a move.
    frame.shift(RIGHT * height * 1e-9)
    ZoomOut(zoom).begin()
    zoom.finish()
    frame.shift(RIGHT * height * 1e-4)
    with pytest.raises(ValueError, match="where its ZoomInto left it"):
        ZoomOut(zoom).begin()
    zoom.finish()
    frame.scale(1.001)
    with pytest.raises(ValueError, match="where its ZoomInto left it"):
        ZoomOut(zoom).begin()


def test_a_return_asks_for_a_zoom():
    with pytest.raises(TypeError, match="ZoomInto"):
        ZoomOut(Block("b"))


@pytest.mark.parametrize(
    ("scene", "named"),
    [
        ("ReturnAfterTheCameraMoved", "where its ZoomInto left it"),
        ("ReturnAfterAPieceLeft", "left the scene"),
        ("ReturnWithoutMovingCamera", "MovingCameraScene"),
        ("NestedOutOfOrder", "where its ZoomInto left it"),
    ],
)
def test_a_scene_s_return_is_refused_by_name(opened, scene, named):
    with pytest.raises(ValueError, match=named):
        opened(scene)


def test_nested_levels_come_back_in_order(opened):
    scene = opened("NestedInOrder").scene
    (points, shown), (points_after, shown_after) = scene.before, scene.after
    # Out of the deep block: back to where the outer zoom left everything.
    (inner_points, inner_shows), (inner_points_after, inner_shows_after) = (
        scene.inner_before,
        scene.inner_after,
    )
    assert np.array_equal(inner_points_after, inner_points)
    assert inner_shows_after == inner_shows
    assert np.array_equal(points_after, points)
    assert shown_after == shown


def test_a_return_under_declared_timing_runs_through_the_clock(opened):
    session = opened("DeclaredZoomAndBack")
    scene = session.scene
    assert id(scene) in hooks.BEFORE_ANIMATIONS
    assert {play.route for play in session.record.plays} == {"ticked"}
    (points, shown), (points_after, shown_after) = scene.before, scene.after
    assert np.array_equal(points_after, points)
    assert shown_after == shown
    assert any(family[0] is scene.picture for family, _, _ in scene.zoom._fading)
    steps = range(0, 2 * TICKS + 1)
    seen = _drawn(
        session,
        [Fraction(3) + Fraction(j, TICKS) for j in steps],
        lambda s: (_shows(s.picture), float(s.camera.frame.height)),
    )
    # The picture's drawing changes only at its boundaries, while it comes
    # back; the camera, undeclared, moves at every tick.
    by_drawing: dict[int, set[float]] = {}
    for j, (picture, _) in zip(steps, seen, strict=True):
        by_drawing.setdefault(j * DRAWINGS // TICKS, set()).add(picture)
    assert all(len(held) == 1 for held in by_drawing.values())
    assert len({next(iter(held)) for held in by_drawing.values()}) > 2
    heights = [height for _, height in seen]
    assert all(a != b for a, b in pairwise(heights))
    # Shown at the end as before the move-in.
    (end,) = _drawn(session, [Fraction(21, 4)], lambda s: shows(s.picture))
    assert end == scene.picture_before


def test_a_second_move_in_after_a_return_draws_like_the_first(opened):
    session = opened("ZoomInOutIn")
    inside = session.scene.middle.inside

    def read(s):
        return (
            float(s.camera.frame.height),
            tuple(s.camera.frame.get_center()),
            _shows(inside),
            _shows(s.neighbour),
        )

    steps = range(0, 2 * TICKS + 1)
    first = _drawn(session, [Fraction(j, TICKS) for j in steps], read)
    second = _drawn(session, [4 + Fraction(j, TICKS) for j in steps], read)
    assert second == first
    assert len(set(first)) > 10


def test_the_picture_after_the_return_is_the_picture_before_the_move_in(opened):
    session = opened("ZoomAndBack")
    before = session.frame(0.25).tobytes()
    assert session.frame(5.25).tobytes() == before
    assert session.frame(4).tobytes() != before


def test_a_return_refuses_a_move_in_that_never_reached_its_block():
    zoom, _ = _played(
        Block("b", width=2.2, height=1.4).shift(RIGHT * 2), rate_func=there_and_back
    )
    with pytest.raises(ValueError, match="never reached its block"):
        ZoomOut(zoom).begin()


def test_a_return_refuses_an_easing_that_does_not_end_where_it_must():
    zoom, _ = _played(Block("b", width=2.2, height=1.4).shift(RIGHT * 2))
    with pytest.raises(ValueError, match="rate_func"):
        ZoomOut(zoom, rate_func=there_and_back).begin()


def test_a_return_refuses_a_frame_stretched_from_where_the_zoom_left_it():
    zoom, frame = _played(Block("b", width=2.2, height=1.4).shift(RIGHT * 2))
    frame.stretch(1.01, 0)
    with pytest.raises(ValueError, match="where its ZoomInto left it"):
        ZoomOut(zoom).begin()


def test_a_return_refuses_an_outer_zoom_while_an_inner_one_holds_the_frame():
    block = Block("b", width=2.2, height=1.4).shift(RIGHT * 2)
    outer, frame = _played(block)
    # A second zoom onto the same block leaves the frame where the first did.
    inner = ZoomInto(block, frame=frame)
    inner.begin()
    inner.finish()
    with pytest.raises(ValueError, match="innermost"):
        ZoomOut(outer).begin()
    back = ZoomOut(inner)
    back.begin()
    back.finish()
    # Back where the outer zoom left it: the outer one comes back, once.
    ZoomOut(outer).begin()
    with pytest.raises(ValueError, match="already come back"):
        ZoomOut(outer).begin()
