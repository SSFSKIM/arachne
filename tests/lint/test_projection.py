"""The screen projection: an object's projected box is where a render draws
it, under every camera, and ``state_at`` reads its boxes through it.

Each probe renders a scene showing one object alone, and holds the drawn
pixels' box against the box the projection gives for the object from the
session's record, within a pixel.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Circle
from arachne.camera.projection import (
    View,
    draw_kind,
    pixel_box,
    record_view,
    render_mapping_centre,
    screen_box,
    view_of,
)
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.draw import Drawing
from arachne.timeline.source import SourceLine
from tests.timeline import scenes as timeline_scenes
from tests.timeline.conftest import render

from . import projection_scenes, shapes

SCENES = Path(projection_scenes.__file__)
FPS = 4
#: Large enough that a pixel is a thirtieth of a scene unit.
SIZE = {"pixel_width": 480, "pixel_height": 270}

#: (scene, frame, the scene's name for the one object the frame shows).
PROBES = [
    ("Flat", 1, "circle"),
    ("Flat", 3, "square"),
    ("Flat", 5, "label"),
    ("MovingBefore", 1, "circle"),
    ("MovingBefore", 3, "cloud"),
    ("MovingBefore", 5, "image"),
    ("MovingDuring", 1, "circle"),
    ("MovingDuring", 2, "circle"),
    ("MovingDuring", 3, "circle"),
    ("MovingDuring", 5, "circle"),
    # The centre set before the first frame: a path's mapping freezes it.
    ("ThreeDCentred", 1, "path"),
    ("ThreeDCentred", 3, "cloud"),
    ("ThreeDCentred", 5, "image"),
    ("ThreeDCentred", 7, "fixed"),
    ("ThreeDCentred", 9, "oriented"),
    # The centre moved after the first frame: a path keeps the frozen one.
    ("ThreeDCentred", 11, "path"),
    ("ThreeDCentred", 13, "cloud"),
    ("ThreeDCentred", 15, "image"),
    ("ThreeDCentred", 17, "fixed"),
    ("ThreeDCentred", 19, "oriented"),
    ("ThreeDMovedLater", 1, "path"),
    ("ThreeDMovedLater", 3, "path"),
    ("ThreeDMovedLater", 5, "cloud"),
    ("ThreeDMovedLater", 7, "image"),
]


class _Probes:
    """Each probe scene's render and session, made once for the module."""

    def __init__(self) -> None:
        self._frames: dict[str, list[np.ndarray]] = {}
        self._sessions: dict[str, Session] = {}

    def frames(self, name: str) -> list[np.ndarray]:
        if name not in self._frames:
            scene_class = getattr(projection_scenes, name)
            self._frames[name] = render(scene_class, FPS, **SIZE)[1]
        return self._frames[name]

    def session(self, name: str, path: Path = SCENES) -> Session:
        if name not in self._sessions:
            self._sessions[name] = Session.open(
                path, name, config={**SIZE, "frame_rate": FPS}
            )
        return self._sessions[name]

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()


@pytest.fixture(scope="module")
def probes() -> Iterator[_Probes]:
    admission.DEMOTED.clear()
    found = _Probes()
    yield found
    found.close()
    admission.DEMOTED.clear()


def _drawn_box(frame: np.ndarray) -> tuple[int, int, int, int]:
    """The box of the frame's pixels that are not the black background, as
    pixel edges: a pixel ``i`` spans ``[i, i + 1)``.
    """
    rows, columns = np.nonzero(frame[..., :3].max(axis=-1) > 0)
    assert rows.size, "the frame shows nothing"
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _projected(
    session: Session, t: Fraction, name: str, *, ink: bool
) -> tuple[tuple[float, float, float, float], View]:
    """The pixel box the projection gives the mobject the scene named
    ``name`` at ``t``, and the view it was mapped with.
    """
    record = session.record
    play, capture = record.capture_at(t)
    assert play is not None
    with Drawing(record) as drawing:
        drawing.show(capture, capture.draw_list())
        view = record_view(record)
        camera = record._scene.renderer.camera
        (mob,) = [
            mob
            for mob, _ in record._walk(capture, capture.draw_list())
            if play.names.get(id(mob)) == name
        ]
        box = pixel_box(camera, view, mob, ink=ink)
    assert box is not None
    return box, view


@pytest.mark.parametrize(("scene", "frame", "name"), PROBES)
def test_an_object_is_drawn_where_its_projected_box_says(probes, scene, frame, name):
    drawn = _drawn_box(probes.frames(scene)[frame])
    session = probes.session(scene)
    t = Fraction(frame, FPS)
    predicted, view = _projected(session, t, name, ink=True)
    assert max(abs(a - b) for a, b in zip(drawn, predicted, strict=True)) <= 1, (
        drawn,
        predicted,
    )
    # state_at's box is the same projection, in home-frame units.
    points_box, _ = _projected(session, t, name, ink=False)
    assert session.state_at(t).named(name).box == pytest.approx(
        screen_box(view, points_box), abs=1e-9
    )


def test_the_mapping_centre_is_the_frame_centre_at_the_first_path_drawn(probes):
    assert render_mapping_centre(probes.session("ThreeDCentred").record) == (
        1.0,
        0.5,
        0.0,
    )
    assert render_mapping_centre(probes.session("ThreeDMovedLater").record) == (
        0.0,
        0.0,
        0.0,
    )
    assert render_mapping_centre(probes.session("Flat").record) == (0.0, 0.0, 0.0)
    # A MovingCamera maps every draw around its frame as it is then.
    assert render_mapping_centre(probes.session("MovingBefore").record) is None


def test_a_live_camera_maps_paths_around_the_centre_its_context_froze():
    """After a render, the camera keeps the context its first path drawn
    made: its view maps paths around that centre, and point clouds and
    images around the frame's centre as it is now.
    """
    scene, _ = render(projection_scenes.ThreeDCentred, FPS, **SIZE)
    view = view_of(scene.renderer.camera)
    assert view.mapping_centre == pytest.approx((1, 0.5, 0), abs=1e-9)
    assert view.frame_center == pytest.approx((-1, -0.5, 0.5), abs=1e-9)
    assert view.camera == "three_d"
    assert view.three_d is not None
    assert view.three_d.phi == pytest.approx(np.pi / 3)


def test_state_at_under_a_moving_camera_follows_the_view(probes):
    """The circle stands still in the scene while the camera pans and
    zooms: its box moves on the picture, by ``(X - cx) * W0 / W``.
    """
    session = probes.session("MovingDuring")
    _, view = _projected(session, Fraction(0), "circle", ink=False)
    home_width = view.home_width
    scene_box = (0.2, -0.8, 1.8, 0.8)

    def expected(centre, width):
        scale = home_width / width
        return (
            (scene_box[0] - centre[0]) * scale,
            (scene_box[1] - centre[1]) * scale,
            (scene_box[2] - centre[0]) * scale,
            (scene_box[3] - centre[1]) * scale,
        )

    start = session.state_at(0).named("circle").box
    end = session.state_at(Fraction(5, 4)).named("circle").box
    assert start == pytest.approx(scene_box, abs=1e-9)
    assert end == pytest.approx(expected((1.5, 0.5), 9), abs=1e-9)


def test_under_the_flat_camera_home_units_are_scene_units_exactly(probes):
    state = probes.session("Flat").state_at(Fraction(3, 4))
    assert state.named("square").box == (-3.0, -2.0, -1.0, 0.0)
    circle = Circle(radius=1).move_to((2, 1, 0)).points
    first = probes.session("Flat").state_at(Fraction(1, 4)).named("circle")
    assert first.box == (
        float(circle[:, 0].min()),
        float(circle[:, 1].min()),
        float(circle[:, 0].max()),
        float(circle[:, 1].max()),
    )


def _drawn_x(frame: np.ndarray) -> tuple[int, int]:
    box = _drawn_box(frame)
    return box[0], box[2]


def test_a_fixed_orientation_label_projects_from_its_recorded_centre(probes):
    """The scene rebinds the variable the dot's centre function reads between
    its two waits: each wait's box is where that wait's frame draws the dot,
    from the centre the forward run recorded, not the live binding.
    """
    path = Path(timeline_scenes.__file__)
    session = probes.session("FixedOrientation", path)
    frames = render(timeline_scenes.FixedOrientation, FPS, **SIZE)[1]
    for k in (1, 3):
        t = Fraction(k, FPS)
        box = session.state_at(t).named("dot").box
        record = session.record
        play, capture = record.capture_at(t)
        with Drawing(record) as drawing:
            drawing.show(capture, capture.draw_list())
            view = record_view(record)
        left, right = _pixel_x(view, box[0]), _pixel_x(view, box[2])
        drawn_left, drawn_right = _drawn_x(frames[k])
        assert abs(left - drawn_left) <= 1, (k, left, drawn_left)
        assert abs(right - drawn_right) <= 1, (k, right, drawn_right)
    first = session.state_at(Fraction(1, FPS)).named("dot").box
    second = session.state_at(Fraction(3, FPS)).named("dot").box
    assert abs(first[0] - second[0]) > 1


def _pixel_x(view: View, x: float) -> float:
    return x * view.pixel_width / view.home_width + view.pixel_width / 2


def test_draw_kinds():
    from arachne import Dot, ImageMobject, PMobject, VGroup

    assert draw_kind(Dot()) == "path"
    assert draw_kind(PMobject()) == "points"
    assert draw_kind(ImageMobject(np.zeros((2, 2, 4), dtype=np.uint8))) == "image"
    assert draw_kind(VGroup()) == "path"


def test_the_scene_line_of_a_mobject_made_in_a_helper_module(probes):
    record = probes.session("HelperMade").record
    play, capture = record.capture_at(Fraction(1, FPS))
    assert play is not None
    (square,) = [
        mob
        for mob, _ in record._walk(capture, capture.draw_list())
        if play.names.get(id(mob)) == "square"
    ]
    origin = record.origin(square)
    helper = Path(shapes.__file__).read_text().splitlines()
    helper_line = next(
        number
        for number, line in enumerate(helper, start=1)
        if "Square(" in line and "return" in line
    )
    assert origin.source_line == SourceLine("shapes.py", helper_line)
    assert origin.scene_line == _tagged("helper-call")


def _tagged(tag: str) -> int:
    for number, line in enumerate(SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)
