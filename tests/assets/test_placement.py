"""Where a rig is placed, kept apart from its pose: a whole-rig move moves
its carrier and a part's own move does not, a one-part rig and a rig that is
one straight line are placed alike, a rig's every measure is its art's, the
carrier taking no space, and a pose and a whole-rig move in one
play give, at every tick and at the end, the pose of the interpolated values
in the interpolated place, whichever comes first.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import DL, RIGHT, UR, Square, VGroup
from arachne.assets import Rig
from arachne.assets.pose import CARRIER_WIDTH
from arachne.session import Session
from arachne.timeline import admission
from arachne.utils.rate_functions import smooth

from . import draw_fixture_pack, rig_scenes
from .helpers import admitted
from .rig_scenes import MOVE, OPEN

SCENES = Path(rig_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54}

ROD_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 100">
  <path d="M 10 50 L 110 50" fill="none" stroke="#3b4a5c" stroke-width="3"/>
</svg>
"""
ROD_MANIFEST = """\
{
  "manifest_version": "1.0", "name": "rod", "version": 1, "kind": "parts",
  "size": {"scene_units_per_unit": 0.02, "origin": [60, 50], "width": 2, "height": 0},
  "parts": [{"name": "rod", "file": "rod.svg", "parent": null, "pivot": [10, 50],
             "order": 0}],
  "parameters": [
    {"name": "tilt", "min": 0, "max": 90, "default": 0, "unit": "deg",
     "drives": [{"part": "rod", "kind": "rotate", "unit": "deg",
                 "keys": [{"at": 0, "value": 0}, {"at": 90, "value": 90}]}]},
    {"name": "grow", "min": 1, "max": 3, "default": 1,
     "drives": [{"part": "rod", "kind": "scale",
                 "keys": [{"at": 1, "value": 1}, {"at": 3, "value": 2}]}]}
  ],
  "provenance": {"source_url": null, "author": "a test", "license": "none",
                 "credit_line": null, "retrieved_at": null}
}
"""


@pytest.fixture
def rod(tmp_path: Path) -> Path:
    """An asset of one part, which is one straight line."""
    directory = tmp_path / "rod"
    directory.mkdir()
    (directory / "rod.svg").write_text(ROD_SVG)
    (directory / "manifest.json").write_text(ROD_MANIFEST)
    return admitted(directory)


def _scene_of(asset, points: np.ndarray) -> np.ndarray:
    """Canvas points where the asset's manifest places them in the scene."""
    size = asset.manifest.size
    (ox, oy), s = size.origin, size.scene_units_per_unit
    return np.column_stack(
        [(points[:, 0] - ox) * s, (oy - points[:, 1]) * s, np.zeros(len(points))]
    )


def _turned(points: np.ndarray, about: np.ndarray, degrees: float) -> np.ndarray:
    """``points`` turned counterclockwise in the scene's plane about ``about``."""
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    d = points - about
    return about + np.column_stack(
        [d[:, 0] * cos - d[:, 1] * sin, d[:, 0] * sin + d[:, 1] * cos, d[:, 2]]
    )


def _art(rig: Rig) -> np.ndarray:
    return np.concatenate([np.asarray(m.points) for m in rig.part("rod").submobjects])


def _assert_measured_as_art(rig: Rig) -> None:
    """Every measure of ``rig`` is its art's, wherever its carrier stands."""
    art = _art(rig)
    low, high = art.min(axis=0), art.max(axis=0)
    assert rig.width == pytest.approx(high[0] - low[0], abs=1e-12)
    assert rig.height == pytest.approx(high[1] - low[1], abs=1e-12)
    np.testing.assert_allclose(rig.get_center(), (low + high) / 2, atol=1e-12)
    np.testing.assert_allclose(rig.get_corner(UR), high, atol=1e-12)
    np.testing.assert_allclose(rig.get_critical_point(DL), low, atol=1e-12)
    # Placed beside something, and centred, by its art.
    square = Square()
    rig.next_to(square, RIGHT, buff=0.5)
    assert _art(rig)[:, 0].min() == pytest.approx(square.get_right()[0] + 0.5)
    rig.center()
    art = _art(rig)
    np.testing.assert_allclose((art.min(axis=0) + art.max(axis=0)) / 2, 0, atol=1e-12)
    # The same in a group that holds it.
    group = VGroup(rig)
    assert group.width == pytest.approx(rig.width, abs=1e-12)
    np.testing.assert_allclose(group.get_center(), rig.get_center(), atol=1e-12)


def test_a_one_part_rig_that_is_one_straight_line_poses_where_it_is_placed(rod):
    rig = Rig(rod)
    line = _scene_of(rig.asset, np.asarray(rig.asset.rest("rod")[0].points)[:, :2])
    pivot = _scene_of(rig.asset, np.array([[10.0, 50.0]]))[0]
    np.testing.assert_allclose(_art(rig), line, atol=1e-12)
    # The box is the line's: the carrier takes no space.
    assert rig.height == 0
    rig.pose(tilt=90)
    np.testing.assert_allclose(_art(rig), _turned(line, pivot, 90), atol=1e-12)
    _assert_measured_as_art(Rig(rod).pose(tilt=90))
    rig.pose(tilt=0, grow=2)
    grown = pivot + (line - pivot) * 1.5
    np.testing.assert_allclose(_art(rig), grown, atol=1e-12)
    rig.rotate(math.pi / 3, about_point=np.zeros(3)).shift(RIGHT)
    rig.pose(tilt=30)
    expected = _turned(_turned(grown, pivot, 30), np.zeros(3), 60) + RIGHT
    np.testing.assert_allclose(_art(rig), expected, atol=1e-9)


def test_a_whole_rig_move_is_kept_by_a_pose_and_a_parts_own_move_is_not(rod):
    moved, part_moved = Rig(rod), Rig(rod)
    v = np.array([0.7, -0.4, 0.0])
    moved.shift(v)
    part_moved.part("rod").shift(v)
    # The art is the same; only the carriers tell the two apart.
    np.testing.assert_allclose(_art(moved), _art(part_moved), atol=1e-12)
    carriers = [
        np.array([p.points[0] for p in rig.carrier.submobjects])
        for rig in (moved, part_moved)
    ]
    assert not np.allclose(*carriers)
    reference = Rig(rod).pose(tilt=45)
    moved.pose(tilt=45)
    part_moved.pose(tilt=45)
    np.testing.assert_allclose(_art(moved), _art(reference) + v, atol=1e-12)
    np.testing.assert_allclose(_art(part_moved), _art(reference), atol=1e-12)


def test_a_part_moved_directly_returns_at_the_next_settle(rod):
    rig = Rig(rod)
    rest = _art(rig)
    rig.part("rod").shift(RIGHT)
    rig.update(0)
    np.testing.assert_allclose(_art(rig), rest, atol=1e-12)


def test_a_rig_scaled_to_nothing_is_not_settled_and_cannot_be_posed():
    from arachne.assets import Pose

    rig = Rig("fixture:lidded-box").scale(0)
    before = [np.asarray(m.points) for m in rig.family_members_with_points()]
    rig.pose(lid_open=50)
    after = [np.asarray(m.points) for m in rig.family_members_with_points()]
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert rig.value("lid_open") == 50
    with pytest.raises(ValueError, match="scaled to nothing"):
        Pose(rig, lid_open=60)


def test_a_rig_transformed_into_another_shape_is_refused_at_its_next_settle():
    from arachne import Transform

    rig = Rig("fixture:lidded-box")
    transform = Transform(rig, Square())
    transform.begin()
    transform.interpolate(1)
    # Its updating resumes as the transform finishes, and the settle refuses.
    with pytest.raises(ValueError, match="changed into something its asset"):
        transform.finish()
    with pytest.raises(ValueError, match="changed into something its asset"):
        rig.pose(lid_open=20)


@pytest.mark.parametrize("name", draw_fixture_pack.NAMES)
def test_the_box_of_a_rig_is_its_arts_in_every_pose(name):
    rig = Rig(f"fixture:{name}")
    poses: list[dict] = [{}]
    for parameter in rig.asset.manifest.parameters:
        middle = (parameter.min + parameter.max) / 2
        poses += [{parameter.name: v} for v in (parameter.min, middle, parameter.max)]
    for slot in rig.asset.manifest.slots:
        poses += [{slot.name: d.name} for d in slot.drawings]
    slots = {slot.name for slot in rig.asset.manifest.slots}
    for pose in poses:
        rig.pose(**{k: v for k, v in pose.items() if k not in slots})
        rig.show(**{k: v for k, v in pose.items() if k in slots})
        art = np.concatenate(
            [
                np.asarray(m.points)
                for m in rig.family_members_with_points()
                if m not in rig.carrier.submobjects
            ]
        )
        whole = np.concatenate(
            [np.asarray(m.points) for m in rig.family_members_with_points()]
        )
        for corner in (0, 1):
            pick = np.min if corner == 0 else np.max
            np.testing.assert_allclose(
                pick(whole, axis=0), pick(art, axis=0), atol=CARRIER_WIDTH
            )


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(scene: str, fps: int) -> Session:
        session = Session.open(SCENES, scene, config={**SMALL, "frame_rate": fps})
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _rig(session: Session, name: str) -> Rig:
    return next(
        m
        for m in session.scene.get_mobject_family_members()
        if isinstance(m, Rig) and m.name == name
    )


def _expected_box(rig: Rig, degrees: float, shift: np.ndarray) -> list[np.ndarray]:
    """Each shape of the box, the lid turned by ``degrees`` about its hinge,
    all moved by ``shift``: computed here from the part files alone.
    """
    asset = rig.asset
    hinge = _scene_of(asset, np.array([asset.parts["lid"].pivot]))[0]
    body = [_scene_of(asset, np.asarray(s.points)[:, :2]) for s in asset.rest("body")]
    lid = [
        _turned(_scene_of(asset, np.asarray(s.points)[:, :2]), hinge, degrees)
        for s in asset.rest("lid")
    ]
    return [points + shift for points in (*body, *lid)]


def _recorded_box(session: Session, rig: Rig, t: Fraction) -> list[np.ndarray]:
    record = session.record
    _, capture = record.capture_at(t)
    shapes = [*rig.part("body").submobjects, *rig.part("lid").submobjects]
    return [np.asarray(record.node_state(s, capture).get("_points")) for s in shapes]


@pytest.mark.parametrize("scene", ["PoseThenMove", "MoveThenPose"])
def test_a_pose_and_a_whole_rig_move_in_one_play_compose_at_every_tick(opened, scene):
    session = opened(scene, 60)
    play = session.record.plays[0]
    assert play.route == "ticked"
    rig = _rig(session, "box")
    instants = [Fraction(k, 60) for k in range(60)]
    assert [capture.instant for capture in play.samples][: len(instants)] == instants
    for t in instants:
        progress = smooth(float(t))
        expected = _expected_box(rig, OPEN * progress, MOVE * progress)
        for got, want in zip(_recorded_box(session, rig, t), expected, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-9, err_msg=f"at {t}")
    # At the end, the targets themselves, and the rig keeps them through the wait.
    for t in (Fraction(1), Fraction(5, 4)):
        expected = _expected_box(rig, OPEN, MOVE)
        for got, want in zip(_recorded_box(session, rig, t), expected, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-9, err_msg=f"at {t}")
    assert rig.value("lid_open") == OPEN
    # The scene's settle left its updaters with the play.
    assert session.scene.updaters == []
    wait = session.record.plays[1]
    assert wait.kind == "wait"
    assert wait.held


def test_a_pose_draws_alike_at_every_shared_frame_instant_of_three_rates(opened):
    by_rate = {fps: opened("PoseThenMove", fps) for fps in (15, 30, 60)}
    rigs = {fps: _rig(session, "box") for fps, session in by_rate.items()}
    for k in range(16):
        t = Fraction(k, 15)
        drawn = [_recorded_box(by_rate[fps], rigs[fps], t) for fps in (15, 30, 60)]
        for fifteen, thirty, sixty in zip(*drawn, strict=True):
            assert np.array_equal(fifteen, thirty), f"at {t}"
            assert np.array_equal(fifteen, sixty), f"at {t}"


def test_a_pose_beside_a_flow_leaves_the_flows_driver_running(opened):
    from arachne.mobject.diagram import flow_driver

    session = opened("PoseBesideFlow", 15)
    scene = session.scene
    driver = flow_driver(scene)
    assert driver is not None
    assert scene.updaters == [driver.updater]
    rig = _rig(session, "box")
    expected = _expected_box(rig, OPEN, np.zeros(3))
    for got, want in zip(
        _recorded_box(session, rig, Fraction(1)), expected, strict=True
    ):
        np.testing.assert_allclose(got, want, atol=1e-9)
    (flow,) = driver.flows
    assert flow.elapsed == pytest.approx(1.5)


def test_the_lint_reads_a_rigs_art_and_not_its_carrier(opened):
    from arachne.lint.parts import build_parts

    session = opened("MoveThenPose", 4)
    record = session.record
    box = _rig(session, "box")
    shapes = {id(m) for m in box.family_members_with_points()} - {
        id(point) for point in box.carrier.submobjects
    }
    parts = build_parts(
        record,
        record.duration,
        None,
        record.final,
        answer="recorded",
        target=(96, 54),
    )
    leaves = [leaf for part in parts.parts for leaf in part.leaves]
    assert leaves
    assert all(leaf.type != "CarrierPoint" for leaf in leaves)
    assert len(leaves) == len(shapes)


ROD_SCENE = """\
from arachne import *


class Tilted(Scene):
    def construct(self):
        rod = Rig({rod!r}, name="rod").pose(tilt=90)
        self.add(rod)
        self.wait(0.5)
"""


def test_the_lint_boxes_a_rig_by_its_art(rod, tmp_path):
    scene = tmp_path / "tilted.py"
    scene.write_text(ROD_SCENE.format(rod=str(rod)))
    admission.DEMOTED.clear()
    session = Session.open(
        scene, "Tilted", config={**SMALL, "frame_rate": 4, "media_dir": str(tmp_path)}
    )
    try:
        rig = _rig(session, "rod")
        art = _art(rig)
        node_id = session.record.origin(rig).id
        (node,) = [n for n in session.state_at(Fraction(1, 4)).nodes if n.id == node_id]
        height = art[:, 1].max() - art[:, 1].min()
        # The line stands upright: every box of it is the line's, as thin as
        # it is, the lint's query and the state's alike.
        for x0, y0, x1, y1 in (
            session.query(Fraction(1, 4)).bbox(node_id),
            node.box,
            node.scene_box,
        ):
            assert x1 - x0 == pytest.approx(0, abs=1e-9)
            assert y1 - y0 == pytest.approx(height, abs=1e-9)
        x0, _, x1, _ = node.screen_box
        assert x1 - x0 == pytest.approx(0, abs=1e-6)
    finally:
        session.close()
        admission.DEMOTED.clear()
