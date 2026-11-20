"""A rig's parts move about their pivots through the parent chain, slide,
warp through their lattices and reshape through their stored outlines, each
checked against a computation made here from the part files; a value
outside its tested range, or a name the manifest does not have, is a
finding; a rig's states are values the record keys apart.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.assets import Pose, Rig
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.record import RegisteredState

from . import rig_scenes
from .helpers import refused

SCENES = Path(rig_scenes.__file__)


def _scene_of(asset, points: np.ndarray) -> np.ndarray:
    size = asset.manifest.size
    (ox, oy), s = size.origin, size.scene_units_per_unit
    points = np.asarray(points, dtype=float)[:, :2]
    return np.column_stack(
        [(points[:, 0] - ox) * s, (oy - points[:, 1]) * s, np.zeros(len(points))]
    )


def _turned(points: np.ndarray, about: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    d = points - about
    return about + np.column_stack(
        [d[:, 0] * cos - d[:, 1] * sin, d[:, 0] * sin + d[:, 1] * cos, d[:, 2]]
    )


def _drawn(group) -> list[np.ndarray]:
    return [np.asarray(m.points) for m in group.family_members_with_points()]


def _rest(rig: Rig, name: str) -> list[np.ndarray]:
    return [_scene_of(rig.asset, s.points) for s in rig.asset.rest(name)]


def _pivot(rig: Rig, name: str) -> np.ndarray:
    spec = rig.asset.parts.get(name) or rig.asset.slots[name]
    return _scene_of(rig.asset, np.array([spec.pivot]))[0]


def _close(got: list[np.ndarray], want: list[np.ndarray]) -> None:
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("degrees", [0, 37.5, 110])
def test_a_part_turns_about_its_pivot(degrees):
    box = Rig("fixture:lidded-box").pose(lid_open=degrees)
    _close(_drawn(box.part("body")), _rest(box, "body"))
    hinge = _pivot(box, "lid")
    _close(
        _drawn(box.part("lid")),
        [_turned(points, hinge, degrees) for points in _rest(box, "lid")],
    )


def test_a_parameter_turns_its_part_and_slides_a_slot_down_the_chain():
    figure = Rig("fixture:figure").pose(reach=0.5)
    shoulder = _pivot(figure, "upper_arm")
    angle = 35  # half of the reach's 70
    for name in ("upper_arm", "lower_arm"):
        _close(
            _drawn(figure.part(name)),
            [_turned(p, shoulder, angle) for p in _rest(figure, name)],
        )
    # The hand slides by half its key in its own place, then rides the arm.
    slide = np.array([6.0, -2.0])
    hand = [
        _scene_of(figure.asset, np.asarray(s.points)[:, :2] + slide)
        for s in figure.asset.drawing("hand", "open")
    ]
    _close(_drawn(figure.part("hand")), [_turned(p, shoulder, angle) for p in hand])
    # The rest of the figure stays.
    _close(_drawn(figure.part("torso")), _rest(figure, "torso"))


def test_a_translate_slides_its_part():
    valve = Rig("fixture:gate-valve").pose(opening=0.25)
    up = np.array([0, 15 * valve.asset.manifest.size.scene_units_per_unit, 0])
    _close(_drawn(valve.part("stem")), [p + up for p in _rest(valve, "stem")])
    _close(_drawn(valve.part("body")), _rest(valve, "body"))


def _bilinear(point, offsets, bounds, cols, rows):
    x0, y0, x1, y1 = bounds
    u = min(max((point[0] - x0) / (x1 - x0) * cols, 0), cols)
    v = min(max((point[1] - y0) / (y1 - y0) * rows, 0), rows)
    i, j = min(int(u), cols - 1), min(int(v), rows - 1)
    fu, fv = u - i, v - j
    at = lambda c, r: np.asarray(offsets[r * (cols + 1) + c], dtype=float)  # noqa: E731
    return (
        (1 - fu) * (1 - fv) * at(i, j)
        + fu * (1 - fv) * at(i + 1, j)
        + (1 - fu) * fv * at(i, j + 1)
        + fu * fv * at(i + 1, j + 1)
    )


@pytest.mark.parametrize("turn", [-25, 12.5, 25])
def test_a_warp_moves_the_points_of_its_part_through_its_lattice(turn):
    figure = Rig("fixture:figure").pose(head_turn=turn)
    head = figure.asset.parts["head"]
    lattice = head.lattice
    (driver,) = figure.asset.parameters["head_turn"].drives
    keys = {key.at: np.asarray(key.value, dtype=float) for key in driver.keys}
    offsets = keys[turn] if turn in keys else keys[0] + (keys[25] - keys[0]) * turn / 25
    expected = []
    for shape in figure.asset.rest("head"):
        canvas = np.asarray(shape.points)[:, :2]
        moved = np.array(
            [
                p + _bilinear(p, offsets, lattice.bounds, lattice.cols, lattice.rows)
                for p in canvas
            ]
        )
        expected.append(_scene_of(figure.asset, moved))
    _close(_drawn(figure.part("head")), expected)


@pytest.mark.parametrize("swell", [0, 0.25, 0.5, 0.9, 1])
def test_a_shape_parameter_carries_its_part_through_its_stored_outlines(swell):
    section = Rig("fixture:layered-section").pose(swell=swell)
    (driver,) = section.asset.parameters["swell"].drives
    outlines = {
        key.at: [np.asarray(s.points)[:, :2] for s in section.asset.outline(key.value)]
        for key in driver.keys
    }
    ats = sorted(outlines)
    low = max(a for a in ats if a <= swell)
    high = min(a for a in ats if a >= swell)
    w = 0 if high == low else (swell - low) / (high - low)
    expected = [
        _scene_of(section.asset, (1 - w) * a + w * b)
        for a, b in zip(outlines[low], outlines[high], strict=True)
    ]
    _close(_drawn(section.part("inner_layer")), expected)
    # The layers about it keep their shapes.
    _close(_drawn(section.part("core")), _rest(section, "core"))


def test_every_pose_starts_from_rest():
    figure = Rig("fixture:figure")
    rest = _drawn(figure)
    figure.pose(reach=1, head_turn=25).pose(reach=0.3).pose(reach=0, head_turn=0)
    _close(_drawn(figure), rest)


def test_a_value_outside_its_tested_range_is_a_finding_with_its_range():
    box = Rig("fixture:lidded-box")
    for value in (-1, 110.5, math.nan, math.inf):
        found = refused("rig.range", lambda value=value: box.pose(lid_open=value))
        assert found.detail["parameter"] == "lid_open"
        assert (found.detail["min"], found.detail["max"]) == (0, 110)
    found = refused("rig.range", lambda: Pose(box, lid_open=120))
    assert found.detail["value"] == 120
    # Nothing was written.
    assert box.value("lid_open") == 0


@pytest.mark.parametrize(
    ("ask", "kind", "nearest"),
    [
        (lambda rig: rig.pose(rech=1), "parameter", ["reach"]),
        (lambda rig: rig.value("head_trun"), "parameter", ["head_turn"]),
        (lambda rig: rig.show(hnad="point"), "slot", ["hand"]),
        (lambda rig: rig.shown("eye"), "slot", ["eyes"]),
        (lambda rig: rig.show(hand="pointing"), "drawing", ["point"]),
        (lambda rig: Pose(rig, reech=1), "parameter", ["reach"]),
        (lambda rig: Pose(rig, hand="opn"), "drawing", ["open"]),
    ],
)
def test_a_name_the_manifest_does_not_have_is_a_finding_of_its_kind(ask, kind, nearest):
    found = refused("rig.name_unknown", lambda: ask(Rig("fixture:figure")))
    assert found.detail["kind"] == kind
    assert found.detail["nearest"] == nearest


@pytest.fixture
def states():
    admission.DEMOTED.clear()
    session = Session.open(
        SCENES,
        "States",
        config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 4},
    )
    yield session
    session.close()
    admission.DEMOTED.clear()


def test_two_rig_states_key_apart_and_an_earlier_capture_is_unchanged(states):
    record = states.record
    box = next(m for m in states.scene.mobjects if isinstance(m, Rig))
    _, during = record.capture_at(Fraction(1, 4))
    _, held = record.capture_at(Fraction(5, 8))
    _, after = record.capture_at(Fraction(7, 8))
    values = [record.node_state(box, c).get("_values") for c in (during, held, after)]
    assert [float(v[0]) for v in values] == [pytest.approx(20), 40, 90]
    # The pose between the plays replaced the values: the capture before it
    # holds what it held.
    assert values[1] is not values[2]
    keys = [RegisteredState(record, c).content_key() for c in (during, held, after)]
    assert len(set(keys)) == 3
    lid = box.part("lid").submobjects[0]
    assert not np.array_equal(
        record.node_state(lid, held).get("_points"),
        record.node_state(lid, after).get("_points"),
    )
