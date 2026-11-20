"""A rig's parts move about their pivots through the parent chain, slide,
warp through their lattices and reshape through their stored outlines, each
checked against a computation made here from the part files; a value
outside its tested range, or a name the manifest does not have, is a
finding, at the instant a rate function carries a pose there too; a rig's
states are values the record keys apart; two poses of one rig in one play
each move their own parameters; a parameter or slot named as a pose's own
option, or both named alike, is refused with the manifest.
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
from .helpers import copied, edited, refused

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


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(scene: str) -> Session:
        session = Session.open(
            SCENES,
            scene,
            config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 30},
        )
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def test_a_pose_eased_past_its_target_inside_the_range_draws_the_value_it_gives(
    opened,
):
    session = opened("Overshooting")
    record = session.record
    box = next(m for m in session.scene.mobjects if isinstance(m, Rig))
    hinge = _pivot(box, "lid")
    rest = _rest(box, "lid")
    past = 0
    for capture in record.plays[0].samples:
        value = float(record.node_state(box, capture).get("_values")[0])
        lid = [
            np.asarray(record.node_state(m, capture).get("_points"))
            for m in box.part("lid").submobjects
        ]
        _close(lid, [_turned(points, hinge, value) for points in rest])
        past += value > 50
    # The rate function ran past the target, and the lid went with it.
    assert past > 3
    assert box.value("lid_open") == 50


def test_a_pose_eased_past_its_tested_range_is_refused_at_the_play():
    from arachne.utils.rate_functions import ease_in_back, ease_out_back

    box = Rig("fixture:lidded-box")
    eased = Pose(box, lid_open=110, rate_func=ease_out_back)
    eased.begin()
    found = refused("rig.range", lambda: eased.interpolate(0.8))
    assert found.detail["parameter"] == "lid_open"
    assert found.detail["value"] > 110
    assert (found.detail["min"], found.detail["max"]) == (0, 110)
    # The rig keeps the last value in range it was given.
    assert 0 <= box.value("lid_open") <= 110
    below = Rig("fixture:lidded-box").pose(lid_open=0)
    dipping = Pose(below, lid_open=60, rate_func=ease_in_back)
    dipping.begin()
    found = refused("rig.range", lambda: dipping.interpolate(0.1))
    assert found.detail["value"] < 0


@pytest.mark.parametrize("scene", ["TwoPoses", "TwoPosesReversed"])
def test_two_poses_of_one_rig_in_one_play_each_move_their_own_parameter(opened, scene):
    session = opened(scene)
    record = session.record
    figure = next(m for m in session.scene.mobjects if isinstance(m, Rig))
    reach, turn = figure.model.index["reach"], figure.model.index["head_turn"]
    _, middle = record.capture_at(Fraction(3, 10))
    values = record.node_state(figure, middle).get("_values")
    assert 0 < values[reach] < 1
    assert 0 < values[turn] < 25
    assert (figure.value("reach"), figure.value("head_turn")) == (1, 25)
    reference = Rig(figure.asset).pose(reach=1, head_turn=25)
    _close(_drawn(figure), _drawn(reference))


@pytest.mark.parametrize("option", ["run_time", "rate_func", "name"])
def test_a_parameter_or_slot_named_as_a_pose_option_is_refused(tmp_path, option):
    def parameter(document: dict) -> None:
        document["parameters"][0]["name"] = option

    def slot(document: dict) -> None:
        document["slots"][0]["name"] = option

    for edit, member in ((parameter, "parameters[0].name"), (slot, "slots[0].name")):
        directory = edited(copied(tmp_path / edit.__name__ / option, "dial"), edit)
        found = refused("asset.manifest_invalid", lambda d=directory: Rig(d))
        assert found.detail["member"] == member
        assert option in found.detail["reason"]


def test_a_slot_and_a_parameter_sharing_a_name_are_refused(tmp_path):
    def shared(document: dict) -> None:
        document["slots"][0]["name"] = "reading"

    directory = edited(copied(tmp_path, "dial"), shared)
    found = refused("asset.manifest_invalid", lambda: Rig(directory))
    assert found.detail["member"] == "slots[0].name"
    assert "reading" in found.detail["reason"]


def test_a_model_lives_as_long_as_its_asset():
    import gc
    import weakref

    from arachne.assets import load_asset
    from arachne.assets.pose import model_of

    asset = load_asset("fixture:dial")
    rig = Rig(asset)
    assert rig.model is model_of(asset)
    gone = weakref.ref(asset)
    model = weakref.ref(rig.model)
    del asset, rig
    gc.collect()
    assert gone() is None
    assert model() is None
