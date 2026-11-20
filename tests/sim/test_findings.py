"""A simulated body's quantities, read from its world's bake, and what the
world's solver found: each finding reported by the lint at its tick, about
its bodies with their boxes, with a clean twin that reports nothing.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("arachne_sim")

from jsonschema import Draft202012Validator  # noqa: E402

from arachne import World, api_json  # noqa: E402
from arachne.lint import bakes as bakes_module  # noqa: E402
from arachne.lint import frames as frames_module  # noqa: E402
from arachne.lint import lint  # noqa: E402
from arachne.session import Session  # noqa: E402
from arachne.timeline import admission  # noqa: E402

from . import finding_scenes, world_scenes  # noqa: E402

SCENES = Path(finding_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}
G = 9.8
TICK = 1 / 60


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, module=finding_scenes) -> Session:
        session = Session.open(Path(module.__file__), name, config=SMALL)
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _valid(form: str, out: dict) -> dict:
    read = json.loads(json.dumps(out, allow_nan=False))
    validator = Draft202012Validator(api_json.schema(form))
    errors = [f"{list(e.path)}: {e.message}" for e in validator.iter_errors(read)]
    assert not errors, errors
    return read


def _bodies(session: Session) -> list[str]:
    """The stable ids of the first world's bodies, by their number."""
    (occurrence, *_) = session.record.bakes.occurrences
    return [b.id for b in occurrence.bindings if b.channel == "pose"]


def _line(tag: str) -> int:
    for number, line in enumerate(SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def _simulation(session: Session, **options):
    result = lint(session, rules=(), **options)
    _valid("lint_result", api_json.lint(result))
    return result, [f for f in result.findings if f.code.startswith("sim.")]


# The quantities


def _read(state, node_id: str) -> dict:
    return {q.name: q for q in state.quantities[node_id]}


def test_a_falling_body_s_quantities_are_its_bake_s_row(opened):
    session = opened("Dropped")
    floor, ball, _ = _bodies(session)
    bakes = session.record.bakes
    pose = bakes.contents[bakes.occurrences[0].content].channels["pose"].values
    # A frame of the 15 fps grid: what it shows is of its own instant, tick 12.
    t = Fraction(1, 5)
    read = _read(session.state_at(t), ball)
    units = {name: q.unit for name, q in read.items()}
    assert units == {
        "sim.position": "u",
        "sim.velocity": "u/s",
        "sim.angular_velocity": "rad/s",
        "sim.contact_times": "s",
        "sim.contact_count": "1",
    }
    assert all(q.unreadable is None for q in read.values())
    # Exact as baked, at tick 12: z is 0 in a plane.
    x, y = pose[12, 1, :2].astype(float)
    assert read["sim.position"].value == (x, y, 0.0)
    # The closed form: v = -g t, which the tick's integration keeps exactly.
    vx, vy, vz = read["sim.velocity"].value
    assert (vx, vz) == (0.0, 0.0)
    assert vy == pytest.approx(-G * 0.2, abs=1e-3)
    assert read["sim.angular_velocity"].value == 0.0
    assert read["sim.contact_times"].value == ()
    assert read["sim.contact_count"].value == 0
    assert read["sim.contact_times"].tolerance == pytest.approx(TICK)


def test_a_bouncing_body_s_contact_times_are_the_closed_form_s_within_a_tick(opened):
    session = opened("Dropped")
    floor, first, second = _bodies(session)
    state = session.state_at(Fraction(7, 10))
    landed = [(2 * drop / G) ** 0.5 for drop in finding_scenes.DROPS]
    on_floor = _read(state, floor)
    assert on_floor["sim.contact_count"].value == 2
    times = on_floor["sim.contact_times"].value
    assert len(times) == 2
    for found, expected in zip(times, landed, strict=True):
        assert abs(found - expected) <= TICK + 1e-9, (found, expected)
    for ball, expected in ((first, landed[0]), (second, landed[1])):
        read = _read(state, ball)
        assert read["sim.contact_count"].value == 1
        (found,) = read["sim.contact_times"].value
        assert abs(found - expected) <= TICK + 1e-9, (found, expected)
    # Each is the instant of the tick its contact began at, as baked.
    bakes = session.record.bakes
    (occurrence,) = bakes.occurrences
    events = bakes.events(occurrence.ordinal, "contact")
    began = [
        float(Fraction(occurrence.first_tick + int(row), 60))
        for row, a, b, kind in events.tolist()
        if kind == 1 and floor in (_bodies(session)[a], _bodies(session)[b])
    ]
    assert list(times) == began
    # Read while the stay was open, from what computed it: the same table,
    # read-only.
    live, writeable = session.module.FOUND["live contacts"]
    assert live == events.tolist()
    assert not writeable
    # Before the second ball lands, the floor has met only the first.
    early = _read(session.state_at(Fraction(1, 2)), floor)
    assert early["sim.contact_count"].value == 1
    out = _valid("state", api_json.state(state, session))
    (entry,) = [e for e in out["quantities"][floor] if e["name"] == "sim.contact_times"]
    assert entry["value"] == list(times)


def test_a_body_in_three_dimensions_turns_about_three_axes(opened):
    session = opened("Tumbling", module=world_scenes)
    (body,) = _bodies(session)
    read = _read(session.state_at(0), body)
    assert read["sim.angular_velocity"].value == pytest.approx(
        (0.3, 1.1, 0.7), abs=1e-6
    )
    assert read["sim.velocity"].value == pytest.approx((0.5, 0.0, 0.0), abs=1e-6)
    assert len(read["sim.position"].value) == 3


def test_a_body_held_on_screen_after_its_world_left_reads_as_of_its_capture(opened):
    """A quantity is read as of its capture. A wait over a body left unmoving
    after its world left at 1/2 s is held as the one capture of 1/2 s, whose
    tick, 30, is the stay's last (inclusive): it reads that row throughout.
    (An open world removed before its callback ran at that tick would hold
    only the tick before it.)
    """
    session = opened("HeldAfterItsWorld")
    (body,) = _bodies(session)
    bakes = session.record.bakes
    (occurrence,) = bakes.occurrences
    assert occurrence.last_tick == 30
    channels = bakes.contents[occurrence.content].channels
    state = session.state_at(Fraction(3, 4))
    assert state.as_of == Fraction(1, 2)
    read = _read(state, body)
    assert all(q.unreadable is None for q in read.values())
    x, y = channels["pose"].values[30, 0, :2].astype(float)
    vx, vy = channels["velocity"].values[30, 0, :2].astype(float)
    assert read["sim.position"].value == (x, y, 0.0)
    assert read["sim.velocity"].value == (vx, vy, 0.0)
    assert read["sim.angular_velocity"].value == 0.0
    assert read["sim.contact_times"].value == ()
    assert read["sim.contact_count"].value == 0
    assert read == _read(session.state_at(Fraction(1, 2)), body)


def test_a_body_shown_after_its_world_left_is_unreadable(opened):
    session = opened("ShownAfterItsWorld")
    (body,) = _bodies(session)
    # Moved, each frame is captured at its own instant, past the stay's last
    # tick, 30: every quantity names the tick it is read at.
    state = session.state_at(Fraction(3, 4))
    tick = math.floor(state.as_of * 60)
    assert tick > 30
    quantities = state.quantities[body]
    assert len(quantities) == 5
    for quantity in quantities:
        assert quantity.value is None
        assert quantity.unreadable == f"no bake holds tick {tick} of this body's stay"
    result, found = _simulation(session)
    assert {f.code for f in found} == {"sim.quantity_unreadable"}
    assert all(f.objects[0].id == body for f in found)
    assert min(f.t_start for f in found) > Fraction(1, 2)
    twin = opened("ShownWithItsWorld")
    assert all(
        q.unreadable is None
        for q in twin.state_at(Fraction(3, 4)).quantities[_bodies(twin)[0]]
    )
    assert _simulation(twin)[1] == []


# The findings


def test_a_world_gone_non_finite_is_unstable_and_its_twin_is_not(opened):
    session = opened("Collapsing")
    bodies = _bodies(session)
    result, found = _simulation(session)
    assert not result.passed
    assert sorted((f.code, f.objects[0].id) for f in found) == sorted(
        ("sim.unstable", body) for body in bodies
    )
    state = session.state_at(0)
    for finding in found:
        assert (finding.t_start, finding.t_end) == (Fraction(1, 60),) * 2
        assert finding.severity == "error"
        assert finding.detail == {
            "tick": 1,
            "signal": "non_finite",
            "value": None,
            "limit": None,
        }
        assert finding.message.endswith("went non-finite at tick 1")
        assert "finite" in finding.fix
        assert finding.line is not None
        # Boxed as the frame on screen at its tick shows it: the first one.
        (thing,) = finding.objects
        node = state.node(thing.id)
        assert thing.bbox == pytest.approx(node.scene_box, abs=1e-9)
        assert thing.screen_bbox == pytest.approx(node.screen_box, abs=1e-6)
        assert thing.source_line is not None
    assert _simulation(opened("Attracting"))[1] == []


def test_a_body_pushed_past_its_world_s_limit_is_unstable_and_its_twin_is_not(
    opened,
):
    session = opened("Speeding")
    (body,) = _bodies(session)
    _, (finding,) = _simulation(session)
    assert (finding.code, finding.objects[0].id) == ("sim.unstable", body)
    assert finding.t_start == Fraction(31, 60)
    assert finding.detail["tick"] == 31
    assert finding.detail["signal"] == "speed"
    assert finding.detail["value"] == pytest.approx(10 * 31 / 60, abs=1e-4)
    assert finding.detail["limit"] == pytest.approx(5.08, abs=1e-6)
    assert "speed_limit" in finding.fix
    assert "above its world's limit of 5.08 u/s" in finding.message
    assert _simulation(opened("Pushed"))[1] == []


def test_a_body_through_a_thin_wall_tunnelled_and_its_twin_did_not(opened):
    session = opened("ThroughTheWall")
    wall, disc = _bodies(session)
    _, (finding,) = _simulation(session)
    assert finding.code == "sim.tunnelled"
    assert [thing.id for thing in finding.objects] == [wall, disc]
    assert finding.detail == {"tick": 1}
    assert finding.t_start == Fraction(1, 60)
    assert finding.message == (
        f"{wall} and {disc} passed through each other in the step to tick 1, with"
        " no contact between them"
    )
    assert "continuous_collision" in finding.fix
    assert all(thing.bbox is not None for thing in finding.objects)
    twin = opened("StoppedByTheWall")
    assert _simulation(twin)[1] == []
    _, disc = _bodies(twin)
    # It stays on its side.
    for t in (Fraction(1, 4), Fraction(1, 2)):
        left, _, right, _ = twin.state_at(t).node(disc).scene_box
        assert (left + right) / 2 < 0, t


def test_findings_never_end_the_recording(opened):
    session = opened("Collapsing")
    assert session.record.duration == Fraction(1, 2)
    (occurrence,) = session.record.bakes.occurrences
    assert (occurrence.first_tick, occurrence.last_tick) == (0, 30)
    assert [each.code for each in occurrence.findings] == ["sim.unstable"] * 2


def test_a_finding_on_a_stay_s_first_row_belongs_to_the_world_s_first_stay(opened):
    """A stay that left before a tick passed ends at the world's first row;
    the next stay begins there, and does not report it again.
    """
    session = opened("ReenteredAtOnce")
    first, second = session.record.bakes.occurrences
    assert (first.first_tick, first.last_tick) == (0, 0)
    assert second.first_tick == 30
    assert [len(first.findings), len(second.findings)] == [1, 0]
    (found,) = first.findings
    assert (found.code, found.row, found.detail["tick"]) == ("sim.unstable", 0, 0)
    _, (finding,) = _simulation(session)
    assert (finding.code, finding.t_start) == ("sim.unstable", 0)


def test_a_finding_at_a_play_s_last_tick_names_that_play(opened):
    """The step to a play's last tick is the play's: a finding there names
    its line, not the next play's.
    """
    session = opened("EndsAtTheCrossing")
    _, (finding,) = _simulation(session)
    assert (finding.t_start, finding.detail["tick"]) == (Fraction(1, 2), 30)
    assert finding.line == _line("first")


def test_a_body_the_frame_does_not_draw_keeps_its_id_and_line(opened):
    """A world that entered between two frames: the frame on screen at its
    first tick does not draw its body, which the finding still names with its
    creating line and kind, its boxes null; its declared state is the play's
    that was running as it entered.
    """
    session = opened("ArrivesTooFast")
    (body,) = _bodies(session)
    _, (finding,) = _simulation(session)
    assert (finding.code, finding.t_start) == ("sim.unstable", Fraction(1, 10))
    (thing,) = finding.objects
    assert (thing.id, thing.source_line, thing.kind) == (body, _line("made"), "Dot")
    assert (thing.bbox, thing.screen_bbox) == (None, None)
    assert finding.line == _line("entered")


def test_a_finding_outside_the_lint_s_range_asks_for_no_frame(opened, monkeypatch):
    """The speeding point's finding is at 31/60 s: a lint over the first
    quarter second neither reports it nor asks for a frame of it.
    """
    session = opened("Speeding")
    asked: list[tuple[object, list[Fraction]]] = []
    request = frames_module.Frames.request

    def spied(self, owner, times, read, **options):
        asked.append((owner, list(times)))
        return request(self, owner, times, read, **options)

    monkeypatch.setattr(frames_module.Frames, "request", spied)
    _, found = _simulation(session, t_range=(0, 0.25))
    assert found == []
    assert [times for owner, times in asked if owner == bakes_module.OWNER] == []
    asked.clear()
    _, (finding,) = _simulation(session)
    assert finding.t_start == Fraction(31, 60)
    assert [times for owner, times in asked if owner == bakes_module.OWNER] == [
        [Fraction(7, 15)]
    ]


# Authoring


@pytest.mark.parametrize(
    "given",
    [
        {"speed_limit": 0},
        {"speed_limit": -1.0},
        {"speed_limit": 400.0},
        {"angular_speed_limit": 50.0},
    ],
)
def test_a_limit_is_positive_and_below_the_solver_s_cap(given):
    with pytest.raises(ValueError, match="below the solver's own cap"):
        World(**given)


def test_a_world_s_limits_and_continuous_detection_are_declared():
    with pytest.raises(TypeError, match="continuous_collision"):
        World(continuous_collision=1)
    plain = World()._declared
    assert World(speed_limit=10)._declared != plain
    assert World(angular_speed_limit=10)._declared != plain
    assert World(continuous_collision=False)._declared != plain
    assert World(speed_limit=10)._declared == World(speed_limit=10.0)._declared
    assert np.float32(World(speed_limit=5.08)._declaration.speed_limit) == np.float32(
        5.08
    )
