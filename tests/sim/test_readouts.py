"""A readout a world makes of one of its bodies: at every capture it shows
the body's quantity there, declared unrounded as ``annotation.count`` and
shown in its format, and it declares under ``sim.`` which body, quantity,
part, scale and tick it shows, so a check compares the number shown with the
body's own quantity read at the same capture. Twins: a plain readout set by
hand declares no such link; a readout of a world that never entered the
scene declares it unreadable.

The rule a check applies: ``annotation.count`` equals the part of the body's
quantity at ``sim.reads_tick`` times ``sim.reads_scale``, to double
precision (exactly, for an axis at scale 1); the text shown is the count in
the readout's format, within half a unit of the format's last digit.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("arachne_sim")

from arachne import BodyReadout, CountUp, Dot, Readout, World  # noqa: E402
from arachne import quantities as quantities_module  # noqa: E402
from arachne.lint import lint  # noqa: E402
from arachne.lint.rules import anchored  # noqa: E402
from arachne.mobject.annotation.attach import placed_point  # noqa: E402
from arachne.session import Session  # noqa: E402
from arachne.timeline import admission  # noqa: E402
from arachne.timeline.record import NodeState, content_fields  # noqa: E402

from . import readout_scenes  # noqa: E402
from .readout_scenes import FALLING  # noqa: E402

SCENES = Path(readout_scenes.__file__)
G = 9.8
#: Half a unit of each of :class:`FallingRead`'s formats' last digit.
HALF_DIGIT = {"{:.1f}": 0.05, "{:.0f} cm/s": 0.5, "{:.6f}": 5e-7}
#: What a body readout declares under ``sim.``, and the unit of each.
LINK = {
    "sim.reads": "name",
    "sim.reads_body": "1",
    "sim.reads_part": "name",
    "sim.reads_scale": "1",
    "sim.reads_tick": "1",
}


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, fps: int = 15) -> Session:
        session = Session.open(
            SCENES,
            name,
            config={"pixel_width": 96, "pixel_height": 54, "frame_rate": fps},
        )
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _read(state, node_id: str) -> dict:
    return {q.name: q for q in state.quantities.get(node_id, ())}


def _bodies(session: Session, ordinal: int = 0) -> list[str]:
    """The stable ids of the bodies of the stay ``ordinal``, by number."""
    occurrence = session.record.bakes.occurrence(ordinal)
    return [b.id for b in occurrence.bindings if b.channel == "pose"]


def _linked(session: Session, read: dict) -> str:
    """The stable id of the body a readout's declarations say it reads."""
    ordinal, body = (int(n) for n in read["sim.reads_body"].value)
    return _bodies(session, ordinal)[body]


def _readouts(state) -> dict[tuple, str]:
    """The body readouts of ``state``, by the quantity, part and scale each
    declares it shows.
    """
    found = {}
    for node in state.nodes:
        if node.type == "BodyReadout":
            read = _read(state, node.id)
            shows = tuple(
                read[name].value
                for name in ("sim.reads", "sim.reads_part", "sim.reads_scale")
            )
            found[shows] = node.id
    return found


def _text(state, readout: str) -> str:
    """The text a readout shows at ``state``: its label's."""
    return state.node(state.node(readout).children[0]).text


def _part(value, part: str) -> float:
    if part == "value":
        return float(value)
    if part == "norm":
        return math.hypot(*value)
    return value["xyz".index(part)]


def _row(session: Session, ordinal: int, channel: str, tick: int) -> np.ndarray:
    bakes = session.record.bakes
    occurrence = bakes.occurrence(ordinal)
    values = bakes.contents[occurrence.content].channels[channel].values
    return values[tick - occurrence.first_tick].astype(float)


#: The frame instants of the 15 fps grid in :class:`FallingRead`'s second.
FRAMES = [Fraction(k, 15) for k in range(15)]


def test_a_falling_body_s_readouts_show_its_quantities_at_every_capture(opened):
    session = opened("FallingRead")
    (ball,) = _bodies(session)
    for t in FRAMES:
        state = session.state_at(t)
        assert state.as_of == t
        tick = math.floor(t * 60)
        body = _read(state, ball)
        readouts = _readouts(state)
        assert set(readouts) == set(FALLING.values())
        for pattern, shows in FALLING.items():
            quantity, part, scale = shows
            read = _read(state, readouts[shows])
            assert {name: read[name].unit for name in LINK} == LINK
            assert all(q.unreadable is None for q in read.values())
            assert _linked(session, read) == ball
            assert read["sim.reads_tick"].value == tick
            count = read["annotation.count"].value
            assert count == pytest.approx(
                scale * _part(body[quantity].value, part), rel=1e-12, abs=1e-12
            )
            # Shown in its format, within half its last digit of the count.
            text = _text(state, readouts[shows])
            assert text == pattern.format(count)
            assert abs(float(text.split()[0]) - count) <= HALF_DIGIT[pattern] + 1e-12
        # The closed form: v = -g t, which the tick's integration keeps.
        vy = _read(state, readouts[FALLING["{:.1f}"]])["annotation.count"].value
        assert vy == pytest.approx(-G * tick / 60, abs=1e-3)


def test_a_readout_declares_its_count_unrounded_and_shows_it_rounded(opened):
    session = opened("FallingRead")
    state = session.state_at(Fraction(7, 15))
    readout = _readouts(state)[FALLING["{:.1f}"]]
    count = _read(state, readout)["annotation.count"].value
    # Tick 28: -9.8 * 28/60 = -4.5733..., which one decimal does not hold.
    assert count != round(count, 1)
    assert _text(state, readout) == f"{count:.1f}" == "-4.6"
    assert abs(-4.6 - count) <= 0.05


def test_readouts_show_the_same_at_15_and_60_frames_per_second(opened):
    shown = {}
    for fps in (15, 60):
        session = opened("FallingRead", fps)
        answers = []
        for t in FRAMES:
            state = session.state_at(t)
            assert state.as_of == t
            for shows, readout in sorted(_readouts(state).items()):
                read = _read(state, readout)
                answers.append(
                    (
                        shows,
                        read["sim.reads_tick"].value,
                        read["annotation.count"].value,
                        _text(state, readout),
                    )
                )
        shown[fps] = answers
    assert shown[15] == shown[60]


def test_a_plain_readout_set_by_hand_declares_no_link(opened):
    session = opened("FallingReadByHand")
    state = session.state_at(Fraction(3, 4))
    (plain,) = [n.id for n in state.nodes if n.type == "Readout"]
    read = _read(state, plain)
    assert set(read) == {"annotation.count"}
    assert not [n for n in state.nodes if n.type == "BodyReadout"]
    for t in FRAMES:
        assert not [
            q.name
            for found in session.state_at(t).quantities.values()
            for q in found
            if q.name.startswith("sim.reads")
        ]


def test_a_readout_of_a_world_never_in_the_scene_declares_its_link_unreadable(
    opened,
):
    session = opened("ReadWithoutItsWorld")
    state = session.state_at(Fraction(1, 3))
    (readout,) = [n.id for n in state.nodes if n.type == "BodyReadout"]
    read = _read(state, readout)
    assert read["annotation.count"].value == 0.0
    assert _text(state, readout) == "0.0"
    assert (read["sim.reads"].value, read["sim.reads_part"].value) == (
        "sim.velocity",
        "norm",
    )
    for name in ("sim.reads_body", "sim.reads_tick"):
        assert read[name].value is None
        assert read[name].unreadable == (
            "the readout has read no stay of its body's world"
        )
    result = lint(session, rules=())
    found = [f for f in result.findings if f.code.startswith("sim.")]
    assert {(f.code, f.objects[0].id, f.detail["name"]) for f in found} == {
        ("sim.quantity_unreadable", readout, "sim.reads_body"),
        ("sim.quantity_unreadable", readout, "sim.reads_tick"),
    }


def test_a_readout_after_its_world_left_holds_a_row_of_its_stay(opened):
    """Once its world has left, it holds the last value its updater set, and
    keeps declaring the row that value is: the body and a tick of the stay,
    at or before its last (30). Set by hand there, it refuses, and its link
    stays true.
    """
    session = opened("ReadAfterItsWorldLeft")
    (occurrence,) = session.record.bakes.occurrences
    assert occurrence.last_tick == 30
    assert "not set by hand" in session.module.FOUND["set by hand"]
    # While the world is in the scene, each capture's own tick.
    early = session.state_at(Fraction(2, 5))
    (readout,) = [n.id for n in early.nodes if n.type == "BodyReadout"]
    assert _read(early, readout)["sim.reads_tick"].value == 24
    held = set()
    for t in (Fraction(1, 2), Fraction(3, 5), Fraction(4, 5), Fraction(14, 15)):
        state = session.state_at(t)
        read = _read(state, readout)
        assert all(q.unreadable is None for q in read.values())
        tick = read["sim.reads_tick"].value
        assert 28 <= tick <= occurrence.last_tick
        count = read["annotation.count"].value
        assert count == _row(session, 0, "velocity", tick)[0, 1]
        assert _text(state, readout) == f"{count:.3f}"
        assert _linked(session, read) == _bodies(session)[0]
        # At 1/2, the play boundary off the frame grid, the answer may be
        # the last frame's or the boundary's: either is a row of the stay.
        if t > Fraction(1, 2):
            held.add((tick, count, read["sim.reads_body"].value))
    # Through the wait after the world left, one value, one row.
    assert len(held) == 1


def test_an_open_world_s_readout_shows_the_tick_the_world_has_stepped_to(opened):
    """Added after its world, a readout shows the capture's tick; before it,
    the tick before, and it says so.
    """
    session = opened("PushedRead")
    (ball,) = _bodies(session)
    for k in range(1, 8):
        t = Fraction(k, 15)
        state = session.state_at(t)
        tick = math.floor(t * 60)
        # Told apart by their formats: four decimals before, five after.
        before, after = sorted(
            (n.id for n in state.nodes if n.type == "BodyReadout"),
            key=lambda found: len(_text(state, found).split(".")[1]),
        )
        shown_after = _read(state, after)
        assert shown_after["sim.reads_tick"].value == tick
        assert (
            shown_after["annotation.count"].value
            == _read(state, ball)["sim.velocity"].value[0]
        )
        shown_before = _read(state, before)
        assert shown_before["sim.reads_tick"].value == tick - 1
        assert _linked(session, shown_before) == ball
        assert (
            shown_before["annotation.count"].value
            == _row(session, 0, "velocity", tick - 1)[0, 0]
        )
        assert shown_before["annotation.count"].value < (
            shown_after["annotation.count"].value
        )


def test_a_readout_shows_a_2d_angular_velocity_and_a_contact_count(opened):
    session = opened("SpunAndDropped")
    square, _, ball = _bodies(session)
    state = session.state_at(Fraction(2, 3))
    readouts = _readouts(state)
    spin = _read(state, readouts[("sim.angular_velocity", "value", 1.0)])
    assert spin["annotation.count"].value == (
        _read(state, square)["sim.angular_velocity"].value
    )
    assert spin["annotation.count"].value == pytest.approx(2.0, abs=1e-5)
    assert _linked(session, spin) == square
    contacts = _read(state, readouts[("sim.contact_count", "value", 1.0)])
    assert contacts["annotation.count"].value == 1.0
    assert _read(state, ball)["sim.contact_count"].value == 1
    assert _linked(session, contacts) == ball
    assert _text(state, readouts[("sim.contact_count", "value", 1.0)]) == "1"


def test_the_lint_finds_a_following_body_readout_at_its_anchor(opened):
    session = opened("FallingRead")
    # What the lint reads of it, its recorded fields and points alone: the
    # anchor's point placement says is the body's centre, as the scene ended.
    (following,) = [
        m
        for m in session.record.mobjects()
        if type(m) is BodyReadout and m.attached is not None
    ]
    dot = following.attached.anchor.mobject
    np.testing.assert_allclose(placed_point(following), dot.get_center(), atol=1e-9)
    assert following.label.original_text == following.format.format(following.value)
    found = lint(session, rules=(anchored,)).findings
    assert [f.code for f in found] == [], [f.message for f in found]
    result = lint(session, rules=())
    assert [f.code for f in result.findings if f.code.startswith("sim.")] == []


# Its declarations, its keys and what it refuses


def _readout(**options) -> BodyReadout:
    world = World()
    ball = world.body(Dot(), shape=None, mass=1.0, velocity=(3.0, 4.0))
    return world.readout(ball, "velocity", **options)


def test_what_a_body_readout_declares_can_be_carried():
    for name, unit in LINK.items():
        assert quantities_module.carriable(name, unit) is None
    reading = _readout(part="x", scale=-2.5)
    (shown,) = reading._quantities
    assert (shown.quantity, shown.part, shown.scale, shown.body) == (
        "sim.velocity",
        "x",
        -2.5,
        0,
    )
    assert isinstance(reading, Readout)


def test_a_body_readout_s_content_is_a_readout_s():
    assert BodyReadout._content_key_omits == frozenset({"_shows", "_quantities"})
    reading = _readout(format="{:.2f} u/s")
    omitted = BodyReadout._content_key_omits

    def keyed(decimals=None) -> dict[str, bytes]:
        state = NodeState.of(reading)
        return {
            key: part
            for key, part, *_ in content_fields(state, decimals)
            if key not in omitted
        }

    before, rounded = keyed(), keyed(6)
    for field in ("_value", "format", "hold"):
        assert field in before
        assert field in rounded
    reading._show(2.5)
    assert keyed()["_value"] != before["_value"]
    assert keyed(6)["_value"] != rounded["_value"]


@pytest.mark.parametrize(
    ("quantity", "options", "error", "message"),
    [
        ("contact_times", {}, ValueError, "velocity"),
        ("sim.velocity", {}, ValueError, "velocity"),
        ("velocity", {"part": "z"}, ValueError, "norm, x, y"),
        ("angular_velocity", {"part": "norm"}, ValueError, "value"),
        ("contact_count", {"part": "x"}, ValueError, "value"),
        ("velocity", {"scale": 0}, ValueError, "scale"),
        ("velocity", {"scale": float("nan")}, ValueError, "scale"),
    ],
)
def test_a_body_readout_refuses_what_it_cannot_show(quantity, options, error, message):
    world = World()
    ball = world.body(Dot(), shape=None, mass=1.0)
    with pytest.raises(error, match=message):
        world.readout(ball, quantity, **options)


def test_a_body_readout_reads_a_body_of_its_world():
    world, other = World(), World()
    ball = other.body(Dot(), shape=None, mass=1.0)
    with pytest.raises(ValueError, match="not a body of this world"):
        world.readout(ball, "velocity")
    with pytest.raises(TypeError, match="reads a body"):
        BodyReadout(Dot(), "velocity")


def test_a_body_readout_is_not_set_by_hand_nor_counted():
    reading = _readout()
    with pytest.raises(TypeError, match="not set by hand"):
        reading.set_value(100)
    count = CountUp(reading, 5)
    with pytest.raises(TypeError, match="CountUp"):
        count.begin()
    assert reading.value == 0.0


# A readout held behind its body


def _stale(session: Session) -> list:
    result = lint(session, rules=())
    return [f for f in result.findings if f.code == "sim.readout_stale"]


def test_a_readout_faded_in_over_a_moving_body_is_stale(opened):
    session = opened("FadedInOverAFall")
    (ball,) = _bodies(session)
    (finding,) = _stale(session)
    assert finding.severity == "error"
    (thing,) = finding.objects
    assert thing.kind == "BodyReadout"
    # Made at 1/5 s, tick 12, and held there through the fade.
    assert finding.detail["body"] == ball
    assert finding.detail["tick"] == 12
    assert finding.detail["scene_tick"] - 12 > 1
    assert Fraction(1, 5) < finding.t_start <= finding.t_end < Fraction(7, 10)
    assert "shows tick 12 while the scene is at tick" in finding.message
    assert "self.add" in finding.fix


def test_a_readout_added_over_a_moving_body_is_not_stale(opened):
    assert _stale(opened("AddedOverAFall")) == []


def test_an_open_world_s_readout_a_tick_behind_is_not_stale(opened):
    assert _stale(opened("PushedRead")) == []
