"""The bake: immutable contents keyed by their canonical bytes, and the
occurrences that place them in a record, read by tick.
"""

from __future__ import annotations

import hashlib
import struct
from fractions import Fraction

import numpy as np
import pytest

from arachne.timeline.bake import (
    BakeContent,
    BakeFinding,
    BakeIdentity,
    Bakes,
    Binding,
    Channel,
    EventTable,
)

IDENTITY = BakeIdentity(
    producer="a producer",
    producer_version="1.0",
    solver="a solver",
    solver_version="2.0",
    integrator="symplectic Euler",
    build="toolchain 1; x86_64; sse2",
    features=("b", "a"),
)
DECLARATION = hashlib.blake2b(b"a declaration", digest_size=16).digest()


def _content(ticks: int = 4, *, scale: float = 1.0, events=None) -> BakeContent:
    pose = np.arange(ticks * 2 * 3, dtype=float).reshape(ticks, 2, 3) * scale
    return BakeContent(
        IDENTITY,
        ticks,
        {
            "pose": Channel("f32", "u", pose),
            "energy": Channel("f64", "1", np.linspace(0, 1, ticks)),
        },
        {"contact": EventTable(events)} if events is not None else None,
    )


def _sealed(bakes: Bakes, content: BakeContent, first_tick: int, **given):
    ordinal = bakes.begin()
    return bakes.seal(
        ordinal,
        content,
        first_tick=first_tick,
        closed=True,
        declaration=DECLARATION,
        **given,
    )


def test_two_occurrences_of_one_content_with_their_own_bindings():
    bakes = Bakes()
    first = _sealed(bakes, _content(), 0, bindings=(Binding("s.py:3#0", "pose", 0),))
    second = _sealed(bakes, _content(), 0, bindings=(Binding("s.py:9#0", "pose", 0),))
    assert len(bakes.contents) == 1
    assert first.content == second.content
    assert (first.ordinal, second.ordinal) == (0, 1)
    assert first.bindings != second.bindings
    assert bakes.occurrences == (first, second)
    assert np.array_equal(bakes.row(0, 2)["pose"], bakes.row(1, 2)["pose"])


def test_two_stays_of_equal_content_answer_on_both_sides_of_their_ends():
    bakes = Bakes()
    early = _sealed(bakes, _content(), 0)
    late = _sealed(bakes, _content(), 30)
    assert (early.first_tick, early.last_tick) == (0, 3)
    assert (late.first_tick, late.last_tick) == (30, 33)
    content = bakes.content_of(1)
    for ordinal, first, last in ((0, 0, 3), (1, 30, 33)):
        assert bakes.row(ordinal, first - 1) is None
        assert bakes.row(ordinal, last + 1) is None
        start, end = bakes.row(ordinal, first), bakes.row(ordinal, last)
        assert (start.index, start.tick, end.index, end.tick) == (0, first, 3, last)
        assert np.array_equal(end["pose"], content.channels["pose"].values[3])
    # An instant reads the last tick at or before it.
    assert bakes.at(1, Fraction(31, 60)).index == 1
    assert bakes.at(1, Fraction(31, 60) + Fraction(1, 121)).index == 1
    assert bakes.at(1, Fraction(30, 60) - Fraction(1, 1000)) is None
    # A row is read-only.
    with pytest.raises(ValueError, match="read-only"):
        bakes.row(0, 0)["pose"][0, 0] = 1.0


def test_a_channel_need_not_be_about_any_mobject():
    bakes = Bakes()
    grid = BakeContent(
        IDENTITY, 2, {"density": Channel("f32", "1", np.ones((2, 8, 8)))}
    )
    occurrence = _sealed(bakes, grid, 5)
    assert occurrence.bindings == ()
    assert bakes.row(0, 6)["density"].shape == (8, 8)


def test_contacts_that_begin_at_one_tick_are_in_the_canonical_order():
    events = [[2, 3, 4, 1], [2, 0, 1, 1], [1, 0, 2, 0], [2, 0, 1, 0]]
    content = _content(events=events)
    assert content.events["contact"].rows.tolist() == [
        [1, 0, 2, 0],
        [2, 0, 1, 0],
        [2, 0, 1, 1],
        [2, 3, 4, 1],
    ]
    shuffled = _content(events=list(reversed(events)))
    assert shuffled.canonical_bytes() == content.canonical_bytes()
    assert shuffled.key == content.key


def test_two_encodings_of_one_content_are_byte_equal_and_one_value_moves_the_key():
    one = _content()
    other = BakeContent(
        _fields(features=("a", "b")),
        4,
        {"energy": one.channels["energy"], "pose": one.channels["pose"]},
    )
    assert other.canonical_bytes() == one.canonical_bytes()
    assert (
        other.key
        == one.key
        == hashlib.blake2b(one.canonical_bytes(), digest_size=16).digest()
    )
    pose = one.channels["pose"].values.copy()
    pose[2, 1, 0] = np.nextafter(pose[2, 1, 0], np.float32(np.inf))
    changed = BakeContent(
        IDENTITY,
        4,
        {"pose": Channel("f32", "u", pose), "energy": one.channels["energy"]},
    )
    assert changed.key != one.key
    assert _content(scale=-0.0).key != _content(scale=0.0).key


def _fields(**changed) -> BakeIdentity:
    values = {
        name: getattr(IDENTITY, name)
        for name in (
            "producer",
            "producer_version",
            "solver",
            "solver_version",
            "integrator",
            "build",
            "features",
        )
    }
    return BakeIdentity(**{**values, **changed})


def test_the_identity_is_in_the_key():
    assert (
        _content().key
        != BakeContent(_fields(solver_version="2.1"), 4, dict(_content().channels)).key
    )


def test_the_canonical_layout():
    content = BakeContent(
        _fields(features=("z", "y")),
        2,
        {"x": Channel("f32", "u", [1.0, 2.0])},
        {"e": EventTable([[1, 7]])},
    )
    data = content.canonical_bytes()

    def text(value: str) -> bytes:
        return struct.pack("<I", len(value.encode())) + value.encode()

    head = b"ARBK" + struct.pack("<I", 1)
    strings = b"".join(
        text(s)
        for s in (
            "a producer",
            "1.0",
            "a solver",
            "2.0",
            "symplectic Euler",
            "toolchain 1; x86_64; sse2",
            "y,z",
        )
    )
    channel = (
        text("x") + struct.pack("<BB", 1, 0) + text("u") + struct.pack("<2f", 1, 2)
    )
    events = text("e") + struct.pack("<BI", 2, 1) + struct.pack("<2I", 1, 7)
    assert data == (
        head
        + strings
        + struct.pack("<II", 2, 1)
        + channel
        + struct.pack("<I", 1)
        + events
    )


def test_a_non_finite_value_is_stored_as_the_one_quiet_nan():
    payload = np.array([np.nan, -np.nan, np.inf, -np.inf], dtype=np.float32)
    payload.view(np.uint32)[1] = 0xFFC00123
    stored = Channel("f32", "1", payload).values
    assert stored.view(np.uint32).tolist() == [0x7FC00000] * 4
    wide = Channel("f64", "1", payload.astype(np.float64)).values
    assert wide.view(np.uint64).tolist() == [0x7FF8000000000000] * 4


def test_a_seal_names_what_its_content_holds():
    bakes = Bakes()
    with pytest.raises(KeyError, match="No stay"):
        bakes.seal(0, _content(), first_tick=0, closed=True, declaration=DECLARATION)
    ordinal = bakes.begin(live="what computes it")
    assert bakes.live(ordinal) == "what computes it"
    assert bakes.row(ordinal, 0) is None
    with pytest.raises(ValueError, match="no channel"):
        bakes.seal(
            ordinal,
            _content(),
            first_tick=0,
            closed=True,
            declaration=DECLARATION,
            bindings=(Binding("s.py:3#0", "velocity", 0),),
        )
    with pytest.raises(ValueError, match="past"):
        bakes.seal(
            ordinal,
            _content(),
            first_tick=0,
            closed=True,
            declaration=DECLARATION,
            bindings=(Binding("s.py:3#0", "pose", 2),),
        )
    with pytest.raises(ValueError, match="outside the stay"):
        bakes.seal(
            ordinal,
            _content(),
            first_tick=0,
            closed=True,
            declaration=DECLARATION,
            findings=(BakeFinding("sim.unstable", 4, ("s.py:3#0",)),),
        )
    sealed = bakes.seal(
        ordinal,
        _content(),
        first_tick=0,
        closed=False,
        declaration=DECLARATION,
        findings=(BakeFinding("sim.unstable", 3, ["s.py:3#0"], {"speed": 9.0}),),
    )
    assert sealed.findings[0].ids == ("s.py:3#0",)
    with pytest.raises(KeyError):
        bakes.live(ordinal)


def test_a_content_holds_its_rows():
    with pytest.raises(ValueError, match="one row or more"):
        BakeContent(IDENTITY, 0, {})
    with pytest.raises(ValueError, match="holds 3 rows, not 4"):
        BakeContent(IDENTITY, 4, {"x": Channel("f32", "1", np.zeros(3))})
    with pytest.raises(ValueError, match="past the last"):
        BakeContent(IDENTITY, 4, {}, {"contact": EventTable([[4, 0, 1, 1]])})
    with pytest.raises(ValueError, match="one of"):
        Channel("f16", "1", np.zeros(3))


def test_a_record_s_bakes_pickle():
    """What a record holds can be handed to another process."""
    import pickle

    bakes = Bakes()
    content = _content(events=[[1, 0, 1, 1]])
    _sealed(
        bakes,
        content,
        3,
        findings=(BakeFinding("sim.unstable", 1, ("s.py:3#0",), {"speed": 9.0}),),
    )
    again = pickle.loads(pickle.dumps(bakes))
    assert again.content_of(0).canonical_bytes() == content.canonical_bytes()
    assert again.occurrences == bakes.occurrences
    assert np.array_equal(again.row(0, 4)["pose"], bakes.row(0, 4)["pose"])


def _assert_immutable(content: BakeContent) -> None:
    key = content.key
    for name, value in (("key", b"0" * 16), ("ticks", 9), ("identity", IDENTITY)):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(content, name, value)
    channel = content.channels["pose"]
    with pytest.raises(AttributeError, match="immutable"):
        channel.unit = "m"
    values = channel.values
    assert not values.flags.owndata
    with pytest.raises(ValueError, match="read-only"):
        values[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        values.flags.writeable = True
    # A read's shape is its own.
    values.shape = (-1,)
    assert channel.values.shape == (4, 2, 3)
    rows = content.events["contact"].rows
    with pytest.raises(ValueError, match="WRITEABLE"):
        rows.flags.writeable = True
    with pytest.raises(AttributeError, match="immutable"):
        content.events["contact"]._data = b""
    with pytest.raises(TypeError):
        content.channels["other"] = channel
    assert (
        content.key
        == key
        == hashlib.blake2b(content.canonical_bytes(), digest_size=16).digest()
    )


def test_a_content_cannot_change_after_it_is_keyed_before_or_after_pickling():
    import pickle

    content = _content(events=[[1, 0, 1, 1]])
    _assert_immutable(content)
    again = pickle.loads(pickle.dumps(content))
    assert again.key == content.key
    assert again.canonical_bytes() == content.canonical_bytes()
    _assert_immutable(again)
    finding = BakeFinding("sim.unstable", 1, ("s.py:3#0",), {"speed": 9.0})
    with pytest.raises(TypeError):
        finding.detail["speed"] = 1.0
    assert pickle.loads(pickle.dumps(finding)) == finding


@pytest.mark.parametrize(
    "rows",
    [
        [[1.5, 0]],
        [[np.nan, 0]],
        [[np.inf, 0]],
        [[True, False]],
        [["1", "0"]],
        [[-1, 0]],
    ],
    ids=["fraction", "nan", "inf", "bool", "text", "negative"],
)
def test_an_event_table_holds_unsigned_integers_only(rows):
    with pytest.raises(ValueError, match="unsigned 32-bit integers"):
        EventTable(rows)


def test_an_event_table_takes_whole_floats():
    assert EventTable([[2.0, 1.0]]).rows.tolist() == [[2, 1]]
