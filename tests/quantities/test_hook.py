"""The hook of declared quantities: ``state_at`` answers each object's
quantities at the instant its boxes are of, from engine-owned providers and
sources only, through one registry that a scene's code cannot stand in for.
"""

from __future__ import annotations

import gc
import math
import weakref
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Square
from arachne import quantities as hook
from arachne.quantities import At, Literal, Quantity, attach, read
from arachne.utils.testing.declared_probe import DeclaredProbe

from . import quantity_scenes

SCENES = Path(quantity_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}

AT_ZERO = At.of(Fraction(0), None)


@pytest.fixture
def opened():
    from arachne.session import Session
    from arachne.timeline import admission

    sessions = []
    admission.DEMOTED.clear()

    def open_scene(name: str):
        session = Session.open(SCENES, name, config=SMALL)
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _values(state, node_id: str) -> dict:
    return {q.name: q.value for q in state.quantities.get(node_id, ())}


def test_state_at_answers_each_instant_s_own_quantities(opened):
    session = opened("ChangedBetweenPlays")
    early = session.state_at(Fraction(1, 2))
    probe = early.named("probe").id
    square = early.named("square").id
    plain = early.named("plain").id
    later = session.state_at(Fraction(3, 2))
    again_early = session.state_at(Fraction(1, 2))
    again_later = session.state_at(Fraction(3, 2))
    for state in (early, again_early):
        assert _values(state, probe) == {"probe.amount": 1.0, "probe.setting": "off"}
        assert _values(state, square) == {"probe.size": 2.0}
    for state in (later, again_later):
        assert _values(state, probe) == {"probe.amount": 2.5, "probe.setting": "on"}
        assert _values(state, square) == {"probe.count": 3, "probe.size": 2.0}
    # A mobject with no provider and no source adds nothing.
    assert plain not in later.quantities
    assert set(later.quantities) == {probe, square}
    # Each list is sorted by name, with its unit and tolerance.
    assert later.quantities[probe] == (
        Quantity("probe.amount", 2.5, "1", tolerance=0.01),
        Quantity("probe.setting", "on", "name"),
    )
    assert later.quantities[square][0] == Quantity("probe.count", 3, "1", 0.5)


def test_a_restored_and_a_forward_answer_read_their_own_instant(opened):
    session = opened("ForwardRead")
    first, second = session.record.plays
    assert (first.route, second.route) == ("state", "state")
    unsampled = Fraction(1, 7)
    restorable = second.start + Fraction(1, 7)
    forward, restored = session.answers([unsampled, restorable], frames=False)
    assert forward.state.answer == "forward"
    assert restored.state.answer == "restored"
    probe = restored.state.named("probe").id
    assert _values(forward.state, probe) == {
        "probe.amount": 1.0,
        "probe.setting": "off",
    }
    assert _values(restored.state, probe)["probe.amount"] == 2.0
    # An attached source on an ordinary mobject, on both routes.
    dot = restored.state.named("dot").id
    assert _values(forward.state, dot) == {"probe.size": 1.0}
    assert _values(restored.state, dot) == {"probe.count": 2, "probe.size": 1.0}
    (again,) = session.answers([unsampled], frames=False)
    assert _values(again.state, probe)["probe.amount"] == 1.0
    assert _values(again.state, dot) == {"probe.size": 1.0}


def test_a_scene_s_subclass_is_answered_by_the_engine_s_provider(opened):
    session = opened("SubclassedProbe")
    state = session.state_at(Fraction(1, 4))
    (node,) = [node for node in state.nodes if node.type == "OverridingProbe"]
    assert _values(state, node.id) == {"probe.amount": 5.0, "probe.setting": "on"}


def test_a_function_of_a_scene_or_a_test_is_refused_as_a_provider():
    def mine(held, at):
        return ()

    with pytest.raises(ValueError, match="engine's own code"):
        hook.provider(Square)(mine)
    with pytest.raises(ValueError, match="engine's own code"):
        hook.source(tuple)(mine)
    with pytest.raises(ValueError, match="engine's own code"):
        hook.point_source(Square, channels=mine, value_at=mine)
    with pytest.raises(TypeError, match="Python function"):
        hook.provider(Square)(print)
    with pytest.raises(ValueError, match="engine's own code"):
        hook.provider(quantity_scenes.OverridingProbe)(
            quantity_scenes.OverridingProbe._declared
        )


def test_a_prefix_has_one_owner_and_a_shape():
    # The probes' prefix is registered by their module, for its owner.
    hook.register("probe.", owner="arachne.utils.testing")
    with pytest.raises(ValueError, match="arachne.utils.testing"):
        hook.register("probe.", owner="someone else")
    for wrong in ("probe", "a.b.", ".", "two words.", ""):
        with pytest.raises(ValueError, match="one segment"):
            hook.register(wrong, owner="someone else")
    with pytest.raises(ValueError, match="without spaces"):
        hook.register_unit("per second")


def test_the_simulation_s_prefix_and_units_are_its_own():
    import arachne.simulation  # noqa: F401

    with pytest.raises(ValueError, match="arachne.simulation"):
        hook.register("sim.", owner="someone else")
    square = Square()
    attach(square, Literal("sim.velocity", (0.0, -1.5, 0.0), "u/s"))
    assert read(Square, square.__dict__, AT_ZERO) == (
        Quantity("sim.velocity", (0.0, -1.5, 0.0), "u/s"),
    )


def test_only_a_registered_source_is_attached():
    with pytest.raises(TypeError, match="no source"):
        attach(Square(), ("probe.x", 1.0, "1"))


def _read_literals(*literals: Literal) -> tuple[Quantity, ...]:
    square = Square()
    attach(square, *literals)
    return read(Square, square.__dict__, AT_ZERO)


@pytest.mark.parametrize(
    ("literal", "reason"),
    [
        (Literal("nothing.x", 1.0, "1"), "prefix 'nothing.' is not registered"),
        (Literal("probe", 1.0, "1"), "is not <prefix>.<rest>"),
        (Literal("probe.a b", 1.0, "1"), "dot-separated segments"),
        (Literal("probe.x..y", 1.0, "1"), "dot-separated segments"),
        (Literal("probe.x", 1.0, "furlong"), "unit 'furlong' is not registered"),
        (Literal("probe.x", math.nan, "1"), "not finite"),
        (Literal("probe.x", math.inf, "1"), "not finite"),
        (Literal("probe.x", (1.0,) * 17, "1"), "at most 16 numbers, not 17"),
        (Literal("probe.x", {"a": 1.0}, "1"), "not a number, a vector or a name"),
        (Literal("probe.x", np.array([1.0]), "1"), "not a number, a vector or a name"),
        (Literal("probe.x", None, "1"), "holds none"),
        (Literal("probe.x", 1.0, ""), "unit '' is not registered"),
        (Literal("probe.x", (1.0, math.nan), "1"), "not finite"),
        (Literal("probe.x", "on", "1"), "unit is 'name'"),
        (Literal("probe.x", 1.0, "name"), "is a name"),
        (Literal("probe.x", "x" * 65, "name"), "1 to 64 bytes, not 65"),
        (Literal("probe.x", True, "1"), "not a number, a vector or a name"),
        (Literal("probe.x", 1.0, "1", tolerance=-0.1), "not below 0"),
    ],
    ids=lambda case: case if isinstance(case, str) else None,
)
def test_a_quantity_that_cannot_be_read_is_null_with_its_reason(literal, reason):
    (found,) = _read_literals(literal)
    assert found.value is None
    assert found.name == literal.name
    assert found.unit == literal.unit
    assert reason in found.unreadable


def test_a_readable_quantity_is_kept_as_stored_and_sorted_by_name():
    found = _read_literals(
        Literal("probe.b", (1, 2.5), "1"),
        Literal("probe.a", "é" * 32, "name"),
        Literal("probe.c", 7, "1"),
        Literal("probe.d", [], "1"),
    )
    assert found == (
        Quantity("probe.a", "é" * 32, "name"),
        Quantity("probe.b", (1.0, 2.5), "1"),
        Quantity("probe.c", 7, "1"),
        # No contact yet: a vector with nothing in it is a value.
        Quantity("probe.d", (), "1"),
    )


def test_a_literal_holds_no_list():
    assert Literal("probe.x", [1, 2.5], "1").value == (1, 2.5)


def test_what_the_hook_finds_of_a_class_lets_the_class_go():
    """A scene's class read once is not kept alive by what the hook
    remembered of it, so a module let go on reload or close goes with it.
    """

    class Discarded(DeclaredProbe):
        pass

    class Plain(Square):
        pass

    read(Discarded, DeclaredProbe().__dict__, AT_ZERO)
    read(Plain, Plain().__dict__, AT_ZERO)
    assert hook._provider_of(Discarded) is not None
    assert hook._point_source_of(Plain) is None
    gone = [weakref.ref(Discarded), weakref.ref(Plain)]
    del Discarded, Plain
    gc.collect()
    assert [ref() for ref in gone] == [None, None]


def test_a_name_declared_twice_on_one_object_is_unreadable():
    (found,) = _read_literals(
        Literal("probe.x", 1.0, "1"), Literal("probe.x", 2.0, "1")
    )
    assert found.value is None
    assert "more than once" in found.unreadable


def test_a_quantity_has_a_value_or_a_reason_exactly():
    with pytest.raises(ValueError, match="exactly one"):
        Quantity("probe.x", None, "1")
    with pytest.raises(ValueError, match="exactly one"):
        Quantity("probe.x", 1.0, "1", unreadable="why")


def test_a_provider_reads_what_it_is_handed():
    probe = DeclaredProbe(amount=0.5, setting="half")
    held = dict(probe.__dict__, amount=0.75)
    found = read(DeclaredProbe, held, AT_ZERO)
    # The attributes handed, not the mobject's own.
    assert found[0] == Quantity("probe.amount", 0.75, "1", tolerance=0.01)
    assert read(Square, Square().__dict__, AT_ZERO) == ()
