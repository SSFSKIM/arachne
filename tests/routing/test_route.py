"""A scene's route is read from its record: ``gpu`` exactly when an object a
drawn capture reaches holds a GPU-only capability there.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from arachne.routing import Where, capabilities, route

from .conftest import line_of


def _made_at(tag: str, n: int = 0) -> str:
    """The stable id of the ``n``-th object the line tagged ``tag`` made."""
    return f"{line_of(tag)}#{n}"


def test_a_scene_that_declares_nothing_routes_cpu(session_of):
    record = session_of("Plain").record
    assert capabilities(record) == {}
    assert route(record) == "cpu"


def test_a_capability_shown_only_in_the_last_play_routes_gpu_and_names_that_play(
    session_of,
):
    record = session_of("LastPlay").record
    found = capabilities(record)
    assert found == {
        "effect.blur": Where(
            play=1,
            t=record.plays[1].start,
            line=line_of("last-play"),
            object=_made_at("last-play-object"),
        )
    }
    assert route(record) == "gpu"


def test_a_capability_set_after_the_last_play_counts(session_of):
    record = session_of("AfterLastPlay").record
    assert capabilities(record) == {
        "effect.soft_mask": Where(
            play=None,
            t=record.duration,
            line=line_of("after-object"),
            object=_made_at("after-object"),
        )
    }
    assert route(record) == "gpu"


def test_a_capability_in_a_scene_with_no_play_counts(session_of):
    record = session_of("NoPlay").record
    assert record.plays == []
    assert capabilities(record) == {
        "layer.depth": Where(
            play=None,
            t=Fraction(0),
            line=line_of("no-play-object"),
            object=_made_at("no-play-object"),
        )
    }
    assert route(record) == "gpu"


@pytest.mark.parametrize("scene", ["NeverShown", "Withdrawn", "RemovedThenDeclared"])
def test_an_object_never_shown_holding_it_routes_cpu(session_of, scene):
    record = session_of(scene).record
    assert capabilities(record) == {}
    assert route(record) == "cpu"


def test_a_capability_that_comes_and_goes_still_counts(session_of):
    record = session_of("ComesAndGoes").record
    found = capabilities(record)
    assert list(found) == ["effect.window"]
    assert (found["effect.window"].play, found["effect.window"].line) == (
        0,
        line_of("comes"),
    )
    assert route(record) == "gpu"


def test_capabilities_are_listed_in_the_order_they_first_appear(session_of):
    record = session_of("Nested").record
    found = capabilities(record)
    assert list(found) == ["layer.depth", "effect.blur", "effect.depth_of_field"]
    assert [(where.play, where.line) for where in found.values()] == [
        (0, line_of("nested-first")),
        (1, line_of("nested-second")),
        (1, line_of("nested-second")),
    ]


def test_a_record_that_refuses_drawing_still_yields_its_capabilities(session_of):
    record = session_of("Refused").record
    assert record.refusal is not None
    found = capabilities(record)
    assert list(found) == ["effect.blur"]
    assert (found["effect.blur"].play, found["effect.blur"].line) == (
        0,
        line_of("refused"),
    )
    assert route(record) == "gpu"


def test_the_route_reads_the_record_not_the_live_scene(session_of):
    """Withdrawing the capability from the live object once the scene has
    run changes nothing the record holds.
    """
    from arachne.routing import withdraw

    session = session_of("AfterLastPlay")
    (square,) = session.scene.mobjects
    withdraw(square, "effect.soft_mask")
    assert route(session.record) == "gpu"


def test_the_route_is_asked_of_a_complete_record_only(session_of):
    record = session_of("LastPlay").record
    final, record.final = record.final, None
    try:
        with pytest.raises(ValueError, match="complete record"):
            route(record)
    finally:
        record.final = final
