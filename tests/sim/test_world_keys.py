"""What a world puts on the mobjects a record captures is plain and keys as
content: every value a capture stored of a world, at each play's start (on
both sides of a stay that ends and one that begins), has a content key, the
same on two reads and in two executions, and none changes with what a
callback force keeps in its closure. (The content key keys every value once
:mod:`arachne.timeline.keys` is part of the engine; until then this file has
nothing to check.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("arachne_sim")
keys = pytest.importorskip("arachne.timeline.keys")

from arachne import World  # noqa: E402
from arachne.session import Session  # noqa: E402
from arachne.timeline import admission  # noqa: E402

from . import world_scenes  # noqa: E402

SCENES = Path(world_scenes.__file__)
CONFIG = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}


def _stored(session: Session) -> list[tuple[str, object]]:
    """Every value each play's start capture stored of each world's family,
    by where it is: the play, the member's place in the worlds' families and
    the field.
    """
    found = []
    for play in session.record.plays:
        nodes = list(play.start_state.nodes())
        places: dict[int, str] = {}
        worlds = [mob for mob, _ in nodes if isinstance(mob, World)]
        for w, world in enumerate(worlds):
            for m, member in enumerate(world.get_family()):
                places.setdefault(id(member), f"{w}.{m}")
        for mob, state in nodes:
            place = places.get(id(mob))
            if place is None:
                continue
            for name, value in zip(state.keys, state.values, strict=True):
                found.append((f"{play.index} {place} {name}", value))
    return found


def _keyed(found: list[tuple[str, object]]) -> list[tuple[str, bytes]]:
    keyed = []
    for where, value in found:
        assert keys.unkeyable(value) is None, where
        key = keys.value_key(value)[0]
        assert key == keys.value_key(value)[0], where
        keyed.append((where, key))
    return keyed


@pytest.mark.parametrize(
    "name", ["ClosedThenOpen", "AwayAndBack", "Tumbling", "Remembering"]
)
def test_every_value_a_capture_stores_of_a_world_keys(name):
    admission.DEMOTED.clear()
    sessions = [Session.open(SCENES, name, config=CONFIG) for _ in range(2)]
    try:
        first, second = (_keyed(_stored(session)) for session in sessions)
        assert first
        assert first == second
        if name == "AwayAndBack":
            # At the start of the play it is away for, no capture holds it.
            assert {where.split()[0] for where, _ in first} == {"0", "2"}
        if name == "Remembering":
            for held in sessions[0].module.HELD:
                held.append(-1)
            assert _keyed(_stored(sessions[0])) == first
    finally:
        for session in sessions:
            session.close()
        admission.DEMOTED.clear()


def test_the_content_key_reads_a_world_by_its_declaration_s_digest():
    """Both content keys encode a string: a world given another declaration
    keys apart at its first capture, its bodies alike.
    """
    admission.DEMOTED.clear()
    sessions = [
        Session.open(SCENES, name, config=CONFIG) for name in ("Falling", "Heavier")
    ]
    try:
        declared = []
        for session in sessions:
            (state,) = [
                state
                for mob, state in session.record.plays[0].start_state.nodes()
                if isinstance(mob, World)
            ]
            fields = dict(zip(state.keys, state.values, strict=True))
            declared.append(keys.value_key(fields["_declared"])[0])
        assert declared[0] != declared[1]
    finally:
        for session in sessions:
            session.close()
        admission.DEMOTED.clear()
