"""A slot shows exactly one of its drawings: the one it says it shows has
its points, posed with the slot's part, and every member of every other
drawing's family holds none, at every capture of a play that switches it;
a style given to the rig reaches the drawings it does not show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arachne.assets import Rig
from arachne.session import Session
from arachne.timeline import admission

from . import rig_scenes

SCENES = Path(rig_scenes.__file__)


def _invariants(slot_group, shown: str, points_of=lambda m: m.points) -> None:
    """The slot's three invariants: one drawing with points, the one shown,
    and every member of the others empty.
    """
    with_points = [
        d.name
        for d in slot_group.submobjects
        if any(len(points_of(m)) for m in d.get_family())
    ]
    assert with_points == [shown]
    for drawing in slot_group.submobjects:
        if drawing.name != shown:
            assert all(len(points_of(m)) == 0 for m in drawing.get_family())


def test_a_slot_shows_its_default_and_switches_to_the_drawing_asked():
    figure = Rig("fixture:figure")
    assert (figure.shown("hand"), figure.shown("eyes")) == ("open", "open")
    _invariants(figure.part("hand"), "open")
    figure.show(hand="point")
    assert figure.shown("hand") == "point"
    _invariants(figure.part("hand"), "point")
    _invariants(figure.part("eyes"), "open")


def test_a_switched_in_drawing_is_posed_with_its_slot():
    figure = Rig("fixture:figure").pose(reach=1)
    figure.show(hand="point")
    reference = Rig("fixture:figure").show(hand="point").pose(reach=1)
    for got, want in zip(
        figure.part("hand").family_members_with_points(),
        reference.part("hand").family_members_with_points(),
        strict=True,
    ):
        np.testing.assert_allclose(got.points, want.points, atol=1e-12)


def test_a_style_given_to_the_rig_reaches_the_drawings_it_does_not_show():
    dial = Rig("fixture:dial")
    dial.set_opacity(0.4)
    dial.show(lamp="on")
    (lamp,) = next(d for d in dial.part("lamp").submobjects if d.name == "on")
    assert lamp.get_fill_opacity() == pytest.approx(0.4)


@pytest.fixture
def switching():
    admission.DEMOTED.clear()
    session = Session.open(
        SCENES,
        "Switching",
        config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 30},
    )
    yield session
    session.close()
    admission.DEMOTED.clear()


def test_a_slot_shows_one_drawing_at_every_capture_of_a_play_that_switches_it(
    switching,
):
    record = switching.record
    dial = next(m for m in switching.scene.mobjects if isinstance(m, Rig))
    lamp = dial.part("lamp")
    checked = 0
    for play in record.plays:
        for capture in play.samples:
            shown = record.node_state(dial, capture).get("_shown")
            # The pose switches the lamp on at its first instant; the scene
            # switches it off before the second play.
            assert shown == ("on" if play.index == 0 else "off")

            def points_of(m, capture=capture):
                state = record.node_state(m, capture)
                return state.get("_points") if state is not None else m.points

            _invariants(lamp, shown, points_of)
            checked += 1
    assert checked > 20
