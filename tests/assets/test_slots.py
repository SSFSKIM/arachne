"""A slot shows exactly one of its drawings: the one it says it shows has
its points, posed with the slot's part, and every member of every other
drawing's family holds none, at every capture of a play that switches it,
the rig moving as a whole in the same play or not; each drawing keeps its
own paint through the switch, and a style given to the rig, at once or by
an animation, reaches the drawings it does not show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arachne import UP
from arachne.assets import Rig
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.draw import Drawing

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


def _paint(drawing) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.asarray(m.get_fill_rgbas()), np.asarray(m.get_stroke_rgbas()))
        for m in drawing.submobjects
    ]


def _drawing(rig: Rig, slot: str, name: str):
    return next(d for d in rig.part(slot).submobjects if d.name == name)


@pytest.mark.parametrize(
    "scene",
    ["SwitchThenMove", "MoveThenSwitch", "SwitchThenGroupMove", "GroupMoveThenSwitch"],
)
def test_a_slot_switched_while_its_rig_moves_shows_its_drawing_in_place_in_its_paint(
    opened, scene
):
    session = opened(scene)
    record = session.record
    (play,) = record.plays
    dial = next(
        m for m in session.scene.get_mobject_family_members() if isinstance(m, Rig)
    )
    asset = dial.asset
    fresh = Rig(asset)
    paints = {name: _paint(_drawing(fresh, "lamp", name)) for name in ("on", "off")}
    start = fresh.placement()
    checked = 0
    with Drawing(record) as drawing:
        for capture in play.samples:
            drawing.show(capture, capture.roots)
            _invariants(dial.part("lamp"), "on")
            # Each part where the pose and the move put it.
            reference = Rig(asset)
            reference.place(dial.placement())
            reference.pose(reading=dial.value("reading")).show(lamp="on")
            for name in ("face", "needle", "lamp"):
                for got, want in zip(
                    dial.part(name).family_members_with_points(),
                    reference.part(name).family_members_with_points(),
                    strict=True,
                ):
                    np.testing.assert_allclose(got.points, want.points, atol=1e-9)
            # Each drawing in its own paint, the one shown and the one hidden.
            for name, paint in paints.items():
                for (fill, stroke), (want_fill, want_stroke) in zip(
                    _paint(_drawing(dial, "lamp", name)), paint, strict=True
                ):
                    np.testing.assert_allclose(fill, want_fill)
                    np.testing.assert_allclose(stroke, want_stroke)
            checked += 1
    assert checked >= 15
    # The play ends with the dial moved up, on, at its target.
    assert dial.shown("lamp") == "on"
    assert dial.value("reading") == 90
    moved = dial.placement()
    np.testing.assert_allclose(moved.offset - start.offset, UP, atol=1e-9)
    np.testing.assert_allclose(moved.linear, start.linear, atol=1e-12)


def test_a_drawing_switched_in_after_an_animated_fade_has_the_faded_opacity(opened):
    session = opened("FadedThenSwitched")
    dial = next(m for m in session.scene.mobjects if isinstance(m, Rig))
    _invariants(dial.part("lamp"), "on")
    for shape in _drawing(dial, "lamp", "on").submobjects:
        assert shape.get_fill_opacity() == pytest.approx(0.4)
        assert shape.get_stroke_opacity() == pytest.approx(0.4)
