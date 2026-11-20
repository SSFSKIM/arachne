"""The layer seam: a spec set on an object routes its scene and records what
its effect reads; nothing of it changes a scene that uses nothing new.
"""

from __future__ import annotations

import importlib
import json
from fractions import Fraction

import pytest

from arachne import Circle, Dot, Square, api_json
from arachne.compose.layer import LayerSpec, layer_of, set_focus, set_layer
from arachne.routing import declare, held
from arachne.timeline.execution import summarize
from arachne.timeline.footprint import ReplayUnsupported, frame_without, object_map

record_module = importlib.import_module("arachne.timeline.record")


def test_a_spec_is_checked_when_made():
    with pytest.raises(ValueError, match="positive"):
        LayerSpec(blur=0)
    with pytest.raises(ValueError, match="need a mask"):
        LayerSpec(mask_feather=0.1)
    with pytest.raises(ValueError, match="at least one"):
        LayerSpec(window=())
    with pytest.raises(ValueError, match="aperture"):
        set_focus(Square(), 0, -1)


def test_set_layer_declares_its_capabilities_and_withdraws_those_it_drops():
    square, mask, dot = Square(), Circle(), Dot()
    set_layer(square, LayerSpec(blur=0.1, mask=mask, depth=1))
    assert held(square) == {"effect.blur", "effect.soft_mask", "layer.depth"}
    assert layer_of(square).mask is mask
    assert vars(square)[record_module.DEPENDENCIES] == (mask,)
    set_layer(square, LayerSpec(window=(dot,)))
    assert held(square) == {"effect.window"}
    assert vars(square)[record_module.DEPENDENCIES] == (dot,)
    set_layer(square, None)
    assert held(square) == frozenset()
    assert layer_of(square) is None
    assert record_module.DEPENDENCIES not in vars(square)
    with pytest.raises(ValueError, match="itself"):
        set_layer(square, LayerSpec(mask=square))


def test_a_row_declared_on_its_own_outlives_every_spec():
    square, mask = Square(), Circle()
    declare(square, "effect.depth_of_field")
    set_layer(square, LayerSpec(blur=0.1))
    set_layer(square, LayerSpec(mask=mask))
    assert held(square) == {"effect.depth_of_field", "effect.soft_mask"}
    set_layer(square, None)
    assert held(square) == {"effect.depth_of_field"}


def test_the_seam_is_declared_by_the_package():
    import arachne.compose as compose

    assert compose.LayerSpec is LayerSpec
    assert compose.set_layer is set_layer
    assert compose.set_focus is set_focus


def test_an_effect_in_the_last_play_routes_the_scene_and_without_it_nothing(
    session_of,
):
    assert session_of("EffectLast").route == "gpu"
    plain = session_of("EffectLastRemoved")
    assert plain.route == "cpu"
    assert plain.renderer is None


def test_an_object_drawn_only_through_a_layer_is_recorded_at_every_capture(
    session_of,
):
    session = session_of("MovingMask")
    record = session.record
    (square,) = [
        mob
        for mob in record.mobjects()
        if layer_of(mob) is not None and layer_of(mob).mask is not None
    ]
    mask = layer_of(square).mask
    centres = []
    for t in (Fraction(1, 4), Fraction(3, 4)):
        _, capture = record.capture_at(t)
        reached = {id(mob): state for mob, state in record._walk(capture, [square])}
        assert id(mask) in reached
        centres.append(reached[id(mask)].get("_points").mean(axis=0)[0])
    # It never joined the scene, yet each capture holds where it was then.
    assert all(mask is not mob for mob in record.final.roots)
    assert centres[1] - centres[0] == pytest.approx(4)


def test_a_scene_with_no_such_object_records_what_it_recorded_before(
    session_of, monkeypatch
):
    """The walk through capture dependencies adds nothing where no object
    holds any: the record, as the store and a render's check read it, is
    the same with it and without it.
    """

    def forms(session):
        record = session.record
        states = [
            (record.origin(node.mob).id, len(node.states))
            for node in record._nodes.values()
        ]
        return (
            summarize(record).to_json(),
            json.dumps(api_json.inspect(session), sort_keys=True, default=str),
            states,
        )

    with_walk = forms(session_of("Plain"))
    monkeypatch.setattr(record_module, "dependencies", lambda state: ())
    assert forms(session_of("Plain")) == with_walk


def test_an_object_drawn_only_through_a_layer_is_never_left_out_alone(session_of):
    session = session_of("MovingMask")
    record = session.record
    holders = [mob for mob in record.mobjects() if layer_of(mob) is not None]
    (square,) = [mob for mob in holders if layer_of(mob).mask is not None]
    (frame,) = [mob for mob in holders if layer_of(mob).window is not None]
    for carrier, inner in (
        (square, layer_of(square).mask),
        (frame, layer_of(frame).window[0]),
    ):
        name, holder = record.origin(inner).id, record.origin(carrier).id
        for ask in (frame_without, object_map):
            with pytest.raises(ReplayUnsupported) as refused:
                ask(session, Fraction(1, 4), [name])
            assert name in str(refused.value)
            assert holder in str(refused.value)
