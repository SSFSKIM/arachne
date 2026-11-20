"""A value at a point: ``query``'s ``value_at`` reads at the request's own
time, from what the record owns, and never runs the scene; outside an
object's domain or its bake's stay it answers ``outside``, never a number.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from arachne import api_json
from arachne import session as session_module
from arachne.quantities import PointProbe, PointRead, point_answers, value_at
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import RpcError
from arachne.session import Session
from arachne.timeline import admission

from . import quantity_scenes

SCENES = Path(quantity_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}


@pytest.fixture(scope="module")
def field():
    admission.DEMOTED.clear()
    session = Session.open(SCENES, "TwoPointField", config=SMALL)
    yield session
    session.close()
    admission.DEMOTED.clear()


def _ids(session: Session) -> dict[str, str]:
    state = session.state_at(0)
    return {node.name: node.id for node in state.nodes if node.name}


def _read(session: Session, t, name: str, position, space="scene", channel=None):
    ids = _ids(session)
    channel = channel or ("flow" if name == "flow" else "temperature")
    probe = PointProbe(ids.get(name, name), channel, tuple(position), space)
    (found,) = value_at(session.record, t, [probe])
    return found


def test_two_point_fixture_one_field_two_positions_and_outside_its_domain(field):
    """One declared scalar field read at two positions in one channel returns
    its two declared values, and a position outside its domain returns
    ``outside``, not a number.
    """
    assert _read(field, 0, "grid", (0, 0, 0), "grid") == PointRead(
        "value", 3.0, "K", "nearest", ((0, 0, 0),), Fraction(0)
    )
    assert _read(field, 0, "grid", (1, 0, 0)) == PointRead(
        "value", 7.0, "K", "nearest", ((1, 0, 0),), Fraction(0)
    )
    assert _read(field, 0, "grid", (5, 0, 0)) == PointRead("outside", why="domain")


def test_the_fixture_s_clock_and_interpolation(field):
    # Tick 1, though no frame of the 15 fps scene is drawn there.
    tick = _read(field, Fraction(1, 60), "grid", (1, 0, 0))
    assert (tick.value, tick.as_of) == (8.0, Fraction(1, 60))
    # Between ticks, the last tick at or before.
    assert _read(field, Fraction(1, 120), "grid", (1, 0, 0)).value == 7.0
    assert _read(field, Fraction(2), "grid", (1, 0, 0)).value == 8.0
    # Past the bake's last tick (120), past the scene, before it.
    for t in (Fraction(5, 2), Fraction(121, 60), 4, -1):
        assert _read(field, t, "grid", (1, 0, 0)) == PointRead("outside", why="time")
    assert _read(field, 0, "linear", (0, 0, 0)) == PointRead(
        "value", 5.0, "K", "linear", ((0, 0, 0), (1, 0, 0)), Fraction(0)
    )
    # Linear reads between the outermost centres only.
    assert _read(field, 0, "linear", (-1.5, 0, 0)).why == "domain"


def test_a_scalar_channel_answers_a_number_and_a_vector_channel_a_vector(field):
    assert _read(field, 1, "grid", (1, 0, 0)).value == 8.0
    flow = _read(field, 1, "flow", (0, 0, 0))
    assert (flow.value, flow.unit) == ((1.0, 2.0), "1")


def test_what_is_not_there_is_unknown(field):
    assert _read(field, 0, "nowhere.py:1#0", (0, 0, 0)).why == "id"
    assert _read(field, 0, "grid", (0, 0, 0), channel="pressure").why == "channel"
    plain = _read(field, 0, "plain", (0, 0, 0))
    assert (plain.result, plain.why) == ("unknown", "not_point_readable")


def test_a_warm_point_read_executes_the_scene_zero_times(field, monkeypatch):
    executions = [0]
    load = session_module._import

    def counting(path):
        executions[0] += 1
        return load(path)

    monkeypatch.setattr(session_module, "_import", counting)
    runs = len(field.forward_runs)
    ids = _ids(field)
    probes = [
        PointProbe(ids["grid"], "temperature", (1.0, 0.0, 0.0)),
        PointProbe(ids["linear"], "temperature", (0.0, 0.0, 0.0)),
    ]
    for t in (0, Fraction(1, 60), Fraction(7, 3), 3):
        point_answers(field, t, probes)
    assert executions[0] == 0
    assert len(field.forward_runs) == runs


def _valid(form: str, out: dict) -> dict:
    read = json.loads(json.dumps(out, allow_nan=False))
    validator = Draft202012Validator(api_json.schema(form))
    errors = [f"{list(e.path)}: {e.message}" for e in validator.iter_errors(read)]
    assert not errors, errors
    return read


@pytest.fixture
def server():
    admission.DEMOTED.clear()
    found = Dispatcher()
    found.handle(
        "open", {"file": str(SCENES), "scene": "TwoPointField", "config": SMALL}
    )
    yield found
    found.handle("close", {})
    admission.DEMOTED.clear()


def test_geometry_and_value_at_in_one_request_each_on_its_clock(server):
    ids = _ids(server.live())
    grid = ids["grid"]
    probes = [
        {"id": grid, "channel": "temperature", "position": [1, 0, 0], "space": "scene"},
        {"id": grid, "channel": "temperature", "position": [5, 0, 0], "space": "scene"},
        {"id": "nowhere.py:1#0", "channel": "temperature", "position": [0]},
    ]
    both = server.handle(
        "query",
        {"t": "1/60", "what": ["bbox", "value_at"], "ids": [grid], "probes": probes},
    )["query"]
    _valid("query", both)
    # The frame on screen at 1/60 is the 15 fps grid's first, at 0.
    assert both["t_exact"] == "0/1"
    assert both["bbox"][grid] is not None
    value, outside, unknown = both["value_at"]
    assert value == {
        **probes[0],
        "result": "value",
        "value": 8.0,
        "unit": "K",
        "interpolation": "nearest",
        "cells": [[1, 0, 0]],
        "as_of": 1 / 60,
        "as_of_exact": "1/60",
    }
    assert outside == {**probes[1], "result": "outside", "outside": "domain"}
    assert unknown == {
        **probes[2],
        "position": [0.0],
        "space": "scene",
        "result": "unknown",
        "unknown": "id",
    }

    alone = server.handle(
        "query", {"t": "1/60", "what": "value_at", "probes": probes[:1]}
    )["query"]
    _valid("query", alone)
    # Asked alone, the answer's time is the request's.
    assert (alone["t_exact"], alone["answer"], alone["play"]) == ("1/60", "recorded", 0)
    assert alone["value_at"] == [value]
    assert alone["frame"] == both["frame"]


@pytest.mark.parametrize(
    "params",
    [
        {"what": "value_at"},
        {"what": "value_at", "probes": []},
        {"what": "value_at", "probes": [{"id": "a", "channel": "c"}]},
        {"what": "value_at", "probes": [{"id": "a", "channel": "c", "position": []}]},
        {
            "what": "value_at",
            "probes": [{"id": "a", "channel": "c", "position": [0], "space": "x"}],
        },
        {
            "what": "value_at",
            "probes": [{"id": "a", "channel": "c", "position": [0], "at": 1}],
        },
        {"what": "value_at", "probes": [{"id": 1, "channel": "c", "position": [0]}]},
    ],
)
def test_a_point_read_s_params_are_checked(server, params):
    with pytest.raises(RpcError) as raised:
        server.handle("query", {"t": 0, **params})
    assert raised.value.code == -32602
