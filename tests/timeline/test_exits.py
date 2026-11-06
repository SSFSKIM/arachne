"""Where each body stood as each play ended (``Record.exits``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arachne.timeline import recorder

from .conftest import render
from .exit_scenes import ArmedLate

EXITS = Path(__file__).parent / "exit_scenes.py"

_FIELDS = ("id", "parent", "members", "own", "visible", "type", "text", "scene_line")


def _key_of(exit, type_name):
    """The stable id of the one top-level body of ``type_name``."""
    (key,) = [
        k for k, b in exit.bodies.items() if b.type == type_name and b.parent is None
    ]
    return key


def test_each_exit_reads_as_a_fresh_reading_would(session_of, monkeypatch):
    """What the recorder keeps from one reading to the next (each play's
    entry, then its exit) is only a shortcut: every one is what reading the
    scene afresh finds.
    """
    readings = []
    read = recorder.bodies

    def compared(*args, **kwargs):
        kept = read(*args, **kwargs)
        kwargs.pop("kept")
        readings.append((kept, read(*args, **kwargs)))
        return kept

    monkeypatch.setattr(recorder, "bodies", compared)
    session = session_of("Changes", EXITS, probe=False)
    assert len(readings) == 2 * len(session.record.plays) == 10
    for (zoom, kept), (fresh_zoom, fresh) in readings:
        assert zoom == fresh_zoom
        assert list(kept) == list(fresh)
        for key, body in kept.items():
            again = fresh[key]
            for field in _FIELDS:
                assert getattr(body, field) == getattr(again, field), (key, field)
            for field in ("centre", "size", "view"):
                assert np.array_equal(getattr(body, field), getattr(again, field))


def test_a_live_camera_picture_is_read_again_at_every_exit(session_of, monkeypatch):
    """A display of a camera holds a view of the camera's own pixels, which
    change under it: the same object is no proof it shows the same picture.
    """
    readings = []
    read = recorder.bodies

    def compared(*args, **kwargs):
        kept = read(*args, **kwargs)
        kwargs.pop("kept")
        readings.append((kept, read(*args, **kwargs)))
        return kept

    monkeypatch.setattr(recorder, "bodies", compared)
    exits = session_of("LiveImage", EXITS, probe=False).record.exits
    display = _key_of(exits[0], "ImageMobjectFromCamera")
    seen = [
        (kept[1][display].visible, again[1][display].visible)
        for kept, again in readings
    ]
    # Each wait's entry, then its exit.
    assert seen == [(False, False), (False, False), (True, True), (True, True)]
    # An image that owns its pixels is still read once.
    still = _key_of(exits[0], "ImageMobject")
    assert exits[1].bodies[still] is exits[0].bodies[still]


def test_what_a_play_changed_is_read_again(session_of):
    exits = session_of("Changes", EXITS, probe=False).record.exits
    square = _key_of(exits[0], "Square")
    assert [exits[i].bodies[square].visible for i in range(5)] == [
        True,
        False,
        False,
        False,
        True,
    ]
    pair = _key_of(exits[0], "VGroup")
    assert [exits[i].bodies[pair].members for i in range(5)] == [2, 2, 3, 3, 3]
    label = _key_of(exits[0], "Text")
    # The pan moves the label on screen, not in the scene.
    assert np.array_equal(exits[3].bodies[label].centre, exits[2].bodies[label].centre)
    moved = exits[3].bodies[label].view - exits[2].bodies[label].view
    assert moved == pytest.approx([1.0, 0.0])


def test_a_session_keeps_exits_from_the_arming_on(session_of):
    record = session_of("ArmedLate", EXITS, probe=False).record
    assert len(record.plays) == 4
    assert record.armed_at == 2
    assert set(record.exits) == {2, 3}


def test_a_scene_that_never_arms_reads_no_exit(session_of, monkeypatch):
    read = []
    monkeypatch.setattr(recorder, "bodies", lambda *a, **k: read.append(a))
    record = session_of("Unarmed", EXITS, probe=False).record
    assert record.exits is None
    assert record.armed_at is None
    assert read == []


def test_a_render_keeps_no_exit_even_armed():
    scene, _ = render(ArmedLate, 6)
    assert scene.timeline.record.exits is None
    assert scene.timeline.record.armed_at is None


def test_a_body_no_play_changed_is_read_once(session_of):
    exits = session_of("Changes", EXITS, probe=False).record.exits
    label = _key_of(exits[0], "Text")
    assert exits[1].bodies[label] is exits[0].bodies[label]
    assert exits[2].bodies[label] is exits[0].bodies[label]
    dot = _key_of(exits[0], "Dot")
    assert exits[1].bodies[dot] is exits[0].bodies[dot]
