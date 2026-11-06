"""The leaf memo answers as reading does: a lint with it builds every part
as one without it, a read it serves leaves the camera as the read would, a
view differing in any one field places a leaf again, a live read is never
kept, and it keeps the places of its latest cameras only.
"""

from __future__ import annotations

import dataclasses
import functools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.camera.projection import ThreeDView, View, record_view
from arachne.lint import RULES, frames, lint, serialise
from arachne.lint import leaf_memo as leaf_memo_module
from arachne.lint.core import parts_reader
from arachne.lint.leaf_memo import LeafMemo
from arachne.lint.parts import PartsCache, build_parts
from arachne.timeline.fallback import visit_frames
from arachne.timeline.routes import _resolve

from . import memo_scenes
from .test_leaf_memo import _counted, _is_probe

SCENES = Path(memo_scenes.__file__)
FPS = 15


def _view_changes(view: View):
    """``view`` with each of its fields, and each of its 3D projection's,
    changed alone.
    """

    def changed(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return value + 0.5
        if isinstance(value, tuple):
            return tuple(changed(v) for v in value)
        if isinstance(value, str):
            return "flat" if value != "flat" else "moving"
        raise AssertionError(f"no change for {value!r}")

    for field in dataclasses.fields(View):
        if field.name == "three_d":
            continue
        yield (
            field.name,
            dataclasses.replace(
                view, **{field.name: changed(getattr(view, field.name))}
            ),
        )
    assert view.three_d is not None
    for field in dataclasses.fields(ThreeDView):
        three_d = dataclasses.replace(
            view.three_d,
            **{field.name: changed(getattr(view.three_d, field.name))},
        )
        yield f"three_d.{field.name}", dataclasses.replace(view, three_d=three_d)


def test_a_view_that_differs_in_any_one_field_places_the_leaf_again(session_of):
    session = session_of("HeldProbe3D", SCENES, frame_rate=FPS)
    record = session.record
    capture = _resolve(record, Fraction(3, FPS)).capture
    camera = record._scene.renderer.camera
    (mob,) = [m for m in capture.draw_list() if _is_probe(m)]
    view = record_view(record)
    view_of = {"as read": view, **dict(_view_changes(view))}
    memo = LeafMemo()
    made: list[str] = []

    def place(name):
        viewing = memo.viewing(record, capture, camera, view_of[name])
        assert viewing is not None
        node = viewing.node(mob, ())
        assert node is not None
        viewing.lays(node, lambda: None)
        viewing.placed(node, lambda: made.append(name))

    place("as read")
    place("as read")
    assert made == ["as read"]
    for name in view_of:
        if name != "as read":
            place(name)
    assert made == list(view_of)


def test_a_live_read_is_never_kept(session_of, monkeypatch):
    session = session_of("HeldProbe3D", SCENES, frame_rate=FPS)
    cache = PartsCache()
    read = parts_reader(session, cache)
    counts = _counted(monkeypatch)

    def visit(t, forward, play, capture):
        read(forward, t, play, capture, True)

    instants = [Fraction(k, FPS) for k in (1, 3, 5, 8)]
    visit_frames(session, instants, visit, to_end=True, states=True)
    assert counts["reads"] == len(instants)
    assert counts["lays"] == counts["placed"] == counts["reads"]
    assert len(cache.memo) == 0


def _builds(session, cache: PartsCache, instants) -> list[np.ndarray]:
    """The parts at each of ``instants``, in turn, through ``cache``; the
    camera's rotation as each build left it.
    """
    record = session.record
    camera = record._scene.renderer.camera
    left = []
    for t in instants:
        resolved = _resolve(record, t)
        cache.build(
            record,
            t,
            resolved.play,
            resolved.capture,
            answer="recorded",
            target=(96, 54),
        )
        left.append(np.array(camera.rotation_matrix))
    return left


def test_a_read_served_from_the_memo_leaves_the_camera_s_rotation_as_a_read_would(
    session_of,
):
    """A ThreeDCamera's next depth sort reads the rotation its last
    projection made: a read whose every leaf is kept leaves it made for its
    own angles, as reading them would.
    """
    session = session_of("TurnsBetweenHolds3D", SCENES, frame_rate=FPS)
    # A hold, then the turn, then the same hold's capture again.
    instants = [Fraction(2, FPS), Fraction(9, FPS), Fraction(4, FPS)]
    assert (
        _resolve(session.record, instants[0]).capture
        is _resolve(session.record, instants[2]).capture
    )
    fresh = _builds(session, PartsCache(memo=False), instants)
    kept = _builds(session, PartsCache(), instants)
    assert not np.array_equal(fresh[0], fresh[1])
    for a, b in zip(fresh, kept, strict=True):
        np.testing.assert_array_equal(a, b)


def test_the_memo_keeps_the_places_of_its_latest_cameras_only(session_of, monkeypatch):
    monkeypatch.setattr(leaf_memo_module, "CAMERAS_KEPT", 2)
    session = session_of("TurningProbe3D", SCENES, frame_rate=FPS)
    record = session.record
    memo = LeafMemo()
    counts = _counted(monkeypatch)
    for k in (1, 2, 3, 1):
        t = Fraction(k, FPS)
        resolved = _resolve(record, t)
        build_parts(
            record,
            t,
            resolved.play,
            resolved.capture,
            answer="recorded",
            target=(96, 54),
            memo=memo,
        )
    assert memo.cameras == 2
    # The first camera was let go before it was read again.
    assert counts["placed"] == 4
    assert counts["lays"] == 1


def _flat(value):
    """``value`` as plain, comparable data: dataclasses field by field,
    arrays by shape, type and bytes.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            tuple(
                (f.name, _flat(getattr(value, f.name)))
                for f in dataclasses.fields(value)
            ),
        )
    if isinstance(value, np.ndarray):
        return ("array", value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, (tuple, list)):
        return tuple(_flat(v) for v in value)
    if isinstance(value, frozenset):
        return tuple(sorted(_flat(v) for v in value))
    if isinstance(value, dict):
        return tuple(sorted((k, _flat(v)) for k, v in value.items()))
    return value


def _lint_and_parts(session_of, monkeypatch, scene, *, memo: bool):
    """The lint's report, and every parts it built in order, flattened."""
    session = session_of(scene, SCENES, frame_rate=FPS)
    built = []
    build = PartsCache.build

    def kept(self, *args, **kwargs):
        parts = build(self, *args, **kwargs)
        built.append(_flat(parts))
        return parts

    with monkeypatch.context() as patched:
        patched.setattr(PartsCache, "build", kept)
        patched.setattr(frames, "PartsCache", functools.partial(PartsCache, memo=memo))
        result = lint(session, rules=RULES)
    report = serialise.report(result)
    report.pop("duration_s")  # a timing
    return report, built


@pytest.mark.parametrize(
    "scene",
    [
        "HeldProbe3D",
        "HeldProbe2D",
        "TurningProbe3D",
        "TurnsBetweenHolds3D",
        "ProbeShifts",
        "ProbeFillChanges",
        "CameraZoom",
        "CameraFrameCentre",
        "ProbeFixedInFrame",
        "FixedOrientationCentreMoves",
        "MovingFrameWidens",
        "ProbeInUpdatedGroup",
        "RedrawnProbe",
    ],
)
def test_a_lint_with_the_memo_reads_every_part_as_one_without_it(
    session_of, monkeypatch, scene
):
    fresh = _lint_and_parts(session_of, monkeypatch, scene, memo=False)
    kept = _lint_and_parts(session_of, monkeypatch, scene, memo=True)
    assert fresh[1]
    assert kept == fresh


def test_a_kept_place_is_read_only(session_of):
    session = session_of("HeldProbe3D", SCENES, frame_rate=FPS)
    record = session.record
    cache = PartsCache()
    for k in (2, 4):
        t = Fraction(k, FPS)
        resolved = _resolve(record, t)
        parts = cache.build(
            record,
            t,
            resolved.play,
            resolved.capture,
            answer="recorded",
            target=(96, 54),
        )
    (leaf,) = [
        leaf for part in parts.parts for leaf in part.leaves if leaf.type == "Probe"
    ]
    with pytest.raises(ValueError, match="read-only"):
        leaf.points[0, 0] = 0.0
