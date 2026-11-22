"""The snap without the point-equality measure: production does not measure
(``canonical.MEASURE`` is off unless set), the test session does, and a snap
that does not measure hands the core the same deltas, byte for byte, as one
that does. The session's guard (``tests/conftest.py``) fails a test that
snaps a shape unmeasured, so it cannot pass on nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from arachne.frame_ir import canonical, joins
from arachne.frame_ir.canonical import Canonical
from arachne.frame_ir.xf import Delta, PaintTable, Record, Shape

from ..conftest import point_equality_guard

#: A flat view of the configured frame at 854x480.
_FLAT = {
    "pixel_width": 854,
    "pixel_height": 480,
    "frame_width": 14.222222222222221,
    "frame_height": 7.993754879000781,
    "frame_center": (0.0, 0.0, 0.0),
    "mapping_centre": (0.0, 0.0, 0.0),
    "cairo_line_width_multiple": 0.01,
    "use_z_index": True,
    "background": (0.0, 0.0, 0.0, 1.0),
    "camera": "flat",
}
#: 1/256 px in frame units at 854 px.
_UNIT = 1 / (60.046875 * 256)
#: The flat view with a frame twice as wide: a pixel spans twice the units,
#: so the input grid changes.
_WIDE = dict(
    _FLAT,
    frame_width=2 * _FLAT["frame_width"],
    frame_height=2 * _FLAT["frame_height"],
)


def _stroked(shape: Shape, **fields) -> Record:
    return Record(
        "path",
        {
            "shape": shape.key,
            "fill": None,
            "stroke": np.array([[1.0, 1.0, 1.0, 1.0]]),
            "stroke_width": 4.0,
            "joint": "auto",
            "cap": "auto",
            "space": "screen",
            "z_index": 0.0,
            "shade_in_3d": False,
            **fields,
        },
    )


def _square_closing(gap: float) -> np.ndarray:
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [gap, 0, 0]], float)
    return np.concatenate(
        [np.linspace(a, b, 4) for a, b in zip(corners[:-1], corners[1:], strict=True)]
    )


def _sharp() -> np.ndarray:
    """A path whose reversal turns on arms of a few 1/256 px: a pan by a
    fraction of a pixel moves Cairo's decision of its join.
    """
    j = np.array([1.0, 0.5, 0.0])
    start, end = j - [1.0, 0, 0], j + [-1.0, 0.05, 0]
    return np.array(
        [
            start,
            start + [0.3, 0, 0],
            j - [6.3 * _UNIT, -1.4 * _UNIT, 0],
            j,
            j,
            j + [-2.4 * _UNIT, 0.7 * _UNIT, 0],
            end + [0.3, 0, 0],
            end,
        ]
    )


def _panned_until_moved(points: np.ndarray) -> list[dict]:
    """Views panned step by step until one moves ``points``' joins, then
    a pan back: views that move them and views that do not.
    """
    views = [dict(_FLAT)]
    seen = {joins.joins(points, _FLAT)}
    for k in range(1, 64):
        shift = (k * _UNIT / 7, 0.0, 0.0)
        moved = dict(_FLAT, frame_center=shift, mapping_centre=shift)
        views.append(moved)
        if joins.joins(points, moved) not in seen:
            break
    else:
        pytest.fail("no pan moved the decision")
    return [*views, dict(_FLAT)]


def _two_faces(shape: Shape) -> tuple[Record, list[PaintTable]]:
    fills = PaintTable(np.array([[1.0, 0, 0, 1], [0, 0, 1, 1]]))
    strokes = PaintTable(np.array([[1.0, 1, 1, 1], [1, 1, 1, 1]]))
    record = Record(
        "face_group",
        {
            "shape": shape.key,
            "fill": None,
            "stroke": None,
            "background_stroke": None,
            "stroke_width": 1.0,
            "background_stroke_width": 0.0,
            "joint": "auto",
            "cap": "auto",
            "face_fill": fills.key,
            "face_stroke": strokes.key,
            "uv": np.array([[0, 0], [1, 0]], dtype=np.uint32),
            "grid": (2, 1),
            "space": "screen",
            "z_index": 0.0,
            "shade_in_3d": False,
        },
    )
    return record, [fills, strokes]


def _stream() -> list[Delta]:
    """A stream through every branch the measure sits on: uploads, a path
    whose joins a pan moves (set again by the view pass), one revealed, a
    face group whose faces a path also draws (measured two ways), a shape a
    record begins to draw another way in a later delta, a release, and a
    wider frame that changes the grid (everything snapped again).
    """
    sharp = Shape("path", _sharp())
    first = _square_closing(0.0)
    pair = Shape("path", np.concatenate([first, first + [2e-6, 0, 0]]))
    group, tables = _two_faces(pair)
    plain = Shape("path", _square_closing(3e-5))
    deltas = [
        Delta(
            seq=0,
            view=dict(_FLAT),
            upload=[sharp, pair, plain],
            upload_paints=tables,
            set=[
                (1, _stroked(sharp)),
                (2, group),
                (3, _stroked(plain, reveal=(0.0, 0.5))),
            ],
        )
    ]
    views = _panned_until_moved(_sharp())
    for seq, view in enumerate(views[1:], start=1):
        deltas.append(Delta(seq=seq, view=view))
    seq = len(deltas)
    deltas.append(Delta(seq=seq, set=[(4, _stroked(pair))]))
    deltas.append(Delta(seq=seq + 1, view=_WIDE))
    deltas.append(Delta(seq=seq + 2, remove=[3], release=[plain.key]))
    deltas.append(Delta(seq=seq + 3, view=dict(_WIDE, frame_center=(0.1, 0, 0))))
    return deltas


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, Shape) and isinstance(b, Shape):
        return a.kind == b.kind and a.key == b.key and _same(a.data, b.data)
    if isinstance(a, Record) and isinstance(b, Record):
        return a.kind == b.kind and _same(a.fields, b.fields)
    if isinstance(a, PaintTable) and isinstance(b, PaintTable):
        return a.key == b.key and _same(a.rows, b.rows)
    return type(a) is type(b) and a == b


def _handed(measure: bool) -> list[Delta]:
    snapper = Canonical(planes=False, measure=measure)
    found = [snapper(delta) for delta in _stream()]
    # The stream's pair of squares is decided otherwise by design (counted
    # when measured): not this comparison's subject.
    canonical.point_equality_disagreements.clear()
    return found


_PARTS = (
    "t",
    "seq",
    "objects",
    "view",
    "upload",
    "upload_paints",
    "release",
    "set",
    "remove",
    "order",
    "source",
)


def test_production_does_not_measure():
    """Imported as production imports it, with no test session around it,
    the snap does not measure.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    probe = (
        "from arachne.frame_ir import canonical\n"
        "print(canonical.MEASURE, canonical.Canonical(planes=False)._measuring)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["False", "False"]


def test_the_test_session_measures():
    assert canonical.MEASURE is True
    assert Canonical(planes=False)._measuring is True
    assert Canonical(planes=False, measure=False)._measuring is False


def test_the_stream_regrids():
    snapper = Canonical(planes=False)
    grids = []
    for delta in _stream():
        snapper(delta)
        grids.append(snapper.grid)
    canonical.point_equality_disagreements.clear()
    assert len(set(grids)) > 1, grids


def test_the_stream_sets_a_path_again_for_its_joins():
    """The stream exercises the view pass: some delta that only moves the
    view sets the sharp path again, with other joins.
    """
    handed = _handed(True)
    first = handed[0].set[0][1].fields["joins"]
    again = [
        d.set
        for d in handed[1:]
        if d.view is not None and any(s == 1 for s, _ in d.set)
    ]
    assert again
    assert any(dict(s)[1].fields["joins"] != first for s in again)


@pytest.mark.snaps_unmeasured
def test_unmeasured_hands_on_the_same_deltas():
    measured, unmeasured = _handed(True), _handed(False)
    assert len(measured) == len(unmeasured)
    for k, (a, b) in enumerate(zip(measured, unmeasured, strict=True)):
        for name in _PARTS:
            assert _same(getattr(a, name), getattr(b, name)), (k, name)


@pytest.mark.snaps_unmeasured
def test_unmeasured_hands_the_core_the_same_bytes():
    pytest.importorskip("arachne_core")
    measured, unmeasured = _handed(True), _handed(False)
    sent = [
        (a.to_core().to_bytes(), b.to_core().to_bytes())
        for a, b in zip(measured, unmeasured, strict=True)
    ]
    assert all(a for a, _ in sent)
    for k, (a, b) in enumerate(sent):
        assert a == b, k


@pytest.mark.snaps_unmeasured
def test_measuring_measures_and_unmeasured_does_not(monkeypatch):
    """Measured, the face group's pair of squares is measured (a path draws
    them as one, which joins where the faces do not) and the disagreement
    counted; unmeasured, nothing is measured and nothing kept for it.
    """
    calls = {True: 0, False: 0}
    real = canonical._measure
    measuring_now = [True]

    def counting(points, snapped, uses):
        calls[measuring_now[0]] += 1
        return real(points, snapped, uses)

    monkeypatch.setattr(canonical, "_measure", counting)
    found, counts = {}, {}
    for measure in (True, False):
        measuring_now[0] = measure
        canonical.point_equality_disagreements.clear()
        canonical.point_equality_counts.update(measured=0, unmeasured=0)
        snapper = Canonical(planes=False, measure=measure)
        for delta in _stream():
            snapper(delta)
        found[measure] = list(canonical.point_equality_disagreements)
        counts[measure] = dict(canonical.point_equality_counts)
        if not measure:
            assert snapper._measured == {}
    canonical.point_equality_disagreements.clear()
    assert calls[True] > 0
    assert found[True]
    assert counts[True]["measured"] > 0
    assert counts[True]["unmeasured"] == 0
    assert calls[False] == 0
    assert found[False] == []
    assert counts[False]["measured"] == 0
    assert counts[False]["unmeasured"] > 0


@pytest.mark.snaps_unmeasured
def test_the_guard_fails_a_test_that_snaps_unmeasured():
    with (
        pytest.raises(AssertionError, match="without the point-equality measure"),
        point_equality_guard(),
    ):
        _handed(False)


def test_the_guard_passes_a_test_that_snaps_measured():
    with point_equality_guard():
        _handed(True)


def test_the_guard_fails_a_session_that_does_not_measure(monkeypatch):
    monkeypatch.setattr(canonical, "MEASURE", False)
    with (
        pytest.raises(AssertionError, match="canonical.MEASURE"),
        point_equality_guard(),
    ):
        pass


def test_the_guard_fails_a_disagreement():
    with (
        pytest.raises(AssertionError, match="decided otherwise"),
        point_equality_guard(),
    ):
        canonical.point_equality_disagreements.append((8, 2, 1))
