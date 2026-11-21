"""``ARACHNE_SNAP_LEAN``: the snap without its two pieces of fixed waste, the
second joins decision on a path the view pass already decided, and the
test-only point-equality measure. Off, nothing changes; on, the deltas the
core is handed are the same.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir import canonical, joins
from arachne.frame_ir.canonical import Canonical
from arachne.frame_ir.xf import Delta, PaintTable, Record, Shape

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
    one more step and a pan back: views that move them and views that do not.
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
    """A stream that takes every branch the switch touches: a path whose
    joins a pan moves (set again by the view pass), one revealed, a face
    group whose faces a path also draws (measured two ways), a shape a
    record begins to draw another way in a later delta, a release, and a
    zoom that changes the grid (everything snapped again).
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
    deltas.append(Delta(seq=seq + 1, view=dict(_FLAT, zoom=2.0)))
    deltas.append(Delta(seq=seq + 2, remove=[3], release=[plain.key]))
    deltas.append(
        Delta(seq=seq + 3, view=dict(_FLAT, zoom=2.0, frame_center=(0.1, 0, 0)))
    )
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


def _handed(lean: bool) -> list[Delta]:
    snapper = Canonical(planes=False, lean=lean)
    found = [snapper(delta) for delta in _stream()]
    # The stream's pair of squares is decided otherwise by design (counted
    # when the switch is off): not this test's subject.
    canonical.point_equality_disagreements.clear()
    return found


def _delta_parts(d: Delta) -> dict:
    return {
        name: getattr(d, name)
        for name in (
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
    }


@pytest.mark.parametrize(
    ("value", "lean"),
    [(None, False), ("", False), ("0", False), ("1", True)],
)
def test_the_switch_is_off_unless_set_to_1(value, lean):
    assert canonical._lean_flag(value) is lean


def test_any_other_value_of_the_switch_is_refused():
    with pytest.raises(ValueError, match="ARACHNE_SNAP_LEAN"):
        canonical._lean_flag("yes")


def test_a_canonical_takes_the_switch_read_at_import(monkeypatch):
    monkeypatch.setattr(canonical, "SNAP_LEAN", True)
    assert Canonical(planes=False)._lean is True
    monkeypatch.setattr(canonical, "SNAP_LEAN", False)
    assert Canonical(planes=False)._lean is False
    assert Canonical(planes=False, lean=True)._lean is True


def test_the_stream_sets_a_path_again_for_its_joins():
    """The stream exercises the view pass: some delta that only moves the
    view sets the sharp path again, with other joins.
    """
    handed = _handed(False)
    first = handed[0].set[0][1].fields["joins"]
    again = [
        d.set
        for d in handed[1:]
        if d.view is not None and any(s == 1 for s, _ in d.set)
    ]
    assert again
    assert any(dict(s)[1].fields["joins"] != first for s in again)


def test_lean_hands_on_the_same_deltas():
    off, on = _handed(False), _handed(True)
    assert len(off) == len(on)
    for k, (a, b) in enumerate(zip(off, on, strict=True)):
        pa, pb = _delta_parts(a), _delta_parts(b)
        for name in pa:
            assert _same(pa[name], pb[name]), (k, name)


def test_lean_decides_a_path_set_again_by_the_view_once(monkeypatch):
    """Off, a path the view pass sets again is decided twice (the pass, then
    the record set); on, once. Every other count is the same.
    """
    decided = {False: [], True: []}
    real = Canonical._joins

    def counting(self, record):
        decided[self._lean][-1] += 1
        return real(self, record)

    monkeypatch.setattr(Canonical, "_joins", counting)
    # Per delta, on the switch off: how many paths the view pass set again
    # (a view that keeps the grid; a regrid sets everything again otherwise).
    again = []
    for lean in (False, True):
        snapper = Canonical(planes=False, lean=lean)
        for delta in _stream():
            decided[lean].append(0)
            grid = snapper.grid
            found = snapper(delta)
            if lean is False:
                own = {slot for slot, _ in delta.set}
                view_pass = delta.view is not None and snapper.grid == grid
                again.append(
                    sum(1 for slot, _ in found.set if slot not in own)
                    if view_pass
                    else 0
                )
    canonical.point_equality_disagreements.clear()
    assert any(again), "the stream set nothing again by the view"
    for k, n in enumerate(again):
        assert decided[True][k] == decided[False][k] - n, k


def test_lean_measures_no_point_equality(monkeypatch):
    """Off, the face group's pair of squares is measured (a path draws them
    as one, which joins where the faces do not) and the disagreement counted;
    on, nothing is measured and nothing kept for it.
    """
    calls = {False: 0, True: 0}
    real = canonical._measure
    lean_now = [False]

    def counting(points, snapped, uses):
        calls[lean_now[0]] += 1
        return real(points, snapped, uses)

    monkeypatch.setattr(canonical, "_measure", counting)
    found = {}
    for lean in (False, True):
        lean_now[0] = lean
        canonical.point_equality_disagreements.clear()
        snapper = Canonical(planes=False, lean=lean)
        for delta in _stream():
            snapper(delta)
        found[lean] = list(canonical.point_equality_disagreements)
        if lean:
            assert snapper._measured == {}
    canonical.point_equality_disagreements.clear()
    assert calls[False] > 0
    assert found[False]
    assert calls[True] == 0
    assert found[True] == []
