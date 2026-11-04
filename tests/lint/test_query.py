"""Queries: numbers about the frame on screen at an instant, in home-frame
units, measured by the lint's own rules on the lint's own parts, with
nothing drawn.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.camera.camera import Camera
from arachne.lint import lint
from arachne.lint.codes import THRESHOLD_HEIGHT
from arachne.lint.rules import contact, frame_edge, occlusion, text_small

from . import defect_scenes, projection_scenes, query_scenes
from .conftest import line_of

DEFECTS = Path(defect_scenes.__file__)
QUERIES = Path(query_scenes.__file__)
PROJECTION = Path(projection_scenes.__file__)


@pytest.fixture
def defect(session_of):
    def open_defect(name):
        return session_of(name, DEFECTS, frame_rate=15)

    return open_defect


def _end(session) -> Fraction:
    return session.record.duration


def _same(found, findings):
    """The query's records are the rule's findings, one for one."""
    assert [r.finding for r in found] == list(findings)


def _rule_findings(session, rule, t):
    return list(lint(session, rules=(rule.where(at=[t]),)).findings)


# bbox ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "scene", "t"),
    [
        (DEFECTS, "WordsOnWords", Fraction(1, 3)),
        (DEFECTS, "RingsCross", Fraction(1)),
        (QUERIES, "DotsOnALine", Fraction(1, 5)),
        (PROJECTION, "MovingDuring", Fraction(7, 15)),
    ],
)
def test_bbox_is_state_at_s_box(session_of, path, scene, t):
    session = session_of(scene, path, frame_rate=15)
    query = session.query(t)
    assert query.t == t
    state = session.state_at(t)
    assert state.nodes
    for node in state.nodes:
        assert query.bbox(node.id) == node.box


def test_bbox_follows_a_moving_camera_s_view(session_of):
    session = session_of("MovingDuring", PROJECTION, frame_rate=15)
    (circle,) = [n for n in session.state_at(0).nodes if n.type == "Circle"]
    start = session.query(0).bbox(circle.id)
    end = session.query(_end(session)).bbox(circle.id)
    # The circle stands still in the scene; the view pans and zooms past it.
    assert start != pytest.approx(end, abs=1e-3)
    width = start[2] - start[0]
    assert end[2] - end[0] > width


def test_a_query_reads_the_frame_on_screen(session_of):
    session = session_of("DotsOnALine", QUERIES, frame_rate=15)
    query = session.query(Fraction(1, 4))
    # 1/4 s lies between frames 3 and 4: frame 3 is on screen.
    assert query.t == Fraction(3, 15)
    assert query.play == 0
    assert query.answer == "recorded"
    past = session.query(10)
    assert past.t == _end(session)
    assert past.play is None


def test_bbox_of_an_unknown_id_names_it(defect):
    query = defect("WordsOnWords").query(1)
    with pytest.raises(KeyError, match="nowhere"):
        query.bbox("nowhere")


# overlaps and occlusions -------------------------------------------------------


@pytest.mark.parametrize(
    ("scene", "code"),
    [("WordsOnWords", "overlap.text"), ("RingsCross", "contact.held")],
)
def test_overlaps_on_the_defect_copies(defect, scene, code):
    session = defect(scene)
    t = _end(session)
    query = session.query(t)
    (found,) = query.overlaps()
    assert found.code == code
    # Closer than held to: two texts overlap, two crossing outlines meet.
    assert found.gap < (found.required if found.required is not None else 0)
    assert {found.a.id, found.b.id} == {o.id for o in found.finding.objects}
    assert query.occlusions() == ()
    _same(query.overlaps(), _rule_findings(session, contact, t))


@pytest.mark.parametrize("scene", ["WordsApart", "RingsApart", "SquareBelowWords"])
def test_a_clean_twin_overlaps_nothing(defect, scene):
    query = defect(scene).query(10)
    assert query.overlaps() == ()
    assert query.occlusions() == ()


def test_crowding_is_asked_only_where_the_scene_declares_a_gap(defect):
    # The scene declares no text gap: rows a hair apart don't touch.
    assert defect("RowsTight").query(10).overlaps() == ()


def test_occlusions_on_the_defect_copies(defect):
    session = defect("SquareOverWords")
    t = _end(session)
    query = session.query(t)
    (found,) = query.occlusions()
    assert found.text.text == "hidden words"
    assert found.over.type == "Square"
    assert found.depth > 0
    assert found.finding.code == "occlusion.glyph"
    assert query.overlaps() == ()
    _same(query.occlusions(), _rule_findings(session, occlusion, t))


def test_a_pair_s_things_are_boxed_in_home_units(defect):
    query = defect("WordsOnWords").query(10)
    (found,) = query.overlaps()
    for thing in (found.a, found.b):
        x0, y0, x1, y1 = thing.box
        assert x0 < x1
        assert y0 < y1
        assert thing.text in ("one phrase", "another phrase")
        assert thing.line is not None


# offscreen -------------------------------------------------------------------


def test_offscreen_names_the_edge_and_how_far_past(defect):
    session = defect("CaptionPushedOut")
    t = _end(session)
    query = session.query(t)
    (found,) = query.offscreen()
    assert found.thing.text == "A caption"
    assert found.edge == "right"
    assert found.past > 0
    assert found.thing.box[2] > query.frame[2]
    _same(query.offscreen(), _rule_findings(session, frame_edge, t))


def test_offscreen_is_empty_for_a_caption_kept_in(defect):
    assert defect("CaptionKeptIn").query(10).offscreen() == ()


# text heights ----------------------------------------------------------------


def test_text_heights_in_units_and_in_pixels_at_1080(defect):
    session = defect("TinyWord")
    t = _end(session)
    query = session.query(t)
    heights = {h.thing.text: h for h in query.text_heights()}
    assert set(heights) == {"tiny", "readable"}
    home_height = query.frame[3] - query.frame[1]
    for height in heights.values():
        assert height.units > 0
        # The picture is wider than tall: its height maps to 1080 pixels.
        assert height.pixels == pytest.approx(
            height.units * THRESHOLD_HEIGHT / home_height
        )
    assert heights["tiny"].pixels < 20 <= heights["readable"].pixels
    assert heights["readable"].units > 4 * heights["tiny"].units
    (note,) = _rule_findings(session, text_small, t)
    assert note.detail["height_px"] == pytest.approx(heights["tiny"].pixels, abs=0.05)


# markers ---------------------------------------------------------------------


def _named(markers) -> list[tuple[str, str]]:
    names = {"section": "name", "subcaption": "content", "mark": "label"}
    return [
        (
            m.kind,
            m.data[names[m.kind]] if m.kind in names else Path(m.data["file"]).name,
        )
        for m in markers
    ]


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        # The first play: its section, nothing else placed in it.
        (Fraction(1, 3), [("section", "intro")]),
        # The wait: its section, the subcaption showing, its tag and sound.
        (
            Fraction(2, 3),
            [
                ("section", "intro"),
                ("mark", "beat"),
                ("sound", "arachne-query-click.wav"),
                ("subcaption", "hello"),
            ],
        ),
        # Still the wait: the subcaption has ended (0.5 + 0.4 s).
        (
            Fraction(14, 15),
            [
                ("section", "intro"),
                ("mark", "beat"),
                ("sound", "arachne-query-click.wav"),
            ],
        ),
        # The last play: the second section only.
        (Fraction(16, 15), [("section", "second")]),
    ],
)
def test_markers_are_the_named_markers_in_force(session_of, t, expected):
    session = session_of("Narrated", QUERIES, frame_rate=15)
    record = session.record
    found = session.query(t).markers()
    assert _named(found) == expected
    # The record's own markers, with the times and lines open reports.
    for marker in found:
        assert any(marker is kept for kept in record.markers)
    by_kind = {m.kind: m for m in found}
    if "mark" in by_kind:
        assert by_kind["mark"].t == Fraction(1, 2)
        assert by_kind["mark"].source_line.line == line_of("beat", QUERIES)
        assert by_kind["mark"].data == {"label": "beat", "n": 1}


def test_the_scene_s_markers_are_all_recorded(session_of):
    """Every kind the query answers is one the record keeps."""
    session = session_of("Narrated", QUERIES, frame_rate=15)
    kinds = [(m.kind, m.t) for m in session.record.markers]
    assert kinds == [
        ("section", Fraction(0)),
        ("mark", Fraction(1, 2)),
        ("sound", Fraction(1, 2)),
        ("subcaption", Fraction(1, 2)),
        ("section", Fraction(1)),
    ]


# no drawing ------------------------------------------------------------------


def test_a_query_draws_nothing(defect, monkeypatch):
    session = defect("SquareOverWords")

    def refuse(*args, **kwargs):
        raise AssertionError("a query drew a frame")

    monkeypatch.setattr(Camera, "capture_mobjects", refuse)
    query = session.query(10)
    assert query.occlusions()
    assert query.text_heights()
    assert session.forward_runs == []


def test_a_refused_record_is_queried_through_one_forward_run(session_of):
    session = session_of("RefusedInstants", frame_rate=15)
    assert session.record.refusal is not None
    query = session.query(Fraction(1, 3))
    assert query.answer == "forward"
    assert len(session.forward_runs) == 1
    assert [part.type for part in query.parts.parts] == ["Dot"]
