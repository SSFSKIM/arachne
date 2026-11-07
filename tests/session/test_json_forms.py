"""The session's answers as JSON: each form validates against its schema,
exact times read back, and the boxes a ``state_at`` node carries are in the
units they say, under every camera.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from arachne import api_json
from arachne.lint import lint
from arachne.lint.rules import sweep, text_small
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.execution import Divergence
from arachne.timeline.source import SourceLine
from tests.lint import defect_scenes, query_scenes, shapes
from tests.lint.report_shape import assert_notes, assert_report
from tests.timeline import probe_scenes
from tests.timeline import scenes as timeline_scenes

from . import json_scenes

SCENES = Path(json_scenes.__file__)
QUERY_SCENES = Path(query_scenes.__file__)
DEFECT_SCENES = Path(defect_scenes.__file__)
TIMELINE_SCENES = Path(timeline_scenes.__file__)
PROBE_SCENES = Path(probe_scenes.__file__)

#: A small frame, so executing and drawing cost little.
SMALL = {"pixel_width": 96, "pixel_height": 54}
#: Large enough that a pixel is a thirtieth of a scene unit.
LARGE = {"pixel_width": 480, "pixel_height": 270}


class _Sessions:
    """Each scene's session, opened once for the module."""

    def __init__(self) -> None:
        self._open: dict[tuple[Path, str, str], Session] = {}

    def __call__(self, path: Path, name: str, size: dict = SMALL) -> Session:
        key = (path, name, str(size))
        if key not in self._open:
            self._open[key] = Session.open(
                path, name, config={**size, "frame_rate": 15}
            )
        return self._open[key]

    def close(self) -> None:
        for session in self._open.values():
            session.close()


@pytest.fixture(scope="module")
def sessions() -> Iterator[_Sessions]:
    admission.DEMOTED.clear()
    found = _Sessions()
    yield found
    found.close()
    admission.DEMOTED.clear()


def _valid(form: str, out: dict) -> dict:
    """``out`` as JSON text read back, checked against ``form``'s schema."""
    read = json.loads(json.dumps(out, allow_nan=False))
    validator = Draft202012Validator(api_json.schema(form))
    errors = sorted(validator.iter_errors(read), key=lambda e: list(e.path))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
    return read


def _line(path: Path, tag: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def test_every_schema_is_a_valid_schema():
    for form in api_json.FORMS:
        Draft202012Validator.check_schema(api_json.schema(form))


def test_the_schemas_share_their_common_definitions():
    """A definition two forms both use is written the same in each."""
    seen: dict[str, tuple[str, object]] = {}
    for form in api_json.FORMS:
        for name, definition in api_json.schema(form).get("$defs", {}).items():
            if name in seen:
                assert definition == seen[name][1], (name, form, seen[name][0])
            else:
                seen[name] = (form, definition)


def test_an_unknown_form_has_no_schema():
    with pytest.raises(ValueError, match="record"):
        api_json.schema("nothing")


#: Scenes of the lint's and the timeline's tests, each form read on each:
#: markers of every kind, a moving camera, state plays the probe reads, a
#: replaced projection, findings and a note.
FORM_SCENES = [
    (TIMELINE_SCENES, "Timeline"),
    (TIMELINE_SCENES, "Panned"),
    (PROBE_SCENES, "ShortPlays"),
    (QUERY_SCENES, "Narrated"),
    (QUERY_SCENES, "DotsOnALine"),
    (QUERY_SCENES, "ReplacesItsProjection"),
    (DEFECT_SCENES, "WordsOnWords"),
    (DEFECT_SCENES, "TinyWord"),
]


@pytest.mark.parametrize(
    ("path", "name"), FORM_SCENES, ids=[name for _, name in FORM_SCENES]
)
def test_every_form_validates_against_its_schema(sessions, tmp_path, path, name):
    session = sessions(path, name)
    record = session.record
    _valid("record", api_json.inspect(session))
    times = [Fraction(0), record.duration / 2, record.duration]
    for t in times:
        _valid("state", api_json.state(session.state_at(t), session))
        # A query reads the frame on screen at t: box what that frame holds.
        found = session.query(t)
        out = _valid("state", api_json.state(session.state_at(found.t), session))
        ids = [node["id"] for node in out["nodes"]]
        _valid("query", api_json.query(found, api_json.QUERY_ANSWERS, ids=ids))
    _valid("key_frames", api_json.key_frames(session.key_frames(tmp_path)))
    _valid("lint_result", api_json.lint(lint(session, rules=(sweep, text_small))))


def test_the_record_s_form_holds_its_timeline_identity_and_provenance(sessions):
    session = sessions(TIMELINE_SCENES, "Timeline")
    record = session.record
    out = _valid("record", api_json.inspect(session))
    assert out["schema_version"] == api_json.SCHEMA_VERSION
    assert (out["scene_class"], out["stale"]) == ("Timeline", False)
    assert out["target_resolution"] == {"width": 96, "height": 54}
    assert out["identity"]["source_digest"] == record.identity.source_digest
    assert [p["index"] for p in out["plays"]] == [p.index for p in record.plays]
    assert [p["route"] for p in out["plays"]] == [p.route for p in record.plays]
    create = out["plays"][0]
    assert create["source_line"] == _line(TIMELINE_SCENES, "create")
    assert create["source"] == f"scenes.py:{create['source_line']}"
    assert create["animations"] == ["Create(Square)"]
    (beat,) = [m for m in out["markers"] if m["kind"] == "mark"]
    assert beat["data"] == {"label": "beat", "n": 1}
    assert beat["source_line"] == _line(TIMELINE_SCENES, "mark")
    # Placed as the wait after the play began, it is among the wait's.
    assert beat in out["plays"][1]["markers"]
    square = out["mobjects"][f"scenes.py:{_line(TIMELINE_SCENES, 'square')}#0"]
    assert square == {
        "type": "Square",
        "name": "square",
        "source_line": _line(TIMELINE_SCENES, "square"),
        "source": f"scenes.py:{_line(TIMELINE_SCENES, 'square')}",
    }
    # Samples and values are not part of it.
    assert not {"samples", "values", "nodes"} & set(out)
    assert not {"samples", "cache_key"} & set(create)


def test_the_probe_s_outcomes_are_in_the_record_s_form(sessions):
    session = sessions(PROBE_SCENES, "ShortPlays")
    out = _valid("record", api_json.inspect(session))
    assert out["probe"]["settled"] is session.probe.settled
    assert {p["play"]: p["outcome"] for p in out["probe"]["plays"]} == {
        0: "unprobed",
        1: "unprobed",
        3: "agreed",
    }


def test_a_marker_s_numpy_values_are_json_values(sessions):
    """A value with no Python number to give is written by its ``repr``;
    one that has one is that number.
    """
    session = sessions(SCENES, "WideMarks")
    out = _valid("record", api_json.inspect(session))
    (marker,) = [m for m in out["markers"] if m["kind"] == "mark"]
    assert marker["data"] == {
        "label": "wide",
        "real": repr(np.longdouble(1.5)),
        "complex": repr(np.clongdouble(1 + 2j)),
        "reals": [repr(np.longdouble(1.5)), repr(np.longdouble(2.5))],
        "complexes": [repr(np.clongdouble(1 + 2j))],
        "plain": 0.25,
    }
    found = session.query(Fraction(1, 4))
    queried = _valid("query", api_json.query(found, "markers"))
    assert queried["markers"] == [marker]


def test_a_divergence_is_in_the_record_s_form(sessions):
    session = sessions(QUERY_SCENES, "DotsOnALine")
    planted = Divergence(
        "execution", 0, SourceLine("query_scenes.py", 46), "planted", ("data.csv",)
    )
    session.divergences.append(planted)
    try:
        out = _valid("record", api_json.inspect(session))
    finally:
        session.divergences.remove(planted)
    assert out["divergences"] == [
        {
            "code": "replay.diverged",
            "kind": "execution",
            "play": 0,
            "source_line": 46,
            "source": "query_scenes.py:46",
            "reason": "planted",
            "changed_inputs": ["data.csv"],
        }
    ]


def test_exact_times_round_trip(sessions, tmp_path):
    """Every instant reads back as the exact one, where its float can't:
    the last play of :class:`Timeline` runs a seventh of a second.
    """
    session = sessions(TIMELINE_SCENES, "Timeline")
    record = session.record
    out = _valid("record", api_json.inspect(session))
    assert Fraction(out["duration_exact"]) == record.duration
    assert record.duration.denominator == 7
    for play, kept in zip(out["plays"], record.plays, strict=True):
        assert Fraction(play["start_exact"]) == kept.start
        assert Fraction(play["end_exact"]) == kept.end
        assert Fraction(play["requested_end_exact"]) == kept.requested_end
        assert play["end"] == float(kept.end)
    for marker, kept in zip(out["markers"], record.markers, strict=True):
        assert Fraction(marker["t_exact"]) == kept.t
    t = record.plays[-1].start + Fraction(1, 21)
    state = _valid("state", api_json.state(session.state_at(t), session))
    assert Fraction(state["t_exact"]) == t
    assert Fraction(state["as_of_exact"]) == session.state_at(t).as_of
    frames = _valid("key_frames", api_json.key_frames(session.key_frames(tmp_path)))
    for frame, kept in zip(
        frames["frames"], session.key_frames(tmp_path).frames, strict=True
    ):
        assert Fraction(frame["t_exact"]) == kept.t


def _node(out: dict, name: str) -> dict:
    (node,) = [n for n in out["nodes"] if n["name"] == name]
    return node


def test_a_zoom_scales_the_screen_box_and_leaves_the_scene_box(sessions):
    session = sessions(SCENES, "Zoomed", LARGE)
    before = _node(api_json.state(session.state_at(Fraction(1, 4)), session), "square")
    after = _node(api_json.state(session.state_at(Fraction(5, 4)), session), "square")
    assert before["bbox"] == pytest.approx([0.5, 0.0, 1.5, 1.0])
    assert after["bbox"] == pytest.approx(before["bbox"])
    # The view halved about the picture's centre: every pixel twice as far
    # from it.
    centre = np.array([240, 135, 240, 135])
    doubled = (np.array(before["screen_bbox"]) - centre) * 2 + centre
    assert after["screen_bbox"] == pytest.approx(doubled.tolist(), abs=1e-6)


def _drawn_box(image) -> tuple[int, int, int, int]:
    """The box of the picture's pixels that are not the black background,
    as pixel edges.
    """
    pixels = np.asarray(image)
    rows, columns = np.nonzero(pixels[..., :3].max(axis=-1) > 0)
    assert rows.size, "the frame shows nothing"
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def test_under_a_three_d_camera_the_screen_box_is_the_projected_one(sessions):
    session = sessions(SCENES, "Tilted", LARGE)
    t = Fraction(1, 4)
    node = _node(api_json.state(session.state_at(t), session), "square")
    assert node["bbox"] == pytest.approx([0.0, -0.5, 2.0, 1.5])
    # Where the frame draws the square, to a pixel and its thin stroke.
    drawn = _drawn_box(session.frame(t))
    assert node["screen_bbox"] == pytest.approx(list(drawn), abs=1.5)
    # Not where the scene box would land under a flat camera.
    width, height = LARGE["pixel_width"], LARGE["pixel_height"]
    scale = width / session.scene.renderer.camera.frame_width
    flat = [
        width / 2 + node["bbox"][0] * scale,
        height / 2 - node["bbox"][3] * scale,
        width / 2 + node["bbox"][2] * scale,
        height / 2 - node["bbox"][1] * scale,
    ]
    assert max(abs(a - b) for a, b in zip(node["screen_bbox"], flat, strict=True)) > 10


def test_a_frame_at_another_quality_leaves_the_screen_box(sessions):
    """The screen box is in the session's pixels, whatever size the camera
    last drew at.
    """
    session = sessions(SCENES, "Tilted", LARGE)
    t = Fraction(1, 4)
    before = _node(api_json.state(session.state_at(t), session), "square")
    session.frame(t, "low_quality")
    try:
        after = _node(api_json.state(session.state_at(t), session), "square")
    finally:
        session.frame(t)
    assert after["screen_bbox"] == pytest.approx(before["screen_bbox"])


def test_the_state_s_resolution_is_the_lint_s(sessions):
    session = sessions(QUERY_SCENES, "DotsOnALine")
    out = api_json.state(session.state_at(0), session)
    report = api_json.lint(lint(session, rules=()))["report"]
    assert out["target_resolution"] == report["target_resolution"]


def test_a_mobject_made_in_a_helper_names_the_scene_file_s_line(sessions):
    session = sessions(SCENES, "MadeInAHelper")
    node = _node(api_json.state(session.state_at(Fraction(1, 4)), session), "square")
    assert node["source_line"] == _line(SCENES, "helper-call")
    helper = Path(shapes.__file__).read_text().splitlines()
    made = next(
        number
        for number, line in enumerate(helper, start=1)
        if "Square(" in line and "return" in line
    )
    assert node["source"].endswith(f"shapes.py:{made}")
    entry = api_json.inspect(session)["mobjects"][node["id"]]
    assert (entry["source_line"], entry["source"]) == (
        node["source_line"],
        node["source"],
    )


def test_a_node_carries_its_kind_label_colour_and_children(sessions):
    session = sessions(QUERY_SCENES, "DotsOnALine")
    out = _valid("state", api_json.state(session.state_at(0), session))
    word = _node(out, "word")
    assert (word["kind"], word["label_or_text"]) == ("Text", "word")
    assert word["children"]
    first = _node(out, "first")
    assert (first["kind"], first["label_or_text"]) == ("Dot", "first")
    assert first["source_line"] == _line(QUERY_SCENES, "first")
    assert first["color"].startswith("#")


def test_a_query_answers_what_it_is_asked(sessions):
    session = sessions(QUERY_SCENES, "Narrated")
    t = Fraction(3, 4)
    found = session.query(t)
    out = _valid("query", api_json.query(found, ["markers", "text_heights"]))
    assert set(out) == {
        "schema_version",
        "t",
        "t_exact",
        "play",
        "answer",
        "frame",
        "markers",
        "text_heights",
    }
    assert Fraction(out["t_exact"]) == found.t
    kinds = [m["kind"] for m in out["markers"]]
    assert kinds == [m.kind for m in found.markers()]
    with pytest.raises(ValueError, match="text_heights"):
        api_json.query(found, ["sizes"])
    with pytest.raises(ValueError, match="ids"):
        api_json.query(found, ["bbox"])


def test_a_query_s_boxes_and_contacts(sessions):
    session = sessions(DEFECT_SCENES, "WordsOnWords")
    t = Fraction(5, 4)
    found = session.query(t)
    ids = [node.id for node in session.state_at(t).nodes]
    out = _valid("query", api_json.query(found, ["bbox", "overlaps"], ids=ids))
    assert out["bbox"] == {
        node_id: (None if found.bbox(node_id) is None else list(found.bbox(node_id)))
        for node_id in ids
    }
    assert out["overlaps"], "the two phrases overlap"
    contact = out["overlaps"][0]
    assert contact["code"] == contact["finding"]["code"]
    assert contact["gap"] < 0


def test_a_replaced_camera_s_measures_are_its_replacement(sessions):
    session = sessions(QUERY_SCENES, "ReplacesItsProjection")
    found = session.query(Fraction(1, 4))
    node_id = session.state_at(Fraction(1, 4)).nodes[0].id
    out = _valid("query", api_json.query(found, api_json.QUERY_ANSWERS, ids=[node_id]))
    for answer in ("bbox", "overlaps", "occlusions", "offscreen", "text_heights"):
        replaced = out[answer]
        assert replaced["code"] == "query.camera_replaced"
        assert replaced["replaced"] == list(found.replaced.replaced)
        assert replaced["message"] == found.replaced.message
    assert isinstance(out["markers"], list)


def test_key_frames_list_each_frame_and_each_play_with_none(sessions, tmp_path):
    session = sessions(TIMELINE_SCENES, "Timeline")
    kept = session.key_frames(tmp_path)
    out = _valid("key_frames", api_json.key_frames(kept))
    assert [f["play"] for f in out["frames"]] == [f.play for f in kept.frames]
    assert [f["path"] for f in out["frames"]] == [str(f.path) for f in kept.frames]
    assert [u["reason"] for u in out["unshown"]] == [u.reason for u in kept.unshown]


def test_the_lint_form_s_report_and_notes_keep_their_shape(sessions):
    session = sessions(DEFECT_SCENES, "WordsOnWords")
    result = lint(session, rules=(sweep, text_small))
    out = _valid("lint_result", api_json.lint(result))
    assert set(out) == {"schema_version", "report", "notes"}
    assert_report(out["report"])
    assert_notes(out["notes"])
    assert out["report"]["violations"], "the phrases overlap"
    assert out["report"]["pass"] is False
    tiny = lint(sessions(DEFECT_SCENES, "TinyWord"), rules=(sweep, text_small))
    out = _valid("lint_result", api_json.lint(tiny))
    assert_notes(out["notes"])
    assert [n["code"] for n in out["notes"]] == ["text.small"]
