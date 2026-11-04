"""The shape of what the serialiser writes, field by field: the report, its
violations and their objects, and the notes kept apart from it.
"""

from __future__ import annotations

import json
from typing import Any

from arachne.lint.codes import CODES, SEVERITIES

#: The severities a report's violations may have; notes are kept apart.
REPORTED = ("error", "warning")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def assert_box(box: Any) -> None:
    """A box is four numbers, or nothing."""
    assert box is None or (
        isinstance(box, list) and len(box) == 4 and all(map(_number, box))
    ), box


def assert_object(obj: dict[str, Any]) -> None:
    assert set(obj) == {
        "id",
        "source_line",
        "kind",
        "label_or_text",
        "name",
        "bbox",
        "screen_bbox",
    }
    assert isinstance(obj["id"], str)
    assert obj["source_line"] is None or _integer(obj["source_line"])
    assert isinstance(obj["kind"], str)
    assert obj["label_or_text"] is None or isinstance(obj["label_or_text"], str)
    assert obj["name"] is None or isinstance(obj["name"], str)
    assert_box(obj["bbox"])
    assert_box(obj["screen_bbox"])


def assert_violation(violation: dict[str, Any], severities: tuple[str, ...]) -> None:
    """A violation of one of ``severities``, every field it must carry
    there and of its type, its class and severity its code's.
    """
    required = {
        "code",
        "class",
        "severity",
        "held",
        "t_start",
        "t_end",
        "objects",
        "message",
        "fix",
        "line",
    }
    assert required <= set(violation) <= required | {"detail"}
    code = CODES[violation["code"]]
    assert violation["class"] == code.cls
    assert violation["severity"] in severities
    assert violation["severity"] in (code.severity, "note")
    assert isinstance(violation["held"], bool)
    for end in ("t_start", "t_end"):
        assert violation[end] is None or _number(violation[end])
    assert isinstance(violation["message"], str)
    assert isinstance(violation["fix"], str)
    assert violation["fix"]
    assert violation["line"] is None or _integer(violation["line"])
    assert isinstance(violation["objects"], list)
    for obj in violation["objects"]:
        assert_object(obj)
    if "detail" in violation:
        detail = violation["detail"]
        assert isinstance(detail, dict)
        assert set(detail) <= {*code.detail, "folded"}
        if "folded" in detail:
            # The look-alikes folded into it, counted, and said in words.
            assert _integer(detail["folded"])
            assert detail["folded"] > 0
            more = f"(and {detail['folded']} more like it)"
            assert violation["message"].endswith(more)


def assert_report(out: dict[str, Any]) -> None:
    """A report: its fields and their types, only errors and warnings in its
    violations, and ``pass`` false exactly when one is an error.
    """
    assert set(out) == {
        "pass",
        "duration_s",
        "target_resolution",
        "violations",
        "timeline",
    }
    assert isinstance(out["pass"], bool)
    assert _number(out["duration_s"])
    resolution = out["target_resolution"]
    assert set(resolution) == {"width", "height"}
    assert all(_integer(v) and v > 0 for v in resolution.values())
    assert isinstance(out["violations"], list)
    for violation in out["violations"]:
        assert_violation(violation, REPORTED)
    errors = any(v["severity"] == "error" for v in out["violations"])
    assert out["pass"] is not errors
    assert isinstance(out["timeline"], list)
    for span in out["timeline"]:
        assert set(span) == {"index", "kind", "t_start", "t_end"}
        assert _integer(span["index"])
        assert isinstance(span["kind"], str)
        assert _number(span["t_start"])
        assert _number(span["t_end"])
    json.dumps(out)


def assert_notes(found: list[dict[str, Any]]) -> None:
    """Notes: violations of severity ``note`` only."""
    assert isinstance(found, list)
    for violation in found:
        assert_violation(violation, ("note",))
    assert set(SEVERITIES) == {*REPORTED, "note"}
    json.dumps(found)
