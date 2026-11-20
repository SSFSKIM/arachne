"""A finding raised while a scene's code runs: its code, what is wrong and
the code's fix, the detail fields its code declares, and the form the server
answers it in.
"""

from __future__ import annotations

import pytest

from arachne.lint.codes import CODES
from arachne.lint.raised import RaisedFinding


def test_its_message_is_the_code_what_is_wrong_and_the_fix():
    error = RaisedFinding(
        "asset.file_missing",
        "the asset box names parts/lid.svg, which is not there.",
        detail={"asset": "box", "file": "parts/lid.svg"},
    )
    fix = CODES["asset.file_missing"].fix
    assert str(error) == (
        f"asset.file_missing: the asset box names parts/lid.svg, which is not there. {fix}"
    )
    assert error.code == "asset.file_missing"
    assert dict(error.detail) == {"asset": "box", "file": "parts/lid.svg"}
    assert error.line is None


def test_what_is_wrong_defaults_to_the_codes_meaning():
    entry = CODES["asset.remote"]
    assert str(RaisedFinding("asset.remote")) == (
        f"asset.remote: {entry.meaning}. {entry.fix}"
    )


def test_a_code_the_table_does_not_hold_is_refused():
    with pytest.raises(KeyError, match="no.such"):
        RaisedFinding("no.such")


def test_detail_its_code_does_not_declare_is_refused():
    with pytest.raises(ValueError, match="carries no colour"):
        RaisedFinding("asset.file_missing", detail={"colour": "red"})


def test_its_finding_is_whole_with_the_line_the_server_found():
    error = RaisedFinding("asset.remote", "far away", detail={"file": "x"})
    assert error.finding(14) == {
        "code": "asset.remote",
        "class": "AssetRemote",
        "severity": "error",
        "message": "far away",
        "fix": CODES["asset.remote"].fix,
        "line": 14,
        "detail": {"file": "x"},
    }
    # A line the raiser gave is the one the finding keeps.
    assert RaisedFinding("asset.remote", line=3).finding(14)["line"] == 3


def test_a_finding_found_outside_a_recording_is_raised():
    from arachne.lint.raised import found

    with pytest.raises(RaisedFinding) as raised:
        found(
            "annotation.anchor_unknown",
            "the asset box has no anchor 'lid_tip'",
            detail={"asset": "box", "name": "lid_tip", "nearest": []},
        )
    assert raised.value.code == "annotation.anchor_unknown"
    assert raised.value.detail["name"] == "lid_tip"


def test_a_scene_whose_bridge_found_nothing_draws_as_before():
    """A scene's own bridge calls (here a check of a point) are no finding
    that refuses drawing: its frames are drawn.
    """
    from pathlib import Path

    from arachne.session import Session

    from . import bridge_scenes

    session = Session.open(
        Path(bridge_scenes.__file__),
        "PointOff",
        config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 4},
    )
    try:
        assert session.record.bridge.calls
        assert session.frame(0) is not None
    finally:
        session.close()
