"""At the session server: stepped timing on exact content, and motion the
engine cannot hold to a declared timing, refuse the scene's opening with
the finding whole, its code, fix and the scene's line, and nothing opens.
"""

from __future__ import annotations

import re

import pytest

from arachne.lint.codes import CODES
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError

from .reading import SCENES, SMALL


def _line(scene: str) -> int:
    """The line marked ``# @refused`` in the scene's class, or in the class
    it declares through.
    """
    lines = SCENES.read_text().splitlines()
    start = next(
        i for i, text in enumerate(lines) if re.match(rf"class {scene}\(", text)
    )
    declared = "(_Declared)" in lines[start]
    if declared:
        start = next(
            i for i, text in enumerate(lines) if text.startswith("class _Declared(")
        )
    return next(
        i + 1 for i in range(start, len(lines)) if lines[i].endswith("# @refused")
    )


def _refused(tmp_path, scene: str, code: str) -> dict:
    server = Dispatcher()
    config = {**SMALL, "frame_rate": 15, "media_dir": str(tmp_path / "media")}
    with pytest.raises(RpcError) as answer:
        server.handle("open", {"file": str(SCENES), "scene": scene, "config": config})
    assert server.handle("ping", {}) == {"generation": None}
    error = answer.value
    assert error.code == REFUSED, error.data
    data = error.data
    entry = CODES[code]
    assert data["type"] == "RaisedFinding"
    assert data["reason"] == code, data["message"]
    finding = data["finding"]
    assert finding["code"] == code
    assert finding["class"] == entry.cls
    assert finding["severity"] == "error"
    assert finding["fix"] == entry.fix
    assert finding["line"] == _line(scene)
    assert data["message"] == f"{code}: {finding['message']}. {entry.fix}"
    return finding["detail"]


@pytest.mark.parametrize(
    ("scene", "kind", "name"),
    [
        ("UpdaterBeforeDeclaring", "mobject updater", "<lambda>"),
        ("UpdaterAfterDeclaring", "mobject updater", "<lambda>"),
        ("SceneUpdaterAfterDeclaring", "scene updater", "<lambda>"),
        ("UpdateFromFuncOnPart", "animation", "UpdateFromFunc"),
        ("OwnAnimationOnBox", "animation", "Nudge"),
    ],
)
def test_motion_the_engine_cannot_hold_is_refused(tmp_path, scene, kind, name):
    detail = _refused(tmp_path, scene, "timing.updater")
    assert detail["kind"] == kind
    assert detail["name"].endswith(name)


@pytest.mark.parametrize(
    ("scene", "found"),
    [
        ("TimedText", "Text"),
        ("TimedNumber", "DecimalNumber"),
        ("TimedAxes", "Axes"),
        ("TimedCallout", "Callout"),
        ("TimedMark", "Capsule"),
        ("TimedDiagram", "Diagram"),
        ("TimedBoxWithText", "Text"),
        ("TextJoinsDeclaredGroup", "Text"),
    ],
)
def test_stepped_timing_on_exact_content_is_refused(tmp_path, scene, found):
    detail = _refused(tmp_path, scene, "timing.exact_content")
    assert detail == {"object": found, "camera": False}


def test_stepped_timing_on_the_scene_s_camera_frame_is_refused(tmp_path):
    detail = _refused(tmp_path, "TimedCameraFrame", "timing.exact_content")
    assert detail["camera"] is True
