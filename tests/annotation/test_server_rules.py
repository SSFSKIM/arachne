"""At the session server, a lint names the annotation and diagram rules as
it names the temporal ones: ``rules`` takes ``"anchored"`` and
``"diagram_clear"`` beside ``"hold_watch"`` and ``"departures"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import RpcError

from . import annotation_scenes

SCENES = Path(annotation_scenes.__file__)


def _linted(tmp_path: Path, scene: str, rules: list[str]) -> list[str]:
    server = Dispatcher()
    config = {
        "pixel_width": 96,
        "pixel_height": 54,
        "frame_rate": 15,
        "media_dir": str(tmp_path / "media"),
    }
    server.handle("open", {"file": str(SCENES), "scene": scene, "config": config})
    found = server.handle("lint", {"rules": rules})
    return [f["code"] for f in found["lint_result"]["report"]["violations"]]


def test_a_lint_names_anchored(tmp_path):
    assert _linted(tmp_path, "CalloutBeforeMovingDot", ["anchored"]) == [
        "annotation.detached"
    ]
    assert _linted(tmp_path, "CalloutAfterMovingDot", ["anchored"]) == []


def test_a_lint_names_diagram_clear_beside_the_temporal_rules(tmp_path):
    rules = ["hold_watch", "departures", "anchored", "diagram_clear"]
    assert "annotation.detached" in _linted(tmp_path, "CalloutBeforeMovingDot", rules)


def test_a_rule_the_server_does_not_name_is_refused(tmp_path):
    with pytest.raises(RpcError):
        _linted(tmp_path, "CalloutAfterMovingDot", ["sweep"])
