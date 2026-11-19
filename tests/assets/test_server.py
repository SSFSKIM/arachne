"""At the session server: an asset that cannot be loaded refuses the scene's
opening with the finding whole, its code, fix and the scene's line, and
nothing is opened.
"""

from __future__ import annotations

import shutil
import socket
import textwrap
from pathlib import Path

import pytest

from arachne.lint.codes import CODES
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError

from .helpers import BROKEN, copied, edited

SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4}

SCENE = """\
from arachne import *


class Stage(Scene):
    def construct(self):
        self.add(Square())
        rig = Rig({asset!r})  # @asset
        self.add(rig)
        self.wait(0.5)
"""


def _refusal(tmp_path: Path, asset: str) -> tuple[dict, int]:
    """The answer to opening a scene that asks for ``asset``, and the line
    that asks.
    """
    scene = tmp_path / "stage.py"
    scene.write_text(textwrap.dedent(SCENE.format(asset=asset)))
    line = next(
        number
        for number, text in enumerate(scene.read_text().splitlines(), start=1)
        if text.endswith("# @asset")
    )
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    with pytest.raises(RpcError) as answer:
        server.handle("open", {"file": str(scene), "scene": "Stage", "config": config})
    assert server.handle("ping", {}) == {"generation": None}
    error = answer.value
    assert error.code == REFUSED, error.data
    return error.data, line


def _assert_refused(data: dict, code: str, line: int) -> dict:
    entry = CODES[code]
    assert data["type"] == "RaisedFinding"
    assert data["reason"] == code
    finding = data["finding"]
    assert finding["code"] == code
    assert finding["class"] == entry.cls
    assert finding["severity"] == "error"
    assert finding["fix"] == entry.fix
    assert finding["line"] == line
    assert data["message"] == f"{code}: {finding['message']}. {entry.fix}"
    return finding["detail"]


def test_an_asset_without_its_manifest_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    (directory / "manifest.json").unlink()
    data, line = _refusal(tmp_path, str(directory))
    detail = _assert_refused(data, "asset.manifest_missing", line)
    assert detail["searched"] == [str(directory / "manifest.json")]


def test_an_svg_asked_for_as_an_asset_is_refused_for_its_missing_manifest(tmp_path):
    svg = copied(tmp_path, "dial") / "pieces" / "face.svg"
    data, line = _refusal(tmp_path, str(svg))
    _assert_refused(data, "asset.manifest_missing", line)


def test_an_asset_missing_a_file_is_refused(tmp_path):
    directory = copied(tmp_path, "figure")
    (directory / "pieces" / "head.svg").unlink()
    data, line = _refusal(tmp_path, str(directory))
    detail = _assert_refused(data, "asset.file_missing", line)
    assert detail == {"asset": str(directory), "file": "pieces/head.svg"}


def test_an_asset_with_an_unreadable_file_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    shutil.copy(BROKEN / "text.svg", directory / "pieces" / "face.svg")
    data, line = _refusal(tmp_path, str(directory))
    detail = _assert_refused(data, "asset.unreadable", line)
    assert detail["file"] == "pieces/face.svg"
    assert "<text>" in detail["reason"]


def test_an_invalid_manifest_is_refused(tmp_path):
    directory = edited(copied(tmp_path, "dial"), lambda d: d.update(kind="clip"))
    data, line = _refusal(tmp_path, str(directory))
    detail = _assert_refused(data, "asset.manifest_invalid", line)
    assert detail["member"] == "kind"


@pytest.fixture
def no_network(monkeypatch):
    """``socket.socket`` replaced by one that records the attempt and fails."""
    attempts: list[tuple] = []

    class Forbidden(socket.socket):
        def __init__(self, *args, **kwargs):
            attempts.append(args)
            raise AssertionError("an asset was fetched from a network")

    monkeypatch.setattr(socket, "socket", Forbidden)
    yield attempts
    assert attempts == []


@pytest.mark.parametrize(
    "file", ["https://example.com/assets/face.svg", "//example.com/face.svg"]
)
def test_a_file_named_by_an_address_is_refused_and_never_fetched(
    tmp_path, no_network, file
):
    directory = edited(
        copied(tmp_path, "dial"), lambda d: d["parts"][0].update(file=file)
    )
    data, line = _refusal(tmp_path, str(directory))
    detail = _assert_refused(data, "asset.remote", line)
    assert detail["file"] == file


def test_an_asset_named_by_an_address_is_refused_and_never_fetched(
    tmp_path, no_network
):
    data, line = _refusal(tmp_path, "https://example.com/assets/dial")
    _assert_refused(data, "asset.remote", line)
