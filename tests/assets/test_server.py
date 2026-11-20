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


@pytest.mark.parametrize(
    "text", ['"newer": 1e400,', '"newer": 9007199254740993,', '"newer": "\\ud800",']
)
def test_a_manifest_outside_the_json_domain_is_refused(tmp_path, text):
    directory = copied(tmp_path, "dial")
    manifest = directory / "manifest.json"
    manifest.write_text(
        manifest.read_text().replace('"version": 1,', f'"version": 1, {text}')
    )
    data, line = _refusal(tmp_path, str(directory))
    detail = _assert_refused(data, "asset.manifest_invalid", line)
    assert detail["member"] == "manifest.json"


def test_a_part_file_that_cannot_be_read_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    (directory / "pieces" / "face.svg").unlink()
    (directory / "pieces" / "face.svg").mkdir()
    data, line = _refusal(tmp_path, str(directory))
    detail = _assert_refused(data, "asset.unreadable", line)
    assert detail["file"] == "pieces/face.svg"


def test_a_dashed_stroke_is_refused(tmp_path):
    directory = copied(tmp_path, "dial")
    shutil.copy(BROKEN / "dashed.svg", directory / "pieces" / "face.svg")
    data, line = _refusal(tmp_path, str(directory))
    detail = _assert_refused(data, "asset.unreadable", line)
    assert "stroke-dasharray" in detail["reason"]


RANGE = """\
from arachne import *


class Stage(Scene):
    def construct(self):
        box = Rig("fixture:lidded-box")
        self.add(box)
        self.play(Pose(box, lid_open={value}))  # @asks
"""


def _line_of(scene: Path, mark: str) -> int:
    return next(
        number
        for number, text in enumerate(scene.read_text().splitlines(), start=1)
        if text.endswith(mark)
    )


def test_a_value_outside_its_tested_range_refuses_the_opening(tmp_path):
    scene = tmp_path / "stage.py"
    scene.write_text(RANGE.format(value=140))
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    with pytest.raises(RpcError) as answer:
        server.handle("open", {"file": str(scene), "scene": "Stage", "config": config})
    assert answer.value.code == REFUSED
    detail = _assert_refused(answer.value.data, "rig.range", _line_of(scene, "# @asks"))
    assert detail == {
        "asset": "lidded-box",
        "parameter": "lid_open",
        "value": 140,
        "min": 0,
        "max": 110,
    }
    # Nothing was opened, so there is no frame to draw.
    with pytest.raises(RpcError):
        server.handle("frame", {"t": 0, "path": str(tmp_path / "frame.png")})


UNKNOWN_ANCHOR = """\
from arachne import *


class Stage(Scene):
    def construct(self):
        box = Rig("fixture:lidded-box")
        self.add(box)
        tip = box.anchor("knbo")  # @anchor
        self.play(Pose(box, lid_open=60), run_time=0.5)
"""


def test_an_unknown_anchor_is_found_by_the_lint_and_refuses_drawing(tmp_path):
    scene = tmp_path / "stage.py"
    scene.write_text(UNKNOWN_ANCHOR)
    line = _line_of(scene, "# @anchor")
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    server.handle("open", {"file": str(scene), "scene": "Stage", "config": config})
    report = server.handle("lint", {"rules": []})["lint_result"]["report"]
    (found,) = report["violations"]
    assert found["code"] == "annotation.anchor_unknown"
    assert found["line"] == line
    assert found["detail"] == {
        "asset": "lidded-box",
        "name": "knbo",
        "nearest": ["knob"],
    }
    assert found["fix"] == CODES["annotation.anchor_unknown"].fix
    # What the lint reads still answers.
    assert server.handle("state_at", {"t": 0.25})
    with pytest.raises(RpcError) as answer:
        server.handle("frame", {"t": 0.25, "path": str(tmp_path / "frame.png")})
    assert answer.value.code == REFUSED
    detail = _assert_refused(answer.value.data, "annotation.anchor_unknown", line)
    assert detail["name"] == "knbo"
    assert not (tmp_path / "frame.png").exists()
    # A clip renders the scene again, and refuses as it draws.
    clip = tmp_path / "clip.mp4"
    with pytest.raises(RpcError) as answer:
        server.handle("clip", {"t0": 0, "t1": 0.5, "out": str(clip)})
    assert answer.value.code == REFUSED
    _assert_refused(answer.value.data, "annotation.anchor_unknown", line)
    assert not clip.exists()


EASED = """\
from arachne import *
from arachne.utils.rate_functions import ease_out_back


class Stage(Scene):
    def construct(self):
        box = Rig("fixture:lidded-box")
        self.add(box)
        self.play(Pose(box, lid_open=110, rate_func=ease_out_back))  # @asks
"""


def test_a_pose_eased_past_its_tested_range_refuses_the_opening_at_its_play(tmp_path):
    scene = tmp_path / "stage.py"
    scene.write_text(EASED)
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    with pytest.raises(RpcError) as answer:
        server.handle("open", {"file": str(scene), "scene": "Stage", "config": config})
    assert answer.value.code == REFUSED
    detail = _assert_refused(answer.value.data, "rig.range", _line_of(scene, "# @asks"))
    assert detail["parameter"] == "lid_open"
    assert detail["value"] > 110


LATE_ANCHOR = """\
from arachne import *


class Stage(Scene):
    def construct(self):
        box = Rig("fixture:lidded-box")
        self.add(box)
        self.play(Pose(box, lid_open=60), run_time=0.5)
        self.play(box.animate.shift(UP), run_time=0.5)
        tip = box.anchor("knbo")  # @anchor
        self.wait(0.5)
"""


def test_a_clip_that_ends_before_an_unknown_anchor_is_asked_is_refused(tmp_path):
    scene = tmp_path / "stage.py"
    scene.write_text(LATE_ANCHOR)
    line = _line_of(scene, "# @anchor")
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    server.handle("open", {"file": str(scene), "scene": "Stage", "config": config})
    clip = tmp_path / "clip.mp4"
    with pytest.raises(RpcError) as answer:
        server.handle("clip", {"t0": 0, "t1": 0.25, "out": str(clip)})
    assert answer.value.code == REFUSED
    _assert_refused(answer.value.data, "annotation.anchor_unknown", line)
    assert not clip.exists()
    rendered = tmp_path / "plays.mp4"
    with pytest.raises(RpcError) as answer:
        server.handle("render", {"plays": [0], "out": str(rendered)})
    _assert_refused(answer.value.data, "annotation.anchor_unknown", line)
    assert not rendered.exists()


LIVE_ONLY = """\
from arachne import *


class Stage(Scene):
    def __init__(self, **kwargs):
        super().__init__(camera_class=MappingCamera, **kwargs)

    def construct(self):
        box = Rig("fixture:lidded-box")
        self.add(box)
        tip = box.anchor("knbo")  # @anchor
        self.play(Pose(box, lid_open=60), run_time=0.5)
"""


def test_a_scene_drawn_only_as_it_runs_lints_and_answers_states_with_an_unknown_anchor(
    tmp_path,
):
    scene = tmp_path / "stage.py"
    scene.write_text(LIVE_ONLY)
    line = _line_of(scene, "# @anchor")
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    server.handle("open", {"file": str(scene), "scene": "Stage", "config": config})
    report = server.handle("lint", {})["lint_result"]["report"]
    codes = [found["code"] for found in report["violations"]]
    assert "annotation.anchor_unknown" in codes
    assert server.handle("state_at", {"t": 0.25})
    with pytest.raises(RpcError) as answer:
        server.handle("frame", {"t": 0.25, "path": str(tmp_path / "frame.png")})
    _assert_refused(answer.value.data, "annotation.anchor_unknown", line)
    assert not (tmp_path / "frame.png").exists()


def _stored_render(tmp_path: Path, text: str, monkeypatch) -> tuple[Dispatcher, list]:
    """A server holding the scene ``text`` opened from the store, a first
    server having executed and kept it, and the executions made since.
    """
    from arachne.server.store import Store
    from arachne.session import Session

    scene = tmp_path / "stage.py"
    scene.write_text(text)
    params = {
        "file": str(scene),
        "scene": "Stage",
        "config": {**SMALL, "media_dir": str(tmp_path / "media")},
    }
    first = Dispatcher(Store())
    first.handle("open", params)
    first.handle("close", {})
    server = Dispatcher(Store())
    server.handle("open", params)
    assert server.session is None
    assert server.stored is not None
    opened: list = []
    real = Session.open.__func__

    def counted(cls, *args, **kwargs):
        opened.append(args)
        return real(cls, *args, **kwargs)

    monkeypatch.setattr(Session, "open", classmethod(counted))
    return server, opened


STORED = """\
from arachne import *


class Stage(Scene):
    def construct(self):
        box = Rig("fixture:lidded-box")
        self.add(box)
        self.play(Pose(box, lid_open=60), run_time=0.5)
"""


def test_a_stored_session_renders_without_executing_again(tmp_path, monkeypatch):
    server, opened = _stored_render(tmp_path, STORED, monkeypatch)
    out = tmp_path / "stored.mp4"
    server.handle("render", {"out": str(out)})
    assert out.exists()
    # Rendered from the store: the session was not executed for it.
    assert opened == []
    assert server.session is None
    server.handle("close", {})


def test_a_stored_session_whose_code_found_an_unknown_anchor_refuses_a_render(
    tmp_path, monkeypatch
):
    server, opened = _stored_render(tmp_path, LATE_ANCHOR, monkeypatch)
    line = _line_of(tmp_path / "stage.py", "# @anchor")
    out = tmp_path / "plays.mp4"
    with pytest.raises(RpcError) as answer:
        server.handle("render", {"plays": [0], "out": str(out)})
    assert answer.value.code == REFUSED
    _assert_refused(answer.value.data, "annotation.anchor_unknown", line)
    assert not out.exists()
    assert opened == []
    assert server.session is None
    server.handle("close", {})


def test_a_stored_session_whose_code_found_an_unknown_anchor_refuses_a_frame(
    tmp_path, monkeypatch
):
    server, opened = _stored_render(tmp_path, LATE_ANCHOR, monkeypatch)
    line = _line_of(tmp_path / "stage.py", "# @anchor")
    with pytest.raises(RpcError) as answer:
        server.handle("frame", {"t": 0.25, "path": str(tmp_path / "frame.png")})
    assert answer.value.code == REFUSED
    _assert_refused(answer.value.data, "annotation.anchor_unknown", line)
    assert not (tmp_path / "frame.png").exists()
    # Refused from what the store kept: the scene was not executed for it.
    assert opened == []
    assert server.session is None
    server.handle("close", {})
