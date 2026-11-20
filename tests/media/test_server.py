"""At the session server: a media asset whose frames the engine cannot
decode exactly is refused by name, with its fix and the scene's line; a
frame whose pixels are not its digest's refuses the frame that needs it,
with the line that made the object; and on a host with no GPU a clip's
scene answers its states and its lint, and refuses its frames.
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arachne.lint.codes import CODES
from arachne.mobject.media import frames
from arachne.mobject.media.frames import FrameCache
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError

from .conftest import SMALL, copied, png

SCENE = """\
from arachne import *


class Stage(Scene):
    def construct(self):
        self.add(Square())
        media = {cls}({asset!r})  # @asset
        self.add(media)
        self.play(media.animate.shift(RIGHT), run_time=0.5)
        self.wait(0.5)
"""


def _scene(tmp_path: Path, cls: str, asset: str) -> tuple[Path, int]:
    """A scene file showing ``asset`` as a ``cls``, and the line that makes
    the object.
    """
    scene = tmp_path / "stage.py"
    scene.write_text(textwrap.dedent(SCENE.format(cls=cls, asset=asset)))
    line = next(
        number
        for number, text in enumerate(scene.read_text().splitlines(), start=1)
        if text.endswith("# @asset")
    )
    return scene, line


def _open(server: Dispatcher, scene: Path, tmp_path: Path) -> dict:
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    return server.handle(
        "open", {"file": str(scene), "scene": "Stage", "config": config}
    )


def _assert_refused(error: RpcError, code: str, line: int) -> dict:
    entry = CODES[code]
    assert error.code == REFUSED, error.data
    data = error.data
    assert data["type"] == "RaisedFinding"
    assert data["reason"] == code
    finding = data["finding"]
    assert finding["code"] == code
    assert finding["severity"] == "error"
    assert finding["fix"] == entry.fix
    assert finding["line"] == line
    return finding["detail"]


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(out, format="JPEG")
    return out.getvalue()


@pytest.mark.parametrize(
    ("data", "found"),
    [
        (png([bytes(64 * 8)] * 64, 64, 64, 16, 6), "16 bits per channel"),
        (
            png([bytes(64)] * 64, 64, 64, 8, 3, chunks=((b"PLTE", bytes(3)),)),
            "indexed colour",
        ),
        (_jpeg(), "not a PNG file"),
    ],
    ids=["sixteen-bit", "palette", "jpeg-named-png"],
)
def test_a_frame_in_no_admitted_form_refuses_the_scene(tmp_path, data, found):
    directory = copied(tmp_path, "swatch")
    (directory / "frames" / "0000.png").write_bytes(data)
    scene, line = _scene(tmp_path, "Still", str(directory))
    server = Dispatcher()
    with pytest.raises(RpcError) as answer:
        _open(server, scene, tmp_path)
    assert server.handle("ping", {}) == {"generation": None}
    detail = _assert_refused(answer.value, "media.inexact", line)
    assert detail["reason"] == "format"
    assert detail["file"] == "frames/0000.png"
    assert found in detail["found"]


@pytest.fixture
def fresh_cache(monkeypatch) -> FrameCache:
    kept = FrameCache()
    monkeypatch.setattr(frames, "cache", kept)
    return kept


def _changed(monkeypatch) -> None:
    """Decode every frame with one pixel other than its file holds."""
    decoded = frames.decode

    def changed(data: bytes) -> np.ndarray:
        pixels = decoded(data)
        pixels[1, 1, 2] ^= 4
        return pixels

    monkeypatch.setattr(frames, "decode", changed)


def test_a_frame_that_decodes_otherwise_draws_nothing_and_names_its_line(
    tmp_path, monkeypatch, fresh_cache
):
    _changed(monkeypatch)
    scene, line = _scene(tmp_path, "Still", "fixture:swatch")
    picture = tmp_path / "frame.png"
    server = Dispatcher()
    try:
        _open(server, scene, tmp_path)
        with pytest.raises(RpcError) as answer:
            server.handle("frame", {"t": 0.25, "path": str(picture)})
    finally:
        server.handle("close", {})
    detail = _assert_refused(answer.value, "media.inexact", line)
    assert detail["reason"] == "digest"
    assert detail["file"] == "frames/0000.png"
    assert detail["expected"] != detail["found"]
    assert not picture.exists()


def test_a_frame_decoded_again_once_the_cache_let_it_go_is_checked_again(
    tmp_path, monkeypatch, fresh_cache
):
    scene, line = _scene(tmp_path, "Still", "fixture:swatch")
    server = Dispatcher()
    try:
        _open(server, scene, tmp_path)
        assert server.handle("frame", {"t": 0.25, "path": str(tmp_path / "a.png")})
        fresh_cache.clear()
        _changed(monkeypatch)
        with pytest.raises(RpcError) as answer:
            server.handle("frame", {"t": 0.75, "path": str(tmp_path / "b.png")})
    finally:
        server.handle("close", {})
    assert _assert_refused(answer.value, "media.inexact", line)["reason"] == "digest"
    assert not (tmp_path / "b.png").exists()


def test_with_no_gpu_a_clip_scene_answers_its_states_and_lint_and_refuses_frames(
    tmp_path, no_gpu
):
    scene, _ = _scene(tmp_path, "VideoMobject", "fixture:numbered-clip")
    server = Dispatcher()
    try:
        _open(server, scene, tmp_path)
        state = server.handle("state_at", {"t": 0.25})["state"]
        assert "VideoMobject" in {node["kind"] for node in state["nodes"]}
        assert "lint_result" in server.handle("lint", {})
        with pytest.raises(RpcError) as answer:
            server.handle("frame", {"t": 0.25, "path": str(tmp_path / "f.png")})
    finally:
        server.handle("close", {})
    assert answer.value.code == REFUSED
    assert answer.value.data["reason"] == "render.gpu_required"
    assert not (tmp_path / "f.png").exists()
