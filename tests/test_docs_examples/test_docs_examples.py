"""The documentation examples as frame tests.

Each ``.. manim::`` block runs its own scene class, as the block writes it, at the
graphical units' configuration, and its frames are compared with the control data in
``control_data/`` through the graphical units' frame testers:

- a still example (``:save_last_frame:``) renders as the directive renders it, skipping
  its animations, and compares its last frame;
- an animated example plays its animations and compares 4 frames spread evenly over it
  plus its last; its frame count is pinned in ``control_data/frame_counts.json``.

Run with ``--set_test`` to write the control data instead.
"""

from __future__ import annotations

import fcntl
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import manim.scene.scene as scene_module
from manim import tempconfig
from manim.renderer.cairo_renderer import CairoRenderer
from manim.utils.testing._frames_testers import _ControlDataWriter, _FramesTester
from manim.utils.testing._test_class_makers import DummySceneFileWriter
from manim.utils.testing.frames_comparison import _config_test

from .collect import DOCS_DIR, EXCLUDED, DocsExample, collect, sample_indices

CONTROL_DIR = Path(__file__).parent / "control_data"
FRAME_COUNTS = CONTROL_DIR / "frame_counts.json"

EXAMPLES = [example for example in collect() if example.name not in EXCLUDED]


class _CapturingWriter(DummySceneFileWriter):
    """Keeps every frame the renderer writes, and the still image it saves."""

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.frames: list[tuple[np.ndarray, int]] = []
        self.still: np.ndarray | None = None

    def write_frame(self, pixels, *, repeat: int = 1) -> None:
        self.frames.append((np.array(pixels, copy=True), repeat))
        super().write_frame(pixels, repeat=repeat)

    def save_image(self, pixels) -> None:
        self.still = np.array(pixels, copy=True)

    def frame(self, index: int) -> np.ndarray:
        for pixels, repeat in self.frames:
            if index < repeat:
                return pixels
            index -= repeat
        raise IndexError(index)

    @property
    def frames_written(self) -> int:
        return sum(repeat for _, repeat in self.frames)


class _CapturingRenderer(CairoRenderer):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["file_writer_class"] = _CapturingWriter
        super().__init__(*args, **kwargs)


def render(example: DocsExample, tmp_path: Path) -> _CapturingWriter:
    """Run the block's own scene class and return what its renderer wrote."""
    settings = _config_test(last_frame=example.still)
    overrides = {
        "text_dir": tmp_path,
        "tex_dir": tmp_path,
        "assets_dir": DOCS_DIR / "_static",
        "progress_bar": "none",
        # The examples draw random colours and layouts; a fixed seed makes them repeatable.
        # The scene seeds itself with it when it is created.
        "seed": 0,
        "format": "png" if example.still else "auto",
    }
    namespace: dict = {"__name__": f"docs_example_{example.scene}"}
    # Some blocks draw at module level, before the scene exists (a random graph).
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002 (the blocks use the global generator)
    # The scene builds its own renderer, with its own camera class; only the file
    # writer is swapped for one that keeps the frames.
    with (
        tempconfig({**settings, **overrides}),
        patch.object(scene_module, "CairoRenderer", _CapturingRenderer),
    ):
        exec(
            compile("from manim import *\n" + example.code, example.origin, "exec"),
            namespace,
        )
        scene = namespace[example.scene]()
        scene.render()
    return scene.renderer.file_writer


def compared_frames(example: DocsExample, writer: _CapturingWriter) -> list[np.ndarray]:
    if example.still:
        assert writer.still is not None, "a still example saved no image"
        return [writer.still]
    return [writer.frame(index) for index in sample_indices(writer.frames_written)]


def _record_frame_count(name: str, count: int) -> None:
    # Control data is written by parallel workers: update the shared file under a lock.
    FRAME_COUNTS.touch()
    with FRAME_COUNTS.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        counts = json.loads(text) if text.strip() else {}
        counts[name] = count
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(dict(sorted(counts.items())), indent=1) + "\n")


def test_control_data_covers_exactly_the_examples(request) -> None:
    if request.config.getoption("--set_test"):
        pytest.skip("the control data is being written")
    stored = {path.stem for path in CONTROL_DIR.glob("*.npz")}
    assert stored == {example.name for example in EXAMPLES}
    counts = json.loads(FRAME_COUNTS.read_text())
    assert set(counts) == {example.name for example in EXAMPLES if not example.still}


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.name)
def test_docs_example(example: DocsExample, request, tmp_path: Path) -> None:
    writer = render(example, tmp_path)
    frames = compared_frames(example, writer)
    path = CONTROL_DIR / f"{example.name}.npz"
    if request.config.getoption("--set_test"):
        CONTROL_DIR.mkdir(exist_ok=True)
        tester = _ControlDataWriter(path, size_frame=frames[0].shape[:2])
        if not example.still:
            _record_frame_count(example.name, writer.frames_written)
    else:
        assert path.is_file(), f"no control data for {example.name} ({example.origin})"
        tester = _FramesTester(path, show_diff=request.config.getoption("--show_diff"))
        if not example.still:
            counts = json.loads(FRAME_COUNTS.read_text())
            assert writer.frames_written == counts[example.name], (
                f"{example.name} wrote {writer.frames_written} frames, "
                f"the control data has {counts[example.name]}"
            )
    with tester.testing():
        for index, frame in enumerate(frames):
            tester.check_frame(index, frame)
