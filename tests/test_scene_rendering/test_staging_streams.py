"""Staging meets the render's other ways of writing: a scene the core
refuses, executed again on Cairo, writes into the render's one staging; and
the plays drawn with caching off share one stream in staging as in place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arachne.scene.scene_file_writer import SceneFileWriter

from .test_hold_gaps import (
    LOSSLESS,
    decoded,
    expanded,
    picture,
    write_plays,
    writer_for,
)
from .test_staging import TwoPlays, _render


def test_a_scene_drawn_again_on_cairo_writes_into_the_render_s_one_staging(
    tmp_path, monkeypatch
):
    """The first execution opens a segment in staging, then the core refuses
    a frame: the execution again on Cairo writes into the same staging, which
    the render publishes whole, and no staging directory is left behind.
    """
    from arachne.frame_ir import lint
    from arachne.frame_ir.extract import Unable
    from arachne.manager import Manager
    from arachne.timeline.draw import Drawing

    monkeypatch.setattr(lint, "raised", [])
    refused: list[int] = []
    sample = Drawing.sample

    def refusing_once(self, *args, **kwargs):
        if not refused:
            refused.append(1)
            raise Unable("a frame the core does not draw")
        return sample(self, *args, **kwargs)

    monkeypatch.setattr(Drawing, "sample", refusing_once)
    monkeypatch.setattr(
        Manager, "_refused_by_the_core", lambda self, error: isinstance(error, Unable)
    )
    published: list[str] = []
    publish = SceneFileWriter.publish

    def noting(self):
        published.append(type(self.staging).__name__)
        publish(self)

    monkeypatch.setattr(SceneFileWriter, "publish", noting)

    media = tmp_path / "media"
    _render(TwoPlays, media)

    assert refused == [1]
    assert "render.fallback" in [f.code for f in lint.raised]
    assert published == ["Staging"]
    assert not list(media.rglob(".staging-*"))
    assert [path.name for path in media.rglob("TwoPlays.mp4")] == ["TwoPlays.mp4"]


def test_a_stream_written_in_staging_holds_every_play(tmp_path):
    """Plays drawn with caching off share one stream, which stays open across
    them while it is written in staging, as it does written in place.
    """
    plays = [[(picture(index), 2)] for index in range(4)]
    writer = writer_for(tmp_path, **LOSSLESS["png"])
    writer.stage()
    movie = write_plays(writer, plays)
    writer.publish()

    found = decoded(Path(movie))
    shown = [
        pixels
        for time, end, pixels in zip(
            found["times"],
            [*found["times"][1:], found["end"]],
            found["pictures"],
            strict=True,
        )
        for _ in range(int(end - time))
    ]
    written = expanded(plays)
    assert len(shown) == len(written) == 8
    for index, (a, b) in enumerate(zip(shown, written, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")
