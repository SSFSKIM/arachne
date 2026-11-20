"""A render's workers write their streams into the render's staging, where
the render joins and publishes them: each worker is forked after the streams
are planned, and writes where the render's own process reads.
"""

from __future__ import annotations

from .test_workers import at, drawn_by_workers, renders


def test_workers_write_their_streams_where_the_render_publishes_them(tmp_path):
    one, two = renders(tmp_path, "Steps", [at(1), at(2)])
    drawn_by_workers(two, 2)
    assert two["movie"]["frames"] == one["drawn"]
    assert not list((tmp_path / "media").rglob(".staging-*"))
