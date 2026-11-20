"""A render's files wait in staging until it ends: where staging is made,
what a render that raised keeps, and how a file is moved into place.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import numpy as np
import pytest

from arachne import (
    Circle,
    Create,
    Dot,
    FadeIn,
    Scene,
    Square,
    ValueTracker,
    VGroup,
    linear,
    tempconfig,
)
from arachne.scene import staging
from arachne.scene.scene_file_writer import SceneFileWriter
from arachne.scene.staging import Staging

SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}


class TwoPlays(Scene):
    def construct(self):
        self.play(Create(Square()))
        self.play(FadeIn(Circle()))


class Still(Scene):
    def construct(self):
        self.add(Square())


class _OwnFill(Dot):
    """A dot whose fill its own method gives: the record can't draw it."""

    def get_fill_rgbas(self):
        return np.array([[1.0, 0.0, 0.0, 1.0]])


class RetriedThenRaises(Scene):
    """A play drawn from the record; then a play an updater adds that dot to
    halfway, which the scene is executed again to draw as it runs; then an
    ordinary error.
    """

    def construct(self):
        self.play(Create(Square()))
        tracker = ValueTracker(0)
        holder = VGroup()

        def show(group):
            if tracker.get_value() >= 0.5 and not group.submobjects:
                group.add(_OwnFill(radius=0.3))

        holder.add_updater(show)
        self.add(holder)
        self.play(tracker.animate.set_value(1), rate_func=linear)
        raise RuntimeError("raised after the plays")


def _render(scene: type, media: Path, **options) -> None:
    with tempconfig({**SMALL, "media_dir": str(media), **options}):
        scene().render()


def _files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


def test_an_automatic_still_writes_nothing_into_the_video_directory(tmp_path):
    """A scene with no play falls back to a still: its staging is made where
    the still goes, so a video directory it can't write is never needed.
    """
    videos, images = tmp_path / "videos", tmp_path / "images"
    videos.mkdir()
    videos.chmod(0o555)
    try:
        _render(
            Still, tmp_path / "media", video_dir=str(videos), images_dir=str(images)
        )
    finally:
        videos.chmod(0o755)
    assert [name for name in _files(images) if name.endswith(".png")]
    assert not any(".staging-" in name for name in _files(images))
    assert list(videos.iterdir()) == []


@pytest.fixture
def assembly_breaks(monkeypatch):
    """Assembling the movie fails once its concat manifest is written."""

    def broken(self, *args, **kwargs):
        raise RuntimeError("assembly broke")

    monkeypatch.setattr(SceneFileWriter, "combine_files", broken)


def test_a_render_that_raised_keeps_only_the_segments_it_finished(
    tmp_path, assembly_breaks
):
    media = tmp_path / "media"
    with pytest.raises(RuntimeError, match="assembly broke"):
        _render(TwoPlays, media)
    found = _files(media)
    # The two segments, in the cache; no concat manifest, no movie, no staging.
    assert len(found) == 2
    assert all("/partial_movie_files/" in name for name in found)
    assert all(name.endswith(".mp4") for name in found)


def test_a_failed_move_leaves_no_staging_and_the_render_s_own_error(
    tmp_path, assembly_breaks, monkeypatch
):
    def failing(source, target):
        raise OSError(errno.EACCES, "no room", str(target))

    monkeypatch.setattr(staging, "_move", failing)
    media = tmp_path / "media"
    with pytest.raises(RuntimeError, match="assembly broke"):
        _render(TwoPlays, media)
    assert not any(".staging-" in name for name in _files(media))
    assert not list(media.rglob(".staging-*"))


def test_across_filesystems_a_file_is_renamed_into_place_from_a_sibling(
    tmp_path, monkeypatch
):
    """The destination is replaced whole: copied into a sibling, then
    renamed, never copied over it.
    """
    target = tmp_path / "out" / "movie.mp4"
    target.parent.mkdir()
    target.write_bytes(b"old")
    stage = Staging()
    staged = stage.path(target)
    staged.write_bytes(b"new")
    replace = os.replace
    renamed: list[tuple[Path, Path]] = []

    def across(source, destination):
        if Path(source).parent != Path(destination).parent:
            raise OSError(errno.EXDEV, "cross-device link")
        renamed.append((Path(source), Path(destination)))
        replace(source, destination)

    monkeypatch.setattr(staging.os, "replace", across)
    assert stage.publish() == [target]
    assert target.read_bytes() == b"new"
    ((sibling, destination),) = renamed
    assert destination == target
    assert sibling.parent == target.parent
    assert sibling != target
    assert sorted(path.name for path in target.parent.iterdir()) == ["movie.mp4"]
    assert not staged.exists()


def test_a_render_executed_again_keeps_the_segments_its_first_execution_finished(
    tmp_path,
):
    """The first execution finished the first play's segment; the second,
    executed to draw the next play as it runs, finds it in staging; the
    render then raises, and both finished segments stay in the cache.
    """
    media = tmp_path / "media"
    with pytest.raises(RuntimeError, match="raised after the plays"):
        _render(RetriedThenRaises, media)
    found = _files(media)
    assert len(found) == 2
    assert all("/partial_movie_files/" in name for name in found)
    assert all(name.endswith(".mp4") for name in found)


def _found_late(scene: Scene) -> None:
    """What a rig records when a line names an anchor no rig has."""
    from arachne.lint.raised import found

    found("annotation.anchor_unknown", "an anchor no rig has")


class FoundAfterAPlay(Scene):
    """A play drawn and finished, then a finding that refuses drawing the
    scene, then another play.
    """

    def construct(self):
        self.play(Create(Square()))
        _found_late(self)
        self.play(FadeIn(Circle()))


class RetriedThenFound(RetriedThenRaises):
    """A play finished in the first execution; the scene executed again for
    the next play, whose code records a finding that refuses drawing it.
    """

    def construct(self):
        self.play(Create(Square()))
        _found_late(self)
        tracker = ValueTracker(0)
        holder = VGroup()

        def show(group):
            if tracker.get_value() >= 0.5 and not group.submobjects:
                group.add(_OwnFill(radius=0.3))

        holder.add_updater(show)
        self.add(holder)
        self.play(tracker.animate.set_value(1), rate_func=linear)


class LiveSequenceFoundLate(Scene):
    """A scene drawn as it runs (its dot's fill is its own), written as a
    numbered sequence, whose finding comes after its plays.
    """

    def construct(self):
        self.add(_OwnFill(radius=0.3))
        self.play(Create(Square()))
        _found_late(self)


@pytest.mark.parametrize("scene", [FoundAfterAPlay, RetriedThenFound])
def test_a_render_refused_by_its_scene_s_finding_keeps_no_segment(tmp_path, scene):
    """A segment drawn before the finding stands for a scene whose drawing
    is refused: none is kept, however many executions the render took.
    """
    from arachne.lint.raised import RaisedFinding

    media = tmp_path / "media"
    with pytest.raises(RaisedFinding):
        _render(scene, media)
    assert _files(media) == []


def test_a_sequence_drawn_as_it_runs_keeps_no_frame_before_a_late_finding(tmp_path):
    from arachne.lint.raised import RaisedFinding

    media = tmp_path / "media"
    with pytest.raises(RaisedFinding):
        _render(LiveSequenceFoundLate, media, format="png-sequence")
    assert _files(media) == []
