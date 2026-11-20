"""A render of a scene holding a GPU-only capability is refused, and leaves
every file that existed before it as it was, with nothing new beside them:
what it drew before the refusal waited in its staging, which goes with it.
"""

from __future__ import annotations

import pytest

from arachne.compose import guard
from arachne.renderer import cairo_renderer
from arachne.routing import GpuLive, GpuRequired

from .compose_scenes import AfterLastPlay, LastPlay, Live, NoPlay, Plain
from .conftest import render, snapshot


@pytest.fixture
def in_flight(monkeypatch):
    """The number of segment encoders still in flight each time the guard
    refused a render.
    """
    seen: list[int] = []
    check = cairo_renderer.check_render

    def counted(record, live=None):
        try:
            check(record, live)
        except (GpuRequired, GpuLive):
            writer = record._scene.renderer.file_writer
            seen.append(len(writer._inflight_encode_jobs))
            raise

    monkeypatch.setattr(cairo_renderer, "check_render", counted)
    return seen


def _warm(media, monkeypatch) -> None:
    """Render ``LastPlay`` with its switch off, so that its first play's
    segment and a movie are in place; then set the switch.
    """
    render(LastPlay, media, max_inflight_encoders=4)
    monkeypatch.setattr(LastPlay, "switch", True)


def test_an_effect_in_the_last_play_is_refused_with_nothing_written(
    tmp_path, monkeypatch, no_gpu, in_flight
):
    media = tmp_path / "media"
    _warm(media, monkeypatch)
    before = snapshot(media)
    assert any(name.endswith("LastPlay.mp4") for name in before)
    with pytest.raises(GpuRequired) as refused:
        render(LastPlay, media, max_inflight_encoders=4)
    assert refused.value.code == "render.gpu_required"
    assert refused.value.finding.detail["play"] == 2
    # The second play was drawn anew (its colour changed) and its encoder was
    # still in flight when the third was refused.
    assert in_flight == [1]
    assert snapshot(media) == before


def test_on_a_host_with_a_gpu_the_render_is_refused_for_now(
    tmp_path, monkeypatch, a_gpu
):
    media = tmp_path / "media"
    _warm(media, monkeypatch)
    before = snapshot(media)
    with pytest.raises(GpuLive) as refused:
        render(LastPlay, media, max_inflight_encoders=4)
    assert refused.value.cause == guard.RECORD_FIRST
    assert snapshot(media) == before


def test_a_png_sequence_leaves_the_numbered_frames_it_found(tmp_path, no_gpu):
    media = tmp_path / "media"
    render(Plain, media, format="png-sequence")
    before = snapshot(media)
    frames = [name for name in before if name.endswith(".png")]
    assert len(frames) > 10
    # The same scene name, so the same destination, now refused late.
    renamed = type("Plain", (AfterLastPlay,), {})
    with pytest.raises(GpuRequired):
        render(renamed, media, format="png-sequence")
    assert snapshot(media) == before


def test_an_effect_set_after_the_last_play_is_refused(tmp_path, no_gpu):
    media = tmp_path / "media"
    with pytest.raises(GpuRequired) as refused:
        render(AfterLastPlay, media)
    assert refused.value.finding.detail["play"] is None
    assert snapshot(media) == {}


def test_a_scene_with_no_play_is_refused_and_its_still_left(tmp_path, no_gpu):
    media = tmp_path / "media"
    render(type("NoPlay", (Plain,), {}), media, format="png")
    before = snapshot(media)
    assert any(name.endswith(".png") for name in before)
    with pytest.raises(GpuRequired):
        render(NoPlay, media, format="png")
    assert snapshot(media) == before


@pytest.mark.parametrize("host", ["no_gpu", "a_gpu"])
def test_a_scene_drawn_as_it_runs_is_refused_live_on_every_host(
    tmp_path, request, host
):
    request.getfixturevalue(host)
    media = tmp_path / "media"
    with pytest.raises(GpuLive) as refused:
        render(Live, media)
    assert refused.value.code == "render.gpu_live"
    assert refused.value.cause is not None
    assert snapshot(media) == {}


def test_a_scene_holding_nothing_renders_and_publishes_everything(tmp_path):
    media = tmp_path / "media"
    render(Plain, media)
    found = snapshot(media)
    assert any(name.endswith("Plain.mp4") for name in found)
    assert sum(name.endswith(".mp4") and "partial" in name for name in found) == 2
    assert not any(".staging-" in name for name in found)
    # Again into the cache it left: both plays from the cache, the same files.
    render(Plain, media)
    assert snapshot(media).keys() == found.keys()
