"""A session of a scene holding a GPU-only capability draws every frame on
the GPU, a forward run's too, and where it can't, refuses before drawing
or writing anything; what needs no drawing answers on every host.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from arachne.compose import guard
from arachne.routing import GpuLive, GpuRequired
from arachne.server import dispatch
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import RpcError
from arachne.server.store import Store

from .conftest import SCENES, SMALL, snapshot

HALF = Fraction(1, 2)


def test_with_no_gpu_every_picture_is_refused_and_nothing_written(
    tmp_path, session_of, no_gpu
):
    session = session_of("EffectLast")
    assert session.route == "gpu"
    with pytest.raises(GpuRequired):
        session.frame(HALF)
    with pytest.raises(GpuRequired):
        session.answers([HALF])
    for draw in (session.key_frames, session.storyboard):
        out = tmp_path / draw.__name__
        with pytest.raises(GpuRequired):
            draw(out)
        assert not out.exists()
    assert session.renderer is None


def test_with_no_gpu_what_needs_no_picture_answers(session_of, no_gpu):
    session = session_of("EffectLast")
    assert session.state_at(HALF).t == HALF
    (answer,) = session.answers([HALF], frames=False)
    assert answer.frame is None
    assert answer.renderer is None
    session.query(HALF)
    codes = {finding.code for finding in session.lint().findings}
    assert not any(code.startswith("render.gpu") for code in codes)


def test_key_frames_beside_a_lint_list_each_refusal_in_place_of_its_file(
    tmp_path, session_of, no_gpu
):
    session = session_of("EffectLast")
    out = tmp_path / "key-frames"
    found = session.key_frames(out, keep_refusals=True)
    plays = len(session.record.plays)
    assert found.frames == ()
    assert [unshown.play for unshown in found.unshown] == list(range(plays))
    assert {finding.code for finding in found.findings} == {"render.gpu_required"}
    assert snapshot(out) == {}


@pytest.mark.parametrize("host", ["no_gpu", "a_gpu"])
def test_a_scene_drawn_as_it_runs_refuses_every_answer_but_its_lint(
    request, session_of, host
):
    request.getfixturevalue(host)
    session = session_of("LiveLayer")
    assert session.record.refusal is not None
    for ask in (session.frame, session.state_at, session.query):
        with pytest.raises(GpuLive) as refused:
            ask(HALF)
        assert refused.value.cause == session.record.refusal
    with pytest.raises(GpuLive):
        session.answers([HALF], frames=False)
    (finding,) = session.lint().findings
    assert (finding.code, finding.severity) == ("render.gpu_live", "error")


def test_a_forward_run_draws_on_the_route_of_the_whole_scene(session_of, no_gpu):
    """The first play holds nothing, but its instants are answered by a
    forward run of the scene, whose frames are drawn on the scene's route.
    """
    session = session_of("ReboundLayer")
    (answer,) = session.answers([HALF], frames=False)
    assert answer.state.answer == "forward"
    with pytest.raises(GpuRequired):
        session.answers([HALF])


@pytest.fixture
def never_rendered(monkeypatch):
    """A server's render, which a test expects never to be reached."""

    def render(*args, **kwargs):
        raise AssertionError("rendered")

    monkeypatch.setattr(dispatch.rendering, "render", render)


def _open(scene: str) -> Dispatcher:
    server = Dispatcher(Store())
    server.handle("open", {"file": str(SCENES), "scene": scene, "config": SMALL})
    return server


@pytest.mark.parametrize(
    ("host", "reason", "cause"),
    [
        ("no_gpu", "render.gpu_required", None),
        ("a_gpu", "render.gpu_live", guard.RECORD_FIRST),
    ],
)
def test_a_clip_of_a_play_holding_nothing_is_refused_before_it_renders(
    tmp_path, request, never_rendered, host, reason, cause
):
    request.getfixturevalue(host)
    server = _open("EffectLast")
    try:
        with pytest.raises(RpcError) as refused:
            server.handle(
                "clip", {"t0": 0, "t1": 0.5, "out": str(tmp_path / "clip.mp4")}
            )
    finally:
        server.handle("close", {})
    assert refused.value.data["reason"] == reason
    if cause is not None:
        assert cause in refused.value.data["message"]
    assert snapshot(tmp_path) == {}


def test_a_session_opened_from_the_store_keeps_its_route(
    tmp_path, no_gpu, never_rendered
):
    _open("EffectLast").handle("close", {})
    server = _open("EffectLast")
    try:
        # Opened from the store: nothing executed, and still refused.
        assert server.session is None
        assert server.stored.route == "gpu"
        picture = tmp_path / "frame.png"
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.5, "path": str(picture)})
        assert refused.value.data["reason"] == "render.gpu_required"
        with pytest.raises(RpcError) as refused:
            server.handle("render", {"out": str(tmp_path / "scene.mp4")})
        assert refused.value.data["reason"] == "render.gpu_required"
    finally:
        server.handle("close", {})
    assert snapshot(tmp_path) == {}


def test_a_cpu_scene_s_stored_frame_is_still_served_from_the_store(tmp_path):
    for _ in range(2):
        server = _open("Plain")
        try:
            from_store = server.session is None
            picture = tmp_path / "frame.png"
            answered = server.handle("frame", {"t": 0.5, "path": str(picture)})
        finally:
            server.handle("close", {})
        assert "renderer" not in answered["frame"]
    # The second server answered from the store without executing the scene.
    assert from_store
    assert server.session is None
