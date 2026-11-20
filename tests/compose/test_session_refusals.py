"""A session of a scene holding a GPU-only capability draws every frame on
the GPU, a forward run's too, and where it can't, refuses before drawing
or writing anything; what needs no drawing answers on every host.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from arachne.compose import guard
from arachne.compose.stack import Unsupported as StackUnsupported
from arachne.lint.keyframes import _refusals_apart
from arachne.routing import GpuLive, GpuRequired
from arachne.server import dispatch
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import RpcError
from arachne.server.store import Store
from arachne.session import Session

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


@pytest.fixture
def never_executed(monkeypatch):
    """From now on, a scene executed is a test failure."""

    def start() -> None:
        def executed(*args, **kwargs):
            raise AssertionError("the scene was executed")

        monkeypatch.setattr(Session, "open", executed)

    return start


@pytest.mark.parametrize(
    ("scene", "host", "reason"),
    [
        ("EffectLast", "no_gpu", "render.gpu_required"),
        ("EffectLast", "a_gpu", "render.gpu_live"),
        ("LiveLayer", "a_gpu", "render.gpu_live"),
    ],
)
def test_a_session_opened_from_the_store_is_refused_without_executing(
    tmp_path, request, never_rendered, never_executed, scene, host, reason
):
    request.getfixturevalue(host)
    _open(scene).handle("close", {})
    never_executed()
    server = _open(scene)
    try:
        # Opened from the store, nothing executed, and refused from it.
        assert server.session is None
        assert server.stored.route == "gpu"
        asked = [("clip", {"t0": 0, "t1": 0.5}), ("render", {})]
        if host == "no_gpu" or scene == "LiveLayer":
            # A frame too, which this host could not draw either.
            asked.append(("frame", {"t": 0.5, "path": str(tmp_path / "frame.png")}))
        for method, params in asked:
            if method != "frame":
                params = {**params, "out": str(tmp_path / f"{method}.mp4")}
            with pytest.raises(RpcError) as refused:
                server.handle(method, params)
            assert refused.value.data["reason"] == reason, method
    finally:
        server.handle("close", {})
    assert snapshot(tmp_path) == {}


def test_the_refusals_kept_in_the_store_are_the_session_s_own(no_gpu):
    """The manifest keeps where each capability first appears, and the
    refusal made from it is the one the session itself makes.
    """
    from arachne.compose.session import Route, route_of

    session = Session.open(SCENES, "LiveLayer", config=SMALL)
    try:
        route = route_of(session)
        kept = Route.from_form(route.name, route.form())
        assert kept == route
        with pytest.raises(GpuLive) as own:
            session.frame(HALF)
    finally:
        session.close()
    with pytest.raises(GpuLive) as stored:
        kept.refuse_here()
    assert str(stored.value) == str(own.value)


def test_a_closed_session_holds_neither_its_record_nor_its_route(session_of):
    session = session_of("EffectLast")
    assert session.route == "gpu"
    session.close()
    assert "record" not in vars(session)
    assert session._route is None


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


def test_a_forward_batch_keeps_a_stack_refusal_apart_and_draws_the_rest(session_of):
    """One forward run answers a batch of key frames: a frame the stack
    refuses is kept as its play's refusal, and the next is still drawn.
    """
    session = session_of("Plain")
    drawn_one = np.full((54, 96, 4), 7, dtype=np.uint8)

    def drawn(record, answers, static):
        (answer,) = answers
        if answer.play.index == 0:
            raise StackUnsupported("layer", "a layer within a layer")
        return [drawn_one]

    session._drawn = drawn
    batch = [SimpleNamespace(play=SimpleNamespace(index=i)) for i in (0, 1)]
    with _refusals_apart(session) as refused:
        found = session._drawn(session.record, batch, None)
    assert list(refused) == [0]
    assert refused[0].code == "render.unsupported"
    assert not found[0].any()
    assert (found[1] == drawn_one).all()
