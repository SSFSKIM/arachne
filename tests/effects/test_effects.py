"""Effects as a scene sets them, read with no device: what each setter
holds and declares, the quantities ``state_at`` answers at the session
server, and a frame refused on a host with no GPU.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from arachne import (
    ChangeBlur,
    Circle,
    RackFocus,
    Square,
    clear_effects,
    routing,
    set_blur,
    set_depth,
    set_focus,
    set_mask,
)
from arachne.compose.layer import layer_of
from arachne.quantities import SOURCES
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, SCENE_ERROR, RpcError

from .conftest import SCENES, SMALL


def test_two_effects_set_one_after_the_other_are_both_held():
    square, mask = Square(), Circle()
    set_blur(square, 0.2)
    set_mask(square, mask, feather=0.1, invert=True)
    set_depth(square, 0)
    spec = layer_of(square)
    assert spec is not None
    assert (spec.blur, spec.mask_feather, spec.mask_invert, spec.depth) == (
        0.2,
        0.1,
        True,
        0.0,
    )
    assert spec.mask is mask
    assert routing.held(square) == {"effect.blur", "effect.soft_mask", "layer.depth"}
    # A blur of 0 removes the blur alone.
    set_blur(square, 0)
    spec = layer_of(square)
    assert spec is not None
    assert (spec.blur, spec.mask, spec.depth) == (None, mask, 0.0)
    assert routing.held(square) == {"effect.soft_mask", "layer.depth"}
    # Its numbers are read through one source, attached once.
    sources = square.__dict__[SOURCES]
    assert [type(each).__name__ for each in sources] == ["EffectNumbers"]


def test_clearing_withdraws_what_the_effects_declared_and_keeps_the_object_s_own():
    square = Square()
    routing.declare(square, "effect.depth_of_field")
    set_blur(square, 0.2)
    set_mask(square, Circle())
    clear_effects(square)
    assert layer_of(square) is None
    assert routing.held(square) == {"effect.depth_of_field"}


def test_an_object_left_with_no_effect_is_no_layer_of_its_own():
    square = Square()
    set_depth(square, 1)
    set_mask(square, Circle(), feather=0.3)
    set_depth(square, None)
    set_mask(square, None)
    assert layer_of(square) is None
    assert routing.held(square) == frozenset()


def test_an_effect_that_cannot_be_drawn_is_refused_where_it_is_set():
    with pytest.raises(ValueError, match="positive length"):
        set_blur(Square(), -0.1)
    with pytest.raises(TypeError, match="mask is a mobject"):
        set_mask(Square(), "circle")
    with pytest.raises(ValueError, match="need a mask"):
        set_mask(Square(), None, feather=0.2)
    with pytest.raises(ValueError, match="depth is a number"):
        set_depth(Square(), float("nan"))
    with pytest.raises(ValueError, match="0 or more"):
        ChangeBlur(Square(), -1)
    with pytest.raises(TypeError, match="takes the scene"):
        set_focus(Square(), 0, 1)
    with pytest.raises(TypeError, match="takes the scene"):
        RackFocus(Square(), 1)


class _Server:
    """A session server with one scene of the scenes file open."""

    def __init__(self, tmp_path, scene: str) -> None:
        self.server = Dispatcher()
        config = {**SMALL, "media_dir": str(tmp_path / "media")}
        self.server.handle(
            "open", {"file": str(SCENES), "scene": scene, "config": config}
        )

    def numbers(self, t: float, name: str) -> dict[str, float]:
        """The quantities of the object the scene named ``name``, at ``t``,
        each readable, by name.
        """
        return self.numbers_as_of(t, name)[1]

    def numbers_as_of(self, t: float, name: str) -> tuple[Fraction, dict[str, float]]:
        """:meth:`numbers`, with the instant the answer is of (``as_of``:
        the last instant at or before ``t`` the session's frames are on).
        """
        state = self.server.handle("state_at", {"t": t})["state"]
        (node,) = [node for node in state["nodes"] if node["name"] == name]
        found = state["quantities"].get(node["id"], [])
        assert not [each for each in found if "unreadable" in each], found
        assert {each["unit"] for each in found} <= {"u", "1"}
        as_of = Fraction(state["as_of_exact"])
        return as_of, {each["name"]: each["value"] for each in found}

    def close(self) -> None:
        self.server.handle("close", {})


@pytest.fixture
def server(tmp_path):
    opened: list[_Server] = []

    def open_scene(scene: str) -> _Server:
        opened.append(_Server(tmp_path, scene))
        return opened[-1]

    yield open_scene
    for each in opened:
        each.close()


def test_state_at_answers_the_numbers_each_instant_s_effects_hold(server):
    opened = server("NumbersOverTime")
    # A blur set twice is one quantity, the second; and the literal another
    # owner attached is still there, beside the effects and after them.
    assert opened.numbers(0.25, "square") == {
        "effect.blur": 0.3,
        "effect.depth": 1.5,
        "effect.mask_feather": 0.2,
        "probe.size": 2.0,
    }
    assert opened.numbers(0.75, "square") == {"probe.size": 2.0}


def test_state_at_answers_a_changing_blur_at_each_instant(server):
    opened = server("ChangingBlur")
    for t in (0.25, 0.5, 0.75):
        as_of, found = opened.numbers_as_of(t, "plane")
        # The play runs from 0 to 1 linearly: its share done is the instant.
        assert 0 < as_of < 1
        sigma = 0.2 * (1 - as_of) + 0.6 * as_of
        assert found == {"effect.blur": pytest.approx(float(sigma))}
    assert opened.numbers(1.0, "plane") == {"effect.blur": 0.6}
    # Brought to 0, the blur is gone from the spec and from the answer.
    assert opened.numbers(1.6, "plane") == {}


def test_a_frame_of_a_scene_with_an_effect_is_refused_with_no_gpu(
    tmp_path, server, no_gpu
):
    opened = server("BlurredStep")
    picture = tmp_path / "frame.png"
    with pytest.raises(RpcError) as refused:
        opened.server.handle("frame", {"t": 0.1, "path": str(picture)})
    assert refused.value.code == REFUSED
    assert refused.value.data["reason"] == "render.gpu_required"
    assert not picture.exists()
    # What needs no picture still answers.
    assert opened.numbers(0.1, "plane") == {"effect.blur": 0.15}


def test_a_focus_racked_in_a_scene_without_one_names_the_call_that_gives_one(
    tmp_path,
):
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    with pytest.raises(RpcError) as raised:
        server.handle(
            "open",
            {"file": str(SCENES), "scene": "RackWithoutFocus", "config": config},
        )
    assert raised.value.code == SCENE_ERROR
    assert raised.value.data["cls"] == "ValueError"
    assert "set_focus(scene, depth, aperture)" in str(raised.value.data)
