"""A session of a scene holding a GPU-only capability, on a host with a GPU
(here a software adapter, through the GPU build's testing door): every
frame is a stack drawn by the compositor, and names the device that drew it.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from arachne_gpu.testing import software_compositor

from arachne import api_json
from arachne.compose import device
from arachne.compose.layer import layer_of
from arachne.compose.stack import Unsupported
from arachne.frame_ir.xf import image_key
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError
from arachne.server.store import Store
from arachne.session import Session
from arachne.timeline.footprint import changed_between, drawables, object_map

from .compose_scenes import TINT
from .conftest import PIN, SCENES, SMALL, _Door

HALF = Fraction(1, 2)


class _Counted:
    """A software compositor that counts its draws and keeps the images
    uploaded to it; with ``pin``, it names that device instead of its own.
    """

    def __init__(self, inner, pin: dict[str, str] | None) -> None:
        self._inner = inner
        self._pin = pin
        self.draws = 0
        self.images: list[tuple[bytes, np.ndarray]] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def pin(self):
        return dict(self._pin) if self._pin is not None else self._inner.pin

    def draw(self, frame):
        self.draws += 1
        return self._inner.draw(frame)

    def upload_image(self, key, rgba):
        self.images.append((key, np.array(rgba)))
        self._inner.upload_image(key, rgba)


class _Software:
    """A host whose GPU is the software adapter."""

    def __init__(self, pin: dict[str, str] | None = None) -> None:
        self._pin = pin
        self.opened: list[_Counted] = []

    def pin(self) -> dict[str, str]:
        return (
            dict(self._pin) if self._pin is not None else software_compositor(4, 4).pin
        )

    def open(self, width: int, height: int) -> _Counted:
        self.opened.append(_Counted(software_compositor(width, height), self._pin))
        return self.opened[-1]

    @property
    def draws(self) -> int:
        return sum(opened.draws for opened in self.opened)


@pytest.fixture
def software(monkeypatch) -> _Software:
    door = _Software()
    monkeypatch.setattr(device, "door", door)
    return door


def _pixels(session, t, quality=None) -> np.ndarray:
    return np.asarray(session.frame(t, quality)).astype(int)


def _close(a: np.ndarray, b: np.ndarray, within: int = 1) -> bool:
    return a.shape == b.shape and int(np.abs(a - b).max()) <= within


def _at(session, x: float, y: float) -> tuple[int, int]:
    """The pixel, ``(column, row)``, of the scene point ``(x, y)``."""
    camera = session.scene.renderer.camera
    column = (x / camera.frame_width + 0.5) * camera.pixel_width
    row = (0.5 - y / camera.frame_height) * camera.pixel_height
    return int(column), int(row)


def test_every_frame_is_drawn_by_the_compositor_and_names_its_device(
    session_of, software
):
    session = session_of("EffectLast")
    assert session.route == "gpu"
    session.frame(HALF)  # in the first play, which holds nothing
    session.frame(Fraction(3, 2))
    assert software.draws == 2
    (opened,) = software.opened
    named = {"name": "gpu", "pin": dict(opened.pin)}
    assert session.renderer == named
    answers = session.answers([HALF, Fraction(3, 2)])
    assert software.draws == 4
    assert [answer.renderer for answer in answers] == [named, named]


def test_a_stack_without_an_effect_draws_what_the_camera_draws(session_of, software):
    stacked = _pixels(session_of("EffectLast"), HALF)
    drawn = _pixels(session_of("EffectLastRemoved"), HALF)
    assert _close(stacked, drawn)
    # The blurred circle of the last play is not the camera's.
    blurred = _pixels(session_of("EffectLast"), Fraction(3, 2))
    assert not _close(blurred, _pixels(session_of("EffectLastRemoved"), Fraction(3, 2)))


def _edge_width(row: np.ndarray) -> float:
    """From where a falling step is at 90 % to where it is at 10 %, in
    pixels, read by linear interpolation between pixel centres.
    """
    level = row / 255.0

    def crossing(at: float) -> float:
        i = int(np.argmax(level < at))
        return i - 1 + (level[i - 1] - at) / (level[i - 1] - level[i])

    return crossing(0.1) - crossing(0.9)


@pytest.mark.parametrize("size", [(96, 54), (384, 216)])
def test_a_blur_spreads_an_edge_by_its_declared_width_at_every_size(
    session_of, software, size
):
    width, height = size
    session = session_of("BlurredEdge", pixel_width=width, pixel_height=height)
    camera = session.scene.renderer.camera
    sigma_px = 0.25 * width / camera.frame_width
    row = _pixels(session, Fraction(1, 10))[height // 2, :, 0]
    assert _edge_width(row) == pytest.approx(2.563 * sigma_px, abs=1)


def test_a_translucent_still_is_uploaded_as_an_image_and_stacked_painted(
    session_of, software
):
    stacked = _pixels(session_of("TintedStill"), Fraction(1, 10))
    alone = _pixels(session_of("TintedStillAlone"), Fraction(1, 10))
    assert _close(stacked, alone)
    images = [image for opened in software.opened for image in opened.images]
    assert images
    for key, rgba in images:
        # Uploaded straight, under the Frame IR's key of those bytes.
        assert key == image_key(rgba)
        assert tuple(rgba[rgba.shape[0] // 2, rgba.shape[1] // 2]) == TINT


def test_on_a_transparent_backdrop_the_frame_is_painted(session_of, software):
    session = session_of("TintedStill", background_opacity=0.0)
    found = _pixels(session, Fraction(1, 10))
    red, green, blue, alpha = TINT
    painted = [round(c * alpha / 255) for c in (red, green, blue)] + [alpha]
    centre = found[found.shape[0] // 2, found.shape[1] // 2]
    assert np.abs(centre - np.array(painted)).max() <= 1


def test_a_moving_mask_and_window_are_drawn_where_they_were_at_each_instant(
    session_of, software
):
    session = session_of("MovingMask")
    early, late = Fraction(1, 4), Fraction(3, 4)
    late_first = _pixels(session, late)
    early_then = _pixels(session, early)
    assert _close(_pixels(session, late), late_first, within=0)
    assert _close(_pixels(session_of("MovingMask"), early), early_then, within=0)
    left, right = _at(session, -2, 0), _at(session, 2, 0)
    # The square shows only inside the mask's circle.
    assert early_then[left[1], left[0], 0] > 200 > early_then[right[1], right[0], 0]
    assert late_first[right[1], right[0], 0] > 200 > late_first[left[1], left[0], 0]
    # The window shows its dot moved.
    (x0, y0), (x1, y1) = _at(session, 2.5, 4), _at(session, 5.5, 2)
    assert not _close(early_then[y0:y1, x0:x1], late_first[y0:y1, x0:x1], within=40)


def _blurred_edge(pixels: np.ndarray) -> int:
    """How many pixels of the middle row are neither background nor fill."""
    row = pixels[pixels.shape[0] // 2, :, 0]
    return int(((row > 20) & (row < 235)).sum())


def test_a_layer_set_and_withdrawn_blurs_only_the_frames_it_holds(session_of, software):
    session = session_of("LayerComesAndGoes")
    plain, blurred, more, again = (
        _pixels(session, Fraction(k, 4)) for k in (1, 3, 5, 7)
    )
    assert _close(plain, again)
    assert _blurred_edge(plain) < _blurred_edge(blurred) < _blurred_edge(more)


def _ids(session):
    record = session.record
    holders = [mob for mob in record.mobjects() if layer_of(mob) is not None]
    (square,) = [mob for mob in holders if layer_of(mob).mask is not None]
    (frame,) = [mob for mob in holders if layer_of(mob).window is not None]
    name = lambda mob: record.origin(mob).id  # noqa: E731
    return (
        (name(square), name(layer_of(square).mask)),
        (name(frame), name(layer_of(frame).window[0])),
    )


def test_a_carrier_changes_when_what_its_layer_reads_does(session_of, software):
    session = session_of("MovingMask")
    pairs = _ids(session)
    moved = changed_between(session, Fraction(1, 4), Fraction(3, 4))
    still = changed_between(session, Fraction(1, 10), Fraction(4, 10))
    for carrier, dependency in pairs:
        assert carrier in moved.ids
        assert dependency not in moved.ids
        assert carrier not in still.ids
    listed = drawables(session, Fraction(1, 4))
    for carrier, dependency in pairs:
        assert carrier in listed
        assert dependency not in listed


def test_what_needs_no_picture_is_the_same_with_and_without_a_gpu(
    request, session_of, software
):
    def answers(session):
        state = api_json.state(session.state_at(HALF), session)
        found = api_json.lint(session.lint())
        found["report"].pop("duration_s")
        query = api_json.query(session.query(HALF), ("overlaps", "occlusions"))
        return state, found, query

    with_gpu = answers(session_of("EffectLast"))
    request.getfixturevalue("no_gpu")
    assert answers(session_of("EffectLast")) == with_gpu


def test_a_forward_run_s_frame_is_drawn_on_the_device(session_of, software):
    session = session_of("ReboundLayer")
    (answer,) = session.answers([HALF])
    assert answer.state.answer == "forward"
    assert answer.frame is not None
    assert answer.renderer == {"name": "gpu", "pin": dict(software.opened[-1].pin)}
    assert software.draws == 1


def test_a_layer_inside_another_is_refused_not_flattened(session_of, software):
    with pytest.raises(Unsupported) as refused:
        session_of("Nested").frame(Fraction(1, 10))
    assert refused.value.code == "render.unsupported"
    assert software.draws == 0


def test_the_server_answers_a_layer_inside_another_as_a_refusal(tmp_path, software):
    server = Dispatcher(Store())
    try:
        server.handle("open", {"file": str(SCENES), "scene": "Nested", "config": SMALL})
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.1, "path": str(tmp_path / "f.png")})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    assert refused.value.data["reason"] == "render.unsupported"


def test_the_store_keeps_no_gpu_frame_and_another_device_draws_it_anew(
    tmp_path, monkeypatch, no_gpu_door
):
    params = {"file": str(SCENES), "scene": "EffectLast", "config": SMALL}

    def frame(door, picture: Path) -> dict:
        monkeypatch.setattr(device, "door", door)
        server = Dispatcher(Store())
        try:
            server.handle("open", params)
            return server.handle("frame", {"t": 1.5, "path": str(picture)})
        finally:
            server.handle("close", {})

    first, other = _Software(), _Software(dict(PIN))
    found = frame(first, tmp_path / "first.png")
    assert found["frame"]["renderer"]["pin"] == dict(first.opened[0].pin)
    assert first.draws == 1
    # Nothing drawn on the GPU was kept: the store answers no frame.
    server = Dispatcher(Store())
    try:
        server.handle("open", params)
        assert server.session is None
        assert server.store.answer(server.stored, "frame", {"t": 1.5}) is None
    finally:
        server.handle("close", {})
    # Opened from the store under another device: drawn again, by it.
    assert frame(other, tmp_path / "other.png")["frame"]["renderer"] == {
        "name": "gpu",
        "pin": PIN,
    }
    assert other.draws == 1
    # With the device gone, refused from the store, nothing executed or written.
    monkeypatch.setattr(Session, "open", _never_executed)
    hidden = tmp_path / "hidden.png"
    with pytest.raises(RpcError) as refused:
        frame(no_gpu_door, hidden)
    assert refused.value.data["reason"] == "render.gpu_required"
    assert not hidden.exists()


def _never_executed(*args, **kwargs):
    raise AssertionError("the scene was executed")


@pytest.fixture
def no_gpu_door():
    """A host with no GPU, as a door to install later."""
    return _Door(None)


def test_a_translucent_backdrop_is_stacked_as_its_painted_values(session_of, software):
    session = session_of(
        "EffectLast", background_color="#FF8000", background_opacity=0.5
    )
    straight = session.scene.renderer.camera.background[0, 0].astype(int)
    alpha = straight[3]
    painted = [round(c * alpha / 255) for c in straight[:3]] + [alpha]
    corner = _pixels(session, HALF)[0, 0]
    assert 0 < alpha < 255
    assert np.abs(corner - np.array(painted)).max() <= 1


def test_a_mask_of_two_kinds_is_stacked_from_its_runs(session_of, software):
    session = session_of("MixedMask")
    found = _pixels(session, Fraction(1, 10))
    left, middle, right = (_at(session, x, 0) for x in (-2, 0, 2))
    assert found[left[1], left[0], 0] > 200
    assert found[right[1], right[0], 0] > 200
    assert found[middle[1], middle[0], 0] < 50


def test_a_window_s_layers_are_defocused_by_the_frame_s_focus(session_of, software):
    session = session_of("FocusInWindow")
    early, late = Fraction(1, 4), Fraction(3, 4)
    late_first = _pixels(session, late)
    early_then = _pixels(session, early)
    assert _close(_pixels(session, late), late_first, within=0)
    assert _blurred_edge(early_then) < _blurred_edge(late_first)


#: Scenes where a layer's mask is shown or listed besides being read, with
#: the scene that only reads it, which each must draw the same.
DUAL_ROLE = [
    ("MaskShownToo", "MaskNotShown"),
    ("WindowMaskShownToo", "WindowMaskNotShown"),
    ("WindowMaskListed", "WindowMaskNotShown"),
]


def _masks_and_carriers(session) -> tuple[set[str], set[str]]:
    """The ids of every mask a layer of the scene reads, at any depth, and
    of the scene's top-level carriers.
    """
    record = session.record
    holders = [mob for mob in record.mobjects() if layer_of(mob) is not None]
    masks = {
        record.origin(layer_of(mob).mask).id
        for mob in holders
        if layer_of(mob).mask is not None
    }
    roots = {id(mob) for mob in record.final.roots}
    carriers = {record.origin(mob).id for mob in holders if id(mob) in roots}
    return masks, carriers


@pytest.mark.parametrize(("scene", "reads_only"), DUAL_ROLE)
def test_a_mask_shown_or_listed_too_is_painted_only_through_its_layer(
    session_of, software, scene, reads_only
):
    t = Fraction(1, 10)
    shown = session_of(scene)
    assert _close(_pixels(shown, t), _pixels(session_of(reads_only), t), within=0)
    masks, _ = _masks_and_carriers(shown)
    listed = drawables(shown, t)
    assert masks
    assert not masks & set(listed)


@pytest.mark.parametrize(("scene", "reads_only"), [*DUAL_ROLE, ("MovingMask", None)])
def test_an_object_map_of_every_drawable_holds_none_of_what_layers_read(
    session_of, software, scene, reads_only
):
    session = session_of(scene)
    t = Fraction(1, 4) if scene == "MovingMask" else Fraction(1, 10)
    masks, carriers = _masks_and_carriers(session)
    asked = list(drawables(session, t))
    found = object_map(session, t, asked)
    assert not masks & set(found.objects)
    assert carriers
    for carrier in carriers:
        footprint = found.objects[carrier]
        assert footprint.mask is not None
        assert footprint.mask.any()


def test_a_picture_a_zoom_fades_is_drawn_faded_and_back_as_before(session_of, software):
    session = session_of("ZoomPastAPictureOnGpu")
    assert session.route == "gpu"
    rate = SMALL["frame_rate"]
    before = _pixels(session, Fraction(3, rate))
    # Drawn through the move-in, the picture's pixels uploaded faded.
    for tick in range(8, 39):
        session.frame(Fraction(tick, rate))
    alphas = {
        int(rgba[..., 3].max())
        for opened in software.opened
        for _, rgba in opened.images
        if rgba.shape[:2] == (16, 16)
    }
    assert any(0 < alpha < int(255 * 0.6) for alpha in alphas)
    # Back, the frame is drawn as before the move-in, to the byte.
    assert _close(_pixels(session, Fraction(78, rate)), before, within=0)
