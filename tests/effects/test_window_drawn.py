"""Windows drawn on this run's device (:mod:`tests.helpers.gpu_door`).

A scene shown in a window at whole pixels is compared with a scene of its
objects alone drawn at the window's size, as a comparison of one renderer
with itself (:mod:`arachne_compare`): its frames, the presence of every
object on both sides, and flicker over a play in which one of them moves.
The window's side has no object map a replay can make (what a window shows
is drawn only through it), so the test makes it: an object's footprint is
where the window, its scene built again without the object, changes.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from arachne_compare import (
    ArrayFrames,
    Footprint,
    HandedMaps,
    ObjectMap,
    cheap,
    compare,
)

from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, RpcError
from arachne.session import Session
from arachne.timeline.footprint import changed_between, object_map
from tests.helpers import gpu_door

from .window_scenes import (
    CARRIER_CORNERS,
    FOCUS,
    FRAME,
    PARENT_RGBA,
    SCALE,
    SIZES,
    shown_name,
)

SCENES = Path(__file__).parent / "window_scenes.py"
NEAREST, BILINEAR = 0, 2

#: The instants each case is compared at: the start, the middle and the end
#: of the play in which its mover moves. At one frame a second each is the
#: first frame of its second, so the comparator samples all three.
INSTANTS = (0, 1, 2)
FPS = Fraction(1)


@pytest.fixture
def opened():
    """Open a scene of the window scenes in a session, in their frame unless
    the settings say otherwise.
    """
    sessions: list[Session] = []

    def open_session(name: str, **settings) -> Session:
        session = Session.open(SCENES, name, config={**FRAME, **settings})
        sessions.append(session)
        return session

    yield open_session
    for session in sessions:
        session.close()


@pytest.fixture
def drawn(door, monkeypatch) -> list:
    """Every frame the stack hands this run's device, in order."""
    frames: list = []
    draw = gpu_door.Drawn.draw

    def recording(self, frame):
        frames.append(frame)
        return draw(self, frame)

    monkeypatch.setattr(gpu_door.Drawn, "draw", recording)
    return frames


def _frame(session: Session, t) -> np.ndarray:
    return np.asarray(session.frame(t))


def _windows(frame) -> list:
    """The windows a frame the stack drew shows, outermost first."""
    import arachne_gpu

    found = []
    pending = list(frame.layers)
    while pending:
        layer = pending.pop(0)
        if isinstance(layer.source, arachne_gpu.Window):
            found.append(layer.source)
            pending.extend(layer.source.frame.layers)
    return found


def _in_pixels(points) -> list[tuple[float, float]]:
    """Scene points in pixels of the window scenes' frame, computed here."""
    return [(x * SCALE + 48, 27 - y * SCALE) for x, y in points]


def _outside(size: str) -> np.ndarray:
    (x0, y0), (width, height) = SIZES[size]
    found = np.ones((FRAME["pixel_height"], FRAME["pixel_width"]), dtype=bool)
    found[y0 : y0 + height, x0 : x0 + width] = False
    return found


def _cut(frame: np.ndarray, size: str) -> np.ndarray:
    (x0, y0), (width, height) = SIZES[size]
    return np.ascontiguousarray(frame[y0 : y0 + height, x0 : x0 + width])


class _Case:
    """One case drawn both ways: the window scene, ``without`` some objects
    for the planted case, and the scene of the objects alone at the
    window's size.
    """

    def __init__(self, opened, case: str, size: str, without=()) -> None:
        self.case, self.size, self.without = case, size, tuple(without)
        self.opened = opened
        (_, (width, height)) = SIZES[size]
        self.reference = opened(f"Alone{case}", pixel_width=width, pixel_height=height)
        self.shown = opened(shown_name(case, size, self.without))
        record = self.reference.record
        self.ids = {
            name: record.origin(mob).id
            for name, mob in self.reference.scene.content.items()
        }
        self.whole = [_frame(self.shown, t) for t in INSTANTS]

    def frames(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """The reference's frames, and the window's rectangle cut from the
        window scene's.
        """
        reference = [_frame(self.reference, t) for t in INSTANTS]
        return reference, [_cut(whole, self.size) for whole in self.whole]

    def reference_maps(self) -> dict[int, ObjectMap]:
        ids = list(self.ids.values())
        return {i: object_map(self.reference, t, ids) for i, t in enumerate(INSTANTS)}

    def window_maps(self) -> dict[int, ObjectMap]:
        """Each object's footprint in the window, by the reference's ids:
        where the window scene, built again without the object, differs
        inside the rectangle (an object the window lacks draws nothing).
        """
        cut = [_cut(whole, self.size) for whole in self.whole]
        height, width = cut[0].shape[:2]
        found: dict[int, dict[str, Footprint]] = {i: {} for i in range(len(INSTANTS))}
        for name, ident in self.ids.items():
            if name in self.without:
                for objects in found.values():
                    objects[ident] = Footprint((0, 0, 0, 0), np.zeros((0, 0), bool))
                continue
            again = self.opened(shown_name(self.case, self.size, (*self.without, name)))
            for i, t in enumerate(INSTANTS):
                changed = (_cut(_frame(again, t), self.size) != cut[i]).any(axis=-1)
                found[i][ident] = Footprint((0, 0, width, height), changed)
        return {
            i: ObjectMap((height, width), objects, complete=True)
            for i, objects in found.items()
        }

    def verdict(self):
        reference, window = self.frames()
        return compare(
            ArrayFrames(reference, FPS),
            ArrayFrames(window, FPS),
            mode="same",
            maps=HandedMaps(self.reference_maps(), self.window_maps()),
            changed=lambda i, j: changed_between(
                self.reference, INSTANTS[i], INSTANTS[j]
            ),
            plays=[(0, len(INSTANTS))],
            fps=FPS,
            seed=0,
            key=shown_name(self.case, self.size, self.without),
        )


#: Each case, its window's size, and whether its objects alone are drawn
#: on the GPU (they hold an effect or a window), where the window's
#: rectangle equals them byte for byte; else they are a Cairo frame, which
#: the window's equals within 1 of 255.
CASES = [
    ("Plain", "Half", False),
    ("Plain", "Third", False),
    ("Blurred", "Half", True),
    ("Masked", "Half", True),
    ("Focused", "Half", True),
    ("Nested", "Half", True),
]


@pytest.mark.parametrize(("case", "size", "gpu_alone"), CASES)
def test_a_scene_shown_in_a_window_matches_it_drawn_alone_at_the_window_s_size(
    opened, drawn, case, size, gpu_alone
):
    shown = _Case(opened, case, size)
    assert shown.reference.route == ("gpu" if gpu_alone else "cpu")
    assert shown.shown.route == "gpu"
    # The window is placed at its whole-pixel corners, at its own size, by
    # copying its texels.
    (x0, y0), (width, height) = SIZES[size]
    (window, *_) = _windows(drawn[-1])
    assert window.size == (width, height)
    assert window.resampling == NEAREST
    corners = [(x0, y0), (x0 + width, y0), (x0, y0 + height), (x0 + width, y0 + height)]
    assert [tuple(corner) for corner in window.corners] == corners
    reference, cut = shown.frames()
    for i, whole in enumerate(shown.whole):
        # Nothing of the window, its mover or its effects, leaves the
        # rectangle: outside it the frame is the parent's.
        assert (whole[_outside(size)] == PARENT_RGBA).all(), INSTANTS[i]
        difference = np.abs(cut[i].astype(int) - reference[i].astype(int))
        assert int(difference.max()) <= (0 if gpu_alone else 1), INSTANTS[i]
    # The mover moved.
    assert not np.array_equal(cut[0], cut[-1])
    verdict = shown.verdict()
    assert verdict.sampled == tuple(range(len(INSTANTS)))
    assert verdict.findings == ()
    assert verdict.counts["presence_untested"] == 0
    assert verdict.counts["flicker_unjudged"] == 0


def test_a_window_lacking_one_small_object_is_blocked_for_its_presence(opened, door):
    planted = _Case(opened, "Plain", "Half", without=("dot",))
    reference, window = planted.frames()
    # Its frames alone pass: the dot is too faint to make a cluster.
    frames = cheap(ArrayFrames(reference, FPS), ArrayFrames(window, FPS), mode="same")
    assert not any(frame.flagged for frame in frames)
    verdict = planted.verdict()
    assert verdict.findings
    assert {finding.code for finding in verdict.findings} == {"gate.presence.missing"}
    assert {finding.object for finding in verdict.findings} == {planted.ids["dot"]}
    assert verdict.counts["presence_untested"] == 0


def test_a_window_s_numbers_are_converted_with_its_own_camera(opened, drawn):
    session = opened("BlurInAndOut")
    _frame(session, Fraction(1, 10))
    frame = drawn[-1]
    (window,) = _windows(frame)
    assert (window.size, window.resampling) == ((48, 27), NEAREST)
    # The frame's 8 pixels to a scene unit, the window's 4.
    (outside,) = [layer for layer in frame.layers if layer.blur is not None]
    assert outside.blur.sigma_px == pytest.approx(0.5 * 8)
    blurred, masked = window.frame.layers
    assert blurred.blur.sigma_px == pytest.approx(0.5 * 4)
    assert masked.mask.feather_px == pytest.approx(0.5 * 4)
    depth, aperture = FOCUS
    assert (frame.focus.depth, frame.focus.aperture) == pytest.approx(
        (depth, aperture * 8)
    )
    assert (window.frame.focus.depth, window.frame.focus.aperture) == pytest.approx(
        (depth, aperture * 4)
    )


def test_a_window_off_the_pixel_grid_is_rounded_and_placed_at_its_true_corners(
    opened, drawn
):
    session = opened("OffGrid")
    _frame(session, Fraction(1, 10))
    (window,) = _windows(drawn[-1])
    # 4.3 x 0.05 scene units: 34.4 x 0.4 pixels.
    assert window.size == (34, 1)
    assert window.resampling == BILINEAR
    left, right, top, bottom = 0.1 - 2.15, 0.1 + 2.15, 0.3 + 0.025, 0.3 - 0.025
    expected = _in_pixels([(left, top), (right, top), (left, bottom), (right, bottom)])
    for found, corner in zip(window.corners, expected, strict=True):
        assert tuple(found) == pytest.approx(corner)


@pytest.mark.parametrize(
    ("scene", "sized"),
    [
        ("ImageCarrier", ((32, 16), NEAREST)),
        ("ImageCarrierMirrored", None),
        ("ImageCarrierQuarter", None),
        ("ImageCarrierTilted", None),
    ],
)
def test_an_image_carrier_is_sized_only_at_a_level_rectangle(
    opened, drawn, scene, sized
):
    session = opened(scene)
    drawn_frame = _frame(session, Fraction(1, 10))
    assert drawn_frame.shape == (54, 96, 4)
    (window,) = _windows(drawn[-1])
    if sized is None:
        # As every window was drawn before windows had sizes: at the frame's
        # size, placed by nearest at the image's corners.
        assert (window.size, window.resampling) == (None, NEAREST)
    else:
        assert (window.size, window.resampling) == sized
    if scene in ("ImageCarrier", "ImageCarrierMirrored", "ImageCarrierQuarter"):
        placed = scene.removeprefix("ImageCarrier").lower() or "level"
        expected = _in_pixels(CARRIER_CORNERS[placed])
        assert [tuple(corner) for corner in window.corners] == expected


def test_a_mask_of_two_kinds_in_a_window_matches_it_drawn_alone(opened, door):
    alone = opened("AloneMixedMask", pixel_width=48, pixel_height=27)
    shown = opened(shown_name("MixedMask", "Half"))
    for t in INSTANTS:
        assert np.array_equal(_cut(_frame(shown, t), "Half"), _frame(alone, t)), t


def test_a_mask_of_two_kinds_in_a_window_larger_than_the_frame_is_refused(opened, door):
    from arachne.compose.stack import Unsupported

    session = opened("WindowTooLargeForMixedMask")
    with pytest.raises(
        Unsupported, match="a mask of several kinds in a window of 128 x 72"
    ):
        session.frame(Fraction(1, 10))


def test_a_blur_too_wide_for_its_window_reaches_the_server_with_the_gpu_build_s_words(
    tmp_path, door
):
    server = Dispatcher()
    config = {**FRAME, "media_dir": str(tmp_path / "media")}
    try:
        server.handle(
            "open",
            {"file": str(SCENES), "scene": "BlurTooWideForWindow", "config": config},
        )
        with pytest.raises(RpcError) as refused:
            server.handle("frame", {"t": 0.1, "path": str(tmp_path / "f.png")})
    finally:
        server.handle("close", {})
    assert refused.value.code == REFUSED
    assert refused.value.data["reason"] == "render.unsupported"
    message = refused.value.data["message"]
    assert "in its window, layer " in message
    assert (
        "a blur of sigma 8 px reaches 24 px, past the window's longer side of 16 px;"
        " at this size the largest sigma is 5.33 px (reduce the blur or show the"
        " window larger)"
    ) in message
    assert not (tmp_path / "f.png").exists()
