"""A frame drawn as a stack: Cairo draws each run of objects, the GPU
compositor stacks the runs and applies the layers' effects.

A drawing given a :class:`Stack` (:class:`~arachne.timeline.draw.Drawing`)
hands it every list of objects it would paint, in the camera's order, and
the stack draws the frame from them. The objects are cut into runs:

- an object that is a layer of its own (:func:`.layer.set_layer`) is a run
  with its whole family, carrying its effects;
- between two such, the objects are cut again wherever the camera's own
  batching by kind changes, because the camera does not leave one kind of
  value behind: Cairo paints vector objects into painted values, images go
  through Pillow's straight-alpha composite, and point clouds are written
  as they are. A run of vector objects is uploaded as a painted picture, a
  run of images or of points as an image (straight values, which the
  compositor converts once), each drawn by the camera on a transparent
  backdrop; no run mixes two kinds. A layer whose family, or a mask, mixes
  kinds is stacked from its runs, never painted into one array.

The camera's background, straight as its colour or image is, is the bottom
layer: an opaque uniform one is the frame's fill, any other an image.
Painted "over" is associative, so a frame whose layers carry no effect
stacks to what the camera paints in one pass, up to rounding.

What a layer reads (its mask, its window's objects) is drawn only through
it, never on its own, even when the scene shows it too (:meth:`Stack.own`).
The camera's focus defocuses the layers with a depth everywhere, inside a
window's stack too.

A layer inside another layer's family is drawn only through a window (the
window's objects are a stack of their own); any other nesting is refused
with ``render.unsupported``, never flattened.
"""

from __future__ import annotations

__all__ = ["Stack", "Unsupported"]

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.frame_ir.xf import image_key

from .layer import LayerSpec, focus_at, layer_of, pixels

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, Record

#: The frame's corners in CE's image order (upper left, upper right, lower
#: left, lower right), as fractions of its width and height.
_FRAME_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))
#: The one resampling this build draws: nearest.
_NEAREST = 0
_TRANSPARENT = bytes(4)


class Unsupported(Exception):
    """``render.unsupported``: the frame asks for what the stack does not
    draw; it is refused whole. ``kind`` names what.
    """

    code = "render.unsupported"

    def __init__(self, kind: str, reason: str) -> None:
        super().__init__(f"{self.code}: {reason}; the frame is not drawn")
        self.kind = kind


class Stack:
    """The stack one drawing builds its frames with, on ``compositor``, for
    ``record``'s scene, drawn by its camera.
    """

    def __init__(self, compositor: Any, record: Record) -> None:
        self.compositor = compositor
        self.record = record
        self.camera: Camera = record._scene.renderer.camera
        self._leaves: list[tuple[Mobject, Mobject | None]] = []
        self._capture: Capture | None = None
        #: The layer each member of the frame's tree belongs to, by id, and
        #: the members of what the layers read, found at the frame's first
        #: list (:func:`_carriers`).
        self._owners: dict[int, Mobject] | None = None
        self._dependents: set[int] = set()
        #: How many frames the compositor drew for this stack.
        self.drawn = 0
        #: The focus of the frame being drawn, every stack of it under it.
        self._focus: Any = None

    def at(self, capture: Capture) -> None:
        """The frame about to be drawn is ``capture``'s."""
        self._capture = capture
        self._owners = None

    def begin(self) -> None:
        """Start a frame: nothing on it but the background."""
        self._leaves = []
        self._owners = None

    def own(self, handed: Sequence[Mobject], leaves: list[Mobject]) -> list[Mobject]:
        """``leaves``, what the camera would paint of ``handed``, without the
        members of what a layer of the frame reads: those are painted only
        through their layer.
        """
        self._find_owners(handed)
        dependents = self._dependents
        return [leaf for leaf in leaves if id(leaf) not in dependents]

    def add(self, handed: Sequence[Mobject], leaves: Iterable[Mobject]) -> None:
        """Put ``leaves``, what the camera would paint of ``handed`` in that
        order, on the frame, over what is on it.
        """
        owners = self._find_owners(handed)
        self._leaves.extend((leaf, owners.get(id(leaf))) for leaf in leaves)

    def _find_owners(self, handed: Sequence[Mobject]) -> dict[int, Mobject]:
        """The layer of each member of the frame's tree (and the members of
        what the layers read), found once a frame.
        """
        if self._owners is None:
            # The lists a frame paints are often flattened (a play's moving
            # objects, a frame's static ones), so a leaf's layer is found
            # from the whole tree of the frame, as it stands now.
            capture = self._capture
            roots = (
                [*capture.draw_list(), *capture.moving]
                if capture is not None
                else list(handed)
            )
            self._owners = _carriers(roots, self._origin)
            carriers = {id(carrier): carrier for carrier in self._owners.values()}
            self._dependents = {
                id(member)
                for carrier in carriers.values()
                for read in _spec(carrier).dependencies
                for member in read.get_family()
            }
        return self._owners

    def draw(self) -> np.ndarray:
        """The frame: its stack composed by the compositor and read back,
        painted, as the camera's pixel array holds a frame.
        """
        import arachne_gpu

        compositor = self.compositor
        uploads: dict[bytes, None] = {}
        camera = self.camera
        try:
            self._focus = None
            if self._capture is not None and self._capture.camera is not None:
                found = focus_at(dict(self._capture.camera.values))
                if found is not None:
                    self._focus = arachne_gpu.Focus(found[0], pixels(camera, found[1]))
            background = camera.background
            assert background is not None
            first = background[0, 0]
            layers: list[Any] = []
            if bool((background == first).all()) and int(first[3]) == 255:
                # Opaque: its straight values are its painted ones.
                fill = bytes(np.asarray(first, dtype=np.uint8))
            else:
                # Straight, as the camera holds it: uploaded as an image,
                # which the compositor converts to painted values once.
                fill = _TRANSPARENT
                rgba = np.ascontiguousarray(background, dtype=np.uint8)
                key = self._upload(rgba, False, uploads)
                layers.append(arachne_gpu.Layer(arachne_gpu.Picture(key)))
            layers.extend(self._layers(self._leaves, uploads))
            frame = arachne_gpu.Frame(fill, layers, self._focus)
            try:
                pixels_drawn = compositor.draw(frame)
            except arachne_gpu.Unsupported as refused:
                raise Unsupported("composite", refused.reason) from None
            self.drawn += 1
            drawn = np.ascontiguousarray(pixels_drawn)
            camera.set_pixel_array(drawn)
            return drawn
        finally:
            for key in uploads:
                compositor.release(key)

    def _origin(self, mob: Mobject) -> str:
        return self.record.origin(mob).id

    def _layers(
        self, leaves: Sequence[tuple[Mobject, Mobject | None]], uploads: dict
    ) -> list[Any]:
        """The layers of ``leaves``, each with the carrier of its layer (or
        ``None``), in paint order.
        """
        import arachne_gpu

        found: list[Any] = []
        done: set[int] = set()
        for run, carrier in _runs(leaves, self._kind):
            if carrier is None:
                key = self._paint(run, uploads)
                found.append(arachne_gpu.Layer(arachne_gpu.Picture(key)))
                continue
            if id(carrier) in done:
                raise Unsupported(
                    "layer",
                    f"{self._origin(carrier)} is a layer whose objects are painted"
                    " apart, other objects between them",
                )
            done.add(id(carrier))
            layer = self._layer(carrier, run, uploads)
            if layer is not None:
                found.append(layer)
        return found

    def _layer(self, carrier: Mobject, run: list[Mobject], uploads: dict) -> Any | None:
        """The layer ``carrier`` makes of ``run``, its painted objects, with
        its effects; ``None`` when it paints nothing.
        """
        import arachne_gpu

        spec = layer_of(carrier)
        assert spec is not None
        camera = self.camera
        if spec.window is not None:
            content = self._stack_of(spec.window, uploads)
            source: Any = arachne_gpu.Window(content, self._corners(carrier), _NEAREST)
        else:
            parts = list(_runs([(leaf, None) for leaf in run], self._kind))
            if not parts:
                return None
            if len(parts) == 1:
                source = arachne_gpu.Picture(self._paint(parts[0][0], uploads))
            else:
                inner = [
                    arachne_gpu.Layer(arachne_gpu.Picture(self._paint(part, uploads)))
                    for part, _ in parts
                ]
                source = arachne_gpu.Window(
                    arachne_gpu.Frame(_TRANSPARENT, inner), self._frame(), _NEAREST
                )
        mask = None
        if spec.mask is not None:
            key = self._mask(spec.mask, uploads)
            mask = arachne_gpu.SoftMask(
                key, pixels(camera, spec.mask_feather), spec.mask_invert
            )
        return arachne_gpu.Layer(
            source,
            depth=spec.depth,
            blur=None
            if spec.blur is None
            else arachne_gpu.Blur(pixels(camera, spec.blur)),
            mask=mask,
            object_id=self._origin(carrier),
        )

    def _mask(self, mask: Mobject, uploads: dict) -> bytes:
        """The key of ``mask``'s picture: one run uploaded as it is; runs of
        several kinds stacked by the compositor into one painted picture.
        """
        import arachne_gpu

        leaves = self.camera.get_mobjects_to_display([mask])
        parts = list(_runs([(leaf, None) for leaf in leaves], self._kind))
        if not parts:
            return self._upload(np.zeros_like(self.camera.pixel_array), True, uploads)
        if len(parts) == 1:
            return self._paint(parts[0][0], uploads)
        inner = [
            arachne_gpu.Layer(arachne_gpu.Picture(self._paint(part, uploads)))
            for part, _ in parts
        ]
        stacked = self.compositor.draw(arachne_gpu.Frame(_TRANSPARENT, inner))
        return self._upload(np.ascontiguousarray(stacked), True, uploads)

    def _stack_of(self, objects: Sequence[Mobject], uploads: dict) -> Any:
        """A window's stack: ``objects`` drawn as a frame of their own, on a
        transparent backdrop, under the frame's focus.
        """
        import arachne_gpu

        leaves = self.camera.get_mobjects_to_display(list(objects))
        owners = _carriers(objects, self._origin)
        layers = self._layers(
            [(leaf, owners.get(id(leaf))) for leaf in leaves], uploads
        )
        return arachne_gpu.Frame(_TRANSPARENT, layers, self._focus)

    def _kind(self, leaf: Mobject) -> str | None:
        """The camera's batch ``leaf`` is painted in: ``"vector"`` (Cairo,
        painted values), ``"image"`` (blended, straight values) or
        ``"points"`` (written over, straight values); ``None`` not at all.
        """
        from arachne.mobject.mobject import Mobject
        from arachne.mobject.types.image_mobject import AbstractImageMobject
        from arachne.mobject.types.vectorized_mobject import VMobject

        kind = self.camera.type_or_raise(leaf)
        if kind is VMobject:
            return "vector"
        if kind is AbstractImageMobject:
            return "image"
        return None if kind is Mobject else "points"

    def _paint(self, run: list[Mobject], uploads: dict) -> bytes:
        """Upload ``run``, one batch of the camera's, drawn by it on a
        transparent backdrop: a painted picture of vector objects, else an
        image of straight values.
        """
        vector = self._kind(run[0]) == "vector"
        return self._upload(self._painted(run), vector, uploads)

    def _painted(self, run: Sequence[Mobject]) -> np.ndarray:
        camera = self.camera
        camera.pixel_array[...] = 0
        camera.capture_mobjects(list(run), include_submobjects=False)
        return camera.pixel_array.copy()

    def _upload(self, rgba: np.ndarray, painted: bool, uploads: dict) -> bytes:
        import arachne_gpu

        key: bytes
        if painted:
            key = arachne_gpu.painted_key(rgba)
            if key not in uploads:
                self.compositor.upload_painted(key, rgba)
        else:
            key = image_key(rgba)
            if key not in uploads:
                self.compositor.upload_image(key, rgba)
        uploads[key] = None
        return key

    def _frame(self) -> list[tuple[float, float]]:
        camera = self.camera
        return [
            (x * camera.pixel_width, y * camera.pixel_height) for x, y in _FRAME_CORNERS
        ]

    def _corners(self, carrier: Mobject) -> list[tuple[float, float]]:
        """Where ``carrier``'s four corners fall, in pixels, in CE's image
        order: an image's own corners, else its bounding box's.
        """
        from arachne.constants import DL, DR, UL, UR
        from arachne.mobject.types.image_mobject import AbstractImageMobject

        if isinstance(carrier, AbstractImageMobject):
            points = carrier.points
        else:
            points = np.array([carrier.get_corner(c) for c in (UL, UR, DL, DR)])
        found = self.camera.points_to_subpixel_coords(carrier, points)
        return [(float(x), float(y)) for x, y in found]


def _spec(carrier: Mobject) -> LayerSpec:
    found = layer_of(carrier)
    assert found is not None
    return found


def _carriers(
    handed: Iterable[Mobject], origin: Callable[[Mobject], str]
) -> dict[int, Mobject]:
    """The layer each member of ``handed``'s families belongs to, by id:
    the member of its family that is a layer of its own. A layer within
    another's family is refused.
    """
    owners: dict[int, Mobject] = {}
    seen: set[int] = set()
    stack: list[tuple[Mobject, Mobject | None]] = [(mob, None) for mob in handed]
    stack.reverse()
    while stack:
        mob, carrier = stack.pop()
        if id(mob) in seen:
            continue
        seen.add(id(mob))
        if layer_of(mob) is not None:
            if carrier is not None:
                raise Unsupported(
                    "layer",
                    f"{origin(mob)} is a layer within the layer {origin(carrier)};"
                    " a layer holds another only through a window",
                )
            carrier = mob
        if carrier is not None:
            owners[id(mob)] = carrier
        stack.extend((sub, carrier) for sub in reversed(mob.submobjects))
    return owners


def _runs(
    leaves: Sequence[tuple[Mobject, Mobject | None]],
    kind: Callable[[Mobject], str | None],
) -> Iterable[tuple[list[Mobject], Mobject | None]]:
    """``leaves`` cut into runs: a layer's leaves together, the rest cut
    where the camera's kind of value changes; leaves the camera paints
    nothing of are left out. Each run with its layer's carrier, or ``None``.
    """
    run: list[Mobject] = []
    current: tuple[Any, ...] | None = None
    carrier_of: Mobject | None = None
    for leaf, carrier in leaves:
        if carrier is not None:
            mark: tuple[Any, ...] = ("layer", id(carrier))
        else:
            painted = kind(leaf)
            if painted is None:
                continue
            mark = ("plain", painted)
        if mark != current and run:
            yield run, carrier_of
            run = []
        current, carrier_of = mark, carrier
        run.append(leaf)
    if run:
        yield run, carrier_of
