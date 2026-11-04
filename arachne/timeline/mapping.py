"""The centre a render maps its paths around, found from the record.

A Camera or ThreeDCamera draws paths through one Cairo context per pixel
array, made at the array's first path draw and kept: its matrix subtracts the
frame centre as it was then, so every later path keeps that centre, wherever
the frame centre has moved since (a MovingCamera makes a new context for
each draw instead). A render draws its frames in order, so the centre is the
frame centre at its first draw that paints a path. Frames drawn from the
record afterwards, in any order, must map around that same centre:
:func:`freeze_mapping_centre` makes the camera's context for its array at
that instant before anything is drawn.
"""

from __future__ import annotations

__all__ = ["first_path_capture", "freeze_mapping_centre", "render_mapping_centre"]

import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from arachne.camera.camera import Camera
    from arachne.mobject.mobject import Mobject

    from .record import Capture, PlayRecord, Record


def first_path_capture(record: Record) -> Capture | None:
    """The capture of the render's first draw that paints a path through the
    frame array's Cairo context, in the order a render draws: each drawn
    play's static picture, then its frames; the final frame when the scene
    played nothing. ``None`` for a camera that makes a context for each draw
    (a MovingCamera), and when no drawn frame paints a path. Found from the
    record's captures, without drawing.
    """
    from arachne.camera.moving_camera import MovingCamera

    if record in _FIRST:
        return _FIRST[record]
    camera = record._scene.renderer.camera
    found = None if isinstance(camera, MovingCamera) else _first(record, camera)
    if record.final is not None:
        # A finished record draws no more frames.
        _FIRST[record] = found
    return found


def render_mapping_centre(record: Record) -> tuple[float, float, float] | None:
    """The frame centre the render of ``record`` froze into its frame
    array's Cairo context (:func:`first_path_capture`), or ``None``.
    """
    from .draw import Drawing

    capture = first_path_capture(record)
    if capture is None:
        return None
    camera = record._scene.renderer.camera
    with Drawing(record) as drawing:
        drawing.show(capture, [])
        x, y, z = (float(v) for v in np.asarray(camera.frame_center).reshape(3))
    return (x, y, z)


def freeze_mapping_centre(record: Record) -> None:
    """Give the record's camera, for its pixel array as it is, the Cairo
    context the render froze (:func:`first_path_capture`), unless it holds
    it already: made with the camera's own code, at that capture's camera.
    Frames drawn after it map paths as the render's do, whatever order they
    are asked in. A camera the record shows no path drawn for keeps its own.
    """
    capture = first_path_capture(record)
    if capture is None:
        return
    from .draw import Drawing

    camera = record._scene.renderer.camera
    array = camera.pixel_array
    made = _MADE.get(camera)
    if made is not None and camera.get_cached_cairo_context(array) is made:
        return
    camera.pixel_array_to_cairo_context.pop(id(array), None)
    with Drawing(record) as drawing:
        drawing.show(capture, [])
        _MADE[camera] = camera.get_cairo_context(array)


#: Per finished record, its first path capture.
_FIRST: weakref.WeakKeyDictionary[Record, Capture | None] = weakref.WeakKeyDictionary()
#: Per camera, the context :func:`freeze_mapping_centre` made last.
_MADE: weakref.WeakKeyDictionary[Camera, Any] = weakref.WeakKeyDictionary()


def _first(record: Record, camera: Camera) -> Capture | None:
    from .clock import exact, frames_in
    from .draw import Drawing

    fps = exact(camera.frame_rate)
    with Drawing(record) as drawing:

        def paints(capture: Capture, targets: Sequence[Mobject]) -> bool:
            drawing.show(capture, targets)
            return _paints_path(camera, targets)

        for play in record.plays:
            frames = frames_in(play.start, play.end, fps)
            drawn = not play.skipped and bool(frames)
            # A frozen frame's static picture is drawn whatever is skipped.
            if play.static and (drawn or play.held):
                static = record._captures[play.static_seq]
                if paints(static, list(play.static)):
                    return static
            if not drawn:
                continue
            captures = [play.samples[0]] if play.held else _frame_captures(play, frames)
            for capture in captures:
                if paints(capture, list(capture.moving) or capture.draw_list()):
                    return capture
        final = record.final
        if not record.plays and final is not None and paints(final, final.draw_list()):
            return final
    return None


def _frame_captures(play: PlayRecord, frames: Sequence[Any]) -> list[Capture]:
    """The captures the play's frames draw, each once, in order."""
    found: list[Capture] = []
    for t in frames:
        try:
            capture = play.capture_at(t)
        except LookupError:
            # A record off the render's grid: the last capture before.
            capture = play.base_at(t)
        if not found or found[-1] is not capture:
            found.append(capture)
    return found


def _paints_path(camera: Camera, targets: Sequence[Mobject]) -> bool:
    """Whether drawing ``targets`` paints a path through the frame array's
    context: a VMobject with points, unless a background image fills it (it
    draws into an array of its own).
    """
    from arachne.mobject.types.vectorized_mobject import VMobject

    for mob in camera.get_mobjects_to_display(targets):
        if camera.type_or_raise(mob) is VMobject and not mob.get_background_image():
            return True
    return False
