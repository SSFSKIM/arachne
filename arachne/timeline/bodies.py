"""The bodies a scene shows, and where each stands as a play ends.

A *body* is something on screen that can be seen to move: every top-level
mobject that paints something, and every direct child of one (but a text's
glyphs and a surface's faces), keyed by its stable id. Its position is the
centre of the points of its members that paint (a path or image with extent,
a point cloud with a visible stroke), in scene units; its size is their
box's; its ``view`` is that centre as the viewer sees it, in home-frame
units: relative to a moving camera's frame and scaled by its zoom, the scene
itself under every other camera. Something that paints nothing (alignment
padding, a tracker) is no body.

:func:`bodies` reads them off a scene as it holds its values now: a frame
shown from a record, or a live scene. The recorder also keeps them as each
play ends (:class:`Exit`, in ``Record.exits``): after its animations have
finished and cleaned up and before the script runs on, a state no capture
holds, and the one the lint's motion rule measures a cut from.
"""

from __future__ import annotations

__all__ = ["Body", "Exit", "bodies"]

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.camera.projection import draw_kind, view_of

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject

    from .origins import Origin

#: Scene units under which two points are one place.
_ONE_PLACE = 1e-3


@dataclass(frozen=True, eq=False)
class Body:
    """One body as the scene holds it: ``parent`` is the stable id of the
    top-level holding it (``None`` for a top-level), ``members`` how many
    members of its family have points, ``own`` whether its own points paint,
    ``visible`` whether some member lays ink a viewer can see. ``type``,
    ``text`` and ``scene_line`` name it.
    """

    id: str
    parent: str | None
    centre: np.ndarray
    size: np.ndarray
    view: np.ndarray
    members: int
    own: bool
    visible: bool
    type: str
    text: str | None
    scene_line: int | None


@dataclass(frozen=True, eq=False)
class Exit:
    """The bodies as a play ended, and the zoom of the view then."""

    zoom: float
    bodies: dict[str, Body]


def bodies(
    roots: list[Mobject],
    camera: Any,
    origin: Callable[[Mobject], Origin],
    *,
    home: tuple[float, float],
    configured_pixels: tuple[int, int],
) -> tuple[float, dict[str, Body]]:
    """The zoom of ``camera``'s view and the bodies of ``roots`` (a scene's
    draw list), each group before its members, as they hold their values now.
    """
    view = view_of(camera, home=home, configured_pixels=configured_pixels)
    zoom_x = view.home_width / view.frame_width
    zoom_y = view.home_height / view.frame_height
    cx, cy = view.frame_center[:2] if view.camera == "moving" else (0.0, 0.0)
    found: dict[str, Body] = {}
    # A child's leaves are its top-level's too: each is read once.
    leaf_facts: dict[int, _LeafFacts] = {}

    def facts(leaf: Mobject) -> _LeafFacts:
        known = leaf_facts.get(id(leaf))
        if known is None:
            known = leaf_facts[id(leaf)] = _LeafFacts(leaf)
        return known

    for mob, parent in _candidates(roots, _viewing(camera)):
        where = origin(mob)
        if where.id in found:
            continue
        leaves = [facts(m) for m in mob.get_family() if len(m.points)]
        lit = [leaf for leaf in leaves if leaf.lays]
        if not lit:
            continue
        low = np.min([leaf.low for leaf in lit], axis=0)
        high = np.max([leaf.high for leaf in lit], axis=0)
        centre = (low + high) / 2
        found[where.id] = Body(
            id=where.id,
            parent=origin(parent).id if parent is not None else None,
            centre=centre,
            size=high - low,
            view=np.array([(centre[0] - cx) * zoom_x, (centre[1] - cy) * zoom_y]),
            members=len(leaves),
            own=bool(len(mob.points)) and facts(mob).lays,
            visible=any(leaf.visible for leaf in leaves),
            type=type(mob).__name__,
            text=_text_of(mob),
            scene_line=where.scene_line,
        )
    return zoom_x, found


class _LeafFacts:
    """One member with points: whether its draw paints, its points' box in
    the scene's x and y, and (read when first asked) whether a viewer sees
    its ink.
    """

    __slots__ = ("_seen", "high", "lays", "leaf", "low")

    def __init__(self, leaf: Mobject) -> None:
        points = np.asarray(leaf.points, dtype=float)[:, :2]
        self.leaf = leaf
        self.lays = _lays(leaf)
        self.low, self.high = points.min(axis=0), points.max(axis=0)
        self._seen: bool | None = None

    @property
    def visible(self) -> bool:
        if self._seen is None:
            self._seen = _visible(self.leaf)
        return self._seen


def _viewing(camera: Any) -> set[int]:
    """The ids of the camera's own mobjects (its frame, trackers, light)."""
    from arachne.mobject.mobject import Mobject

    return {id(v) for v in vars(camera).values() if isinstance(v, Mobject)}


def _text_types() -> tuple[type, ...]:
    """What is read whole, as the lint reads texts (:mod:`arachne.lint.parts`)."""
    from arachne.mobject.text.numbers import DecimalNumber
    from arachne.mobject.text.tex_mobject import SingleStringMathTex
    from arachne.mobject.text.text_mobject import MarkupText, Paragraph, Text
    from arachne.mobject.text.typst_mobject import Typst

    return (Text, MarkupText, Paragraph, SingleStringMathTex, DecimalNumber, Typst)


def _candidates(
    roots: list[Mobject], hidden: set[int]
) -> Iterator[tuple[Mobject, Mobject | None]]:
    """The top-level mobjects with points, each followed by its direct
    children with points (none of a text's or a surface's), with the
    top-level holding each child; the camera's own are none of them.
    """
    from arachne.mobject.three_d.three_dimensions import Surface
    from arachne.mobject.value_tracker import ValueTracker

    whole = (*_text_types(), Surface)

    def drawn(mob: Mobject) -> bool:
        return (
            id(mob) not in hidden
            and not isinstance(mob, ValueTracker)
            and any(len(m.points) for m in mob.get_family())
        )

    seen: set[int] = set()
    for top in roots:
        if id(top) in seen or id(top) in hidden:
            continue
        seen.add(id(top))
        if drawn(top):
            yield top, None
        if isinstance(top, whole):
            continue
        for child in top.submobjects:
            if id(child) not in seen and drawn(child):
                seen.add(id(child))
                yield child, top


def _lays(mob: Mobject) -> bool:
    """Whether one member's own draw paints anything: a path or an image
    whose points are not all in one place, or a point cloud with a visible
    stroke.
    """
    kind = draw_kind(mob)
    points = np.asarray(mob.points)
    if not len(points) or kind is None:
        return False
    if kind == "points":
        rgbas = np.asarray(getattr(mob, "rgbas", np.zeros((0, 4))))
        alpha = float(rgbas[:, 3].max()) if rgbas.size else 0.0
        return alpha > 0 and float(np.max(getattr(mob, "stroke_width", 0))) > 0
    return bool((np.abs(points - points[0]) >= _ONE_PLACE).any())


def _visible(mob: Mobject) -> bool:
    """Whether a member lays ink a viewer can see move: a fill, or a stroke
    with width and opacity (its background stroke too), over more than one
    place; a visible point cloud; an image not wholly transparent.
    """
    kind = draw_kind(mob)
    if kind is None or not len(mob.points):
        return False
    if kind == "image":
        raster = np.asarray(getattr(mob, "pixel_array", np.zeros((0, 0, 4))))
        return not (raster.ndim == 3 and raster.shape[-1] == 4) or bool(
            raster[..., 3].any()
        )
    if kind == "points":
        return _lays(mob)
    inked = float(np.max(mob.get_fill_opacities(), initial=0.0)) > 0 or any(
        float(np.max(mob.get_stroke_width(background), initial=0.0)) > 0
        and float(np.max(mob.get_stroke_opacities(background), initial=0.0)) > 0
        for background in (False, True)
    )
    spread = np.ptp(np.asarray(mob.points)[:, :2], axis=0).max()
    return inked and float(spread) > 0


def _text_of(mob: Mobject) -> str | None:
    from .state import _text

    return _text(mob) if isinstance(mob, _text_types()) else None
