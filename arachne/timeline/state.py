"""The scene's state at an instant, as ``state_at`` answers it."""

from __future__ import annotations

__all__ = ["NodeView", "State", "build_state"]

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .draw import Drawing, camera_nodes

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject

    from .record import Capture, NodeState, PlayRecord, Record
    from .source import SourceLine


@dataclass(frozen=True)
class NodeView:
    """One mobject of the scene at an instant.

    ``box`` is ``(x_min, y_min, x_max, y_max)`` over the points of the
    mobject and its family, in scene units on the screen's plane: the points
    as the camera hands them to its pixel map, so projected under a
    ThreeDCamera and as they are under 2D cameras. ``None`` when the family
    has no points.
    """

    id: str
    source_line: SourceLine | None
    name: str | None
    type: str
    box: tuple[float, float, float, float] | None
    z: float
    color: str | None
    text: str | None
    children: tuple[str, ...]


@dataclass(frozen=True)
class State:
    """The scene at ``t``: its mobjects depth first from the top-level ones
    (trackers, the camera's own mobjects and the fully transparent padding
    that family alignment adds are left out), the play holding ``t`` and the
    route that answered.

    ``answer`` says how: ``"recorded"`` is a capture of the forward run,
    ``"analytic"`` a trusted play evaluated at ``t`` from its last capture
    before it.
    """

    t: Fraction
    play: int | None
    route: str | None
    answer: str
    roots: tuple[str, ...]
    nodes: tuple[NodeView, ...]

    def node(self, node_id: str) -> NodeView:
        """The node with stable id ``node_id``."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def named(self, name: str) -> NodeView:
        """The node the scene's code named ``name``."""
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)


def build_state(
    record: Record,
    t: Fraction,
    play: PlayRecord | None,
    capture: Capture,
    *,
    route: str | None,
    answer: str,
    prepare: Callable[[Drawing], None] | None = None,
) -> State:
    """The :class:`State` a capture holds, read with every reachable mobject
    given its captured values; ``prepare`` runs once they are (a trusted
    play's evaluation at an instant between its captures).
    """
    camera = record._scene.renderer.camera
    hidden = {id(mob) for mob in camera_nodes(camera)}
    names = play.names if play is not None else record.final_names
    views: list[NodeView] = []
    with Drawing(record) as drawing:
        drawing.show(capture, capture.draw_list())
        if prepare is not None:
            prepare(drawing)
        if hasattr(camera, "reset_rotation_matrix"):
            # A ThreeDCamera projects with the matrix of its recorded trackers.
            camera.reset_rotation_matrix()
        visible = _visible_ids(record, capture, hidden)
        for mob, state in record._walk(capture, capture.draw_list()):
            if id(mob) not in visible:
                continue
            origin = record.origin(mob)
            children = tuple(
                record.origin(child).id
                for child in state.submobjects
                if id(child) in visible
            )
            views.append(
                NodeView(
                    id=origin.id,
                    source_line=origin.source_line,
                    name=names.get(id(mob)),
                    type=type(mob).__name__,
                    box=_box(camera, mob),
                    z=float(getattr(mob, "z_index", 0)),
                    color=_color(mob),
                    text=_text(mob),
                    children=children,
                )
            )
        roots = tuple(
            record.origin(mob).id for mob in capture.draw_list() if id(mob) in visible
        )
    return State(t, play.index if play else None, route, answer, roots, tuple(views))


def _visible_ids(record: Record, capture: Capture, hidden: set[int]) -> set[int]:
    """The mobjects ``state_at`` reports: every reachable one but trackers,
    the camera's own, and padding that is fully transparent with all it holds.
    """
    from arachne.mobject.mobject import is_alignment_padding
    from arachne.mobject.value_tracker import ValueTracker

    visible: set[int] = set()

    def keep(mob: Mobject, state: NodeState, seen: set[int]) -> bool:
        if id(mob) in seen:
            return id(mob) in visible
        seen.add(id(mob))
        shown = False
        for child in state.submobjects:
            child_state = record.node_state(child, capture)
            if child_state is not None and keep(child, child_state, seen):
                shown = True
        if isinstance(mob, ValueTracker) or id(mob) in hidden:
            return False
        if shown or not (is_alignment_padding(mob) and _transparent(mob)):
            visible.add(id(mob))
            return True
        return False

    seen: set[int] = set()
    for mob in capture.draw_list():
        state = record.node_state(mob, capture)
        if state is not None:
            keep(mob, state, seen)
    return visible


def _transparent(mob: Mobject) -> bool:
    """Whether nothing of ``mob`` itself can be seen: every colour it holds
    has zero opacity.
    """
    for getter in ("get_fill_opacities", "get_stroke_opacities"):
        method = getattr(mob, getter, None)
        if method is None:
            continue
        try:
            opacities = np.asarray(method(), dtype=float)
        except Exception:
            return False
        if opacities.size and np.any(opacities > 0):
            return False
    opacity = getattr(mob, "opacity", None)
    return not (isinstance(opacity, (int, float)) and opacity > 0)


def _box(camera: Any, mob: Mobject) -> tuple[float, float, float, float] | None:
    blocks = []
    for member in mob.family_members_with_points():
        points = camera.transform_points_pre_display(member, member.points)
        blocks.append(np.asarray(points)[:, :2])
    if not blocks:
        return None
    points = np.vstack(blocks)
    low, high = points.min(axis=0), points.max(axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _color(mob: Mobject) -> str | None:
    try:
        return mob.get_color().to_hex()
    except Exception:
        return None


def _text(mob: Mobject) -> str | None:
    from arachne.mobject.text.numbers import DecimalNumber

    if isinstance(mob, DecimalNumber):
        return mob._get_num_string(mob.number)
    for attribute in ("original_text", "tex_string", "text", "code_string"):
        value = mob.__dict__.get(attribute)
        if isinstance(value, str):
            return value
    return None
